#include "PolyMBAPass.h"

#include "mbagen/PolyIREmit.h"
#include "mbagen/PolyMBA.h"
#include "mbagen/PolyMBACatalog.h"
#include "mbagen/PolySinkFinder.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

using namespace llvm;

llvm::cl::opt<bool> PolyMBAEnable("obfpolymba", llvm::cl::init(true), llvm::cl::desc("Polynomial MBA on i32/i64 Add/Sub/Mul/Shl sites with a ConstantInt operand."));

static constexpr unsigned Density = 100;
static constexpr unsigned FamilyCountMin = 32;
static constexpr unsigned FamilyCountMax = 128;
static constexpr unsigned A2Valuation = 0;
static constexpr unsigned InverseCapD = 80;
static constexpr unsigned VerifyBits = 18;
static constexpr unsigned VerifyRandom = 100000;
static constexpr unsigned MaxDecodeInstrs = 220;
static constexpr bool WrapAdd = true;
static constexpr bool WrapSub = true;
static constexpr bool Wrap32 = true;
static constexpr bool Wrap64 = true;
static constexpr bool SkipLoops = true;
static constexpr bool SkipDegenerateOutput = true;
static constexpr unsigned VerifySamples = 100;
static constexpr bool ChainEnable = true;
static constexpr unsigned DegreeMixPct = 25;
static constexpr bool WrapMul = true;
static constexpr bool WrapShl = true;
static constexpr unsigned BarrierPoolSize = 16;
static constexpr unsigned BarrierWeightLegacy = 0;
static constexpr unsigned BarrierWeightXor = 50;
static constexpr unsigned BarrierWeightAdd = 25;
static constexpr unsigned BarrierWeightSub = 25;

static uint64_t HashDJB2(const char* s)
{
    uint64_t h = 5381;
    while (*s)
    {
        h = ((h << 5) + h) + (uint64_t)(uint8_t)(*s);
        ++s;
    }
    return h;
}

static bool HasAnnotationTag(Function& f, const char* tag)
{
    Module* m = f.getParent();
    GlobalVariable* ga = m->getGlobalVariable("llvm.global.annotations");
    if (!ga || !ga->hasInitializer()) return false;
    auto* ca = dyn_cast<ConstantArray>(ga->getOperand(0));
    if (!ca) return false;
    for (unsigned i = 0; i < ca->getNumOperands(); ++i)
    {
        auto* cs = dyn_cast<ConstantStruct>(ca->getOperand(i));
        if (!cs || cs->getNumOperands() < 2) continue;
        if (cs->getOperand(0)->stripPointerCasts() != &f) continue;
        auto* annogv = dyn_cast<GlobalVariable>(cs->getOperand(1)->stripPointerCasts());
        if (!annogv || !annogv->hasInitializer()) continue;
        auto* cds = dyn_cast<ConstantDataSequential>(annogv->getInitializer());
        if (!cds || !cds->isString()) continue;
        if (strcmp(cds->getAsString().str().c_str(), tag) == 0) return true;
    }
    return false;
}

static bool HasObfuscateAnnotation(Function& f)
{
    return HasAnnotationTag(f, "obfus");
}

static bool HasNoObfuscateAnnotation(Function& f)
{
    return HasAnnotationTag(f, "noobfus");
}

static bool HasNoMBAAnnotation(Function& f)
{
    return HasAnnotationTag(f, "nomba");
}

static std::vector<polymba::PolyMBAFamily> Families32;
static std::vector<polymba::PolyMBAFamily> Families64;
static std::mt19937_64 Rng;
static mbagen::ModuleBarrierState BarrierState;
static DenseMap<Value*, SmallVector<uint8_t, 32>> FamilyUsedByRoot32;
static DenseMap<Value*, SmallVector<uint8_t, 32>> FamilyUsedByRoot64;

static bool ClassifyAddSub(Instruction* i, mbagen::PolySiteKind& kind, ConstantInt*& kc, unsigned& aopidx, unsigned& bw, uint64_t& koverride, bool& koverridevalid)
{
    koverridevalid = false;
    koverride = 0;
    auto* bo = dyn_cast<BinaryOperator>(i);
    if (!bo) return false;
    const auto opc = bo->getOpcode();
    if (opc != Instruction::Add && opc != Instruction::Sub && opc != Instruction::Mul && opc != Instruction::Shl) return false;
    auto* ity = dyn_cast<IntegerType>(bo->getType());
    if (!ity) return false;
    bw = ity->getBitWidth();
    if (bw == 32) { if (!Wrap32) return false; }
    else if (bw == 64) { if (!Wrap64) return false; }
    else return false;
    if (opc == Instruction::Add && !WrapAdd) return false;
    if (opc == Instruction::Sub && !WrapSub) return false;
    if (opc == Instruction::Mul && !WrapMul) return false;
    if (opc == Instruction::Shl && !WrapShl) return false;
    auto* op0const = dyn_cast<ConstantInt>(bo->getOperand(0));
    auto* op1const = dyn_cast<ConstantInt>(bo->getOperand(1));
    if (op0const && op1const) return false;
    if (!op0const && !op1const) return false;
    const uint64_t mask = (bw >= 64) ? ~uint64_t(0) : ((uint64_t(1) << bw) - 1);
    if (opc == Instruction::Add)
    {
        kind = mbagen::PolySiteKind::AddAK;
        if (op1const) { kc = op1const; aopidx = 0; }
        else { kc = op0const; aopidx = 1; }
        return true;
    }
    if (opc == Instruction::Sub)
    {
        if (op1const) { kind = mbagen::PolySiteKind::SubAK; kc = op1const; aopidx = 0; }
        else { kind = mbagen::PolySiteKind::SubKA; kc = op0const; aopidx = 1; }
        return true;
    }
    if (opc == Instruction::Mul)
    {
        ConstantInt* kconst = op1const ? op1const : op0const;
        const uint64_t kw = kconst->getZExtValue() & mask;
        // reject K==0 (trivial zero, LLVM folds) and K==1 (identity); larger valuation K reduces top coefficient magnitude but keeps correctness
        if (kw == 0 || kw == 1) return false;
        kind = mbagen::PolySiteKind::MulAK;
        if (op1const) { kc = op1const; aopidx = 0; }
        else { kc = op0const; aopidx = 1; }
        return true;
    }
    if (opc == Instruction::Shl)
    {
        if (!op1const) return false;
        const uint64_t shiftamt = op1const->getZExtValue();
        //  Shl at shiftamt == 0 is identity and shiftamt >= bw is poison in LLVM; reject both to keep the synthesised K = 2^shiftamt a nontrivial encoding
        if (shiftamt == 0 || shiftamt >= (uint64_t)bw) return false;
        const uint64_t kmul = (uint64_t(1) << shiftamt) & mask;
        koverride = kmul;
        koverridevalid = true;
        kc = op1const;
        aopidx = 0;
        kind = mbagen::PolySiteKind::MulAK;
        return true;
    }
    return false;
}

static bool HasPHIUser(Instruction* i)
{
    for (User* u : i->users())
        if (isa<PHINode>(u)) return true;
    return false;
}

static bool HasDegenerateOutputUse(Instruction* i)
{
    for (User* u : i->users())
    {
        auto* cmp = dyn_cast<ICmpInst>(u);
        if (!cmp) continue;
        Value* op0 = cmp->getOperand(0);
        Value* op1 = cmp->getOperand(1);
        Value* other = (op0 == i) ? op1 : op0;
        if (isa<ConstantInt>(other)) return true;
    }
    return false;
}

static size_t TrimmedFinvDegree(const polymba::Poly& p)
{
    size_t deg = p.C.size();
    while (deg > 1 && p.C[deg - 1] == 0) --deg;
    return deg;
}

struct Site
{
    Instruction* ins;
    uint64_t k;
    mbagen::PolySiteKind kind;
    unsigned bitwidth;
    unsigned aopidx;
    Value* chainroot = nullptr;
    uint64_t chainkcum = 0;
    unsigned chaindepth = 0;
};


static void CollectSites(Function& f, SmallVectorImpl<Site>& out)
{
    for (BasicBlock& bb : f)
    {
        for (Instruction& i : bb)
        {
            if (auto* bo = dyn_cast<BinaryOperator>(&i))
            {
                const auto opc = bo->getOpcode();
                if (opc != Instruction::Add && opc != Instruction::Sub && opc != Instruction::Mul && opc != Instruction::Shl)
                    continue;
            }
            else continue;
            mbagen::PolySiteKind kind{};
            ConstantInt* kc = nullptr;
            unsigned aopidx = 0;
            unsigned bw = 0;
            uint64_t koverride = 0;
            bool koverridevalid = false;
            if (!ClassifyAddSub(&i, kind, kc, aopidx, bw, koverride, koverridevalid)) continue;
            if (HasPHIUser(&i)) continue;
            if (SkipDegenerateOutput && HasDegenerateOutputUse(&i)) continue;
            Site s{};
            s.ins = &i;
            s.k = koverridevalid ? koverride : kc->getZExtValue();
            s.kind = kind;
            s.bitwidth = bw;
            s.aopidx = aopidx;
            out.push_back(s);
        }
    }
}

static bool DensityGateOpen(const Function& f, size_t siteidx)
{
    if (Density >= 100u) return true;
    if (Density == 0u) return false;
    uint64_t x = HashDJB2(f.getName().str().c_str());
    x ^= (uint64_t)siteidx * 0x9E3779B97F4A7C15ULL;
    // domain separation XOR distinguishes PolyMBA's density roll from Const/LinearMBA's so sites don't correlate across the three passes
    x ^= 0x506F6C794D424150ULL;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    const uint32_t roll = (uint32_t)(x & 0xFFu);
    const uint32_t thr = (Density * 256u) / 100u;
    return roll < thr;
}

static void AnalyzeChains(SmallVectorImpl<Site>& sites)
{
    for (auto& s : sites)
    {
        Value* op0 = s.ins->getOperand(s.aopidx);
        s.chainroot = op0;
        s.chainkcum = 0;
        s.chaindepth = 0;
        if (s.kind != mbagen::PolySiteKind::AddAK && s.kind != mbagen::PolySiteKind::SubAK) continue;
        const unsigned w = s.bitwidth;
        const uint64_t mask = (w >= 64) ? ~uint64_t(0) : ((uint64_t(1) << w) - 1);
        Value* cur = op0;
        uint64_t kacc = 0;
        unsigned depth = 0;
        // cap chain walk at 32 to bound compile time on pathological long chains; LLVM reassoc normally collapses linear chains before we see them
        const unsigned depthcap = 32;
        while (depth < depthcap)
        {
            auto* bo = dyn_cast<BinaryOperator>(cur);
            if (!bo) break;
            const unsigned opc = bo->getOpcode();
            if (opc != Instruction::Add && opc != Instruction::Sub) break;
            Value* lhs = bo->getOperand(0);
            Value* rhs = bo->getOperand(1);
            auto* lhsc = dyn_cast<ConstantInt>(lhs);
            auto* rhsc = dyn_cast<ConstantInt>(rhs);
            if (bo->getType()->getIntegerBitWidth() != w) break;
            if (rhsc && !lhsc)
            {
                const uint64_t c = rhsc->getZExtValue() & mask;
                if (opc == Instruction::Add) kacc = (kacc + c) & mask;
                else kacc = (kacc - c) & mask;
                cur = lhs;
            }
            else if (lhsc && !rhsc && opc == Instruction::Add)
            {
                const uint64_t c = lhsc->getZExtValue() & mask;
                kacc = (kacc + c) & mask;
                cur = rhs;
            }
            else break;
            ++depth;
        }
        s.chainroot = cur;
        s.chainkcum = kacc;
        s.chaindepth = depth;
    }
}

static bool VerifyHornerShape(const polymba::PolyMBAFamily& fam, uint64_t k, mbagen::PolySiteKind kind, unsigned w, unsigned samples, std::mt19937_64& rng)
{
    if (samples == 0) return true;
    const uint64_t m = (w >= 64) ? ~uint64_t(0) : ((uint64_t(1) << w) - 1);
    auto mask = [m](uint64_t v) { return v & m; };
    const uint64_t kw = mask(k);
    const uint64_t a0 = fam.A0, a1 = fam.A1, a2 = fam.A2, a3 = fam.A3, cxy = fam.CXY;
    uint64_t lead = 0, c0 = 0, c1 = 0, c2 = 0;
    if (fam.Degree == 2)
    {
        lead = a2;
        switch (kind)
        {
            case mbagen::PolySiteKind::AddAK:
            {
                const uint64_t ak = mask(a2 * kw);
                const uint64_t ak2 = mask(ak * kw);
                c0 = mask(a0 + mask(a1 * kw) + ak2);
                c1 = mask(a1 + mask(cxy * kw));
                break;
            }
            case mbagen::PolySiteKind::SubAK:
            {
                const uint64_t negk = mask(uint64_t(0) - kw);
                const uint64_t ak = mask(a2 * negk);
                const uint64_t ak2 = mask(ak * negk);
                c0 = mask(a0 + mask(a1 * negk) + ak2);
                c1 = mask(a1 - mask(cxy * kw));
                break;
            }
            case mbagen::PolySiteKind::SubKA:
            {
                const uint64_t ak = mask(a2 * kw);
                const uint64_t ak2 = mask(ak * kw);
                c0 = mask(a0 + mask(a1 * kw) + ak2);
                c1 = mask(uint64_t(0) - mask(a1 + mask(cxy * kw)));
                break;
            }
            case mbagen::PolySiteKind::MulAK:
            {
                lead = mask(a2 * mask(kw * kw));
                c1 = mask(a1 * kw);
                c0 = a0;
                break;
            }
        }
    }
    else
    {
        const uint64_t kw2 = mask(kw * kw);
        const uint64_t kw3 = mask(kw2 * kw);
        const uint64_t threea3 = mask(a3 + mask(a3 << 1));
        const uint64_t twoa2 = cxy;
        switch (kind)
        {
            case mbagen::PolySiteKind::AddAK:
            {
                lead = a3;
                c2 = mask(a2 + mask(threea3 * kw));
                c1 = mask(a1 + mask(twoa2 * kw) + mask(threea3 * kw2));
                c0 = mask(a0 + mask(a1 * kw) + mask(a2 * kw2) + mask(a3 * kw3));
                break;
            }
            case mbagen::PolySiteKind::SubAK:
            {
                const uint64_t negk = mask(uint64_t(0) - kw);
                const uint64_t negk2 = mask(negk * negk);
                const uint64_t negk3 = mask(negk2 * negk);
                lead = a3;
                c2 = mask(a2 + mask(threea3 * negk));
                c1 = mask(a1 + mask(twoa2 * negk) + mask(threea3 * negk2));
                c0 = mask(a0 + mask(a1 * negk) + mask(a2 * negk2) + mask(a3 * negk3));
                break;
            }
            case mbagen::PolySiteKind::SubKA:
            {
                lead = mask(uint64_t(0) - a3);
                c2 = mask(a2 + mask(threea3 * kw));
                c1 = mask(uint64_t(0) - mask(a1 + mask(twoa2 * kw) + mask(threea3 * kw2)));
                c0 = mask(a0 + mask(a1 * kw) + mask(a2 * kw2) + mask(a3 * kw3));
                break;
            }
            case mbagen::PolySiteKind::MulAK:
            {
                lead = mask(a3 * kw3);
                c2 = mask(a2 * kw2);
                c1 = mask(a1 * kw);
                c0 = a0;
                break;
            }
        }
    }
    for (unsigned i = 0; i < samples; ++i)
    {
        const uint64_t a = mask(rng());
        uint64_t aeff = 0;
        switch (kind)
        {
            case mbagen::PolySiteKind::AddAK: aeff = mask(a + kw); break;
            case mbagen::PolySiteKind::SubAK: aeff = mask(a - kw); break;
            case mbagen::PolySiteKind::SubKA: aeff = mask(kw - a); break;
            case mbagen::PolySiteKind::MulAK: aeff = mask(a * kw); break;
        }
        const uint64_t expected = fam.F.Eval(aeff, w);
        uint64_t horner;
        if (fam.Degree == 2)
        {
            horner = mask(mask(mask(lead * a + c1) * a) + c0);
        }
        else
        {
            const uint64_t t1 = mask(lead * a);
            const uint64_t t2 = mask(t1 + c2);
            const uint64_t t3 = mask(t2 * a);
            const uint64_t t4 = mask(t3 + c1);
            const uint64_t t5 = mask(t4 * a);
            horner = mask(t5 + c0);
        }
        if (horner != expected) return false;
    }
    return true;
}

static bool ProcessSite(Function& f, const Site& s, size_t siteidx, DominatorTree& dt, LoopInfo& li)
{
    if (SkipLoops && li.getLoopFor(s.ins->getParent()) != nullptr) return false;
    if (!DensityGateOpen(f, siteidx)) return false;
    const auto& familyset = (s.bitwidth == 32) ? Families32 : Families64;
    if (familyset.empty()) return false;
    const unsigned weff = s.bitwidth;
    const uint64_t maskeff = (weff >= 64) ? ~uint64_t(0) : ((uint64_t(1) << weff) - 1);
    //  chainroot is populated by AnalyzeChains (always) or by ProcessFunction's chain disabled fallback (also always); the ternary is defensive against future refactors that might leave it null, but in the shipped path chainroot is never null
    Value* effectivea = s.chainroot ? s.chainroot : s.ins->getOperand(s.aopidx);
    uint64_t effectivek = s.k;
    mbagen::PolySiteKind effectivekind = s.kind;
    if (ChainEnable && s.chaindepth > 0)
    {
        if (s.kind == mbagen::PolySiteKind::AddAK)
        {
            effectivek = (s.chainkcum + s.k) & maskeff;
            effectivekind = mbagen::PolySiteKind::AddAK;
        }
        else if (s.kind == mbagen::PolySiteKind::SubAK)
        {
            // normalise SubAK with absorbed chain into AddAK with K' = chainKcum - K, since negative K is representable as AddAK in Z/2^w
            effectivek = (s.chainkcum - s.k) & maskeff;
            effectivekind = mbagen::PolySiteKind::AddAK;
        }
    }
    auto& usedmap = (s.bitwidth == 32) ? FamilyUsedByRoot32 : FamilyUsedByRoot64;
    auto& usedbits = usedmap[effectivea];
    if (usedbits.size() < familyset.size()) usedbits.resize(familyset.size(), 0);
    size_t familyidx = 0;
    {
        size_t usedcount = 0;
        for (uint8_t b : usedbits) if (b) ++usedcount;
        if (usedcount >= familyset.size())
        {
            familyidx = Rng() % familyset.size();
        }
        else
        {
            const size_t poolsz = familyset.size();
            size_t guard = 0;
            for (;;)
            {
                const size_t cand = Rng() % poolsz;
                if (!usedbits[cand]) { familyidx = cand; break; }
                if (++guard > 8)
                {
                    const size_t start = Rng() % poolsz;
                    for (size_t k = 0; k < poolsz; ++k)
                    {
                        const size_t kk = (start + k) % poolsz;
                        if (!usedbits[kk]) { familyidx = kk; break; }
                    }
                    break;
                }
            }
        }
        usedbits[familyidx] = 1;
    }
    const polymba::PolyMBAFamily& family = familyset[familyidx];
    const size_t tdeg = TrimmedFinvDegree(family.FInv);
    const uint64_t decodeinstrs = (tdeg > 0) ? 2ull * (uint64_t)(tdeg - 1) : 0;
    if (decodeinstrs > (uint64_t)MaxDecodeInstrs) return false;
    if (VerifySamples > 0)
    {
        std::mt19937_64 vrng(Rng());
        if (!VerifyHornerShape(family, effectivek, effectivekind, s.bitwidth, VerifySamples, vrng)) return false;
    }
    SmallVector<Instruction*, 8> users;
    for (User* u : s.ins->users())
    {
        if (auto* ui = dyn_cast<Instruction>(u))
        {
            if (isa<PHINode>(ui)) return false;
            users.push_back(ui);
        }
    }
    if (users.empty()) return false;
    IRBuilder<> encodebuilder(s.ins);
    Value* enc = mbagen::EmitPolyEncode(encodebuilder, effectivea, family, effectivek, effectivekind, BarrierState);
    Instruction* insertpt = polymba::FindLatestDecodeInsertionPoint(users, dt);
    IRBuilder<> decodebuilder(insertpt);
    Value* dec = mbagen::EmitPolyDecode(decodebuilder, enc, family, BarrierState);
    s.ins->replaceAllUsesWith(dec);
    s.ins->eraseFromParent();
    return true;
}

static bool ProcessFunction(Function& f)
{
    if (f.isDeclaration()) return false;
    if (HasNoObfuscateAnnotation(f) || HasNoMBAAnnotation(f)) return false;
    if (!HasObfuscateAnnotation(f)) return false;
    SmallVector<Site, 16> sites;
    CollectSites(f, sites);
 
    if (!sites.empty() && ChainEnable) AnalyzeChains(sites);
    else
    {
        
        for (auto& s : sites)
        {
            s.chainroot = s.ins->getOperand(s.aopidx);
            s.chainkcum = 0;
            s.chaindepth = 0;
        }
    }
    // reset perroot family used maps per function so Value* keys from an earlier function don't collide with recycled allocator slots in this one
    FamilyUsedByRoot32.clear();
    FamilyUsedByRoot64.clear();
    bool mutated = false;
    if (!sites.empty())
    {
        DominatorTree dt;
        dt.recalculate(f);
        LoopInfo li;
        li.analyze(dt);
        for (size_t i = 0; i < sites.size(); ++i)
            mutated |= ProcessSite(f, sites[i], i, dt, li);
    }
    return mutated;
}

bool RunPolyMBA(Module& m)
{
    if (!PolyMBAEnable) return false;
    std::random_device rd;
    uint64_t seed = (uint64_t)rd();
    seed ^= (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    seed ^= ((uint64_t)rd()) << 32;
    seed ^= 0x506F6C794D424152ULL;
    Rng.seed(seed);
    uint64_t barrierseed = Rng();
    if (barrierseed == 0) barrierseed = 0x504F4C59424152ULL;
    BarrierState.Init(&m, BarrierPoolSize, barrierseed, BarrierWeightLegacy, BarrierWeightXor, BarrierWeightAdd, BarrierWeightSub, ".mba.barrier.pmba");
    unsigned lo = FamilyCountMin;
    unsigned hi = FamilyCountMax;
    if (lo > hi) std::swap(lo, hi);
    const uint64_t span = (uint64_t)(hi - lo) + 1ull;
    const unsigned counttarget32 = Wrap32 ? (unsigned)(lo + (Rng() % span)) : 0u;
    const unsigned counttarget64 = Wrap64 ? (unsigned)(lo + (Rng() % span)) : 0u;
    polymba::BuildOptions bocommon;
    bocommon.InverseCapD = InverseCapD;
    bocommon.VerifyBits = VerifyBits;
    bocommon.VerifyRandom = VerifyRandom;
    bocommon.A2Valuation = A2Valuation;
    bocommon.DegreeMixPct = DegreeMixPct;
    if (Wrap64)
    {
        auto bo = bocommon;
        bo.Seed = Rng();
        bo.Width = 64;
        bo.Count = counttarget64;
        Families64 = polymba::BuildFamilies(bo);
    }
    if (Wrap32)
    {
        auto bo = bocommon;
        bo.Seed = Rng();
        bo.Width = 32;
        bo.Count = counttarget32;
        Families32 = polymba::BuildFamilies(bo);
    }
    if (Families32.empty() && Families64.empty()) return false;
    bool mutated = false;
    for (Function& f : m) mutated |= ProcessFunction(f);
    // finalize runs BEFORE the family pool clears so BarrierFinalizeModule can still observe the live state (clear is only tearing down the Poly family tables, not the barrier state, but keeping the ordering unambiguous avoids a trap when a future refactor couples the two)
    mbagen::BarrierFinalizeModule(BarrierState);
    Families32.clear();
    Families64.clear();
    return mutated;
}

PreservedAnalyses PolyMBAPass::run(Module& m, ModuleAnalysisManager&)
{
    bool mutated = RunPolyMBA(m);
    return mutated ? PreservedAnalyses::none() : PreservedAnalyses::all();
}