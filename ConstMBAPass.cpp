#include "ConstMBAPass.h"

#include "mbagen/AuxSource.h"
#include "mbagen/ConstMBA.h"
#include "mbagen/FoldBarrier.h"
#include "mbagen/IREmit.h"
#include "mbagen/IdentityGenerator.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using namespace llvm;

llvm::cl::opt<bool> ConstMBAEnable("obfconstmba", llvm::cl::init(true), llvm::cl::desc("Bitwise MBA on ConstantInt operand sites."));

static constexpr unsigned Density = 100;
static constexpr unsigned NormalTier = 8;
static constexpr unsigned AuxCount = 4;
static constexpr unsigned GroupCount = 3;
static constexpr bool SkipLoops = true;
static constexpr bool SkipGEP = false;
static constexpr bool SkipPHI = false;
static constexpr unsigned VerifySamples = 100;
static constexpr unsigned CrossWidthPct = 15;
static constexpr bool EscalateEnable = true;
static constexpr unsigned EscalatedTier = 6;
static constexpr unsigned EscalatedAuxCount = 6;
static constexpr unsigned EscalatedGroupCount = 5;
static constexpr bool ReactorEnable = true;
static constexpr unsigned ReactorTier = 4;
static constexpr unsigned ReactorAuxCount = 3;
static constexpr unsigned ReactorGroupCount = 2;
static constexpr uint64_t ReactorMinCoef = 4;
static constexpr unsigned ReactorPerFnCap = 256;
static constexpr unsigned BarrierPoolSize = 16;
static constexpr unsigned BarrierWeightLegacy = 0;
static constexpr unsigned BarrierWeightXor = 50;
static constexpr unsigned BarrierWeightAdd = 25;
static constexpr unsigned BarrierWeightSub = 25;

static constexpr uint64_t FNV1a64(const char* s)
{
    uint64_t h = 0xcbf29ce484222325ULL;
    while (*s)
    {
        h ^= (uint8_t)(*s);
        h *= 0x100000001b3ULL;
        ++s;
    }
    return h;
}

static uint64_t FNV1a64Range(llvm::StringRef s)
{
    uint64_t h = 0xcbf29ce484222325ULL;
    for (char c : s)
    {
        h ^= (uint8_t)c;
        h *= 0x100000001b3ULL;
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

static bool HasObfuscateAnnotation(Function& f) { return HasAnnotationTag(f, "obfus"); }
static bool HasNoObfuscateAnnotation(Function& f) { return HasAnnotationTag(f, "noobfus"); }
static bool HasNoMBAAnnotation(Function& f) { return HasAnnotationTag(f, "nomba"); }

struct Site
{
    enum class Kind : uint8_t { InstrOp, PHIIncoming };
    Kind kind;
    Instruction* ins;
    unsigned opidx;
    unsigned bitwidth;
    uint64_t value;
};

static mbagen::GeneratorState GenState32;
static mbagen::GeneratorState GenState64;
static std::mt19937_64 Rng;
static mbagen::ModuleBarrierState BarrierState;
static std::vector<Site> ReactorQueue;

static mbagen::Tier TierFromFlag(unsigned t)
{
    switch (t)
    {
        case 6: return mbagen::Tier::T6;
        case 7: return mbagen::Tier::T7;
        case 8:
        default: return mbagen::Tier::T8;
    }
}

static bool ClassifyIntegerWidth(Type* ty, unsigned& bw)
{
    auto* ity = dyn_cast<IntegerType>(ty);
    if (!ity) return false;
    bw = ity->getBitWidth();
    return bw == 32 || bw == 64;
}

static bool IsRewritableConstOp(Instruction& i, unsigned opidx, unsigned& outbw, uint64_t& outvalue, bool& outisgepindex)
{
    outisgepindex = false;
    Value* op = i.getOperand(opidx);
    auto* ci = dyn_cast<ConstantInt>(op);
    if (!ci) return false;
    unsigned bw;
    if (!ClassifyIntegerWidth(ci->getType(), bw)) return false;
    switch (i.getOpcode())
    {
        case Instruction::Add:
        case Instruction::Sub:
        case Instruction::Mul:
        case Instruction::And:
        case Instruction::Or:
        case Instruction::Xor:
        case Instruction::Shl:
        case Instruction::LShr:
        case Instruction::AShr:
            break;
        case Instruction::ICmp:
            break;
        case Instruction::Store:
            if (opidx != 0) return false;
            break;
        case Instruction::Call:
        case Instruction::Invoke:
        {
            auto* cb = dyn_cast<CallBase>(&i);
            if (!cb) return false;
            if (cb->getIntrinsicID() != Intrinsic::not_intrinsic) return false;
            if (opidx >= cb->arg_size()) return false;
            break;
        }
        case Instruction::GetElementPtr:
        {
            if (opidx == 0) return false;
            auto* gep = cast<GetElementPtrInst>(&i);
            unsigned idxn = opidx - 1;
            gep_type_iterator gti = gep_type_begin(gep);
            gep_type_iterator gte = gep_type_end(gep);
            for (unsigned k = 0; k < idxn && gti != gte; ++k) ++gti;
            if (gti == gte) return false;
            if (gti.isStruct()) return false;
            outisgepindex = true;
            break;
        }
        default:
            return false;
    }
    outbw = bw;
    outvalue = ci->getZExtValue();
    return true;
}

static bool IsRewritablePHIIncoming(PHINode& phi, unsigned idx, unsigned& outbw, uint64_t& outvalue)
{
    Value* v = phi.getIncomingValue(idx);
    auto* ci = dyn_cast<ConstantInt>(v);
    if (!ci) return false;
    unsigned bw;
    if (!ClassifyIntegerWidth(ci->getType(), bw)) return false;
    outbw = bw;
    outvalue = ci->getZExtValue();
    return true;
}

static void CollectSites(Function& f, std::vector<Site>& out)
{
    for (BasicBlock& bb : f)
    {
        for (Instruction& i : bb)
        {
            if (auto* phi = dyn_cast<PHINode>(&i))
            {
                for (unsigned k = 0; k < phi->getNumIncomingValues(); ++k)
                {
                    unsigned bw;
                    uint64_t v;
                    if (!IsRewritablePHIIncoming(*phi, k, bw, v)) continue;
                    Site s{};
                    s.kind = Site::Kind::PHIIncoming;
                    s.ins = &i;
                    s.opidx = k;
                    s.bitwidth = bw;
                    s.value = v;
                    out.push_back(s);
                }
                continue;
            }
            if (isa<SwitchInst>(&i)) continue;
            for (unsigned k = 0; k < i.getNumOperands(); ++k)
            {
                unsigned bw;
                uint64_t v;
                bool isgep = false;
                if (!IsRewritableConstOp(i, k, bw, v, isgep)) continue;
                Site s{};
                s.kind = Site::Kind::InstrOp;
                s.ins = &i;
                s.opidx = k;
                s.bitwidth = bw;
                s.value = v;
                out.push_back(s);
            }
        }
    }
}

static bool DensityGateOpen(const Function& f, size_t siteidx)
{
    if (Density >= 100u) return true;
    if (Density == 0u) return false;
    uint64_t x = FNV1a64Range(f.getName());
    x ^= (uint64_t)siteidx * 0x9E3779B97F4A7C15ULL;
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    const uint32_t roll = (uint32_t)(x & 0xFFu);
    const uint32_t thr = (Density * 256u) / 100u;
    return roll < thr;
}

static mbagen::GeneratorState& PickGenState(unsigned bw)
{
    return (bw == 32) ? GenState32 : GenState64;
}

static void CollectReactorMuls(Value* raw, SmallVectorImpl<Instruction*>& out, unsigned maxdepth = 256)
{
    SmallPtrSet<Value*, 32> visited;
    SmallVector<Value*, 32> work;
    work.push_back(raw);
    while (!work.empty())
    {
        Value* v = work.pop_back_val();
        if (!visited.insert(v).second) continue;
        if (visited.size() > maxdepth) break;
        auto* ins = dyn_cast<Instruction>(v);
        if (!ins) continue;
        auto* bo = dyn_cast<BinaryOperator>(ins);
        if (!bo) continue;
        if (bo->getOpcode() == Instruction::Add)
        {
            work.push_back(bo->getOperand(0));
            work.push_back(bo->getOperand(1));
        }
        else if (bo->getOpcode() == Instruction::Mul)
        {
            if (isa<ConstantInt>(bo->getOperand(0))) out.push_back(bo);
        }
    }
}

static bool ProcessSite(Function& f, const Site& s, size_t siteidx, DominatorTree& dt, LoopInfo& li)
{
    if (SkipGEP)
    {
        if (s.kind == Site::Kind::InstrOp && isa<GetElementPtrInst>(s.ins) && s.opidx > 0) return false;
    }
    if (SkipPHI && s.kind == Site::Kind::PHIIncoming) return false;
    if (SkipLoops)
    {
        BasicBlock* bb = s.ins->getParent();
        if (li.getLoopFor(bb) != nullptr) return false;
    }
    bool escalated = false;
    bool isreactor = false;
    if (EscalateEnable && s.kind == Site::Kind::InstrOp && s.ins != nullptr)
    {
        if (MDNode* md = s.ins->getMetadata("mbagen.escalate"))
        {
            if (md->getNumOperands() >= 1 && isa<MDString>(md->getOperand(0)))
            {
                auto tag = cast<MDString>(md->getOperand(0))->getString();
                
                if (tag == "rhopq_skip") return false;
                if (tag == "const_reactor") isreactor = true;
                else escalated = true;
            }
        }
    }
    if (!escalated && !isreactor && !DensityGateOpen(f, siteidx)) return false;
    mbagen::ConstMBAParams params;
    if (isreactor)
    {
        params.AuxVarCount = ReactorAuxCount;
        params.GroupCount = ReactorGroupCount;
        params.GeneratorPolicyV.TierV = TierFromFlag(ReactorTier);
    }
    else if (escalated)
    {
        params.AuxVarCount = EscalatedAuxCount;
        params.GroupCount = EscalatedGroupCount;
        params.GeneratorPolicyV.TierV = TierFromFlag(EscalatedTier);
    }
    else
    {
        params.AuxVarCount = AuxCount;
        params.GroupCount = GroupCount;
        params.GeneratorPolicyV.TierV = TierFromFlag(NormalTier);
    }
    params.Width = s.bitwidth;
    params.GenState = &PickGenState(s.bitwidth);
    mbagen::ConstMBAExpr expr = mbagen::BuildConstMBA(s.value, params, Rng);
    if (expr.Terms.empty()) return false;
    if (VerifySamples > 0)
    {
        std::mt19937_64 vrng(Rng());
        if (!mbagen::VerifyConst(expr, s.value, vrng, (int)VerifySamples)) return false;
    }
    Instruction* insertbefore = nullptr;
    PHINode* phi = nullptr;
    if (s.kind == Site::Kind::PHIIncoming)
    {
        phi = cast<PHINode>(s.ins);
        BasicBlock* pred = phi->getIncomingBlock(s.opidx);
        insertbefore = pred->getTerminator();
    }
    else
    {
        insertbefore = s.ins;
    }
    IRBuilder<> builder(insertbefore);
    IntegerType* resultty = IntegerType::get(f.getContext(), s.bitwidth);
    std::vector<Value*> auxvalues;
    
    if (isreactor)
    {
        Function* ff = insertbefore->getFunction();
        AllocaInst* pool = mbagen::GetOrCreateAuxPool(*ff, Rng);
        auxvalues.reserve(params.AuxVarCount);
        unsigned slot = (unsigned)(Rng() % mbagen::AuxPoolSlots);
        for (unsigned k = 0; k < params.AuxVarCount; ++k)
        {
            auxvalues.push_back(mbagen::EmitAuxPoolLoad(builder, pool, resultty, slot));
            slot = (slot + 1) % mbagen::AuxPoolSlots;
        }
    }
    else
    {
        auxvalues = mbagen::GatherAuxValues(builder, dt, resultty, params.AuxVarCount, Rng, CrossWidthPct);
    }
    SmallVector<Value*, 8> auxarr(auxvalues.begin(), auxvalues.end());
    Value* raw = mbagen::EmitConstMBA(builder, expr, auxarr);
    
    Value* barriered;
    if (isreactor) barriered = raw;
    else barriered = mbagen::InsertFoldBarrier(builder, raw, BarrierState);
    if (phi) phi->setIncomingValue(s.opidx, barriered);
    else s.ins->setOperand(s.opidx, barriered);
    if (escalated && ReactorEnable && !isreactor)
    {
        SmallVector<Instruction*, 32> termmuls;
        CollectReactorMuls(raw, termmuls);
        const uint64_t wmask = (s.bitwidth >= 64) ? ~uint64_t(0) : ((uint64_t(1) << s.bitwidth) - 1);
        const uint64_t mincoef = ReactorMinCoef;
        const unsigned perfncap = ReactorPerFnCap;
        for (Instruction* mul : termmuls)
        {
            if (perfncap != 0 && ReactorQueue.size() >= perfncap) break;
            auto* ci = dyn_cast<ConstantInt>(mul->getOperand(0));
            if (!ci) continue;
            const uint64_t coef = ci->getZExtValue() & wmask;
            const uint64_t negcoef = (uint64_t(0) - coef) & wmask;
            if (coef < mincoef || negcoef < mincoef) continue;
            LLVMContext& c = f.getContext();
            MDNode* md = MDNode::get(c, { MDString::get(c, "const_reactor") });
            mul->setMetadata("mbagen.escalate", md);
            Site rs{};
            rs.kind = Site::Kind::InstrOp;
            rs.ins = mul;
            rs.opidx = 0;
            rs.bitwidth = s.bitwidth;
            rs.value = coef;
            ReactorQueue.push_back(rs);
        }
    }
    (void)li;
    return true;
}

static bool ProcessFunction(Function& f)
{
    if (f.isDeclaration()) return false;
    if (HasNoObfuscateAnnotation(f) || HasNoMBAAnnotation(f)) return false;
    if (!HasObfuscateAnnotation(f)) return false;
    ReactorQueue.clear();
    std::vector<Site> sites;
    CollectSites(f, sites);
    bool mutated = false;
    DominatorTree dt;
    LoopInfo li;
    if (!sites.empty())
    {
        dt.recalculate(f);
        li.analyze(dt);
        for (size_t i = 0; i < sites.size(); ++i) mutated |= ProcessSite(f, sites[i], i, dt, li);
    }
    if (!ReactorQueue.empty())
    {
        if (sites.empty())
        {
            dt.recalculate(f);
            li.analyze(dt);
        }
        for (size_t i = 0; i < ReactorQueue.size(); ++i) mutated |= ProcessSite(f, ReactorQueue[i], sites.size() + i, dt, li);
        ReactorQueue.clear();
    }
    return mutated;
}

bool RunConstMBA(Module& m)
{
    if (!ConstMBAEnable) return false;
    GenState32 = mbagen::GeneratorState{};
    GenState64 = mbagen::GeneratorState{};
    mbagen::ResetAuxPoolCache();
    std::random_device rd;
    uint64_t seed = (uint64_t)rd();
    seed ^= (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    seed ^= ((uint64_t)rd()) << 32;
    Rng.seed(seed);
    uint64_t barrierseed = Rng();
    if (barrierseed == 0) barrierseed = 0x9E3779B97F4A7C15ULL;
    BarrierState.Init(&m, BarrierPoolSize, barrierseed, BarrierWeightLegacy, BarrierWeightXor, BarrierWeightAdd, BarrierWeightSub, ".mba.barrier.cmba");
    bool mutated = false;
    for (Function& f : m) mutated |= ProcessFunction(f);
    
    mbagen::BarrierFinalizeModule(BarrierState);
    return mutated;
}

PreservedAnalyses ConstMBAPass::run(Module& m, ModuleAnalysisManager&)
{
    bool mutated = RunConstMBA(m);
    return mutated ? PreservedAnalyses::none() : PreservedAnalyses::all();
}