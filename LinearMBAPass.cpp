#include "LinearMBAPass.h"

#include "mbagen/AuxSource.h"
#include "mbagen/FoldBarrier.h"
#include "mbagen/IREmit.h"
#include "mbagen/IdentityGenerator.h"
#include "mbagen/LinearMBA.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
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

llvm::cl::opt<bool> LinearMBAEnable("obflinearmba", llvm::cl::init(true), llvm::cl::desc("Identity wrap MBA at forced liveness sites (Store value, Call/Invoke arg)."));

static constexpr unsigned Density = 100;
static constexpr unsigned Tier = 8;
static constexpr unsigned AuxCount = 3;
static constexpr unsigned GroupCount = 2;
static constexpr bool WrapStore = true;
static constexpr bool WrapCallArg = true;
static constexpr bool SkipLoops = true;
static constexpr unsigned VerifySamples = 100;
static constexpr unsigned XYPct = 40;
static constexpr unsigned CrossWidthPct = 15;
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

struct Site
{
    enum class Kind : uint8_t { StoreValue, CallArg };
    Kind kind;
    Instruction* ins;
    unsigned opidx;
    unsigned bitwidth;
};

static mbagen::GeneratorState GenState32;
static mbagen::GeneratorState GenState64;
static std::mt19937_64 Rng;
static mbagen::ModuleBarrierState BarrierState;

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

static mbagen::GeneratorState& PickGenState(unsigned bw)
{
    return (bw == 32) ? GenState32 : GenState64;
}

static bool IsWrappableValue(Value* v, unsigned& outbw)
{
    if (!v) return false;
    return ClassifyIntegerWidth(v->getType(), outbw);
}

static void CollectSites(Function& f, std::vector<Site>& out)
{
    for (BasicBlock& bb : f)
    {
        for (Instruction& i : bb)
        {
            if (auto* st = dyn_cast<StoreInst>(&i))
            {
                if (!WrapStore) continue;
                unsigned bw;
                if (!IsWrappableValue(st->getValueOperand(), bw)) continue;
                Site s{};
                s.kind = Site::Kind::StoreValue;
                s.ins = &i;
                s.opidx = 0;
                s.bitwidth = bw;
                out.push_back(s);
                continue;
            }
            if (auto* cb = dyn_cast<CallBase>(&i))
            {
                if (!WrapCallArg) continue;
                if (cb->getIntrinsicID() != Intrinsic::not_intrinsic) continue;
                for (unsigned k = 0; k < cb->arg_size(); ++k)
                {
                    unsigned bw;
                    if (!IsWrappableValue(cb->getArgOperand(k), bw)) continue;
                    Site s{};
                    s.kind = Site::Kind::CallArg;
                    s.ins = &i;
                    s.opidx = k;
                    s.bitwidth = bw;
                    out.push_back(s);
                }
                continue;
            }
        }
    }
}

static bool DensityGateOpen(const Function& f, size_t siteidx)
{
    if (Density >= 100u) return true;
    if (Density == 0u) return false;
    uint64_t x = HashDJB2(f.getName().str().c_str());
    x ^= (uint64_t)siteidx * 0x9E3779B97F4A7C15ULL;
    // domain separation XOR distinguishes LinearMBA's density roll from ConstMBA's and PolyMBA's so a site stays independently gated per pass instead of correlating across them
    x ^= 0xA17E4CB91EDBF017ULL;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    const uint32_t roll = (uint32_t)(x & 0xFFu);
    const uint32_t thr = (Density * 256u) / 100u;
    return roll < thr;
}

static bool ProcessSite(Function& f, const Site& s, size_t siteidx, DominatorTree& dt, LoopInfo& li)
{
    if (SkipLoops)
    {
        if (li.getLoopFor(s.ins->getParent()) != nullptr) return false;
    }
    if (!DensityGateOpen(f, siteidx)) return false;
    Value* orig = (s.kind == Site::Kind::StoreValue)
                      ? cast<StoreInst>(s.ins)->getValueOperand()
                      : cast<CallBase>(s.ins)->getArgOperand(s.opidx);
    if (!orig) return false;
    auto* ity = dyn_cast<IntegerType>(orig->getType());
    if (!ity) return false;
    const unsigned bw = ity->getBitWidth();
    if (bw != s.bitwidth) return false;
    mbagen::LinearMBAParams params;
    params.AuxVarCount = AuxCount;
    params.RealVarCount = 1;
    params.GroupCount = GroupCount;
    params.Width = bw;
    params.GeneratorPolicyV.TierV = TierFromFlag(Tier);
    params.GenState = &PickGenState(bw);
    params.XYPct = XYPct;
    mbagen::LinearMBAExpr expr = mbagen::BuildLinearMBA(mbagen::LinearMBATarget::Identity, 0, params, Rng);
    if (expr.Terms.empty()) return false;
    if (VerifySamples > 0)
    {
        std::mt19937_64 vrng(Rng());
        if (!mbagen::VerifyLinear(expr, vrng, (int)VerifySamples)) return false;
    }
    IRBuilder<> builder(s.ins);
    auto auxvalues = mbagen::GatherAuxValues(builder, dt, ity, params.AuxVarCount, Rng, CrossWidthPct);
    SmallVector<Value*, 1> realvalues{orig};
    SmallVector<Value*, 8> auxarr(auxvalues.begin(), auxvalues.end());
    Value* raw = mbagen::EmitLinearMBA(builder, expr, realvalues, auxarr);
    Value* barriered = mbagen::InsertFoldBarrier(builder, raw, BarrierState);
    if (s.kind == Site::Kind::StoreValue) s.ins->setOperand(0, barriered);
    else cast<CallBase>(s.ins)->setArgOperand(s.opidx, barriered);
    return true;
}

static bool ProcessFunction(Function& f)
{
    if (f.isDeclaration()) return false;
    if (HasNoObfuscateAnnotation(f) || HasNoMBAAnnotation(f)) return false;
    if (!HasObfuscateAnnotation(f)) return false;
    std::vector<Site> sites;
    CollectSites(f, sites);
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

bool RunLinearMBA(Module& m)
{
    if (!LinearMBAEnable) return false;
    // do NOT ResetAuxPoolCache() here; ConstMBA (which runs first) already set it up and we deliberately share the pool to avoid duplicate pool allocas per function
    GenState32 = mbagen::GeneratorState{};
    GenState64 = mbagen::GeneratorState{};
    std::random_device rd;
    uint64_t seed = (uint64_t)rd();
    seed ^= (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    seed ^= ((uint64_t)rd()) << 32;
    // domain separation XOR with "LINEARMB" keeps the LinearMBA RNG stream independent from ConstMBA's even on identical seed inputs
    seed ^= 0x4C494E4541524D42ULL;
    Rng.seed(seed);
    uint64_t barrierseed = Rng();
    if (barrierseed == 0) barrierseed = 0x4C494E42415252ULL;
    BarrierState.Init(&m, BarrierPoolSize, barrierseed, BarrierWeightLegacy, BarrierWeightXor, BarrierWeightAdd, BarrierWeightSub, ".mba.barrier.lmba");
    bool mutated = false;
    for (Function& f : m) mutated |= ProcessFunction(f);
    // finalize runs after the whole module site loop so every lazily created slot gets an init-store entry; the per pass BarrierState carries its own SlotNamePrefix (".mba.barrier.lmba") so the registered __wp_barrier_init symbol stays distinct from ConstMBA's and PolyMBA's and all three append cleanly into llvm.global_ctors
    mbagen::BarrierFinalizeModule(BarrierState);
    return mutated;
}

PreservedAnalyses LinearMBAPass::run(Module& m, ModuleAnalysisManager&)
{
    bool mutated = RunLinearMBA(m);
    return mutated ? PreservedAnalyses::none() : PreservedAnalyses::all();
}