#include "RiggedHornerOpaquePass.h"

#include "mbagen/AuxSource.h"
#include "mbagen/FoldBarrier.h"
#include "mbagen/PolyMBA.h"
#include "mbagen/RiggedHornerCatalog.h"
#include "mbagen/RiggedHornerEmit.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace llvm;

llvm::cl::opt<bool> RiggedHornerOpaqueEnable("obfrhopq", llvm::cl::init(true), llvm::cl::desc("Rigged Polynomial Horner MBA"));

static constexpr unsigned Density = 100;
static constexpr unsigned PerFunctionCap = 16;
static constexpr unsigned MinBBInsts = 2;
static constexpr bool SkipLoops = false;
static constexpr bool SkipEntry = false;
static constexpr bool Wrap32 = true;
static constexpr bool Wrap64 = true;

static constexpr unsigned CountA = 8;
static constexpr unsigned CountB = 4;
static constexpr unsigned CountC = 4;
static constexpr unsigned CountD = 8;
static constexpr unsigned CountE = 4;
static constexpr unsigned CountF = 4;

static constexpr unsigned LowBitsK = 8;
static constexpr unsigned BitIdxMin = 8;
static constexpr unsigned BitIdxMax = 48;
static constexpr unsigned ImageEnumSamples = 8192;
static constexpr unsigned ImageMaskBits = 8;
static constexpr unsigned InverseCapD = 80;
static constexpr unsigned VerifyBits = 18;
static constexpr unsigned VerifyRandom = 50000;
static constexpr unsigned RiggedVerifySamples = 10000;
static constexpr unsigned DegreeMixPct = 25;

static constexpr unsigned BarrierPoolSize = 16;
static constexpr unsigned BarrierWeightLegacy = 0;
static constexpr unsigned BarrierWeightXor = 50;
static constexpr unsigned BarrierWeightAdd = 25;
static constexpr unsigned BarrierWeightSub = 25;

static constexpr unsigned DiamondCloneCap = 16;
static constexpr unsigned PolluteSizeFloor = 8;
static constexpr unsigned PolluteSizeCeiling = 48;
static constexpr unsigned StealCeiling = 16;
static constexpr unsigned StealFloor = 4;
static constexpr unsigned RealBBMinRemain = 5;

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
static bool HasNoRHOPQAnnotation(Function& f) { return HasAnnotationTag(f, "norhopq"); }

static std::vector<polymba::RiggedFamily> Families32;
static std::vector<polymba::RiggedFamily> Families64;
static std::mt19937_64 Rng;
static mbagen::ModuleBarrierState BarrierState;

static GlobalVariable* CSaltGV = nullptr;

static void EnsureCSalt(Module& m)
{
    if (CSaltGV) return;
    LLVMContext& c = m.getContext();
    IntegerType* i64 = Type::getInt64Ty(c);

    CSaltGV = new GlobalVariable(m, i64, false, GlobalValue::InternalLinkage, ConstantInt::get(i64, 0), ".mba.barrier.rhopq.csalt");
    CSaltGV->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
    CSaltGV->setAlignment(Align(8));

    FunctionType* ft = FunctionType::get(Type::getVoidTy(c), false);
    Function* saltinit = Function::Create(ft, GlobalValue::InternalLinkage, "RhopqCSaltInit", &m);
    saltinit->addFnAttr(Attribute::NoInline);
    BasicBlock* entry = BasicBlock::Create(c, "Entry", saltinit);
    IRBuilder<> ib(entry);
    Function* rdtsc = Intrinsic::getDeclaration(&m, Intrinsic::readcyclecounter);
    Value* tsc = ib.CreateCall(rdtsc, {}, "RhopqRdtsc");
    StoreInst* st = ib.CreateStore(tsc, CSaltGV);
    st->setVolatile(true);
    st->setAlignment(Align(8));
    ib.CreateRetVoid();

    // priority 102 MUST sit after the barrier ctor (priority 101) so the barrier pool is up by the time any downstream code that depends on both reads the salt; a lower priority would leave barrier reads undefined on first use
    appendToGlobalCtors(m, saltinit, 102);
}

static bool BBAlreadyProcessed(BasicBlock* bb)
{
    Instruction* term = bb->getTerminator();
    if (!term) return true;
    return term->getMetadata("rhopq.done") != nullptr;
}

static void MarkBBProcessed(BasicBlock* bb)
{
    Instruction* term = bb->getTerminator();
    if (!term) return;
    term->setMetadata("rhopq.done", MDNode::get(bb->getContext(), {}));
}

static bool IsPolluteBlock(BasicBlock* bb)
{
    Instruction* term = bb->getTerminator();
    if (!term) return false;
    return term->getMetadata("rhopq.pollute") != nullptr;
}

static void MarkPolluteBlock(BasicBlock* bb)
{
    Instruction* term = bb->getTerminator();
    if (!term) return;
    term->setMetadata("rhopq.pollute", MDNode::get(bb->getContext(), {}));
}

static Instruction* PickInsertionPoint(BasicBlock* bb)
{
    Instruction* first = bb->getFirstNonPHIOrDbgOrLifetime();
    if (!first) return nullptr;

    if ((unsigned)bb->size() < MinBBInsts) return nullptr;

    // splitting the entry block before an alloca relocates it into realbb and demotes it from static to dynamic, which breaks AllocaInst::isStaticAlloca downstream. clamp `first` past the last entry alloca so every alloca stays in the head
    if (bb == &bb->getParent()->getEntryBlock())
    {
        Instruction* lastalloca = nullptr;
        for (Instruction& i : *bb)
        {
            if (isa<AllocaInst>(&i)) lastalloca = &i;
        }
        if (lastalloca)
        {
            Instruction* after = lastalloca->getNextNode();
            if (!after) return nullptr;
            BasicBlock::iterator seek(first);
            while (seek != bb->end() && &*seek != after) ++seek;
            if (seek != bb->end()) first = &*seek;
            else first = after;
        }
    }

    unsigned splittable = 0;
    for (auto it = BasicBlock::iterator(first); it != bb->end(); ++it)
    {
        if (&*it == bb->getTerminator()) break;
        ++splittable;
    }
    if (splittable == 0) return nullptr;

    const unsigned pick = (unsigned)(Rng() % splittable);
    BasicBlock::iterator it(first);
    for (unsigned i = 0; i < pick; ++i) ++it;

    if (it == bb->end()) return nullptr;
    if (&*it == bb->getTerminator()) return nullptr;
    if (isa<PHINode>(&*it)) return nullptr;
    return &*it;
}

struct PolluteSlot
{
    AllocaInst* Base;
    int FieldIdx;
    Type* BaseTy;
    Type* RawTy;
    bool IsPtr;
    IntegerType* IntTy;
};

static Value* MaterializeSlotPtr(IRBuilder<>& b, const PolluteSlot& s)
{
    if (s.FieldIdx < 0) return s.Base;
    LLVMContext& c = b.getContext();
    IntegerType* i32 = Type::getInt32Ty(c);
    return b.CreateInBoundsGEP(s.BaseTy, s.Base, { ConstantInt::get(i32, 0), ConstantInt::get(i32, s.FieldIdx) }, "RhopqGep");
}

static Value* EmitSlotLoad(IRBuilder<>& b, const PolluteSlot& s)
{
    Value* ptr = MaterializeSlotPtr(b, s);
    LoadInst* ld = b.CreateLoad(s.RawTy, ptr, "RhopqLoad");
    if (!s.IsPtr) return ld;
    return b.CreatePtrToInt(ld, s.IntTy, "RhopqPtrToInt");
}

static void EmitSlotStore(IRBuilder<>& b, const PolluteSlot& s, Value* intval, bool volatilestore)
{
    Value* ptr = MaterializeSlotPtr(b, s);
    Value* v = intval;
    if (s.IsPtr) v = b.CreateIntToPtr(intval, s.RawTy, "RhopqIntToPtr");
    StoreInst* st = b.CreateStore(v, ptr);
    if (volatilestore) st->setVolatile(true);
}

static void CollectPolluteSlots(Function& f, IntegerType* i32, IntegerType* i64, SmallVectorImpl<PolluteSlot>& out32, SmallVectorImpl<PolluteSlot>& out64)
{
    for (Instruction& i : f.getEntryBlock())
    {
        auto* a = dyn_cast<AllocaInst>(&i);
        if (!a || !a->isStaticAlloca()) continue;
        Type* ty = a->getAllocatedType();

        if (ty->isIntegerTy(32)) { out32.push_back({a, -1, ty, ty, false, i32}); continue; }
        if (ty->isIntegerTy(64)) { out64.push_back({a, -1, ty, ty, false, i64}); continue; }

        if (ty->isPointerTy())
        {
            out64.push_back({a, -1, ty, ty, true, i64});
            continue;
        }

        if (auto* st = dyn_cast<StructType>(ty))
        {
            for (unsigned fi = 0; fi < st->getNumElements(); ++fi)
            {
                Type* ft = st->getElementType(fi);
                if (ft->isIntegerTy(32)) out32.push_back({a, (int)fi, ty, ft, false, i32});
                else if (ft->isIntegerTy(64)) out64.push_back({a, (int)fi, ty, ft, false, i64});
                else if (ft->isPointerTy()) out64.push_back({a, (int)fi, ty, ft, true, i64});
            }
        }
    }
}

static unsigned DiamondCloneRealIntoPollute(BasicBlock* realbb, BasicBlock* pollutebb, std::mt19937_64& rng, unsigned maxclone)
{
    if (maxclone == 0) return 0;

    ValueToValueMapTy vmap;
    unsigned cloned = 0;
    const uint64_t mutnonce = rng();

    for (Instruction& i : *realbb)
    {
        if (cloned >= maxclone) break;
        if (i.isTerminator()) break;

        if (isa<PHINode>(&i)) continue;
        if (isa<AllocaInst>(&i)) continue;
        if (isa<CallBase>(&i)) continue;
        if (isa<LandingPadInst>(&i)) continue;
        if (isa<CatchPadInst>(&i)) continue;
        if (isa<CleanupPadInst>(&i)) continue;
        if (i.isAtomic()) continue;
        if (isa<FenceInst>(&i)) continue;

        bool allok = true;
        for (Use& u : i.operands())
        {
            Value* op = u.get();
            auto* oi = dyn_cast<Instruction>(op);
            if (!oi) continue;
            if (vmap.count(op)) continue;
            if (oi->getParent() != realbb && oi->getParent() != pollutebb) continue;
            allok = false;
            break;
        }
        if (!allok) continue;

        Instruction* clone = i.clone();
        clone->setName("");
        clone->insertInto(pollutebb, pollutebb->end());
        RemapInstruction(clone, vmap, RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);

        // Instruction::clone() carries inherited mbagen.escalate tags (poly_const from PolyMBA, const_reactor from earlier ConstMBA walks) and the blanket rhopq_skip post-walk preserves poly_const by design, so an inherited poly_const on a clone would survive as a T6 escalate site amplifying roughly 15kx each. strip explicitly before the blanket runs
        clone->setMetadata("mbagen.escalate", nullptr);

        if (auto* bo = dyn_cast<BinaryOperator>(clone))
        {
            if ((rng() & 1u) == 0)
            {
                Instruction::BinaryOps newop = bo->getOpcode();
                switch (bo->getOpcode())
                {
                    case Instruction::Add: newop = Instruction::Sub; break;
                    case Instruction::Sub: newop = Instruction::Add; break;
                    case Instruction::And: newop = Instruction::Or;  break;
                    case Instruction::Or:  newop = Instruction::And; break;
                    case Instruction::Xor: newop = Instruction::Or;  break;
                    case Instruction::Mul: newop = Instruction::Add; break;
                    default: break;
                }
                if (newop != bo->getOpcode())
                {
                    IRBuilder<> b(bo);
                    Value* nb = b.CreateBinOp(newop, bo->getOperand(0), bo->getOperand(1));
                    bo->replaceAllUsesWith(nb);
                    vmap[&i] = nb;
                    bo->eraseFromParent();
                    ++cloned;
                    continue;
                }
            }
            for (unsigned op = 0; op < bo->getNumOperands(); ++op)
            {
                if (auto* ci = dyn_cast<ConstantInt>(bo->getOperand(op)))
                {
                    if ((rng() % 100u) < 50u)
                    {
                        bo->setOperand(op, ConstantInt::get(ci->getType(), ci->getZExtValue() ^ mutnonce));
                    }
                }
            }
        }
        else if (auto* si = dyn_cast<StoreInst>(clone))
        {
            if ((rng() % 100u) < 30u)
            {
                Value* v = si->getValueOperand();
                if (auto* iv = dyn_cast<IntegerType>(v->getType()))
                {
                    IRBuilder<> b(si);
                    Value* nv = b.CreateXor(v, ConstantInt::get(iv, mutnonce), "");
                    si->setOperand(0, nv);
                }
            }
        }

        vmap[&i] = clone;
        ++cloned;
    }

    return cloned;
}

static void FillPolluteBlock(Function& f, BasicBlock* pollutebb, BasicBlock* realbb, BasicBlock* headbb, std::mt19937_64& rng)
{
    LLVMContext& c = f.getContext();
    IRBuilder<> pollb(pollutebb);
    IntegerType* i32 = Type::getInt32Ty(c);
    IntegerType* i64 = Type::getInt64Ty(c);

    SmallVector<PolluteSlot, 32> slots32, slots64;
    CollectPolluteSlots(f, i32, i64, slots32, slots64);

    SmallVector<Value*, 16> vals32, vals64;
    for (Argument& a : f.args())
    {
        if (a.getType()->isIntegerTy(32)) vals32.push_back(&a);
        else if (a.getType()->isIntegerTy(64)) vals64.push_back(&a);
    }
    for (Instruction& i : *headbb)
    {
        if (i.getType()->isIntegerTy(32)) vals32.push_back(&i);
        else if (i.getType()->isIntegerTy(64)) vals64.push_back(&i);
    }

    const bool have32 = !slots32.empty();
    const bool have64 = !slots64.empty();

    const unsigned diamondcap = std::min<unsigned>((unsigned)(realbb->size() > 1 ? realbb->size() - 1 : 0), DiamondCloneCap);
    const unsigned diamondcloned = DiamondCloneRealIntoPollute(realbb, pollutebb, rng, diamondcap);

    if (!have32 && !have64)
    {
        AllocaInst* pool = mbagen::GetOrCreateAuxPool(f, rng);
        unsigned sidx = (unsigned)(rng() % mbagen::AuxPoolSlots);
        pollb.SetInsertPoint(pollutebb);
        (void)mbagen::EmitAuxPoolLoad(pollb, pool, i64, sidx);
        pollb.CreateBr(realbb);
        return;
    }

    pollb.SetInsertPoint(pollutebb);

    unsigned targetops = (unsigned)realbb->size();
    if (targetops < PolluteSizeFloor) targetops = PolluteSizeFloor;
    if (targetops > PolluteSizeCeiling) targetops = PolluteSizeCeiling;
    if (diamondcloned >= targetops) targetops = 2;
    else targetops -= diamondcloned;

    Value* acc = nullptr;
    IntegerType* accty = nullptr;

    auto pickslot = [&](IntegerType* ity) -> const PolluteSlot&
    {
        auto& slots = (ity == i64) ? slots64 : slots32;
        return slots[rng() % slots.size()];
    };

    for (unsigned n = 0; n < targetops; ++n)
    {
        IntegerType* ity;
        if (have32 && have64) ity = (rng() & 1) ? i64 : i32;
        else ity = have64 ? i64 : i32;

        if (acc && accty != ity) acc = nullptr;

        if (!acc || (rng() % 5) == 0)
        {
            acc = EmitSlotLoad(pollb, pickslot(ity));
            accty = ity;
            continue;
        }

        auto& vals = (ity == i64) ? vals64 : vals32;
        Value* rhs = nullptr;
        const unsigned pick = (unsigned)(rng() % 10u);
        if (pick < 6) rhs = EmitSlotLoad(pollb, pickslot(ity));
        else if (pick < 9 && !vals.empty()) rhs = vals[rng() % vals.size()];
        else rhs = ConstantInt::get(ity, rng());

        // Shl shamt must sit in [1, w-1] because shamt >= w is LLVM poison and shamt == 0 collapses to the identity
        switch (rng() % 7)
        {
            case 0: acc = pollb.CreateAdd(acc, rhs, "RhopqAdd"); break;
            case 1: acc = pollb.CreateSub(acc, rhs, "RhopqSub"); break;
            case 2: acc = pollb.CreateXor(acc, rhs, "RhopqXor"); break;
            case 3: acc = pollb.CreateAnd(acc, rhs, "RhopqAnd"); break;
            case 4: acc = pollb.CreateOr(acc, rhs, "RhopqOr"); break;
            case 5: acc = pollb.CreateMul(acc, rhs, "RhopqMul"); break;
            default:
            {
                const unsigned w = ity->getBitWidth();
                unsigned shamt = (unsigned)(rng() % (w - 1u)) + 1u;
                acc = pollb.CreateShl(acc, ConstantInt::get(ity, shamt), "RhopqShl");
                break;
            }
        }

        if ((rng() % 100) < 15) EmitSlotStore(pollb, pickslot(ity), acc, false);
    }

    if (acc) EmitSlotStore(pollb, pickslot(accty), acc, true);

    pollb.CreateBr(realbb);
}

static Value* BuildPredicate(Function& f, IRBuilder<>& builder, DominatorTree& dt, const polymba::RiggedFamily& fam, bool& outrealontrue)
{
    const unsigned w = fam.Width;
    IntegerType* ity = IntegerType::get(f.getContext(), w);

    Value* x = nullptr;
    if (fam.Kind == polymba::RiggedKind::DZeroOnC)
    {
        x = mbagen::InsertFoldBarrier(builder, ConstantInt::get(ity, fam.KnownC), BarrierState);
    }
    else
    {
        const unsigned auxcount = 2u + (unsigned)(Rng() % 2u);
        std::vector<Value*> aux = mbagen::GatherAuxValues(builder, dt, ity, auxcount, Rng, 0u);
        Value* raw;
        if (aux.empty())
        {
            raw = ConstantInt::get(ity, Rng());
        }
        else
        {
            raw = aux[0];
            for (size_t ai = 1; ai < aux.size(); ++ai) raw = builder.CreateXor(raw, aux[ai], "RhopqXMix");
        }

        // salt mixing only preserves the invariant for variants whose theorem is "for all y, invariant(F(y))" (A/C/E/F) — B's F∘FInv composition and D's F(known_c)=0 don't translate through input XOR and MUST be excluded or the invariant silently breaks at runtime
        const bool saltable =
            fam.Kind == polymba::RiggedKind::ALowByte ||
            fam.Kind == polymba::RiggedKind::CCorrelatedPair ||
            fam.Kind == polymba::RiggedKind::EBitPosition ||
            fam.Kind == polymba::RiggedKind::FImageSet;
        if (saltable)
        {
            EnsureCSalt(*f.getParent());
            Value* saltld = builder.CreateLoad(Type::getInt64Ty(f.getContext()), CSaltGV, true, "RhopqSalt");
            cast<LoadInst>(saltld)->setAlignment(Align(8));
            Value* saltw = saltld;
            if (w != 64) saltw = builder.CreateTrunc(saltld, ity, "RhopqSaltTrunc");
            raw = builder.CreateXor(raw, saltw, "RhopqXSalt");
        }

        x = mbagen::InsertFoldBarrier(builder, raw, BarrierState);
    }

    outrealontrue = true;

    const uint64_t wmask = (w >= 64u) ? ~uint64_t(0) : ((uint64_t(1) << w) - 1u);
    const uint64_t nsite = Rng() & wmask;

    const bool usene = (Rng() & 1u) != 0;
    auto emitcmp = [&](Value* lhs, Value* rhs, const char* name) -> Value*
    {
        return usene ? builder.CreateICmpNE(lhs, rhs, name) : builder.CreateICmpEQ(lhs, rhs, name);
    };

    bool baserealontrue = true;

    Value* result = nullptr;
    switch (fam.Kind)
    {
        case polymba::RiggedKind::ALowByte:
        {
            Value* h = mbagen::EmitPolyHorner(builder, x, fam, BarrierState);
            Value* shifted = builder.CreateAdd(h, ConstantInt::get(ity, nsite), "RhopqASh");
            Value* masked = builder.CreateAnd(shifted, ConstantInt::get(ity, fam.InvariantMask), "RhopqAMask");
            const uint64_t expected = (fam.InvariantValue + nsite) & fam.InvariantMask;
            result = emitcmp(masked, ConstantInt::get(ity, expected), "RhopqACmp");
            mbagen::TagEscalatePoly(builder, result);
            baserealontrue = true;
            break;
        }
        case polymba::RiggedKind::BFInvComp:
        {
            Value* h1 = mbagen::EmitPolyHorner(builder, x, fam, BarrierState);
            Value* h2 = mbagen::EmitPolyHornerFromPoly(builder, h1, fam.FInv, w, BarrierState);
            result = emitcmp(h2, x, "RhopqBCmp");
            baserealontrue = true;
            break;
        }
        case polymba::RiggedKind::CCorrelatedPair:
        {
            Value* h1 = mbagen::EmitPolyHorner(builder, x, fam, BarrierState);
            Value* h2 = mbagen::EmitPolyHornerFromPoly(builder, x, fam.FSib, w, BarrierState);
            Value* sum = builder.CreateAdd(h1, h2, "RhopqCSum");
            Value* kv = mbagen::InsertFoldBarrier(builder, ConstantInt::get(ity, fam.PairK), BarrierState);
            result = emitcmp(sum, kv, "RhopqCCmp");
            baserealontrue = true;
            break;
        }
        case polymba::RiggedKind::DZeroOnC:
        {
            Value* h = mbagen::EmitPolyHorner(builder, x, fam, BarrierState);
            Value* shifted = builder.CreateAdd(h, ConstantInt::get(ity, nsite), "RhopqDSh");
            result = emitcmp(shifted, ConstantInt::get(ity, nsite), "RhopqDCmp");
            mbagen::TagEscalatePoly(builder, result);
            baserealontrue = true;
            break;
        }
        case polymba::RiggedKind::EBitPosition:
        {
            Value* h = mbagen::EmitPolyHorner(builder, x, fam, BarrierState);
            Value* shifted = builder.CreateXor(h, ConstantInt::get(ity, nsite), "RhopqESh");
            Value* shr = builder.CreateLShr(shifted, ConstantInt::get(ity, fam.BitIdx), "RhopqEShr");
            Value* bit = builder.CreateAnd(shr, ConstantInt::get(ity, 1), "RhopqEBit");
            const unsigned expected = fam.BitExpected ^ (unsigned)((nsite >> fam.BitIdx) & 1u);
            result = emitcmp(bit, ConstantInt::get(ity, expected), "RhopqECmp");
            mbagen::TagEscalatePoly(builder, result);
            baserealontrue = true;
            break;
        }
        case polymba::RiggedKind::FImageSet:
        {
            Value* h = mbagen::EmitPolyHorner(builder, x, fam, BarrierState);
            Value* shifted = builder.CreateAdd(h, ConstantInt::get(ity, nsite), "RhopqFSh");
            Value* masked = builder.CreateAnd(shifted, ConstantInt::get(ity, fam.InvariantMask), "RhopqFMask");
            const uint64_t expected = (fam.ImpossibleValue + nsite) & fam.InvariantMask;
            // F is a PF opaque (image-set membership). EQ against the impossible value is always false, so real must flow out of the FALSE edge; setting baserealontrue=true here would route every live execution into the pollute block
            baserealontrue = false;
            result = emitcmp(masked, ConstantInt::get(ity, expected), "RhopqFCmp");
            mbagen::TagEscalatePoly(builder, result);
            break;
        }
        default:
            outrealontrue = true;
            return ConstantInt::getTrue(f.getContext());
    }

    outrealontrue = baserealontrue ^ usene;
    return result;
}

static bool IsEHOrFuncletInterior(BasicBlock* bb)
{
    if (bb->isEHPad()) return true;
    Instruction* term = bb->getTerminator();
    if (!term) return true;
    if (isa<CatchReturnInst>(term) || isa<CleanupReturnInst>(term) || isa<ResumeInst>(term)) return true;
    for (Instruction& i : *bb)
    {
        if (auto* cb = dyn_cast<CallBase>(&i))
        {
            if (cb->countOperandBundlesOfType(LLVMContext::OB_funclet) > 0) return true;
        }
    }
    return false;
}

static unsigned StealIntoHead(BasicBlock* headbb, BasicBlock* realbb, unsigned maxsteal)
{
    unsigned stolen = 0;
    while (stolen < maxsteal)
    {
        Instruction* first = realbb->getFirstNonPHIOrDbgOrLifetime();
        if (!first || first->isTerminator()) break;

        if (isa<CallBase>(first)) break;
        if (isa<AllocaInst>(first)) break;
        if (isa<LandingPadInst>(first)) break;
        if (isa<CatchPadInst>(first)) break;
        if (isa<CleanupPadInst>(first)) break;
        if (first->isAtomic()) break;
        if (isa<FenceInst>(first)) break;
        if (first->mayHaveSideEffects()) break;

        // the operand check must STOP the chain the first time a candidate depends on something still later in realbb; moving such an instruction would hoist a use above its def and the verifier catches it
        bool opsok = true;
        for (Use& op : first->operands())
        {
            Value* v = op.get();
            if (auto* oi = dyn_cast<Instruction>(v))
            {
                if (oi->getParent() == realbb) { opsok = false; break; }
            }
        }
        if (!opsok) break;

        first->removeFromParent();
        first->insertInto(headbb, headbb->end());
        ++stolen;
    }
    return stolen;
}

static bool TryEmitAtBB(Function& f, BasicBlock* bb, DominatorTree& dt)
{
    if (BBAlreadyProcessed(bb)) return false;
    if (SkipEntry && bb == &f.getEntryBlock()) return false;
    if (IsEHOrFuncletInterior(bb)) return false;
    if (bb->size() < MinBBInsts) return false;

    Instruction* splitat = PickInsertionPoint(bb);
    if (!splitat) return false;

    const size_t total = Families32.size() + Families64.size();
    if (total == 0) return false;
    const size_t pick = Rng() % total;
    const polymba::RiggedFamily& fam = (pick < Families32.size()) ? Families32[pick] : Families64[pick - Families32.size()];

    BasicBlock* realbb = bb->splitBasicBlock(splitat, "Cont");

    BasicBlock* pollutebb = BasicBlock::Create(f.getContext(), "Cont", &f, realbb);

    // lastexisting MUST be snapshotted before erasing splitbr so the blanket skip loop can start AFTER the last pre pass instruction; everything emitted below belongs to this pass and must be swept, so the boundary has to be captured first
    Instruction* splitbr = bb->getTerminator();
    Instruction* lastexisting = splitbr->getPrevNode();
    splitbr->eraseFromParent();

    IRBuilder<> builder(bb);

    bool realontrue = true;
    Value* cond = BuildPredicate(f, builder, dt, fam, realontrue);

    FillPolluteBlock(f, pollutebb, realbb, bb, Rng);

    // MarkPolluteBlock MUST run after FillPolluteBlock emits the terminator because the pollute tag attaches to the terminator and setting metadata on a non existent terminator silently drops the tag
    MarkPolluteBlock(pollutebb);

    {
        LLVMContext& c = f.getContext();
        Metadata* skipops[1] = { MDString::get(c, "rhopq_skip") };
        MDNode* skipnode = MDNode::get(c, skipops);

        // preservation is VALUE specific; only the one intentional poly_const tag on the opaque's ICmp survives; const_reactor from InsertFoldBarrier's K_adjs and any other inherited tag on a diamond clone gets overridden to rhopq_skip here, which is why diamond clones strip their inherited metadata earlier
        auto blanketskip = [&](BasicBlock* target, Instruction* afterthis)
        {
            BasicBlock::iterator it = afterthis ? std::next(BasicBlock::iterator(afterthis)) : target->begin();
            for (; it != target->end(); ++it)
            {
                MDNode* existing = it->getMetadata("mbagen.escalate");
                if (existing && existing->getNumOperands() > 0)
                {
                    if (auto* ms = dyn_cast<MDString>(existing->getOperand(0).get()))
                    {
                        if (ms->getString() == "poly_const") continue;
                    }
                }
                it->setMetadata("mbagen.escalate", skipnode);
            }
        };

        blanketskip(bb, lastexisting);
        blanketskip(pollutebb, nullptr);
    }

    // StealIntoHead MUST run after the blanket skip so stolen application instructions are NOT tagged rhopq_skip; they are real user code and must be ConstMBA'd normally; reversing the order silences every stolen instruction and tanks coverage
    unsigned stealcap = 0;
    {
        const unsigned rbsize = (unsigned)realbb->size();
        if (rbsize > RealBBMinRemain)
        {
            const unsigned headroom = rbsize - RealBBMinRemain;
            const unsigned ceiling = std::min(StealCeiling, headroom);
            stealcap = 1u + (unsigned)(Rng() % ceiling);
            const unsigned floor = std::min(StealFloor, ceiling);
            if (stealcap < floor) stealcap = floor;
        }
    }
    if (stealcap > 0) (void)StealIntoHead(bb, realbb, stealcap);

    builder.SetInsertPoint(bb);
    if (realontrue) builder.CreateCondBr(cond, realbb, pollutebb);
    else builder.CreateCondBr(cond, pollutebb, realbb);

    MarkBBProcessed(bb);

    return true;
}

static BasicBlock* RealBBFromSplit(BasicBlock* bb)
{
    Instruction* t = bb->getTerminator();
    auto* br = dyn_cast<BranchInst>(t);
    if (!br || !br->isConditional()) return nullptr;
    for (unsigned i = 0; i < br->getNumSuccessors(); ++i)
    {
        BasicBlock* s = br->getSuccessor(i);
        if (!IsPolluteBlock(s)) return s;
    }
    return nullptr;
}

static bool ProcessFunction(Function& f)
{
    if (f.isDeclaration()) return false;
    if (HasNoObfuscateAnnotation(f) || HasNoRHOPQAnnotation(f)) return false;
    if (!HasObfuscateAnnotation(f)) return false;

    SmallVector<BasicBlock*, 64> worklist;
    SmallPtrSet<BasicBlock*, 32> seen;
    for (BasicBlock& bb : f)
    {
        worklist.push_back(&bb);
        seen.insert(&bb);
    }

    unsigned emitted = 0;
    bool mutated = false;

    while (!worklist.empty() && emitted < PerFunctionCap)
    {
        BasicBlock* bb = worklist.pop_back_val();

        if (IsPolluteBlock(bb)) continue;

        DominatorTree dt;
        dt.recalculate(f);
        LoopInfo li;
        li.analyze(dt);
        if (SkipLoops && li.getLoopFor(bb) != nullptr) continue;

        if (!TryEmitAtBB(f, bb, dt)) continue;

        ++emitted;
        mutated = true;

        if (BasicBlock* realbb = RealBBFromSplit(bb))
        {
            if (seen.insert(realbb).second) worklist.push_back(realbb);
        }
    }

    return mutated;
}

static void BuildCatalogues(Module& m)
{
    Families32.clear();
    Families64.clear();

    polymba::RiggedBuildOptions common;
    common.CountA = CountA;
    common.CountB = CountB;
    common.CountC = CountC;
    common.CountD = CountD;
    common.CountE = CountE;
    common.CountF = CountF;
    common.LowBitsK = LowBitsK;
    common.BitIdxMin = BitIdxMin;
    common.BitIdxMax = BitIdxMax;
    common.ImageEnumSamples = ImageEnumSamples;
    common.ImageMaskBits = ImageMaskBits;
    common.InverseCapD = InverseCapD;
    common.VerifyBits = VerifyBits;
    common.VerifyRandom = VerifyRandom;
    common.RiggedVerifySamples = RiggedVerifySamples;
    common.DegreeMixPct = DegreeMixPct;

    if (Wrap64)
    {
        auto opt = common;
        opt.Width = 64;
        opt.Seed = Rng();
        Families64 = polymba::BuildRiggedFamilies(opt);
    }
    if (Wrap32)
    {
        auto opt = common;
        opt.Width = 32;
        opt.Seed = Rng();
        // subvariant E draws a bit index below width, so BitIdxMax must be clamped to w1 before the i32 catalogue builds; leaving it at the i64 default of 48 would let E produce out of range shifts on i32 and emit poison
        if (opt.BitIdxMax >= opt.Width) opt.BitIdxMax = opt.Width - 1u;
        if (opt.BitIdxMin > opt.BitIdxMax) opt.BitIdxMin = opt.BitIdxMax;
        Families32 = polymba::BuildRiggedFamilies(opt);
    }

    (void)m;
}

bool RunRiggedHornerOpaque(Module& m)
{
    if (!RiggedHornerOpaqueEnable) return false;

    std::random_device rd;
    uint64_t seed = (uint64_t)rd();
    seed ^= (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    seed ^= ((uint64_t)rd()) << 32;
    seed ^= 0x52484F50515F5F5FULL;
    Rng.seed(seed);

    // CSaltGV is module scoped; the pointer MUST be reset per run so EnsureCSalt creates a new slot in THIS module when a C/E/F/A variant first demands it; carrying a stale pointer from a previous module would attempt to reach a GV that has already been reclaimed
    CSaltGV = nullptr;

    uint64_t barrierseed = Rng();
    if (barrierseed == 0) barrierseed = 0x5248425152494747ULL;
    BarrierState.Init(&m, BarrierPoolSize, barrierseed, BarrierWeightLegacy, BarrierWeightXor, BarrierWeightAdd, BarrierWeightSub, ".mba.barrier.rhopq");

    BuildCatalogues(m);
    if (Families32.empty() && Families64.empty()) return false;

    bool mutated = false;
    for (Function& f : m) mutated |= ProcessFunction(f);

    mbagen::BarrierFinalizeModule(BarrierState);

    Families32.clear();
    Families64.clear();
    return mutated;
}

PreservedAnalyses RiggedHornerOpaquePass::run(Module& m, ModuleAnalysisManager&)
{
    bool mutated = RunRiggedHornerOpaque(m);
    return mutated ? PreservedAnalyses::none() : PreservedAnalyses::all();
}