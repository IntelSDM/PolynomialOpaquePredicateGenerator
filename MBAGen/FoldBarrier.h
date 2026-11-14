#pragma once

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <cassert>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace mbagen
{

enum class BarrierShape : uint8_t
{
    Legacy = 0,
    XorHidden = 1,
    AddHidden = 2,
    SubHidden = 3,
};

struct BarrierSlot
{
    llvm::GlobalVariable* GV = nullptr;
    llvm::GlobalVariable* NonceGV = nullptr;
    llvm::GlobalVariable* CipherGV = nullptr;
    uint64_t K = 0;
    uint64_t Nonce = 0;
    uint64_t Cipher = 0;
};

struct ModuleBarrierState
{
    llvm::Module* M = nullptr;
    unsigned PoolSize = 16;
    std::vector<BarrierSlot> SlotsI32;
    std::vector<BarrierSlot> SlotsI64;
    std::mt19937_64 Rng;
    unsigned WLegacy = 0;
    unsigned WXor = 50;
    unsigned WAdd = 25;
    unsigned WSub = 25;
    std::string SlotNamePrefix = ".mba.barrier";

    llvm::GlobalVariable* SeedGV = nullptr;
    llvm::GlobalVariable* SeedBootNonceGV = nullptr;
    llvm::GlobalVariable* SeedBootSaltGV = nullptr;
    uint64_t SeedK = 0;
    uint64_t SeedBootNonce = 0;
    uint64_t SeedBootSalt = 0;
    bool InitEmitted = false;

    void Init(llvm::Module* module, unsigned poolsz, uint64_t seed, unsigned wleg, unsigned wx, unsigned wa, unsigned ws, const std::string& nameprefix)
    {
        M = module;
        PoolSize = (poolsz == 0) ? 1u : poolsz;
        SlotsI32.assign(PoolSize, BarrierSlot{});
        SlotsI64.assign(PoolSize, BarrierSlot{});
        Rng.seed(seed ? seed : (uint64_t)(uintptr_t)module ^ 0x9E3779B97F4A7C15ull);
        WLegacy = wleg;
        WXor = wx;
        WAdd = wa;
        WSub = ws;
        SlotNamePrefix = nameprefix;

        SeedGV = nullptr;
        SeedBootNonceGV = nullptr;
        SeedBootSaltGV = nullptr;
        SeedBootNonce = Rng() | 1ull;
        SeedBootSalt = Rng() | 1ull;
        SeedK = 0;
        InitEmitted = false;
    }
};

static inline uint64_t BarrierArxMix(uint64_t nonce, uint64_t seed)
{
    uint64_t x = nonce ^ seed;
    x = ((x << 13) | (x >> 51)) + 0x9E3779B97F4A7C15ULL;
    x = x ^ ((x << 7) | (x >> 57));
    x = ((x << 17) | (x >> 47)) + seed;
    x = x ^ nonce;
    x = ((x << 23) | (x >> 41)) * 0xBF58476D1CE4E5B9ULL;
    return x;
}

static BarrierShape BarrierPickShape(ModuleBarrierState& st)
{
    const unsigned total = st.WLegacy + st.WXor + st.WAdd + st.WSub;
    if (total == 0) return BarrierShape::Legacy;
    unsigned r = (unsigned)(st.Rng() % total);
    if (r < st.WLegacy) return BarrierShape::Legacy;
    r -= st.WLegacy;
    if (r < st.WXor) return BarrierShape::XorHidden;
    r -= st.WXor;
    if (r < st.WAdd) return BarrierShape::AddHidden;
    return BarrierShape::SubHidden;
}

static uint64_t BarrierDrawK(ModuleBarrierState& st, unsigned w)
{
    const uint64_t mask = (w >= 64) ? ~uint64_t(0) : ((uint64_t(1) << w) - 1);
    uint64_t k = 0;
    do
    {
        k = st.Rng() & mask;
    } while (k == 0 || k == 1 || k == 2 || k == mask || k == (mask - 1) || k == (mask - 2));
    return k;
}

static llvm::GlobalVariable* BarrierCreateConstDataGV(llvm::Module* m, llvm::IntegerType* ity, uint64_t value, const std::string& name)
{
    using namespace llvm;
    GlobalVariable* gv = new GlobalVariable(*m, ity, false, GlobalValue::InternalLinkage, ConstantInt::get(ity, value), name);
    gv->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
    gv->setAlignment(Align(ity->getBitWidth() / 8));
    return gv;
}

static void BarrierEnsureSeedGVs(ModuleBarrierState& st)
{
    using namespace llvm;
    if (st.SeedGV) return;
    LLVMContext& c = st.M->getContext();
    IntegerType* i64 = IntegerType::get(c, 64);

    st.SeedK = BarrierArxMix(st.SeedBootNonce, st.SeedBootSalt);

    st.SeedGV = new GlobalVariable(*st.M, i64, false, GlobalValue::InternalLinkage, ConstantInt::get(i64, 0), st.SlotNamePrefix + ".seed");
    st.SeedGV->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
    st.SeedGV->setAlignment(Align(8));

    st.SeedBootNonceGV = BarrierCreateConstDataGV(st.M, i64, st.SeedBootNonce, st.SlotNamePrefix + ".seed.nonce");
    st.SeedBootSaltGV = BarrierCreateConstDataGV(st.M, i64, st.SeedBootSalt, st.SlotNamePrefix + ".seed.salt");
}

static BarrierSlot& BarrierGetOrCreateSlot(ModuleBarrierState& st, llvm::IntegerType* ity, unsigned idx)
{
    using namespace llvm;
    const unsigned w = ity->getBitWidth();
    auto& pool = (w == 32) ? st.SlotsI32 : st.SlotsI64;
    unsigned i = idx % (unsigned)pool.size();
    BarrierSlot& slot = pool[i];
    if (slot.GV) return slot;

    BarrierEnsureSeedGVs(st);

    slot.K = BarrierDrawK(st, w);
    slot.Nonce = st.Rng() | 1ull;
    const uint64_t derived = BarrierArxMix(slot.Nonce, st.SeedK);
    const uint64_t wmask = (w >= 64) ? ~uint64_t(0) : ((uint64_t(1) << w) - 1);
    slot.Cipher = (slot.K ^ derived) & wmask;

    const std::string base = st.SlotNamePrefix + ".g" + std::to_string(w) + "." + std::to_string(i);

    slot.GV = new GlobalVariable(*st.M, ity, false, GlobalValue::InternalLinkage, ConstantInt::get(ity, 0), base);
    slot.GV->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
    slot.GV->setAlignment(Align(w / 8));

    IntegerType* i64 = IntegerType::get(st.M->getContext(), 64);
    slot.NonceGV = BarrierCreateConstDataGV(st.M, i64, slot.Nonce, base + ".nonce");
    slot.CipherGV = BarrierCreateConstDataGV(st.M, ity, slot.Cipher, base + ".cipher");

    return slot;
}

static unsigned BarrierPickSecondSlotIndex(ModuleBarrierState& st, unsigned firstidx)
{
    const unsigned pool = st.PoolSize;
    if (pool < 2) return firstidx;

    const unsigned off = 1u + (unsigned)(st.Rng() % (pool - 1u));
    return (firstidx + off) % pool;
}

static void BarrierTagConstBarrierEscalate(llvm::IRBuilder<>& builder, llvm::Value* v)
{
    using namespace llvm;
    auto* inst = dyn_cast<Instruction>(v);
    if (!inst) return;
    LLVMContext& c = builder.getContext();
    MDNode* md = MDNode::get(c, { MDString::get(c, "const_barrier") });
    inst->setMetadata("mbagen.escalate", md);
}

static llvm::LoadInst* BarrierEmitVolatileLoad(llvm::IRBuilder<>& builder, llvm::GlobalVariable* gv, llvm::IntegerType* ity, const char* name)
{
    using namespace llvm;
    LoadInst* ld = builder.CreateLoad(ity, gv, true, name);
    ld->setAlignment(Align(ity->getBitWidth() / 8));
    return ld;
}

inline llvm::Value* InsertFoldBarrierLegacy(llvm::IRBuilder<>& builder, llvm::Value* v)
{
    using namespace llvm;
    Type* ty = v->getType();
    assert(ty->isIntegerTy() && "FoldBarrier only supports integer types");
    FunctionType* ft = FunctionType::get(ty, { ty }, false);
    InlineAsm* ia = InlineAsm::get(ft, "", "=r,0", true, false, InlineAsm::AD_ATT);
    return builder.CreateCall(ia, { v }, "MBABarrierLeg");
}

inline llvm::Value* InsertFoldBarrier(llvm::IRBuilder<>& builder, llvm::Value* v, ModuleBarrierState& st)
{
    using namespace llvm;
    Type* ty = v->getType();
    assert(ty->isIntegerTy() && "FoldBarrier only supports integer types");
    auto* ity = cast<IntegerType>(ty);
    const unsigned w = ity->getBitWidth();

    if (w != 32 && w != 64) return InsertFoldBarrierLegacy(builder, v);

    BarrierShape shape = BarrierPickShape(st);
    if (shape == BarrierShape::Legacy) return InsertFoldBarrierLegacy(builder, v);

    const unsigned slotidxa = (unsigned)(st.Rng() % st.PoolSize);
    const unsigned slotidxb = BarrierPickSecondSlotIndex(st, slotidxa);
    BarrierSlot& slota = BarrierGetOrCreateSlot(st, ity, slotidxa);
    
    BarrierSlot& slotb = (st.PoolSize >= 2 && slotidxb != slotidxa) ? BarrierGetOrCreateSlot(st, ity, slotidxb) : slota;

    const bool dualslot = (&slota != &slotb);
    const uint64_t wmask = (w >= 64) ? ~uint64_t(0) : ((uint64_t(1) << w) - 1);

    LoadInst* lda = BarrierEmitVolatileLoad(builder, slota.GV, ity, "MBABarrierLdA");
    LoadInst* ldb = dualslot ? BarrierEmitVolatileLoad(builder, slotb.GV, ity, "MBABarrierLdB") : nullptr;

    Value* result = nullptr;
    switch (shape)
    {
        case BarrierShape::XorHidden:
        {
            Value* t1 = builder.CreateXor(v, lda, "MBABarrierX1");
            if (dualslot) t1 = builder.CreateXor(t1, ldb, "MBABarrierX2");
          
            const uint64_t kadj = dualslot ? ((slota.K ^ slotb.K) & wmask) : (slota.K & wmask);
            Value* kc = ConstantInt::get(ity, kadj);
            Value* t2 = builder.CreateXor(t1, kc, "MBABarrierXor");
            BarrierTagConstBarrierEscalate(builder, t2);
            result = t2;
            break;
        }
        case BarrierShape::AddHidden:
        {
            Value* t1 = builder.CreateAdd(v, lda, "MBABarrierA1");
            if (dualslot) t1 = builder.CreateAdd(t1, ldb, "MBABarrierA2");
            const uint64_t kadj = dualslot ? ((slota.K + slotb.K) & wmask) : (slota.K & wmask);
            Value* kc = ConstantInt::get(ity, kadj);
            Value* t2 = builder.CreateSub(t1, kc, "MBABarrierAdd");
            BarrierTagConstBarrierEscalate(builder, t2);
            result = t2;
            break;
        }
        case BarrierShape::SubHidden:
        {
            Value* t1 = builder.CreateSub(v, lda, "MBABarrierS1");
            if (dualslot) t1 = builder.CreateSub(t1, ldb, "MBABarrierS2");
            const uint64_t kadj = dualslot ? ((slota.K + slotb.K) & wmask) : (slota.K & wmask);
            Value* kc = ConstantInt::get(ity, kadj);
            Value* t2 = builder.CreateAdd(t1, kc, "MBABarrierSub");
            BarrierTagConstBarrierEscalate(builder, t2);
            result = t2;
            break;
        }
        case BarrierShape::Legacy:
        default:
            return InsertFoldBarrierLegacy(builder, v);
    }

    return result;
}

inline llvm::Value* InsertFoldBarrier(llvm::IRBuilder<>& builder, llvm::Value* v)
{
    return InsertFoldBarrierLegacy(builder, v);
}

static llvm::Value* BarrierEmitArxBody(llvm::IRBuilder<>& builder, llvm::Value* nonce, llvm::Value* seed)
{
    using namespace llvm;
    LLVMContext& c = builder.getContext();
    IntegerType* i64 = IntegerType::get(c, 64);

    auto rotl = [&](Value* v, unsigned n) -> Value*
    {
        Value* lo = builder.CreateShl(v, ConstantInt::get(i64, n));
        Value* hi = builder.CreateLShr(v, ConstantInt::get(i64, 64 - n));
        return builder.CreateOr(lo, hi);
    };

    Value* x = builder.CreateXor(nonce, seed);
    x = builder.CreateAdd(rotl(x, 13), ConstantInt::get(i64, 0x9E3779B97F4A7C15ULL));
    x = builder.CreateXor(x, rotl(x, 7));
    x = builder.CreateAdd(rotl(x, 17), seed);
    x = builder.CreateXor(x, nonce);
    x = builder.CreateMul(rotl(x, 23), ConstantInt::get(i64, 0xBF58476D1CE4E5B9ULL));
    return x;
}

static void BarrierEmitInitBody(llvm::IRBuilder<>& builder, ModuleBarrierState& st)
{
    using namespace llvm;
    LLVMContext& c = builder.getContext();
    IntegerType* i64 = IntegerType::get(c, 64);

    LoadInst* bootnoncev = BarrierEmitVolatileLoad(builder, st.SeedBootNonceGV, i64, "BarrierBootNonce");
    LoadInst* bootsaltv = BarrierEmitVolatileLoad(builder, st.SeedBootSaltGV, i64, "BarrierBootSalt");
    Value* seedval = BarrierEmitArxBody(builder, bootnoncev, bootsaltv);
    StoreInst* seedstore = builder.CreateStore(seedval, st.SeedGV);
    seedstore->setVolatile(true);
    seedstore->setAlignment(Align(8));

    LoadInst* seedld = BarrierEmitVolatileLoad(builder, st.SeedGV, i64, "BarrierSeed");

    IntegerType* i32 = IntegerType::get(c, 32);

    auto emitslotstore = [&](const BarrierSlot& slot, IntegerType* ity)
    {
        if (!slot.GV || !slot.NonceGV || !slot.CipherGV) return;
        const unsigned w = ity->getBitWidth();
        LoadInst* nonceld = BarrierEmitVolatileLoad(builder, slot.NonceGV, i64, "BarrierNonce");
        Value* derived = BarrierEmitArxBody(builder, nonceld, seedld);
        if (w != 64) derived = builder.CreateTrunc(derived, ity, "BarrierDerivedTrunc");
        LoadInst* cipherld = BarrierEmitVolatileLoad(builder, slot.CipherGV, ity, "BarrierCipher");
        Value* kval = builder.CreateXor(cipherld, derived, "BarrierK");
        StoreInst* kst = builder.CreateStore(kval, slot.GV);
        kst->setVolatile(true);
        kst->setAlignment(Align(w / 8));
    };

    for (const BarrierSlot& s : st.SlotsI64) emitslotstore(s, i64);
    for (const BarrierSlot& s : st.SlotsI32) emitslotstore(s, i32);
}

inline void BarrierFinalizeModule(ModuleBarrierState& st)
{
    using namespace llvm;
    if (!st.M || st.InitEmitted) return;
    bool anyslot = false;
    for (const auto& s : st.SlotsI32) if (s.GV) { anyslot = true; break; }
    if (!anyslot) for (const auto& s : st.SlotsI64) if (s.GV) { anyslot = true; break; }

    if (!anyslot) return;

    LLVMContext& c = st.M->getContext();
    FunctionType* ft = FunctionType::get(Type::getVoidTy(c), false);
    const std::string fname = "WPBarrierInit" + st.SlotNamePrefix;
    Function* initfn = Function::Create(ft, GlobalValue::InternalLinkage, fname, st.M);
    
    initfn->addFnAttr(Attribute::NoInline);
    BasicBlock* entry = BasicBlock::Create(c, "Entry", initfn);
    IRBuilder<> ib(entry);
    BarrierEmitInitBody(ib, st);
    ib.CreateRetVoid();

   
    appendToGlobalCtors(*st.M, initfn, 101);

    st.InitEmitted = true;
}

}