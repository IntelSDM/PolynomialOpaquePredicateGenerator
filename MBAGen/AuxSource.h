#pragma once

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include <algorithm>
#include <random>
#include <vector>

namespace mbagen
{

inline constexpr unsigned AuxPoolSlots = 8;

inline llvm::DenseMap<llvm::Function*, llvm::AllocaInst*> AuxPoolCaches;

struct AuxHarvestCache
{
    llvm::SmallVector<llvm::Value*, 64> I32;
    llvm::SmallVector<llvm::Value*, 64> I64;
    bool Built = false;
};

inline llvm::DenseMap<llvm::Function*, AuxHarvestCache> AuxHarvestCaches;

inline void ResetAuxPoolCache()
{
    AuxPoolCaches.clear();
    AuxHarvestCaches.clear();
}

// per function cache of every integer-typed Value (args + instructions) grouped by bitwidth, built once per function on first use; the uncached whole function scan ran O(sites*insns) per call and dominated ConstMBA runtime on realistic modules, so ResetAuxPoolCache MUST invalidate both pools at pass start or stale entries point into reclaimed functions
inline const AuxHarvestCache& GetOrBuildHarvestCache(llvm::Function& f)
{
    auto& cache = AuxHarvestCaches[&f];
    if (cache.Built) return cache;
    for (llvm::Argument& a : f.args())
    {
        if (a.getType()->isIntegerTy(32)) cache.I32.push_back(&a);
        else if (a.getType()->isIntegerTy(64)) cache.I64.push_back(&a);
    }
    for (llvm::BasicBlock& bb : f)
    {
        for (llvm::Instruction& i : bb)
        {
            if (i.getType()->isIntegerTy(32)) cache.I32.push_back(&i);
            else if (i.getType()->isIntegerTy(64)) cache.I64.push_back(&i);
        }
    }
    cache.Built = true;
    return cache;
}

inline llvm::AllocaInst* GetOrCreateAuxPool(llvm::Function& f, std::mt19937_64& rng)
{
    using namespace llvm;

    auto it = AuxPoolCaches.find(&f);
    if (it != AuxPoolCaches.end()) return it->second;

    BasicBlock& entry = f.getEntryBlock();
    BasicBlock::iterator insertpt = entry.getFirstInsertionPt();

    LLVMContext& c = f.getContext();
    IRBuilder<> builder(&entry, insertpt);

    Type* i64 = Type::getInt64Ty(c);
    ArrayType* poolty = ArrayType::get(i64, AuxPoolSlots);

    AllocaInst* pool = builder.CreateAlloca(poolty, nullptr, "MBAGenAuxPool");
    pool->setAlignment(Align(8));

    Type* i32 = Type::getInt32Ty(c);
    for (unsigned i = 0; i < AuxPoolSlots; ++i)
    {
        Value* idx[2] = { ConstantInt::get(i32, 0), ConstantInt::get(i32, i) };
        Value* slotptr = builder.CreateInBoundsGEP(poolty, pool, idx, "MBAGenAuxSlot");
        uint64_t init = rng();
        // stores MUST be volatile; without the volatile flag InstCombine propagates the stored initial constants through to every volatile load at use sites, collapsing the aux pool at compile time and leaving the MBA expression as a visible constant
        StoreInst* st = builder.CreateStore(ConstantInt::get(i64, init), slotptr);
        st->setVolatile(true);
        st->setAlignment(Align(8));
    }

    AuxPoolCaches[&f] = pool;
    return pool;
}

// pull the per-function harvest cache instead of walking every BB on every call; dominance filter still runs per candidate but on the cached width list rather than the whole function; O(sites*insns) collapsed to O(sites*cached_width_list) which was the dominant cost on ConstMBA-heavy modules
inline std::vector<llvm::Value*> HarvestLiveIntegerValues(llvm::Instruction* insertionpoint, const llvm::DominatorTree& dt, llvm::IntegerType* requiredtype, unsigned maxcount)
{
    using namespace llvm;

    std::vector<Value*> candidates;
    candidates.reserve(maxcount);

    Function* f = insertionpoint->getFunction();
    if (!f) return candidates;

    const unsigned bw = requiredtype->getBitWidth();
    const AuxHarvestCache& cache = GetOrBuildHarvestCache(*f);
    const auto& pool = (bw == 32) ? cache.I32 : (bw == 64) ? cache.I64 : llvm::SmallVector<llvm::Value*, 64>{};

    for (Value* v : pool)
    {
        if (candidates.size() >= maxcount) break;
        if (v == insertionpoint) continue;
        if (!isa<Argument>(v))
        {
            auto* vi = cast<Instruction>(v);
            if (!dt.dominates(vi, insertionpoint)) continue;
        }
        candidates.push_back(v);
    }

    return candidates;
}

inline llvm::Value* EmitAuxPoolLoad(llvm::IRBuilder<>& builder, llvm::AllocaInst* pool, llvm::IntegerType* resulttype, unsigned slotidx)
{
    using namespace llvm;

    LLVMContext& c = builder.getContext();
    Type* i32 = Type::getInt32Ty(c);
    Type* i64 = Type::getInt64Ty(c);

    ArrayType* poolty = ArrayType::get(i64, AuxPoolSlots);
    Value* idx[2] = { ConstantInt::get(i32, 0), ConstantInt::get(i32, slotidx % AuxPoolSlots) };
    Value* slotptr = builder.CreateInBoundsGEP(poolty, pool, idx, "MBAGenAuxSlot");

    LoadInst* ld = builder.CreateLoad(i64, slotptr, "MBAGenAuxLoad");
    ld->setVolatile(true);
    ld->setAlignment(Align(8));

    if (resulttype == i64) return ld;
    // the trunc here sits in the backfill path only, it does NOT contaminate the MBA shape proper (which sees a plain integer-typed Value*); auto zext/sext/trunc on the Tier A harvest path was deliberately excluded for exactly this reason
    return builder.CreateTrunc(ld, resulttype, "MBAGenAuxTrunc");
}

inline std::vector<llvm::Value*> GatherAuxValues(llvm::IRBuilder<>& builder, const llvm::DominatorTree& dt, llvm::IntegerType* resulttype, unsigned v, std::mt19937_64& rng, unsigned crosswidthpct = 0)
{
    using namespace llvm;

    if (crosswidthpct > 100) crosswidthpct = 100;

    Instruction* insertpt = &*builder.GetInsertPoint();
    std::vector<Value*> harvested = HarvestLiveIntegerValues(insertpt, dt, resulttype, v * 3);

    std::vector<Value*> crossharvested;
    IntegerType* crosstype = nullptr;
    if (crosswidthpct > 0)
    {
        LLVMContext& c = builder.getContext();
        if (resulttype->getBitWidth() == 32) crosstype = Type::getInt64Ty(c);
        else if (resulttype->getBitWidth() == 64) crosstype = Type::getInt32Ty(c);
        if (crosstype)
        {
            crossharvested = HarvestLiveIntegerValues(insertpt, dt, crosstype, v * 3);
            if (crossharvested.size() > 1) std::shuffle(crossharvested.begin(), crossharvested.end(), rng);
        }
    }

    // function arguments always rank first in the harvest (dominance by construction), and without the shuffle that ordering becomes a static tell; "first V aux slots are always function args"; so an attacker can scan for exactly that header in each MBA site
    if (harvested.size() > 1) std::shuffle(harvested.begin(), harvested.end(), rng);

    std::vector<Value*> result;
    result.reserve(v);

    const unsigned fromharvest = std::min(v, (unsigned)harvested.size());
    size_t crosscursor = 0;
    unsigned samecursor = 0;
    while (result.size() < fromharvest || (crosswidthpct > 0 && crosscursor < crossharvested.size() && result.size() < v))
    {
        if (result.size() >= v) break;
        const bool havesame = (samecursor < harvested.size());
        const bool havecross = (crosswidthpct > 0 && crosscursor < crossharvested.size());
        const bool rollcross = havecross && ((unsigned)(rng() % 100u) < crosswidthpct);
        if (rollcross)
        {
            Value* cv = crossharvested[crosscursor++];
            Value* bridged;
            if (crosstype->getBitWidth() > resulttype->getBitWidth()) bridged = builder.CreateTrunc(cv, resulttype, "MBAGenAuxCrossTrunc");
            else bridged = builder.CreateZExt(cv, resulttype, "MBAGenAuxCrossZext");
            result.push_back(bridged);
        }
        else if (havesame)
        {
            result.push_back(harvested[samecursor++]);
        }
        else
        {
            break;
        }
    }

    if (result.size() < v)
    {
        Function* f = insertpt->getFunction();
        AllocaInst* pool = GetOrCreateAuxPool(*f, rng);

        unsigned slot = (unsigned)(rng() % AuxPoolSlots);
        while (result.size() < v)
        {
            result.push_back(EmitAuxPoolLoad(builder, pool, resulttype, slot));
            slot = (slot + 1) % AuxPoolSlots;
        }
    }

    return result;
}

}