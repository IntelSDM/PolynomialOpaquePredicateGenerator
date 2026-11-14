#pragma once

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"

#include <cassert>

namespace polymba
{

inline llvm::Instruction* FindLatestDecodeInsertionPoint(const llvm::SmallVectorImpl<llvm::Instruction*>& users, llvm::DominatorTree& dt)
{
    assert(!users.empty() && "no users means no site should have been emitted");

#ifndef NDEBUG
    // PHI users must be filtered at site collection; a PHI's "use" is logically at the end of the matching incoming predecessor, not at the PHI itself, so per-incoming-edge decode insertion would be required and this layer does not support that
    for (llvm::Instruction* u : users)
        assert(!llvm::isa<llvm::PHINode>(u) && "PHI users must be filtered at site collection");
#endif

    llvm::BasicBlock* firstbb = users.front()->getParent();
    bool allsame = true;
    for (llvm::Instruction* u : users)
    {
        if (u->getParent() != firstbb) { allsame = false; break; }
    }
    if (allsame)
    {
        llvm::Instruction* earliest = users.front();
        for (llvm::Instruction* u : users)
            if (u->comesBefore(earliest)) earliest = u;
        return earliest;
    }

    llvm::BasicBlock* dom = users.front()->getParent();
    for (size_t i = 1; i < users.size(); ++i)
    {
        dom = dt.findNearestCommonDominator(dom, users[i]->getParent());
        assert(dom && "function CFG lacks a common dominator for user set");
    }

    // if the dominator block itself contains any users, insert before the earliest in-block user (strictly later than inserting at the terminator would be); only when the dominator holds no user do we fall through to its terminator
    llvm::Instruction* earliestindom = nullptr;
    for (llvm::Instruction* u : users)
    {
        if (u->getParent() != dom) continue;
        if (!earliestindom || u->comesBefore(earliestindom))
            earliestindom = u;
    }
    if (earliestindom) return earliestindom;

    return dom->getTerminator();
}

}