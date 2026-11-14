#pragma once

#include "FoldBarrier.h"
#include "PolyIREmit.h"
#include "PolyMBA.h"
#include "RiggedHornerCatalog.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"

#include <cassert>

namespace mbagen
{

static void TagRhopqSkip(llvm::IRBuilder<>& builder, llvm::Value* v)
{
    if (auto* i = llvm::dyn_cast<llvm::Instruction>(v))
    {
        llvm::LLVMContext& c = builder.getContext();
        llvm::Metadata* ops[1] = { llvm::MDString::get(c, "rhopq_skip") };
        i->setMetadata("mbagen.escalate", llvm::MDNode::get(c, ops));
    }
}

inline llvm::Value* EmitPolyHornerFromPoly(llvm::IRBuilder<>& builder, llvm::Value* x, const polymba::Poly& p, unsigned w, ModuleBarrierState& barrierstate)
{
    auto* ity = llvm::dyn_cast<llvm::IntegerType>(x->getType());
    assert(ity && ity->getBitWidth() == w && "EmitPolyHornerFromPoly width mismatch");

    // trailing zero coefficients MUST be trimmed before the Horner walk because FInv from CompositionalInverse at D=80 commonly carries zero coefficients past the family's effective degree, and without the trim the chain emits a long tail of (acc*x + 0) multiplications that blow up IR size for no obfuscation gain
    size_t deg = p.C.size();
    while (deg > 1 && p.C[deg - 1] == 0) --deg;

    // each a_i MUST go through InsertFoldBarrier so the Mul/Add operand is a Value (barrier output) rather than a ConstantInt; a raw ConstantInt operand would let LLVM ConstantFold the Horner step at compile time and leak the family coefficients into the disassembly
    llvm::Value* acc = InsertFoldBarrier(builder, llvm::ConstantInt::get(ity, p.C[deg - 1]), barrierstate);
    for (size_t k = deg - 1; k-- > 0; )
    {
        llvm::Value* mul = builder.CreateMul(acc, x, "RHornerM");
        llvm::Value* ck = InsertFoldBarrier(builder, llvm::ConstantInt::get(ity, p.C[k]), barrierstate);
        acc = builder.CreateAdd(mul, ck, "RHornerA");
    }

    return InsertFoldBarrier(builder, acc, barrierstate);
}

inline llvm::Value* EmitPolyHorner(llvm::IRBuilder<>& builder, llvm::Value* x, const polymba::RiggedFamily& fam, ModuleBarrierState& barrierstate)
{
    return EmitPolyHornerFromPoly(builder, x, fam.F, fam.Width, barrierstate);
}

}