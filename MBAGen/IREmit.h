#pragma once

#include "ConstMBA.h"
#include "LinearMBA.h"
#include "MBABasis.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include <cassert>

namespace mbagen
{

inline llvm::Value* EmitAtomTreeIR(llvm::IRBuilder<>& builder, const AtomTree& t, llvm::ArrayRef<llvm::Value*> inputs, llvm::IntegerType* resulttype)
{
    using namespace llvm;

    switch (t.Op)
    {
        case AtomOp::Var:
        {
            assert(t.VarIdx < inputs.size() && "AtomTree Var index out of range");
            Value* v = inputs[t.VarIdx];
            assert(v->getType() == resulttype && "AtomTree input has wrong width for this emit");
            return v;
        }
        case AtomOp::Const:
        {
            const unsigned bw = resulttype->getBitWidth();
            const u64 wm = (bw >= 64) ? ~u64(0) : ((u64(1) << bw) - 1);
            // RewriteIdentities only produces Const(0) and Const(all-ones) in practice, so this path is almost always a Const(-1) trailing residue; LLVM constant folds the outer `coef * ConstantInt(-1)` to one terminal add imm at the sum accumulator, which is the one-terminal constant invariant the hardening plan relies on
            return ConstantInt::get(resulttype, t.ConstVal & wm);
        }
        case AtomOp::Not:
        {
            assert(t.Kids.size() == 1 && "Not arity != 1");
            Value* k = EmitAtomTreeIR(builder, *t.Kids[0], inputs, resulttype);
            return builder.CreateNot(k, "MBANot");
        }
        case AtomOp::And:
        {
            assert(!t.Kids.empty() && "And with no kids");
            Value* acc = EmitAtomTreeIR(builder, *t.Kids[0], inputs, resulttype);
            for (size_t i = 1; i < t.Kids.size(); ++i)
            {
                Value* k = EmitAtomTreeIR(builder, *t.Kids[i], inputs, resulttype);
                acc = builder.CreateAnd(acc, k, "MBAAnd");
            }
            return acc;
        }
        case AtomOp::Or:
        {
            assert(!t.Kids.empty() && "Or with no kids");
            Value* acc = EmitAtomTreeIR(builder, *t.Kids[0], inputs, resulttype);
            for (size_t i = 1; i < t.Kids.size(); ++i)
            {
                Value* k = EmitAtomTreeIR(builder, *t.Kids[i], inputs, resulttype);
                acc = builder.CreateOr(acc, k, "MBAOr");
            }
            return acc;
        }
        case AtomOp::Xor:
        {
            assert(!t.Kids.empty() && "Xor with no kids");
            Value* acc = EmitAtomTreeIR(builder, *t.Kids[0], inputs, resulttype);
            for (size_t i = 1; i < t.Kids.size(); ++i)
            {
                Value* k = EmitAtomTreeIR(builder, *t.Kids[i], inputs, resulttype);
                acc = builder.CreateXor(acc, k, "MBAXor");
            }
            return acc;
        }
    }
    return llvm::ConstantInt::get(resulttype, 0);
}

inline llvm::Value* EmitMBATerms(llvm::IRBuilder<>& builder, llvm::ArrayRef<ConstMBATerm> terms, llvm::ArrayRef<llvm::Value*> inputs, llvm::IntegerType* resulttype)
{
    using namespace llvm;

    if (terms.empty())
        return ConstantInt::get(resulttype, 0);

    const unsigned bw = resulttype->getBitWidth();
    const u64 widthmask = (bw >= 64) ? ~u64(0) : ((u64(1) << bw) - 1);

    Value* acc = nullptr;
    for (const ConstMBATerm& term : terms)
    {
        // defensive drop of zero coef terms; the mbagen builder already drops them in debug, but a release mode generator miss would otherwise emit coef=0 muls that LLVM may fail to fold at low O levels
        const u64 coefmasked = term.Coef & widthmask;
        if (coefmasked == 0) continue;

        Value* treeval = EmitAtomTreeIR(builder, *term.Tree, inputs, resulttype);
        Value* coefconst = ConstantInt::get(resulttype, coefmasked);

        Value* contrib = builder.CreateMul(coefconst, treeval, "MBATerm");

        acc = acc ? builder.CreateAdd(acc, contrib, "MBAAcc") : contrib;
    }

    return acc ? acc : ConstantInt::get(resulttype, 0);
}

inline llvm::Value* EmitConstMBA(llvm::IRBuilder<>& builder, const ConstMBAExpr& expr, llvm::ArrayRef<llvm::Value*> auxvalues)
{
    using namespace llvm;

    assert(auxvalues.size() == expr.V && "EmitConstMBA: aux_values count must equal expr.V");

    IntegerType* resulttype = IntegerType::get(builder.getContext(), expr.W);

    return EmitMBATerms(builder, expr.Terms, auxvalues, resulttype);
}

inline llvm::Value* EmitLinearMBA(llvm::IRBuilder<>& builder, const LinearMBAExpr& expr, llvm::ArrayRef<llvm::Value*> realvalues, llvm::ArrayRef<llvm::Value*> auxvalues)
{
    using namespace llvm;

    assert(realvalues.size() == expr.VReal && "EmitLinearMBA: real_values count must equal expr.V_real");
    assert(auxvalues.size() == expr.VAux && "EmitLinearMBA: aux_values count must equal expr.V_aux");

    IntegerType* resulttype = IntegerType::get(builder.getContext(), expr.W);

    SmallVector<Value*, 16> inputs;
    inputs.reserve(realvalues.size() + auxvalues.size());
    inputs.append(realvalues.begin(), realvalues.end());
    inputs.append(auxvalues.begin(), auxvalues.end());

    return EmitMBATerms(builder, expr.Terms, inputs, resulttype);
}

}