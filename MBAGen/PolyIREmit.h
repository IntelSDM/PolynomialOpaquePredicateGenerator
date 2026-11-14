#pragma once

#include "FoldBarrier.h"
#include "PolyMBA.h"
#include "PolyMBACatalog.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include <cassert>
#include <cstdint>

namespace mbagen
{

static void TagEscalatePoly(llvm::IRBuilder<>& builder, llvm::Value* v)
{
    if (auto* i = llvm::dyn_cast<llvm::Instruction>(v))
    {
        llvm::LLVMContext& ctx = builder.getContext();
        llvm::Metadata* ops[1] = { llvm::MDString::get(ctx, "poly_const") };
        // "mbagen.escalate" with "poly_const" is the downstream contract ConstMBAPass reads to route the parent Mul/Add's ConstantInt operands into its T6 (richer V, deeper trees) tier, bypassing the module's base-tier density gate so a2/C1/C0 Horner constants are hidden by the richer encoding
        i->setMetadata("mbagen.escalate", llvm::MDNode::get(ctx, ops));
    }
}

enum class PolySiteKind : uint8_t
{
    AddAK,
    SubAK,
    SubKA,
    MulAK,
};

inline llvm::Value* EmitPolyEncode(llvm::IRBuilder<>& builder, llvm::Value* A, const polymba::PolyMBAFamily& family, uint64_t K, PolySiteKind kind, ModuleBarrierState& barrierstate)
{
    auto* ity = llvm::dyn_cast<llvm::IntegerType>(A->getType());
    assert(ity && "EmitPolyEncode requires integer-typed A");
    const unsigned w = ity->getBitWidth();
    assert(w == family.Width && "family width must match the operand bitwidth");

    const uint64_t MASK = (w >= 64) ? ~uint64_t(0) : ((uint64_t(1) << w) - 1);
    auto mask = [MASK](uint64_t v) { return v & MASK; };

    const uint64_t kw = mask(K);
    const uint64_t a0 = family.A0;
    const uint64_t a1 = family.A1;
    const uint64_t a2 = family.A2;
    const uint64_t a3 = family.A3;
    const uint64_t cxy = family.CXY;

    uint64_t lead = 0;
    uint64_t c0 = 0;
    uint64_t c1 = 0;
    uint64_t c2 = 0;
    if (family.Degree == 2)
    {
        lead = a2;
        switch (kind)
        {
            case PolySiteKind::AddAK:
            {
                const uint64_t ak = mask(a2 * kw);
                const uint64_t ak2 = mask(ak * kw);
                c0 = mask(a0 + mask(a1 * kw) + ak2);
                c1 = mask(a1 + mask(cxy * kw));
                break;
            }
            case PolySiteKind::SubAK:
            {
                // Horner K terms are folded through mask() at every step to keep the arithmetic in Z/2^w; without this intermediate wrap, 64-bit multiplications of unmasked operands silently bleed into the i32 path and produce wrong C0 constants
                const uint64_t negk = mask(uint64_t(0) - kw);
                const uint64_t ak = mask(a2 * negk);
                const uint64_t ak2 = mask(ak * negk);
                c0 = mask(a0 + mask(a1 * negk) + ak2);
                c1 = mask(a1 - mask(cxy * kw));
                break;
            }
            case PolySiteKind::SubKA:
            {
                const uint64_t ak = mask(a2 * kw);
                const uint64_t ak2 = mask(ak * kw);
                c0 = mask(a0 + mask(a1 * kw) + ak2);
                c1 = mask(uint64_t(0) - mask(a1 + mask(cxy * kw)));
                break;
            }
            case PolySiteKind::MulAK:
            {
                // f(A*K) stays degree matched with the family since each coefficient picks up a K^i factor; permutation-ness is preserved as long as K is odd (a2*K^2 shares a2's valuation), and the classifier filters K==0 and K==1 upstream so this never emits a trivial encoding
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
            case PolySiteKind::AddAK:
            {
                lead = a3;
                c2 = mask(a2 + mask(threea3 * kw));
                c1 = mask(a1 + mask(twoa2 * kw) + mask(threea3 * kw2));
                c0 = mask(a0 + mask(a1 * kw) + mask(a2 * kw2) + mask(a3 * kw3));
                break;
            }
            case PolySiteKind::SubAK:
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
            case PolySiteKind::SubKA:
            {
                // f(K-A) has (K-A)^3 expanded = -A^3 + 3*A^2*K - 3*A*K^2 + K^3, so the leading A^3 coefficient flips sign to -a3; omitting that sign flip emits a decode that returns K+A rather than K-A
                lead = mask(uint64_t(0) - a3);
                c2 = mask(a2 + mask(threea3 * kw));
                c1 = mask(uint64_t(0) - mask(a1 + mask(twoa2 * kw) + mask(threea3 * kw2)));
                c0 = mask(a0 + mask(a1 * kw) + mask(a2 * kw2) + mask(a3 * kw3));
                break;
            }
            case PolySiteKind::MulAK:
            {
                lead = mask(a3 * kw3);
                c2 = mask(a2 * kw2);
                c1 = mask(a1 * kw);
                c0 = a0;
                break;
            }
        }
    }

    llvm::Value* vlead = llvm::ConstantInt::get(ity, lead);
    llvm::Value* vc0 = llvm::ConstantInt::get(ity, c0);
    llvm::Value* vc1 = llvm::ConstantInt::get(ity, c1);

    llvm::Value* t1 = builder.CreateMul(vlead, A, "PolyMBAEncT1");
    TagEscalatePoly(builder, t1);

    llvm::Value* tail;
    if (family.Degree == 2)
    {
        llvm::Value* t2 = builder.CreateAdd(t1, vc1, "PolyMBAEncT2");
        TagEscalatePoly(builder, t2);
        llvm::Value* t3 = builder.CreateMul(t2, A, "PolyMBAEncT3");
        tail = builder.CreateAdd(t3, vc0, "PolyMBAEnc");
        TagEscalatePoly(builder, tail);
    }
    else
    {
        llvm::Value* vc2 = llvm::ConstantInt::get(ity, c2);
        llvm::Value* t2 = builder.CreateAdd(t1, vc2, "PolyMBAEncT2");
        TagEscalatePoly(builder, t2);
        llvm::Value* t3 = builder.CreateMul(t2, A, "PolyMBAEncT3");
        llvm::Value* t4 = builder.CreateAdd(t3, vc1, "PolyMBAEncT4");
        TagEscalatePoly(builder, t4);
        llvm::Value* t5 = builder.CreateMul(t4, A, "PolyMBAEncT5");
        tail = builder.CreateAdd(t5, vc0, "PolyMBAEnc");
        TagEscalatePoly(builder, tail);
    }

    return InsertFoldBarrier(builder, tail, barrierstate);
}

inline llvm::Value* EmitPolyDecode(llvm::IRBuilder<>& builder, llvm::Value* enc, const polymba::PolyMBAFamily& family, ModuleBarrierState& barrierstate)
{
    auto* ity = llvm::dyn_cast<llvm::IntegerType>(enc->getType());
    assert(ity && "EmitPolyDecode requires integer-typed enc");
    const unsigned w = ity->getBitWidth();
    assert(w == family.Width && "family width must match the operand bitwidth");

    const auto& coefs = family.FInv.C;
    assert(!coefs.empty() && "finv must have at least one coefficient — call CompositionalInverse first");

    // trim trailing zero coefficients so the Horner emit matches the actual degree, not the truncation cap D from inverse construction; a 196 coef finv with 130 trailing zeros would otherwise emit 260 pointless (acc*enc+0) instructions at a2_valuation=0 on w=64
    size_t deg = coefs.size();
    while (deg > 1 && coefs[deg - 1] == 0) --deg;

    llvm::Value* acc = llvm::ConstantInt::get(ity, coefs[deg - 1]);

    bool accisconst = true;
    for (size_t k = deg - 1; k-- > 0; )
    {
        llvm::Value* mul = builder.CreateMul(acc, enc, "PolyMBADecM");
        // the FIRST iteration's Mul takes a bare ConstantInt as `acc`, so it has a ConstantInt operand to tag; subsequent iterations' Mul takes an Add result and has no Const operand, so we only tag on the first pass; the Add in every iteration always has `ck` as its Const operand and gets tagged unconditionally
        if (accisconst) TagEscalatePoly(builder, mul);
        llvm::Value* ck = llvm::ConstantInt::get(ity, coefs[k]);
        acc = builder.CreateAdd(mul, ck, "PolyMBADecA");
        TagEscalatePoly(builder, acc);
        accisconst = false;
    }

    return InsertFoldBarrier(builder, acc, barrierstate);
}

}