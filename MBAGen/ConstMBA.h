#pragma once

#include "AtomTree.h"
#include "IdentityGenerator.h"
#include "MBABasis.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

namespace mbagen
{

struct ConstMBAParams
{
    unsigned AuxVarCount = 4;
    unsigned GroupCount = 3;
    unsigned Width = 64;
    GeneratorPolicy GeneratorPolicyV{};
    GeneratorState* GenState = nullptr;
};

struct ConstMBATerm
{
    u64 Coef;
    AtomTreePtr Tree;
};

struct ConstMBAExpr
{
    std::vector<ConstMBATerm> Terms;
    unsigned V = 0;
    unsigned W = 64;
    u64 Target = 0;
};

inline u64 ModInvW(u64 a, unsigned w)
{
    a = MaskW(a, w);
    assert((a & 1) && "ModInvW requires odd input");
    u64 x = a;
    // 6 Newton doublings give at least 192 bits of precision; the same algorithm as IdentityGenerator's ModInvPow2 is kept locally so callers that already include this header don't pull another dependency
    for (int i = 0; i < 6; ++i)
        x = MaskW(x * MaskW(2u - MaskW(a * x, w), w), w);
    return x;
}

inline void CanonicalizeTerms(std::vector<ConstMBATerm>& terms, unsigned w)
{
    for (auto& t : terms)
        t.Tree = RewriteIdentities(t.Tree, w);

    terms.erase(
        std::remove_if(terms.begin(), terms.end(),
            [w](const ConstMBATerm& t)
            {
                return t.Tree->Op == AtomOp::Const && MaskW(t.Tree->ConstVal, w) == 0;
            }),
        terms.end());

    struct GroupSlot
    {
        AtomTreePtr Tree;
        u64 CoefSum = 0;
    };
    std::vector<GroupSlot> groups;
    std::unordered_map<u64, size_t> hashtoidx;

    for (auto& t : terms)
    {
        const u64 h = CanonicalHash(t.Tree);
        auto it = hashtoidx.find(h);
        if (it == hashtoidx.end())
        {
            hashtoidx.emplace(h, groups.size());
            groups.push_back({ t.Tree, MaskW(t.Coef, w) });
        }
        else
        {
            auto& g = groups[it->second];
            g.CoefSum = MaskW(g.CoefSum + MaskW(t.Coef, w), w);
        }
    }

    terms.clear();
    for (auto& g : groups)
        if (g.CoefSum != 0)
            terms.push_back({ g.CoefSum, std::move(g.Tree) });
}

inline ConstMBAExpr BuildConstMBA(u64 target, ConstMBAParams params, std::mt19937_64& rng)
{
    assert(params.AuxVarCount >= 1 && "ConstMBA requires at least one aux variable");
    assert(params.GroupCount >= 1 && "ConstMBA requires at least one group");

    ConstMBAExpr expr;
    expr.V = params.AuxVarCount;
    expr.W = params.Width;
    expr.Target = MaskW(target, params.Width);

    // K must be odd so modinv(K) exists on Z/2^w; forcing the low bit via; rng() | 1, keeps the entropy of the remaining 63 bits untouched
    const u64 K = MaskW(rng() | 1ull, params.Width);

    std::vector<u64> ksplit(params.GroupCount);
    u64 kacc = 0;
    for (unsigned i = 0; i + 1 < params.GroupCount; ++i)
    {
        ksplit[i] = MaskW(rng(), params.Width);
        kacc = MaskW(kacc + ksplit[i], params.Width);
    }
    ksplit[params.GroupCount - 1] = MaskW(K - kacc, params.Width);

    // caller owned GeneratorState gives cross call diversity; a local fallback gives only within call diversity across group_count groups, so threading one through from the pass driver is strongly preferred for binary wide uniqueness
    GeneratorState localstate;
    GeneratorState* state = params.GenState ? params.GenState : &localstate;

    auto emit = [&](u64 c, AtomTreePtr t)
    {
        expr.Terms.push_back({ MaskW(c, params.Width), std::move(t) });
    };

    for (unsigned i = 0; i < params.GroupCount; ++i)
        BuildGroupFromGenerator(ksplit[i], params.AuxVarCount, params.GeneratorPolicyV, rng, *state, params.Width, emit);

    const u64 M = MaskW(ModInvW(K, params.Width) * expr.Target, params.Width);
    for (auto& t : expr.Terms)
        t.Coef = MaskW(t.Coef * M, params.Width);

    // scaling by M commutes with hash-grouping — (c1 + c2) * M == c1*M + c2*M mod 2^w; so running canonicalisation after the multiplier is algebraically equivalent to running it before and keeps the pipeline linear
    CanonicalizeTerms(expr.Terms, params.Width);

    std::shuffle(expr.Terms.begin(), expr.Terms.end(), rng);

    return expr;
}

inline u64 EvalConstMBA(const ConstMBAExpr& expr, const u64* auxvalues)
{
    u64 acc = 0;
    for (const auto& t : expr.Terms)
    {
        const u64 v = EvalAtomTree(*t.Tree, auxvalues, expr.W);
        acc = MaskW(acc + MaskW(t.Coef * v, expr.W), expr.W);
    }
    return acc;
}

inline bool VerifyConst(const ConstMBAExpr& expr, u64 target, std::mt19937_64& rng, int samples = 10000)
{
    target = MaskW(target, expr.W);
    std::vector<u64> aux(expr.V);
    for (int s = 0; s < samples; ++s)
    {
        for (auto& v : aux) v = MaskW(rng(), expr.W);
        if (EvalConstMBA(expr, aux.data()) != target) return false;
    }
    return true;
}

}