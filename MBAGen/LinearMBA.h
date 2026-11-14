#pragma once

#include "ConstMBA.h"
#include "IdentityGenerator.h"
#include "MBABasis.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <random>
#include <vector>

namespace mbagen
{

enum class LinearMBATarget
{
    Zero,
    Identity
};

struct LinearMBAParams
{
    unsigned AuxVarCount = 4;
    unsigned RealVarCount = 0;
    unsigned GroupCount = 3;
    unsigned Width = 64;
    GeneratorPolicy GeneratorPolicyV{};
    GeneratorState* GenState = nullptr;
    unsigned XYPct = 0;
};

struct LinearMBAExpr
{
    std::vector<ConstMBATerm> Terms;
    unsigned VReal = 0;
    unsigned VAux = 0;
    unsigned W = 64;
    LinearMBATarget TargetKind = LinearMBATarget::Zero;
    unsigned TargetRealVarIdx = 0;
};

inline LinearMBAExpr BuildLinearMBA(LinearMBATarget target, unsigned targetrealvaridx, LinearMBAParams params, std::mt19937_64& rng)
{
    assert(params.AuxVarCount >= 1 && "LinearMBA requires at least one aux variable");
    assert(params.GroupCount >= 1 && "LinearMBA requires at least one group");
    if (target == LinearMBATarget::Identity)
    {
        assert(params.RealVarCount >= 1 && "LinearMBA Identity target requires real_var_count >= 1");
        assert(targetrealvaridx < params.RealVarCount && "LinearMBA target_real_var_idx out of range");
    }

    LinearMBAExpr expr;
    expr.VReal = params.RealVarCount;
    expr.VAux = params.AuxVarCount;
    expr.W = params.Width;
    expr.TargetKind = target;
    expr.TargetRealVarIdx = targetrealvaridx;

    const unsigned vtotal = params.RealVarCount + params.AuxVarCount;
    assert(vtotal >= 1 && "LinearMBA requires V_real + V_aux >= 1");

    // K must be odd so the ClassA(-K) corrective below hits the valid envelope-subtraction path; forcing low bit on preserves the entropy of the remaining 63 bits
    const u64 K = MaskW(rng() | 1ull, params.Width);

    std::vector<u64> ksplit(params.GroupCount);
    u64 kacc = 0;
    for (unsigned i = 0; i + 1 < params.GroupCount; ++i)
    {
        ksplit[i] = MaskW(rng(), params.Width);
        kacc = MaskW(kacc + ksplit[i], params.Width);
    }
    ksplit[params.GroupCount - 1] = MaskW(K - kacc, params.Width);

    GeneratorState localstate;
    GeneratorState* state = params.GenState ? params.GenState : &localstate;

    auto emit = [&](u64 c, AtomTreePtr t)
    {
        expr.Terms.push_back({ MaskW(c, params.Width), std::move(t) });
    };

    for (unsigned i = 0; i < params.GroupCount; ++i)
        BuildGroupFromGenerator(ksplit[i], vtotal, params.GeneratorPolicyV, rng, *state, params.Width, emit);

    const u64 minusk = MaskW(u64(0) - K, params.Width);

    GenerateIdentityRequest rconst;
    rconst.TargetV.Kind = TargetKind::ClassA;
    rconst.TargetV.Constant = minusk;
    rconst.TierV = params.GeneratorPolicyV.TierV;
    IdentityResult corrconst = GenerateIdentityWithRetry(rconst, rng, *state, params.Width);
    for (const auto& t : corrconst.Terms)
        emit(t.first, t.second);

    if (target == LinearMBATarget::Identity)
    {
        // ClassXY decomposition x_i = (x_i | x_j) + (x_i & x_j) - x_j is correct over Z/2^w via a+b = (a|b)+(a&b); it introduces non affine real var coupling that SiMBA descended simplifiers can't peel linearly, so it's a stronger emit when available but falls back cleanly when real_var_count < 2 can't supply a distinct j
        bool usexy = (params.XYPct > 0) && (params.RealVarCount >= 2) && ((unsigned)(rng() % 100u) < params.XYPct);

        if (usexy)
        {
            const unsigned iidx = targetrealvaridx;
            unsigned jidx;
            if (params.RealVarCount == 2)
            {
                jidx = (iidx == 0u) ? 1u : 0u;
            }
            else
            {
                jidx = (unsigned)(rng() % params.RealVarCount);
                while (jidx == iidx)
                    jidx = (unsigned)(rng() % params.RealVarCount);
            }

            const u64 one = MaskW(1ull, params.Width);
            const u64 minusone = MaskW(u64(0) - 1ull, params.Width);

            {
                GenerateIdentityRequest ror;
                ror.TargetV.Kind = TargetKind::ClassXY;
                ror.TargetV.XCoef = one;
                ror.TargetV.XVarIdx = iidx;
                ror.TargetV.YVarIdx = jidx;
                ror.TargetV.XYOpV = XYOp::Or;
                ror.TierV = params.GeneratorPolicyV.TierV;
                ror.VOverride = vtotal;
                IdentityResult corr = GenerateIdentityWithRetry(ror, rng, *state, params.Width);
                for (const auto& t : corr.Terms) emit(t.first, t.second);
            }
            {
                GenerateIdentityRequest rand;
                rand.TargetV.Kind = TargetKind::ClassXY;
                rand.TargetV.XCoef = one;
                rand.TargetV.XVarIdx = iidx;
                rand.TargetV.YVarIdx = jidx;
                rand.TargetV.XYOpV = XYOp::And;
                rand.TierV = params.GeneratorPolicyV.TierV;
                rand.VOverride = vtotal;
                IdentityResult corr = GenerateIdentityWithRetry(rand, rng, *state, params.Width);
                for (const auto& t : corr.Terms) emit(t.first, t.second);
            }
            {
                GenerateIdentityRequest rx;
                rx.TargetV.Kind = TargetKind::ClassX;
                rx.TargetV.XCoef = minusone;
                rx.TargetV.XVarIdx = jidx;
                rx.TierV = params.GeneratorPolicyV.TierV;
                rx.VOverride = vtotal;
                IdentityResult corr = GenerateIdentityWithRetry(rx, rng, *state, params.Width);
                for (const auto& t : corr.Terms) emit(t.first, t.second);
            }
        }
        else
        {
            GenerateIdentityRequest rvar;
            rvar.TargetV.Kind = TargetKind::ClassX;
            rvar.TargetV.XCoef = 1u;
            rvar.TargetV.XVarIdx = targetrealvaridx;
            rvar.TierV = params.GeneratorPolicyV.TierV;
            // V_override pins the generator's V to vtotal so atom trees can reference any variable in the surrounding context; without this, the generator may pick V smaller than target_real_var_idx+1 from the tier range and silently emit terms that reference out-of-bounds variables
            rvar.VOverride = vtotal;
            IdentityResult corrvar = GenerateIdentityWithRetry(rvar, rng, *state, params.Width);
            for (const auto& t : corrvar.Terms)
                emit(t.first, t.second);
        }
    }

    CanonicalizeTerms(expr.Terms, params.Width);

    std::shuffle(expr.Terms.begin(), expr.Terms.end(), rng);

    return expr;
}

inline u64 EvalLinearMBA(const LinearMBAExpr& expr, const u64* realvalues, const u64* auxvalues)
{
    const unsigned vtotal = expr.VReal + expr.VAux;
    std::vector<u64> inputs(vtotal);
    for (unsigned i = 0; i < expr.VReal; ++i) inputs[i] = realvalues[i];
    for (unsigned i = 0; i < expr.VAux; ++i) inputs[expr.VReal + i] = auxvalues[i];

    u64 acc = 0;
    for (const auto& t : expr.Terms)
    {
        const u64 v = EvalAtomTree(*t.Tree, inputs.data(), expr.W);
        acc = MaskW(acc + MaskW(t.Coef * v, expr.W), expr.W);
    }
    return acc;
}

inline u64 ExpectedLinearMBAValue(const LinearMBAExpr& expr, const u64* realvalues)
{
    switch (expr.TargetKind)
    {
        case LinearMBATarget::Zero: return 0;
        case LinearMBATarget::Identity: return MaskW(realvalues[expr.TargetRealVarIdx], expr.W);
    }
    return 0;
}

inline bool VerifyLinear(const LinearMBAExpr& expr, std::mt19937_64& rng, int samples = 1000)
{
    std::vector<u64> reals(expr.VReal);
    std::vector<u64> auxs(expr.VAux);
    for (int s = 0; s < samples; ++s)
    {
        for (auto& v : reals) v = MaskW(rng(), expr.W);
        for (auto& v : auxs) v = MaskW(rng(), expr.W);
        const u64 got = EvalLinearMBA(expr, reals.empty() ? nullptr : reals.data(), auxs.empty() ? nullptr : auxs.data());
        const u64 want = ExpectedLinearMBAValue(expr, reals.empty() ? nullptr : reals.data());
        if (got != want) return false;
    }
    return true;
}

}