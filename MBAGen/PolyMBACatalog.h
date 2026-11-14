#pragma once

#include "PolyMBA.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace polymba
{

struct PolyMBAFamily
{
    Poly F;
    Poly FInv;
    u64 A0 = 0;
    u64 A1 = 0;
    u64 A2 = 0;
    u64 A3 = 0;
    u64 CXY = 0;
    unsigned Width = 64;
    unsigned Degree = 2;
};

inline bool VerifyCorrections(const PolyMBAFamily& fam, size_t samples = 200000, uint64_t seed = 0xA17B)
{
    // the f(x+y) / f(x-y) encoded domain identities only hold at degree 2; the degree 3 extension has cubic cross-terms (3*a3*x^2*y + 3*a3*x*y^2) that break them, so VerifyInverse is the sole correctness gate at degree 3 and this returns true immediately there
    if (fam.Degree != 2) return true;
    std::mt19937_64 rng(seed);
    const Poly& f = fam.F;
    const unsigned w = fam.Width;
    for (size_t i = 0; i < samples; ++i)
    {
        const u64 x = MaskW(rng(), w);
        const u64 y = MaskW(rng(), w);
        const u64 fx = f.Eval(x, w);
        const u64 fy = f.Eval(y, w);
        const u64 encadd = MaskW(fx + fy - fam.A0 + MaskW(fam.CXY * MaskW(x * y, w), w), w);
        if (encadd != f.Eval(MaskW(x + y, w), w)) return false;
        const u64 encsub = MaskW(fx - fy + fam.A0 + MaskW(fam.CXY * MaskW(y * y, w), w) - MaskW(fam.CXY * MaskW(x * y, w), w), w);
        if (encsub != f.Eval(MaskW(x - y, w), w)) return false;
    }
    return true;
}

struct BuildOptions
{
    uint64_t Seed = 0;
    size_t Count = 64;
    size_t InverseCapD = 80;
    unsigned VerifyBits = 18;
    size_t VerifyRandom = 100000;
    size_t MaxTriesMult = 8;
    unsigned Width = 64;
    unsigned A2Valuation = 0;
    unsigned DegreeMixPct = 0;
};

inline Poly SamplePermutationDegree2(std::mt19937_64& rng, unsigned v, unsigned w = 64)
{
    Poly f(3);
    f.C[0] = MaskW(rng(), w);
    f.C[1] = MaskW(rng() | 1ull, w);
    if (v == 0) f.C[2] = MaskW(rng() & ~1ull, w);
    else if (v >= w) f.C[2] = 0;
    else f.C[2] = MaskW((rng() | 1ull) << v, w);
    return f;
}

inline Poly SamplePermutationDegree3(std::mt19937_64& rng, unsigned v, unsigned w = 64)
{
    Poly f(4);
    f.C[0] = MaskW(rng(), w);
    f.C[1] = MaskW(rng() | 1ull, w);
    // Rivest's permutation condition on Z/2^w at degree 3 reduces to a1 odd AND a2 even AND a3 even; a2 and a3 share the valuation knob here because the compositional inverse's degree is bounded by min(valuation(a2), valuation(a3))
    if (v == 0)
    {
        f.C[2] = MaskW(rng() & ~1ull, w);
        f.C[3] = MaskW(rng() & ~1ull, w);
    }
    else if (v >= w)
    {
        f.C[2] = 0;
        f.C[3] = 0;
    }
    else
    {
        f.C[2] = MaskW((rng() | 1ull) << v, w);
        f.C[3] = MaskW((rng() | 1ull) << v, w);
    }
    return f;
}

inline std::vector<PolyMBAFamily> BuildFamilies(const BuildOptions& opt)
{
    std::mt19937_64 rng;
    if (opt.Seed) rng.seed(opt.Seed);
    else { std::random_device rd; rng.seed((u64(rd()) << 32) | rd()); }

    std::vector<PolyMBAFamily> out;
    out.reserve(opt.Count);

    const size_t maxtries = opt.Count * opt.MaxTriesMult + 16;
    size_t tries = 0;
    const unsigned mixpct = opt.DegreeMixPct > 100 ? 100 : opt.DegreeMixPct;

    while (out.size() < opt.Count && tries < maxtries)
    {
        ++tries;
        PolyMBAFamily fam;
        fam.Width = opt.Width;

        const bool wantd3 = (mixpct > 0) && ((unsigned)(rng() % 100u) < mixpct);

        if (!wantd3)
        {
            fam.F = SamplePermutationDegree2(rng, opt.A2Valuation, opt.Width);
            fam.A0 = fam.F.C[0];
            fam.A1 = fam.F.C[1];
            fam.A2 = fam.F.C[2];
            fam.A3 = 0;
            fam.CXY = MaskW(fam.A2 << 1, opt.Width);
            fam.Degree = 2;
        }
        else
        {
            fam.F = SamplePermutationDegree3(rng, opt.A2Valuation, opt.Width);
            // reject a3==0 so the polynomial is genuinely degree 3; SamplePermutationDegree2 uses c[2] presence to differentiate degree 2 vs affine, but at degree 3 we insist on cubic-ness or we're not giving the attacker anything new to simplify through
            if (fam.F.C[3] == 0) continue;
            fam.A0 = fam.F.C[0];
            fam.A1 = fam.F.C[1];
            fam.A2 = fam.F.C[2];
            fam.A3 = fam.F.C[3];
            fam.CXY = MaskW(fam.A2 << 1, opt.Width);
            fam.Degree = 3;
        }

        fam.FInv = CompositionalInverse(fam.F, opt.InverseCapD, opt.Width);
        if (!VerifyInverse(fam.F, fam.FInv, opt.VerifyBits, opt.VerifyRandom, 0xC0FFEEu, opt.Width)) continue;
        // VerifyCorrections is a no op for degree 3 (identity doesn't hold there), returns true unconditionally, so this gate only trips on degree 2 anomalies and never fails a degree 3 family that already passed VerifyInverse
        if (!VerifyCorrections(fam)) continue;

        out.push_back(std::move(fam));
    }

    return out;
}

}