#pragma once

#include "PolyMBA.h"
#include "PolyMBACatalog.h"

#include <cassert>
#include <cstdint>
#include <random>
#include <vector>

namespace polymba
{

enum class RiggedKind : uint8_t
{
    ALowByte,
    BFInvComp,
    CCorrelatedPair,
    DZeroOnC,
    EBitPosition,
    FImageSet,
};

struct RiggedFamily
{
    Poly F;
    Poly FInv;
    Poly FSib;

    u64 A0 = 0;
    u64 A1 = 0;
    u64 A2 = 0;
    u64 A3 = 0;

    unsigned Width = 64;
    unsigned Degree = 2;
    RiggedKind Kind = RiggedKind::ALowByte;

    u64 InvariantMask = 0;
    u64 InvariantValue = 0;
    u64 ImpossibleValue = 0;
    u64 KnownC = 0;
    u64 PairK = 0;

    unsigned BitIdx = 0;
    unsigned BitExpected = 0;
};

struct RiggedBuildOptions
{
    uint64_t Seed = 0;
    unsigned Width = 64;
    size_t CountA = 8;
    size_t CountB = 4;
    size_t CountC = 4;
    size_t CountD = 8;
    size_t CountE = 4;
    size_t CountF = 4;
    size_t MaxTriesMult = 8;

    unsigned LowBitsK = 8;

    unsigned BitIdxMin = 8;
    unsigned BitIdxMax = 48;

    unsigned ImageEnumSamples = 8192;
    unsigned ImageMaskBits = 8;

    size_t InverseCapD = 80;
    unsigned VerifyBits = 18;
    size_t VerifyRandom = 50000;

    unsigned DegreeMixPct = 25;

    unsigned RiggedVerifySamples = 10000;
};

// subvariant A samplers produce a non-permutation on Z/2^w (a_1 is even once low k bits are cleared); A has no FInv path, so leaving a_i low bits zero for i>=1 is the whole invariant and must not be relaxed to "all coefficients random" or the low-byte identity breaks
inline Poly SampleRiggedLowBitsDegree2(std::mt19937_64& rng, unsigned k, unsigned w)
{
    const u64 lowmask = (k >= w) ? MaskW(~u64(0), w) : ((u64(1) << k) - 1u);
    const u64 clearlow = MaskW(~lowmask, w);
    Poly f(3);
    f.C[0] = MaskW(rng(), w);
    f.C[1] = MaskW(rng() & clearlow, w);
    f.C[2] = MaskW(rng() & clearlow, w);
    return f;
}

inline Poly SampleRiggedLowBitsDegree3(std::mt19937_64& rng, unsigned k, unsigned w)
{
    const u64 lowmask = (k >= w) ? MaskW(~u64(0), w) : ((u64(1) << k) - 1u);
    const u64 clearlow = MaskW(~lowmask, w);
    Poly f(4);
    f.C[0] = MaskW(rng(), w);
    f.C[1] = MaskW(rng() & clearlow, w);
    f.C[2] = MaskW(rng() & clearlow, w);
    f.C[3] = MaskW(rng() & clearlow, w);
    return f;
}

inline Poly SampleZeroOnCDegree2(std::mt19937_64& rng, u64 c, unsigned w)
{
    Poly f(3);
    f.C[1] = MaskW(rng(), w);
    f.C[2] = MaskW(rng(), w);
    const u64 cm = MaskW(c, w);
    const u64 c2 = MaskW(cm * cm, w);
    const u64 contrib = MaskW(MaskW(f.C[1] * cm, w) + MaskW(f.C[2] * c2, w), w);
    f.C[0] = MaskW(u64(0) - contrib, w);
    return f;
}

inline Poly SampleZeroOnCDegree3(std::mt19937_64& rng, u64 c, unsigned w)
{
    Poly f(4);
    f.C[1] = MaskW(rng(), w);
    f.C[2] = MaskW(rng(), w);
    f.C[3] = MaskW(rng(), w);
    const u64 cm = MaskW(c, w);
    const u64 c2 = MaskW(cm * cm, w);
    const u64 c3 = MaskW(c2 * cm, w);
    const u64 contrib = MaskW(MaskW(f.C[1] * cm, w) + MaskW(f.C[2] * c2, w) + MaskW(f.C[3] * c3, w), w);
    f.C[0] = MaskW(u64(0) - contrib, w);
    return f;
}

// subvariant F requires a_1 == 0 AND a_2 odd so a_2*x^2 is non-bijective on Z/2^w and the image is a proper subset; relaxing either coefficient constraint restores a bijection and the image set membership invariant no longer rules out any value
inline Poly SampleNonBijectiveDegree2(std::mt19937_64& rng, unsigned w)
{
    Poly f(3);
    f.C[0] = MaskW(rng(), w);
    f.C[1] = 0;
    f.C[2] = MaskW(rng() | 1ull, w);
    return f;
}

inline bool VerifyRiggedA(const RiggedFamily& fam, size_t samples, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    const unsigned w = fam.Width;
    for (size_t i = 0; i < samples; ++i)
    {
        const u64 x = MaskW(rng(), w);
        if ((fam.F.Eval(x, w) & fam.InvariantMask) != fam.InvariantValue) return false;
    }
    return true;
}

inline bool VerifyRiggedC(const RiggedFamily& fam, size_t samples, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    const unsigned w = fam.Width;
    for (size_t i = 0; i < samples; ++i)
    {
        const u64 x = MaskW(rng(), w);
        const u64 s = MaskW(fam.F.Eval(x, w) + fam.FSib.Eval(x, w), w);
        if (s != fam.PairK) return false;
    }
    return true;
}

inline bool VerifyRiggedD(const RiggedFamily& fam)
{
    return fam.F.Eval(fam.KnownC, fam.Width) == 0;
}

inline bool VerifyRiggedE(const RiggedFamily& fam, size_t samples, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    const unsigned w = fam.Width;
    for (size_t i = 0; i < samples; ++i)
    {
        const u64 x = MaskW(rng(), w);
        const unsigned bit = (unsigned)((fam.F.Eval(x, w) >> fam.BitIdx) & 1u);
        if (bit != fam.BitExpected) return false;
    }
    return true;
}

inline bool VerifyRiggedF(const RiggedFamily& fam, size_t samples, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    const unsigned w = fam.Width;
    for (size_t i = 0; i < samples; ++i)
    {
        const u64 x = MaskW(rng(), w);
        if ((fam.F.Eval(x, w) & fam.InvariantMask) == fam.ImpossibleValue) return false;
    }
    return true;
}

inline bool BuildOneA(RiggedFamily& out, const RiggedBuildOptions& opt, std::mt19937_64& rng)
{
    const unsigned w = opt.Width;
    const bool use3 = (opt.DegreeMixPct > 0) && ((unsigned)(rng() % 100u) < opt.DegreeMixPct);
    out.Width = w;
    out.Kind = RiggedKind::ALowByte;
    out.Degree = use3 ? 3u : 2u;
    out.F = use3 ? SampleRiggedLowBitsDegree3(rng, opt.LowBitsK, w) : SampleRiggedLowBitsDegree2(rng, opt.LowBitsK, w);
    out.A0 = out.F.C[0];
    out.A1 = out.F.C[1];
    out.A2 = (out.F.C.size() > 2) ? out.F.C[2] : 0;
    out.A3 = (out.F.C.size() > 3) ? out.F.C[3] : 0;

    const unsigned k = opt.LowBitsK;
    out.InvariantMask = (k >= w) ? MaskW(~u64(0), w) : ((u64(1) << k) - 1u);
    out.InvariantValue = out.F.C[0] & out.InvariantMask;
    return VerifyRiggedA(out, opt.RiggedVerifySamples, rng());
}

inline bool BuildOneB(RiggedFamily& out, const RiggedBuildOptions& opt, std::mt19937_64& rng)
{
    const unsigned w = opt.Width;
    const bool use3 = (opt.DegreeMixPct > 0) && ((unsigned)(rng() % 100u) < opt.DegreeMixPct);
    Poly f = use3 ? SamplePermutationDegree3(rng, 0u, w) : SamplePermutationDegree2(rng, 0u, w);
    if (use3 && f.C[3] == 0) return false;
    Poly finv = CompositionalInverse(f, opt.InverseCapD, w);
    if (!VerifyInverse(f, finv, opt.VerifyBits, opt.VerifyRandom, 0xC0FFEEu, w)) return false;

    out.Width = w;
    out.Kind = RiggedKind::BFInvComp;
    out.Degree = use3 ? 3u : 2u;
    out.F = std::move(f);
    out.FInv = std::move(finv);
    out.A0 = out.F.C[0];
    out.A1 = out.F.C[1];
    out.A2 = (out.F.C.size() > 2) ? out.F.C[2] : 0;
    out.A3 = (out.F.C.size() > 3) ? out.F.C[3] : 0;
    return true;
}

inline bool BuildOneC(RiggedFamily& out, const RiggedBuildOptions& opt, std::mt19937_64& rng)
{
    const unsigned w = opt.Width;
    const bool use3 = (opt.DegreeMixPct > 0) && ((unsigned)(rng() % 100u) < opt.DegreeMixPct);
    const unsigned deg = use3 ? 3u : 2u;

    Poly f1(deg + 1);
    for (unsigned i = 0; i <= deg; ++i) f1.C[i] = MaskW(rng(), w);
    if (f1.C[deg] == 0) f1.C[deg] = MaskW(rng() | 1ull, w);

    const u64 k = MaskW(rng(), w);

    // F2 MUST be built by coefficient wise construction (F2.C[0] = K - F1.C[0]; F2.C[i] = -F1.C[i] for i>=1) so F1(x)+F2(x)=K holds identically; computing F2 by evaluation at sample points would only hold at those points and the invariant would silently break on any other x
    Poly f2(deg + 1);
    f2.C[0] = MaskW(k - f1.C[0], w);
    for (unsigned i = 1; i <= deg; ++i) f2.C[i] = MaskW(u64(0) - f1.C[i], w);

    out.Width = w;
    out.Kind = RiggedKind::CCorrelatedPair;
    out.Degree = deg;
    out.F = std::move(f1);
    out.FSib = std::move(f2);
    out.A0 = out.F.C[0];
    out.A1 = out.F.C[1];
    out.A2 = (out.F.C.size() > 2) ? out.F.C[2] : 0;
    out.A3 = (out.F.C.size() > 3) ? out.F.C[3] : 0;
    out.PairK = k;
    return VerifyRiggedC(out, opt.RiggedVerifySamples, rng());
}

inline bool BuildOneD(RiggedFamily& out, const RiggedBuildOptions& opt, std::mt19937_64& rng)
{
    const unsigned w = opt.Width;
    const bool use3 = (opt.DegreeMixPct > 0) && ((unsigned)(rng() % 100u) < opt.DegreeMixPct);
    // c=0 makes F(0)=a_0 so the baked Horner collapses to a trivial compare against 0, and c=1 makes F(1)=sum(a_i) which is pattern matchable; both must be rejected to keep the Horner chain none degenerate
    u64 c = 0;
    for (int tries = 0; tries < 8; ++tries)
    {
        c = MaskW(rng(), w);
        if (c != 0 && c != 1) break;
    }
    out.Width = w;
    out.Kind = RiggedKind::DZeroOnC;
    out.Degree = use3 ? 3u : 2u;
    out.KnownC = c;
    out.F = use3 ? SampleZeroOnCDegree3(rng, c, w) : SampleZeroOnCDegree2(rng, c, w);
    out.A0 = out.F.C[0];
    out.A1 = out.F.C[1];
    out.A2 = (out.F.C.size() > 2) ? out.F.C[2] : 0;
    out.A3 = (out.F.C.size() > 3) ? out.F.C[3] : 0;
    return VerifyRiggedD(out);
}

inline bool BuildOneE(RiggedFamily& out, const RiggedBuildOptions& opt, std::mt19937_64& rng)
{
    const unsigned w = opt.Width;
    unsigned bmin = opt.BitIdxMin;
    unsigned bmax = opt.BitIdxMax;
    if (bmax >= w) bmax = w - 1u;
    if (bmin > bmax) bmin = bmax;
    const unsigned bitidx = bmin + (unsigned)(rng() % (bmax - bmin + 1u));

    // for bit `bitidx` to be invariant across x, every a_i with i>=1 must have bits 0..bitidx zero so no carry from a_i*x^i can propagate into bit `bitidx`; passing k = bitidx (not bitidx+1) leaves bit `bitidx` of a_1 free and the invariant breaks on roughly half of x
    const unsigned k = bitidx + 1u;
    const bool use3 = (opt.DegreeMixPct > 0) && ((unsigned)(rng() % 100u) < opt.DegreeMixPct);

    out.Width = w;
    out.Kind = RiggedKind::EBitPosition;
    out.Degree = use3 ? 3u : 2u;
    out.F = use3 ? SampleRiggedLowBitsDegree3(rng, k, w) : SampleRiggedLowBitsDegree2(rng, k, w);
    out.A0 = out.F.C[0];
    out.A1 = out.F.C[1];
    out.A2 = (out.F.C.size() > 2) ? out.F.C[2] : 0;
    out.A3 = (out.F.C.size() > 3) ? out.F.C[3] : 0;
    out.BitIdx = bitidx;
    out.BitExpected = (unsigned)((out.F.C[0] >> bitidx) & 1u);
    return VerifyRiggedE(out, opt.RiggedVerifySamples, rng());
}

inline bool BuildOneF(RiggedFamily& out, const RiggedBuildOptions& opt, std::mt19937_64& rng)
{
    const unsigned w = opt.Width;
    Poly f = SampleNonBijectiveDegree2(rng, w);

    unsigned maskbits = opt.ImageMaskBits;
    if (maskbits == 0 || maskbits >= w) maskbits = 8u;
    const u64 mask = (u64(1) << maskbits) - 1u;

    std::vector<uint8_t> seen(size_t(1) << maskbits, 0);
    unsigned enumn = opt.ImageEnumSamples ? opt.ImageEnumSamples : 8192u;
    std::mt19937_64 erng(rng());
    for (unsigned i = 0; i < enumn; ++i)
    {
        const u64 x = MaskW(erng(), w);
        const u64 v = f.Eval(x, w) & mask;
        seen[(size_t)v] = 1;
    }

    std::vector<u64> holes;
    holes.reserve(seen.size());
    for (size_t i = 0; i < seen.size(); ++i)
    {
        if (!seen[i]) holes.push_back((u64)i);
    }
    if (holes.empty()) return false;
    const u64 impossible = holes[rng() % holes.size()];

    out.Width = w;
    out.Kind = RiggedKind::FImageSet;
    out.Degree = 2;
    out.F = std::move(f);
    out.A0 = out.F.C[0];
    out.A1 = out.F.C[1];
    out.A2 = out.F.C[2];
    out.A3 = 0;
    out.InvariantMask = mask;
    out.ImpossibleValue = impossible;
    return VerifyRiggedF(out, opt.RiggedVerifySamples, rng());
}

inline std::vector<RiggedFamily> BuildRiggedFamilies(const RiggedBuildOptions& opt)
{
    std::mt19937_64 rng;
    if (opt.Seed) rng.seed(opt.Seed);
    else { std::random_device rd; rng.seed(((u64)rd() << 32) | rd()); }

    std::vector<RiggedFamily> out;
    out.reserve(opt.CountA + opt.CountB + opt.CountC + opt.CountD + opt.CountE + opt.CountF);

    auto runkind = [&](size_t count, bool (*fn)(RiggedFamily&, const RiggedBuildOptions&, std::mt19937_64&))
    {
        const size_t maxtries = count * opt.MaxTriesMult + 16u;
        size_t tries = 0;
        size_t got = 0;
        while (got < count && tries < maxtries)
        {
            ++tries;
            RiggedFamily fam;
            if (fn(fam, opt, rng))
            {
                out.push_back(std::move(fam));
                ++got;
            }
        }
    };

    if (opt.CountA) runkind(opt.CountA, &BuildOneA);
    if (opt.CountB) runkind(opt.CountB, &BuildOneB);
    if (opt.CountC) runkind(opt.CountC, &BuildOneC);
    if (opt.CountD) runkind(opt.CountD, &BuildOneD);
    if (opt.CountE) runkind(opt.CountE, &BuildOneE);
    if (opt.CountF) runkind(opt.CountF, &BuildOneF);

    return out;
}

}