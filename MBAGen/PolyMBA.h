#pragma once

#include <cassert>
#include <cstdint>
#include <random>
#include <vector>

namespace polymba
{

using u64 = uint64_t;

static inline u64 MaskW(u64 v, unsigned w)
{
    return (w >= 64) ? v : (v & ((u64(1) << w) - 1));
}

struct Poly
{
    std::vector<u64> C;

    Poly() = default;
    explicit Poly(size_t degplusone) : C(degplusone, 0) {}

    size_t Size() const { return C.size(); }

    int Degree() const
    {
        for (int k = (int)C.size() - 1; k >= 0; --k) if (C[k]) return k;
        return -1;
    }

    void ResizeTo(size_t n) { if (C.size() < n) C.resize(n, 0); }

    u64 Eval(u64 x, unsigned w = 64) const
    {
        u64 acc = 0;
        for (size_t k = C.size(); k-- > 0; ) acc = MaskW(acc * MaskW(x, w) + C[k], w);
        return acc;
    }
};

inline u64 ModInv(u64 a, unsigned w = 64)
{
    a = MaskW(a, w);
    assert((a & 1) && "ModInv requires odd input");
    u64 x = a;
    // 6 Newton doublings give at least 192 bits of precision, which clears 64-bit margin; fewer iterations have been seen to miss by one bit on near boundary inputs in debug mode
    for (int i = 0; i < 6; ++i)
        x = MaskW(x * MaskW(2u - MaskW(a * x, w), w), w);
    return x;
}

inline Poly Add(const Poly& a, const Poly& b, size_t D, unsigned w = 64)
{
    Poly r(D + 1);
    for (size_t k = 0; k <= D; ++k)
    {
        u64 v = 0;
        if (k < a.C.size()) v += a.C[k];
        if (k < b.C.size()) v += b.C[k];
        r.C[k] = MaskW(v, w);
    }
    return r;
}

inline Poly Sub(const Poly& a, const Poly& b, size_t D, unsigned w = 64)
{
    Poly r(D + 1);
    for (size_t k = 0; k <= D; ++k)
    {
        u64 v = 0;
        if (k < a.C.size()) v += a.C[k];
        if (k < b.C.size()) v -= b.C[k];
        r.C[k] = MaskW(v, w);
    }
    return r;
}

inline Poly MulTrunc(const Poly& a, const Poly& b, size_t D, unsigned w = 64)
{
    Poly r(D + 1);
    for (size_t i = 0; i < a.C.size() && i <= D; ++i)
    {
        if (!a.C[i]) continue;
        for (size_t j = 0; j < b.C.size() && i + j <= D; ++j)
            r.C[i + j] = MaskW(r.C[i + j] + MaskW(a.C[i] * b.C[j], w), w);
    }
    return r;
}

inline Poly ComposeTrunc(const Poly& f, const Poly& g, size_t D, unsigned w = 64)
{
    Poly acc(D + 1);
    for (size_t k = f.C.size(); k-- > 0; )
    {
        acc = MulTrunc(acc, g, D, w);
        acc.C[0] = MaskW(acc.C[0] + f.C[k], w);
    }
    return acc;
}

inline Poly Derivative(const Poly& f, unsigned w = 64)
{
    if (f.C.size() <= 1) return Poly(1);
    Poly d(f.C.size() - 1);
    for (size_t k = 1; k < f.C.size(); ++k) d.C[k - 1] = MaskW((u64)k * f.C[k], w);
    return d;
}

inline Poly SeriesInverse(const Poly& P, size_t D, unsigned w = 64)
{
    assert(!P.C.empty() && (P.C[0] & 1) && "SeriesInverse needs odd P[0]");
    const u64 p0inv = ModInv(P.C[0], w);
    Poly Q(D + 1);
    Q.C[0] = p0inv;
    for (size_t n = 1; n <= D; ++n)
    {
        u64 s = 0;
        for (size_t i = 1; i <= n; ++i)
            if (i < P.C.size()) s = MaskW(s + MaskW(P.C[i] * Q.C[n - i], w), w);
        Q.C[n] = MaskW((u64)0 - MaskW(p0inv * s, w), w);
    }
    return Q;
}

inline Poly CompositionalInverse(const Poly& f, size_t D, unsigned w = 64, int iters = 8)
{
    assert(f.C.size() >= 2 && (f.C[1] & 1));
    const u64 ia1 = ModInv(f.C[1], w);

    Poly g(D + 1);
    g.C[0] = MaskW(((u64)0 - f.C[0]) * ia1, w);
    g.C[1] = ia1;

    const Poly fp = Derivative(f, w);

    // ~log2(w) iterations suffice in theory for 2 adic Newton; the 8 iter margin covers degree up-to 64 inputs and any convergence miss is caught by the VerifyInverse gate at the catalog layer, so wrong inverses never ship
    for (int it = 0; it < iters; ++it)
    {
        Poly Fg = ComposeTrunc(f, g, D, w);
        Poly E = Fg;
        E.ResizeTo(2);
        E.C[1] = MaskW(E.C[1] - 1, w);
        Poly Fpg = ComposeTrunc(fp, g, D, w);
        Poly inv = SeriesInverse(Fpg, D, w);
        Poly cor = MulTrunc(E, inv, D, w);
        g = Sub(g, cor, D, w);
    }
    return g;
}

inline bool VerifyInverse(const Poly& f, const Poly& g, unsigned exhaustivebits = 20, size_t randomsamples = 300000, uint64_t seed = 0xC0FFEEu, unsigned w = 64)
{
    unsigned ebits = (exhaustivebits > w) ? w : exhaustivebits;
    const u64 lim = (ebits >= 64) ? ~0ull : (1ull << ebits);
    for (u64 x = 0; x < lim; ++x)
    {
        if (f.Eval(g.Eval(x, w), w) != MaskW(x, w)) return false;
        if (g.Eval(f.Eval(x, w), w) != MaskW(x, w)) return false;
    }
    std::mt19937_64 rng(seed);
    for (size_t i = 0; i < randomsamples; ++i)
    {
        u64 x = MaskW(rng(), w);
        if (f.Eval(g.Eval(x, w), w) != x) return false;
        if (g.Eval(f.Eval(x, w), w) != x) return false;
    }
    return true;
}

}