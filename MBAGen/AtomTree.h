#pragma once

#include "MBABasis.h"

#include <algorithm>
#include <memory>
#include <random>
#include <vector>

namespace mbagen
{

enum class AtomOp
{
    Var,
    Not,
    And,
    Or,
    Xor,
    Const
};

struct AtomTree
{
    AtomOp Op;
    unsigned VarIdx;
    u64 ConstVal;
    std::vector<std::shared_ptr<const AtomTree>> Kids;
};

using AtomTreePtr = std::shared_ptr<const AtomTree>;

inline AtomTreePtr MakeVar(unsigned k)
{
    auto t = std::make_shared<AtomTree>();
    t->Op = AtomOp::Var;
    t->VarIdx = k;
    return t;
}

inline AtomTreePtr MakeConst(u64 v)
{
    auto t = std::make_shared<AtomTree>();
    t->Op = AtomOp::Const;
    t->ConstVal = v;
    return t;
}

inline AtomTreePtr MakeNot(AtomTreePtr a)
{
    auto t = std::make_shared<AtomTree>();
    t->Op = AtomOp::Not;
    t->Kids = { std::move(a) };
    return t;
}

inline AtomTreePtr MakeAnd(AtomTreePtr a, AtomTreePtr b)
{
    auto t = std::make_shared<AtomTree>();
    t->Op = AtomOp::And;
    t->Kids = { std::move(a), std::move(b) };
    return t;
}

inline AtomTreePtr MakeOr(AtomTreePtr a, AtomTreePtr b)
{
    auto t = std::make_shared<AtomTree>();
    t->Op = AtomOp::Or;
    t->Kids = { std::move(a), std::move(b) };
    return t;
}

inline AtomTreePtr MakeXor(AtomTreePtr a, AtomTreePtr b)
{
    auto t = std::make_shared<AtomTree>();
    t->Op = AtomOp::Xor;
    t->Kids = { std::move(a), std::move(b) };
    return t;
}

inline u64 EvalAtomTree(const AtomTree& t, const u64* inputs, unsigned w)
{
    switch (t.Op)
    {
        case AtomOp::Var: return MaskW(inputs[t.VarIdx], w);
        case AtomOp::Const: return MaskW(t.ConstVal, w);
        case AtomOp::Not: return MaskW(~EvalAtomTree(*t.Kids[0], inputs, w), w);
        case AtomOp::And:
        {
            u64 r = MaskW(~0ull, w);
            for (const auto& k : t.Kids) r = MaskW(r & EvalAtomTree(*k, inputs, w), w);
            return r;
        }
        case AtomOp::Or:
        {
            u64 r = 0;
            for (const auto& k : t.Kids) r = MaskW(r | EvalAtomTree(*k, inputs, w), w);
            return r;
        }
        case AtomOp::Xor:
        {
            u64 r = 0;
            for (const auto& k : t.Kids) r = MaskW(r ^ EvalAtomTree(*k, inputs, w), w);
            return r;
        }
    }
    return 0;
}

inline bool TreesEqual(const AtomTreePtr& a, const AtomTreePtr& b)
{
    if (a.get() == b.get()) return true;
    if (!a || !b) return false;
    if (a->Op != b->Op) return false;
    switch (a->Op)
    {
        case AtomOp::Var: return a->VarIdx == b->VarIdx;
        case AtomOp::Const: return a->ConstVal == b->ConstVal;
        case AtomOp::Not: return TreesEqual(a->Kids[0], b->Kids[0]);
        case AtomOp::And:
        case AtomOp::Or:
        case AtomOp::Xor:
        {
            if (a->Kids.size() != b->Kids.size()) return false;
            // order independent match for commutative ops; O(N^2) in child count but T8 generator output stays shallow (depth 1-2 typical) so overhead is noise, and a canonical sort alternative would require a total order on AtomTrees that the shared_ptr identity breaks
            std::vector<bool> used(b->Kids.size(), false);
            for (const auto& ak : a->Kids)
            {
                bool found = false;
                for (size_t i = 0; i < b->Kids.size(); ++i)
                {
                    if (used[i]) continue;
                    if (TreesEqual(ak, b->Kids[i])) { used[i] = true; found = true; break; }
                }
                if (!found) return false;
            }
            return true;
        }
    }
    return false;
}

inline u64 HashCombine(u64 h, u64 x)
{
    h ^= x + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    return h;
}

inline u64 CanonicalHash(const AtomTreePtr& t)
{
    if (!t) return 0;
    u64 h = (u64)(unsigned)t->Op * 0x9E3779B97F4A7C15ull + 0xA701u;
    switch (t->Op)
    {
        case AtomOp::Var: h = HashCombine(h, (u64)t->VarIdx); break;
        case AtomOp::Const: h = HashCombine(h, t->ConstVal); break;
        case AtomOp::Not: h = HashCombine(h, CanonicalHash(t->Kids[0])); break;
        case AtomOp::And:
        case AtomOp::Or:
        case AtomOp::Xor:
        {
            // bug: sort child hashes BEFORE combining so commutative ops hash order-independently; without the sort, (a op b) and (b op a) would hash to different buckets and CanonicalizeTerms would fail to merge them
            std::vector<u64> childh;
            childh.reserve(t->Kids.size());
            for (const auto& k : t->Kids) childh.push_back(CanonicalHash(k));
            std::sort(childh.begin(), childh.end());
            for (u64 ch : childh) h = HashCombine(h, ch);
            break;
        }
    }
    return h;
}

inline AtomTreePtr NormalizeNot(AtomTreePtr t);

static AtomTreePtr PushNotThrough(const AtomTree& node)
{
    switch (node.Op)
    {
        case AtomOp::And:
        {
            std::vector<AtomTreePtr> negatedkids;
            negatedkids.reserve(node.Kids.size());
            for (const auto& k : node.Kids)
                negatedkids.push_back(NormalizeNot(MakeNot(k)));
            AtomTreePtr r = std::move(negatedkids[0]);
            for (size_t i = 1; i < negatedkids.size(); ++i)
                r = MakeOr(r, std::move(negatedkids[i]));
            return r;
        }
        case AtomOp::Or:
        {
            std::vector<AtomTreePtr> negatedkids;
            negatedkids.reserve(node.Kids.size());
            for (const auto& k : node.Kids)
                negatedkids.push_back(NormalizeNot(MakeNot(k)));
            AtomTreePtr r = std::move(negatedkids[0]);
            for (size_t i = 1; i < negatedkids.size(); ++i)
                r = MakeAnd(r, std::move(negatedkids[i]));
            return r;
        }
        case AtomOp::Xor:
        {
            // Xor negation absorbs into exactly one operand (~(a^b) = ~a ^ b, not ~a ^ ~b); picking the first operand by convention is arbitrary but must be consistent, otherwise two emit paths produce structurally different trees for the same input
            std::vector<AtomTreePtr> kids;
            kids.reserve(node.Kids.size());
            kids.push_back(NormalizeNot(MakeNot(node.Kids[0])));
            for (size_t i = 1; i < node.Kids.size(); ++i)
                kids.push_back(NormalizeNot(node.Kids[i]));
            AtomTreePtr r = std::move(kids[0]);
            for (size_t i = 1; i < kids.size(); ++i)
                r = MakeXor(r, std::move(kids[i]));
            return r;
        }
        default:
            return std::make_shared<AtomTree>(node);
    }
}

inline AtomTreePtr NormalizeNot(AtomTreePtr t)
{
    switch (t->Op)
    {
        case AtomOp::Var:
        case AtomOp::Const:
            return t;
        case AtomOp::Not:
        {
            const AtomTree& child = *t->Kids[0];
            if (child.Op == AtomOp::Var) return t;
            if (child.Op == AtomOp::Const)
                return MakeConst(~child.ConstVal);
            if (child.Op == AtomOp::Not)
                return NormalizeNot(child.Kids[0]);
            return PushNotThrough(child);
        }
        case AtomOp::And:
        case AtomOp::Or:
        case AtomOp::Xor:
        {
            auto nt = std::make_shared<AtomTree>();
            nt->Op = t->Op;
            nt->Kids.reserve(t->Kids.size());
            for (const auto& k : t->Kids)
                nt->Kids.push_back(NormalizeNot(k));
            return nt;
        }
    }
    return t;
}

inline AtomTreePtr RewriteIdentities(AtomTreePtr t, unsigned w);

static bool IsComplementPair(const AtomTreePtr& a, const AtomTreePtr& b)
{
    if (a->Op == AtomOp::Not && TreesEqual(a->Kids[0], b)) return true;
    if (b->Op == AtomOp::Not && TreesEqual(b->Kids[0], a)) return true;
    return false;
}

static void FlattenInto(const AtomTreePtr& t, AtomOp op, std::vector<AtomTreePtr>& out)
{
    if (t->Op == op)
        for (const auto& k : t->Kids) FlattenInto(k, op, out);
    else
        out.push_back(t);
}

static AtomTreePtr RebuildChain(AtomOp op, std::vector<AtomTreePtr> kids)
{
    if (kids.empty())
    {
        // empty And is "all ones", empty Or is 0, empty Xor is 0; caller should have handled these cases but defensive fallback here prevents a wrong Const(0) landing in an empty And slot
        if (op == AtomOp::And) return MakeConst(~u64(0));
        return MakeConst(0);
    }
    if (kids.size() == 1) return kids[0];
    auto t = std::make_shared<AtomTree>();
    t->Op = op;
    t->Kids = std::move(kids);
    return t;
}

static AtomTreePtr RewriteAnd(std::vector<AtomTreePtr> kids, unsigned w)
{
    const u64 ONES = MaskW(~u64(0), w);
    std::vector<AtomTreePtr> out;
    for (auto& k : kids)
    {
        if (k->Op == AtomOp::Const)
        {
            const u64 c = MaskW(k->ConstVal, w);
            if (c == 0) return MakeConst(0);
            if (c == ONES) continue;
            out.push_back(k);
        }
        else
        {
            out.push_back(std::move(k));
        }
    }
    std::vector<AtomTreePtr> dedup;
    for (auto& k : out)
    {
        bool dup = false;
        for (auto& e : dedup)
            if (TreesEqual(k, e)) { dup = true; break; }
        if (!dup) dedup.push_back(std::move(k));
    }
    out = std::move(dedup);
    for (size_t i = 0; i < out.size(); ++i)
        for (size_t j = i + 1; j < out.size(); ++j)
            if (IsComplementPair(out[i], out[j])) return MakeConst(0);
    std::vector<AtomTreePtr> afterabs;
    afterabs.reserve(out.size());
    for (auto& k : out)
    {
        bool absorbed = false;
        if (k->Op == AtomOp::Or)
        {
            std::vector<AtomTreePtr> orkids;
            FlattenInto(k, AtomOp::Or, orkids);
            for (auto& ok : orkids)
            {
                for (auto& other : out)
                {
                    if (other.get() == k.get()) continue;
                    if (TreesEqual(ok, other)) { absorbed = true; break; }
                }
                if (absorbed) break;
            }
        }
        if (!absorbed) afterabs.push_back(std::move(k));
    }
    out = std::move(afterabs);
    {
        // snapshot sibling view before any std::move into aftercons; IsComplementPair dereferences both operands without null checks, so a sibling slot that an earlier iteration moved out would be a null shared_ptr and crash the solve; "self" is identified by index into the snapshot rather than by get()-identity
        const std::vector<AtomTreePtr> snapshot = out;
        std::vector<AtomTreePtr> aftercons;
        aftercons.reserve(out.size());
        bool anyshrunktozero = false;
        for (size_t ki = 0; ki < out.size(); ++ki)
        {
            auto& k = out[ki];
            if (k->Op != AtomOp::Or)
            {
                aftercons.push_back(std::move(k));
                continue;
            }
            std::vector<AtomTreePtr> orkids;
            FlattenInto(k, AtomOp::Or, orkids);
            std::vector<AtomTreePtr> kept;
            kept.reserve(orkids.size());
            for (auto& ok : orkids)
            {
                bool drop = false;
                for (size_t si = 0; si < snapshot.size(); ++si)
                {
                    if (si == ki) continue;
                    if (IsComplementPair(ok, snapshot[si])) { drop = true; break; }
                }
                if (!drop) kept.push_back(std::move(ok));
            }
            if (kept.size() == orkids.size()) aftercons.push_back(std::move(k));
            else if (kept.empty()) { anyshrunktozero = true; break; }
            else if (kept.size() == 1) aftercons.push_back(std::move(kept[0]));
            else aftercons.push_back(RebuildChain(AtomOp::Or, std::move(kept)));
        }
        if (anyshrunktozero) return MakeConst(0);
        out = std::move(aftercons);
    }
    if (out.empty()) return MakeConst(ONES);
    if (out.size() == 1) return out[0];
    return RebuildChain(AtomOp::And, std::move(out));
}

static AtomTreePtr RewriteOr(std::vector<AtomTreePtr> kids, unsigned w)
{
    const u64 ONES = MaskW(~u64(0), w);
    std::vector<AtomTreePtr> out;
    for (auto& k : kids)
    {
        if (k->Op == AtomOp::Const)
        {
            const u64 c = MaskW(k->ConstVal, w);
            if (c == ONES) return MakeConst(ONES);
            if (c == 0) continue;
            out.push_back(k);
        }
        else
        {
            out.push_back(std::move(k));
        }
    }
    std::vector<AtomTreePtr> dedup;
    for (auto& k : out)
    {
        bool dup = false;
        for (auto& e : dedup)
            if (TreesEqual(k, e)) { dup = true; break; }
        if (!dup) dedup.push_back(std::move(k));
    }
    out = std::move(dedup);
    for (size_t i = 0; i < out.size(); ++i)
        for (size_t j = i + 1; j < out.size(); ++j)
            if (IsComplementPair(out[i], out[j])) return MakeConst(ONES);
    std::vector<AtomTreePtr> afterabs;
    afterabs.reserve(out.size());
    for (auto& k : out)
    {
        bool absorbed = false;
        if (k->Op == AtomOp::And)
        {
            std::vector<AtomTreePtr> andkids;
            FlattenInto(k, AtomOp::And, andkids);
            for (auto& ak : andkids)
            {
                for (auto& other : out)
                {
                    if (other.get() == k.get()) continue;
                    if (TreesEqual(ak, other)) { absorbed = true; break; }
                }
                if (absorbed) break;
            }
        }
        if (!absorbed) afterabs.push_back(std::move(k));
    }
    out = std::move(afterabs);
    {
        const std::vector<AtomTreePtr> snapshot = out;
        std::vector<AtomTreePtr> aftercons;
        aftercons.reserve(out.size());
        bool anyshrunktoones = false;
        for (size_t ki = 0; ki < out.size(); ++ki)
        {
            auto& k = out[ki];
            if (k->Op != AtomOp::And)
            {
                aftercons.push_back(std::move(k));
                continue;
            }
            std::vector<AtomTreePtr> andkids;
            FlattenInto(k, AtomOp::And, andkids);
            std::vector<AtomTreePtr> kept;
            kept.reserve(andkids.size());
            for (auto& ak : andkids)
            {
                bool drop = false;
                for (size_t si = 0; si < snapshot.size(); ++si)
                {
                    if (si == ki) continue;
                    if (IsComplementPair(ak, snapshot[si])) { drop = true; break; }
                }
                if (!drop) kept.push_back(std::move(ak));
            }
            if (kept.size() == andkids.size()) aftercons.push_back(std::move(k));
            else if (kept.empty()) { anyshrunktoones = true; break; }
            else if (kept.size() == 1) aftercons.push_back(std::move(kept[0]));
            else aftercons.push_back(RebuildChain(AtomOp::And, std::move(kept)));
        }
        if (anyshrunktoones) return MakeConst(ONES);
        out = std::move(aftercons);
    }
    if (out.empty()) return MakeConst(0);
    if (out.size() == 1) return out[0];
    return RebuildChain(AtomOp::Or, std::move(out));
}

static AtomTreePtr RewriteXor(std::vector<AtomTreePtr> kids, unsigned w)
{
    const u64 ONES = MaskW(~u64(0), w);
    bool invertparity = false;
    std::vector<AtomTreePtr> out;
    for (auto& k : kids)
    {
        if (k->Op == AtomOp::Const)
        {
            const u64 c = MaskW(k->ConstVal, w);
            if (c == 0) continue;
            if (c == ONES) { invertparity ^= true; continue; }
            out.push_back(k);
        }
        else
        {
            out.push_back(std::move(k));
        }
    }
    std::vector<AtomTreePtr> cancelled;
    cancelled.reserve(out.size());
    std::vector<bool> killed(out.size(), false);
    for (size_t i = 0; i < out.size(); ++i)
    {
        if (killed[i]) continue;
        bool paired = false;
        for (size_t j = i + 1; j < out.size(); ++j)
        {
            if (killed[j]) continue;
            if (TreesEqual(out[i], out[j])) { killed[j] = true; paired = true; break; }
        }
        if (!paired) cancelled.push_back(out[i]);
    }
    out = std::move(cancelled);
    {
        std::vector<AtomTreePtr> aftercomp;
        std::vector<bool> killed2(out.size(), false);
        for (size_t i = 0; i < out.size(); ++i)
        {
            if (killed2[i]) continue;
            bool paired = false;
            for (size_t j = i + 1; j < out.size(); ++j)
            {
                if (killed2[j]) continue;
                if (IsComplementPair(out[i], out[j]))
                {
                    killed2[j] = true;
                    // each complement pair toggles parity, not cancels, because a ^ ~a == all-ones in Z/2^w; cancelling instead would produce wrong results on any emission with an odd count of complement pairs
                    invertparity ^= true;
                    paired = true;
                    break;
                }
            }
            if (!paired) aftercomp.push_back(out[i]);
        }
        out = std::move(aftercomp);
    }
    if (out.empty())
        return invertparity ? MakeConst(ONES) : MakeConst(0);
    AtomTreePtr core = (out.size() == 1) ? out[0] : RebuildChain(AtomOp::Xor, std::move(out));
    if (invertparity)
    {
        // wrap in Not and let NormalizeNot push it to leaves on re entry; the push may unlock further rewrites (e.g. a leaf Not matching another sibling's complement pair detection) that this round missed
        auto n = std::make_shared<AtomTree>();
        n->Op = AtomOp::Not;
        n->Kids = { core };
        return n;
    }
    return core;
}

static AtomTreePtr RewriteStep(AtomTreePtr t, unsigned w)
{
    switch (t->Op)
    {
        case AtomOp::Var:
        case AtomOp::Const:
            return t;
        case AtomOp::Not:
        {
            AtomTreePtr child = RewriteStep(t->Kids[0], w);
            if (child->Op == AtomOp::Const)
                return MakeConst(MaskW(~child->ConstVal, w));
            if (child->Op == AtomOp::Not)
                return child->Kids[0];
            auto n = std::make_shared<AtomTree>();
            n->Op = AtomOp::Not;
            n->Kids = { child };
            return n;
        }
        case AtomOp::And:
        case AtomOp::Or:
        case AtomOp::Xor:
        {
            std::vector<AtomTreePtr> flat;
            for (const auto& k : t->Kids)
            {
                AtomTreePtr simp = RewriteStep(k, w);
                if (simp->Op == t->Op)
                    for (const auto& kk : simp->Kids) flat.push_back(kk);
                else
                    flat.push_back(std::move(simp));
            }
            if (t->Op == AtomOp::And) return RewriteAnd(std::move(flat), w);
            if (t->Op == AtomOp::Or) return RewriteOr(std::move(flat), w);
            return RewriteXor(std::move(flat), w);
        }
    }
    return t;
}

inline AtomTreePtr RewriteIdentities(AtomTreePtr t, unsigned w)
{
    // 8 iteration cap is a safety net; healthy T8 output stabilises in 1-2, and interleaving NormalizeNot with the step lets complement pair detection see leaf NOT form on the next iteration which unlocks further rewrites that would otherwise be missed
    for (int iter = 0; iter < 8; ++iter)
    {
        AtomTreePtr next = RewriteStep(t, w);
        next = NormalizeNot(next);
        if (TreesEqual(next, t)) return next;
        t = next;
    }
    return t;
}

inline AtomTreePtr RandomAtomTree(std::mt19937_64& rng, unsigned V, unsigned depth)
{
    // 1 in 5 early terminate per node makes trees vary in shape, not just size; without the roll, trees at the same depth are structurally similar and give a pattern-learning attacker a size signal
    if (depth == 0 || (rng() % 5u) == 0u)
        return MakeVar((unsigned)(rng() % V));
    switch (rng() % 4u)
    {
        case 0: return MakeNot(RandomAtomTree(rng, V, depth - 1));
        case 1: return MakeAnd(RandomAtomTree(rng, V, depth - 1), RandomAtomTree(rng, V, depth - 1));
        case 2: return MakeOr(RandomAtomTree(rng, V, depth - 1), RandomAtomTree(rng, V, depth - 1));
        default: return MakeXor(RandomAtomTree(rng, V, depth - 1), RandomAtomTree(rng, V, depth - 1));
    }
}

}