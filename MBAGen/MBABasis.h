#pragma once

#include <cstdint>

namespace mbagen
{

using u64 = uint64_t;

static inline u64 MaskW(u64 v, unsigned w)
{
    return (w >= 64) ? v : (v & ((u64(1) << w) - 1));
}

}