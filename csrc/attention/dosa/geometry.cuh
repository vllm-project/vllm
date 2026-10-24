#pragma once
#include "common.cuh"

namespace DOSA_NAMESPACE {
#ifndef DOSA_SCORE_HEADS
  #define DOSA_SCORE_HEADS 32
#endif
namespace dosa_geometry {
constexpr int Heads = DOSA_SCORE_HEADS;
static_assert(Heads == 32 || Heads == 64, "DoSA supports H32 or H64");
// Every pipeline consumer accepts the same registered certificate families.
__host__ __device__ constexpr bool valid_certificate_budget(int budget) {
  return budget == 14 || budget == 16 || budget == 20 ||
         (Heads == 64 && budget == 28);
}
using HeadMask = std::conditional_t<Heads == 32, int32_t, int64_t>;
using UnsignedHeadMask = std::make_unsigned_t<HeadMask>;
}  // namespace dosa_geometry

}  // namespace DOSA_NAMESPACE
