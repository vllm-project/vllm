#pragma once
#include "common.cuh"

namespace DOSA_NAMESPACE {
namespace record5 {
inline constexpr uint32_t Invalid = 0xFFFFFu;
// A caller first applies its domain-specific encoding. Both record writers
// then use this same low8/high12 plus exact20-bit-ID storage representation.
__device__ __forceinline__ uint32_t pack_word(uint32_t code, uint32_t id) {
  return ((code >> 8) << 20) | (id & 0xFFFFFu);
}
}  // namespace record5

// Completed original H32 scores use a global order cell. This is a different
// mathematical domain from Full's bounded surrogate, with the same5B layout.
namespace h32_record5 {
inline constexpr uint32_t Min = 0x00800u, Max = 0xFF7FFu;
inline constexpr uint32_t Invalid = record5::Invalid;

__device__ __forceinline__ uint32_t encode(float score) {
  uint32_t bits = __float_as_uint(score);
  if ((bits & 0x7FFFFFFFu) == 0) bits = 0;
  const uint32_t ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
  return (~ordered) >> 12;
}

__device__ __forceinline__ uint32_t pack_word(uint32_t cp, uint32_t id) {
  return record5::pack_word(cp, id);
}
}  // namespace h32_record5

namespace record5 {
// Keep the independent validation statement in its original word loop. In
// particular, continue must target that loop instead of returning from a visit
// callback: the latter changes NVCC's hot/cold basic-block placement here.
// Arguments must have no side effects; callers provide already-validated
// bounds.
#define DOSA_RECORD5_INDEPENDENT_WORDS(Threads, T, Words, Base, Count, Word,   \
                                       Slot, ...)                              \
  do {                                                                         \
    int dosa_r5_scalar_begin = (T);                                            \
    if ((reinterpret_cast<uintptr_t>((Words) + (Base)) & 15u) == 0) {          \
      const int dosa_r5_vectors = (Count) / 4;                                 \
      for (int dosa_r5_vector = (T); dosa_r5_vector < dosa_r5_vectors;         \
           dosa_r5_vector += (Threads)) {                                      \
        const uint4 dosa_r5_quad =                                             \
            reinterpret_cast<const uint4*>((Words) + (Base))[dosa_r5_vector];  \
        const unsigned dosa_r5_values[4] = {dosa_r5_quad.x, dosa_r5_quad.y,    \
                                            dosa_r5_quad.z, dosa_r5_quad.w};   \
        _Pragma("unroll") for (int dosa_r5_item = 0; dosa_r5_item < 4;         \
                               ++dosa_r5_item) {                               \
          const unsigned Word = dosa_r5_values[dosa_r5_item];                  \
          const int Slot = dosa_r5_vector * 4 + dosa_r5_item;                  \
          __VA_ARGS__                                                          \
        }                                                                      \
      }                                                                        \
      dosa_r5_scalar_begin = dosa_r5_vectors * 4 + (T);                        \
    }                                                                          \
    for (int Slot = dosa_r5_scalar_begin; Slot < (Count); Slot += (Threads)) { \
      const unsigned Word = unsigned((Words)[(Base) + (Slot)]);                \
      __VA_ARGS__                                                              \
    }                                                                          \
  } while (false)

// A collective visitor is declared in the caller's kernel scope. Keeping the
// loop here as statements avoids extra pointer/metadata copies through a
// helper.
#define DOSA_RECORD5_COLLECTIVE_WORDS(Threads, Warp, Lane, Words, Base, Count, \
                                      Visit)                                   \
  do {                                                                         \
    int dosa_r5_scalar_first = 0;                                              \
    if ((reinterpret_cast<uintptr_t>((Words) + (Base)) & 15u) == 0) {          \
      dosa_r5_scalar_first = ((Count) / 128) * 128;                            \
      for (int dosa_r5_begin = (Warp) * 128;                                   \
           dosa_r5_begin < dosa_r5_scalar_first;                               \
           dosa_r5_begin += (Threads) * 4) {                                   \
        const int dosa_r5_slot = dosa_r5_begin + (Lane) * 4;                   \
        const uint4 dosa_r5_quad =                                             \
            *reinterpret_cast<const uint4*>((Words) + (Base) + dosa_r5_slot);  \
        const unsigned dosa_r5_values[4] = {dosa_r5_quad.x, dosa_r5_quad.y,    \
                                            dosa_r5_quad.z, dosa_r5_quad.w};   \
        _Pragma("unroll") for (int dosa_r5_item = 0; dosa_r5_item < 4;         \
                               ++dosa_r5_item)                                 \
            Visit(dosa_r5_values[dosa_r5_item], dosa_r5_slot + dosa_r5_item,   \
                  true);                                                       \
      }                                                                        \
    }                                                                          \
    for (int dosa_r5_begin = dosa_r5_scalar_first + (Warp) * 32;               \
         dosa_r5_begin < (Count); dosa_r5_begin += (Threads)) {                \
      const int dosa_r5_slot = dosa_r5_begin + (Lane);                         \
      const bool dosa_r5_live = dosa_r5_slot < (Count);                        \
      Visit(dosa_r5_live ? unsigned((Words)[(Base) + dosa_r5_slot]) : 0u,      \
            dosa_r5_slot, dosa_r5_live);                                       \
    }                                                                          \
  } while (false)

// Used in a block scope after the caller's certified B/E ballots and optional
// uniformly-empty-warp exit. Keep scalar locals instead of an aggregate return;
// the lifetime and register layout then match the qualified reserve sequence.
#define DOSA_RECORD5_RESERVE(Full, Lane, BM, EM, BC, EC, BBase, EBase, Before) \
  int BBase = 0, EBase = 0;                                                    \
  if ((Lane) == 0) {                                                           \
    if (BM) BBase = atomicAdd((BC), __popc(BM));                               \
    if (EM) EBase = atomicAdd((EC), __popc(EM));                               \
  }                                                                            \
  BBase = __shfl_sync((Full), BBase, 0);                                       \
  EBase = __shfl_sync((Full), EBase, 0);                                       \
  const unsigned Before = (Lane) ? ((1u << (Lane)) - 1u) : 0u
}  // namespace record5

}  // namespace DOSA_NAMESPACE
