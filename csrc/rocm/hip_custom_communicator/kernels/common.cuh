#pragma once

// Device primitives the kernels share: load, store, send, recv, barrier,
// reduce, and the INT4 quantize and dequantize.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace hip_custom_communicator {

#define DINLINE __device__ __forceinline__

template <typename DTYPE, int VEC_SIZE>
struct __align__(sizeof(DTYPE) * VEC_SIZE) vec {
  DTYPE d[VEC_SIZE];
};

template <typename DTYPE, int PACK_BYTES>
struct traits {
  static constexpr int N = PACK_BYTES / sizeof(DTYPE);
};

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(1))) u32x4 global_u32x4;

typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));

#ifndef HIP_CUSTOM_COMMUNICATOR_DEBUG
  #define HIP_CUSTOM_COMMUNICATOR_DEBUG 0
#endif

constexpr int kInt4Group = 32;
constexpr int kInt4Lowest = -8;
constexpr int kInt4Highest = 7;
constexpr int kInt4WordBytes = 4;
constexpr int kInt4ScaleBytes = 2;
constexpr uint32_t kInt4ZeroWord = 0x88888888u;

constexpr int kInt4GroupPacks = kInt4Group / (2 * kInt4WordBytes);

__host__ __device__ constexpr int64_t int4_groups(int64_t n) {
  return (n + kInt4Group - 1) / kInt4Group;
}

constexpr int64_t kInt4ScalesAlign = 256;
__host__ __device__ constexpr int64_t int4_scales_offset(int64_t m, int64_t n) {
  return (m * n / 2 + kInt4ScalesAlign - 1) / kInt4ScalesAlign *
         kInt4ScalesAlign;
}

__host__ __device__ constexpr int64_t int4_payload_bytes(int64_t m, int64_t n) {
  return int4_scales_offset(m, n) + m * int4_groups(n) * kInt4ScaleBytes;
}

constexpr int kLineBytes = 16;
constexpr int kLinePayloadBytes = kLineBytes / 2;

enum class Where : uint32_t {
  start = 1,
  peer_barrier,
};

enum class FlagState : uint32_t {
  clear = 0,
  arrived = 1,
};

enum class FaultState : uint32_t {
  none = 0,
  writing = 1,
  written = 2,
};

enum class StatusField : int {
  abort,
  state,
  where,
  rank,
  block,
  peer,
  count,
  want,
  elapsed_low,

  elapsed_high,
};

constexpr int kStatusFields = static_cast<int>(StatusField::elapsed_high) + 1;

typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));

template <typename U>
constexpr bool kBf16 = sizeof(U) == 2 && std::numeric_limits<U>::digits == 8;

typedef __attribute__((address_space(1))) uint32_t global_u32;
typedef __attribute__((address_space(1))) __fp16 global_f16;

enum class Layout { tile, run, flat, group_run, group_flat, line };

template <int LOWEST = kInt4Lowest, int HIGHEST = kInt4Highest, typename U,
          int P>
// An encoded INT4 pack is a word of 8 values and its group's fp16 scale.
// Only quantize and dequantize read inside it.
DINLINE void dequantize(const uint32_t (&encoded)[2], U (&output)[P]);
template <typename DTYPE, int LOWEST = kInt4Lowest, int HIGHEST = kInt4Highest,
          int SLOTS, int K, int P>
DINLINE void dequantize(const uint32_t (&encoded)[SLOTS][1][K][2],
                        float (&sum)[1][K][P]);
template <int LOWEST = kInt4Lowest, int HIGHEST = kInt4Highest, int PACKS,
          typename T, int P>
DINLINE void quantize(const T (&input)[PACKS][P],
                      uint32_t (&encoded)[PACKS][2]);
template <typename T, int K, int P>
DINLINE void quantize(const T (&input)[1][K][P], uint32_t (&encoded)[1][K][2]);

namespace internal {

template <typename U = void, typename T>
DINLINE std::conditional_t<std::is_void_v<U>, T, U>* ptr_array_at(
    T* const* x_ptr_array, int x_ptr_array_size, int r) {
  r = __builtin_amdgcn_readfirstlane(r);
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG && (r < 0 || r >= x_ptr_array_size))
    __builtin_trap();
  typedef T* Entry;
  return reinterpret_cast<std::conditional_t<std::is_void_v<U>, T, U>*>(
      ((const __attribute__((address_space(4))) Entry*)x_ptr_array)[r]);
}

template <typename U, int WORLD, typename T>
DINLINE std::conditional_t<std::is_void_v<U>, T, U>* lane_ptr_array_at(
    T* const* x_ptr_array, int x_ptr_array_size, int r) {
  static_assert(WORLD >= 1, "a world has a rank");
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
      (WORLD - 1 < 0 || WORLD - 1 >= x_ptr_array_size))
    __builtin_trap();
  const auto scalar = [](T* p) {
    const auto u = reinterpret_cast<uint64_t>(p);
    const uint32_t lo =
        __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(u));
    const uint32_t hi =
        __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(u >> 32));
    return (uint64_t{hi} << 32) | lo;
  };
  uint64_t at = scalar(ptr_array_at(x_ptr_array, x_ptr_array_size, 0));
#pragma unroll
  for (int k = 1; k < WORLD; ++k) {
    const uint64_t p = scalar(ptr_array_at(x_ptr_array, x_ptr_array_size, k));
    at = r == k ? p : at;
  }
  auto* const global =
      reinterpret_cast<__attribute__((address_space(1))) char*>(at);
  return reinterpret_cast<std::conditional_t<std::is_void_v<U>, T, U>*>(
      (char*)global);
}

DINLINE void give_up(uint32_t* status_ptr, int64_t status_stride_field,
                     int status_size_field, Where where, int rank,
                     uint32_t peer, uint32_t count, uint32_t want,
                     uint64_t waited) {
  const auto word = [&](StatusField f) {
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
        (static_cast<int64_t>(f) < 0 ||
         static_cast<int64_t>(f) >= status_size_field))
      __builtin_trap();
    return status_ptr + static_cast<int64_t>(f) * status_stride_field;
  };
  uint32_t none = static_cast<uint32_t>(FaultState::none);
  if (__scoped_atomic_compare_exchange_n(
          word(StatusField::state), &none,
          static_cast<uint32_t>(FaultState::writing), false, __ATOMIC_RELAXED,
          __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM)) {
    *word(StatusField::where) = static_cast<uint32_t>(where);
    *word(StatusField::rank) = static_cast<uint32_t>(rank);
    *word(StatusField::block) = blockIdx.x;
    *word(StatusField::peer) = peer;
    *word(StatusField::count) = count;
    *word(StatusField::want) = want;
    *word(StatusField::elapsed_low) = static_cast<uint32_t>(waited);
    *word(StatusField::elapsed_high) = static_cast<uint32_t>(waited >> 32);
    __scoped_atomic_store_n(word(StatusField::state),
                            static_cast<uint32_t>(FaultState::written),
                            __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
  }
  __builtin_trap();
}

template <int WORLD, int TILE_M, int THREADS_PER_BLOCK, int WAVE_SIZE,
          bool QUANTIZE_INT4, Layout LAYOUT, bool TAIL, typename OUTPUT>
DINLINE void read_buffer(
    const void* input, int input_slot, int64_t input_stride_slot,
    int64_t input_stride_m, int64_t input_stride_n, int input_size_slot,
    int input_size_m, int input_size_n, int clip_m, int clip_n, OUTPUT& output,
    int offs_m, int offs_n, int row_step, int rank = 0,
    const uint32_t* flags = nullptr, uint32_t* status_ptr = nullptr,
    int64_t status_stride_field = 0, int status_size_field = 0,
    uint64_t timeout_ticks = 0) {
#if defined(__AMDGCN_WAVEFRONT_SIZE)
  static_assert(__AMDGCN_WAVEFRONT_SIZE == WAVE_SIZE,
                "the compiler's wave is not the one handed");
#endif
  constexpr bool kSlots = std::rank_v<OUTPUT> == 4;
  constexpr int kCopies = kSlots ? std::extent_v<OUTPUT, 0> : 1;
  constexpr int kRows = std::extent_v < OUTPUT, kSlots ? 1 : 0 > ;
  constexpr int kPacks = std::extent_v < OUTPUT, kSlots ? 2 : 1 > ;
  constexpr int kPackElems = std::extent_v < OUTPUT, kSlots ? 3 : 2 > ;
  using U = std::remove_all_extents_t<OUTPUT>;
  constexpr bool kRaw = std::is_same_v<U, uint32_t>;
  static_assert(!kRaw || (QUANTIZE_INT4 && kPackElems == 2),
                "uint32_t registers are an INT4 payload's words and scales");

  constexpr int kThreadsM = TILE_M / kRows;
  constexpr int kThreadsN = THREADS_PER_BLOCK / kThreadsM;
  static_assert(kThreadsM * kRows == TILE_M, "a tile is whole rows a thread");
  static_assert(kThreadsM * kThreadsN == THREADS_PER_BLOCK,
                "the layout is the block");
  using V = vec<U, kPackElems>;

  using E = std::conditional_t<!QUANTIZE_INT4, const U, const uint8_t>;
  const auto base = [&](int c) -> E* {
    const int slot = kSlots ? (input_slot + c) % WORLD : input_slot;
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG && (slot < 0 || slot >= input_size_slot))
      __builtin_trap();
    return reinterpret_cast<E*>(input) + slot * input_stride_slot;
  };
  const int tid = static_cast<int>(threadIdx.x);

  const auto row_at = [&](int i) {
    return offs_m +
           (kThreadsM == 1 ? i : tid / kThreadsN + i * kThreadsM) * row_step;
  };
  const auto row_of = [&](int i) {
    return row_at(i) < clip_m ? row_at(i) : clip_m - 1;
  };

  const auto skips = [&](int i) {
    return !(kRows == 1 && kThreadsM == 1) && row_at(i) >= clip_m;
  };

  constexpr int kElems = kRaw ? kInt4Group / kInt4GroupPacks : kPackElems;
  const auto pack_of = [&](int k) {
    const int n = offs_n / kElems + tid % kThreadsN + k * kThreadsN;
    return n < clip_n / kElems ? n : clip_n / kElems - 1;
  };

  [[maybe_unused]] const auto element_at =
      [](auto* ptr, int64_t stride_m, int64_t stride_n, int64_t e, int n) {
        if (stride_m == int64_t{n} * stride_n) return ptr + e * stride_n;
        const int64_t row =
            e < (int64_t{1} << 31)
                ? int64_t{static_cast<uint32_t>(e) / static_cast<uint32_t>(n)}
                : e / n;
        return ptr + row * stride_m + (e - row * n) * stride_n;
      };
  constexpr bool kGroup =
      LAYOUT == Layout::group_run || LAYOUT == Layout::group_flat;
  constexpr bool kRun = LAYOUT == Layout::run || LAYOUT == Layout::group_run;
  if constexpr (LAYOUT == Layout::line) {
    static_assert(kSlots && kCopies == WORLD,
                  "a line read is every writer's copy");

    constexpr int kLines = QUANTIZE_INT4 || sizeof(V) <= kLinePayloadBytes
                               ? 1
                               : sizeof(V) / kLinePayloadBytes;
    static_assert(
        (QUANTIZE_INT4 ? kRaw || kPackElems == kInt4Group / kInt4GroupPacks
                       : sizeof(V) <= kLinePayloadBytes ||
                             sizeof(V) % kLinePayloadBytes == 0) &&
            kLines >= 1 && kLines <= 2 && sizeof(u32x4) == kLineBytes,
        "a pack is one or two lines, an INT4 pack a word of 8 values");
    constexpr int kLineScale = kLineBytes / kLinePayloadBytes;
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
        (input_slot < 0 || input_slot + WORLD > input_size_slot ||
         flags == nullptr))
      __builtin_trap();

    const char* const first = reinterpret_cast<const char*>(input) +
                              int64_t{input_slot} * input_stride_slot *
                                  int64_t{sizeof(U)} * kLineScale;
    const uint32_t slot_bytes = static_cast<uint32_t>(
        input_stride_slot * int64_t{sizeof(U)} * kLineScale);

    const uint32_t half =
        static_cast<uint32_t>(input_stride_m * int64_t{sizeof(U)});
    const int bank_rows = input_size_m / 2;
    uint32_t line_at[kRows][kPacks];
    bool waits[kRows][kPacks];
#pragma unroll
    for (int i = 0; i < kRows; ++i)
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        const int row = row_at(i) + static_cast<int>(flags[i] & 1u) * bank_rows;
        if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
            (row < 0 || row >= input_size_m || flags[i] == 0))
          __builtin_trap();
        line_at[i][k] =
            static_cast<uint32_t>(row) *
                static_cast<uint32_t>(input_stride_m * int64_t{sizeof(U)}) *
                kLineScale +
            static_cast<uint32_t>(pack_of(k)) * uint32_t{kLineBytes};
        waits[i][k] =
            row_at(i) < clip_m &&
            offs_n / kElems + tid % kThreadsN + k * kThreadsN < clip_n / kElems;
      }
    u32x4 got[WORLD][kRows][kPacks][kLines];
    const uint64_t t0 = wall_clock64();
    for (uint32_t n = 1;; ++n) {
#pragma unroll
      for (int c = 0; c < WORLD; ++c)
#pragma unroll
        for (int i = 0; i < kRows; ++i)
#pragma unroll
          for (int k = 0; k < kPacks; ++k)
#pragma unroll
            for (int h = 0; h < kLines; ++h) {
              uint32_t at = static_cast<uint32_t>(c) * slot_bytes +
                            line_at[i][k] + h * half;
              asm volatile("" : "+v"(at));
              got[c][i][k][h] =
                  __builtin_nontemporal_load((const global_u32x4*)(first + at));
            }

      asm volatile("" ::: "memory");
      bool missing = false;
#pragma unroll
      for (int c = 0; c < WORLD; ++c)
#pragma unroll
        for (int i = 0; i < kRows; ++i)
#pragma unroll
          for (int k = 0; k < kPacks; ++k)
#pragma unroll
            for (int h = 0; h < kLines; ++h)
              missing |= waits[i][k] && (got[c][i][k][h][1] != flags[i] ||
                                         got[c][i][k][h][3] != flags[i]);

      if (__builtin_amdgcn_ballot_w64(missing) == 0) break;
      if ((n & 255u) != 0) continue;

      const uint64_t waited = wall_clock64() - t0;
      const bool aborted =
          __scoped_atomic_load_n(
              status_ptr + static_cast<int64_t>(StatusField::abort) *
                               status_stride_field,
              __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM) != 0;
      if (__builtin_amdgcn_ballot_w64(aborted) == 0 && waited <= timeout_ticks)
        continue;
      if (!aborted) {
        uint32_t peer = 0, seen = flags[0], want = flags[0];
#pragma unroll
        for (int c = 0; c < WORLD; ++c)
#pragma unroll
          for (int i = 0; i < kRows; ++i)
#pragma unroll
            for (int k = 0; k < kPacks; ++k)
#pragma unroll
              for (int h = 0; h < kLines; ++h)
                if (waits[i][k] && (got[c][i][k][h][1] != flags[i] ||
                                    got[c][i][k][h][3] != flags[i])) {
                  peer = c;
                  seen = got[c][i][k][h][1];
                  want = flags[i];
                }
        give_up(status_ptr, status_stride_field, status_size_field,
                Where::start, rank, peer, seen, want, waited);
      }
      __builtin_amdgcn_endpgm();
    }

#pragma unroll
    for (int c = 0; c < WORLD; ++c)
#pragma unroll
      for (int i = 0; i < kRows; ++i)
#pragma unroll
        for (int k = 0; k < kPacks; ++k) {
          if constexpr (QUANTIZE_INT4 && kRaw) {
            output[c][i][k][0] = got[c][i][k][0][0];
            output[c][i][k][1] = got[c][i][k][0][2];
          } else if constexpr (QUANTIZE_INT4) {
            uint32_t line[4];
            // Copy the payload out whole. Reading it as vector elements decoded
            // the wrong word.
            __builtin_memcpy(line, &got[c][i][k][0], sizeof(line));
            const uint32_t encoded[2] = {line[0], line[2]};
            dequantize(encoded, output[c][i][k]);
          } else {
            uint32_t words[2 * kLines];
#pragma unroll
            for (int h = 0; h < kLines; ++h) {
              words[2 * h] = got[c][i][k][h][0];
              words[2 * h + 1] = got[c][i][k][h][2];
            }
            __builtin_memcpy(output[c][i][k], words, sizeof(V));
          }
        }
  } else if constexpr (LAYOUT != Layout::tile && !kGroup) {
    static_assert(!QUANTIZE_INT4 && !kSlots && kRows == 1,
                  "the flat walk reads one exact copy a row of threads");

    constexpr int kLanes = THREADS_PER_BLOCK / WORLD;
    static_assert(kLanes * WORLD == THREADS_PER_BLOCK,
                  "a block is a group of threads a rank");
    const U* const from =
        reinterpret_cast<const U*>(input) + int64_t{offs_m} * input_stride_m;
    const int64_t first = int64_t{input_slot} * input_stride_slot;
    if constexpr (kRun) {
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        const int last = clip_n / kPackElems - 1;
        const V* run =
            reinterpret_cast<const V*>(from + first * input_stride_n);
        const int n = offs_n / kPackElems + tid % kLanes + k * kLanes;
        const int col = n < clip_n / kPackElems ? n : last;
        uint32_t offset = static_cast<uint32_t>(col) * uint32_t{sizeof(V)};
        asm volatile("" : "+v"(offset));
        V v;
        {
          using W = std::conditional_t<
              sizeof(V) == 16, u32x4,
              std::conditional_t<
                  sizeof(V) == 8, u32x2,
                  std::conditional_t<sizeof(V) == 4, uint32_t, uint16_t>>>;
          const W raw = *(const __attribute__((address_space(1)))
                          W*)(reinterpret_cast<const char*>(run) + offset);
          __builtin_memcpy(&v, &raw, sizeof(V));
        }
        __builtin_memcpy(output[0][k], &v, sizeof(v));
      }
    } else {
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        const int n = offs_n / kPackElems + tid % kLanes + k * kLanes;
        const int col = n < clip_n / kPackElems ? n : clip_n / kPackElems - 1;
        V v;
        {
          using W = std::conditional_t<
              sizeof(V) == 16, u32x4,
              std::conditional_t<
                  sizeof(V) == 8, u32x2,
                  std::conditional_t<sizeof(V) == 4, uint32_t, uint16_t>>>;
          const W raw =
              *(const __attribute__((address_space(1))) W*)(element_at(
                  from, input_stride_m, input_stride_n,
                  first + int64_t{col} * kPackElems, input_size_n));
          __builtin_memcpy(&v, &raw, sizeof(V));
        }
        __builtin_memcpy(output[0][k], &v, sizeof(v));
      }
    }

    asm volatile("" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
  } else if constexpr (kGroup) {
    static_assert(kRows == 1 && kPacks == kInt4GroupPacks,
                  "a group a thread here: several are load's, recv's, store's "
                  "and send's loop");

    const int g = offs_n / kInt4Group + tid % kThreadsN;

    const int at = TAIL ? min(g, static_cast<int>(int4_groups(clip_n)) - 1)
                        : min(g, clip_n / kInt4Group - 1);
    if constexpr (!QUANTIZE_INT4) {
      static_assert(!kSlots, "an exact group is one copy");
      const U* const from =
          reinterpret_cast<const U*>(input) + int64_t{offs_m} * input_stride_m;
      const int64_t first = int64_t{input_slot} * input_stride_slot;
      if constexpr (TAIL) {
        [[maybe_unused]] const V* group = nullptr;
        if constexpr (kRun) {
          uint32_t offset =
              static_cast<uint32_t>(at) * (kInt4Group * sizeof(U));
          asm volatile("" : "+v"(offset));
          group = reinterpret_cast<const V*>(
              reinterpret_cast<const char*>(from + first * input_stride_n) +
              offset);
        }
        const auto pack_at = [&](int kk) -> const V* {
          if constexpr (kRun)
            return group + kk;
          else
            return reinterpret_cast<const V*>(element_at(
                from, input_stride_m, input_stride_n,
                first + int64_t{at} * kInt4Group + int64_t{kk} * kPackElems,
                input_size_n));
        };

        static_assert(sizeof(V) % sizeof(u32x4) == 0,
                      "a pack is whole 16-byte loads");
        const int last = clip_n / kPackElems - 1 - at * kInt4GroupPacks;
#pragma unroll
        for (int k = 0; k < kInt4GroupPacks; ++k) {
          const int kk = k < last ? k : last;
          if constexpr (sizeof(V) == sizeof(u32x4)) {
            const u32x4 raw = *(const global_u32x4*)(pack_at(kk));
            __builtin_memcpy(output[0][k], &raw, sizeof(raw));
          } else {
#pragma unroll
            for (int h = 0; h < static_cast<int>(sizeof(V) / sizeof(u32x4));
                 ++h) {
              const u32x4 raw =
                  *(const global_u32x4*)(reinterpret_cast<const char*>(
                                             pack_at(kk)) +
                                         h * sizeof(u32x4));
              __builtin_memcpy(
                  reinterpret_cast<char*>(output[0][k]) + h * sizeof(u32x4),
                  &raw, sizeof(raw));
            }
          }
        }
#pragma unroll
        for (int k = 0; k < kInt4GroupPacks; ++k)
          if (k > last)
#pragma unroll
            for (int j = 0; j < kPackElems; ++j)
              output[0][k][j] = static_cast<U>(0.0f);
      } else {
        const V* group;
        if constexpr (kRun) {
          uint32_t offset =
              static_cast<uint32_t>(at) * (kInt4Group * sizeof(U));
          asm volatile("" : "+v"(offset));
          group = reinterpret_cast<const V*>(
              reinterpret_cast<const char*>(from + first * input_stride_n) +
              offset);
        } else
          group = reinterpret_cast<const V*>(
              element_at(from, input_stride_m, input_stride_n,
                         first + int64_t{at} * kInt4Group, input_size_n));

        static_assert(sizeof(V) % sizeof(u32x4) == 0,
                      "a pack is whole 16-byte loads");
#pragma unroll
        for (int k = 0; k < kInt4GroupPacks; ++k) {
          if constexpr (sizeof(V) == sizeof(u32x4)) {
            const u32x4 raw = *(const global_u32x4*)(group + k);
            __builtin_memcpy(output[0][k], &raw, sizeof(raw));
          } else {
#pragma unroll
            for (int h = 0; h < static_cast<int>(sizeof(V) / sizeof(u32x4));
                 ++h) {
              const u32x4 raw =
                  *(const global_u32x4*)(reinterpret_cast<const char*>(group +
                                                                       k) +
                                         h * sizeof(u32x4));
              __builtin_memcpy(
                  reinterpret_cast<char*>(output[0][k]) + h * sizeof(u32x4),
                  &raw, sizeof(raw));
            }
          }
        }
      }
    } else {
      static_assert(kRaw && kRun,
                    "an INT4 group is read raw from a dense payload");
      const int64_t scales_offset =
          int4_scales_offset(input_size_m, input_size_n);
      const uint8_t* values[kCopies];
      const __fp16* scales[kCopies];
#pragma unroll
      for (int c = 0; c < kCopies; ++c) {
        const int64_t slot =
            int64_t{kSlots ? (input_slot + c) % WORLD : input_slot} *
            input_stride_slot;
        values[c] = reinterpret_cast<const uint8_t*>(input) + slot / 2;
        scales[c] =
            reinterpret_cast<const __fp16*>(
                reinterpret_cast<const uint8_t*>(input) + scales_offset) +
            slot / kInt4Group;
      }
      uint32_t value_at = static_cast<uint32_t>(at) * 16;
      uint32_t scale_at = static_cast<uint32_t>(at) * kInt4ScaleBytes;
      asm volatile("" : "+v"(value_at));
      asm volatile("" : "+v"(scale_at));
#pragma unroll
      for (int c = 0; c < kCopies; ++c) {
        const u32x4 words = *(const global_u32x4*)(values[c] + value_at);

        const uint32_t scale = __builtin_bit_cast(
            uint16_t,
            *(const global_f16*)(reinterpret_cast<const char*>(scales[c]) +
                                 scale_at));
        auto& out = [&]() -> U(&)[kPacks][kPackElems] {
          if constexpr (kSlots)
            return output[c][0];
          else
            return output[0];
        }();

        const int last =
            clip_n / (kInt4Group / kInt4GroupPacks) - 1 - at * kInt4GroupPacks;
#pragma unroll
        for (int k = 0; k < kInt4GroupPacks; ++k) {
          out[k][0] = TAIL && k > last ? kInt4ZeroWord : words[k];
          out[k][1] = scale;
        }
      }

      asm volatile("" ::: "memory");
      __builtin_amdgcn_sched_barrier(0);
    }
  } else if constexpr (!QUANTIZE_INT4 && !kSlots) {
    const V* at = reinterpret_cast<const V*>(base(0));
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      constexpr bool kLaneRow = !(kRows == 1 && kThreadsM == 1);
      uint32_t lane_offset[kPacks];
      if constexpr (kLaneRow) {
        if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
            int64_t{row_of(i)} * input_stride_m * int64_t{sizeof(U)} +
                    input_size_n * int64_t{sizeof(U)} >
                (int64_t{1} << 32))
          __builtin_trap();
        const uint32_t row_bytes =
            static_cast<uint32_t>(row_of(i)) *
            static_cast<uint32_t>(input_stride_m * int64_t{sizeof(U)});
#pragma unroll
        for (int k = 0; k < kPacks; ++k) {
          lane_offset[k] = row_bytes + static_cast<uint32_t>(pack_of(k)) *
                                           uint32_t{sizeof(V)};
          asm("" : "+v"(lane_offset[k]));
        }
      }
      if (skips(i)) continue;
      const V* row =
          kLaneRow ? at
                   : at + int64_t{row_of(i)} * (input_stride_m / kPackElems);
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        uint32_t offset = 0;
        if constexpr (kLaneRow)
          offset = lane_offset[k];
        else {
          offset = static_cast<uint32_t>(pack_of(k)) * uint32_t{sizeof(V)};
          asm volatile("" : "+v"(offset));
        }
        const V* p = reinterpret_cast<const V*>(
            reinterpret_cast<const char*>(row) + offset);
        V v;
        if constexpr (sizeof(V) == 16) {
          const u32x4 raw = *(const global_u32x4*)(p);
          __builtin_memcpy(&v, &raw, 16);
        } else {
          v = *p;
        }
        __builtin_memcpy(output[i][k], &v, sizeof(v));
      }
    }

    asm volatile("" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
  } else if constexpr (!QUANTIZE_INT4) {
    const U* at_copy[kCopies];
#pragma unroll
    for (int c = 0; c < kCopies; ++c) at_copy[c] = base(c);
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      if (skips(i)) continue;
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        int64_t at =
            int64_t{row_of(i)} * (input_stride_m / kPackElems) + pack_of(k);
        asm volatile("" : "+v"(at));
#pragma unroll
        for (int c = 0; c < kCopies; ++c) {
          const V* p = reinterpret_cast<const V*>(at_copy[c]) + at;
          V v;
          if constexpr (sizeof(V) == 16) {
            const u32x4 raw = *(const global_u32x4*)(p);
            __builtin_memcpy(&v, &raw, 16);
          } else {
            v = *p;
          }
          __builtin_memcpy(output[c][i][k], &v, sizeof(v));
        }
      }
    }
    asm volatile("" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
  } else {
    static_assert(kThreadsN % kInt4GroupPacks == 0,
                  "a group is 4 lanes' packs");

    const int64_t values_stride_m = input_size_n / 2;
    const int64_t scales_stride_m =
        TAIL ? int4_groups(input_size_n) : input_size_n / kInt4Group;
    const int64_t scales_at = int4_scales_offset(input_size_m, input_size_n);

    const uint8_t* values[kCopies];
    const char* scales[kCopies];
#pragma unroll
    for (int c = 0; c < kCopies; ++c) values[c] = base(c);
#pragma unroll
    for (int c = 0; c < kCopies; ++c)
      scales[c] = reinterpret_cast<const char*>(values[c] + scales_at);
    uint32_t words[kCopies][kRows][kPacks];
    uint32_t group_scales[kCopies][kRows][kPacks];
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      if (skips(i)) continue;
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        const int64_t row = row_of(i);
        const int pack = pack_of(k);
        uint32_t value_at = static_cast<uint32_t>(row * values_stride_m) +
                            static_cast<uint32_t>(pack) * kInt4WordBytes;
        uint32_t scale_at = static_cast<uint32_t>(
            (row * scales_stride_m + pack / kInt4GroupPacks) * kInt4ScaleBytes);
        asm volatile("" : "+v"(value_at));
        asm volatile("" : "+v"(scale_at));
#pragma unroll
        for (int c = 0; c < kCopies; ++c) {
          words[c][i][k] = *(const global_u32*)(values[c] + value_at);
          group_scales[c][i][k] = __builtin_bit_cast(
              uint16_t, *(const global_f16*)(scales[c] + scale_at));
        }
      }
    }
    asm volatile("" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int c = 0; c < kCopies; ++c)
#pragma unroll
      for (int i = 0; i < kRows; ++i)
#pragma unroll
        for (int k = 0; k < kPacks; ++k) {
          auto& out = [&]() -> U(&)[kPackElems] {
            if constexpr (kSlots)
              return output[c][i][k];
            else
              return output[i][k];
          }();
          const uint32_t encoded[2] = {words[c][i][k], group_scales[c][i][k]};
          if constexpr (kRaw) {
            out[0] = encoded[0];
            out[1] = encoded[1];
          } else {
            dequantize(encoded, out);
          }
        }
  }
}

template <int WORLD, int TILE_M, int THREADS_PER_BLOCK, int WAVE_SIZE,
          bool QUANTIZE_INT4, Layout LAYOUT, bool TAIL, typename INPUT>
DINLINE void write_buffer(const INPUT& input, void* output, int output_slot,
                          int64_t output_stride_slot, int64_t output_stride_m,
                          int64_t output_stride_n, int output_size_slot,
                          int output_size_m, int output_size_n, int clip_m,
                          int clip_n, int offs_m, int offs_n, int row_step,
                          const uint32_t* flags = nullptr) {
#if defined(__AMDGCN_WAVEFRONT_SIZE)
  static_assert(__AMDGCN_WAVEFRONT_SIZE == WAVE_SIZE,
                "the compiler's wave is not the one handed");
#endif
  constexpr int kRows = std::extent_v<INPUT, 0>;
  constexpr int kPacks = std::extent_v<INPUT, 1>;
  constexpr int kPackElems = std::extent_v<INPUT, 2>;
  using U = std::remove_all_extents_t<INPUT>;
  constexpr bool kRaw = std::is_same_v<U, uint32_t>;
  static_assert(!kRaw || (QUANTIZE_INT4 && kPackElems == 2),
                "uint32_t registers are an INT4 payload's words and scales");

  constexpr int kThreadsM = TILE_M / kRows;
  constexpr int kThreadsN = THREADS_PER_BLOCK / kThreadsM;
  static_assert(kThreadsM * kRows == TILE_M, "a tile is whole rows a thread");
  static_assert(kThreadsM * kThreadsN == THREADS_PER_BLOCK,
                "the layout is the block");
  using V = vec<U, kPackElems>;
  using E = std::conditional_t<!QUANTIZE_INT4, U, uint8_t>;
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
      (output_slot < 0 || output_slot >= output_size_slot))
    __builtin_trap();
  [[maybe_unused]] E* const at_copy =
      reinterpret_cast<E*>(output) + output_slot * output_stride_slot;
  const int tid = static_cast<int>(threadIdx.x);
  const auto row_at = [&](int i) {
    return offs_m +
           (kThreadsM == 1 ? i : tid / kThreadsN + i * kThreadsM) * row_step;
  };
  const auto col_at = [&](int k) {
    return offs_n / kPackElems + tid % kThreadsN + k * kThreadsN;
  };

  [[maybe_unused]] const auto element_at =
      [](auto* ptr, int64_t stride_m, int64_t stride_n, int64_t e, int n) {
        if (stride_m == int64_t{n} * stride_n) return ptr + e * stride_n;
        const int64_t row =
            e < (int64_t{1} << 31)
                ? int64_t{static_cast<uint32_t>(e) / static_cast<uint32_t>(n)}
                : e / n;
        return ptr + row * stride_m + (e - row * n) * stride_n;
      };
  constexpr bool kGroup =
      LAYOUT == Layout::group_run || LAYOUT == Layout::group_flat;
  constexpr bool kRun = LAYOUT == Layout::run || LAYOUT == Layout::group_run;
  if constexpr (LAYOUT == Layout::line) {
    constexpr int kLines = QUANTIZE_INT4 || sizeof(V) <= kLinePayloadBytes
                               ? 1
                               : sizeof(V) / kLinePayloadBytes;
    static_assert(
        (QUANTIZE_INT4 ? kRaw || kPackElems == kInt4Group / kInt4GroupPacks
                       : sizeof(V) <= kLinePayloadBytes ||
                             sizeof(V) % kLinePayloadBytes == 0) &&
            kLines >= 1 && kLines <= 2 && sizeof(u32x4) == kLineBytes,
        "a pack is one or two lines, an INT4 pack a word of 8 values");
    constexpr int kLineScale = kLineBytes / kLinePayloadBytes;
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG && flags == nullptr) __builtin_trap();

    [[maybe_unused]] uint32_t payload[kRows][kPacks][2];
    if constexpr (QUANTIZE_INT4) {
#pragma unroll
      for (int i = 0; i < kRows; ++i)
#pragma unroll
        for (int k = 0; k < kPacks; ++k) {
          if constexpr (kRaw) {
            payload[i][k][0] = input[i][k][0];
            payload[i][k][1] = input[i][k][1];
          } else {
            quantize(reinterpret_cast<const U(&)[1][kPackElems]>(input[i][k]),
                     reinterpret_cast<uint32_t (&)[1][2]>(payload[i][k]));
          }
        }
    }
    const int bank_rows = output_size_m / 2;

    constexpr int64_t kElemBytes = kRaw ? 2 : int64_t{sizeof(U)};
    char* const slot =
        reinterpret_cast<char*>(output) +
        int64_t{output_slot} * output_stride_slot * kElemBytes * kLineScale;
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      if (!(kRows == 1 && kThreadsM == 1) && row_at(i) >= clip_m) continue;

      const int row = row_at(i) + static_cast<int>(flags[i] & 1u) * bank_rows;
      if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
          (row >= output_size_m || flags[i] == 0))
        __builtin_trap();
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        constexpr int kElems = kRaw ? kInt4Group / kInt4GroupPacks : kPackElems;

        if constexpr (QUANTIZE_INT4) {
          if (col_at(k) >= clip_n / kElems) continue;
        }
        const int col = min(col_at(k), clip_n / kElems - 1);
        uint32_t words[2 * kLines] = {};
        if constexpr (QUANTIZE_INT4) {
          words[0] = payload[i][k][0];
          words[1] = payload[i][k][1];
        } else {
          __builtin_memcpy(words, input[i][k], sizeof(V));
        }

        const uint32_t half =
            static_cast<uint32_t>(output_stride_m * kElemBytes);
        uint32_t at = static_cast<uint32_t>(row) * half * kLineScale +
                      static_cast<uint32_t>(col) * uint32_t{kLineBytes};
        asm volatile("" : "+v"(at));
#pragma unroll
        for (int h = 0; h < kLines; ++h)
          __builtin_nontemporal_store(
              // The flag is in both 8-byte halves, so a reader that sees both
              // has the whole line.
              u32x4{words[2 * h], flags[i], words[2 * h + 1], flags[i]},
              (global_u32x4*)(slot + at + h * half));
      }
    }
  } else if constexpr (LAYOUT != Layout::tile && !kGroup) {
    static_assert(!QUANTIZE_INT4 && kRows == 1,
                  "the flat walk writes one exact copy a row of threads");

    constexpr int kLanes = THREADS_PER_BLOCK / WORLD;
    static_assert(kLanes * WORLD == THREADS_PER_BLOCK,
                  "a block is a group of threads a rank");
    U* const to =
        reinterpret_cast<U*>(output) + int64_t{offs_m} * output_stride_m;
    const int64_t first = int64_t{output_slot} * output_stride_slot;
    U* const run = to + first * output_stride_n;
#pragma unroll
    for (int k = 0; k < kPacks; ++k) {
      const int c = offs_n / kPackElems + tid % kLanes + k * kLanes;
      if (c >= clip_n / kPackElems) continue;
      V* p;
      if constexpr (kRun) {
        uint32_t offset = static_cast<uint32_t>(c) * uint32_t{sizeof(V)};
        asm volatile("" : "+v"(offset));
        p = reinterpret_cast<V*>(reinterpret_cast<char*>(run) + offset);
      } else {
        p = reinterpret_cast<V*>(
            element_at(to, output_stride_m, output_stride_n,
                       first + int64_t{c} * kPackElems, output_size_n));
      }
      V v;
      __builtin_memcpy(&v, input[0][k], sizeof(v));
      {
        using W = std::conditional_t<
            sizeof(V) == 16, u32x4,
            std::conditional_t<
                sizeof(V) == 8, u32x2,
                std::conditional_t<sizeof(V) == 4, uint32_t, uint16_t>>>;
        W raw;
        __builtin_memcpy(&raw, &v, sizeof(V));
        *(__attribute__((address_space(1))) W*)(p) = raw;
      }
    }
  } else if constexpr (kGroup) {
    static_assert(kRows == 1 && kPacks == kInt4GroupPacks,
                  "a group a thread here: several are load's, recv's, store's "
                  "and send's loop");
    const int g = offs_n / kInt4Group + tid % kThreadsN;
    if constexpr (!QUANTIZE_INT4) {
      if constexpr (TAIL) {
        if (g >= int4_groups(clip_n)) return;
        U* const to =
            reinterpret_cast<U*>(output) + int64_t{offs_m} * output_stride_m;
        const int64_t first = int64_t{output_slot} * output_stride_slot;

        [[maybe_unused]] V* group = nullptr;
        if constexpr (kRun) {
          uint32_t offset = static_cast<uint32_t>(g) * (kInt4Group * sizeof(U));
          asm volatile("" : "+v"(offset));
          group = reinterpret_cast<V*>(
              reinterpret_cast<char*>(to + first * output_stride_n) + offset);
        }
        const auto pack_at = [&](int k) -> V* {
          if constexpr (kRun)
            return group + k;
          else
            return reinterpret_cast<V*>(element_at(
                to, output_stride_m, output_stride_n,
                first + int64_t{g} * kInt4Group + int64_t{k} * kPackElems,
                output_size_n));
        };
#pragma unroll
        for (int k = 0; k < kInt4GroupPacks; ++k) {
          if (g * kInt4GroupPacks + k >= clip_n / kPackElems) continue;
          u32x4 raw;
          __builtin_memcpy(&raw, input[0][k], sizeof(raw));
          *(global_u32x4*)(pack_at(k)) = raw;
        }
      } else {
        if (g >= clip_n / kInt4Group) return;
        U* const to =
            reinterpret_cast<U*>(output) + int64_t{offs_m} * output_stride_m;
        const int64_t first = int64_t{output_slot} * output_stride_slot;
        V* group;
        if constexpr (kRun) {
          uint32_t offset = static_cast<uint32_t>(g) * (kInt4Group * sizeof(U));
          asm volatile("" : "+v"(offset));
          group = reinterpret_cast<V*>(
              reinterpret_cast<char*>(to + first * output_stride_n) + offset);
        } else
          group = reinterpret_cast<V*>(
              element_at(to, output_stride_m, output_stride_n,
                         first + int64_t{g} * kInt4Group, output_size_n));
#pragma unroll
        for (int k = 0; k < kInt4GroupPacks; ++k) {
          u32x4 raw;
          __builtin_memcpy(&raw, input[0][k], sizeof(raw));
          *(global_u32x4*)(group + k) = raw;
        }
      }
    } else {
      static_assert(kRun, "an INT4 payload is dense");
      uint32_t encoded[kInt4GroupPacks][2];
      if constexpr (kRaw) {
#pragma unroll
        for (int k = 0; k < kInt4GroupPacks; ++k) {
          encoded[k][0] = input[0][k][0];
          encoded[k][1] = input[0][k][1];
        }
      } else {
        quantize(input[0], encoded);
      }
      if (g < (TAIL ? static_cast<int>(int4_groups(clip_n))
                    : clip_n / kInt4Group)) {
        const int64_t slot = int64_t{output_slot} * output_stride_slot;
        uint8_t* const values = reinterpret_cast<uint8_t*>(output) + slot / 2;
        __fp16* const scales =
            reinterpret_cast<__fp16*>(
                reinterpret_cast<uint8_t*>(output) +
                int4_scales_offset(output_size_m, output_size_n)) +
            slot / kInt4Group;
        uint32_t value_at = static_cast<uint32_t>(g) * 16;
        uint32_t scale_at = static_cast<uint32_t>(g) * kInt4ScaleBytes;
        asm volatile("" : "+v"(value_at));
        asm volatile("" : "+v"(scale_at));

        const int words_in =
            clip_n / (kInt4Group / kInt4GroupPacks) - g * kInt4GroupPacks;
        if (!TAIL || words_in >= kInt4GroupPacks) {
          *(global_u32x4*)(values + value_at) =
              u32x4{encoded[0][0], encoded[1][0], encoded[2][0], encoded[3][0]};
        } else {
#pragma unroll
          for (int k = 0; k < kInt4GroupPacks; ++k)
            if (k < words_in)
              *(global_u32*)(values + value_at + k * kInt4WordBytes) =
                  encoded[k][0];
        }
        *(global_f16*)(reinterpret_cast<char*>(scales) + scale_at) =
            __builtin_bit_cast(__fp16, static_cast<uint16_t>(encoded[0][1]));
      }
    }
  } else if constexpr (!QUANTIZE_INT4) {
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      if (row_at(i) >= clip_m) continue;
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        if (col_at(k) >= clip_n / kPackElems) continue;
        V v;
        __builtin_memcpy(&v, input[i][k], sizeof(v));

        uint32_t offset =
            static_cast<uint32_t>(col_at(k)) * uint32_t{sizeof(V)};
        asm volatile("" : "+v"(offset));
        V* row = reinterpret_cast<V*>(at_copy) +
                 int64_t{row_at(i)} * (output_stride_m / kPackElems);
        V* p = reinterpret_cast<V*>(reinterpret_cast<char*>(row) + offset);
        if constexpr (sizeof(V) == 16) {
          u32x4 raw;
          __builtin_memcpy(&raw, &v, 16);
          *(global_u32x4*)(p) = raw;
        } else {
          *p = v;
        }
      }
    }
  } else {
    static_assert(kThreadsN % kInt4GroupPacks == 0,
                  "a group is 4 lanes' packs");
    constexpr int kInt4PackElems = kInt4Group / kInt4GroupPacks;
    uint32_t encoded[kRows][kPacks][2];
#pragma unroll
    for (int i = 0; i < kRows; ++i)
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        if constexpr (kRaw) {
          encoded[i][k][0] = input[i][k][0];
          encoded[i][k][1] = input[i][k][1];
        } else {
          quantize(reinterpret_cast<const U(&)[1][kPackElems]>(input[i][k]),
                   reinterpret_cast<uint32_t (&)[1][2]>(encoded[i][k]));
        }
      }
    const int64_t values_stride_m = output_size_n / 2;
    const int64_t scales_stride_m =
        TAIL ? int4_groups(output_size_n) : output_size_n / kInt4Group;
    const int64_t scales_at = int4_scales_offset(output_size_m, output_size_n);
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      if (row_at(i) >= clip_m) continue;
      const int64_t row = row_at(i);
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        const int pack =
            offs_n / kInt4PackElems + tid % kThreadsN + k * kThreadsN;
        if (pack >= clip_n / kInt4PackElems) continue;
        uint32_t value_at = static_cast<uint32_t>(row * values_stride_m) +
                            static_cast<uint32_t>(pack) * kInt4WordBytes;
        uint32_t scale_at = static_cast<uint32_t>(
            (row * scales_stride_m + pack / kInt4GroupPacks) * kInt4ScaleBytes);
        asm volatile("" : "+v"(value_at));
        asm volatile("" : "+v"(scale_at));
        *(global_u32*)(at_copy + value_at) = encoded[i][k][0];
        if (pack % kInt4GroupPacks == 0)
          *(global_f16*)(at_copy + scales_at + scale_at) = __builtin_bit_cast(
              __fp16, static_cast<uint16_t>(encoded[i][k][1]));
      }
    }
  }
}

}  // namespace internal

namespace ReduceAxis {
constexpr int all = 0;
constexpr int m = 1;
}  // namespace ReduceAxis

template <int TILE_M, int THREADS_PER_BLOCK, int WAVE_SIZE, int AXIS,
          typename T, typename U, int K, int P>
DINLINE void reduce(const T (&input)[K][P], U (&output)[K][P]) {
  static_assert(AXIS == ReduceAxis::all || AXIS == ReduceAxis::m,
                "AXIS is all or m");

#if defined(__AMDGCN_WAVEFRONT_SIZE)
  static_assert(__AMDGCN_WAVEFRONT_SIZE == WAVE_SIZE,
                "the compiler's wave is not the one handed");
#endif

  constexpr int kThreadsM = TILE_M;
  constexpr int kThreadsN = THREADS_PER_BLOCK / TILE_M;
  static_assert(kThreadsM * kThreadsN == THREADS_PER_BLOCK,
                "the block is TILE_M whole rows of threads");
  if constexpr (AXIS == ReduceAxis::all) {
    static_assert(TILE_M == 1, "AXIS all is one row of the block's threads");
    constexpr int kValues = K * P;
    constexpr int kWaves = (kThreadsN + WAVE_SIZE - 1) / WAVE_SIZE;
    __shared__ float partial[kValues][kWaves];
    const int lane = threadIdx.x % WAVE_SIZE;
    const int wave = threadIdx.x / WAVE_SIZE;
    float v[kValues];
#pragma unroll
    for (int n = 0; n < kValues; ++n)
      v[n] = static_cast<float>(input[n / P][n % P]);

    static_assert(WAVE_SIZE == 64, "the DPP sequence is for 64-lane waves");
    const auto bits = [](float x) { return __builtin_bit_cast(int, x); };
    const auto real = [](int x) { return __builtin_bit_cast(float, x); };
    constexpr int kIdentity = 0;
#pragma unroll
    for (int n = 0; n < kValues; ++n) {
      float x = v[n];

      x = real(__builtin_amdgcn_mov_dpp(bits(x), 0xb1, 0xf, 0xf, false)) + x;
      x = real(__builtin_amdgcn_mov_dpp(bits(x), 0x4e, 0xf, 0xf, false)) + x;
      x = real(__builtin_amdgcn_mov_dpp(bits(x), 0x141, 0xf, 0xf, false)) + x;
      x = real(__builtin_amdgcn_mov_dpp(bits(x), 0x140, 0xf, 0xf, false)) + x;

      x = real(__builtin_amdgcn_update_dpp(kIdentity, bits(x), 0x142, 0xa, 0xf,
                                           false)) +
          x;
      x = real(__builtin_amdgcn_update_dpp(kIdentity, bits(x), 0x143, 0xc, 0xf,
                                           false)) +
          x;
      v[n] = real(__builtin_amdgcn_readlane(bits(x), 63));
    }

    if (lane == 0) {
#pragma unroll
      for (int n = 0; n < kValues; ++n) partial[n][wave] = v[n];
    }
    __syncthreads();
#pragma unroll
    for (int n = 0; n < kValues; ++n) {
      float x = partial[n][0];
#pragma unroll
      for (int w = 1; w < kWaves; ++w) x = x + partial[n][w];
      output[n / P][n % P] = static_cast<U>(x);
    }
    __syncthreads();
  } else {
    using V = vec<T, P>;
    __shared__ V part[kThreadsM][K][kThreadsN];
    const int mine = static_cast<int>(threadIdx.x) % kThreadsN;
#pragma unroll
    for (int k = 0; k < K; ++k)
      __builtin_memcpy(
          &part[static_cast<int>(threadIdx.x) / kThreadsN][k][mine], input[k],
          sizeof(V));
    __syncthreads();

    if (kThreadsM == 1 || static_cast<int>(threadIdx.x) < kThreadsN) {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        float x[P];
        V p = part[0][k][mine];
#pragma unroll
        for (int j = 0; j < P; ++j)
          x[j] = static_cast<float>(reinterpret_cast<const T*>(&p)[j]);
#pragma unroll
        for (int r = 1; r < kThreadsM; ++r) {
          p = part[r][k][mine];
#pragma unroll
          for (int j = 0; j < P; ++j)
            x[j] = x[j] + static_cast<float>(reinterpret_cast<const T*>(&p)[j]);
        }
#pragma unroll
        for (int j = 0; j < P; ++j) output[k][j] = static_cast<U>(x[j]);
      }
    }
    __syncthreads();
  }
}

enum class Until { launched, visible, read, started };

template <int WORLD>
// A barrier across ranks through flags in every peer's signal buffer.
DINLINE void barrier(uint32_t* const* signals_ptr_array,
                     int signals_ptr_array_size, int64_t signals_stride_parity,
                     int64_t signals_stride_block,
                     int64_t signals_stride_writer, int signals_size_parity,
                     int signals_size_block, int signals_size_writer,
                     uint32_t* status_ptr, int64_t status_stride_field,
                     int status_size_field, int rank, uint64_t timeout_ticks,
                     Until until, uint32_t parity) {
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG && parity > 1) __builtin_trap();
  const bool ordered = until == Until::launched || until == Until::visible;
  const bool first = until == Until::launched || until == Until::started;

  if (HIP_CUSTOM_COMMUNICATOR_DEBUG) {
    uint32_t h = static_cast<uint32_t>(rank) * 73856093u ^
                 blockIdx.x * 19349663u ^ parity * 83492791u;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    for (uint32_t n = (h ^ (h >> 15)) % 32; n > 0; --n)
      __builtin_amdgcn_s_sleep(127);
  }

  const auto flag = [&](uint32_t* owner_signals, auto owner, unsigned block,
                        auto writer) {
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
        (static_cast<int64_t>(owner) < 0 ||
         static_cast<int64_t>(owner) >= WORLD ||
         static_cast<int64_t>(parity) >= signals_size_parity ||
         static_cast<int64_t>(block) >= signals_size_block ||
         static_cast<int64_t>(writer) < 0 ||
         static_cast<int64_t>(writer) >= signals_size_writer))
      __builtin_trap();
    return owner_signals + parity * signals_stride_parity +
           block * signals_stride_block + writer * signals_stride_writer;
  };

  uint32_t* theirs_ptr = nullptr;
  uint32_t* mine_ptr = nullptr;
  if (threadIdx.x < WORLD) {
    if (HIP_CUSTOM_COMMUNICATOR_DEBUG && WORLD > signals_ptr_array_size)
      __builtin_trap();

    const uint64_t entry = ((const __attribute__((address_space(1)))
                             uint64_t*)signals_ptr_array)[threadIdx.x];
    theirs_ptr =
        flag((uint32_t*)(__attribute__((address_space(1))) uint32_t*)entry,
             threadIdx.x, blockIdx.x, rank);
    mine_ptr = flag((uint32_t*)(__attribute__((address_space(1))) uint32_t*)
                        internal::ptr_array_at(signals_ptr_array,
                                               signals_ptr_array_size, rank),
                    rank, blockIdx.x, threadIdx.x);
  }

  if (ordered) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  if (threadIdx.x < WORLD) {
    if (!ordered) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

    __scoped_atomic_store_n(
        theirs_ptr, static_cast<uint32_t>(FlagState::arrived),
        ordered ? __ATOMIC_RELEASE : __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);

    const uint32_t want = static_cast<uint32_t>(FlagState::arrived);

    const auto word = [&](StatusField f) {
      if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
          (static_cast<int64_t>(f) < 0 ||
           static_cast<int64_t>(f) >= status_size_field))
        __builtin_trap();
      return status_ptr + static_cast<int64_t>(f) * status_stride_field;
    };
    const uint64_t t0 = wall_clock64();
    for (uint32_t n = 1; __scoped_atomic_load_n(mine_ptr, __ATOMIC_RELAXED,
                                                __MEMORY_SCOPE_SYSTEM) < want;
         ++n) {
      if ((n & 255u) != 0) continue;
      const uint64_t waited = wall_clock64() - t0;
      const bool aborted =
          __scoped_atomic_load_n(word(StatusField::abort), __ATOMIC_RELAXED,
                                 __MEMORY_SCOPE_SYSTEM) != 0;
      if (!aborted && waited <= timeout_ticks) continue;
      if (!aborted)
        internal::give_up(status_ptr, status_stride_field, status_size_field,
                          first ? Where::start : Where::peer_barrier, rank,
                          threadIdx.x,
                          __scoped_atomic_load_n(mine_ptr, __ATOMIC_RELAXED,
                                                 __MEMORY_SCOPE_SYSTEM),
                          want, waited);
      __builtin_amdgcn_endpgm();
    }

    if (ordered) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");

    __scoped_atomic_store_n(mine_ptr, static_cast<uint32_t>(FlagState::clear),
                            __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
  __syncthreads();
}

template <int LOWEST, int HIGHEST, typename U, int P>
DINLINE void dequantize(const uint32_t (&encoded)[2], U (&output)[P]) {
  const uint32_t word = encoded[0];
  const float scale = static_cast<float>(
      __builtin_bit_cast(__fp16, static_cast<uint16_t>(encoded[1])));
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
      ((encoded[1] >> 16) != 0 || !__builtin_isfinite(scale)))
    __builtin_trap();
  static_assert(P == 8, "a word is 8 values");
  static_assert(std::is_same_v<U, float> || sizeof(U) == 2,
                "fp32 terms or a 16-bit dtype");
  const float scale16 = scale * 0.0625f;

  const float bias = scale * -(1024.0f - LOWEST);
  const float bias16 = scale * -(64.0f - LOWEST);
  float term[P];
  uint32_t p0, p1, p2, p3;
  // 0x6400 is 1024 in fp16. OR-ing a nibble into it gives 1024 + q exactly.
  asm volatile(
      "v_and_or_b32 %[p0], %[w], %[m0], %[magic]\n\t"
      "v_and_or_b32 %[p1], %[w], %[m1], %[magic]\n\t"
      "v_lshrrev_b32 %[p2], 8, %[w]\n\t"
      "v_and_or_b32 %[p3], %[p2], %[m1], %[magic]\n\t"
      "v_and_or_b32 %[p2], %[p2], %[m0], %[magic]\n\t"
      "v_fma_mix_f32 %[t0], %[p0], %[s], %[b] op_sel_hi:[1,0,0]\n\t"
      "v_fma_mix_f32 %[t1], %[p0], %[s], %[b] op_sel:[1,0,0] "
      "op_sel_hi:[1,0,0]\n\t"
      "v_fma_mix_f32 %[t2], %[p1], %[s16], %[b16] op_sel_hi:[1,0,0]\n\t"
      "v_fma_mix_f32 %[t3], %[p1], %[s16], %[b16] op_sel:[1,0,0] "
      "op_sel_hi:[1,0,0]\n\t"
      "v_fma_mix_f32 %[t4], %[p2], %[s], %[b] op_sel_hi:[1,0,0]\n\t"
      "v_fma_mix_f32 %[t5], %[p2], %[s], %[b] op_sel:[1,0,0] "
      "op_sel_hi:[1,0,0]\n\t"
      "v_fma_mix_f32 %[t6], %[p3], %[s16], %[b16] op_sel_hi:[1,0,0]\n\t"
      "v_fma_mix_f32 %[t7], %[p3], %[s16], %[b16] op_sel:[1,0,0] "
      "op_sel_hi:[1,0,0]"
      : [p0] "=&v"(p0), [p1] "=&v"(p1), [p2] "=&v"(p2), [p3] "=&v"(p3),
        [t0] "=&v"(term[0]), [t1] "=&v"(term[1]), [t2] "=&v"(term[2]),
        [t3] "=&v"(term[3]), [t4] "=&v"(term[4]), [t5] "=&v"(term[5]),
        [t6] "=&v"(term[6]), [t7] "=&v"(term[7])
      : [w] "v"(word), [m0] "s"(0x000F000Fu), [m1] "s"(0x00F000F0u),
        [magic] "v"(0x64006400u), [s] "v"(scale), [b] "v"(bias),
        [s16] "v"(scale16), [b16] "v"(bias16));
  if constexpr (std::is_same_v<U, float>) {
#pragma unroll
    for (int j = 0; j < P; ++j) output[j] = term[j];
  } else {
    u32x4 packed;
#pragma unroll
    for (int w = 0; w < P / 2; ++w) {
      const float a = term[2 * w], b = term[2 * w + 1];
      if constexpr (kBf16<U>)
        packed[w] = __builtin_bit_cast(
            uint32_t, __builtin_convertvector(f32x2{a, b}, bf16x2));
      else
        packed[w] = static_cast<uint32_t>(
                        __builtin_bit_cast(uint16_t, static_cast<U>(a))) |
                    static_cast<uint32_t>(
                        __builtin_bit_cast(uint16_t, static_cast<U>(b)))
                        << 16;
    }
    __builtin_memcpy(output, &packed, sizeof(packed));
  }
}

template <typename DTYPE, int LOWEST, int HIGHEST, int SLOTS, int K, int P>
DINLINE void dequantize(const uint32_t (&encoded)[SLOTS][1][K][2],
                        float (&sum)[1][K][P]) {
  const auto& words = encoded;

  const auto scale_of = [&](int w, int k) {
    if constexpr (K == kInt4GroupPacks)
      return words[w][0][0][1];
    else
      return words[w][0][k / kInt4GroupPacks * kInt4GroupPacks][1];
  };
#pragma unroll
  for (int k = 0; k < K; ++k) {
#pragma unroll
    for (int j = 0; j < P; ++j) sum[0][k][j] = 0.0f;
    int w = 0;
    if constexpr (kBf16<DTYPE>) {
      const bf16x2 ones = __builtin_bit_cast(bf16x2, 0x3f803f80u);
#pragma unroll
      for (; w + 1 < SLOTS; w += 2) {
        float a[P], b[P];
        const uint32_t pack_a[2] = {words[w][0][k][0], scale_of(w, k)};
        dequantize<LOWEST, HIGHEST>(pack_a, a);
        const uint32_t pack_b[2] = {words[w + 1][0][k][0], scale_of(w + 1, k)};
        dequantize<LOWEST, HIGHEST>(pack_b, b);
#pragma unroll
        for (int j = 0; j < P; ++j)
          sum[0][k][j] = __builtin_amdgcn_fdot2_f32_bf16(
              __builtin_convertvector(f32x2{a[j], b[j]}, bf16x2), ones,
              sum[0][k][j], false);

#pragma unroll
        for (int j = 0; j < P; ++j) asm volatile("" ::"v"(sum[0][k][j]));
      }
    }
#pragma unroll
    for (; w < SLOTS; ++w) {
      float a[P];
      const uint32_t pack_a[2] = {words[w][0][k][0], scale_of(w, k)};
      dequantize<LOWEST, HIGHEST>(pack_a, a);
#pragma unroll
      for (int j = 0; j < P; ++j)
        sum[0][k][j] += static_cast<float>(static_cast<DTYPE>(a[j]));
#pragma unroll
      for (int j = 0; j < P; ++j) asm volatile("" ::"v"(sum[0][k][j]));
    }
  }
}

template <int LOWEST, int HIGHEST, int PACKS, typename T, int P>
DINLINE void quantize(const T (&input)[PACKS][P],
                      uint32_t (&encoded)[PACKS][2]) {
  constexpr int THREAD_PACKS = PACKS;
  uint32_t words[PACKS];
  float scale;
  static_assert(P * 4 == 32, "a pack is one word of 8 values");
  static_assert(THREAD_PACKS == 1 || THREAD_PACKS == kInt4GroupPacks,
                "a pack a thread (its quad's group) or a group a thread");

  float x[THREAD_PACKS][P];
  uint32_t key = 0;
#pragma unroll
  for (int k = 0; k < THREAD_PACKS; ++k)
#pragma unroll
    for (int j = 0; j < P; ++j) {
      x[k][j] = static_cast<float>(input[k][j]);
      const uint32_t bits = __builtin_bit_cast(uint32_t, x[k][j]);
      key = max(key, (bits << 1) | (bits >> 31));
    }
  if constexpr (THREAD_PACKS == 1) {
    key = max(key, static_cast<uint32_t>(__builtin_amdgcn_mov_dpp(
                       static_cast<int>(key), 0xb1, 0xf, 0xf, false)));
    key = max(key, static_cast<uint32_t>(__builtin_amdgcn_mov_dpp(
                       static_cast<int>(key), 0x4e, 0xf, 0xf, false)));
  }
  const float extreme = __builtin_bit_cast(float, (key >> 1) | (key << 31));

  // The group's signed extreme lands on the lowest level, so it is never
  // clipped.
  scale = static_cast<float>(static_cast<__fp16>(
      fminf(fmaxf(extreme * (1.0f / LOWEST), -65504.0f), 65504.0f)));
  const float inv = scale != 0.0f ? 1.0f / scale : 0.0f;

  const auto word = [](const float (&v)[P], float inverse) {
    constexpr float kLowest = 12582912.0f;
    constexpr float kBias = -LOWEST;
    constexpr float kLevels = HIGHEST - LOWEST;
    uint32_t q[P];
#pragma unroll
    for (int p = 0; p < P / 2; ++p) {
#pragma clang fp contract(off)
      const f32x2 y = f32x2{v[2 * p], v[2 * p + 1]} * f32x2{inverse, inverse} +
                      f32x2{kLowest + kBias, kLowest + kBias};
#pragma unroll
      for (int e = 0; e < 2; ++e)
        q[2 * p + e] = __builtin_bit_cast(
            uint32_t,
            __builtin_amdgcn_fmed3f(y[e], kLowest, kLowest + kLevels));
    }

    const auto nibbles = [](uint32_t high, uint32_t low) {
      uint32_t byte;

      asm("v_lshl_or_b32 %0, %1, 4, %2" : "=v"(byte) : "v"(high), "v"(low));
      return byte;
    };
    const uint32_t even_low = nibbles(q[2], q[0]),
                   even_high = nibbles(q[6], q[4]);
    const uint32_t odd_low = nibbles(q[3], q[1]),
                   odd_high = nibbles(q[7], q[5]);
    return __builtin_amdgcn_perm(
        __builtin_amdgcn_perm(odd_high, odd_low, 0x0c0c0400u),
        __builtin_amdgcn_perm(even_high, even_low, 0x0c0c0400u), 0x05040100u);
  };
#pragma unroll
  for (int k = 0; k < THREAD_PACKS; ++k) words[k] = word(x[k], inv);

  const uint32_t bits =
      __builtin_bit_cast(uint16_t, static_cast<__fp16>(scale));
#pragma unroll
  for (int k = 0; k < PACKS; ++k) {
    encoded[k][0] = words[k];
    encoded[k][1] = bits;
  }
}

template <typename T, int K, int P>
DINLINE void quantize(const T (&input)[1][K][P], uint32_t (&encoded)[1][K][2]) {
  static_assert(K % kInt4GroupPacks == 0, "a thread holds whole groups");
#pragma unroll
  for (int jg = 0; jg < K / kInt4GroupPacks; ++jg)
    quantize(reinterpret_cast<const T(&)[kInt4GroupPacks][P]>(
                 input[0][jg * kInt4GroupPacks]),
             reinterpret_cast<uint32_t (&)[kInt4GroupPacks][2]>(
                 encoded[0][jg * kInt4GroupPacks]));
}

template <int WORLD, int TILE_M, int THREADS_PER_BLOCK, int WAVE_SIZE,
          bool QUANTIZE_INT4, Layout LAYOUT, bool TAIL = false, typename PTR,
          typename OUTPUT>
DINLINE void load(PTR const* input_ptr_array, int input_ptr_array_size,
                  int input_rank, int input_slot, int64_t input_stride_slot,
                  int64_t input_stride_m, int64_t input_stride_n,
                  int input_size_slot, int input_size_m, int input_size_n,
                  int clip_m, int clip_n, OUTPUT& output, int offs_m,
                  int offs_n, int row_step, const uint32_t* flags = nullptr,
                  uint32_t* status_ptr = nullptr,
                  int64_t status_stride_field = 0, int status_size_field = 0,
                  uint64_t timeout_ticks = 0) {
  constexpr bool kSlots = std::rank_v<OUTPUT> == 4;
  constexpr int kPacks = std::extent_v < OUTPUT, kSlots ? 2 : 1 > ;
  if constexpr ((LAYOUT == Layout::group_run || LAYOUT == Layout::group_flat) &&
                kPacks != kInt4GroupPacks) {
    static_assert(kPacks % kInt4GroupPacks == 0, "a thread holds whole groups");
    constexpr int kCopies = kSlots ? std::extent_v<OUTPUT, 0> : 1;
    constexpr int kPackElems = std::extent_v < OUTPUT, kSlots ? 3 : 2 > ;
    using U = std::remove_all_extents_t<OUTPUT>;
    using Group =
        std::conditional_t<kSlots, U[kCopies][1][kInt4GroupPacks][kPackElems],
                           U[1][kInt4GroupPacks][kPackElems]>;
#pragma unroll
    for (int jg = 0; jg < kPacks / kInt4GroupPacks; ++jg) {
      Group group;
      load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4, LAYOUT,
           TAIL>(input_ptr_array, input_ptr_array_size, input_rank, input_slot,
                 input_stride_slot, input_stride_m, input_stride_n,
                 input_size_slot, input_size_m, input_size_n, clip_m, clip_n,
                 group, offs_m,
                 offs_n + jg * (THREADS_PER_BLOCK / TILE_M) * kInt4Group,
                 row_step, flags, status_ptr, status_stride_field,
                 status_size_field, timeout_ticks);
#pragma unroll
      for (int c = 0; c < kCopies; ++c)
#pragma unroll
        for (int k = 0; k < kInt4GroupPacks; ++k)
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) {
            if constexpr (kSlots)
              output[c][0][jg * kInt4GroupPacks + k][j] = group[c][0][k][j];
            else
              output[0][jg * kInt4GroupPacks + k][j] = group[0][k][j];
          }
    }
  } else {
    const auto* const entries =
        (decltype(+input_ptr_array[0]) const*)input_ptr_array;
    const void* const input =
        __builtin_constant_p(input_ptr_array_size) && input_ptr_array_size == 1
            ? entries[0]
            : internal::ptr_array_at(entries, input_ptr_array_size, input_rank);
    internal::read_buffer<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE,
                          QUANTIZE_INT4, LAYOUT, TAIL>(
        input, input_slot, input_stride_slot, input_stride_m, input_stride_n,
        input_size_slot, input_size_m, input_size_n, clip_m, clip_n, output,
        offs_m, offs_n, row_step, input_rank, flags, status_ptr,
        status_stride_field, status_size_field, timeout_ticks);
  }
}

template <typename T>
DINLINE T* load(T* const* x_ptr_array, int x_ptr_array_size, int x_rank) {
  return internal::ptr_array_at(x_ptr_array, x_ptr_array_size, x_rank);
}

template <int WORLD, int TILE_M, int THREADS_PER_BLOCK, int WAVE_SIZE,
          bool QUANTIZE_INT4, Layout LAYOUT, bool TAIL = false, typename PTR,
          typename OUTPUT>
DINLINE void recv(PTR const* input_ptr_array, int input_ptr_array_size,
                  int input_rank, int input_slot, int64_t input_stride_slot,
                  int64_t input_stride_m, int64_t input_stride_n,
                  int input_size_slot, int input_size_m, int input_size_n,
                  int clip_m, int clip_n, OUTPUT& output, int offs_m,
                  int offs_n, int row_step) {
  constexpr bool kSlots = std::rank_v<OUTPUT> == 4;
  constexpr int kPacks = std::extent_v < OUTPUT, kSlots ? 2 : 1 > ;
  if constexpr ((LAYOUT == Layout::group_run || LAYOUT == Layout::group_flat) &&
                kPacks != kInt4GroupPacks) {
    static_assert(kPacks % kInt4GroupPacks == 0, "a thread holds whole groups");
    constexpr int kCopies = kSlots ? std::extent_v<OUTPUT, 0> : 1;
    constexpr int kPackElems = std::extent_v < OUTPUT, kSlots ? 3 : 2 > ;
    using U = std::remove_all_extents_t<OUTPUT>;
    using Group =
        std::conditional_t<kSlots, U[kCopies][1][kInt4GroupPacks][kPackElems],
                           U[1][kInt4GroupPacks][kPackElems]>;
#pragma unroll
    for (int jg = 0; jg < kPacks / kInt4GroupPacks; ++jg) {
      Group group;
      recv<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4, LAYOUT,
           TAIL>(
          input_ptr_array, input_ptr_array_size, input_rank, input_slot,
          input_stride_slot, input_stride_m, input_stride_n, input_size_slot,
          input_size_m, input_size_n, clip_m, clip_n, group, offs_m,
          offs_n + jg * (THREADS_PER_BLOCK / TILE_M) * kInt4Group, row_step);
#pragma unroll
      for (int c = 0; c < kCopies; ++c)
#pragma unroll
        for (int k = 0; k < kInt4GroupPacks; ++k)
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) {
            if constexpr (kSlots)
              output[c][0][jg * kInt4GroupPacks + k][j] = group[c][0][k][j];
            else
              output[0][jg * kInt4GroupPacks + k][j] = group[0][k][j];
          }
    }
  } else {
    const auto* const entries =
        (decltype(+input_ptr_array[0]) const*)input_ptr_array;
    const int first = __builtin_amdgcn_readfirstlane(input_rank);
    if (__builtin_constant_p(input_ptr_array_size) && input_ptr_array_size == 1)
      internal::read_buffer<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE,
                            QUANTIZE_INT4, LAYOUT, TAIL>(
          entries[0], input_slot, input_stride_slot, input_stride_m,
          input_stride_n, input_size_slot, input_size_m, input_size_n, clip_m,
          clip_n, output, offs_m, offs_n, row_step);
    else if (__builtin_amdgcn_ballot_w64(input_rank != first) == 0)
      internal::read_buffer<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE,
                            QUANTIZE_INT4, LAYOUT, TAIL>(
          internal::ptr_array_at(entries, input_ptr_array_size, first),
          input_slot, input_stride_slot, input_stride_m, input_stride_n,
          input_size_slot, input_size_m, input_size_n, clip_m, clip_n, output,
          offs_m, offs_n, row_step);
    else
      internal::read_buffer<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE,
                            QUANTIZE_INT4, LAYOUT, TAIL>(
          internal::lane_ptr_array_at<void, WORLD>(
              entries, input_ptr_array_size, input_rank),
          input_slot, input_stride_slot, input_stride_m, input_stride_n,
          input_size_slot, input_size_m, input_size_n, clip_m, clip_n, output,
          offs_m, offs_n, row_step);
  }
}

template <int WORLD, int TILE_M, int THREADS_PER_BLOCK, int WAVE_SIZE,
          bool QUANTIZE_INT4, Layout LAYOUT, bool TAIL = false, typename INPUT,
          typename PTR>
DINLINE void store(const INPUT& input, PTR const* output_ptr_array,
                   int output_ptr_array_size, int output_rank, int output_slot,
                   int64_t output_stride_slot, int64_t output_stride_m,
                   int64_t output_stride_n, int output_size_slot,
                   int output_size_m, int output_size_n, int clip_m, int clip_n,
                   int offs_m, int offs_n, int row_step) {
  constexpr int kPacks = std::extent_v<INPUT, 1>;
  if constexpr ((LAYOUT == Layout::group_run || LAYOUT == Layout::group_flat) &&
                kPacks != kInt4GroupPacks) {
    static_assert(std::extent_v<INPUT, 0> == 1 && kPacks % kInt4GroupPacks == 0,
                  "a thread holds whole groups");
    constexpr int kPackElems = std::extent_v<INPUT, 2>;
    using U = std::remove_all_extents_t<INPUT>;
#pragma unroll
    for (int jg = 0; jg < kPacks / kInt4GroupPacks; ++jg) {
      U group[1][kInt4GroupPacks][kPackElems];
#pragma unroll
      for (int k = 0; k < kInt4GroupPacks; ++k)
#pragma unroll
        for (int j = 0; j < kPackElems; ++j)
          group[0][k][j] = input[0][jg * kInt4GroupPacks + k][j];
      store<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4, LAYOUT,
            TAIL>(group, output_ptr_array, output_ptr_array_size, output_rank,
                  output_slot, output_stride_slot, output_stride_m,
                  output_stride_n, output_size_slot, output_size_m,
                  output_size_n, clip_m, clip_n, offs_m,
                  offs_n + jg * (THREADS_PER_BLOCK / TILE_M) * kInt4Group,
                  row_step);
    }
  } else {
    const auto* const entries =
        (decltype(+output_ptr_array[0]) const*)output_ptr_array;
    void* const output = __builtin_constant_p(output_ptr_array_size) &&
                                 output_ptr_array_size == 1
                             ? entries[0]
                             : internal::ptr_array_at(
                                   entries, output_ptr_array_size, output_rank);
    internal::write_buffer<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE,
                           QUANTIZE_INT4, LAYOUT, TAIL>(
        input, output, output_slot, output_stride_slot, output_stride_m,
        output_stride_n, output_size_slot, output_size_m, output_size_n, clip_m,
        clip_n, offs_m, offs_n, row_step);
  }
}

template <int WORLD, int TILE_M, int THREADS_PER_BLOCK, int WAVE_SIZE,
          bool QUANTIZE_INT4, Layout LAYOUT, bool TAIL = false, typename INPUT,
          typename PTR>
DINLINE void send(const INPUT& input, PTR const* output_ptr_array,
                  int output_ptr_array_size, int output_rank, int output_slot,
                  int64_t output_stride_slot, int64_t output_stride_m,
                  int64_t output_stride_n, int output_size_slot,
                  int output_size_m, int output_size_n, int clip_m, int clip_n,
                  int offs_m, int offs_n, int row_step,
                  const uint32_t* flags = nullptr) {
  constexpr int kPacks = std::extent_v<INPUT, 1>;
  if constexpr ((LAYOUT == Layout::group_run || LAYOUT == Layout::group_flat) &&
                kPacks != kInt4GroupPacks) {
    static_assert(std::extent_v<INPUT, 0> == 1 && kPacks % kInt4GroupPacks == 0,
                  "a thread holds whole groups");
    constexpr int kPackElems = std::extent_v<INPUT, 2>;
    using U = std::remove_all_extents_t<INPUT>;
#pragma unroll
    for (int jg = 0; jg < kPacks / kInt4GroupPacks; ++jg) {
      U group[1][kInt4GroupPacks][kPackElems];
#pragma unroll
      for (int k = 0; k < kInt4GroupPacks; ++k)
#pragma unroll
        for (int j = 0; j < kPackElems; ++j)
          group[0][k][j] = input[0][jg * kInt4GroupPacks + k][j];
      send<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4, LAYOUT,
           TAIL>(group, output_ptr_array, output_ptr_array_size, output_rank,
                 output_slot, output_stride_slot, output_stride_m,
                 output_stride_n, output_size_slot, output_size_m,
                 output_size_n, clip_m, clip_n, offs_m,
                 offs_n + jg * (THREADS_PER_BLOCK / TILE_M) * kInt4Group,
                 row_step, flags);
    }
  } else {
    const auto* const entries =
        (decltype(+output_ptr_array[0]) const*)output_ptr_array;
    const int first = __builtin_amdgcn_readfirstlane(output_rank);
    if (__builtin_amdgcn_ballot_w64(output_rank != first) == 0)
      internal::write_buffer<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE,
                             QUANTIZE_INT4, LAYOUT, TAIL>(
          input, internal::ptr_array_at(entries, output_ptr_array_size, first),
          output_slot, output_stride_slot, output_stride_m, output_stride_n,
          output_size_slot, output_size_m, output_size_n, clip_m, clip_n,
          offs_m, offs_n, row_step, flags);
    else
      internal::write_buffer<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE,
                             QUANTIZE_INT4, LAYOUT, TAIL>(
          input,
          internal::lane_ptr_array_at<void, WORLD>(
              entries, output_ptr_array_size, output_rank),
          output_slot, output_stride_slot, output_stride_m, output_stride_n,
          output_size_slot, output_size_m, output_size_n, clip_m, clip_n,
          offs_m, offs_n, row_step, flags);
  }
}

}  // namespace hip_custom_communicator
