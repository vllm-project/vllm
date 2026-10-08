#pragma once
#include <cutlass/arch/barrier.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda_fp16.h>
constexpr unsigned FallbackSegment = 1024, FallbackRetain = 4096;
struct FallbackBuffers {
  float* ring;
  float* retained;
  int* indices;
  int* output;
};
struct alignas(16) FallbackShared {
  cutlass::arch::ClusterBarrier full[2], empty[2];
  float ring[2][8][FallbackSegment];
  unsigned hist[8][1056], groups[8][32], gates[8], kept[8];
  unsigned work[264], prefix, rank, written, added, emit_counts[8];
  __device__ void init(unsigned tid) {
    if (tid < 2) {
      full[tid].init(256);
      empty[tid].init(256);
    }
    for (unsigned j = tid; j < 8 * 1056; j += 384)
      reinterpret_cast<unsigned*>(hist)[j] = 0;
    if (tid < 256) reinterpret_cast<unsigned*>(groups)[tid] = 0;
    if (tid < 8) {
      gates[tid] = __float_as_uint(-CUDART_INF_F);
      kept[tid] = 0;
    }
    cutlass::arch::fence_barrier_init();
  }
};
__device__ __forceinline__ unsigned fb_code(float x) {
  unsigned u = __float_as_uint(x);
  return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}
__device__ __forceinline__ float fb_float(unsigned u) {
  return __uint_as_float((u & 0x80000000u) ? (u ^ 0x80000000u) : ~u);
}
__device__ __forceinline__ unsigned fb_half_bin(float x) {
  unsigned u = __half_as_ushort(__float2half_rd(x));
  return ((u & 0x8000u) ? ((~u) & 65535u) : (u ^ 0x8000u)) >> 6;
}
__device__ __forceinline__ float fb_bin_floor(unsigned b) {
  unsigned o = b << 6, u = (o & 0x8000u) ? (o ^ 0x8000u) : ((~o) & 65535u);
  return __half2float(__ushort_as_half(u));
}
__device__ __forceinline__ void fb_sync() {
  asm volatile("" ::: "memory");
  cutlass::arch::NamedBarrier(256, 2).sync();
  asm volatile("" ::: "memory");
}
__device__ void fb_compact(FallbackShared& s, FallbackBuffers b, unsigned local,
                           unsigned row, unsigned tid, bool final = false) {
  constexpr unsigned K = 2048, Full = 0xffffffffu, T = 256,
                     Items = FallbackRetain / T;
  unsigned lane = tid & 31, count = s.kept[local];
  uint64_t base = (uint64_t)row * FallbackRetain;
  float values[Items];
  int indices[Items];
#pragma unroll
  for (unsigned i = 0; i < Items; ++i) {
    unsigned j = tid + i * T;
    values[i] = j < count ? b.retained[base + j] : -CUDART_INF_F;
    indices[i] = j < count ? b.indices[base + j] : -1;
  }
  auto emit = [&](float gate, unsigned threshold, int mode,
                  unsigned base_offset) {
    unsigned total = 0;
#pragma unroll
    for (unsigned i = 0; i < Items; ++i) {
      float v = values[i];
      bool pass =
          isfinite(v) && (mode == 0 ? v >= gate
                                    : (mode == 1 ? fb_code(v) > threshold
                                                 : fb_code(v) == threshold));
      total += pass;
    }
    unsigned warp_total = __reduce_add_sync(Full, total);
    if (lane == 0) s.emit_counts[tid / 32] = warp_total;
    fb_sync();
    unsigned start = base_offset;
    for (unsigned w = 0; w < tid / 32; ++w) start += s.emit_counts[w];
    unsigned all = 0;
    for (unsigned w = 0; w < 8; ++w) all += s.emit_counts[w];
#pragma unroll
    for (unsigned i = 0; i < Items; ++i) {
      float v = values[i];
      bool pass =
          isfinite(v) && (mode == 0 ? v >= gate
                                    : (mode == 1 ? fb_code(v) > threshold
                                                 : fb_code(v) == threshold));
      unsigned mask = __ballot_sync(Full, pass),
               pos = start + __popc(mask & ((1u << lane) - 1));
      if (pass && (mode != 2 || pos < K)) {
        b.retained[base + pos] = v;
        b.indices[base + pos] = indices[i];
      }
      start += __popc(mask);
    }
    fb_sync();
    return all;
  };
  if (!final) {
    if (tid == 0) {
      s.rank = 0;
      s.written = 0;
    }
    fb_sync();
    float gate = __uint_as_float(s.gates[local]);
    unsigned kept = 0;
#pragma unroll
    for (unsigned i = 0; i < Items; ++i)
      kept += isfinite(values[i]) && values[i] >= gate;
    kept = __reduce_add_sync(Full, kept);
    if (lane == 0) atomicAdd(&s.rank, kept);
    fb_sync();
    kept = s.rank;
    if (kept >= K && kept <= FallbackRetain - FallbackSegment) {
      emit(gate, 0, 0, 0);
      fb_sync();
      if (tid == 0) s.kept[local] = kept;
      fb_sync();
      return;
    }
  }
  if (tid == 0) {
    s.prefix = 0;
    s.rank = K;
    s.written = 0;
  }
  fb_sync();
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (unsigned j = tid; j < 264; j += T) s.work[j] = 0;
    fb_sync();
    unsigned prefix = s.prefix;
#pragma unroll
    for (unsigned i = 0; i < Items; ++i) {
      unsigned code = fb_code(values[i]);
      bool valid = isfinite(values[i]) &&
                   (shift == 24 || (code >> (shift + 8)) == prefix);
      unsigned mask = __ballot_sync(Full, valid);
      if (valid) {
        unsigned bin = (code >> shift) & 255u,
                 peers = __match_any_sync(mask, bin);
        if (lane == unsigned(__ffs(peers) - 1))
          atomicAdd(s.work + bin + bin / 32, __popc(peers));
      }
    }
    fb_sync();
    if (tid < 32) {
      unsigned sum = 0;
      for (unsigned j = 0; j < 8; ++j) {
        unsigned bin = 255 - lane * 8 - j;
        sum += s.work[bin + bin / 32];
      }
      unsigned inclusive = sum;
      for (unsigned d = 1; d < 32; d *= 2) {
        unsigned v = __shfl_up_sync(Full, inclusive, d);
        if (lane >= d) inclusive += v;
      }
      unsigned before = inclusive - sum, rank = s.rank,
               crossing =
                   __ballot_sync(Full, before < rank && inclusive >= rank),
               chosen_lane = __ffs(crossing) - 1, chosen = 0, above = before;
      if (lane == chosen_lane) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned bin = 255 - lane * 8 - j;
          if (above + s.work[bin + bin / 32] >= rank) {
            chosen = bin;
            break;
          }
          above += s.work[bin + bin / 32];
        }
      }
      chosen = __shfl_sync(Full, chosen, chosen_lane);
      above = __shfl_sync(Full, above, chosen_lane);
      if (lane == 0) {
        s.prefix = (prefix << 8) | chosen;
        s.rank = rank - above;
      }
    }
    fb_sync();
  }
  unsigned threshold = s.prefix;
  unsigned greater = emit(0, threshold, 1, 0);
  emit(0, threshold, 2, greater);
  if (tid == 0) {
    s.kept[local] = K;
    float gate = fmaxf(__uint_as_float(s.gates[local]), fb_float(threshold));
    atomicExch(s.gates + local, __float_as_uint(gate));
  }
  fb_sync();
}
__device__ void fb_select(FallbackShared& s, FallbackBuffers b, unsigned local,
                          unsigned row, unsigned epoch, unsigned first,
                          unsigned n, unsigned tid, bool final) {
  constexpr unsigned K = 2048, Full = 0xffffffffu, T = 32,
                     Items = FallbackSegment / T;
  unsigned lane = tid, bank = epoch & 1;
  float values[Items], gate = __uint_as_float(s.gates[local]);
  unsigned count = 0;
#pragma unroll
  for (unsigned i = 0; i < Items; ++i) {
    unsigned j = tid + i * T;
    float v = j < n ? s.ring[bank][local][j] : -CUDART_INF_F;
    if (v <= gate) v = -CUDART_INF_F;
    values[i] = v;
    bool valid = isfinite(v);
    count += valid;
    unsigned mask = __ballot_sync(Full, valid);
    if (valid) {
      unsigned bin = fb_half_bin(v), peers = __match_any_sync(mask, bin);
      if (lane == unsigned(__ffs(peers) - 1))
        atomicAdd(&s.hist[local][bin + bin / 32], __popc(peers));
    }
  }
  count = __reduce_add_sync(Full, count);
  __syncwarp();
  if (count) {
    unsigned sum = 0;
#pragma unroll
    for (unsigned j = 0; j < 32; ++j)
      sum += s.hist[local][(31 - lane) * 33 + j];
    unsigned inclusive = sum;
    for (unsigned d = 1; d < 32; d *= 2) {
      unsigned v = __shfl_up_sync(Full, inclusive, d);
      if (lane >= d) inclusive += v;
    }
    unsigned before = inclusive - sum,
             mask = __ballot_sync(Full, before < K && inclusive >= K);
    if (mask) {
      unsigned chosen = __ffs(mask) - 1, group = 31 - chosen,
               rank = K - __shfl_sync(Full, before, chosen),
               bin = group * 32 + 31 - lane;
      sum = s.hist[local][bin + bin / 32];
      inclusive = sum;
      for (unsigned d = 1; d < 32; d *= 2) {
        unsigned v = __shfl_up_sync(Full, inclusive, d);
        if (lane >= d) inclusive += v;
      }
      before = inclusive - sum;
      chosen =
          __ffs(__ballot_sync(Full, before < rank && inclusive >= rank)) - 1;
      bin = __shfl_sync(Full, bin, chosen);
      gate = fmaxf(gate, fb_bin_floor(bin));
      if (lane == 0) atomicExch(s.gates + local, __float_as_uint(gate));
    }
    unsigned written = s.kept[local];
    uint64_t base = (uint64_t)row * FallbackRetain;
#pragma unroll
    for (unsigned i = 0; i < Items; ++i) {
      float v = values[i];
      bool pass = isfinite(v) && v >= gate;
      unsigned mask = __ballot_sync(Full, pass),
               pos = written + __popc(mask & ((1u << lane) - 1));
      if (pass) {
        b.retained[base + pos] = v;
        b.indices[base + pos] = first + tid + i * T;
      }
      written += __popc(mask);
    }
    __syncwarp();
    if (lane == 0) s.kept[local] = written;
    __syncwarp();
  }
}
