// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// HIP versions of the four Lamport P2P kernels in ../lamport.py (see its module
// docstring for the protocol). Buffers come from PyTorch symmetric memory or
// from the hipIpc helpers below; either way the host passes a device array of
// peer base pointers plus the u32 offset of the active generation.
//
// Every payload word is its own ready flag: a u32 holds two bf16s and
// 0x80008000 (two -0.0) means "empty". Kernels move 64-bit "quads" (two words,
// four bf16) with system-scope atomics: that bypasses the non-coherent caches,
// and since each 32-bit half is validated on its own, a quad whose halves land
// at different times is simply re-polled.

#include <hip/hip_runtime.h>
#include <c10/hip/HIPStream.h>
#include <torch/extension.h>

#include <cstdint>
#include <tuple>
#include <vector>

namespace {

constexpr uint32_t kEmptyPair = 0x80008000u;
constexpr uint64_t kEmptyQuad = (static_cast<uint64_t>(kEmptyPair) << 32) | kEmptyPair;

__device__ __forceinline__ uint64_t ld_sys(const uint64_t* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void st_sys(uint64_t* p, uint64_t v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ bool has_empty(uint64_t q) {
  return static_cast<uint32_t>(q) == kEmptyPair || static_cast<uint32_t>(q >> 32) == kEmptyPair;
}

__device__ __forceinline__ float lane_f32(uint64_t q, int lane) {  // bf16 lane 0..3
  return __uint_as_float(static_cast<uint32_t>((q >> (16 * lane)) & 0xffffu) << 16);
}

// Round-to-nearest-even, NaN kept quiet (matches torch / Triton .to(bf16)).
__device__ __forceinline__ uint64_t f32_to_bf16(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (u >> 16) | 0x40u;
  u += 0x7fffu + ((u >> 16) & 1u);
  return u >> 16;
}

// Four bf16 lanes -> one publishable quad; -0.0 is reserved for "empty".
__device__ __forceinline__ uint64_t sanitize(uint64_t q) {
  #pragma unroll
  for (int lane = 0; lane < 4; ++lane) {
    const uint64_t mask = 0xffffull << (16 * lane);
    if ((q & mask) == (0x8000ull << (16 * lane))) q &= ~mask;
  }
  return q;
}

__device__ __forceinline__ uint64_t* peer_base(const uint64_t* peer_ptrs, int peer,
                                               int64_t offset_u32) {
  return reinterpret_cast<uint64_t*>(reinterpret_cast<uint32_t*>(peer_ptrs[peer]) + offset_u32);
}

// Spin until both words of the quad are published. max_spins bounds the wait
// for diagnostics; on give-up the stall counters say which phase/token/split/
// source starved.
__device__ __forceinline__ uint64_t wait_quad(const uint64_t* p, int64_t max_spins,
                                              int32_t* stall, int phase, int token,
                                              int split, int source) {
  uint64_t v = ld_sys(p);
  int64_t spins = 0;
  while (has_empty(v) && spins < max_spins) {
    v = ld_sys(p);
    ++spins;
  }
  if (has_empty(v)) {
    atomicAdd(stall + phase, 1);
    atomicMax(stall + 2, 1);
    atomicMin(stall + 3, token);
    atomicMax(stall + 4, split);
    if (source >= 0) atomicAdd(stall + 8 + source, 1);
  }
  return v;
}

// 1) Publish this rank's partial row into every shard owner's input slots.
//    Input slot layout (u32): [token][source rank][CS / 2].
__global__ void publish_input_kernel(const uint64_t* __restrict__ stage,  // bf16 quads
                                     int64_t stage_stride_q, const uint64_t* peer_ptrs,
                                     int64_t in_off_u32, int CS, int splits, int rank,
                                     int world) {
  const int token = blockIdx.x, split = blockIdx.y;
  const int quads = CS / splits / 4;
  const uint64_t* src_row = stage + static_cast<int64_t>(token) * stage_stride_q;
  for (int owner = 0; owner < world; ++owner) {
    uint64_t* dst = peer_base(peer_ptrs, owner, in_off_u32) +
                    (static_cast<int64_t>(token) * world + rank) * (CS / 4) + split * quads;
    const uint64_t* src = src_row + (owner * CS) / 4 + split * quads;
    for (int q = threadIdx.x; q < quads; q += blockDim.x) st_sys(dst + q, sanitize(src[q]));
  }
}

// 2) Wait for every source's partial of this rank's shard, reduce in fp32 (in
//    source order), re-arm the slots, and insert the reduced row into the cache.
//    All sources' quads are requested at once; only still-empty ones are re-polled.
constexpr int kMaxWorld = 8;

template <bool VEC>
__global__ void reduce_insert_kernel(const uint64_t* peer_ptrs, int64_t in_off_u32,
                                     uint16_t* __restrict__ cache, const int64_t* slot_mapping,
                                     int64_t s_blk, int64_t s_head, int64_t s_tok, int64_t s_dim,
                                     int64_t block_size, int CS, int splits, int head_size,
                                     int cache_offset, int rank, int world) {
  const int token = blockIdx.x, split = blockIdx.y;
  const int quads = CS / splits / 4;
  uint64_t* in = peer_base(peer_ptrs, rank, in_off_u32);
  const int64_t slot = slot_mapping[token];
  const bool valid = slot >= 0;
  const int64_t slot_base = valid ? (slot / block_size) * s_blk + (slot % block_size) * s_tok : 0;
  const int n_heads = CS / head_size;
  for (int q = threadIdx.x; q < quads; q += blockDim.x) {
    uint64_t* w[kMaxWorld];
    uint64_t v[kMaxWorld];
    #pragma unroll
    for (int src = 0; src < kMaxWorld; ++src) {
      if (src < world) {
        w[src] = in + (static_cast<int64_t>(token) * world + src) * (CS / 4) + split * quads + q;
        v[src] = ld_sys(w[src]);
      }
    }
    int64_t spins = 0;
    while (true) {
      bool pending = false;
      #pragma unroll
      for (int src = 0; src < kMaxWorld; ++src) pending |= src < world && has_empty(v[src]);
      if (!pending) break;
      #pragma unroll
      for (int src = 0; src < kMaxWorld; ++src)
        if (src < world && has_empty(v[src])) v[src] = ld_sys(w[src]);
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int src = 0; src < kMaxWorld; ++src) {
      if (src < world) {
        #pragma unroll
        for (int lane = 0; lane < 4; ++lane) acc[lane] += lane_f32(v[src], lane);
        st_sys(w[src], kEmptyQuad);
      }
    }
    if (valid) {
      const int c0 = split * (quads * 4) + 4 * q;
      if constexpr (VEC) {
        // 4 channels share a head (head_size % 4 == 0) and are contiguous.
        const int head = min(c0 / head_size, n_heads - 1);
        uint64_t out = 0;
        #pragma unroll
        for (int lane = 0; lane < 4; ++lane) out |= f32_to_bf16(acc[lane]) << (16 * lane);
        *reinterpret_cast<uint64_t*>(cache + slot_base + head * s_head + cache_offset + c0 % head_size) = out;
      } else {
        #pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
          const int c = c0 + lane;
          const int head = min(c / head_size, n_heads - 1);
          cache[slot_base + head * s_head + (cache_offset + c % head_size) * s_dim] =
              static_cast<uint16_t>(f32_to_bf16(acc[lane]));
        }
      }
    }
  }
}

// 3) Short conv over the 4-tap window (3 pre-forward taps from the paged cache
//    + the just-inserted current value), add the residual shard, and publish
//    the bf16 result into every rank's output row.
template <bool VEC>
__global__ void sconv_publish_kernel(const uint64_t* peer_ptrs, int64_t out_off_u32,
                                     const uint16_t* __restrict__ residual, int64_t res_stride,
                                     const uint16_t* __restrict__ weight,  // [CS, 4]
                                     const uint16_t* __restrict__ cache, const int64_t* positions,
                                     const int32_t* seq_idx, const int64_t* slot_mapping,
                                     const int32_t* block_table, int64_t bt_stride,
                                     int64_t max_blocks, int64_t s_blk, int64_t s_head,
                                     int64_t s_tok, int64_t s_dim, int64_t block_size, int C,
                                     int CS, int splits, int head_size, int cache_offset,
                                     int rank, int world) {
  constexpr int kWindow = 4;
  const int token = blockIdx.x, split = blockIdx.y;
  const int quads = CS / splits / 4;
  const int64_t slot = slot_mapping[token];
  const bool valid = slot >= 0;
  const int64_t position = positions[token];
  const int64_t sequence = seq_idx[token];
  const int n_heads = CS / head_size;
  // Physical base of each pre-forward tap (same for every channel of the token).
  int64_t tap_base[kWindow - 1];
  bool tap_take[kWindow - 1];
  #pragma unroll
  for (int tap = 0; tap < kWindow - 1; ++tap) {
    const int64_t src_pos = position - (kWindow - 1) + tap;
    tap_take[tap] = valid && src_pos >= 0;
    tap_base[tap] = 0;
    if (tap_take[tap]) {
      const int64_t lblk = min(src_pos / block_size, max_blocks - 1);
      tap_base[tap] = static_cast<int64_t>(block_table[sequence * bt_stride + lblk]) * s_blk +
                      (src_pos % block_size) * s_tok;
    }
  }
  const int64_t own_base = valid ? (slot / block_size) * s_blk + (slot % block_size) * s_tok : 0;
  const bool cur_take = valid && position >= 0;
  const int64_t out_off = static_cast<int64_t>(token) * (C / 4) + rank * (CS / 4) + split * quads;
  for (int q = threadIdx.x; q < quads; q += blockDim.x) {
    const int c0 = split * (quads * 4) + 4 * q;
    // Gather the 4 channels' inputs: current value, 3 taps, residual, 4x4 weights.
    float cur[4], tapv[kWindow - 1][4], res[4], wt[4][kWindow];
    if constexpr (VEC) {
      const int head = min(c0 / head_size, n_heads - 1);
      const int64_t chan = head * s_head + cache_offset + c0 % head_size;
      const uint64_t cq = valid ? *reinterpret_cast<const uint64_t*>(cache + own_base + chan) : 0;
      uint64_t tq[kWindow - 1];
      #pragma unroll
      for (int tap = 0; tap < kWindow - 1; ++tap)
        tq[tap] = tap_take[tap] ? *reinterpret_cast<const uint64_t*>(cache + tap_base[tap] + chan) : 0;
      const uint64_t rq = *reinterpret_cast<const uint64_t*>(residual + token * res_stride + rank * CS + c0);
      const uint64_t* wq = reinterpret_cast<const uint64_t*>(weight + c0 * kWindow);
      #pragma unroll
      for (int lane = 0; lane < 4; ++lane) {
        cur[lane] = lane_f32(cq, lane);
        res[lane] = lane_f32(rq, lane);
        const uint64_t wl = wq[lane];  // channel c0 + lane: its 4 taps
        #pragma unroll
        for (int tap = 0; tap < kWindow; ++tap) wt[lane][tap] = lane_f32(wl, tap);
        #pragma unroll
        for (int tap = 0; tap < kWindow - 1; ++tap) tapv[tap][lane] = lane_f32(tq[tap], lane);
      }
    } else {
      #pragma unroll
      for (int lane = 0; lane < 4; ++lane) {
        const int c = c0 + lane;
        const int head = min(c / head_size, n_heads - 1);
        const int64_t chan = head * s_head + (cache_offset + c % head_size) * s_dim;
        auto f = [](uint16_t b) { return __uint_as_float(static_cast<uint32_t>(b) << 16); };
        cur[lane] = valid ? f(cache[own_base + chan]) : 0.f;
        res[lane] = f(residual[token * res_stride + rank * CS + c]);
        #pragma unroll
        for (int tap = 0; tap < kWindow; ++tap) wt[lane][tap] = f(weight[c * kWindow + tap]);
        #pragma unroll
        for (int tap = 0; tap < kWindow - 1; ++tap)
          tapv[tap][lane] = tap_take[tap] ? f(cache[tap_base[tap] + chan]) : 0.f;
      }
    }
    uint64_t packed = 0;
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      float conv = 0.f;
      #pragma unroll
      for (int tap = 0; tap < kWindow - 1; ++tap) conv += tapv[tap][lane] * wt[lane][tap];
      conv += (cur_take ? cur[lane] : 0.f) * wt[lane][kWindow - 1];
      // Both bf16 rounding points of the original sublayer.
      const float skip = __uint_as_float(static_cast<uint32_t>(f32_to_bf16(conv + cur[lane])) << 16);
      packed |= f32_to_bf16(res[lane] + skip) << (16 * lane);
    }
    packed = sanitize(packed);
    for (int dst = 0; dst < world; ++dst) st_sys(peer_base(peer_ptrs, dst, out_off_u32) + out_off + q, packed);
  }
}

// 4) Wait for the complete output row, write the new residual, RMS-normalize,
//    and re-arm the row. One block per token; each thread requests all of its
//    quads before waiting on any of them.
constexpr int kMaxQuadsPerThread = 4;

__global__ void gather_norm_kernel(const uint64_t* peer_ptrs, int64_t out_off_u32,
                                   const uint64_t* __restrict__ norm_weight,
                                   uint64_t* __restrict__ normed, uint64_t* __restrict__ residual_out,
                                   int64_t out_stride_q, float eps, int C, int rank, bool has_norm) {
  extern __shared__ float partial[];
  const int token = blockIdx.x;
  const int quads = C / 4;
  uint64_t* row = peer_base(peer_ptrs, rank, out_off_u32) + static_cast<int64_t>(token) * quads;
  uint64_t* res_out = residual_out + token * out_stride_q;
  uint64_t v[kMaxQuadsPerThread];
  #pragma unroll
  for (int i = 0; i < kMaxQuadsPerThread; ++i) {
    const int q = threadIdx.x + i * blockDim.x;
    v[i] = q < quads ? ld_sys(row + q) : 0;
  }
  int64_t spins = 0;
  while (true) {
    bool pending = false;
    #pragma unroll
    for (int i = 0; i < kMaxQuadsPerThread; ++i)
      pending |= threadIdx.x + i * blockDim.x < quads && has_empty(v[i]);
    if (!pending) break;
    #pragma unroll
    for (int i = 0; i < kMaxQuadsPerThread; ++i) {
      const int q = threadIdx.x + i * blockDim.x;
      if (q < quads && has_empty(v[i])) v[i] = ld_sys(row + q);
    }
  }
  float sumsq = 0.f;
  #pragma unroll
  for (int i = 0; i < kMaxQuadsPerThread; ++i) {
    const int q = threadIdx.x + i * blockDim.x;
    if (q < quads) {
      res_out[q] = v[i];
      #pragma unroll
      for (int lane = 0; lane < 4; ++lane) {
        const float x = lane_f32(v[i], lane);
        sumsq += x * x;
      }
    }
  }
  if (has_norm) {
    partial[threadIdx.x] = sumsq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
      __syncthreads();
    }
    const float inv_rms = rsqrtf(partial[0] / C + eps);
    uint64_t* y = normed + token * out_stride_q;
    #pragma unroll
    for (int i = 0; i < kMaxQuadsPerThread; ++i) {
      const int q = threadIdx.x + i * blockDim.x;
      if (q < quads) {
        const uint64_t wv = norm_weight[q];
        uint64_t out = 0;
        #pragma unroll
        for (int lane = 0; lane < 4; ++lane)
          out |= f32_to_bf16(lane_f32(v[i], lane) * inv_rms * lane_f32(wv, lane)) << (16 * lane);
        y[q] = out;
      }
    }
  }
  #pragma unroll
  for (int i = 0; i < kMaxQuadsPerThread; ++i) {
    const int q = threadIdx.x + i * blockDim.x;
    if (q < quads) st_sys(row + q, kEmptyQuad);
  }
}

int threads_for(int quads) {
  const int t = ((quads + 63) / 64) * 64;
  return t < 64 ? 64 : (t > 256 ? 256 : t);
}

void check_bf16(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kBFloat16, name, " must be a bf16 GPU tensor");
}

// 8-byte cache access needs a contiguous D dim and 4-element-aligned channels.
bool cache_vec_ok(const torch::Tensor& cache, int64_t head_size, int64_t cache_offset) {
  return cache.stride(3) == 1 && head_size % 4 == 0 && cache_offset % 4 == 0 &&
         cache.stride(0) % 4 == 0 && cache.stride(1) % 4 == 0 && cache.stride(2) % 4 == 0 &&
         reinterpret_cast<uintptr_t>(cache.data_ptr()) % 8 == 0;
}

void check_quads(int64_t CS, int64_t splits, int64_t off_u32) {
  TORCH_CHECK(CS % (4 * splits) == 0, "shard must split into whole bf16 quads");
  TORCH_CHECK(off_u32 % 2 == 0, "generation offset must be 8-byte aligned");
}

}  // namespace

void publish_input(torch::Tensor stage, int64_t peer_ptrs, int64_t in_off_u32, int64_t CS,
                   int64_t splits, int64_t rank, int64_t world) {
  check_bf16(stage, "stage");
  check_quads(CS, splits, in_off_u32);
  TORCH_CHECK(stage.stride(1) == 1 && stage.stride(0) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(stage.data_ptr()) % 8 == 0,
              "stage rows must be 8-byte aligned contiguous quads");
  const int tokens = stage.size(0), quads = CS / splits / 4;
  auto stream = c10::hip::getCurrentHIPStream().stream();
  hipLaunchKernelGGL(publish_input_kernel, dim3(tokens, splits), dim3(threads_for(quads)), 0, stream,
                     reinterpret_cast<const uint64_t*>(stage.data_ptr()), stage.stride(0) / 4,
                     reinterpret_cast<const uint64_t*>(peer_ptrs), in_off_u32, (int)CS, (int)splits,
                     (int)rank, (int)world);
}

void reduce_insert(int64_t peer_ptrs, int64_t in_off_u32, 
                   torch::Tensor cache, torch::Tensor slot_mapping, int64_t tokens,
                   int64_t block_size, int64_t CS, int64_t splits, int64_t head_size,
                   int64_t cache_offset, int64_t rank, int64_t world) {
  check_bf16(cache, "cache");
  check_quads(CS, splits, in_off_u32);
  TORCH_CHECK(slot_mapping.scalar_type() == torch::kInt64, "slot_mapping must be int64");
  TORCH_CHECK(world <= kMaxWorld, "world size above ", kMaxWorld);
  const int quads = CS / splits / 4;
  auto stream = c10::hip::getCurrentHIPStream().stream();
  auto kernel = cache_vec_ok(cache, head_size, cache_offset) ? reduce_insert_kernel<true>
                                                              : reduce_insert_kernel<false>;
  hipLaunchKernelGGL(kernel, dim3(tokens, splits), dim3(threads_for(quads)), 0, stream,
                     reinterpret_cast<const uint64_t*>(peer_ptrs), in_off_u32, 
                     reinterpret_cast<uint16_t*>(cache.data_ptr()),
                     slot_mapping.data_ptr<int64_t>(), cache.stride(0), cache.stride(1),
                     cache.stride(2), cache.stride(3), block_size, (int)CS, (int)splits,
                     (int)head_size, (int)cache_offset, (int)rank, (int)world);
}

void sconv_publish(int64_t peer_ptrs, int64_t out_off_u32, torch::Tensor residual,
                   torch::Tensor weight, torch::Tensor cache, torch::Tensor positions,
                   torch::Tensor seq_idx, torch::Tensor slot_mapping, torch::Tensor block_table,
                   int64_t block_size, int64_t CS, int64_t splits, int64_t head_size,
                   int64_t cache_offset, int64_t rank, int64_t world) {
  check_bf16(residual, "residual");
  check_bf16(weight, "conv_weight");
  check_bf16(cache, "cache");
  check_quads(CS, splits, out_off_u32);
  TORCH_CHECK(positions.scalar_type() == torch::kInt64, "positions must be int64");
  TORCH_CHECK(seq_idx.scalar_type() == torch::kInt32, "seq_idx must be int32");
  TORCH_CHECK(slot_mapping.scalar_type() == torch::kInt64, "slot_mapping must be int64");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "block_table must be int32");
  TORCH_CHECK(weight.is_contiguous() && weight.size(1) == 4, "conv_weight must be contiguous [CS, 4]");
  TORCH_CHECK(residual.stride(1) == 1, "residual rows must be contiguous");
  const int tokens = residual.size(0), C = residual.size(1), quads = CS / splits / 4;
  const bool vec = cache_vec_ok(cache, head_size, cache_offset) && residual.stride(0) % 4 == 0 &&
                   reinterpret_cast<uintptr_t>(residual.data_ptr()) % 8 == 0 &&
                   reinterpret_cast<uintptr_t>(weight.data_ptr()) % 8 == 0;
  auto stream = c10::hip::getCurrentHIPStream().stream();
  auto kernel = vec ? sconv_publish_kernel<true> : sconv_publish_kernel<false>;
  hipLaunchKernelGGL(kernel, dim3(tokens, splits), dim3(threads_for(quads)), 0, stream,
                     reinterpret_cast<const uint64_t*>(peer_ptrs), out_off_u32,
                     reinterpret_cast<const uint16_t*>(residual.data_ptr()), residual.stride(0),
                     reinterpret_cast<const uint16_t*>(weight.data_ptr()),
                     reinterpret_cast<const uint16_t*>(cache.data_ptr()),
                     positions.data_ptr<int64_t>(), seq_idx.data_ptr<int32_t>(),
                     slot_mapping.data_ptr<int64_t>(), block_table.data_ptr<int32_t>(),
                     block_table.stride(0), block_table.size(1), cache.stride(0), cache.stride(1),
                     cache.stride(2), cache.stride(3), block_size, C, (int)CS, (int)splits,
                     (int)head_size, (int)cache_offset, (int)rank, (int)world);
}

void gather_norm(int64_t peer_ptrs, int64_t out_off_u32, 
                 torch::Tensor norm_weight, torch::Tensor normed, torch::Tensor residual_out,
                 double eps, int64_t rank, bool has_norm) {
  check_bf16(residual_out, "residual_out");
  const int tokens = residual_out.size(0), C = residual_out.size(1);
  TORCH_CHECK(C % 4 == 0 && out_off_u32 % 2 == 0, "rows must be whole, aligned quads");
  TORCH_CHECK(residual_out.stride(1) == 1 && residual_out.stride(0) % 4 == 0 &&
                  normed.stride(0) == residual_out.stride(0),
              "normed / residual_out must share a row-contiguous, quad-aligned layout");
  // Power of two (the sum-of-squares tree halves it), up to one quad per thread.
  const int quads = C / 4;
  int threads = 64;
  while (threads < quads && threads < 1024) threads *= 2;
  TORCH_CHECK(quads <= threads * kMaxQuadsPerThread, "row too wide for gather_norm");
  auto stream = c10::hip::getCurrentHIPStream().stream();
  hipLaunchKernelGGL(gather_norm_kernel, dim3(tokens), dim3(threads), threads * sizeof(float), stream,
                     reinterpret_cast<const uint64_t*>(peer_ptrs), out_off_u32, 
                     reinterpret_cast<const uint64_t*>(norm_weight.data_ptr()),
                     reinterpret_cast<uint64_t*>(normed.data_ptr()),
                     reinterpret_cast<uint64_t*>(residual_out.data_ptr()), residual_out.stride(0) / 4,
                     (float)eps, C, (int)rank, has_norm);
}

// hipIpc-shared buffers, the same scheme as vLLM's / aiter's custom all-reduce
// (csrc/libtorch_stable/custom_all_reduce.cu): allocate, export a handle,
// exchange handles on the host, open the peers' handles.
#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t e = (cmd);                                                      \
    TORCH_CHECK(e == hipSuccess, #cmd " failed: ", hipGetErrorString(e));      \
  } while (0)

// mode: 0 = uncached, 1 = fine-grained, 2 = plain hipMalloc (coarse-grained).
std::tuple<int64_t, torch::Tensor> ipc_alloc(int64_t nbytes, int64_t mode) {
  TORCH_CHECK(mode >= 0 && mode <= 2, "ipc_alloc mode must be 0, 1 or 2");
  hipStreamCaptureMode capture = hipStreamCaptureModeRelaxed;
  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&capture));
  void* ptr = nullptr;
  if (mode == 2) {
    HIP_CHECK(hipMalloc(&ptr, nbytes));
  } else {
    HIP_CHECK(hipExtMallocWithFlags(
        &ptr, nbytes, mode == 0 ? hipDeviceMallocUncached : hipDeviceMallocFinegrained));
  }
  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&capture));
  auto handle = torch::empty({static_cast<int64_t>(sizeof(hipIpcMemHandle_t))},
                             torch::TensorOptions().dtype(torch::kUInt8));
  HIP_CHECK(hipIpcGetMemHandle(reinterpret_cast<hipIpcMemHandle_t*>(handle.data_ptr()), ptr));
  return {reinterpret_cast<int64_t>(ptr), handle};
}

int64_t ipc_open(torch::Tensor handle) {
  TORCH_CHECK(handle.device().is_cpu() && handle.numel() == sizeof(hipIpcMemHandle_t),
              "expected a CPU uint8 hipIpcMemHandle_t");
  void* ptr = nullptr;
  HIP_CHECK(hipIpcOpenMemHandle(&ptr, *reinterpret_cast<const hipIpcMemHandle_t*>(handle.data_ptr()),
                                hipIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<int64_t>(ptr);
}

void ipc_close(int64_t ptr) { HIP_CHECK(hipIpcCloseMemHandle(reinterpret_cast<void*>(ptr))); }

void ipc_free(int64_t ptr) { HIP_CHECK(hipFree(reinterpret_cast<void*>(ptr))); }

// Non-owning bf16 view of a local buffer (the caller keeps the allocation alive).
torch::Tensor as_tensor(int64_t ptr, std::vector<int64_t> sizes, int64_t device_index) {
  return torch::from_blob(reinterpret_cast<void*>(ptr), sizes,
                          torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, device_index));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("publish_input", &publish_input);
  m.def("reduce_insert", &reduce_insert);
  m.def("sconv_publish", &sconv_publish);
  m.def("gather_norm", &gather_norm);
  m.def("ipc_alloc", &ipc_alloc);
  m.def("ipc_open", &ipc_open);
  m.def("ipc_close", &ipc_close);
  m.def("ipc_free", &ipc_free);
  m.def("as_tensor", &as_tensor);
}
