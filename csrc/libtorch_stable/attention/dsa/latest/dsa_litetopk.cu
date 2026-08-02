// LiteTopK DSA V3 hybrid host wrapper: DeepGEMM-2.5 scoring loop + V1 KV-split;
// scoring kernel (sm100_dsa_litetopk.cuh) with the sparse candidate epilogue,
// plus the architecture-agnostic radix-select post-kernels (copied verbatim
// from dsa_litetopk.cu). Build against the DeepGEMM 2.5 include tree + its
// bundled CUTLASS (NOT the legacy deep_gemm include tree V1 uses).

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>

#include "sm100_dsa_litetopk.cuh"
#include "dense_topk_litetopk.cuh"

namespace {

using CandidateValue = dsa_litetopk::CandidateValue;

static torch::TensorOptions candidate_options(
    const torch::TensorOptions& options) {
  // torch.float16 is only the owning 16-bit storage type here.  CUDA treats
  // its payload as an opaque uint16 score code; no half arithmetic occurs.
  return options.dtype(torch::kHalf);
}

static CandidateValue* candidate_data_ptr(torch::Tensor& tensor) {
  return reinterpret_cast<CandidateValue*>(tensor.data_ptr<at::Half>());
}

static void check_candidate_dtype(const torch::Tensor& tensor) {
  TORCH_CHECK(tensor.scalar_type() == torch::kHalf,
              "cand_val must use float16 as opaque packed storage");
}

static void* driver_handle() {
  static void* h = nullptr;
  if (!h) {
    h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    TORCH_CHECK(h, "failed to load libcuda.so.1");
  }
  return h;
}

static CUresult enc_tiled(CUtensorMap* tm, CUtensorMapDataType dt,
                          cuuint32_t rank, void* addr, const cuuint64_t* dims,
                          const cuuint64_t* strides, const cuuint32_t* box,
                          const cuuint32_t* estrides, CUtensorMapInterleave il,
                          CUtensorMapSwizzle sw, CUtensorMapL2promotion l2,
                          CUtensorMapFloatOOBfill oob) {
  using FT =
      CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                   const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                   const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                   CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  static FT f = nullptr;
  if (!f) {
    f = reinterpret_cast<FT>(dlsym(driver_handle(), "cuTensorMapEncodeTiled"));
    TORCH_CHECK(f, "failed to load cuTensorMapEncodeTiled");
  }
  return f(tm, dt, rank, addr, dims, strides, box, estrides, il, sw, l2, oob);
}

static CUtensorMap make_2d(void* ptr, CUtensorMapDataType dt, int elem_size,
                           int gmem_inner, int gmem_outer, int smem_inner,
                           int smem_outer, long gmem_outer_stride,
                           int swizzle_mode) {
  if (swizzle_mode != 0) smem_inner = swizzle_mode / elem_size;
  CUtensorMap tm;
  const cuuint64_t gdims[2] = {(cuuint64_t)gmem_inner, (cuuint64_t)gmem_outer};
  const cuuint32_t sdims[2] = {(cuuint32_t)smem_inner, (cuuint32_t)smem_outer};
  const cuuint64_t gstrides[1] = {(cuuint64_t)(gmem_outer_stride * elem_size)};
  const cuuint32_t estrides[2] = {1, 1};
  CUtensorMapSwizzle swizzle = swizzle_mode == 128  ? CU_TENSOR_MAP_SWIZZLE_128B
                               : swizzle_mode == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
                               : swizzle_mode == 32
                                   ? CU_TENSOR_MAP_SWIZZLE_32B
                                   : CU_TENSOR_MAP_SWIZZLE_NONE;
  CUresult r = enc_tiled(&tm, dt, 2, ptr, gdims, gstrides, sdims, estrides,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                         CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return tm;
}

static inline int align_up(int x, int a) { return (x + a - 1) / a * a; }

constexpr int NUM_HEADS = 32;
constexpr int HEAD_DIM = 128;
constexpr int BLOCK_Q = 4;  // 128 q*h rows per UMMA tile / 32 heads
constexpr int BLOCK_KV = 256;
constexpr int NUM_Q_STAGES = 1;  // one q-block per CTA
constexpr int NUM_KV_STAGES = 4;
constexpr int SPEC_THREADS = 128;
constexpr int MATH_THREADS = 256;  // 2 math warpgroups on SM100
constexpr int NUM_SMS = 148;       // B200

// Gather the compact hot sample's FP8 rows and their per-row FP32 scales in
// one launch. One warp owns one output row; its first eight lanes issue a
// single coalesced 128-byte vector copy and lane zero also copies the scale.
// A capped persistent grid avoids launching one CTA per hot row at N=8192.
template <typename IndexT>
__global__ void gather_hot_sample_litetopk_kernel(
    const uint4* __restrict__ k, const float* __restrict__ k_scale,
    const IndexT* __restrict__ idx, uint4* __restrict__ out_k,
    float* __restrict__ out_scale, int64_t seq_len, int hot_n) {
  constexpr int kWarpsPerBlock = 8;
  constexpr int kVecsPerRow = HEAD_DIM / static_cast<int>(sizeof(uint4));
  static_assert(kVecsPerRow == 8, "the fused gather requires D=128");
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int first_row = blockIdx.x * kWarpsPerBlock + warp;
  const int row_stride = gridDim.x * kWarpsPerBlock;

  for (int out_row = first_row; out_row < hot_n; out_row += row_stride) {
    int64_t src_row =
        lane == 0 ? static_cast<int64_t>(idx[out_row]) : int64_t{0};
    src_row = __shfl_sync(0xffffffffu, src_row, 0);

    // The host can validate dtype/layout/device without synchronizing,
    // but only the device sees index values. Assert instead of clamping:
    // clamping would silently duplicate a row and could tighten the DSA
    // threshold enough to reduce recall.
    if (src_row < 0 || src_row >= seq_len) {
      if (lane == 0) {
        assert(src_row >= 0 && src_row < seq_len);
      }
      continue;
    }
    if (lane < kVecsPerRow) {
      out_k[static_cast<int64_t>(out_row) * kVecsPerRow + lane] =
          k[src_row * kVecsPerRow + lane];
    }
    if (lane == 0) {
      out_scale[out_row] = k_scale[src_row];
    }
  }
}

void gather_hot_sample_litetopk_(torch::Tensor k, torch::Tensor k_scale,
                                 torch::Tensor idx, torch::Tensor out_k,
                                 torch::Tensor out_scale) {
  TORCH_CHECK(k.is_cuda() && k_scale.is_cuda() && idx.is_cuda() &&
                  out_k.is_cuda() && out_scale.is_cuda(),
              "k, k_scale, idx, out_k, and out_scale must be CUDA tensors");
  TORCH_CHECK(k.device() == k_scale.device() && k.device() == idx.device() &&
                  k.device() == out_k.device() &&
                  k.device() == out_scale.device(),
              "all fused hot-gather tensors must be on the same CUDA device");
  TORCH_CHECK(k.is_contiguous() && k_scale.is_contiguous() &&
                  idx.is_contiguous() && out_k.is_contiguous() &&
                  out_scale.is_contiguous(),
              "all fused hot-gather tensors must be contiguous");
  TORCH_CHECK(k.scalar_type() == torch::kFloat8_e4m3fn &&
                  out_k.scalar_type() == torch::kFloat8_e4m3fn,
              "k and out_k must be fp8_e4m3fn");
  TORCH_CHECK(k_scale.scalar_type() == torch::kFloat &&
                  out_scale.scalar_type() == torch::kFloat,
              "k_scale and out_scale must be fp32");
  TORCH_CHECK(
      idx.scalar_type() == torch::kLong || idx.scalar_type() == torch::kInt,
      "idx must be int64 or int32");
  TORCH_CHECK(k.dim() == 2 && k.size(1) == HEAD_DIM,
              "k must have shape [S, 128]");
  TORCH_CHECK(k_scale.dim() == 1 && k_scale.numel() == k.size(0),
              "k_scale must have shape [S]");
  TORCH_CHECK(idx.dim() == 1, "idx must have shape [hot_n]");
  TORCH_CHECK(out_k.dim() == 2 && out_k.size(0) == idx.numel() &&
                  out_k.size(1) == HEAD_DIM,
              "out_k must have shape [hot_n, 128]");
  TORCH_CHECK(out_scale.dim() == 1 && out_scale.numel() == idx.numel(),
              "out_scale must have shape [hot_n]");
  TORCH_CHECK(idx.numel() <= 8192,
              "fused hot gather supports at most 8192 indices");
  TORCH_CHECK(k.size(0) <= std::numeric_limits<int>::max(),
              "fused hot gather supports at most INT_MAX source rows");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(k.data_ptr()) % alignof(uint4) == 0 &&
          reinterpret_cast<uintptr_t>(out_k.data_ptr()) % alignof(uint4) == 0,
      "k and out_k must be 16-byte aligned");
  TORCH_CHECK(k.data_ptr() != out_k.data_ptr() &&
                  k_scale.data_ptr() != out_scale.data_ptr(),
              "fused hot gather does not support aliased input/output storage");

  const int hot_n = static_cast<int>(idx.numel());
  if (hot_n == 0) {
    return;
  }
  const c10::cuda::CUDAGuard device_guard(k.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  constexpr int kThreads = 256;
  constexpr int kRowsPerBlock = kThreads / 32;
  const int blocks =
      std::min((hot_n + kRowsPerBlock - 1) / kRowsPerBlock, NUM_SMS);
  if (idx.scalar_type() == torch::kLong) {
    gather_hot_sample_litetopk_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const uint4*>(k.data_ptr()), k_scale.data_ptr<float>(),
        idx.data_ptr<int64_t>(), reinterpret_cast<uint4*>(out_k.data_ptr()),
        out_scale.data_ptr<float>(), k.size(0), hot_n);
  } else {
    gather_hot_sample_litetopk_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const uint4*>(k.data_ptr()), k_scale.data_ptr<float>(),
        idx.data_ptr<int32_t>(), reinterpret_cast<uint4*>(out_k.data_ptr()),
        out_scale.data_ptr<float>(), k.size(0), hot_n);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void seed_bcount_kernel(const float* __restrict__ seed_val,
                                   int seed_k, const float* __restrict__ origin,
                                   const float* __restrict__ inv_delta,
                                   int32_t* __restrict__ bcount, int R,
                                   int NB) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= R) return;
  float o = origin[row];
  float inv = inv_delta[row];
  for (int i = tid; i < seed_k; i += blockDim.x) {
    float x = seed_val[(size_t)row * seed_k + i];
    int braw = static_cast<int>((x - o) * inv);
    int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
    if (braw < NB) atomicAdd(&bcount[(size_t)row * NB + b], 1);
  }
}

__global__ void refresh_threshold_from_bcount_kernel(
    int32_t* __restrict__ th_bucket, const int32_t* __restrict__ bcount, int R,
    int NB, int K) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= R) return;
  int old_th = th_bucket[row];
  int cum = 0;
  int new_th = old_th;
  bool found = false;
  for (int b = 0; b < NB; ++b) {
    cum += bcount[(size_t)row * NB + b];
    if (cum >= K) {
      new_th = b;
      found = true;
      break;
    }
  }
  if (found && new_th < old_th) th_bucket[row] = new_th;
}

// Fused seed/prep kernel (one block per row, all state in smem — borrows the
// vLLM top_k_per_row engineering): from the sample scores [Q, head] derive the
// per-row bucket params (origin, inv_delta), the initial gate threshold
// (bucket of the K-th best sample score), write the FULL sample histogram into
// bcount (a valid, conservative refresh base: counting genuine row elements
// can only tighten th safely), and emit every sample position with
// bucket <= th as initial candidates — a SUPERSET of the sample top-K, which
// the exact final select trims. Replaces: aminmax + torch.topk/radix seed +
// neg/contiguous copies + host seed copies + seed_bcount_kernel (~6 passes,
// ~10 launches) with 3 passes in 1 launch.
constexpr int kSeedThreads = 256;

__global__ void seed_prep_kernel(
    const float* __restrict__ slog, const int64_t slog_stride, const int head,
    const int NB, const int K,
    const float headroom,  // extend the bucket scale ABOVE the sample max by
                           // headroom*span (absolute, resolution-preserving
                           // when NB is scaled up with it): drifted scores
                           // land in real buckets instead of clamping to
                           // bucket 0 where refresh can never resolve them
    float* __restrict__ origin, float* __restrict__ inv_delta,
    int32_t* __restrict__ th_bucket, int32_t* __restrict__ cand_cnt) {
  constexpr int BT = kSeedThreads;
  constexpr int NSUB = 4;  // sub-histograms to spread smem atomic conflicts
  constexpr int kMaxRetainedHead = 8192;
  constexpr int kRetainVecs = kMaxRetainedHead / (BT * 4);
  const int row = gridDim.x - 1 - blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const float* srow = slog + (size_t)row * slog_stride;
  extern __shared__ int s_hist[];  // NSUB * NB ints

  // pass 1: min/max of the row's FINITE scores (vectorized). -inf appears
  // when the caller passes clean_logits=True full-row logits (dense-select
  // mode) for the out-of-range causal tail; it must not poison the range.
  __shared__ float s_mx[BT / 32];
  __shared__ float s_mn[BT / 32];
  float mx = -INFINITY, mn = INFINITY;
  const auto acc = [&](const float s) {
    if (isfinite(s)) {
      mx = fmaxf(mx, s);
      mn = fminf(mn, s);
    }
  };
  // HOT<=8192 and BT=256 needs at most eight float4 values (32 scores) per
  // thread. Keep them live across the CTA reduction so pass 2 does not
  // reread the score matrix. Missing tail lanes carry -inf and are ignored.
  static_assert(BT == 256, "HOT<=8192 retention requires BT=256");
  float4 retained[kRetainVecs];
  if (head == kMaxRetainedHead) {
#pragma unroll
    for (int it = 0; it < kRetainVecs; ++it) {
      const int j = tid * 4 + it * BT * 4;
      const float4 s4 = *reinterpret_cast<const float4*>(srow + j);
      retained[it] = s4;
      acc(s4.x);
      acc(s4.y);
      acc(s4.z);
      acc(s4.w);
    }
  } else {
#pragma unroll
    for (int it = 0; it < kRetainVecs; ++it) {
      const int j = tid * 4 + it * BT * 4;
      float4 s4 = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
      if (j + 3 < head) {
        s4 = *reinterpret_cast<const float4*>(srow + j);
      } else {
        if (j < head) s4.x = srow[j];
        if (j + 1 < head) s4.y = srow[j + 1];
        if (j + 2 < head) s4.z = srow[j + 2];
      }
      retained[it] = s4;
      acc(s4.x);
      acc(s4.y);
      acc(s4.z);
      acc(s4.w);
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, off));
  }
  if (lane == 0) {
    s_mx[tid >> 5] = mx;
    s_mn[tid >> 5] = mn;
  }
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int wgi = 1; wgi < BT / 32; ++wgi) {
      s_mx[0] = fmaxf(s_mx[0], s_mx[wgi]);
      s_mn[0] = fminf(s_mn[0], s_mn[wgi]);
    }
  }
  __syncthreads();
  float o = -s_mx[0];         // min over x = -score
  const float hi = -s_mn[0];  // max over x
  const float span = fmaxf(hi - o, 1e-20f);
  o -= headroom * span;  // forward (above-max) drift headroom
  float inv = (NB - 1) / (span * (1.0f + headroom));

  // pass 2: histogram in [o, inv] bucket space, NSUB sub-histograms to cut
  // smem atomic conflicts, vectorized loads.
  for (int b = tid; b < NSUB * NB; b += BT) s_hist[b] = 0;
  __syncthreads();
  int* my_hist = s_hist + (tid / (BT / NSUB)) * NB;
  const auto bucket_of = [&](const float s) -> int {
    const float x = -s;
    int b = static_cast<int>((x - o) * inv);
    return b < 0 ? 0 : (b > NB - 1 ? NB - 1 : b);
  };
#pragma unroll
  for (int it = 0; it < kRetainVecs; ++it) {
    const float4 s4 = retained[it];
    if (isfinite(s4.x)) atomicAdd(&my_hist[bucket_of(s4.x)], 1);
    if (isfinite(s4.y)) atomicAdd(&my_hist[bucket_of(s4.y)], 1);
    if (isfinite(s4.z)) atomicAdd(&my_hist[bucket_of(s4.z)], 1);
    if (isfinite(s4.w)) atomicAdd(&my_hist[bucket_of(s4.w)], 1);
  }
  __syncthreads();
  // merge sub-histograms into s_hist[0..NB)
  for (int b = tid; b < NB; b += BT) {
    int c = s_hist[b];
#pragma unroll
    for (int g = 1; g < NSUB; ++g) c += s_hist[g * NB + b];
    s_hist[b] = c;
  }
  __syncthreads();
  // Coarse K-th estimate, then REBUILD the scale over just the useful
  // range: [~K-th value (+1 coarse bucket slack) .. sample max + drift
  // headroom]. The bottom of [min,max] cannot contain the final threshold,
  // while headroom prevents scores above the sample maximum from collapsing
  // into bucket 0. The resulting scale concentrates bins around the useful
  // threshold range.
  // The production U16 contract uses emit_limit==0 and a single KV split.
  // Its scan covers the complete KV range and initializes the CTA-local
  // histogram itself, so writing Q*NB zeros to global memory is dead work.
  // Find the first histogram prefix that reaches K in parallel.  The old
  // single-thread walk serialized 256 dependent shared-memory loads while
  // the other 1023 threads waited.  NB <= BT gives every bin one owner;
  // the half-open prefix ranges are disjoint, so exactly one thread writes
  // the same threshold as the serial "first cumulative sum >= K" rule.
  __shared__ int s_th;
  __shared__ int s_wsum[BT / 32];
  if (tid == 0) s_th = NB - 1;
  const int h = (tid < NB) ? s_hist[tid] : 0;
  int x = h;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, off);
    if ((tid & 31) >= off) x += y;
  }
  if ((tid & 31) == 31) s_wsum[tid >> 5] = x;
  __syncthreads();
  int base = 0;
#pragma unroll
  for (int w = 0; w < BT / 32; ++w)
    if (w < (tid >> 5)) base += s_wsum[w];
  const int incl = base + x;
  const int excl = incl - h;
  if (tid < NB && excl < K && K <= incl) s_th = tid;
  __syncthreads();
  if (tid == 0) {
    th_bucket[row] = s_th;
    origin[row] = o;
    inv_delta[row] = inv;
  }
  __syncthreads();
  if (tid == 0) cand_cnt[row] = 0;
}

__device__ __forceinline__ uint32_t compact_enc_float(float v) {
  uint32_t bits = __float_as_uint(v);
  return (bits & 0x80000000u) ? (~bits) : (bits ^ 0x80000000u);
}

// Find the first radix digit whose inclusive histogram prefix reaches kfind.
//
// The old selector assigned this to tid==0, serializing 256 dependent shared
// loads while the other 255 threads waited. Warp 0 instead treats the radix
// as 32 groups of eight bins:
//   1. every lane sums one eight-bin group and warp-scans the group totals;
//   2. the winning group is broadcast, lanes 0..7 scan its eight bins.
//
// This keeps the exact "first prefix >= k" rule, including empty bins and
// ties, and needs no extra CTA barrier: callers already synchronize after the
// histogram fill and again after desired/kfind are published.
__device__ __forceinline__ void compact_find_radix_digit_warp0(
    const uint32_t* __restrict__ hist, uint32_t* __restrict__ desired,
    uint32_t* __restrict__ kfind, const uint32_t desired_base, const int shift,
    const int tid) {
  if (tid >= 32) return;
  constexpr unsigned FULL = 0xffffffffu;
  const int lane = tid;
  const int group_start = lane * 8;
  uint32_t group_count = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) group_count += hist[group_start + i];

  uint32_t group_inclusive = group_count;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const uint32_t other = __shfl_up_sync(FULL, group_inclusive, offset);
    if (lane >= offset) group_inclusive += other;
  }

  const uint32_t target = *kfind;
  if (target == 0u) return;
  const unsigned group_mask = __ballot_sync(FULL, group_inclusive >= target);
  // Match the serial fallback exactly for an underfilled histogram: leave
  // desired/kfind unchanged instead of deriving an invalid -1 group.
  if (group_mask == 0u) return;
  const int winning_group = __ffs(group_mask) - 1;
  const uint32_t group_before =
      __shfl_sync(FULL, group_inclusive - group_count, winning_group);

  const uint32_t digit_count = lane < 8 ? hist[winning_group * 8 + lane] : 0u;
  uint32_t digit_inclusive = digit_count;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const uint32_t other = __shfl_up_sync(FULL, digit_inclusive, offset);
    if (lane >= offset) digit_inclusive += other;
  }
  const unsigned digit_mask =
      __ballot_sync(FULL, lane < 8 && group_before + digit_inclusive >= target);
  if (digit_mask == 0u) return;
  const int winning_lane = __ffs(digit_mask) - 1;
  const uint32_t digit_before =
      group_before +
      __shfl_sync(FULL, digit_inclusive - digit_count, winning_lane);

  if (lane == 0) {
    const uint32_t digit =
        static_cast<uint32_t>(winning_group * 8 + winning_lane);
    *desired = desired_base | (digit << static_cast<uint32_t>(shift));
    *kfind = target - digit_before;
  }
}

// Deferred overflow telemetry used to be expressed as
//
//   stack({cand_cnt.max(), cand_cnt.float().mean().int()})
//
// which launches five ATen kernels before the asynchronous eight-byte D2H
// copy.  Candidate counts are already contiguous int32 values, so one CTA is
// enough for all supported Q (8128/8192).  Accumulate in signed 64 bits:
// even the supported 1M-token upper bound sums to about 8.6e9 and therefore
// cannot be represented by int32.
//
// The second output is the exact integer mean, truncated toward zero.  ATen's
// FP32 tree reduction can very rarely round that value across an integer
// boundary.  The production HOTONLY path marks every successful probe as
// strided, where the mean is telemetry-only and AUTO state is not updated;
// Python retains the old ATen expression for any future non-strided path.
__global__ void cand_count_stats_litetopk_kernel(
    const int32_t* __restrict__ cand_cnt, int count,
    int32_t* __restrict__ stats) {
  constexpr int kThreads = 256;
  constexpr int kWarps = kThreads / 32;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  int32_t local_max = (-2147483647 - 1);
  int64_t local_sum = 0;
  for (int i = tid; i < count; i += kThreads) {
    const int32_t value = cand_cnt[i];
    local_max = max(local_max, value);
    local_sum += static_cast<int64_t>(value);
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_max =
        max(local_max, __shfl_down_sync(0xffffffffu, local_max, offset));
    local_sum += __shfl_down_sync(0xffffffffu, local_sum, offset);
  }

  __shared__ int32_t warp_max[kWarps];
  __shared__ int64_t warp_sum[kWarps];
  if (lane == 0) {
    warp_max[warp] = local_max;
    warp_sum[warp] = local_sum;
  }
  __syncthreads();

  if (warp == 0) {
    int32_t block_max = lane < kWarps ? warp_max[lane] : (-2147483647 - 1);
    int64_t block_sum = lane < kWarps ? warp_sum[lane] : 0;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      block_max =
          max(block_max, __shfl_down_sync(0xffffffffu, block_max, offset));
      block_sum += __shfl_down_sync(0xffffffffu, block_sum, offset);
    }
    if (lane == 0) {
      stats[0] = block_max;
      stats[1] = static_cast<int32_t>(block_sum / static_cast<int64_t>(count));
    }
  }
}

// Selector-fused carry votes have a much smaller value domain than their
// corpus-index domain.  A selected corpus position can receive at most one
// vote from each sampled query row, hence max_vote=ceil(Q/row_stride)<=8192
// while the histogram itself can contain up to 1M positions.  Exploit that
// bounded domain directly instead of sending the 1M int32 values through a
// general-purpose topk.
//
// The operation is deliberately split into exactly two kernels:
//
//   1. Each CTA builds a local count-of-counts histogram for a contiguous
//      8192-position tile and writes it to caller-owned int16 partial storage.
//      The CTA that receives the final completion ticket reduces those
//      partials, finds the exact vote threshold, resolves threshold ties by
//      ascending corpus index, and publishes deterministic per-CTA offsets.
//   2. CTAs stably compact the selected indices to int64 output while clearing
//      every live vote.  The output is ascending by corpus index, which is a
//      friendlier order for the next index_select than vote-sorted output.
//
// The partial workspace and state are exclusive to one ordered stream.  The
// final CTA resets the completion ticket before kernel exit, so the same
// workspace can be reused by the next call without a memset or host sync.
constexpr int kCarryTileItems = 8192;
constexpr int kCarryMaxItems = 1 << 20;
constexpr int kCarryMaxK = 8192;
constexpr int kCarryMaxBlocks =
    (kCarryMaxItems + kCarryTileItems - 1) / kCarryTileItems;
constexpr int kCarryThreads = 256;
constexpr int kCarryWarps = kCarryThreads / 32;

enum CarryStateOffset : int {
  kCarryTicket = 0,
  kCarryThreshold = 1,
  kCarryTieBlock = 2,
  kCarryTieTake = 3,
  kCarryOutK = 4,
  kCarryNumBlocks = 5,
  kCarryBlockOffsets = 6,
};
constexpr int kCarryStateInts = kCarryBlockOffsets + kCarryMaxBlocks + 1;

__device__ __forceinline__ int carry_warp_sum(int value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__global__ void carry_votes_plan_litetopk_kernel(
    const int32_t* __restrict__ votes, int count, int min_index, int out_k,
    int max_vote, volatile int16_t* __restrict__ partial, int partial_stride,
    int32_t* __restrict__ state) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int block = blockIdx.x;
  const int begin = block * kCarryTileItems;
  const int end = min(begin + kCarryTileItems, count);
  const int bins = max_vote + 1;

  extern __shared__ uint32_t s_freq[];
  __shared__ int s_warp_sum[kCarryWarps];
  __shared__ int s_last;
  __shared__ int s_scan_base;
  __shared__ int s_found;
  __shared__ int s_threshold;
  __shared__ int s_count_gt;
  __shared__ int s_tie_block;
  __shared__ int s_tie_take;
  __shared__ int s_block_count[kCarryMaxBlocks];

  for (int bin = tid; bin < bins; bin += kCarryThreads) {
    s_freq[bin] = 0;
  }
  __syncthreads();

  // Zero votes dominate most corpora. Count them in registers and reduce
  // once instead of serializing every zero through one shared atomic.
  int local_zero = 0;
  for (int index = begin + tid; index < end; index += kCarryThreads) {
    if (index < min_index) {
      continue;
    }
    int value = votes[index];
    // The selector emits unique winners per sampled row, so this clamp is
    // unreachable under the public ABI. Keep release builds memory-safe
    // if an upstream invariant is violated.
    value = value < 0 ? 0 : (value > max_vote ? max_vote : value);
    if (value == 0) {
      ++local_zero;
    } else {
      atomicAdd(&s_freq[value], 1u);
    }
  }
  local_zero = carry_warp_sum(local_zero);
  if (lane == 0) {
    s_warp_sum[warp] = local_zero;
  }
  __syncthreads();
  if (warp == 0) {
    int value = lane < kCarryWarps ? s_warp_sum[lane] : 0;
    value = carry_warp_sum(value);
    if (lane == 0) {
      s_freq[0] = static_cast<uint32_t>(value);
    }
  }
  __syncthreads();

  volatile int16_t* block_partial =
      partial + static_cast<size_t>(block) * partial_stride;
  for (int bin = tid; bin < bins; bin += kCarryThreads) {
    // A CTA owns at most 8192 positions, safely inside signed int16.
    block_partial[bin] = static_cast<int16_t>(s_freq[bin]);
  }
  // Every thread publishes its own global stores. A fence in tid0 alone
  // would not release the other 255 writers before the completion ticket.
  __threadfence();
  __syncthreads();

  // CUDA's canonical "last block" reduction pattern. No CTA spins: every
  // non-last block exits, while the last ticket holder sees all partial
  // writes made visible before the atomic increment.
  if (tid == 0) {
    const int old = atomicAdd(&state[kCarryTicket], 1);
    s_last = old == gridDim.x - 1;
  }
  __syncthreads();
  if (!s_last) {
    return;
  }

  for (int bin = tid; bin < bins; bin += kCarryThreads) {
    int total = 0;
    for (int source_block = 0; source_block < gridDim.x; ++source_block) {
      total += static_cast<int>(
          partial[static_cast<size_t>(source_block) * partial_stride + bin]);
    }
    s_freq[bin] = static_cast<uint32_t>(total);
  }
  __syncthreads();

  // Descending 256-bin tiles. This is the seed-prep parallel prefix in the
  // opposite direction, extended to the dynamic [0,max_vote] domain.
  if (tid == 0) {
    s_scan_base = 0;
    s_found = 0;
    s_threshold = 0;
    s_count_gt = 0;
  }
  __syncthreads();
  for (int tile = 0; tile < bins; tile += kCarryThreads) {
    const int bin = max_vote - tile - tid;
    const int count_here = bin >= 0 ? static_cast<int>(s_freq[bin]) : 0;
    int inclusive = count_here;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
      if (lane >= offset) {
        inclusive += other;
      }
    }
    if (lane == 31) {
      s_warp_sum[warp] = inclusive;
    }
    __syncthreads();
    int warp_base = 0;
#pragma unroll
    for (int source_warp = 0; source_warp < kCarryWarps; ++source_warp) {
      if (source_warp < warp) {
        warp_base += s_warp_sum[source_warp];
      }
    }
    const int exclusive = s_scan_base + warp_base + inclusive - count_here;
    const int inclusive_global = exclusive + count_here;
    if (bin >= 0 && exclusive < out_k && out_k <= inclusive_global) {
      s_threshold = bin;
      s_count_gt = exclusive;
      s_found = 1;
    }
    __syncthreads();
    if (s_found) {
      break;
    }
    if (tid == 0) {
      int tile_total = 0;
#pragma unroll
      for (int source_warp = 0; source_warp < kCarryWarps; ++source_warp) {
        tile_total += s_warp_sum[source_warp];
      }
      s_scan_base += tile_total;
    }
    __syncthreads();
  }

  if (tid == 0) {
    const int need_equal = out_k - s_count_gt;
    int equal_before = 0;
    s_tie_block = gridDim.x - 1;
    s_tie_take = 0;
    for (int source_block = 0; source_block < gridDim.x; ++source_block) {
      const int equal_here = static_cast<int>(
          partial[static_cast<size_t>(source_block) * partial_stride +
                  s_threshold]);
      if (equal_before < need_equal &&
          need_equal <= equal_before + equal_here) {
        s_tie_block = source_block;
        s_tie_take = need_equal - equal_before;
        break;
      }
      equal_before += equal_here;
    }
  }
  __syncthreads();

  // Compute each block's exact stable-output size. Warps read one partial
  // row at a time so the second partial pass remains coalesced.
  for (int source_block = warp; source_block < gridDim.x;
       source_block += kCarryWarps) {
    int selected = 0;
    for (int bin = s_threshold + 1 + lane; bin < bins; bin += 32) {
      selected += static_cast<int>(
          partial[static_cast<size_t>(source_block) * partial_stride + bin]);
    }
    selected = carry_warp_sum(selected);
    if (lane == 0) {
      int equal_take = 0;
      if (source_block < s_tie_block) {
        equal_take = static_cast<int>(
            partial[static_cast<size_t>(source_block) * partial_stride +
                    s_threshold]);
      } else if (source_block == s_tie_block) {
        equal_take = s_tie_take;
      }
      s_block_count[source_block] = selected + equal_take;
    }
  }
  __syncthreads();

  if (tid == 0) {
    int offset = 0;
    for (int source_block = 0; source_block < gridDim.x; ++source_block) {
      state[kCarryBlockOffsets + source_block] = offset;
      offset += s_block_count[source_block];
    }
    state[kCarryBlockOffsets + gridDim.x] = offset;
    state[kCarryThreshold] = s_threshold;
    state[kCarryTieBlock] = s_tie_block;
    state[kCarryTieTake] = s_tie_take;
    state[kCarryOutK] = out_k;
    state[kCarryNumBlocks] = gridDim.x;
    __threadfence();
    atomicExch(&state[kCarryTicket], 0);
  }
}

__global__ void carry_votes_emit_reset_litetopk_kernel(
    int32_t* __restrict__ votes, int count, int min_index, int max_vote,
    int64_t* __restrict__ out_idx, const int32_t* __restrict__ state) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int block = blockIdx.x;
  const int begin = block * kCarryTileItems;
  const int threshold = state[kCarryThreshold];
  const int tie_block = state[kCarryTieBlock];
  const int tie_take = state[kCarryTieTake];
  const int output_base = state[kCarryBlockOffsets + block];

  __shared__ int s_warp_count[kCarryWarps];
  __shared__ int s_warp_prefix[kCarryWarps];
  __shared__ int s_tile_output_base;
  __shared__ int s_tie_seen;
  __shared__ int s_tile_total;
  if (tid == 0) {
    s_tile_output_base = 0;
    s_tie_seen = 0;
  }
  __syncthreads();

  constexpr unsigned kFullMask = 0xffffffffu;
  const unsigned lane_mask = lane == 0 ? 0u : ((1u << lane) - 1u);
  for (int tile = 0; tile < kCarryTileItems; tile += kCarryThreads) {
    const int index = begin + tile + tid;
    const bool valid = index < count;
    const int raw_value = valid ? votes[index] : 0;
    const int value =
        raw_value < 0 ? 0 : (raw_value > max_vote ? max_vote : raw_value);
    if (valid) {
      votes[index] = 0;
    }
    const bool eligible = valid && index >= min_index;
    const bool is_equal = eligible && value == threshold;

    bool take_equal = is_equal && block < tie_block;
    if (block == tie_block) {
      const unsigned equal_mask = __ballot_sync(kFullMask, is_equal);
      if (lane == 0) {
        s_warp_count[warp] = __popc(equal_mask);
      }
      __syncthreads();
      if (tid == 0) {
        int prefix = 0;
        for (int source_warp = 0; source_warp < kCarryWarps; ++source_warp) {
          s_warp_prefix[source_warp] = prefix;
          prefix += s_warp_count[source_warp];
        }
        s_tile_total = prefix;
      }
      __syncthreads();
      const int equal_rank =
          s_tie_seen + s_warp_prefix[warp] + __popc(equal_mask & lane_mask);
      take_equal = is_equal && equal_rank < tie_take;
      __syncthreads();
      if (tid == 0) {
        s_tie_seen += s_tile_total;
      }
      __syncthreads();
    }

    const bool selected = eligible && (value > threshold || take_equal);
    const unsigned selected_mask = __ballot_sync(kFullMask, selected);
    if (lane == 0) {
      s_warp_count[warp] = __popc(selected_mask);
    }
    __syncthreads();
    if (tid == 0) {
      int prefix = 0;
      for (int source_warp = 0; source_warp < kCarryWarps; ++source_warp) {
        s_warp_prefix[source_warp] = prefix;
        prefix += s_warp_count[source_warp];
      }
      s_tile_total = prefix;
    }
    __syncthreads();
    if (selected) {
      const int local_rank =
          s_warp_prefix[warp] + __popc(selected_mask & lane_mask);
      out_idx[output_base + s_tile_output_base + local_rank] =
          static_cast<int64_t>(index);
    }
    __syncthreads();
    if (tid == 0) {
      s_tile_output_base += s_tile_total;
    }
    __syncthreads();
  }
}

// DSA specialization of the GitHub FlashTopK boundary-bucket strategy.
//
// The generic selector above logically restricts the radix set to bucket
// `th`, but every radix pass still rereads all `n` candidates and filters
// them. Sparse refresh normally leaves:
//
//     count(bucket < th) < K <= count(bucket <= th).
//
// Make that saving physical: one tiled pass writes bucket<th directly to the
// final output and compacts bucket==th in-place at the front of the candidate
// buffer. The four radix passes then read only that compact boundary. A tile
// is loaded completely before any write, and the compacted prefix can never
// extend beyond the end of the processed tile, so aliasing input/output is
// race-free and needs no second multi-GiB candidate slab.
//
// The two fallback modes mirror compact_topk_min_thr_litetopk_kernel:
//   * threshold too loose (lt >= K): compact/radix the lt set;
//   * threshold underfilled: compact/radix every finite buffered candidate.
__device__ __forceinline__ void dsa_litetopk_accumulate_inplace_votes(
    const int32_t* __restrict__ out_idx, int K, int tid, int threads,
    int32_t* __restrict__ votes, int votes_len, int row, int row_stride) {
  // The selector output is always produced for every row. Carry voting is
  // auxiliary, so sample a deterministic phase (row % stride == 0). Keep
  // this block-uniform return ahead of the barrier: unsampled CTAs pay no
  // vote-side synchronization or atomics.
  if (votes == nullptr || votes_len <= 0 || (row & (row_stride - 1)) != 0) {
    return;
  }
  // Every call site is a block-uniform exit. Wait until all winner stores
  // are visible, then count this row's K outputs while they are still hot.
  __syncthreads();
  for (int j = tid; j < K; j += threads) {
    int32_t col = out_idx[j];
    col = col < 0 ? 0 : (col >= votes_len ? votes_len - 1 : col);
    atomicAdd(votes + col, 1);
  }
}

__global__ void compact_topk_min_thr_inplace_idx_out_litetopk_kernel(
    CandidateValue* __restrict__ val, int32_t* __restrict__ idx,
    const int32_t* __restrict__ cnt, const int32_t* __restrict__ th_in,
    const int32_t* __restrict__ boundary_meta, int R, int CAP, int K, int NB,
    int32_t* __restrict__ out_idx, int32_t* __restrict__ votes, int votes_len,
    int vote_row_stride) {
  constexpr int BT = 256;
  constexpr int RADIX = 256;
  const unsigned FULL = 0xffffffffu;
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const unsigned lane_mask = lane == 0 ? 0u : ((1u << lane) - 1u);
  if (row >= R) return;

  CandidateValue* vrow = val + static_cast<size_t>(row) * CAP;
  int32_t* irow = idx + static_cast<size_t>(row) * CAP;
  int32_t* oi = out_idx + static_cast<size_t>(row) * K;
  const int raw_n = cnt[row];
  int n = raw_n;
  if (n > CAP) n = CAP;
  if (n < 0) n = 0;
  if (n == 0) {
    for (int j = tid; j < K; j += BT) {
      oi[j] = 0;
    }
    dsa_litetopk_accumulate_inplace_votes(oi, K, tid, BT, votes, votes_len, row,
                                          vote_row_stride);
    return;
  }

  const int th = th_in[row];
  // The packed boundary remains bit-exact only above its compile-time
  // lower bound.  Fail loudly instead of silently turning the exact path
  // into an approximation.
  constexpr int kPackedExactThreshold = 1;
  if (th < kPackedExactThreshold) {
    asm volatile("trap;");
    return;
  }

  // mode 0: standard boundary path; 1: loose threshold; 2: underfilled.
  __shared__ int s_count_lt;
  __shared__ int s_count_eq;
  __shared__ int s_count_valid;
  __shared__ int s_have_boundary_meta;
  __shared__ int s_mode;
  __shared__ int s_k_target;
  constexpr int BOUNDARY_SMEM_CAP = 256;
  __shared__ uint32_t s_boundary_val[BOUNDARY_SMEM_CAP];
  __shared__ int32_t s_boundary_idx[BOUNDARY_SMEM_CAP];
  __shared__ int s_fast_lt_cursor;
  __shared__ int s_fast_eq_cursor;
  __shared__ uint32_t s_fast_hist[RADIX];
  __shared__ uint32_t s_fast_desired;
  __shared__ uint32_t s_fast_kfind;
  __shared__ int s_fast_pivot_lt;
  __shared__ int s_fast_write_lt;
  __shared__ int s_fast_write_eq;
  if (tid == 0) {
    const int32_t* meta = boundary_meta + static_cast<size_t>(row) * NB;
    const int tag = meta[0];
    const int meta_th = ~tag;
    const int meta_lt = meta[1];
    const int meta_eq = meta[2];
    const int meta_need = K - meta_lt;
    s_have_boundary_meta = tag < 0 && meta_th == th && meta_th >= 0 &&
                           meta_th < NB && raw_n >= 0 && raw_n <= CAP &&
                           meta_lt >= 0 && meta_eq >= 0 && meta_lt < K &&
                           meta_need > 0 && meta_need <= meta_eq &&
                           meta_lt + meta_eq <= n;
    s_count_lt = s_have_boundary_meta ? meta_lt : 0;
    s_count_eq = s_have_boundary_meta ? meta_eq : 0;
    s_count_valid = 0;
  }
  __syncthreads();

  // The six-byte representation is conditionally exact for the certified
  // sparse-refresh boundary path.  A missing certificate could require a
  // top-K selection within collapsed bucket 0, so it is an explicit error.
  if (!s_have_boundary_meta) {
    asm volatile("trap;");
    return;
  }

  if (!s_have_boundary_meta) {
    int local_lt = 0;
    int local_eq = 0;
    int local_valid = 0;
    for (int j = tid; j < n; j += BT) {
      const float v = dsa_litetopk::candidate_decode_score(vrow[j], irow[j]);
      if (!isfinite(v)) continue;
      ++local_valid;
      int braw = static_cast<int>(v);
      const int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
      local_lt += b < th;
      local_eq += b == th;
    }
    atomicAdd(&s_count_lt, local_lt);
    atomicAdd(&s_count_eq, local_eq);
    atomicAdd(&s_count_valid, local_valid);
  }
  __syncthreads();
  if (tid == 0) {
    const int need = K - s_count_lt;
    if (s_count_lt < K && need > 0 && need <= s_count_eq) {
      s_mode = 0;
      s_k_target = need;
    } else if (s_count_lt >= K) {
      s_mode = 1;
      s_k_target = K;
    } else {
      s_mode = 2;
      s_k_target = min(K, s_count_valid);
    }
  }
  __syncthreads();

  // Production sparse-refresh distribution (Q=8192, K=2048):
  // boundary E averages ~97 candidates, P99 ~163, max 212 on the 1M
  // corpus. Keep that boundary entirely in shared memory. Unlike the
  // generic in-place path below, this pass has no aliasing stores, so
  // warp-local shared-atomic reservations need no CTA barrier per tile.
  if (s_have_boundary_meta && s_count_eq <= BOUNDARY_SMEM_CAP) {
    if (tid == 0) {
      s_fast_lt_cursor = 0;
      s_fast_eq_cursor = 0;
    }
    __syncthreads();
    for (int tile = 0; tile < n; tile += BT) {
      const int j = tile + tid;
      uint32_t score_code = 0u;
      bool valid = false;
      if (j < n) {
        score_code = dsa_litetopk::candidate_load_score_code(vrow[j], irow[j]);
        valid = true;
      }
      // Writers canonicalize negative bucket-0 values to code zero.
      // Positive FP32 high24 codes remain monotonic, and truncating the
      // low byte cannot cross an exactly represented integer edge.
      const uint32_t th_code = __float_as_uint(static_cast<float>(th)) >> 8;
      const uint32_t next_th_code =
          __float_as_uint(static_cast<float>(th + 1)) >> 8;
      const bool is_lt = valid && score_code < th_code;
      const bool is_eq =
          valid && score_code >= th_code && score_code < next_th_code;
      const unsigned lt_mask = __ballot_sync(FULL, is_lt);
      const unsigned eq_mask = __ballot_sync(FULL, is_eq);
      int warp_lt_base = 0;
      int warp_eq_base = 0;
      if (lane == 0) {
        const int lt_count = __popc(lt_mask);
        const int eq_count = __popc(eq_mask);
        if (lt_count != 0)
          warp_lt_base = atomicAdd(&s_fast_lt_cursor, lt_count);
        if (eq_count != 0)
          warp_eq_base = atomicAdd(&s_fast_eq_cursor, eq_count);
      }
      warp_lt_base = __shfl_sync(FULL, warp_lt_base, 0);
      warp_eq_base = __shfl_sync(FULL, warp_eq_base, 0);

      if (is_lt) {
        const int pos = warp_lt_base + __popc(lt_mask & lane_mask);
        if (pos < K) {
          const int32_t raw_idx = irow[j];
          oi[pos] = dsa_litetopk::candidate_decode_index(raw_idx);
        }
      }
      if (is_eq) {
        const int pos = warp_eq_base + __popc(eq_mask & lane_mask);
        if (pos < BOUNDARY_SMEM_CAP) {
          s_boundary_val[pos] = score_code;
          s_boundary_idx[pos] = dsa_litetopk::candidate_decode_index(irow[j]);
        }
      }
    }
    __syncthreads();

    const int boundary_n = s_fast_eq_cursor;
    const int output_base = s_fast_lt_cursor;
    const int k_target = K - output_base;
    if (boundary_n == k_target) {
      for (int j = tid; j < boundary_n; j += BT) {
        oi[output_base + j] = s_boundary_idx[j];
      }
      dsa_litetopk_accumulate_inplace_votes(oi, K, tid, BT, votes, votes_len,
                                            row, vote_row_stride);
      return;
    }

    if (tid == 0) {
      // Boundary values lie in [th, th+1), so their positive FP32
      // high byte is fixed. Select only the remaining 16 code bits.
      s_fast_desired = s_boundary_val[0] & 0xff0000u;
      s_fast_kfind = static_cast<uint32_t>(k_target);
    }
    __syncthreads();
    uint32_t fast_mask = 0u;
#pragma unroll
    // The high byte was fixed above. The remaining 16 bits preserve
    // high24 ordering inside the certified boundary bucket.
    for (int pass = 0; pass < 2; ++pass) {
      const int shift = 8 - pass * 8;
      s_fast_hist[tid] = 0;
      __syncthreads();
      const uint32_t desired = s_fast_desired;
      if (tid < boundary_n) {
        const uint32_t encoded = s_boundary_val[tid];
        if ((encoded & fast_mask) == (desired & fast_mask)) {
          atomicAdd(&s_fast_hist[(encoded >> shift) & 0xffu], 1u);
        }
      }
      __syncthreads();
      compact_find_radix_digit_warp0(s_fast_hist, &s_fast_desired,
                                     &s_fast_kfind, desired, shift, tid);
      __syncthreads();
      fast_mask |= 0xffu << shift;
    }
    const uint32_t pivot = s_fast_desired;

    if (tid == 0) {
      s_fast_pivot_lt = 0;
      s_fast_write_lt = 0;
      s_fast_write_eq = 0;
    }
    __syncthreads();
    if (tid < boundary_n && s_boundary_val[tid] < pivot)
      atomicAdd(&s_fast_pivot_lt, 1);
    __syncthreads();
    const int eq_take = max(k_target - s_fast_pivot_lt, 0);
    if (tid < boundary_n) {
      const uint32_t encoded = s_boundary_val[tid];
      if (encoded < pivot) {
        const int pos = atomicAdd(&s_fast_write_lt, 1);
        if (pos < k_target) {
          oi[output_base + pos] = s_boundary_idx[tid];
        }
      } else if (encoded == pivot) {
        const int equal_rank = atomicAdd(&s_fast_write_eq, 1);
        if (equal_rank < eq_take) {
          const int pos = output_base + s_fast_pivot_lt + equal_rank;
          if (pos < K) {
            oi[pos] = s_boundary_idx[tid];
          }
        }
      }
    }
    dsa_litetopk_accumulate_inplace_votes(oi, K, tid, BT, votes, votes_len, row,
                                          vote_row_stride);
    return;
  }

  // Tiled, alias-safe in-place compaction. In the standard mode, lt
  // candidates bypass the compact buffer and go straight to output.
  __shared__ int s_compact_base;
  __shared__ int s_direct_base;
  if (tid == 0) {
    s_compact_base = 0;
    s_direct_base = 0;
  }
  __syncthreads();

  for (int tile = 0; tile < n; tile += BT) {
    const int j = tile + tid;
    CandidateValue raw_value{};
    float v = INFINITY;
    int32_t raw_idx = 0;
    int b = NB;
    bool valid = false;
    if (j < n) {
      raw_value = vrow[j];
      raw_idx = irow[j];
      v = dsa_litetopk::candidate_decode_score(raw_value, raw_idx);
      valid = isfinite(v);
      if (valid) {
        int braw = static_cast<int>(v);
        b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
      }
    }

    const bool is_lt = valid && b < th;
    bool selected = false;
    if (s_mode == 0)
      selected = valid && b == th;
    else if (s_mode == 1)
      selected = is_lt;
    else
      selected = valid;
    const bool direct = s_mode == 0 && is_lt;

    const unsigned selected_mask = __ballot_sync(FULL, selected);
    const unsigned direct_mask = __ballot_sync(FULL, direct);
    int warp_compact_base = 0;
    int warp_direct_base = 0;
    if (lane == 0) {
      const int selected_count = __popc(selected_mask);
      const int direct_count = __popc(direct_mask);
      if (selected_count != 0)
        warp_compact_base = atomicAdd(&s_compact_base, selected_count);
      if (direct_count != 0)
        warp_direct_base = atomicAdd(&s_direct_base, direct_count);
    }
    warp_compact_base = __shfl_sync(FULL, warp_compact_base, 0);
    warp_direct_base = __shfl_sync(FULL, warp_direct_base, 0);

    // One CTA barrier per tile is sufficient for alias safety: every
    // source element is already in a register and every warp has reserved
    // its compact ranges before any in-place store starts. Compact output
    // never reaches the next (unread) tile.
    __syncthreads();

    if (direct) {
      const int pos = warp_direct_base + __popc(direct_mask & lane_mask);
      if (pos < K) {
        oi[pos] = dsa_litetopk::candidate_decode_index(raw_idx);
      }
    }
    if (selected) {
      const int pos = warp_compact_base + __popc(selected_mask & lane_mask);
      vrow[pos] = raw_value;
      irow[pos] = raw_idx;
    }
  }
  __syncthreads();

  const int selected_n = s_compact_base;
  const int output_base = s_mode == 0 ? s_count_lt : 0;
  const int k_target = s_k_target;

  // Exact fallback with fewer than K finite buffered candidates.
  if (s_mode == 2 && selected_n <= K) {
    for (int j = tid; j < selected_n; j += BT) {
      oi[j] = dsa_litetopk::candidate_decode_index(irow[j]);
    }
    for (int j = selected_n + tid; j < K; j += BT) {
      oi[j] = 0;
    }
    dsa_litetopk_accumulate_inplace_votes(oi, K, tid, BT, votes, votes_len, row,
                                          vote_row_stride);
    return;
  }
  if (selected_n == 0 || k_target == 0) {
    for (int j = output_base + tid; j < K; j += BT) {
      oi[j] = 0;
    }
    dsa_litetopk_accumulate_inplace_votes(oi, K, tid, BT, votes, votes_len, row,
                                          vote_row_stride);
    return;
  }

  // Radix-select only the compacted set. In the expected sparse-refresh
  // case this is exactly the threshold bucket and k_target == K-count_lt.
  __shared__ uint32_t hist[RADIX];
  __shared__ uint32_t desired;
  __shared__ uint32_t kfind;
  __shared__ int s_pivot_lt;
  __shared__ int s_write_lt;
  __shared__ int s_write_eq;
  if (tid == 0) {
    desired = 0u;
    kfind = static_cast<uint32_t>(k_target);
  }
  __syncthreads();

  uint32_t mask = 0u;
#pragma unroll
  for (int pass = 0; pass < 4; ++pass) {
    const int shift = 24 - pass * 8;
    hist[tid] = 0;
    __syncthreads();
    const uint32_t d = desired;
    for (int j = tid; j < selected_n; j += BT) {
      const uint32_t e = compact_enc_float(
          dsa_litetopk::candidate_decode_score(vrow[j], irow[j]));
      if ((e & mask) == (d & mask)) atomicAdd(&hist[(e >> shift) & 0xffu], 1u);
    }
    __syncthreads();
    compact_find_radix_digit_warp0(hist, &desired, &kfind, d, shift, tid);
    __syncthreads();
    mask |= 0xffu << shift;
  }
  const uint32_t pivot = desired;

  if (tid == 0) {
    s_pivot_lt = 0;
    s_write_lt = 0;
    s_write_eq = 0;
  }
  __syncthreads();
  int pivot_lt = 0;
  for (int j = tid; j < selected_n; j += BT)
    pivot_lt += compact_enc_float(dsa_litetopk::candidate_decode_score(
                    vrow[j], irow[j])) < pivot;
  atomicAdd(&s_pivot_lt, pivot_lt);
  __syncthreads();
  const int eq_take = max(k_target - s_pivot_lt, 0);

  for (int j = tid; j < selected_n; j += BT) {
    const float v = dsa_litetopk::candidate_decode_score(vrow[j], irow[j]);
    const uint32_t e = compact_enc_float(v);
    if (e < pivot) {
      const int w = atomicAdd(&s_write_lt, 1);
      const int pos = output_base + w;
      if (pos < K) {
        oi[pos] = dsa_litetopk::candidate_decode_index(irow[j]);
      }
    } else if (e == pivot) {
      const int equal_rank = atomicAdd(&s_write_eq, 1);
      if (equal_rank < eq_take) {
        const int pos = output_base + s_pivot_lt + equal_rank;
        if (pos < K) {
          oi[pos] = dsa_litetopk::candidate_decode_index(irow[j]);
        }
      }
    }
  }
  dsa_litetopk_accumulate_inplace_votes(oi, K, tid, BT, votes, votes_len, row,
                                        vote_row_stride);
}

static int compute_smem_bytes() {
  const int esz_fp8 = 1, esz_f32 = 4;
  const int smem_q = BLOCK_Q * NUM_HEADS * HEAD_DIM * esz_fp8;
  const int smem_w = BLOCK_Q * NUM_HEADS * esz_f32;
  const int smem_kv = BLOCK_KV * HEAD_DIM * esz_fp8;
  const int smem_ks = align_up(BLOCK_KV * esz_f32, 512);
  const int num_barriers = NUM_Q_STAGES * 2 + NUM_KV_STAGES * 2 +
                           (MATH_THREADS / 128) * dsa_litetopk::kUmmaStages * 2;
  const int smem_barriers = num_barriers * 8;
  const int smem_slots =
      4 * (int)sizeof(uint32_t);  // tmem ptr + daemon mailboxes
  constexpr int emit_record_bytes = (int)sizeof(uint32_t);
  const int smem_warpq = (MATH_THREADS / 32) * BLOCK_Q *
                         ((int)sizeof(int32_t) + dsa_litetopk::kEmitLaneSlots *
                                                     32 * emit_record_bytes);
  const int smem_hist =
      BLOCK_Q * 256 * (int)sizeof(int32_t);  // per-CTA refresh
                                             // histogram (NB<=256)
  return NUM_Q_STAGES * smem_q + NUM_Q_STAGES * smem_w +
         NUM_KV_STAGES * smem_kv + NUM_KV_STAGES * smem_ks + smem_barriers +
         smem_slots + smem_warpq + smem_hist;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mqa_logits_dsa_litetopk(
    torch::Tensor q, torch::Tensor kv, torch::Tensor kv_scales,
    torch::Tensor weights, torch::Tensor cu_start, torch::Tensor cu_end,
    torch::Tensor origin, torch::Tensor inv_delta, torch::Tensor th_bucket,
    torch::Tensor seed_val, torch::Tensor seed_idx, int64_t num_buckets64,
    int64_t cand_cap64, int64_t topk64, int64_t refresh_every64,
    int64_t num_kv_splits_override) {
  TORCH_CHECK(q.is_cuda() && kv.is_cuda() && kv_scales.is_cuda() &&
                  weights.is_cuda() && origin.is_cuda() &&
                  inv_delta.is_cuda() && th_bucket.is_cuda() &&
                  seed_val.is_cuda() && seed_idx.is_cuda(),
              "all tensors must be CUDA");
  TORCH_CHECK(q.is_contiguous() && kv.is_contiguous() &&
                  kv_scales.is_contiguous() && weights.is_contiguous() &&
                  cu_start.is_contiguous() && cu_end.is_contiguous() &&
                  origin.is_contiguous() && inv_delta.is_contiguous() &&
                  th_bucket.is_contiguous() && seed_val.is_contiguous() &&
                  seed_idx.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn, "q must be fp8_e4m3fn");
  TORCH_CHECK(kv.scalar_type() == torch::kFloat8_e4m3fn,
              "kv must be fp8_e4m3fn");
  TORCH_CHECK(kv_scales.scalar_type() == torch::kFloat,
              "kv_scales must be fp32");
  TORCH_CHECK(weights.scalar_type() == torch::kFloat, "weights must be fp32");
  TORCH_CHECK(origin.scalar_type() == torch::kFloat &&
                  inv_delta.scalar_type() == torch::kFloat,
              "origin/inv_delta must be fp32");
  TORCH_CHECK(th_bucket.scalar_type() == torch::kInt,
              "th_bucket must be int32");
  TORCH_CHECK(seed_val.scalar_type() == torch::kFloat,
              "seed_val must be fp32 x=-score");
  TORCH_CHECK(seed_idx.scalar_type() == torch::kInt, "seed_idx must be int32");
  TORCH_CHECK(cu_start.scalar_type() == torch::kInt &&
                  cu_end.scalar_type() == torch::kInt,
              "cu_start/cu_end must be int32");

  const int seq_len = (int)q.size(0);
  const int num_heads = (int)q.size(1);
  const int head_dim = (int)q.size(2);
  const int seq_len_kv = (int)kv.size(0);
  TORCH_CHECK(seq_len_kv <= (1 << dsa_litetopk::kCandidateIndexBits),
              "packed candidates support at most 1M KV positions");
  const int topk = static_cast<int>(topk64);
  // Sparse-only: honor a caller-provided cap in [topk, S).
  const int cand_cap = (cand_cap64 >= topk && cand_cap64 < seq_len_kv)
                           ? static_cast<int>(cand_cap64)
                           : seq_len_kv;
  TORCH_CHECK(num_heads == NUM_HEADS && head_dim == HEAD_DIM,
              "only GLM DSA H=32 D=128 is supported");
  TORCH_CHECK(kv.size(1) == HEAD_DIM, "kv D mismatch");
  TORCH_CHECK(origin.numel() == seq_len && inv_delta.numel() == seq_len &&
                  th_bucket.numel() == seq_len,
              "bucket params must have Q elements");
  const int num_buckets = static_cast<int>(num_buckets64);
  TORCH_CHECK(refresh_every64 > 0, "sparse refresh requires refresh_every>0");
  const int refresh_every = static_cast<int>(refresh_every64);
  TORCH_CHECK(num_buckets >= 3 && num_buckets <= 256,
              "in-place boundary select requires 3 <= num_buckets <= 256");
  TORCH_CHECK(topk >= 1 && topk <= cand_cap, "topk must be in [1, cand_cap]");
  TORCH_CHECK(refresh_every64 >= -1, "refresh_every must be >= -1");
  TORCH_CHECK(seed_val.dim() == 2 && seed_idx.dim() == 2,
              "seed tensors must be [Q, seed_k]");
  TORCH_CHECK(seed_val.size(0) == seq_len && seed_idx.size(0) == seq_len &&
                  seed_val.size(1) == seed_idx.size(1),
              "seed tensor shape mismatch");
  const int seed_k = static_cast<int>(seed_val.size(1));
  TORCH_CHECK(seed_k <= cand_cap, "seed_k must be <= cand_cap");
  TORCH_CHECK(seed_k == 0,
              "production scan requires empty seeds; use the "
              "prepared ext API so sampled positions are not double-counted");

  auto cand_val =
      torch::empty({seq_len, cand_cap}, candidate_options(q.options()));
  auto cand_idx =
      torch::empty({seq_len, cand_cap}, q.options().dtype(torch::kInt));
  auto cand_cnt =
      torch::full({seq_len}, seed_k, q.options().dtype(torch::kInt));
  auto bcount =
      torch::zeros({seq_len, num_buckets}, q.options().dtype(torch::kInt));

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  const int esz_fp8 = 1, esz_f32 = 4;
  const int ks_aligned = align_up(seq_len_kv, 16 / esz_f32);
  auto tm_q = make_2d(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8,
                      HEAD_DIM, seq_len * NUM_HEADS, HEAD_DIM,
                      BLOCK_Q * NUM_HEADS, HEAD_DIM, HEAD_DIM);
  auto tm_kv =
      make_2d(kv.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8, HEAD_DIM,
              seq_len_kv, HEAD_DIM, BLOCK_KV, HEAD_DIM, HEAD_DIM);
  auto tm_ks = make_2d(kv_scales.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                       esz_f32, ks_aligned, 1, BLOCK_KV, 1, 0, 0);
  auto tm_w =
      make_2d(weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32,
              NUM_HEADS, seq_len, NUM_HEADS, BLOCK_Q, NUM_HEADS, 0);

  const int smem = compute_smem_bytes();

  // Production packed-candidate path: one CTA owns each row's complete
  // histogram and publishes an exact boundary certificate.
  const int num_q_blocks = (seq_len + BLOCK_Q - 1) / BLOCK_Q;
  TORCH_CHECK(num_kv_splits_override <= 0 || num_kv_splits_override == 1,
              "production packed candidates require num_kv_splits=1");
  constexpr int num_kv_splits = 1;
  auto kernel = &dsa_litetopk::sm100_dsa_litetopk<
      NUM_HEADS, HEAD_DIM, BLOCK_Q, BLOCK_KV, NUM_Q_STAGES, NUM_KV_STAGES,
      NUM_SMS, SPEC_THREADS, MATH_THREADS, MATH_THREADS / 128>;
  C10_CUDA_CHECK(
      cudaFuncSetAttribute(reinterpret_cast<void*>(kernel),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  dim3 grid((unsigned)num_q_blocks, (unsigned)num_kv_splits, 1);
  kernel<<<grid, SPEC_THREADS + MATH_THREADS, smem, stream>>>(
      (uint32_t)seq_len, (uint32_t)seq_len_kv,
      (uint32_t*)cu_start.data_ptr<int>(), (uint32_t*)cu_end.data_ptr<int>(),
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
      th_bucket.data_ptr<int32_t>(), bcount.data_ptr<int32_t>(),
      (uint32_t)num_buckets, (uint32_t)topk, (uint32_t)refresh_every,
      (uint32_t)num_kv_splits, 0u, 0ULL, 0u, candidate_data_ptr(cand_val),
      cand_idx.data_ptr<int32_t>(), cand_cnt.data_ptr<int32_t>(),
      (uint32_t)cand_cap, tm_q, tm_kv, tm_ks, tm_w);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_tuple(cand_val, cand_idx, cand_cnt);
}

void launch_seed_prep(const float* slog, int64_t slog_stride, int Q, int head,
                      int NB, int K, float headroom, float* origin,
                      float* inv_delta, int32_t* th_bucket, int32_t* cand_cnt,
                      cudaStream_t stream) {
  const int seed_smem = 4 * NB * static_cast<int>(sizeof(int));
  seed_prep_kernel<<<Q, kSeedThreads, seed_smem, stream>>>(
      slog, slog_stride, head, NB, K, headroom, origin, inv_delta, th_bucket,
      cand_cnt);
}

// Fused seed/prep: sample scores -> (origin, inv_delta, th_bucket, cand_val,
// cand_idx, cand_cnt, bcount), everything the scan needs, in one launch.
void seed_prep_litetopk_(torch::Tensor slog, int64_t num_buckets64,
                         int64_t topk64, int64_t cand_cap64,
                         int64_t emit_limit64, double headroom,
                         int64_t probe_stride_tok64, int64_t hist_stride64,
                         torch::Tensor origin, torch::Tensor inv_delta,
                         torch::Tensor th_bucket, torch::Tensor bcount,
                         torch::Tensor cand_val, torch::Tensor cand_idx,
                         torch::Tensor cand_cnt) {
  TORCH_CHECK(slog.is_cuda() && slog.dim() == 2, "slog must be CUDA [Q, head]");
  TORCH_CHECK(slog.scalar_type() == torch::kFloat, "slog must be fp32 scores");
  TORCH_CHECK(slog.stride(1) == 1, "slog rows must be inner-contiguous");
  const int Q = (int)slog.size(0);
  const int head = (int)slog.size(1);
  const int NB = (int)num_buckets64;
  const int K = (int)topk64;
  const int cap = (int)cand_cap64;
  TORCH_CHECK(head >= K && head <= 8192,
              "production seed prep requires topk <= HOT <= 8192");
  TORCH_CHECK(NB >= 3 && NB <= 256, "num_buckets out of range");
  TORCH_CHECK(K >= 1 && cap >= K, "need cap >= topk >= 1");
  TORCH_CHECK(cand_val.size(0) >= Q && cand_val.size(1) == cap,
              "cand_val shape");
  check_candidate_dtype(cand_val);
  TORCH_CHECK(bcount.size(0) >= Q && bcount.size(1) == NB, "bcount shape");
  TORCH_CHECK((slog.stride(0) % 4) == 0 &&
                  (reinterpret_cast<uintptr_t>(slog.data_ptr()) % 16) == 0,
              "slog rows must be 16B aligned");
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int emit_limit =
      emit_limit64 == 0 ? 0 : (emit_limit64 > 0 ? (int)emit_limit64 : head);
  TORCH_CHECK(emit_limit == 0,
              "production seed prep requires the hot-only no-seed contract");
  TORCH_CHECK(hist_stride64 == 1,
              "production seed prep requires hist_stride=1");
  (void)probe_stride_tok64;
  launch_seed_prep(slog.data_ptr<float>(), slog.stride(0), Q, head, NB, K,
                   static_cast<float>(headroom), origin.data_ptr<float>(),
                   inv_delta.data_ptr<float>(), th_bucket.data_ptr<int32_t>(),
                   cand_cnt.data_ptr<int32_t>(), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
seed_prep_litetopk(torch::Tensor slog, int64_t num_buckets64, int64_t topk64,
                   int64_t cand_cap64, int64_t emit_limit64, double headroom,
                   int64_t probe_stride_tok64, int64_t hist_stride64) {
  TORCH_CHECK(slog.is_cuda() && slog.dim() == 2, "slog must be CUDA [Q, head]");
  TORCH_CHECK(slog.scalar_type() == torch::kFloat, "slog must be fp32 scores");
  TORCH_CHECK(slog.stride(1) == 1, "slog rows must be inner-contiguous");
  const int Q = (int)slog.size(0);
  const int head = (int)slog.size(1);
  const int NB = (int)num_buckets64;
  const int K = (int)topk64;
  const int cap = (int)cand_cap64;
  TORCH_CHECK(head >= K && head <= 8192,
              "production seed prep requires topk <= HOT <= 8192");
  TORCH_CHECK(NB >= 3 && NB <= 256, "num_buckets out of range");
  TORCH_CHECK(K >= 1 && cap >= K, "need cap >= topk >= 1");
  // float4 pass requires 16B-aligned rows; fall back is not implemented.
  TORCH_CHECK((slog.stride(0) % 4) == 0 &&
                  (reinterpret_cast<uintptr_t>(slog.data_ptr()) % 16) == 0,
              "slog rows must be 16B aligned");

  auto opts_f = slog.options();
  auto opts_i = slog.options().dtype(torch::kInt);
  auto origin = torch::empty({Q}, opts_f);
  auto inv_delta = torch::empty({Q}, opts_f);
  auto th_bucket = torch::empty({Q}, opts_i);
  auto bcount =
      torch::empty({Q, NB}, opts_i);  // scan publishes boundary metadata
  auto cand_val = torch::empty({Q, cap}, candidate_options(opts_f));
  auto cand_idx = torch::empty({Q, cap}, opts_i);
  auto cand_cnt = torch::empty({Q}, opts_i);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int emit_limit =
      emit_limit64 == 0 ? 0 : (emit_limit64 > 0 ? (int)emit_limit64 : head);
  TORCH_CHECK(emit_limit == 0,
              "production seed prep requires the hot-only no-seed contract");
  TORCH_CHECK(hist_stride64 == 1,
              "production seed prep requires hist_stride=1");
  (void)probe_stride_tok64;
  launch_seed_prep(slog.data_ptr<float>(), slog.stride(0), Q, head, NB, K,
                   static_cast<float>(headroom), origin.data_ptr<float>(),
                   inv_delta.data_ptr<float>(), th_bucket.data_ptr<int32_t>(),
                   cand_cnt.data_ptr<int32_t>(), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(origin, inv_delta, th_bucket, cand_val, cand_idx,
                         cand_cnt, bcount);
}

// Scan into buffers prepared by seed_prep_litetopk (no seeding of any kind).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
mqa_logits_dsa_litetopk_ext(torch::Tensor q, torch::Tensor kv,
                            torch::Tensor kv_scales, torch::Tensor weights,
                            torch::Tensor cu_start, torch::Tensor cu_end,
                            torch::Tensor origin, torch::Tensor inv_delta,
                            torch::Tensor th_bucket, torch::Tensor cand_val,
                            torch::Tensor cand_idx, torch::Tensor cand_cnt,
                            torch::Tensor bcount, int64_t num_buckets64,
                            int64_t topk64, int64_t refresh_every64,
                            int64_t num_kv_splits_override,
                            int64_t probe_group64, int64_t probe_add_max64) {
  TORCH_CHECK(
      q.is_cuda() && kv.is_cuda() && kv_scales.is_cuda() && weights.is_cuda(),
      "all tensors must be CUDA");
  TORCH_CHECK(q.is_contiguous() && kv.is_contiguous() &&
                  kv_scales.is_contiguous() && weights.is_contiguous() &&
                  cu_start.is_contiguous() && cu_end.is_contiguous() &&
                  origin.is_contiguous() && inv_delta.is_contiguous() &&
                  th_bucket.is_contiguous() && cand_val.is_contiguous() &&
                  cand_idx.is_contiguous() && cand_cnt.is_contiguous() &&
                  bcount.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn &&
                  kv.scalar_type() == torch::kFloat8_e4m3fn,
              "q/kv must be fp8_e4m3fn");
  check_candidate_dtype(cand_val);
  const int seq_len = (int)q.size(0);
  const int seq_len_kv = (int)kv.size(0);
  TORCH_CHECK(seq_len_kv <= (1 << dsa_litetopk::kCandidateIndexBits),
              "packed candidates support at most 1M KV positions");
  const int cand_cap = (int)cand_val.size(1);
  const int num_buckets = (int)num_buckets64;
  const int topk = (int)topk64;
  TORCH_CHECK(q.size(1) == NUM_HEADS && q.size(2) == HEAD_DIM,
              "only GLM DSA H=32 D=128 is supported");
  TORCH_CHECK(num_buckets >= 3 && num_buckets <= 256,
              "prepared scan requires 3 <= num_buckets <= 256");
  TORCH_CHECK(topk >= 1 && topk <= cand_cap, "topk must be in [1, cand_cap]");
  TORCH_CHECK(refresh_every64 > 0, "sparse refresh requires refresh_every>0");
  TORCH_CHECK(cand_val.size(0) == seq_len &&
                  cand_idx.sizes() == cand_val.sizes() &&
                  cand_cnt.numel() == seq_len && bcount.size(0) == seq_len &&
                  bcount.size(1) == num_buckets,
              "prepared buffer shape mismatch");
  const int refresh_every = static_cast<int>(refresh_every64);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int esz_fp8 = 1, esz_f32 = 4;
  const int ks_aligned = align_up(seq_len_kv, 16 / esz_f32);
  auto tm_q = make_2d(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8,
                      HEAD_DIM, seq_len * NUM_HEADS, HEAD_DIM,
                      BLOCK_Q * NUM_HEADS, HEAD_DIM, HEAD_DIM);
  auto tm_kv =
      make_2d(kv.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8, HEAD_DIM,
              seq_len_kv, HEAD_DIM, BLOCK_KV, HEAD_DIM, HEAD_DIM);
  auto tm_ks = make_2d(kv_scales.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                       esz_f32, ks_aligned, 1, BLOCK_KV, 1, 0, 0);
  auto tm_w =
      make_2d(weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32,
              NUM_HEADS, seq_len, NUM_HEADS, BLOCK_Q, NUM_HEADS, 0);

  const int smem = compute_smem_bytes();

  const int num_q_blocks = (seq_len + BLOCK_Q - 1) / BLOCK_Q;
  TORCH_CHECK(num_kv_splits_override <= 0 || num_kv_splits_override == 1,
              "production packed candidates require num_kv_splits=1");
  constexpr int num_kv_splits = 1;
  auto kernel = &dsa_litetopk::sm100_dsa_litetopk<
      NUM_HEADS, HEAD_DIM, BLOCK_Q, BLOCK_KV, NUM_Q_STAGES, NUM_KV_STAGES,
      NUM_SMS, SPEC_THREADS, MATH_THREADS, MATH_THREADS / 128>;
  C10_CUDA_CHECK(
      cudaFuncSetAttribute(reinterpret_cast<void*>(kernel),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  dim3 grid((unsigned)num_q_blocks, (unsigned)num_kv_splits, 1);
  kernel<<<grid, SPEC_THREADS + MATH_THREADS, smem, stream>>>(
      (uint32_t)seq_len, (uint32_t)seq_len_kv,
      (uint32_t*)cu_start.data_ptr<int>(), (uint32_t*)cu_end.data_ptr<int>(),
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
      th_bucket.data_ptr<int32_t>(), bcount.data_ptr<int32_t>(),
      (uint32_t)num_buckets, (uint32_t)topk, (uint32_t)refresh_every,
      (uint32_t)num_kv_splits, (uint32_t)probe_group64,
      probe_group64 > 0 ? (((1ULL << 42) + (uint64_t)probe_group64 - 1) /
                           (uint64_t)probe_group64)
                        : 0ULL,
      (uint32_t)probe_add_max64, candidate_data_ptr(cand_val),
      cand_idx.data_ptr<int32_t>(), cand_cnt.data_ptr<int32_t>(),
      (uint32_t)cand_cap, tm_q, tm_kv, tm_ks, tm_w);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_tuple(cand_val, cand_idx, cand_cnt);
}

void cand_count_stats_litetopk_(torch::Tensor cand_cnt, torch::Tensor stats) {
  TORCH_CHECK(cand_cnt.is_cuda() && stats.is_cuda(),
              "cand_cnt/stats must be CUDA tensors");
  TORCH_CHECK(cand_cnt.is_contiguous() && stats.is_contiguous(),
              "cand_cnt/stats must be contiguous");
  TORCH_CHECK(cand_cnt.scalar_type() == torch::kInt &&
                  stats.scalar_type() == torch::kInt,
              "cand_cnt/stats must be int32");
  TORCH_CHECK(cand_cnt.dim() == 1 && cand_cnt.numel() > 0,
              "cand_cnt must be a non-empty 1-D tensor");
  TORCH_CHECK(cand_cnt.numel() <= std::numeric_limits<int32_t>::max(),
              "cand_cnt is too large for the single-CTA stats ABI");
  TORCH_CHECK(stats.dim() == 1 && stats.numel() == 2, "stats must be int32[2]");
  TORCH_CHECK(cand_cnt.device() == stats.device(),
              "cand_cnt/stats must be on the same CUDA device");

  cand_count_stats_litetopk_kernel<<<1, 256, 0,
                                     c10::cuda::getCurrentCUDAStream()>>>(
      cand_cnt.data_ptr<int32_t>(), static_cast<int>(cand_cnt.numel()),
      stats.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void carry_votes_topk_reset_litetopk_(torch::Tensor votes,
                                      torch::Tensor out_idx,
                                      torch::Tensor partial,
                                      torch::Tensor state, int64_t k64,
                                      int64_t max_vote64, int64_t min_index64) {
  TORCH_CHECK(votes.is_cuda() && out_idx.is_cuda() && partial.is_cuda() &&
                  state.is_cuda(),
              "votes/out_idx/partial/state must be CUDA tensors");
  TORCH_CHECK(votes.is_contiguous() && out_idx.is_contiguous() &&
                  partial.is_contiguous() && state.is_contiguous(),
              "votes/out_idx/partial/state must be contiguous");
  TORCH_CHECK(votes.scalar_type() == torch::kInt, "votes must be int32");
  TORCH_CHECK(out_idx.scalar_type() == torch::kLong, "out_idx must be int64");
  TORCH_CHECK(partial.scalar_type() == torch::kShort, "partial must be int16");
  TORCH_CHECK(state.scalar_type() == torch::kInt, "state must be int32");
  TORCH_CHECK(votes.device() == out_idx.device() &&
                  votes.device() == partial.device() &&
                  votes.device() == state.device(),
              "votes/out_idx/partial/state must be on the same CUDA device");
  TORCH_CHECK(votes.dim() == 1, "votes must be a 1-D histogram");
  TORCH_CHECK(out_idx.dim() == 1, "out_idx must be 1-D");
  TORCH_CHECK(partial.dim() == 2, "partial must be [blocks,bins]");
  TORCH_CHECK(state.dim() == 1 && state.numel() >= kCarryStateInts,
              "state is too small for the carry top-k ABI");

  const int64_t count64 = votes.numel();
  TORCH_CHECK(count64 >= 1 && count64 <= kCarryMaxItems,
              "votes length must be in [1,1048576]");
  TORCH_CHECK(k64 >= 1 && k64 <= kCarryMaxK, "k must be in [1,8192]");
  TORCH_CHECK(max_vote64 >= 1 && max_vote64 <= kCarryMaxK,
              "max_vote must be in [1,8192]");
  TORCH_CHECK(min_index64 >= 0 && min_index64 < count64,
              "min_index must be in [0,votes.numel())");
  const int count = static_cast<int>(count64);
  const int min_index = static_cast<int>(min_index64);
  const int eligible = count - min_index;
  const int out_k = static_cast<int>(min(k64, static_cast<int64_t>(eligible)));
  const int max_vote = static_cast<int>(max_vote64);
  const int bins = max_vote + 1;
  const int blocks = (count + kCarryTileItems - 1) / kCarryTileItems;
  TORCH_CHECK(out_idx.numel() == out_k,
              "out_idx must have min(k,votes.numel()-min_index) elements");
  TORCH_CHECK(partial.size(0) >= blocks && partial.size(1) >= bins,
              "partial must provide at least [ceil(N/8192),max_vote+1]");

  const int partial_stride = static_cast<int>(partial.size(1));
  const size_t dynamic_smem = static_cast<size_t>(bins) * sizeof(uint32_t);
  auto stream = c10::cuda::getCurrentCUDAStream();
  carry_votes_plan_litetopk_kernel<<<blocks, kCarryThreads, dynamic_smem,
                                     stream>>>(
      votes.data_ptr<int32_t>(), count, min_index, out_k, max_vote,
      partial.data_ptr<int16_t>(), partial_stride, state.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  carry_votes_emit_reset_litetopk_kernel<<<blocks, kCarryThreads, 0, stream>>>(
      votes.data_ptr<int32_t>(), count, min_index, max_vote,
      out_idx.data_ptr<int64_t>(), state.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Destructive single-use selector for the fused indexer. This entry point
// consumes cand_val/cand_idx by compacting its selected subset in place.
// Candidate indices must already be in final corpus
// space, as they are for the current chunked-flush scan path. Gate4 candidate
// values are already in bucket space, and the caller owns the final idx
// output, so this specialization allocates and writes no discarded values or
// temporary index tensor.
void compact_topk_min_thr_inplace_idx_out_litetopk(
    torch::Tensor cand_val, torch::Tensor cand_idx, torch::Tensor cand_cnt,
    torch::Tensor th_bucket, torch::Tensor boundary_meta, int64_t num_buckets64,
    int64_t k64, torch::Tensor out_idx, torch::Tensor votes,
    int64_t vote_row_stride64) {
  TORCH_CHECK(cand_val.is_cuda() && cand_idx.is_cuda() && cand_cnt.is_cuda() &&
                  th_bucket.is_cuda() && boundary_meta.is_cuda() &&
                  out_idx.is_cuda() && votes.is_cuda(),
              "tensors must be CUDA");
  TORCH_CHECK(cand_val.is_contiguous() && cand_idx.is_contiguous() &&
                  cand_cnt.is_contiguous() && th_bucket.is_contiguous() &&
                  boundary_meta.is_contiguous() && out_idx.is_contiguous() &&
                  votes.is_contiguous(),
              "tensors must be contiguous");
  check_candidate_dtype(cand_val);
  TORCH_CHECK(cand_idx.scalar_type() == torch::kInt &&
                  cand_cnt.scalar_type() == torch::kInt &&
                  out_idx.scalar_type() == torch::kInt,
              "idx/cnt/out_idx must be int32");
  TORCH_CHECK(th_bucket.scalar_type() == torch::kInt,
              "th_bucket must be int32");
  TORCH_CHECK(boundary_meta.scalar_type() == torch::kInt,
              "boundary_meta must be int32");
  TORCH_CHECK(votes.scalar_type() == torch::kInt, "votes must be int32");
  TORCH_CHECK(cand_val.dim() == 2 && cand_idx.sizes() == cand_val.sizes(),
              "candidate tensors must be [R,CAP]");
  const int R = static_cast<int>(cand_val.size(0));
  const int CAP = static_cast<int>(cand_val.size(1));
  TORCH_CHECK(cand_cnt.dim() == 1 && cand_cnt.numel() == R,
              "cand_cnt must have R elements");
  const int K = static_cast<int>(k64);
  const int NB = static_cast<int>(num_buckets64);
  TORCH_CHECK(K >= 1 && K <= CAP, "K must be in [1,CAP]");
  TORCH_CHECK(NB >= 3 && NB <= 256,
              "in-place boundary select requires 3 <= num_buckets <= 256");
  TORCH_CHECK(th_bucket.numel() == R, "th_bucket must have R elements");
  TORCH_CHECK(boundary_meta.dim() == 2 && boundary_meta.size(0) == R &&
                  boundary_meta.size(1) == NB,
              "boundary_meta must be [R,num_buckets]");
  TORCH_CHECK(
      out_idx.dim() == 2 && out_idx.size(0) == R && out_idx.size(1) == K,
      "out_idx must be [R,K]");
  TORCH_CHECK(votes.dim() == 1, "votes must be a 1-D histogram (or empty)");
  const int votes_len = static_cast<int>(votes.numel());
  TORCH_CHECK(vote_row_stride64 == 1 || vote_row_stride64 == 8 ||
                  vote_row_stride64 == 16,
              "vote_row_stride must be one of {1, 8, 16}");
  const int vote_row_stride = static_cast<int>(vote_row_stride64);
  auto stream = c10::cuda::getCurrentCUDAStream();
  compact_topk_min_thr_inplace_idx_out_litetopk_kernel<<<R, 256, 0, stream>>>(
      candidate_data_ptr(cand_val), cand_idx.data_ptr<int32_t>(),
      cand_cnt.data_ptr<int32_t>(), th_bucket.data_ptr<int32_t>(),
      boundary_meta.data_ptr<int32_t>(), R, CAP, K, NB,
      out_idx.data_ptr<int32_t>(),
      votes_len > 0 ? votes.data_ptr<int32_t>() : nullptr, votes_len,
      vote_row_stride);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "candidate_fp24_global_litetopk", []() { return true; },
      "Reports the production high24 FP32 local/global candidate ABI");
  m.def(
      "candidate_value_u16_litetopk", []() { return true; },
      "Reports the packed six-byte candidate ABI");
  m.def("gather_hot_sample_litetopk_", &gather_hot_sample_litetopk_,
        "One-launch FP8 K + FP32 scale gather into caller-owned buffers",
        pybind11::arg("k"), pybind11::arg("k_scale"), pybind11::arg("idx"),
        pybind11::arg("out_k"), pybind11::arg("out_scale"));
  m.def("dense_topk_litetopk_", &litetopk_dense::dense_topk_litetopk_,
        "Single-launch exact dense prefill top-k", pybind11::arg("logits"),
        pybind11::arg("row_starts"), pybind11::arg("row_ends"),
        pybind11::arg("out"), pybind11::arg("rows"), pybind11::arg("stride0"),
        pybind11::arg("stride1"), pybind11::arg("topk"),
        pybind11::arg("num_init_tokens"), pybind11::arg("num_local_tokens"));
  m.def("dense_hist_meta_litetopk_", &litetopk_dense::dense_hist_meta_litetopk_,
        "Exact FP16-coarse histogram metadata for dense prefill top-k",
        pybind11::arg("logits"), pybind11::arg("row_starts"),
        pybind11::arg("row_ends"), pybind11::arg("threshold"),
        pybind11::arg("count_lt"), pybind11::arg("count_eq"),
        pybind11::arg("rows"), pybind11::arg("stride0"),
        pybind11::arg("stride1"), pybind11::arg("topk"),
        pybind11::arg("bins") = 2048);
  m.def("dense_prehist_select_litetopk_",
        &litetopk_dense::dense_prehist_select_litetopk_,
        "Exact dense prefill top-k from precomputed coarse metadata",
        pybind11::arg("logits"), pybind11::arg("row_starts"),
        pybind11::arg("row_ends"), pybind11::arg("threshold"),
        pybind11::arg("count_lt"), pybind11::arg("count_eq"),
        pybind11::arg("out"), pybind11::arg("rows"), pybind11::arg("stride0"),
        pybind11::arg("stride1"), pybind11::arg("topk"),
        pybind11::arg("bins") = 2048);
  m.def("seed_prep_litetopk_", &seed_prep_litetopk_,
        "In-place fused sample prep (caller-owned buffers)",
        pybind11::arg("slog"), pybind11::arg("num_buckets"),
        pybind11::arg("topk"), pybind11::arg("cand_cap"),
        pybind11::arg("emit_limit"), pybind11::arg("headroom"),
        pybind11::arg("probe_stride_tok"), pybind11::arg("hist_stride"),
        pybind11::arg("origin"), pybind11::arg("inv_delta"),
        pybind11::arg("th_bucket"), pybind11::arg("bcount"),
        pybind11::arg("cand_val"), pybind11::arg("cand_idx"),
        pybind11::arg("cand_cnt"));
  m.def(
      "seed_prep_litetopk", &seed_prep_litetopk,
      "Fused sample prep: scores -> (origin, inv, th, cand bufs, cnt, bcount)",
      pybind11::arg("slog"), pybind11::arg("num_buckets"),
      pybind11::arg("topk"), pybind11::arg("cand_cap"),
      pybind11::arg("emit_limit") = -1, pybind11::arg("headroom") = 0.0,
      pybind11::arg("probe_stride_tok") = 0, pybind11::arg("hist_stride") = 1);
  m.def("mqa_logits_dsa_litetopk_ext", &mqa_logits_dsa_litetopk_ext,
        "V3 scan into buffers prepared by seed_prep_litetopk",
        pybind11::arg("q"), pybind11::arg("kv"), pybind11::arg("kv_scales"),
        pybind11::arg("weights"), pybind11::arg("cu_start"),
        pybind11::arg("cu_end"), pybind11::arg("origin"),
        pybind11::arg("inv_delta"), pybind11::arg("th_bucket"),
        pybind11::arg("cand_val"), pybind11::arg("cand_idx"),
        pybind11::arg("cand_cnt"), pybind11::arg("bcount"),
        pybind11::arg("num_buckets"), pybind11::arg("topk"),
        pybind11::arg("refresh_every"), pybind11::arg("num_kv_splits") = -1,
        pybind11::arg("probe_group") = 0, pybind11::arg("probe_add_max") = 0);
  m.def("mqa_logits_dsa_litetopk", &mqa_logits_dsa_litetopk,
        "DSA ReLU-MQA scoring V3 hybrid (DeepGEMM-2.5 loop + V1 KV-split) with "
        "sparse epilogue",
        pybind11::arg("q"), pybind11::arg("kv"), pybind11::arg("kv_scales"),
        pybind11::arg("weights"), pybind11::arg("cu_start"),
        pybind11::arg("cu_end"), pybind11::arg("origin"),
        pybind11::arg("inv_delta"), pybind11::arg("th_bucket"),
        pybind11::arg("seed_val"), pybind11::arg("seed_idx"),
        pybind11::arg("num_buckets"), pybind11::arg("cand_cap"),
        pybind11::arg("topk"), pybind11::arg("refresh_every"),
        pybind11::arg("num_kv_splits") = -1);
  m.def("cand_count_stats_litetopk_", &cand_count_stats_litetopk_,
        "Single-CTA candidate-count max and exact integer mean",
        pybind11::arg("cand_cnt"), pybind11::arg("stats"));
  m.def("carry_votes_topk_reset_", &carry_votes_topk_reset_litetopk_,
        "Deterministic carry-vote top-k with fused histogram reset",
        pybind11::arg("votes"), pybind11::arg("out_idx"),
        pybind11::arg("partial"), pybind11::arg("state"), pybind11::arg("k"),
        pybind11::arg("max_vote"), pybind11::arg("min_index") = 0);
  m.def("compact_topk_min_thr_inplace_idx_out_litetopk",
        &compact_topk_min_thr_inplace_idx_out_litetopk,
        "Single-use Gate4 threshold top-k directly into caller idx output",
        pybind11::arg("cand_val"), pybind11::arg("cand_idx"),
        pybind11::arg("cand_cnt"), pybind11::arg("th_bucket"),
        pybind11::arg("boundary_meta"), pybind11::arg("num_buckets"),
        pybind11::arg("topk"), pybind11::arg("out_idx"), pybind11::arg("votes"),
        pybind11::arg("vote_row_stride") = 1);
}
