// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SGLang K3 KDA fused-decode kernel, vendored for the raon-1-mini vLLM plugin.
//
// Provenance: sgl-project/sglang @ f65c70bb7d46ceecaacb42689816f3036cffd03f
//   python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh (PR #30113,
//   NVIDIA x Moonshot Kimi K3 package; see that file's own patch list).
// The DEVICE section below is byte-identical to upstream except for the
// anchored local patches, each marked "[RAON RDA patch]" or "[RAON port]":
//   1. rda_ratio / rda_row_stride kernel parameters + post-SiLU v scale in
//      both conv branches -- raon's RDA gate scales v at the unfused chain's
//      post-conv seam (_scale_packed_v); the fused kernel has no seam, so the
//      ratio is folded in as a conv epilogue (the study's C2 prescription,
//      bottleneck-opt-candidates-2026-09-17.md).
//   2. This prelude replaces the sgl_kernel/tvm-ffi headers with the handful
//      of device helpers the kernel body actually uses (verbatim bodies from
//      sgl_kernel/math.cuh, warp.cuh, utils.cuh at the same commit).
//   3. The tvm-ffi host wrapper (KdaFusedDecodeKernel) is replaced by a torch
//      extension entry point at the bottom, with kFixedHeads=40 instantiated
//      for raon (upstream instantiates only 3/6/12; the head axis is
//      grid-parallel, so the count is an allowlist, not a structural limit).
// Do not "improve" the device section: re-vendor from upstream and re-apply
// the anchored patches instead.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define SGL_DEVICE __device__ __forceinline__

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  #define SGL_ARCH_HOPPER_OR_GREATER 1
#else
  #define SGL_ARCH_HOPPER_OR_GREATER 0
#endif

namespace sglang {
namespace device {
namespace math {
// Verbatim from sgl_kernel/math.cuh @ f65c70bb7d46ceecaacb42689816f3036cffd03f.
SGL_DEVICE float sigmoid_fast(float x) { return 1.0f / (1.0f + __expf(-x)); }
SGL_DEVICE float silu_fast(float x) { return x * sigmoid_fast(x); }
SGL_DEVICE float softplus_fast(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}
}  // namespace math
namespace warp {
// sgl_kernel/warp.cuh reduce_sum<32, true, float>: the fp32 SUM case takes the
// generic butterfly (redux.sync covers int SUM and fp32 MAX/MIN only).
SGL_DEVICE float reduce_sum(float value) {
#pragma unroll
  for (int mask = 16; mask >= 1; mask >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
  }
  return value;
}
}  // namespace warp
// Verbatim from sgl_kernel/utils.cuh @
// f65c70bb7d46ceecaacb42689816f3036cffd03f.
template <bool kUsePDL>
SGL_DEVICE void PDLWaitPrimary() {
#if SGL_ARCH_HOPPER_OR_GREATER
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
#endif
}
template <bool kUsePDL>
SGL_DEVICE void PDLTriggerSecondary() {
#if SGL_ARCH_HOPPER_OR_GREATER
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
#endif
}
}  // namespace device
}  // namespace sglang

// ---------------------------------------------------------------------------
// DEVICE SECTION (vendored, anchored patches only -- see header)
// ---------------------------------------------------------------------------
namespace sglang {

namespace ptx {

// Generic ptr -> 32-bit `.shared` address: inline-PTX `.shared` instructions
// take a byte offset in the shared window, not a generic pointer.
template <typename T>
static SGL_DEVICE uint32_t to_shared(T* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// ---- non-bulk cp.async (PTX ISA §9.7.9.24) ---------------------------------

// One 16-byte cache-global segment, global -> shared. Both pointers must be
// 16-byte aligned.
static SGL_DEVICE void cp_async_cg_16b(void* smem_dst, const void* gmem_src) {
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(to_shared(smem_dst)),
      "l"(gmem_src)
      : "memory");
}

static SGL_DEVICE void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;");
}

// Wait until at most N committed cp.async groups remain pending.
template <int N>
static SGL_DEVICE void cp_async_wait_group() {
  static_assert(N >= 0 && N <= 7,
                "cp.async wait-group count must be in [0, 7]");
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

static SGL_DEVICE void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;" ::: "memory");
}

// ---- bulk 1D TMA (PTX ISA §9.7.9.25) ---------------------------------------

// global -> shared::cluster, completed by an smem mbarrier. Arm `bar` with
// `mbar_arrive_expect_tx(bar, bytes)` before issuing; `bytes` and both
// endpoints must be 16-byte aligned.
static SGL_DEVICE void cp_async_bulk_1d_load(void* smem_dst,
                                             const void* gmem_src,
                                             uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];" ::"r"(to_shared(smem_dst)),
      "l"(gmem_src), "r"(bytes), "r"(to_shared(bar))
      : "memory");
}

// ---- mbarrier (PTX ISA §9.7.13.15) -----------------------------------------
//
// Only the `try_wait.parity` waiter is wrapped; the caller owns the phase
// counter and flips it at the stage wrap. After `mbar_init` the bar is at
// parity 0 and each full cycle (count arrivals -> fire -> reset) flips it, so a
// consumer-first waiter starts at 0 and a producer-first waiter (whose first
// wait must be a no-op skip) starts at 1. Getting this backwards deadlocks on
// the second wait.
static SGL_DEVICE void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(to_shared(bar)),
               "r"(count));
}

// Combined arrive + set tx-count, for TMA-load completion.
static SGL_DEVICE void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(to_shared(bar)),
      "r"(bytes));
}

// Wait for phase `parity` to complete. Looped because the spec allows spurious
// early wakeups. The default `.acquire` semantics make prior `cp.async.bulk`
// writes tracked by this mbarrier visible to later generic-proxy reads on this
// thread with no `fence.proxy.async` (spec §9.7.13.15.16 point 3).
static SGL_DEVICE void mbar_wait_parity(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_%=;\n\t}\n" ::"r"(to_shared(bar)),
      "r"(parity));
}

// ---- async-proxy fence (PTX ISA §9.7.13) -----------------------------------

// Make generic-proxy smem writes visible to the async proxy. Required before
// any bulk store that reads smem written by regular ld/st.
static SGL_DEVICE void fence_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;");
}

}  // namespace ptx

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kKernelWidth = 4;
constexpr int kConvStateWidth = kKernelWidth - 1;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunkV = 32;
constexpr int kNumChunks = kDimV / kChunkV;
constexpr int kRowsPerWarp = kChunkV / kWarps;

SGL_DEVICE float bf16_load(const __nv_bfloat16* ptr, int idx) {
  return __bfloat162float(ptr[idx]);
}

SGL_DEVICE __nv_bfloat16 bf16_store(float value) {
  return __float2bfloat16(value);
}

template <bool kUseCacheGlobalStore>
SGL_DEVICE void store_state_float4(float* ptr, float4 value) {
  if constexpr (kUseCacheGlobalStore) {
    __stcg(reinterpret_cast<float4*>(ptr), value);
  } else {
    *reinterpret_cast<float4*>(ptr) = value;
  }
}

template <int kCopyThreads>
SGL_DEVICE void cp_async_state_chunk_for(float* s_state, const float* state,
                                         int slot, int i_hv,
                                         int64_t state_slot_stride, int chunk) {
  constexpr int kFloat4PerChunk = kChunkV * kDimK / 4;
  const int tid = threadIdx.x;
  const int stage = chunk & 1;
  const int v_base = chunk * kChunkV;
  const int64_t slot_base = static_cast<int64_t>(slot) * state_slot_stride;
  for (int linear4 = tid; linear4 < kFloat4PerChunk; linear4 += kCopyThreads) {
    const int elem = linear4 * 4;
    const int row = elem / kDimK;
    const int k = elem - row * kDimK;
    float* dst = s_state + (stage * kChunkV + row) * kDimK + k;
    const float* src =
        state + slot_base + ((i_hv * kDimV + v_base + row) * kDimK + k);
    ptx::cp_async_cg_16b(dst, src);
  }
  ptx::cp_async_commit_group();
}

SGL_DEVICE void cp_async_state_chunk(float* s_state, const float* state,
                                     int slot, int i_hv,
                                     int64_t state_slot_stride, int chunk) {
  cp_async_state_chunk_for<kThreads>(s_state, state, slot, i_hv,
                                     state_slot_stride, chunk);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  #define KDA_FUSED_DECODE_HAS_TMA 1
#else
  #define KDA_FUSED_DECODE_HAS_TMA 0
#endif

// 1D TMA bulk copy gmem -> smem; the issuing thread arrives with expect-tx on
// the mbarrier, completion is observed via ptx::mbar_wait_parity.
template <int kStageChunkV>
SGL_DEVICE void tma_state_chunk_stage(float* s_state, const float* state,
                                      int slot, int i_hv,
                                      int64_t state_slot_stride, int chunk,
                                      int stage, uint64_t* bar) {
  constexpr uint32_t kBytes = kStageChunkV * kDimK * sizeof(float);
  float* dst = s_state + stage * kStageChunkV * kDimK;
  // 16B-aligned for any slot iff state_slot_stride*sizeof(float) % 16 == 0
  // (host gates TMA off otherwise); the intra-slot chunk offset is a multiple
  // of kStageChunkV*kDimK*4, always 16B-aligned.
  const int64_t slot_base = static_cast<int64_t>(slot) * state_slot_stride;
  const float* src =
      state + slot_base + ((i_hv * kDimV + chunk * kStageChunkV) * kDimK);
#if KDA_FUSED_DECODE_HAS_TMA
  ptx::fence_async_smem();
  ptx::mbar_arrive_expect_tx(bar, kBytes);
  ptx::cp_async_bulk_1d_load(dst, src, kBytes, bar);
#else
  __trap();
#endif
}

SGL_DEVICE float block_reduce_sum(float value, float* scratch) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  float warp_total = device::warp::reduce_sum(value);
  if (lane == 0) {
    scratch[warp] = warp_total;
  }
  __syncthreads();

  float block_total = 0.0f;
  if (warp == 0) {
    block_total = lane < kWarps ? scratch[lane] : 0.0f;
    block_total = device::warp::reduce_sum(block_total);
    if (lane == 0) {
      scratch[0] = block_total;
    }
  }
  __syncthreads();
  return scratch[0];
}

struct Sum2 {
  float x;
  float y;
};

SGL_DEVICE Sum2 warp_reduce_sum_pair(float x, float y) {
  return {device::warp::reduce_sum(x), device::warp::reduce_sum(y)};
}

template <int kReduceWarps>
SGL_DEVICE Sum2 block_reduce_sum2_for(float x, float y, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  const float warp_x = device::warp::reduce_sum(x);
  const float warp_y = device::warp::reduce_sum(y);
  if (lane == 0) {
    scratch[warp] = warp_x;
    scratch[kReduceWarps + warp] = warp_y;
  }
  __syncthreads();

  float block_x = 0.0f;
  float block_y = 0.0f;
  if (warp == 0) {
    block_x = lane < kReduceWarps ? scratch[lane] : 0.0f;
    block_y = lane < kReduceWarps ? scratch[kReduceWarps + lane] : 0.0f;
    block_x = device::warp::reduce_sum(block_x);
    block_y = device::warp::reduce_sum(block_y);
    if (lane == 0) {
      scratch[0] = block_x;
      scratch[1] = block_y;
    }
  }
  __syncthreads();
  return {scratch[0], scratch[1]};
}

SGL_DEVICE Sum2 block_reduce_sum2(float x, float y, float* scratch) {
  return block_reduce_sum2_for<kWarps>(x, y, scratch);
}

template <int kReduceWarps>
SGL_DEVICE float block_reduce_sum_active_for(float value, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float warp_total = 0.0f;
  if (warp < kReduceWarps) {
    warp_total = device::warp::reduce_sum(value);
  }
  if (lane == 0 && warp < kReduceWarps) {
    scratch[warp] = warp_total;
  }
  __syncthreads();

  float block_total = 0.0f;
  if (warp == 0) {
    block_total = lane < kReduceWarps ? scratch[lane] : 0.0f;
    block_total = device::warp::reduce_sum(block_total);
    if (lane == 0) {
      scratch[0] = block_total;
    }
  }
  __syncthreads();
  return scratch[0];
}

template <int kReduceWarps>
SGL_DEVICE Sum2 block_reduce_sum2_active_for(float x, float y, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float warp_x = 0.0f;
  float warp_y = 0.0f;
  if (warp < kReduceWarps) {
    warp_x = device::warp::reduce_sum(x);
    warp_y = device::warp::reduce_sum(y);
  }
  if (lane == 0 && warp < kReduceWarps) {
    scratch[warp] = warp_x;
    scratch[kReduceWarps + warp] = warp_y;
  }
  __syncthreads();

  float block_x = 0.0f;
  float block_y = 0.0f;
  if (warp == 0) {
    block_x = lane < kReduceWarps ? scratch[lane] : 0.0f;
    block_y = lane < kReduceWarps ? scratch[kReduceWarps + lane] : 0.0f;
    block_x = device::warp::reduce_sum(block_x);
    block_y = device::warp::reduce_sum(block_y);
    if (lane == 0) {
      scratch[0] = block_x;
      scratch[1] = block_y;
    }
  }
  __syncthreads();
  return {scratch[0], scratch[1]};
}

SGL_DEVICE float fp32_at(const float* ptr, int64_t idx) { return ptr[idx]; }

// One-token KDA (Kimi Delta Attention) decode step, fused end to end:
//  1. Causal conv1d update: q/k/v = SiLU(bias + w_{q,k,v}_t (dot) [conv_state,
//  x_{q,k,v}]),
//     shift-registers cs_q/cs_k/cs_v advanced in place (x_q/x_k/x_v are the raw
//     per-token projections, w_*_t the depthwise conv taps).
//  2. Per-head decay gate from a_log/g/dt_bias:
//       decay = exp(lower_bound * sigmoid(exp(a_log) * (g + dt_bias)))   if
//       kUseLowerBound
//             = exp(-exp(a_log) * softplus(g + dt_bias)) otherwise
//  3. q, k are L2-normalized over kDimK (q additionally scaled by `scale`);
//  beta
//     is sigmoid(raw) when kApplyBetaSigmoid, else used as-is.
//  4. Delta-rule recurrent state update per value row v of the [kDimV, kDimK]
//  state h:
//       h_decay = h * decay
//       v_new   = (v - h_decay . k) * beta
//       h'      = h_decay + k (outer) v_new        (written back to `state`)
//       o       = h' . q
//  5. Optional gated RMSNorm (onorm) over o: out = o * rsqrt(mean(o^2) +
//  onorm_eps)
//     * onorm_weight * sigmoid(onorm_g).
// Grid maps one block per (batch/token, value-head); kUseStaticDecodeLayout
// picks a fixed (B, HV) launch shape for CUDA-graph capture,
// cu_seqlens/ssm_state_indices otherwise resolve the token's batch slot and
// recurrent-state slot.
template <
    bool kApplyOnorm, bool kUseStaticDecodeLayout = false, int kFixedHeads = 0,
    int kFixedValueHeads = 0, bool kUseHeadGrid = false,
    bool kAccumulateOnormSumsq = false, bool kUseActiveQkReduction = false,
    bool kUseCacheGlobalStore = false, bool kComputeOutputBeforeStore = false,
    bool kSkipWarpSync = false, bool kPreloadOnormParams = false,
    bool kPrefetchNextStateChunk = false, bool kUseActiveOnormReduction = false,
    bool kUpdateConvState = false, bool kUseLowerBound = false,
    bool kApplyBetaSigmoid = true, bool kUseTmaLoad = false,
    int kTmaStages = kNumChunks, bool kUsePDL = false>
// Shapes below use K3's per-TP-rank sizing (linear_attn_config: num_heads=96
// over TP={8,16,32} -> H=HV={12,6,3} local heads; head_dim=128 ->
// kDimK=kDimV=128; kSeg = H*128; short_conv_kernel_size=4 -> kKernelWidth=4,
// kConvStateWidth=3). B is the live (post-padding, under
// kUseStaticDecodeLayout) decode batch size; slots is the recurrent-state /
// conv-cache pool capacity, addressed by ssm_state_indices, not B.
__global__
__launch_bounds__(kThreads, 2) void kda_decode_fusion_many_heads_kernel(
    const __nv_bfloat16* __restrict__ x_q,  // [B, H*128] row bos*x_row_stride +
                                            // hk, sliced from mixed_qkv
                                            // q-segment
    const __nv_bfloat16* __restrict__ x_k,  // [B, H*128] row bos*x_row_stride +
                                            // hk, sliced from mixed_qkv
                                            // k-segment
    const __nv_bfloat16* __restrict__ x_v,  // [B, HV*128] row bos*x_row_stride
                                            // + hvv, sliced from mixed_qkv
                                            // v-segment
    const float* __restrict__ w_q_t,  // [kKernelWidth=4, H*128] dense conv taps
                                      // for q, indexed w*hkv_dim + hk
    const float* __restrict__ w_k_t,  // [4, H*128] dense conv taps for k
    const float* __restrict__ w_v_t,  // [4, HV*128] dense conv taps for v
    const float* __restrict__ bias_q,  // [H*128] conv bias for q, sliced from
                                       // conv_bias
    const float* __restrict__ bias_k,  // [H*128] conv bias for k
    const float* __restrict__ bias_v,  // [HV*128] conv bias for v
    __nv_bfloat16* __restrict__ cs_q,  // [slots, kConvStateWidth=3, H*128] q
                                       // shift-register, sliced from
                                       // conv_states
    __nv_bfloat16* __restrict__ cs_k,  // [slots, 3, H*128] k shift-register
    __nv_bfloat16* __restrict__ cs_v,  // [slots, 3, HV*128] v shift-register
    const float* __restrict__ a_log,  // [H] per-head log-decay base, indexed by
                                      // i_h
    const __nv_bfloat16* __restrict__ g,  // [B, HV*128] raw forget gate, row
                                          // bos*g_row_stride + i_hv*kDimK + k
    const float* __restrict__ dt_bias,    // [H*128] gate bias added to g before
                                          // the decay nonlinearity
    const __nv_bfloat16* __restrict__ beta,     // [B, HV] raw beta logit, row
                                                // bos*beta_row_stride + i_hv
    const __nv_bfloat16* __restrict__ onorm_g,  // [B, HV*128] onorm sigmoid
                                                // gate, row
                                                // i_n*onormg_row_stride +
                                                // i_hv*128
                                                // + v
    const float* __restrict__ onorm_weight,     // [128] onorm RMSNorm scale,
                                                // shared across all heads
    const int* __restrict__ ssm_state_indices,  // [B] recurrent-state slot per
                                                // token; <0 marks a padded
                                                // cuda-graph slot
    const int* __restrict__ cu_seqlens,         // [B+1] token offsets into
                                         // x_q/x_k/x_v/g/beta; unused under
                                         // kUseStaticDecodeLayout
    float* __restrict__ state,  // [slots, HV, 128, 128] recurrent KDA state h,
                                // inner [V,K] contiguous, slot pitch =
                                // state_slot_stride
    __nv_bfloat16* __restrict__ out,  // [B, hv_count*128] fused-decode output,
                                      // row i_n, col i_hv*128 + v
    int B,   // live decode batch size (token count for this launch)
    int H,   // local key/query heads on this TP rank
    int HV,  // local value heads on this TP rank (H==HV since KDA is MHA not
             // GQA)
    float lower_bound,     // linear_attn_config.gate_lower_bound (-5.0 for K3)
                           // when kUseLowerBound
    float scale,           // query scale applied after L2-normalization
    float onorm_eps,       // onorm RMSNorm epsilon
    int64_t x_row_stride,  // element stride between consecutive tokens' rows in
                           // x_q/x_k/x_v (>= 3*H*128)
    int64_t g_row_stride,  // element stride between consecutive tokens' rows in
                           // g (>= HV*128)
    int64_t beta_row_stride,  // element stride between consecutive tokens' rows
                              // in beta (>= HV)
    int64_t onormg_row_stride,  // element stride between consecutive tokens'
                                // rows in onorm_g (>= HV*128)
    int64_t cs_slot_stride,     // element stride between consecutive slots in
                                // cs_q/cs_k/cs_v
    int64_t cs_w_stride,  // element stride between shift-register taps (w=0..2)
                          // within a slot
    int64_t
        state_slot_stride,  // element stride between consecutive slots in state
    // [RAON RDA patch] fp32 per-(token, head) value ratio
    // sigmoid(beta+delta)/sigmoid(beta), applied to the SiLU'd conv output
    // of v (the unfused chain's post-conv _scale_packed_v seam, folded
    // in-kernel because the fused kernel has no such seam). nullptr = off.
    const float* __restrict__ rda_ratio,  // [B, HV], row bos*rda_row_stride +
                                          // i_hv
    int64_t
        rda_row_stride) {  // element stride between consecutive slots in state
                           // (dense HV*128*128, or larger for shared pools)
  device::PDLWaitPrimary<kUsePDL>();
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  int i_n;
  int i_hv;
  int i_h;
  int bos;
  int slot;
  if constexpr (kUseStaticDecodeLayout) {
    if constexpr (kUseHeadGrid) {
      i_n = blockIdx.x;
      i_hv = blockIdx.y;
    } else {
      const int nhv = blockIdx.x;
      i_n = nhv / kFixedValueHeads;
      i_hv = nhv - i_n * kFixedValueHeads;
    }
    i_h = i_hv;
    bos = i_n;
    slot = ssm_state_indices == nullptr ? i_n : ssm_state_indices[i_n];
  } else {
    const int nhv = blockIdx.x;
    i_n = nhv / HV;
    i_hv = nhv - i_n * HV;
    const int hv_per_h = HV / H;
    i_h = i_hv / hv_per_h;

    bos = cu_seqlens == nullptr ? i_n : cu_seqlens[i_n];
    const int eos = cu_seqlens == nullptr ? i_n + 1 : cu_seqlens[i_n + 1];
    if (eos <= bos) {
      device::PDLTriggerSecondary<kUsePDL>();
      return;
    }
    slot = ssm_state_indices == nullptr ? i_n : ssm_state_indices[i_n];
  }

  if (slot < 0) {
    // Padded cuda-graph slot: zero the output row, leave the pools untouched.
    const int hv_count_pad = kUseStaticDecodeLayout ? kFixedValueHeads : HV;
    if (tid < kDimV) {
      out[(i_n * hv_count_pad + i_hv) * kDimV + tid] = __float2bfloat16(0.0f);
    }
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }

  const int hk_off = i_h * kDimK;
  const int hv_off = i_hv * kDimV;
  const int h_count = kUseStaticDecodeLayout ? kFixedHeads : H;
  const int hv_count = kUseStaticDecodeLayout ? kFixedValueHeads : HV;
  const int hkv_dim = h_count * kDimK;
  const int hvv_dim = hv_count * kDimV;

  // Dynamic size: 2 cp.async stages (32KB) or kTmaStages TMA stages (16KB per
  // stage; kTmaStages == kNumChunks means no buffer reuse).
  extern __shared__ __align__(16) float s_state[];
  __shared__ float s_q[kDimK];
  __shared__ float s_k[kDimK];
  __shared__ float s_decay[kDimK];
  __shared__ float s_v[kDimV];
  __shared__ float s_o[kDimV];
  __shared__ float s_reduce[kThreads];
  __shared__ float s_beta;
  __shared__ uint64_t s_tma_bar[kNumChunks];
  float pre_onorm_gate = 0.0f;
  float pre_onorm_weight = 0.0f;

  if constexpr (kUseTmaLoad) {
    // Chunk c lives in stage c % kTmaStages behind barrier c % kTmaStages
    // (wait parity (c / kTmaStages) & 1). Chunks 0/1 are issued here; chunk+2
    // is issued at the loop top while its stage is still fresh, and chunks
    // that reuse a stage are issued at the loop bottom behind a
    // __syncthreads(). Barrier-init visibility for waiting threads is
    // covered by the __syncthreads() before the chunk loop.
    if (tid == 0) {
#pragma unroll
      for (int c = 0; c < kTmaStages; ++c) {
        ptx::mbar_init(&s_tma_bar[c], 1);
      }
      tma_state_chunk_stage<kChunkV>(s_state, state, slot, i_hv,
                                     state_slot_stride, 0, 0, &s_tma_bar[0]);
      if (kTmaStages > 1 && kNumChunks > 1) {
        tma_state_chunk_stage<kChunkV>(s_state, state, slot, i_hv,
                                       state_slot_stride, 1, 1, &s_tma_bar[1]);
      }
    }
  } else {
    cp_async_state_chunk(s_state, state, slot, i_hv, state_slot_stride, 0);
  }

  if constexpr (kUpdateConvState) {
    if (tid < kDimK) {
      const int k = tid;
      const int hk = hk_off + k;
      const int64_t cs_base = slot * cs_slot_stride + hk;
      const int64_t xq_idx = bos * x_row_stride + hk;
      const float exp_a =
          __shfl_sync(0xffffffffu, lane == 0 ? __expf(a_log[i_h]) : 0.0f, 0);

      float q_acc = bias_q[hk];
      float k_acc = bias_k[hk];
      __nv_bfloat16 q_shift0 = __float2bfloat16(0.0f);
      __nv_bfloat16 q_shift1 = __float2bfloat16(0.0f);
      __nv_bfloat16 k_shift0 = __float2bfloat16(0.0f);
      __nv_bfloat16 k_shift1 = __float2bfloat16(0.0f);
#pragma unroll
      for (int w = 0; w < kConvStateWidth; ++w) {
        const __nv_bfloat16 q_state = cs_q[cs_base + w * cs_w_stride];
        const __nv_bfloat16 k_state = cs_k[cs_base + w * cs_w_stride];
        q_acc += __bfloat162float(q_state) * fp32_at(w_q_t, w * hkv_dim + hk);
        k_acc += __bfloat162float(k_state) * fp32_at(w_k_t, w * hkv_dim + hk);
        if (w == 1) {
          q_shift0 = q_state;
          k_shift0 = k_state;
        } else if (w == 2) {
          q_shift1 = q_state;
          k_shift1 = k_state;
        }
      }
      const __nv_bfloat16 q_new = x_q[xq_idx];
      const __nv_bfloat16 k_new = x_k[xq_idx];
      q_acc += __bfloat162float(q_new) *
               fp32_at(w_q_t, (kKernelWidth - 1) * hkv_dim + hk);
      k_acc += __bfloat162float(k_new) *
               fp32_at(w_k_t, (kKernelWidth - 1) * hkv_dim + hk);

      cs_q[cs_base + 0] = q_shift0;
      cs_q[cs_base + cs_w_stride] = q_shift1;
      cs_q[cs_base + 2 * cs_w_stride] = q_new;
      cs_k[cs_base + 0] = k_shift0;
      cs_k[cs_base + cs_w_stride] = k_shift1;
      cs_k[cs_base + 2 * cs_w_stride] = k_new;

      s_q[k] = device::math::silu_fast(q_acc);
      s_k[k] = device::math::silu_fast(k_acc);

      const float g_raw =
          bf16_load(g, bos * g_row_stride + i_hv * kDimK + k) + dt_bias[hk];
      if constexpr (kUseLowerBound) {
        s_decay[k] =
            __expf(lower_bound * device::math::sigmoid_fast(exp_a * g_raw));
      } else {
        s_decay[k] = __expf(-exp_a * device::math::softplus_fast(g_raw));
      }
    }
  } else {
    if (tid < kDimK) {
      const int k = tid;
      const int hk = hk_off + k;
      const float exp_a =
          __shfl_sync(0xffffffffu, lane == 0 ? __expf(a_log[i_h]) : 0.0f, 0);

      float q_acc = bias_q[hk];
      float k_acc = bias_k[hk];
#pragma unroll
      for (int w = 0; w < kConvStateWidth; ++w) {
        const int64_t cs_idx = slot * cs_slot_stride + hk + w * cs_w_stride;
        q_acc += bf16_load(cs_q, cs_idx) * fp32_at(w_q_t, w * hkv_dim + hk);
        k_acc += bf16_load(cs_k, cs_idx) * fp32_at(w_k_t, w * hkv_dim + hk);
      }
      q_acc += bf16_load(x_q, bos * x_row_stride + hk) *
               fp32_at(w_q_t, (kKernelWidth - 1) * hkv_dim + hk);
      k_acc += bf16_load(x_k, bos * x_row_stride + hk) *
               fp32_at(w_k_t, (kKernelWidth - 1) * hkv_dim + hk);

      s_q[k] = device::math::silu_fast(q_acc);
      s_k[k] = device::math::silu_fast(k_acc);

      const float g_raw =
          bf16_load(g, bos * g_row_stride + i_hv * kDimK + k) + dt_bias[hk];
      if constexpr (kUseLowerBound) {
        s_decay[k] =
            __expf(lower_bound * device::math::sigmoid_fast(exp_a * g_raw));
      } else {
        s_decay[k] = __expf(-exp_a * device::math::softplus_fast(g_raw));
      }
    }
  }

  if constexpr (kUpdateConvState) {
    if (tid < kDimV) {
      const int v = tid;
      const int hvv = hv_off + v;
      const int64_t cs_base = slot * cs_slot_stride + hvv;
      const int64_t xv_idx = bos * x_row_stride + hvv;

      float v_acc = bias_v[hvv];
      __nv_bfloat16 v_shift0 = __float2bfloat16(0.0f);
      __nv_bfloat16 v_shift1 = __float2bfloat16(0.0f);
#pragma unroll
      for (int w = 0; w < kConvStateWidth; ++w) {
        const __nv_bfloat16 v_state = cs_v[cs_base + w * cs_w_stride];
        v_acc += __bfloat162float(v_state) * fp32_at(w_v_t, w * hvv_dim + hvv);
        if (w == 1) {
          v_shift0 = v_state;
        } else if (w == 2) {
          v_shift1 = v_state;
        }
      }
      const __nv_bfloat16 v_new = x_v[xv_idx];
      v_acc += __bfloat162float(v_new) *
               fp32_at(w_v_t, (kKernelWidth - 1) * hvv_dim + hvv);
      cs_v[cs_base + 0] = v_shift0;
      cs_v[cs_base + cs_w_stride] = v_shift1;
      cs_v[cs_base + 2 * cs_w_stride] = v_new;
      float v_post = device::math::silu_fast(v_acc);
      // [RAON RDA patch] post-conv, post-SiLU v scale (matches
      // _scale_packed_v applied to the causal_conv1d_update output).
      if (rda_ratio != nullptr) {
        v_post *= rda_ratio[bos * rda_row_stride + i_hv];
      }
      s_v[v] = v_post;

      if constexpr (kApplyOnorm && kPreloadOnormParams) {
        const int64_t onorm_idx = i_n * onormg_row_stride + i_hv * kDimV + v;
        pre_onorm_gate =
            device::math::sigmoid_fast(bf16_load(onorm_g, onorm_idx));
        pre_onorm_weight = onorm_weight[v];
      }
    }
  } else {
    if (tid < kDimV) {
      const int v = tid;
      const int hvv = hv_off + v;

      float v_acc = bias_v[hvv];
#pragma unroll
      for (int w = 0; w < kConvStateWidth; ++w) {
        const int64_t cs_idx = slot * cs_slot_stride + hvv + w * cs_w_stride;
        v_acc += bf16_load(cs_v, cs_idx) * fp32_at(w_v_t, w * hvv_dim + hvv);
      }
      v_acc += bf16_load(x_v, bos * x_row_stride + hvv) *
               fp32_at(w_v_t, (kKernelWidth - 1) * hvv_dim + hvv);
      float v_post = device::math::silu_fast(v_acc);
      // [RAON RDA patch] post-conv, post-SiLU v scale (matches
      // _scale_packed_v applied to the causal_conv1d_update output).
      if (rda_ratio != nullptr) {
        v_post *= rda_ratio[bos * rda_row_stride + i_hv];
      }
      s_v[v] = v_post;

      if constexpr (kApplyOnorm && kPreloadOnormParams) {
        const int64_t onorm_idx = i_n * onormg_row_stride + i_hv * kDimV + v;
        pre_onorm_gate =
            device::math::sigmoid_fast(bf16_load(onorm_g, onorm_idx));
        pre_onorm_weight = onorm_weight[v];
      }
    }
  }

  if (tid == 0) {
    const float beta_raw = bf16_load(beta, bos * beta_row_stride + i_hv);
    if constexpr (kApplyBetaSigmoid) {
      s_beta = device::math::sigmoid_fast(beta_raw);
    } else {
      s_beta = beta_raw;
    }
  }
  __syncthreads();

  if constexpr (!kUseTmaLoad && kPrefetchNextStateChunk && kNumChunks > 1) {
    cp_async_state_chunk(s_state, state, slot, i_hv, state_slot_stride, 1);
  }

  const float q_sq = tid < kDimK ? s_q[tid] * s_q[tid] : 0.0f;
  const float k_sq = tid < kDimK ? s_k[tid] * s_k[tid] : 0.0f;
  Sum2 qk_sum;
  if constexpr (kUseActiveQkReduction) {
    qk_sum = block_reduce_sum2_active_for<kDimK / 32>(q_sq, k_sq, s_reduce);
  } else {
    qk_sum = block_reduce_sum2(q_sq, k_sq, s_reduce);
  }
  if (tid < kDimK) {
    s_q[tid] *= rsqrtf(qk_sum.x + 1.0e-6f) * scale;
    s_k[tid] *= rsqrtf(qk_sum.y + 1.0e-6f);
  }
  __syncthreads();

  const int k_base = lane * 4;
  const float4 q4 = *reinterpret_cast<const float4*>(s_q + k_base);
  const float4 k4 = *reinterpret_cast<const float4*>(s_k + k_base);
  const float4 decay4 = *reinterpret_cast<const float4*>(s_decay + k_base);
  float r_q[4] = {q4.x, q4.y, q4.z, q4.w};
  float r_k[4] = {k4.x, k4.y, k4.z, k4.w};
  float r_decay[4] = {decay4.x, decay4.y, decay4.z, decay4.w};
  float o_sumsq = 0.0f;

#pragma unroll
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    if constexpr (kUseTmaLoad) {
      if (tid == 0 && chunk + 2 < kNumChunks && chunk + 2 < kTmaStages) {
        tma_state_chunk_stage<kChunkV>(s_state, state, slot, i_hv,
                                       state_slot_stride, chunk + 2, chunk + 2,
                                       &s_tma_bar[chunk + 2]);
      }
      ptx::mbar_wait_parity(&s_tma_bar[chunk % kTmaStages],
                            (chunk / kTmaStages) & 1);
    } else if constexpr (kPrefetchNextStateChunk && kNumChunks > 1) {
      if (chunk + 1 < kNumChunks) {
        ptx::cp_async_wait_group<1>();
      } else {
        ptx::cp_async_wait_all();
      }
    } else {
      ptx::cp_async_wait_all();
    }
    if constexpr (!kUseTmaLoad && !kSkipWarpSync) {
      __syncwarp();
    }

    if constexpr (!kUseTmaLoad && !kPrefetchNextStateChunk) {
      if (chunk + 1 < kNumChunks) {
        cp_async_state_chunk(s_state, state, slot, i_hv, state_slot_stride,
                             chunk + 1);
      }
    }

    const float* state_stage =
        s_state +
        (kUseTmaLoad ? (chunk % kTmaStages) : (chunk & 1)) * kChunkV * kDimK;

#pragma unroll
    for (int row = 0; row < kRowsPerWarp; row += 2) {
      const int v_row_a = warp + row * kWarps;
      const int v_row_b = warp + (row + 1) * kWarps;
      const int v0 = chunk * kChunkV + v_row_a;
      const int v1 = chunk * kChunkV + v_row_b;
      float h_a_vals[4];
      float h_b_vals[4];
      float dot_hk_a = 0.0f;
      float dot_hk_b = 0.0f;

      const float4 raw_h_a = *reinterpret_cast<const float4*>(
          state_stage + v_row_a * kDimK + k_base);
      const float4 raw_h_b = *reinterpret_cast<const float4*>(
          state_stage + v_row_b * kDimK + k_base);
      h_a_vals[0] = raw_h_a.x * r_decay[0];
      h_a_vals[1] = raw_h_a.y * r_decay[1];
      h_a_vals[2] = raw_h_a.z * r_decay[2];
      h_a_vals[3] = raw_h_a.w * r_decay[3];
      h_b_vals[0] = raw_h_b.x * r_decay[0];
      h_b_vals[1] = raw_h_b.y * r_decay[1];
      h_b_vals[2] = raw_h_b.z * r_decay[2];
      h_b_vals[3] = raw_h_b.w * r_decay[3];
      dot_hk_a = h_a_vals[0] * r_k[0] + h_a_vals[1] * r_k[1] +
                 h_a_vals[2] * r_k[2] + h_a_vals[3] * r_k[3];
      dot_hk_b = h_b_vals[0] * r_k[0] + h_b_vals[1] * r_k[1] +
                 h_b_vals[2] * r_k[2] + h_b_vals[3] * r_k[3];

      const Sum2 dot_hk = warp_reduce_sum_pair(dot_hk_a, dot_hk_b);
      const float v_new0 = (s_v[v0] - dot_hk.x) * s_beta;
      const float v_new1 = (s_v[v1] - dot_hk.y) * s_beta;

      float dot_hq_a = 0.0f;
      float dot_hq_b = 0.0f;
      // Writeback mirrors the load addressing: slot pitch from the host stride
      // (int64, envelope-safe), intra-slot offset i_hv*V*K + v*K + k
      // contiguous.
      const int64_t slot_base_wb =
          static_cast<int64_t>(slot) * state_slot_stride;
      const int64_t state_idx_a =
          slot_base_wb + ((i_hv * kDimV + v0) * kDimK + k_base);
      const int64_t state_idx_b =
          slot_base_wb + ((i_hv * kDimV + v1) * kDimK + k_base);
      const float h_a_0 = h_a_vals[0] + r_k[0] * v_new0;
      const float h_a_1 = h_a_vals[1] + r_k[1] * v_new0;
      const float h_a_2 = h_a_vals[2] + r_k[2] * v_new0;
      const float h_a_3 = h_a_vals[3] + r_k[3] * v_new0;
      const float h_b_0 = h_b_vals[0] + r_k[0] * v_new1;
      const float h_b_1 = h_b_vals[1] + r_k[1] * v_new1;
      const float h_b_2 = h_b_vals[2] + r_k[2] * v_new1;
      const float h_b_3 = h_b_vals[3] + r_k[3] * v_new1;
      if constexpr (kComputeOutputBeforeStore) {
        dot_hq_a =
            h_a_0 * r_q[0] + h_a_1 * r_q[1] + h_a_2 * r_q[2] + h_a_3 * r_q[3];
        dot_hq_b =
            h_b_0 * r_q[0] + h_b_1 * r_q[1] + h_b_2 * r_q[2] + h_b_3 * r_q[3];
        store_state_float4<kUseCacheGlobalStore>(
            state + state_idx_a, make_float4(h_a_0, h_a_1, h_a_2, h_a_3));
        store_state_float4<kUseCacheGlobalStore>(
            state + state_idx_b, make_float4(h_b_0, h_b_1, h_b_2, h_b_3));
      } else {
        store_state_float4<kUseCacheGlobalStore>(
            state + state_idx_a, make_float4(h_a_0, h_a_1, h_a_2, h_a_3));
        store_state_float4<kUseCacheGlobalStore>(
            state + state_idx_b, make_float4(h_b_0, h_b_1, h_b_2, h_b_3));
        dot_hq_a =
            h_a_0 * r_q[0] + h_a_1 * r_q[1] + h_a_2 * r_q[2] + h_a_3 * r_q[3];
        dot_hq_b =
            h_b_0 * r_q[0] + h_b_1 * r_q[1] + h_b_2 * r_q[2] + h_b_3 * r_q[3];
      }

      const Sum2 dot_hq = warp_reduce_sum_pair(dot_hq_a, dot_hq_b);
      if (lane == 0) {
        s_o[v0] = dot_hq.x;
        s_o[v1] = dot_hq.y;
        if constexpr (kApplyOnorm && kAccumulateOnormSumsq) {
          o_sumsq += dot_hq.x * dot_hq.x + dot_hq.y * dot_hq.y;
        }
      }
    }

    if constexpr (kUseTmaLoad && kTmaStages < kNumChunks) {
      // chunk + kTmaStages reuses the stage this chunk just read; every warp
      // must be done with it before the single issuing thread overwrites it.
      if (chunk + kTmaStages < kNumChunks) {
        __syncthreads();
        if (tid == 0) {
          tma_state_chunk_stage<kChunkV>(
              s_state, state, slot, i_hv, state_slot_stride, chunk + kTmaStages,
              chunk % kTmaStages, &s_tma_bar[chunk % kTmaStages]);
        }
      }
    } else if constexpr (!kUseTmaLoad && kPrefetchNextStateChunk) {
      if (chunk + 2 < kNumChunks) {
        cp_async_state_chunk(s_state, state, slot, i_hv, state_slot_stride,
                             chunk + 2);
      }
    }
  }
  __syncthreads();

  device::PDLTriggerSecondary<kUsePDL>();

  if constexpr (kApplyOnorm) {
    if constexpr (kAccumulateOnormSumsq) {
      if (lane == 0) {
        s_reduce[warp] = o_sumsq;
      }
      __syncthreads();

      float total_sumsq = 0.0f;
      if (warp == 0) {
        total_sumsq = lane < kWarps ? s_reduce[lane] : 0.0f;
        total_sumsq = device::warp::reduce_sum(total_sumsq);
        if (lane == 0) {
          s_reduce[0] = total_sumsq;
        }
      }
      __syncthreads();

      if (tid < kDimV) {
        const int out_idx = (i_n * hv_count + i_hv) * kDimV + tid;
        const float raw_o = s_o[tid];
        const float rstd =
            rsqrtf(s_reduce[0] / static_cast<float>(kDimV) + onorm_eps);
        float gate;
        float weight;
        if constexpr (kPreloadOnormParams) {
          gate = pre_onorm_gate;
          weight = pre_onorm_weight;
        } else {
          gate = device::math::sigmoid_fast(
              bf16_load(onorm_g, i_n * onormg_row_stride + i_hv * kDimV + tid));
          weight = onorm_weight[tid];
        }
        const float y = raw_o * rstd * weight * gate;
        out[out_idx] = bf16_store(y);
      }
    } else {
      const float raw_o = tid < kDimV ? s_o[tid] : 0.0f;
      const float o_sq = raw_o * raw_o;
      float sumsq;
      if constexpr (kUseActiveOnormReduction || kUseActiveQkReduction) {
        sumsq = block_reduce_sum_active_for<kDimV / 32>(o_sq, s_reduce);
      } else {
        sumsq = block_reduce_sum(o_sq, s_reduce);
      }

      if (tid < kDimV) {
        const int out_idx = (i_n * hv_count + i_hv) * kDimV + tid;
        const float rstd =
            rsqrtf(sumsq / static_cast<float>(kDimV) + onorm_eps);
        float gate;
        float weight;
        if constexpr (kPreloadOnormParams) {
          gate = pre_onorm_gate;
          weight = pre_onorm_weight;
        } else {
          gate = device::math::sigmoid_fast(
              bf16_load(onorm_g, i_n * onormg_row_stride + i_hv * kDimV + tid));
          weight = onorm_weight[tid];
        }
        const float y = raw_o * rstd * weight * gate;
        out[out_idx] = bf16_store(y);
      }
    }
  } else {
    if (tid < kDimV) {
      const int out_idx = (i_n * hv_count + i_hv) * kDimV + tid;
      out[out_idx] = bf16_store(s_o[tid]);
    }
  }
}

// K3 decode configuration of the many-heads kernel: onorm fused, static
// H = HV in {3, 6, 12} layout with a (B, HV) grid, onorm params preloaded, next
// state chunk prefetched, active onorm reduction, conv cache updated in place,
// beta sigmoid in-kernel. Both forget-gate variants are compiled (softplus
// and lower-bounded sigmoid) and selected at launch from the model config.
// kUseTmaLoad/kTmaStages select the 1D-TMA state-staging path in place of the
// cp.async fallback used for a misaligned recurrent-state slot stride.
template <int kFixedHeads, bool kUseLowerBound, bool kUsePDL,
          bool kUseTmaLoad = false, int kTmaStages = kNumChunks>
constexpr auto kda_fused_decode_k3_kernel = kda_decode_fusion_many_heads_kernel<
    /*kApplyOnorm=*/true,
    /*kUseStaticDecodeLayout=*/true,
    /*kFixedHeads=*/kFixedHeads,
    /*kFixedValueHeads=*/kFixedHeads,
    /*kUseHeadGrid=*/true,
    /*kAccumulateOnormSumsq=*/false,
    /*kUseActiveQkReduction=*/false,
    /*kUseCacheGlobalStore=*/false,
    /*kComputeOutputBeforeStore=*/false,
    /*kSkipWarpSync=*/false,
    /*kPreloadOnormParams=*/true,
    /*kPrefetchNextStateChunk=*/true,
    /*kUseActiveOnormReduction=*/true,
    /*kUpdateConvState=*/true, kUseLowerBound,
    /*kApplyBetaSigmoid=*/true, kUseTmaLoad, kTmaStages, kUsePDL>;

template <int kFixedHeads, bool kUsePDL>
auto select_kda_fused_decode_k3_kernel(bool use_lower_bound, int tma_stages) {
  if (tma_stages == 3) {
    return use_lower_bound
               ? kda_fused_decode_k3_kernel<kFixedHeads, true, kUsePDL, true, 3>
               : kda_fused_decode_k3_kernel<kFixedHeads, false, kUsePDL, true,
                                            3>;
  }
  if (tma_stages == 4) {
    return use_lower_bound
               ? kda_fused_decode_k3_kernel<kFixedHeads, true, kUsePDL, true, 4>
               : kda_fused_decode_k3_kernel<kFixedHeads, false, kUsePDL, true,
                                            4>;
  }
  return use_lower_bound
             ? kda_fused_decode_k3_kernel<kFixedHeads, true, kUsePDL>
             : kda_fused_decode_k3_kernel<kFixedHeads, false, kUsePDL>;
}

}  // namespace sglang

// ---------------------------------------------------------------------------
// [RAON port] torch-extension host wrapper (replaces the tvm-ffi
// KdaFusedDecodeKernel struct; the TensorMatcher contract is preserved as
// TORCH_CHECK calls). kUsePDL is pinned false: PDL is an upstream launch
// optimization that needs the surrounding launches to cooperate; it is not
// required for correctness and is out of scope for this port.
// ---------------------------------------------------------------------------

namespace {

void kda_fused_decode_sgl(
    at::Tensor mixed_qkv,     // [B, 3*H*128] bf16, row-strided ok
    at::Tensor a,             // [B, H*128] bf16 raw forget gate, row-strided ok
    at::Tensor b,             // [B, H] bf16 raw beta logits, row-strided ok
    at::Tensor conv_states,   // [slots, 3, 3*H*128] bf16 pool (time-major)
    at::Tensor w_q_t,         // [4, H*128] fp32 dense (width-major taps)
    at::Tensor w_k_t,         // [4, H*128] fp32 dense
    at::Tensor w_v_t,         // [4, H*128] fp32 dense
    at::Tensor conv_bias,     // [3*H*128] fp32 (zeros if the conv has no bias)
    at::Tensor A_log,         // [H] fp32
    at::Tensor dt_bias,       // [H*128] fp32
    at::Tensor onorm_g,       // [B, H*128] bf16, row-strided ok
    at::Tensor onorm_weight,  // [128] fp32
    at::Tensor state,         // [slots, H, 128, 128] fp32, inner contiguous
    at::Tensor indices,       // [B] int32 (< 0 = padded cuda-graph slot)
    at::Tensor out,           // [B, H*128] bf16 dense
    std::optional<at::Tensor> rda_ratio,  // [B, H] fp32, row-strided ok
    double scale, double onorm_eps, double lower_bound, bool use_lower_bound) {
  const int64_t kH = A_log.size(0);
  TORCH_CHECK(kH == 3 || kH == 6 || kH == 12 || kH == 40,
              "kda_fused_decode_sgl: instantiated local head counts are "
              "3/6/12/40, got ",
              kH);
  const int64_t kSeg = kH * 128;
  const int64_t B = mixed_qkv.size(0);

  const auto dev = mixed_qkv.device();
  auto chk = [&](const at::Tensor& t, const char* n) {
    TORCH_CHECK(t.is_cuda() && t.device() == dev, n, ": wrong device");
  };
  chk(mixed_qkv, "mixed_qkv");
  chk(a, "a");
  chk(b, "b");
  chk(conv_states, "conv_states");
  chk(w_q_t, "w_q_t");
  chk(w_k_t, "w_k_t");
  chk(w_v_t, "w_v_t");
  chk(conv_bias, "conv_bias");
  chk(A_log, "A_log");
  chk(dt_bias, "dt_bias");
  chk(onorm_g, "onorm_g");
  chk(onorm_weight, "onorm_weight");
  chk(state, "state");
  chk(indices, "indices");
  chk(out, "out");

  TORCH_CHECK(mixed_qkv.dim() == 2 && mixed_qkv.size(1) == 3 * kSeg &&
                  mixed_qkv.scalar_type() == at::kBFloat16 &&
                  mixed_qkv.stride(1) == 1,
              "mixed_qkv contract");
  TORCH_CHECK(a.dim() == 2 && a.size(0) == B && a.size(1) == kSeg &&
                  a.scalar_type() == at::kBFloat16 && a.stride(1) == 1,
              "a (raw forget gate) contract");
  TORCH_CHECK(b.dim() == 2 && b.size(0) == B && b.size(1) == kH &&
                  b.scalar_type() == at::kBFloat16 && b.stride(1) == 1,
              "b (raw beta) contract");
  TORCH_CHECK(conv_states.dim() == 3 && conv_states.size(1) == 3 &&
                  conv_states.size(2) == 3 * kSeg &&
                  conv_states.scalar_type() == at::kBFloat16 &&
                  conv_states.stride(2) == 1,
              "conv_states contract");
  for (const auto* wt : {&w_q_t, &w_k_t, &w_v_t}) {
    TORCH_CHECK(wt->dim() == 2 && wt->size(0) == 4 && wt->size(1) == kSeg &&
                    wt->scalar_type() == at::kFloat && wt->is_contiguous(),
                "conv tap contract");
  }
  TORCH_CHECK(conv_bias.dim() == 1 && conv_bias.size(0) == 3 * kSeg &&
                  conv_bias.scalar_type() == at::kFloat &&
                  conv_bias.is_contiguous(),
              "conv_bias contract");
  TORCH_CHECK(A_log.scalar_type() == at::kFloat && A_log.is_contiguous(),
              "A_log contract");
  TORCH_CHECK(dt_bias.dim() == 1 && dt_bias.size(0) == kSeg &&
                  dt_bias.scalar_type() == at::kFloat &&
                  dt_bias.is_contiguous(),
              "dt_bias contract");
  TORCH_CHECK(
      onorm_g.dim() == 2 && onorm_g.size(0) == B && onorm_g.size(1) == kSeg &&
          onorm_g.scalar_type() == at::kBFloat16 && onorm_g.stride(1) == 1,
      "onorm_g contract");
  TORCH_CHECK(onorm_weight.dim() == 1 && onorm_weight.size(0) == 128 &&
                  onorm_weight.scalar_type() == at::kFloat &&
                  onorm_weight.is_contiguous(),
              "onorm_weight contract");
  TORCH_CHECK(
      state.dim() == 4 && state.size(1) == kH && state.size(2) == 128 &&
          state.size(3) == 128 && state.scalar_type() == at::kFloat &&
          state.stride(3) == 1 && state.stride(2) == 128 &&
          state.stride(1) == 128 * 128,
      "state contract (inner [HV, V, K] must be dense; slot pitch free)");
  TORCH_CHECK(indices.dim() == 1 && indices.size(0) == B &&
                  indices.scalar_type() == at::kInt && indices.is_contiguous(),
              "indices contract");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == B && out.size(1) == kSeg &&
                  out.scalar_type() == at::kBFloat16 && out.stride(1) == 1 &&
                  out.stride(0) == kSeg,
              "out contract (dense)");

  const float* rda_ptr = nullptr;
  int64_t rda_stride = 0;
  if (rda_ratio.has_value()) {
    const at::Tensor& r = rda_ratio.value();
    chk(r, "rda_ratio");
    TORCH_CHECK(r.dim() == 2 && r.size(0) == B && r.size(1) == kH &&
                    r.scalar_type() == at::kFloat && r.stride(1) == 1,
                "rda_ratio contract");
    rda_ptr = r.data_ptr<float>();
    rda_stride = r.stride(0);
  }

  if (B == 0) return;

  const auto* mixed_ptr =
      reinterpret_cast<const __nv_bfloat16*>(mixed_qkv.data_ptr());
  auto* cs_ptr = reinterpret_cast<__nv_bfloat16*>(conv_states.data_ptr());
  const auto* bias_ptr = conv_bias.data_ptr<float>();
  const int64_t state_slot_stride = state.stride(0);
  // Both state staging paths (TMA 1D-bulk AND the cp.async fallback) issue
  // 16B loads from state + slot*stride, so every slot base must be 16B
  // aligned: fp32 pitch % 4 == 0.  (Upstream's K3 envelope and dense pitches
  // both satisfy this; reject rather than address incorrectly.)
  TORCH_CHECK(state_slot_stride % 4 == 0,
              "state slot pitch must be a multiple of 4 elements (16B state "
              "loads); got ",
              state_slot_stride);

  // Upstream TMA staging policy: 1D-bulk needs every slot base 16B-aligned.
  int tma_stages = 0;
  if ((state_slot_stride % 4) == 0) {
    tma_stages = (B * kH >= 512) ? 3 : 4;
  }

  const bool use_lb = use_lower_bound;
  auto kernel =
      kH == 3
          ? sglang::select_kda_fused_decode_k3_kernel<3, false>(use_lb,
                                                                tma_stages)
          : (kH == 6
                 ? sglang::select_kda_fused_decode_k3_kernel<6, false>(
                       use_lb, tma_stages)
                 : (kH == 12
                        ? sglang::select_kda_fused_decode_k3_kernel<12, false>(
                              use_lb, tma_stages)
                        : sglang::select_kda_fused_decode_k3_kernel<40, false>(
                              use_lb, tma_stages)));

  const int smem_stages = tma_stages == 0 ? 2 : tma_stages;
  const size_t smem_bytes = static_cast<size_t>(smem_stages) * sglang::kChunkV *
                            sglang::kDimK * sizeof(float);
  C10_CUDA_CHECK(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(smem_bytes)));

  const c10::cuda::CUDAGuard guard(dev);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned>(B), static_cast<unsigned>(kH));
  kernel<<<grid, dim3(sglang::kThreads), smem_bytes, stream>>>(
      /*x_q=*/mixed_ptr,
      /*x_k=*/mixed_ptr + kSeg,
      /*x_v=*/mixed_ptr + 2 * kSeg, w_q_t.data_ptr<float>(),
      w_k_t.data_ptr<float>(), w_v_t.data_ptr<float>(),
      /*bias_q=*/bias_ptr,
      /*bias_k=*/bias_ptr + kSeg,
      /*bias_v=*/bias_ptr + 2 * kSeg,
      /*cs_q=*/cs_ptr,
      /*cs_k=*/cs_ptr + kSeg,
      /*cs_v=*/cs_ptr + 2 * kSeg, A_log.data_ptr<float>(),
      /*g=*/reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
      dt_bias.data_ptr<float>(),
      /*beta=*/reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(onorm_g.data_ptr()),
      onorm_weight.data_ptr<float>(), indices.data_ptr<int32_t>(),
      /*cu_seqlens=*/static_cast<const int32_t*>(nullptr),
      state.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      static_cast<int>(B),
      /*H=*/static_cast<int>(kH),
      /*HV=*/static_cast<int>(kH), static_cast<float>(lower_bound),
      static_cast<float>(scale), static_cast<float>(onorm_eps),
      mixed_qkv.stride(0), a.stride(0), b.stride(0), onorm_g.stride(0),
      conv_states.stride(0), conv_states.stride(1), state_slot_stride, rda_ptr,
      rda_stride);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &kda_fused_decode_sgl,
        "Fused KDA decode step (conv1d update + RDA v-scale + delta-rule "
        "recurrence + sigmoid-gated RMSNorm), vendored from SGLang @ f65c70bb");
}
