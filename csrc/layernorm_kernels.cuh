// Shared kernel templates for RMS LayerNorm (unfused and fused-residual).
//
// Two template families are provided:
//
//   rms_norm_kernel<scalar_t, out_t, VEC_SIZE, NUM_DIMS>
//     Unfused RMS norm (no residual).
//     When out_t == scalar_t  → plain rms_norm  (vectorised scalar write)
//     Otherwise               → rms_norm_static_fp8_quant (vectorised fp8
//     write)
//
//   fused_add_rms_norm_kernel<scalar_t, out_t, VEC_SIZE>
//     Fused residual-add + RMS norm.
//     When out_t == scalar_t  → normalised output written back to `input`
//     Otherwise               → normalised output written to `out_quant` (fp8)
//
// Using a single template per family eliminates ~200 lines of duplicated
// variance-reduction code that previously existed between
// layernorm_kernels.cu and layernorm_quant_kernels.cu.
//
// Vectorisation:
//   - Unfused kernels use vectorize_read_with_alignment<VEC_SIZE> for the
//     variance pass, both scalar and fp8 branches.
//   - Fused kernels use vec_n_t<scalar_t, VEC_SIZE> for both passes.
//     VEC_SIZE = 16/sizeof(scalar_t), giving VEC_SIZE=8 for fp16/bf16 and
//     VEC_SIZE=4 for fp32 (FP32 was previously unvectorised).
//   - Output writes:
//       scalar branch → vec_n_t<scalar_t, VEC_SIZE> (128-bit store)
//       fp8 branch    → q8_n_t<fp8_t, VEC_SIZE>     (vectorised fp8 store,
//                       previously scalar-per-element)
#pragma once

#include "cub_helpers.h"
#include "libtorch_stable/quantization/vectorization.cuh"
#include "libtorch_stable/quantization/vectorization_utils.cuh"
// scaled_fp8_conversion is used inside an `if constexpr` branch conditioned on
// out_t != scalar_t.  It must still be declared so that the discarded branch
// can be parsed.  The include is therefore unconditional.
#include "quantization/w8a8/fp8/common.cuh"

namespace vllm {

// ============================================================================
// rms_norm_kernel — unfused; shared between scalar and fp8-quant variants
// ============================================================================
// Template parameters:
//   scalar_t  — input (and weight) element type (float / half / bfloat16)
//   out_t     — output element type
//               • scalar_t         → plain rms_norm, vectorised scalar write
//               • fp8 c10 type     → rms_norm_fp8_quant, vectorised fp8 write
//   VEC_SIZE  — elements per 128-bit vector (computed as gcd(16/sizeof, hid))
//   NUM_DIMS  — tensor rank (2, 3, or 4); only 2 is used by the fp8 variant
//
// The `scale` argument is only dereferenced when out_t != scalar_t; pass
// nullptr for the plain scalar case (the branch is compiled away).
// ============================================================================
template <typename scalar_t, typename out_t, int VEC_SIZE, int NUM_DIMS>
__global__ void rms_norm_kernel(
    out_t* __restrict__ out, const scalar_t* __restrict__ input,
    const int64_t input_stride_d2, const int64_t input_stride_d3,
    const int64_t input_stride_d4, const int64_t input_shape_d2,
    const int64_t input_shape_d3, const scalar_t* __restrict__ weight,
    const float* __restrict__ scale,  // null when out_t == scalar_t
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  // Determine the start of this block's input row (handles non-contiguous
  // multi-dimensional inputs for qk-norm use-cases).
  const scalar_t* input_row;
  if constexpr (NUM_DIMS == 2) {
    input_row = input + blockIdx.x * input_stride_d2;
  } else if constexpr (NUM_DIMS == 3) {
    int batch_idx = blockIdx.x / input_shape_d2;
    int head_idx = blockIdx.x % input_shape_d2;
    input_row =
        input + batch_idx * input_stride_d3 + head_idx * input_stride_d2;
  } else {  // NUM_DIMS == 4
    int batch_idx = blockIdx.x / (input_shape_d3 * input_shape_d2);
    int remaining = blockIdx.x % (input_shape_d3 * input_shape_d2);
    int seq_idx = remaining / input_shape_d2;
    int head_idx = remaining % input_shape_d2;
    input_row = input + batch_idx * input_stride_d4 +
                seq_idx * input_stride_d3 + head_idx * input_stride_d2;
  }

  // ---- Pass 1: accumulate per-thread sum-of-squares (vectorised) -----------
  auto vec_op = [&variance](const vec_n_t<scalar_t, VEC_SIZE>& vec) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float x = static_cast<float>(vec.val[i]);
      variance += x * x;
    }
  };
  auto scalar_op = [&variance](const scalar_t& val) {
    float x = static_cast<float>(val);
    variance += x * x;
  };
  vllm::vectorize_read_with_alignment<VEC_SIZE>(
      input_row, hidden_size, threadIdx.x, blockDim.x, vec_op, scalar_op);

  // ---- Block-wide reduction -------------------------------------------------
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  // ---- Pass 2: normalise and write output -----------------------------------
  const int64_t out_base = static_cast<int64_t>(blockIdx.x) * hidden_size;
  auto* v_in = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(input_row);
  auto* v_w = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(weight);

  if constexpr (std::is_same_v<out_t, scalar_t>) {
    // Vectorised scalar write (unchanged from original rms_norm_kernel)
    auto* v_out = reinterpret_cast<vec_n_t<scalar_t, VEC_SIZE>*>(
        reinterpret_cast<scalar_t*>(out) + out_base);
    for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += blockDim.x) {
      vec_n_t<scalar_t, VEC_SIZE> src1 = v_in[i];
      vec_n_t<scalar_t, VEC_SIZE> src2 = v_w[i];
      vec_n_t<scalar_t, VEC_SIZE> dst;
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        float x = static_cast<float>(src1.val[j]);
        dst.val[j] = static_cast<scalar_t>(x * s_variance) * src2.val[j];
      }
      v_out[i] = dst;
    }
  } else {
    // Vectorised fp8 write (previously scalar per-element;
    // q8_n_t<out_t, VEC_SIZE> issues a single VEC_SIZE-byte store)
    float const scale_inv = 1.0f / *scale;
    using vout_t = q8_n_t<out_t, VEC_SIZE>;
    auto* v_out = reinterpret_cast<vout_t*>(out + out_base);
    for (int idx = threadIdx.x; idx < hidden_size / VEC_SIZE;
         idx += blockDim.x) {
      vec_n_t<scalar_t, VEC_SIZE> src1 = v_in[idx];
      vec_n_t<scalar_t, VEC_SIZE> src2 = v_w[idx];
      vout_t out_vec;
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        float x = static_cast<float>(src1.val[j]);
        float norm = static_cast<float>(static_cast<scalar_t>(x * s_variance) *
                                        src2.val[j]);
        out_vec.val[j] = scaled_fp8_conversion<true, out_t>(norm, scale_inv);
      }
      v_out[idx] = out_vec;
    }
  }
}

// ============================================================================
// fused_add_rms_norm_kernel — fused residual-add; shared between variants
// ============================================================================
// Computes:
//   residual[i] = input[i] + residual[i]       (contiguous residual, in-place)
//   output[i]   = rms_norm(residual[i]) * weight[i]
//
// When out_t == scalar_t:
//   The normalised output is written back to `input` with stride
//   (mirrors the original fused_add_rms_norm behaviour).
//   Pass `out_quant = nullptr`.
//
// When out_t is an fp8 type:
//   The quantised normalised output is written to `out_quant` (contiguous).
//   `residual` is updated in-place in Pass 1 (same as the scalar path).
//   `input` is read but never written; normalised output goes to `out_quant`.
//   `scale` must be non-null; pass a valid fp8 pointer for `out_quant`.
//
// Vectorisation (NEW vs. old _f16Vec approach):
//   Uses vec_n_t<scalar_t, VEC_SIZE> instead of _f16Vec<scalar_t, width>,
//   which:
//   - Enables FP32 vectorisation (VEC_SIZE=4) — FP32 was previously scalar
//   - Removes the enable_if specialisation pattern
//   - Shares the same vec_n_t infrastructure as rms_norm_kernel above
//
// The `scale` argument is only dereferenced when out_t != scalar_t.
// ============================================================================
template <typename scalar_t, typename out_t, int VEC_SIZE>
__global__ void fused_add_rms_norm_kernel(
    out_t* __restrict__ out_quant,  // fp8 output; null/unused for scalar
    scalar_t* __restrict__ input,   // strided; receives scalar output
    const int64_t input_stride,
    scalar_t* __restrict__ residual,  // contiguous; updated in-place
    const scalar_t* __restrict__ weight,
    const float* __restrict__ scale,  // null when out_t == scalar_t
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  const int64_t vec_hidden_size = hidden_size / VEC_SIZE;
  const int64_t vec_input_stride = input_stride / VEC_SIZE;

  using vin_t = vec_n_t<scalar_t, VEC_SIZE>;
  auto* input_v = reinterpret_cast<vin_t*>(input);
  auto* residual_v = reinterpret_cast<vin_t*>(residual);
  const auto* weight_v = reinterpret_cast<const vin_t*>(weight);

  // ---- Pass 1: residual += input; accumulate variance; save residual -------
  for (int64_t idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int64_t rid = static_cast<int64_t>(blockIdx.x) * vec_hidden_size + idx;
    int64_t iid = static_cast<int64_t>(blockIdx.x) * vec_input_stride + idx;
    vin_t inp = input_v[iid];
    vin_t res = residual_v[rid];
    vin_t tmp;
#pragma unroll
    for (int j = 0; j < VEC_SIZE; j++) {
      float x = static_cast<float>(inp.val[j]) + static_cast<float>(res.val[j]);
      tmp.val[j] = static_cast<scalar_t>(x);
      variance += x * x;
    }
    residual_v[rid] = tmp;
  }

  // ---- Block-wide reduction -------------------------------------------------
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  // ---- Pass 2: normalise; write output -------------------------------------
  if constexpr (std::is_same_v<out_t, scalar_t>) {
    // Scalar output: write normalised result back to input (strided)
    for (int64_t idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
      int64_t rid = static_cast<int64_t>(blockIdx.x) * vec_hidden_size + idx;
      int64_t iid = static_cast<int64_t>(blockIdx.x) * vec_input_stride + idx;
      vin_t res = residual_v[rid];
      vin_t wt = weight_v[idx];
      vin_t dst;
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        float x = static_cast<float>(res.val[j]);
        dst.val[j] = static_cast<scalar_t>(x * s_variance) * wt.val[j];
      }
      input_v[iid] = dst;
    }
  } else {
    // FP8 output: write quantised result to out_quant (contiguous).
    // q8_n_t<out_t, VEC_SIZE> issues a single VEC_SIZE-byte store per vector.
    float const scale_inv = 1.0f / *scale;
    using vout_t = q8_n_t<out_t, VEC_SIZE>;
    auto* v_out = reinterpret_cast<vout_t*>(
        out_quant + static_cast<int64_t>(blockIdx.x) * hidden_size);
    for (int64_t idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
      int64_t rid = static_cast<int64_t>(blockIdx.x) * vec_hidden_size + idx;
      vin_t res = residual_v[rid];
      vin_t wt = weight_v[idx];
      vout_t out_vec;
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        float x = static_cast<float>(res.val[j]);
        float norm = static_cast<float>(static_cast<scalar_t>(x * s_variance) *
                                        wt.val[j]);
        out_vec.val[j] = scaled_fp8_conversion<true, out_t>(norm, scale_inv);
      }
      v_out[idx] = out_vec;
    }
  }
}

}  // namespace vllm
