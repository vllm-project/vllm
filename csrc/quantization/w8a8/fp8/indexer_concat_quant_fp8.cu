#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "cuda_vec_utils.cuh"
#include "dispatch_utils.h"

namespace vllm {

constexpr int WARP_SIZE = 32;

union BF16x4 {
  int2 vec;
  __nv_bfloat162 bf16x2[2];
};

union HALFx4 {
  int2 vec;
  __half2 half2[2];
};

union FLOATx4 {
  int4 vec;
  float4 f4;
};

union FP8x4 {
  int packed;
  __nv_fp8_e4m3 fp8[4];
};

__device__ __forceinline__ float WarpReduceMax(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
  }
  return val;
}

template <typename InputType, int VEC_SIZE, int HEAD_DIM, bool SCALE_UE8M0>
__global__ __launch_bounds__(256) void indexer_concat_quant_fp8_kernel(
    __nv_fp8_e4m3* __restrict__ q_out, float* __restrict__ scale_out,
    const InputType* __restrict__ q_pe, const InputType* __restrict__ q_nope,
    const int num_tokens, const int num_heads, const int rope_dim,
    const int nope_dim, const int64_t out_stride_0, const int64_t out_stride_1,
    const int64_t nope_stride_0, const int64_t nope_stride_1,
    const int64_t pe_stride_0, const int64_t pe_stride_1, const double eps,
    const double min_8bit, const double max_8bit) {
  static_assert(std::is_same_v<InputType, float> ||
                    std::is_same_v<InputType, at::BFloat16> ||
                    std::is_same_v<InputType, at::Half>,
                "InputType must be float, bfloat16, or half");

  const int global_warp_id =
      (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  if (global_warp_id >= num_tokens * num_heads) {
    return;
  }

  const int token_id = global_warp_id / num_heads;
  const int head_id = global_warp_id % num_heads;
  const int lane_id = threadIdx.x % WARP_SIZE;

  int base = lane_id * VEC_SIZE;
  const InputType* pe_row =
      q_pe + token_id * pe_stride_0 + head_id * pe_stride_1;
  const InputType* nope_row =
      q_nope + token_id * nope_stride_0 + head_id * nope_stride_1;
  const InputType* input_row =
      (base < rope_dim) ? (pe_row + base) : (nope_row + (base - rope_dim));

  float v0, v1, v2, v3;
  if constexpr (std::is_same_v<InputType, float>) {
    FLOATx4 src;
    src.vec = ld128_cs(reinterpret_cast<const int4*>(input_row));
    v0 = src.f4.x;
    v1 = src.f4.y;
    v2 = src.f4.z;
    v3 = src.f4.w;
  } else if constexpr (std::is_same_v<InputType, at::BFloat16>) {
    BF16x4 src;
    src.vec = ld64_cs(reinterpret_cast<const int2*>(input_row));
    float2 f0 = __bfloat1622float2(src.bf16x2[0]);
    float2 f1 = __bfloat1622float2(src.bf16x2[1]);
    v0 = f0.x;
    v1 = f0.y;
    v2 = f1.x;
    v3 = f1.y;
  } else if constexpr (std::is_same_v<InputType, at::Half>) {
    HALFx4 src;
    src.vec = ld64_cs(reinterpret_cast<const int2*>(input_row));
    float2 f0 = __half22float2(src.half2[0]);
    float2 f1 = __half22float2(src.half2[1]);
    v0 = f0.x;
    v1 = f0.y;
    v2 = f1.x;
    v3 = f1.y;
  }

  // Quantization
  float local_absmax =
      fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
  float absmax = WarpReduceMax(local_absmax);
  absmax = fmaxf(absmax, eps);

  float y_s = absmax / max_8bit;
  if constexpr (SCALE_UE8M0) {
    y_s = exp2f(ceilf(log2f(fmaxf(fabsf(y_s), 1e-10f))));
  }

  if (lane_id == 0) {
    scale_out[global_warp_id] = y_s;
  }

  auto quant_op = [&](float val) -> __nv_fp8_e4m3 {
    float q = fminf(fmaxf(val / y_s, min_8bit), max_8bit);
    return __nv_fp8_e4m3(q);
  };

  FP8x4 dst;
  dst.fp8[0] = quant_op(v0);
  dst.fp8[1] = quant_op(v1);
  dst.fp8[2] = quant_op(v2);
  dst.fp8[3] = quant_op(v3);

  int out_base =
      token_id * out_stride_0 + head_id * out_stride_1 + lane_id * VEC_SIZE;
  st32_cs(reinterpret_cast<int*>(q_out + out_base), dst.packed);
}

template <typename InputType, int VEC_SIZE, int HEAD_DIM>
void invokeConcatIndexerQFp8(
    __nv_fp8_e4m3* q_out, float* scale_out, const InputType* q_pe,
    const InputType* q_nope, const int num_tokens, const int num_heads,
    const int rope_dim, const int nope_dim, const int64_t out_stride_0,
    const int64_t out_stride_1, const int64_t nope_stride_0,
    const int64_t nope_stride_1, const int64_t pe_stride_0,
    const int64_t pe_stride_1, const int64_t group_size, const double eps,
    const double fp8_min, const double fp8_max, bool scale_ue8m0,
    cudaStream_t stream) {
  TORCH_CHECK(rope_dim % VEC_SIZE == 0,
              "rope_dim (%d) must be a multiple of VEC_SIZE (%d)", rope_dim,
              VEC_SIZE);
  TORCH_CHECK(pe_stride_1 % VEC_SIZE == 0,
              "pe_stride_1 (%d) must be multiple of VEC_SIZE (%d)", pe_stride_1,
              VEC_SIZE);
  TORCH_CHECK(nope_stride_1 % VEC_SIZE == 0,
              "nope_stride_1 (%d) must be multiple of VEC_SIZE (%d)",
              nope_stride_1, VEC_SIZE);

  const int num_threads = 256;
  const int total_warps = num_tokens * num_heads;
  const int warps_per_block = num_threads / WARP_SIZE;
  const int num_blocks = (total_warps + warps_per_block - 1) / warps_per_block;
  dim3 grid(num_blocks);
  dim3 block(num_threads);

  if (scale_ue8m0) {
    indexer_concat_quant_fp8_kernel<InputType, VEC_SIZE, HEAD_DIM, true>
        <<<grid, block, 0, stream>>>(
            q_out, scale_out, q_pe, q_nope, num_tokens, num_heads, rope_dim,
            nope_dim, out_stride_0, out_stride_1, nope_stride_0, nope_stride_1,
            pe_stride_0, pe_stride_1, eps, fp8_min, fp8_max);
  } else {
    indexer_concat_quant_fp8_kernel<InputType, VEC_SIZE, HEAD_DIM, false>
        <<<grid, block, 0, stream>>>(
            q_out, scale_out, q_pe, q_nope, num_tokens, num_heads, rope_dim,
            nope_dim, out_stride_0, out_stride_1, nope_stride_0, nope_stride_1,
            pe_stride_0, pe_stride_1, eps, fp8_min, fp8_max);
  }
}
}  // namespace vllm

// Concatenates q_pe and q_nope, then FP8 quantization.
void indexer_concat_quant_fp8(
    torch::Tensor& q_out,         // [num_tokens, num_heads, head_dim]
    torch::Tensor& scale_out,     // [num_tokens, num_heads, 1]
    torch::Tensor const& q_pe,    // [num_tokens, num_heads, rope_dim]
    torch::Tensor const& q_nope,  // [num_tokens, num_heads, nope_dim]
    int64_t group_size, double eps, double fp8_min, double fp8_max,
    bool scale_ue8m0) {
  TORCH_CHECK(scale_out.scalar_type() == at::ScalarType::Float,
              "scale_out must be float");
  TORCH_CHECK(q_pe.stride(-1) == 1, "q_pe must be contiguous");
  TORCH_CHECK(q_nope.stride(-1) == 1, "q_nope must be contiguous");
  TORCH_CHECK(q_out.stride(-1) == 1, "q_out must be contiguous");

  const int num_tokens = q_nope.size(0);
  const int num_heads = q_nope.size(1);
  const int nope_dim = q_nope.size(-1);
  const int rope_dim = q_pe.size(-1);
  const int head_dim = rope_dim + nope_dim;

  TORCH_CHECK(group_size == 128, "group_size must be 128");
  TORCH_CHECK(head_dim == 128, "head_dim must be 128");
  TORCH_CHECK(rope_dim > 0, "rope_dim must be > 0");
  TORCH_CHECK(nope_dim > 0, "nope_dim must be > 0");
  TORCH_CHECK(head_dim == rope_dim + nope_dim,
              "head_dim must be equal to rope_dim + nope_dim");

  if (num_tokens == 0) {
    return;
  }

  auto stream = at::cuda::getCurrentCUDAStream(q_pe.get_device());

  VLLM_DISPATCH_FLOATING_TYPES(
      q_nope.scalar_type(), "indexer_concat_quant_fp8", [&] {
        vllm::invokeConcatIndexerQFp8<scalar_t, 4, 128>(
            reinterpret_cast<__nv_fp8_e4m3*>(q_out.data_ptr()),
            scale_out.data_ptr<float>(), q_pe.data_ptr<scalar_t>(),
            q_nope.data_ptr<scalar_t>(), num_tokens, num_heads, rope_dim,
            nope_dim, q_out.stride(0), q_out.stride(1), q_nope.stride(0),
            q_nope.stride(1), q_pe.stride(0), q_pe.stride(1), group_size, eps,
            fp8_min, fp8_max, scale_ue8m0, stream);
      });
}
