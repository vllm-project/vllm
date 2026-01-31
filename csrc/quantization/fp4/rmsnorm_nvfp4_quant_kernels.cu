#include <torch/all.h>

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp8.h>
#include <cub/cub.cuh>
#include "dispatch_utils.h"
#include "cub_helpers.h"

#include "cuda_utils.h"
#include "launch_bounds_utils.h"

// Define before including nvfp4_utils.cuh so the header
// can use this macro during compilation.
#define NVFP4_ENABLE_ELTS16 1
#include "nvfp4_utils.cuh"
#include "../fused_kernels/layernorm_utils.cuh"

namespace vllm {

// Use UE4M3 by default.
template <typename scalar_t, bool UE8M0_SF = false>
__global__ void __launch_bounds__(1024, VLLM_BLOCKS_PER_SM(1024))
    rms_norm_cvt_fp16_to_fp4(const int num_tokens, const int hidden_size,
                             scalar_t const* __restrict__ input,
                             scalar_t const* __restrict__ weight,
                             float const* __restrict__ scale,
                             const float epsilon, uint32_t* __restrict__ output,
                             uint32_t* __restrict__ output_scale) {
  using PackedVecT = vllm::PackedVec<scalar_t>;

  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;

  __shared__ float s_rms_inv;

  const int32_t num_k_tiles = (hidden_size + 63) / 64;
  const float global_scale = (scale == nullptr) ? 1.0f : scale[0];
  const int vecs_per_row = hidden_size / CVT_FP4_ELTS_PER_THREAD;

  // SF layout requires rows padded to 128 and cols padded to 64
  const int sf_rows = (num_tokens + 127) / 128 * 128;
  const int sf_cols = (hidden_size + 63) / 64 * 64;
  const int vecs_per_row_padded = sf_cols / CVT_FP4_ELTS_PER_THREAD;

  for (int row_idx = blockIdx.x; row_idx < sf_rows; row_idx += gridDim.x) {
    const bool valid_row = row_idx < num_tokens;
    const scalar_t* row_input = input + row_idx * hidden_size;

    float variance = 0.0f;
    for (int col_idx = threadIdx.x; col_idx < vecs_per_row_padded;
         col_idx += blockDim.x) {
      const int elem_idx = col_idx * CVT_FP4_ELTS_PER_THREAD;
      PackedVecT vec{};

      bool valid = valid_row && (elem_idx < hidden_size);
      if constexpr (CVT_FP4_PACK16) {
        ld256_or_zero_cg_u32<scalar_t>(
            vec, &reinterpret_cast<const uint32_t*>(row_input)[col_idx * 8],
            valid);
      } else {
        ld128_or_zero_cg_u32<scalar_t>(
            vec, &reinterpret_cast<const uint32_t*>(row_input)[col_idx * 4],
            valid);
      }

      if (valid) {
        variance += compute_packed_sum_squares(vec);
      }
    }

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;
    variance =
        BlockReduce(reduce_storage).Reduce(variance, CubAddOp{}, blockDim.x);

    if (threadIdx.x == 0) {
      s_rms_inv = rsqrtf(variance / static_cast<float>(hidden_size) + epsilon);
    }
    __syncthreads();

    const float rms_inv = s_rms_inv;
    uint32_t* row_out = output + row_idx * vecs_per_row;

    for (int col_idx = threadIdx.x; col_idx < vecs_per_row_padded;
         col_idx += blockDim.x) {
      const int elem_idx = col_idx * CVT_FP4_ELTS_PER_THREAD;
      const bool valid_col = elem_idx < hidden_size;
      const bool valid = valid_row && valid_col;

      PackedVecT in_vec{}, w_vec{};

      if constexpr (CVT_FP4_PACK16) {
        ld256_or_zero_cg_u32<scalar_t>(
            in_vec, &reinterpret_cast<const uint32_t*>(row_input)[col_idx * 8],
            valid);
        ld256_or_zero_cg_u32<scalar_t>(
            w_vec, &reinterpret_cast<const uint32_t*>(weight)[col_idx * 8],
            valid);
      } else {
        ld128_or_zero_cg_u32<scalar_t>(
            in_vec, &reinterpret_cast<const uint32_t*>(row_input)[col_idx * 4],
            valid);
        ld128_or_zero_cg_u32<scalar_t>(
            w_vec, &reinterpret_cast<const uint32_t*>(weight)[col_idx * 4],
            valid);
      }

      PackedVecT norm_vec = compute_rms_norm(in_vec, w_vec, rms_inv);

      uint8_t* sf_out =
          cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                             CVT_FP4_NUM_THREADS_PER_SF>(
              row_idx, col_idx, num_k_tiles, output_scale);

      auto fp4_packed =
          cvt_warp_fp16_to_fp4<scalar_t, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
              norm_vec, global_scale, sf_out);

      if (valid) {
        if constexpr (CVT_FP4_PACK16) {
          int64_t out_offset = row_idx * (hidden_size / 8) + col_idx * 2;
          uint64_t packed64 =
              (uint64_t(fp4_packed.hi) << 32) | uint64_t(fp4_packed.lo);
          reinterpret_cast<uint64_t*>(output)[out_offset >> 1] = packed64;
        } else {
          row_out[col_idx] = fp4_packed;
        }
      }
    }

    __syncthreads();
  }
}

}  // namespace vllm

void rms_norm_nvfp4_quant_sm1xxa(
    torch::Tensor& output,        // [..., hidden_size/2] uint8 (packed FP4)
    torch::Tensor& output_scale,  // block scale, int32 (swizzled layout)
    torch::Tensor& input,         // [..., hidden_size] BF16/FP16
    torch::Tensor& weight,        // [hidden_size] BF16/FP16
    torch::Tensor& scale,         // [1] float32 (global scale)
    double epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  TORCH_CHECK(hidden_size % 16 == 0, "The hidden_size must be multiple of 16.");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half ||
                  input.scalar_type() == at::ScalarType::BFloat16,
              "Unsupported input data type for rms_norm_nvfp4_quant.");

  int multi_processor_count =
      get_device_attribute(cudaDevAttrMultiProcessorCount, -1);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 block(std::min(hidden_size / ELTS_PER_THREAD, 1024));
  int const num_blocks_per_sm =
      vllm_runtime_blocks_per_sm(static_cast<int>(block.x));
  dim3 grid(std::min(num_tokens, multi_processor_count * num_blocks_per_sm));

  VLLM_DISPATCH_HALF_TYPES(
      input.scalar_type(), "rms_norm_nvfp4_quant_kernel", [&] {
        using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
        vllm::rms_norm_cvt_fp16_to_fp4<cuda_type><<<grid, block, 0, stream>>>(
            num_tokens, hidden_size,
            reinterpret_cast<cuda_type const*>(input.data_ptr()),
            reinterpret_cast<cuda_type const*>(weight.data_ptr()),
            scale.data_ptr<float>(), static_cast<float>(epsilon),
            reinterpret_cast<uint32_t*>(output.data_ptr()),
            reinterpret_cast<uint32_t*>(output_scale.data_ptr()));
      });
}
