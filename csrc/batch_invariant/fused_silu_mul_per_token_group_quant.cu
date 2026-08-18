/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Source provenance:
// - SGLang base commit 0b3bb0cbe31873994c9f989fddfe2f87ca839fdd,
//   sgl-kernel/csrc/gemm/per_token_group_quant_8bit_v2.cu.
// - Deterministic numeric changes from Slime commit
//   a74ae3a0ad16bd8b769d5386738e8ae3d1269d7e,
//   docker/patch/latest/sglang-deterministic.patch, for the source above.
// - Local semantics: rounded and unrounded scale paths, packed UE8M0
//   validation, 64-bit large-shape offsets, masked and contiguous layouts,
//   and a standalone vLLM Torch operator binding.

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Float8_e4m3fn.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <sstream>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_EQ(a, b) \
  TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

#define DISPATCH_INPUT_DTYPE(pytorch_dtype, c_type, ...)                 \
  [&]() -> bool {                                                        \
    switch (pytorch_dtype) {                                             \
      case at::ScalarType::Half: {                                       \
        using c_type = __half;                                           \
        return __VA_ARGS__();                                            \
      }                                                                  \
      case at::ScalarType::BFloat16: {                                   \
        using c_type = __nv_bfloat16;                                    \
        return __VA_ARGS__();                                            \
      }                                                                  \
      default:                                                           \
        TORCH_CHECK(false, "unsupported input dtype: ", pytorch_dtype);  \
        return false;                                                    \
    }                                                                    \
  }()

namespace vllm::batch_invariant {

inline bool get_bool_env(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

inline int get_sm_version() {
  int device = -1;
  int major = 0;
  int minor = 0;
  TORCH_CHECK(cudaGetDevice(&device) == cudaSuccess, "failed to get CUDA device");
  TORCH_CHECK(
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess,
      "failed to get CUDA compute capability major");
  TORCH_CHECK(
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess,
      "failed to get CUDA compute capability minor");
  return major * 10 + minor;
}

inline bool get_env_enable_pdl() {
  static std::once_flag flag;
  static bool enabled = false;
  std::call_once(flag, [] {
    enabled = get_sm_version() >= 90 && get_bool_env("TRTLLM_ENABLE_PDL");
  });
  return enabled;
}

template <int THREADS_PER_SUBWARP>
__device__ __forceinline__ float GroupReduceMax(float val, const int tid) {
  unsigned mask = 0xffffffff;

  static_assert(
      (THREADS_PER_SUBWARP & (THREADS_PER_SUBWARP - 1)) == 0 && THREADS_PER_SUBWARP <= 16 && THREADS_PER_SUBWARP >= 1,
      "THREADS_PER_SUBWARP must be 1, 2, 4, 8, or 16");

  if constexpr (THREADS_PER_SUBWARP >= 16) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  }
  if constexpr (THREADS_PER_SUBWARP >= 8) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  }
  if constexpr (THREADS_PER_SUBWARP >= 4) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  }
  if constexpr (THREADS_PER_SUBWARP >= 2) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
  }
  return val;
}

__device__ __forceinline__ float silu(const float& val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  float half = 0.5f * val;
  float t = __tanhf(half);
  return half * (1.0f + t);
#else
  // Match the legacy SM90 activation kernel.  That kernel uses expf and
  // exposes only the final SiLU*up BF16 rounding boundary.
  return val / (1.0f + expf(-val));
#endif
}

__device__ float2 fmul2_rn(float2 a, float2 b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  return __fmul2_rn(a, b);
#else
  float2 result;
  result.x = a.x * b.x;
  result.y = a.y * b.y;
  return result;
#endif
}

// Copied and modified from DeepEP
__forceinline__ __device__ float fast_pow2(int x) {
  // We can ensure `-126 <= x and x <= 127`
  uint32_t bits_x = (x + 127) << 23;
  return *reinterpret_cast<float*>(&bits_x);
}

// Copied and modified from DeepEP
__forceinline__ __device__ int fast_log2_ceil(float x) {
  auto bits_x = *reinterpret_cast<uint32_t*>(&x);
  auto exp_x = (bits_x >> 23) & 0xff;
  auto man_bits = bits_x & ((1 << 23) - 1);
  return exp_x - 127 + (man_bits != 0);
}

// Copied and modified from DeepEP
template <bool ROUND_SCALE, typename dtype_info>
__forceinline__ __device__ void calculate_fp8_scales(float amax, float& scale, float& scale_inv) {
  constexpr float MAX_8BIT_INV = 1.0f / dtype_info::MAX;
  if constexpr (ROUND_SCALE) {
    auto exp_scale_inv = fast_log2_ceil(amax * MAX_8BIT_INV);
    scale = fast_pow2(-exp_scale_inv);
    scale_inv = fast_pow2(exp_scale_inv);
  } else {
    scale_inv = amax * MAX_8BIT_INV;
    scale = dtype_info::MAX / amax;
  }
}

// Copied and modified from DeepEP
template <bool SCALE_UE8M0, typename OUT_DTYPE_T = std::conditional_t<SCALE_UE8M0, uint8_t, float>>
__forceinline__ __device__ OUT_DTYPE_T extract_required_scale_format(float value) {
  if constexpr (SCALE_UE8M0) {
    return static_cast<uint8_t>((*reinterpret_cast<uint32_t*>(&value)) >> 23);
  } else {
    return value;
  }
}

__device__ __forceinline__ void st_global(const int4* ptr, const int4& value) {
#ifndef USE_MUSA
  asm volatile(
      "st.global.v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
#else
  int4* p = const_cast<int4*>(ptr);
  *p = value;
#endif
}

__device__ __forceinline__ int4 ld_global_nc(const int4* ptr) {
#ifndef USE_MUSA
  int4 ret;
  asm volatile("ld.global.nc.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
               : "l"(ptr));
  return ret;
#else
  return *ptr;
#endif
}

template <typename T>
struct DtypeInfo;

template <>
struct DtypeInfo<int8_t> {
  static constexpr float MIN = -128;
  static constexpr float MAX = 127;
};

template <>
struct DtypeInfo<c10::Float8_e4m3fn> {
  static constexpr float MIN = -448;
  static constexpr float MAX = 448;
};

template <bool FUSE_SILU_AND_MUL>
__device__ __forceinline__ int64_t compute_input_group_start_offset(
    int expert_idx,
    int token_idx,
    int hidden_dim_group_idx,
    int hidden_size,
    int num_tokens_per_expert,
    int group_size) {
  return static_cast<int64_t>(expert_idx) * num_tokens_per_expert * hidden_size *
             (FUSE_SILU_AND_MUL ? 2 : 1) +
         static_cast<int64_t>(token_idx) * hidden_size * (FUSE_SILU_AND_MUL ? 2 : 1) +
         static_cast<int64_t>(hidden_dim_group_idx) * group_size;
}

constexpr float LOCAL_ABSMAX_ABS = 1e-10;
constexpr uint32_t INPUT_PRIMARY_VEC_NUM_BYTES = 32;

struct NaiveScheduler {
  static void compute_exec_config(
      int threads_per_subwarp,
      int num_local_experts,
      int hidden_dim_num_groups,
      int num_groups,
      int& subwarps_per_block,
      dim3& grid,
      dim3& block) {
    subwarps_per_block = ([=]() -> int {
      if (num_groups % 16 == 0) {
        return 16;
      } else if (num_groups % 8 == 0) {
        return 8;
      } else if (num_groups % 4 == 0) {
        return 4;
      } else if (num_groups % 2 == 0) {
        return 2;
      }
      return 1;
    })();
    grid = dim3(num_groups / subwarps_per_block);
    block = dim3(subwarps_per_block * threads_per_subwarp);
  }

  template <bool FUSE_SILU_AND_MUL, int GROUP_SIZE, int THREADS_PER_SUBWARP, typename FUNC>
  __device__ __forceinline__ static void execute(
      const int subwarps_per_block,
      const int hidden_dim_num_groups,
      const int32_t* masked_m,
      const int num_tokens_per_expert,
      FUNC fn) {
    constexpr int expert_idx = 0;

    const int64_t subwarp_id = threadIdx.x / THREADS_PER_SUBWARP;
    const int lane_id = threadIdx.x % THREADS_PER_SUBWARP;

    const int64_t block_group_id = blockIdx.x * subwarps_per_block;
    const int64_t group_id = block_group_id + subwarp_id;

    int64_t input_group_start_offset;
    if constexpr (!FUSE_SILU_AND_MUL) {
      input_group_start_offset = group_id * GROUP_SIZE;
    }

    const int token_idx = group_id / hidden_dim_num_groups;
    // At the hidden_size dimension, we are handling idx-th group
    const int hidden_dim_group_idx = group_id % hidden_dim_num_groups;

    if constexpr (FUSE_SILU_AND_MUL) {
      const int hidden_size = hidden_dim_num_groups * GROUP_SIZE;
      input_group_start_offset = compute_input_group_start_offset<FUSE_SILU_AND_MUL>(
          expert_idx, token_idx, hidden_dim_group_idx, hidden_size, num_tokens_per_expert, GROUP_SIZE);
    }

    fn(expert_idx, token_idx, hidden_dim_group_idx, lane_id, input_group_start_offset);
  }
};

struct MaskedLayoutScheduler {
  // TODO can be dynamically determined (which may be good when num rank is small)
  static constexpr int TOKEN_DIM_BLOCK_NUM_PER_EXPERT = 1024;
  static constexpr int SUBWARPS_PER_BLOCK = 16;

  static void compute_exec_config(
      int threads_per_subwarp,
      int num_local_experts,
      int hidden_dim_num_groups,
      int num_groups,
      int& subwarps_per_block,
      dim3& grid,
      dim3& block) {
    subwarps_per_block = SUBWARPS_PER_BLOCK;
    TORCH_CHECK(hidden_dim_num_groups % subwarps_per_block == 0);
    grid = dim3(hidden_dim_num_groups / subwarps_per_block, TOKEN_DIM_BLOCK_NUM_PER_EXPERT, num_local_experts);
    block = dim3(subwarps_per_block * threads_per_subwarp);
  }

  template <bool FUSE_SILU_AND_MUL, int GROUP_SIZE, int THREADS_PER_SUBWARP, typename FUNC>
  __device__ __forceinline__ static void execute(
      const int subwarps_per_block,
      const int hidden_dim_num_groups,
      const int32_t* masked_m,
      const int num_tokens_per_expert,
      FUNC fn) {
    const int64_t subwarp_id = threadIdx.x / THREADS_PER_SUBWARP;
    const int lane_id = threadIdx.x % THREADS_PER_SUBWARP;

    const int expert_idx = blockIdx.z;
    const int token_idx_start = blockIdx.y;

    const int64_t hidden_dim_group_idx = blockIdx.x * SUBWARPS_PER_BLOCK + subwarp_id;

    const int curr_expert_token_num = masked_m[expert_idx];

    for (int token_idx = token_idx_start; token_idx < curr_expert_token_num;
         token_idx += TOKEN_DIM_BLOCK_NUM_PER_EXPERT) {
      const int hidden_size = hidden_dim_num_groups * GROUP_SIZE;
      const int64_t input_group_start_offset = compute_input_group_start_offset<FUSE_SILU_AND_MUL>(
          expert_idx, token_idx, hidden_dim_group_idx, hidden_size, num_tokens_per_expert, GROUP_SIZE);
      fn(expert_idx, token_idx, hidden_dim_group_idx, lane_id, input_group_start_offset);
    }
  }
};

template <
    typename SCHEDULER,
    int GROUP_SIZE,
    int THREADS_PER_SUBWARP,
    typename T,
    typename DST_DTYPE,
    bool IS_COLUMN_MAJOR = false,
    bool SCALE_UE8M0 = false,
    bool FUSE_SILU_AND_MUL = false,
    bool ROUND_SCALE = SCALE_UE8M0,
    typename scale_packed_t = std::conditional_t<SCALE_UE8M0, uint32_t, float>>
__global__ void per_token_group_quant_8bit_kernel(
    const T* __restrict__ input,
    DST_DTYPE* __restrict__ output_q,
    scale_packed_t* __restrict__ output_s,
    const int32_t* __restrict__ masked_m,
    const int subwarps_per_block,
    const int hidden_dim_num_groups,
    // TODO can this be removed?
    const int scale_expert_stride,
    const int scale_hidden_stride,
    const int num_tokens_per_expert,
    const float max_8bit) {
  using dst_dtype_info = DtypeInfo<DST_DTYPE>;
  using scale_element_t = std::conditional_t<SCALE_UE8M0, uint8_t, float>;
  static_assert(sizeof(scale_packed_t) % sizeof(scale_element_t) == 0);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif

  SCHEDULER::execute<FUSE_SILU_AND_MUL, GROUP_SIZE, THREADS_PER_SUBWARP>(
      subwarps_per_block,
      hidden_dim_num_groups,
      masked_m,
      num_tokens_per_expert,
      [&](const int expert_idx,
          const int token_idx,
          const int hidden_dim_group_idx,
          const int lane_id,
          const int64_t input_group_start_offset) {
        constexpr uint32_t INPUT_PRIMARY_VEC_SIZE = INPUT_PRIMARY_VEC_NUM_BYTES / sizeof(T);
        constexpr uint32_t INPUT_PRIMARY_INT4_SIZE = INPUT_PRIMARY_VEC_NUM_BYTES / sizeof(int4);

        const int offset_num_groups = expert_idx * num_tokens_per_expert * hidden_dim_num_groups +
                                      token_idx * hidden_dim_num_groups + hidden_dim_group_idx;

        int4 input_primary_int4[INPUT_PRIMARY_INT4_SIZE];
        T* input_primary_vec = reinterpret_cast<T*>(input_primary_int4);
        static_assert(sizeof(input_primary_vec[0]) * INPUT_PRIMARY_VEC_SIZE == sizeof(input_primary_int4));

        int4 input_secondary_int4[INPUT_PRIMARY_INT4_SIZE];
        T* input_secondary_vec = reinterpret_cast<T*>(input_secondary_int4);
        static_assert(sizeof(input_secondary_vec[0]) * INPUT_PRIMARY_VEC_SIZE == sizeof(input_secondary_int4));

#pragma unroll
        for (uint32_t j = 0; j < INPUT_PRIMARY_INT4_SIZE; ++j) {
          input_primary_int4[j] = ld_global_nc(
              reinterpret_cast<const int4*>(input + input_group_start_offset + lane_id * INPUT_PRIMARY_VEC_SIZE) + j);
        }
        if constexpr (FUSE_SILU_AND_MUL) {
          const int secondary_offset = hidden_dim_num_groups * GROUP_SIZE;
#pragma unroll
          for (uint32_t j = 0; j < INPUT_PRIMARY_INT4_SIZE; ++j) {
            input_secondary_int4[j] = ld_global_nc(
                reinterpret_cast<const int4*>(
                    input + input_group_start_offset + lane_id * INPUT_PRIMARY_VEC_SIZE + secondary_offset) +
                j);
          }
        }

        constexpr int num_elems_per_pack = static_cast<int>(sizeof(scale_packed_t) / sizeof(scale_element_t));
        scale_element_t* scale_output;
        if constexpr (IS_COLUMN_MAJOR) {
          constexpr int scale_token_stride = 1;

          const int hidden_idx_packed = hidden_dim_group_idx / num_elems_per_pack;
          const int pack_idx = hidden_dim_group_idx % num_elems_per_pack;
          scale_output = reinterpret_cast<scale_element_t*>(output_s) +
                         (expert_idx * scale_expert_stride * num_elems_per_pack +
                          hidden_idx_packed * scale_hidden_stride * num_elems_per_pack +
                          token_idx * scale_token_stride * num_elems_per_pack + pack_idx);
        } else {
          static_assert(!SCALE_UE8M0);
          scale_output = output_s + offset_num_groups;
        }

        // can speed up if too slow
        if constexpr (IS_COLUMN_MAJOR and SCALE_UE8M0) {
          const int remainder_num_groups = hidden_dim_num_groups % num_elems_per_pack;
          if ((remainder_num_groups != 0) and (hidden_dim_group_idx == hidden_dim_num_groups - 1) and
              (lane_id < num_elems_per_pack - remainder_num_groups)) {
            const int shift = 1 + lane_id;
            *(scale_output + shift) = 0;
          }
        }

        float local_absmax = LOCAL_ABSMAX_ABS;

#pragma unroll
        for (uint32_t j = 0; j < INPUT_PRIMARY_VEC_SIZE; ++j) {
          float val;
          if constexpr (FUSE_SILU_AND_MUL) {
            // Keep the same single visible low-precision boundary as the
            // contiguous activation path.  Rounding SiLU before multiplying
            // changes the second MoE GEMM's FP8 inputs.
            T val_lowprec = static_cast<T>(
                silu(static_cast<float>(input_primary_vec[j])) * static_cast<float>(input_secondary_vec[j]));
            val = static_cast<float>(val_lowprec);
            input_primary_vec[j] = val_lowprec;
          } else {
            val = static_cast<float>(input_primary_vec[j]);
          }

          float abs_val = fabsf(val);
          local_absmax = fmaxf(local_absmax, abs_val);
        }

        local_absmax = GroupReduceMax<THREADS_PER_SUBWARP>(local_absmax, lane_id);

        float y_scale, y_scale_inv;
        if constexpr (ROUND_SCALE) {
          calculate_fp8_scales<true, dst_dtype_info>(local_absmax, y_scale, y_scale_inv);
        } else {
          // Preserve the scale and division sequence of the legacy unmasked
          // quantizer.  Reciprocal reuse under -use_fast_math moves a few
          // values across an FP8 rounding boundary.
          y_scale_inv = __fdiv_rn(local_absmax, max_8bit);
          y_scale = 1.0f / y_scale_inv;
        }

        if (lane_id == 0) {
          *scale_output = extract_required_scale_format<SCALE_UE8M0>(y_scale_inv);
        }

        int4 output_buf;
        static_assert(sizeof(output_buf) == INPUT_PRIMARY_VEC_SIZE * sizeof(DST_DTYPE));

        if constexpr (std::is_same_v<DST_DTYPE, c10::Float8_e4m3fn>) {
          if constexpr (SCALE_UE8M0) {
            const auto output_buf_ptr = reinterpret_cast<__nv_fp8x2_storage_t*>(&output_buf);
            static_assert(sizeof(output_buf) == INPUT_PRIMARY_VEC_SIZE / 2 * sizeof(__nv_fp8x2_storage_t));
            static_assert(INPUT_PRIMARY_VEC_SIZE % 2 == 0);

#pragma unroll
            for (uint32_t j = 0; j < INPUT_PRIMARY_VEC_SIZE; j += 2) {
              float2 inputx2 = {
                  static_cast<float>(input_primary_vec[j]), static_cast<float>(input_primary_vec[j + 1])};
              float2 outputx2 = fmul2_rn(inputx2, {y_scale, y_scale});

              outputx2.x = fminf(fmaxf(outputx2.x, dst_dtype_info::MIN), dst_dtype_info::MAX);
              outputx2.y = fminf(fmaxf(outputx2.y, dst_dtype_info::MIN), dst_dtype_info::MAX);

              output_buf_ptr[j / 2] = __nv_cvt_float2_to_fp8x2(outputx2, __NV_SATFINITE, __NV_E4M3);
            }
          } else {
            // Match the legacy SM90 quantizer's scalar conversion exactly;
            // packed conversion makes a different tie choice for rare values.
            const auto output_buf_ptr = reinterpret_cast<DST_DTYPE*>(&output_buf);
#pragma unroll
            for (uint32_t j = 0; j < INPUT_PRIMARY_VEC_SIZE; ++j) {
              float val = static_cast<float>(input_primary_vec[j]);
              float q_val =
                  fminf(fmaxf(__fdiv_rn(val, y_scale_inv), dst_dtype_info::MIN), dst_dtype_info::MAX);
              output_buf_ptr[j] = DST_DTYPE(q_val);
            }
          }
        } else {
          const auto output_buf_ptr = reinterpret_cast<DST_DTYPE*>(&output_buf);

#pragma unroll
          for (uint32_t j = 0; j < INPUT_PRIMARY_VEC_SIZE; ++j) {
            float val = static_cast<float>(input_primary_vec[j]);
            float q_val = fminf(fmaxf(val * y_scale, dst_dtype_info::MIN), dst_dtype_info::MAX);
            output_buf_ptr[j] = DST_DTYPE(q_val);
          }
        }

        st_global(
            reinterpret_cast<int4*>(output_q + offset_num_groups * GROUP_SIZE + lane_id * INPUT_PRIMARY_VEC_SIZE),
            output_buf);
      });

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void fused_silu_mul_per_token_group_quant(
    // vanilla: (num_tokens, hidden_size)
    // fuse_silu_and_mul: (num_tokens, hidden_size * 2)
    // fuse_silu_and_mul + masked_layout: (num_experts, num_tokens-with-padding, hidden_size * 2)
    torch::Tensor input,
    torch::Tensor output_q,
    torch::Tensor output_s,
    int64_t group_size,
    double eps,
    double min_8bit,
    double max_8bit,
    bool round_scale,
    bool scale_ue8m0,
    bool fuse_silu_and_mul,
    const std::optional<torch::Tensor>& masked_m) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  TORCH_CHECK(input.numel() > 0);

  TORCH_CHECK(std::abs(LOCAL_ABSMAX_ABS - eps) < 1e-13);

  CHECK_EQ(input.numel() % group_size, 0);
  const int num_groups = static_cast<int>(input.numel()) / group_size / (fuse_silu_and_mul ? 2 : 1);

  const bool masked_layout = masked_m.has_value();
  TORCH_CHECK(output_s.dim() == (masked_layout ? 3 : 2));

  const int num_local_experts = masked_layout ? input.size(0) : 1;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto dst_type = output_q.scalar_type();

  const bool is_column_major = output_s.stride(-2) < output_s.stride(-1);
  TORCH_CHECK(!scale_ue8m0 || round_scale, "packed UE8M0 scales require rounding");
  TORCH_CHECK(!round_scale || is_column_major, "rounded FP32 scales require column-major output");
  const int hidden_dim_num_groups = static_cast<int>(output_q.size(-1)) / group_size;
  const int num_tokens_per_expert = static_cast<int>(output_q.size(-2));
  const int scale_expert_stride = masked_layout ? static_cast<int>(output_s.stride(0)) : 0;
  const int scale_hidden_stride = static_cast<int>(output_s.stride(-1));

#define LAUNCH_KERNEL_INNER(SCHEDULER, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, output_s_dtype, ...)           \
  do {                                                                                                               \
    int subwarps_per_block;                                                                                          \
    dim3 grid, block;                                                                                                \
    SCHEDULER::compute_exec_config(                                                                                  \
        THREADS_PER_SUBWARP, num_local_experts, hidden_dim_num_groups, num_groups, subwarps_per_block, grid, block); \
                                                                                                                     \
    cudaLaunchConfig_t config;                                                                                       \
    config.gridDim = grid;                                                                                           \
    config.blockDim = block;                                                                                         \
    config.dynamicSmemBytes = 0;                                                                                     \
    config.stream = stream;                                                                                          \
    cudaLaunchAttribute attrs[1];                                                                                    \
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;                                                \
    attrs[0].val.programmaticStreamSerializationAllowed = get_env_enable_pdl();                                      \
    config.numAttrs = 1;                                                                                             \
    config.attrs = attrs;                                                                                            \
    cudaLaunchKernelEx(                                                                                              \
        &config,                                                                                                     \
        per_token_group_quant_8bit_kernel<SCHEDULER, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, __VA_ARGS__>,    \
        static_cast<T*>(input.data_ptr()),                                                                           \
        static_cast<DST_DTYPE*>(output_q.data_ptr()),                                                                \
        static_cast<output_s_dtype*>(output_s.data_ptr()),                                                           \
        static_cast<int32_t*>(masked_m.has_value() ? masked_m->data_ptr() : 0),                                      \
        subwarps_per_block,                                                                                          \
        hidden_dim_num_groups,                                                                                       \
        scale_expert_stride,                                                                                         \
        scale_hidden_stride,                                                                                         \
        num_tokens_per_expert,                                                                                       \
        static_cast<float>(max_8bit));                                                                               \
  } while (0)

#define LAUNCH_KERNEL(GROUP_SIZE, T, DST_DTYPE)                                                                     \
  do {                                                                                                              \
    constexpr int THREADS_PER_SUBWARP = GROUP_SIZE / 16;                                                            \
    TORCH_CHECK(THREADS_PER_SUBWARP * INPUT_PRIMARY_VEC_NUM_BYTES == group_size * sizeof(T));                       \
                                                                                                                    \
    using dst_dtype_info = DtypeInfo<DST_DTYPE>;                                                                    \
    CHECK_EQ(dst_dtype_info::MIN, min_8bit);                                                                        \
    CHECK_EQ(dst_dtype_info::MAX, max_8bit);                                                                        \
                                                                                                                    \
    if (is_column_major) {                                                                                          \
      if (scale_ue8m0) {                                                                                            \
        if (fuse_silu_and_mul) {                                                                                    \
          if (masked_layout) {                                                                                      \
            LAUNCH_KERNEL_INNER(                                                                                    \
                MaskedLayoutScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, uint32_t, true, true, true);  \
          } else {                                                                                                  \
            LAUNCH_KERNEL_INNER(                                                                                    \
                NaiveScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, uint32_t, true, true, true);         \
          }                                                                                                         \
        } else {                                                                                                    \
          if (masked_layout) {                                                                                       \
            LAUNCH_KERNEL_INNER(                                                                                    \
                MaskedLayoutScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, uint32_t, true, true, false); \
          } else {                                                                                                  \
            LAUNCH_KERNEL_INNER(                                                                                    \
                NaiveScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, uint32_t, true, true, false);        \
          }                                                                                                         \
        }                                                                                                           \
      } else {                                                                                                      \
        if (fuse_silu_and_mul) {                                                                                     \
          if (masked_layout) {                                                                                       \
            if (round_scale) {                                                                                       \
              LAUNCH_KERNEL_INNER(                                                                                  \
                  MaskedLayoutScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, true, false, true, true); \
            } else {                                                                                                \
              LAUNCH_KERNEL_INNER(                                                                                  \
                  MaskedLayoutScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, true, false, true);  \
            }                                                                                                       \
          } else {                                                                                                  \
            if (round_scale) {                                                                                       \
              LAUNCH_KERNEL_INNER(                                                                                  \
                  NaiveScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, true, false, true, true);   \
            } else {                                                                                                \
              LAUNCH_KERNEL_INNER(                                                                                  \
                  NaiveScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, true, false, true);         \
            }                                                                                                       \
          }                                                                                                         \
        } else {                                                                                                    \
          if (masked_layout) {                                                                                       \
            if (round_scale) {                                                                                       \
              LAUNCH_KERNEL_INNER(                                                                                  \
                  MaskedLayoutScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, true, false, false, true); \
            } else {                                                                                                \
              LAUNCH_KERNEL_INNER(                                                                                  \
                  MaskedLayoutScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, true, false, false); \
            }                                                                                                       \
          } else {                                                                                                  \
            if (round_scale) {                                                                                       \
              LAUNCH_KERNEL_INNER(                                                                                  \
                  NaiveScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, true, false, false, true);  \
            } else {                                                                                                \
              LAUNCH_KERNEL_INNER(                                                                                  \
                  NaiveScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, true, false, false);        \
            }                                                                                                       \
          }                                                                                                         \
        }                                                                                                           \
      }                                                                                                             \
    } else {                                                                                                        \
      if (fuse_silu_and_mul) {                                                                                       \
        if (masked_layout) {                                                                                         \
          LAUNCH_KERNEL_INNER(                                                                                      \
              MaskedLayoutScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, false, false, true);     \
        } else {                                                                                                    \
          LAUNCH_KERNEL_INNER(                                                                                      \
              NaiveScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, false, false, true);            \
        }                                                                                                           \
      } else {                                                                                                      \
        if (masked_layout) {                                                                                         \
          LAUNCH_KERNEL_INNER(                                                                                      \
              MaskedLayoutScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, false, false, false);    \
        } else {                                                                                                    \
          LAUNCH_KERNEL_INNER(                                                                                      \
              NaiveScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, false, false, false);           \
        }                                                                                                           \
      }                                                                                                             \
    }                                                                                                               \
  } while (0)

#define LAUNCH_KERNEL_OUTER(...)                    \
  switch (group_size) {                             \
    case 16:                                        \
      LAUNCH_KERNEL(16, __VA_ARGS__);               \
      break;                                        \
    case 32:                                        \
      LAUNCH_KERNEL(32, __VA_ARGS__);               \
      break;                                        \
    case 64:                                        \
      LAUNCH_KERNEL(64, __VA_ARGS__);               \
      break;                                        \
    case 128:                                       \
      LAUNCH_KERNEL(128, __VA_ARGS__);              \
      break;                                        \
    default:                                        \
      TORCH_CHECK(false, "Unsupported group_size"); \
  }                                                 \
  while (0)

  DISPATCH_INPUT_DTYPE(input.scalar_type(), scalar_t, [&] {
    if (dst_type == at::ScalarType::Char) {
      LAUNCH_KERNEL_OUTER(scalar_t, int8_t);
      return true;
    } else if (dst_type == at::ScalarType::Float8_e4m3fn) {
      LAUNCH_KERNEL_OUTER(scalar_t, c10::Float8_e4m3fn);
      return true;
    }
    return false;
  });

#undef LAUNCH_KERNEL
#undef LAUNCH_KERNEL_INNER
}

}  // namespace vllm::batch_invariant
