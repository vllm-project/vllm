#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/core/ScalarType.h>

#include "libtorch_stable/quantization/w8a8/per_token_group_quant_8bit.h"

#include <cmath>

#include <cuda_fp8.h>

#include "libtorch_stable/quantization/vectorization.cuh"
#include "libtorch_stable/quantization/vectorization_utils.cuh"
#include "libtorch_stable/dispatch_utils.h"
#include "libtorch_stable/torch_utils.h"

__device__ __forceinline__ float GroupReduceMax(float val) {
  unsigned mask = threadIdx.x % 32 >= 16 ? 0xffff0000 : 0x0000ffff;

  val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
  return val;
}

template <typename T, bool SCALE_UE8M0>
__device__ __forceinline__ float ComputeGroupScale(
    const T* __restrict__ group_input, T* __restrict__ smem_group,
    const int group_size, const int lane_id, const int threads_per_group,
    const float eps, const float max_8bit) {
  float local_absmax = eps;

  constexpr int vec_size = 16 / sizeof(T);

  // copy global -> shared & compute absmax
  auto scalar_op_cache = [&] __device__(T & dst, const T& src) {
    float abs_v = fabsf(static_cast<float>(src));
    local_absmax = fmaxf(local_absmax, abs_v);
    dst = src;
  };

  vllm::vectorize_with_alignment<vec_size>(
      group_input,        // in
      smem_group,         // out (shared)
      group_size,         // elements per group
      lane_id,            // thread id
      threads_per_group,  // stride in group
      scalar_op_cache);   // scalar handler

  local_absmax = GroupReduceMax(local_absmax);

  float y_s = local_absmax / max_8bit;
  if constexpr (SCALE_UE8M0) {
    y_s = exp2f(ceilf(log2f(fmaxf(fabsf(y_s), 1e-10f))));
  }

  return y_s;
}

template <typename T, typename DST_DTYPE>
__device__ __forceinline__ void QuantizeGroup(
    const T* __restrict__ smem_group, DST_DTYPE* __restrict__ group_output,
    const int group_size, const int lane_id, const int threads_per_group,
    const float y_s, const float min_8bit, const float max_8bit) {
  constexpr int vec_size = 16 / sizeof(T);

  // quantize shared -> global 8-bit
  auto scalar_op_quant = [&] __device__(DST_DTYPE & dst, const T& src) {
    float q = fminf(fmaxf(static_cast<float>(src) / y_s, min_8bit), max_8bit);
    dst = DST_DTYPE(q);
  };

  vllm::vectorize_with_alignment<vec_size>(
      smem_group,         // in (shared)
      group_output,       // out (global quant tensor)
      group_size,         // elements
      lane_id,            // tid
      threads_per_group,  // stride
      scalar_op_quant);   // scalar handler
}

template <typename T, typename DST_DTYPE, bool IS_COLUMN_MAJOR = false,
          bool SCALE_UE8M0 = false, typename scale_packed_t = float>
__global__ void per_token_group_quant_8bit_kernel(
    const T* __restrict__ input, void* __restrict__ output_q,
    scale_packed_t* __restrict__ output_s, const int group_size,
    const int num_groups, const int groups_per_block, const float eps,
    const float min_8bit, const float max_8bit, const int scale_num_rows = 0,
    const int scale_stride = 0) {
  const int threads_per_group = 16;
  const int64_t local_group_id = threadIdx.x / threads_per_group;
  const int lane_id = threadIdx.x % threads_per_group;

  const int64_t block_group_id = blockIdx.x * groups_per_block;
  const int64_t global_group_id = block_group_id + local_group_id;
  const int64_t block_group_offset = global_group_id * group_size;

  using scale_element_t = float;
  static_assert(sizeof(scale_packed_t) % sizeof(scale_element_t) == 0);

  const T* group_input = input + block_group_offset;
  DST_DTYPE* group_output =
      static_cast<DST_DTYPE*>(output_q) + block_group_offset;
  scale_element_t* scale_output;

  if constexpr (IS_COLUMN_MAJOR) {
    const int num_elems_per_pack =
        static_cast<int>(sizeof(scale_packed_t) / sizeof(scale_element_t));
    const int scale_num_rows_element = scale_num_rows * num_elems_per_pack;
    const int row_idx = global_group_id / scale_num_rows_element;
    const int col_idx_raw = global_group_id % scale_num_rows_element;
    const int col_idx = col_idx_raw / num_elems_per_pack;
    const int pack_idx = col_idx_raw % num_elems_per_pack;
    scale_output = reinterpret_cast<scale_element_t*>(output_s) +
                   (col_idx * scale_stride * num_elems_per_pack +
                    row_idx * num_elems_per_pack + pack_idx);
  } else {
    scale_output = output_s + global_group_id;
  }

  // shared memory to cache each group's data to avoid double DRAM reads.
  extern __shared__ __align__(16) char smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);
  T* smem_group = smem + local_group_id * group_size;

  const float y_s = ComputeGroupScale<T, SCALE_UE8M0>(
      group_input, smem_group, group_size, lane_id, threads_per_group, eps,
      max_8bit);

  scale_element_t y_s_quant = y_s;

  if (lane_id == 0) {
    *scale_output = y_s_quant;
  }

  __syncthreads();

  QuantizeGroup<T, DST_DTYPE>(smem_group, group_output, group_size, lane_id,
                              threads_per_group, y_s, min_8bit, max_8bit);
}

inline int GetGroupsPerBlock(int64_t num_groups) {
  if (num_groups % 16 == 0) {
    return 16;
  }
  if (num_groups % 8 == 0) {
    return 8;
  }
  if (num_groups % 4 == 0) {
    return 4;
  }
  if (num_groups % 2 == 0) {
    return 2;
  }
  return 1;
}

// Largest divisor of padded_groups_per_row that is <= 16. ry = 16 / kx.
inline int GetGroupsPerBlockX(int64_t padded_groups_per_row) {
  if (padded_groups_per_row % 16 == 0) {
    return 16;
  }
  if (padded_groups_per_row % 8 == 0) {
    return 8;
  }
  return 4;
}

void per_token_group_quant_8bit(const torch::stable::Tensor& input,
                                torch::stable::Tensor& output_q,
                                torch::stable::Tensor& output_s,
                                int64_t group_size, double eps, double min_8bit,
                                double max_8bit, bool scale_ue8m0) {
  STD_TORCH_CHECK(input.is_contiguous());
  STD_TORCH_CHECK(output_q.is_contiguous());

  const int num_groups = input.numel() / group_size;

  STD_TORCH_CHECK(input.numel() % group_size == 0);
  STD_TORCH_CHECK(output_s.dim() == 2);

  cudaStream_t stream = get_current_cuda_stream();

  constexpr int THREADS_PER_GROUP = 16;

  const int groups_per_block = GetGroupsPerBlock(num_groups);

  auto dst_type = output_q.scalar_type();
  const int num_blocks = num_groups / groups_per_block;
  const int num_threads = groups_per_block * THREADS_PER_GROUP;

  const bool is_column_major = output_s.stride(0) < output_s.stride(1);
  const int scale_num_rows = output_s.size(1);
  const int scale_stride = output_s.stride(1);

#define LAUNCH_KERNEL(T, DST_DTYPE)                                        \
  do {                                                                     \
    dim3 grid(num_blocks);                                                 \
    dim3 block(num_threads);                                               \
    size_t smem_bytes =                                                    \
        static_cast<size_t>(groups_per_block) * group_size * sizeof(T);    \
    if (is_column_major) {                                                 \
      if (scale_ue8m0) {                                                   \
        per_token_group_quant_8bit_kernel<T, DST_DTYPE, true, true>        \
            <<<grid, block, smem_bytes, stream>>>(                         \
                static_cast<T*>(input.data_ptr()), output_q.data_ptr(),    \
                static_cast<float*>(output_s.data_ptr()), group_size,      \
                num_groups, groups_per_block, (float)eps, (float)min_8bit, \
                (float)max_8bit, scale_num_rows, scale_stride);            \
      } else {                                                             \
        per_token_group_quant_8bit_kernel<T, DST_DTYPE, true, false>       \
            <<<grid, block, smem_bytes, stream>>>(                         \
                static_cast<T*>(input.data_ptr()), output_q.data_ptr(),    \
                static_cast<float*>(output_s.data_ptr()), group_size,      \
                num_groups, groups_per_block, (float)eps, (float)min_8bit, \
                (float)max_8bit, scale_num_rows, scale_stride);            \
      }                                                                    \
    } else {                                                               \
      if (scale_ue8m0) {                                                   \
        per_token_group_quant_8bit_kernel<T, DST_DTYPE, false, true>       \
            <<<grid, block, smem_bytes, stream>>>(                         \
                static_cast<T*>(input.data_ptr()), output_q.data_ptr(),    \
                static_cast<float*>(output_s.data_ptr()), group_size,      \
                num_groups, groups_per_block, (float)eps, (float)min_8bit, \
                (float)max_8bit);                                          \
      } else {                                                             \
        per_token_group_quant_8bit_kernel<T, DST_DTYPE, false, false>      \
            <<<grid, block, smem_bytes, stream>>>(                         \
                static_cast<T*>(input.data_ptr()), output_q.data_ptr(),    \
                static_cast<float*>(output_s.data_ptr()), group_size,      \
                num_groups, groups_per_block, (float)eps, (float)min_8bit, \
                (float)max_8bit);                                          \
      }                                                                    \
    }                                                                      \
  } while (0)

  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "per_token_group_quant_8bit", ([&] {
        if (dst_type == torch::headeronly::ScalarType::Float8_e4m3fn) {
          LAUNCH_KERNEL(scalar_t, __nv_fp8_e4m3);
        } else if (dst_type == torch::headeronly::ScalarType::Char) {
          LAUNCH_KERNEL(scalar_t, int8_t);
        }
      }));

#undef LAUNCH_KERNEL
}

// Register-resident fast path for group_size==128.
//
// Each thread holds 16 source elements (32 B = uint4 x 2) in registers across
// the absmax reduce -> scale compute -> quantize pipeline. No shared memory.
// UE8M0 scale extracted via bit math (bit-exact with exp2f(ceilf(log2f))).
//
// Loads two contiguous uint4s (16 B + 16 B = 32 B) per thread; on Blackwell
// nvcc fuses these into a single 256-bit LDG.E.256.
//
// Constraints: GROUP_SIZE % (THREADS_PER_GROUP * VEC_SIZE) == 0; for
// THREADS_PER_GROUP=8 and bf16/fp16 (VEC_SIZE=16), this means GROUP_SIZE=128.
template <typename T, typename DST_DTYPE, int GROUP_SIZE, int kGroupsPerBlockX,
          int kRowsPerBlock>
__global__ void per_token_group_quant_8bit_packed_register_kernel(
    const T* __restrict__ input, void* __restrict__ output_q,
    unsigned int* __restrict__ output_s_packed, const int padded_groups_per_row,
    const int groups_per_row, const int mn, const int output_q_mn_extent,
    const int tma_aligned_mn, const int64_t num_scale_elems, const float eps,
    const float min_8bit, const float max_8bit) {
  static_assert(GROUP_SIZE == 128, "fast path supports GROUP_SIZE==128");
  constexpr int THREADS_PER_GROUP = 8;
  constexpr int VEC_SIZE = 32 / sizeof(T);  // 16 for bf16/fp16
  static_assert(GROUP_SIZE == THREADS_PER_GROUP * VEC_SIZE,
                "GROUP_SIZE must equal THREADS_PER_GROUP * VEC_SIZE");
  static_assert(32 % THREADS_PER_GROUP == 0,
                "THREADS_PER_GROUP must divide warp size for the shuffle "
                "mask to be valid");
  static_assert(
      kGroupsPerBlockX > 0 && (kGroupsPerBlockX & (kGroupsPerBlockX - 1)) == 0,
      "kGroupsPerBlockX must be a positive power of 2");
  static_assert(kRowsPerBlock > 0, "kRowsPerBlock must be positive");

  const int local_group_id = threadIdx.x / THREADS_PER_GROUP;
  const int lane_id = threadIdx.x % THREADS_PER_GROUP;

  const int sf_k_local = local_group_id % kGroupsPerBlockX;
  const int row_local = local_group_id / kGroupsPerBlockX;
  const int sf_k_idx = blockIdx.x * kGroupsPerBlockX + sf_k_local;
  const int mn_idx = blockIdx.y * kRowsPerBlock + row_local;

  if (mn_idx >= tma_aligned_mn) {
    return;
  }
  const bool is_valid_group = (mn_idx < mn) && (sf_k_idx < groups_per_row);

  // Load 16 input elements (32 B) into registers as two adjacent uint4
  // loads. nvcc keeps these as 2x LDG.E.128 on sm_100; the per-thread cost
  // is dominated by HBM bandwidth at large MN, so a fused 256-bit load via
  // inline PTX gave no measurable speedup.
  // alignas(16) is required so the uint4* reinterpret_cast below is
  // well-defined for T == bf16/fp16 (default alignof is 2).
  alignas(16) T regs[VEC_SIZE];
  float local_absmax = eps;
  if (is_valid_group) {
    const T* group_input =
        input + static_cast<int64_t>(mn_idx) * groups_per_row * GROUP_SIZE +
        sf_k_idx * GROUP_SIZE + lane_id * VEC_SIZE;
    uint4* dst = reinterpret_cast<uint4*>(&regs[0]);
    const uint4* src = reinterpret_cast<const uint4*>(group_input);
    dst[0] = src[0];
    dst[1] = src[1];
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float v = fabsf(static_cast<float>(regs[i]));
      local_absmax = fmaxf(local_absmax, v);
    }
  }

  // 8-lane subgroup shuffle reduce (octet of the warp). The mask selects the
  // 8 lanes within the warp that share a group.
  unsigned mask = 0xffu << (threadIdx.x & 24u);
  local_absmax = fmaxf(local_absmax, __shfl_xor_sync(mask, local_absmax, 4));
  local_absmax = fmaxf(local_absmax, __shfl_xor_sync(mask, local_absmax, 2));
  local_absmax = fmaxf(local_absmax, __shfl_xor_sync(mask, local_absmax, 1));

  float y_s = local_absmax / max_8bit;
  y_s = fmaxf(y_s, 1e-10f);
  uint32_t bits = __float_as_uint(y_s);
  uint32_t exp_bits = (bits >> 23) & 0xffu;
  uint32_t mant_bits = bits & 0x7fffffu;
  uint8_t exp_byte =
      static_cast<uint8_t>(exp_bits + (mant_bits != 0u ? 1u : 0u));

  // Lane 0 writes the packed scale byte.
  if (lane_id == 0) {
    const int sf_k_pack_idx = sf_k_idx / 4;
    const int pos = sf_k_idx % 4;
    const int out_idx = sf_k_pack_idx * tma_aligned_mn + mn_idx;
    if (is_valid_group) {
      reinterpret_cast<uint8_t*>(output_s_packed)[out_idx * 4 + pos] = exp_byte;
    } else if (out_idx < num_scale_elems) {
      reinterpret_cast<uint8_t*>(output_s_packed)[out_idx * 4 + pos] = 0;
    }
  }

  // For padded mn rows that fall within output_q's allocated extent, write
  // a uint4 of zeros to keep the buffer clean for downstream TMA loads.
  // Skip writes for sf_k padding (those positions don't exist in output_q).
  if (!is_valid_group) {
    if (sf_k_idx < groups_per_row && mn_idx >= mn &&
        mn_idx < output_q_mn_extent) {
      DST_DTYPE* group_output =
          static_cast<DST_DTYPE*>(output_q) +
          static_cast<int64_t>(mn_idx) * groups_per_row * GROUP_SIZE +
          sf_k_idx * GROUP_SIZE + lane_id * VEC_SIZE;
      *reinterpret_cast<uint4*>(group_output) = make_uint4(0, 0, 0, 0);
    }
    return;
  }

  // Reconstruct y_s as a power-of-2 float and use its reciprocal.
  float y_s_q = __uint_as_float(static_cast<uint32_t>(exp_byte) << 23);
  float inv_y = 1.0f / y_s_q;

  // Quantize and pack into 16 fp8/int8 bytes (= uint4). VEC_SIZE==16 so we
  // fill four 32-bit words, four bytes each.
  uint32_t packed_lo = 0;
  uint32_t packed_lo_hi = 0;
  uint32_t packed_hi_lo = 0;
  uint32_t packed_hi = 0;
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    float q =
        fminf(fmaxf(static_cast<float>(regs[i]) * inv_y, min_8bit), max_8bit);
    DST_DTYPE qb = DST_DTYPE(q);
    uint8_t byte = *reinterpret_cast<uint8_t*>(&qb);
    const int shift = (i & 3) * 8;
    if (i < 4) {
      packed_lo |= static_cast<uint32_t>(byte) << shift;
    } else if (i < 8) {
      packed_lo_hi |= static_cast<uint32_t>(byte) << shift;
    } else if (i < 12) {
      packed_hi_lo |= static_cast<uint32_t>(byte) << shift;
    } else {
      packed_hi |= static_cast<uint32_t>(byte) << shift;
    }
  }

  uint4 packed_out =
      make_uint4(packed_lo, packed_lo_hi, packed_hi_lo, packed_hi);
  DST_DTYPE* group_output =
      static_cast<DST_DTYPE*>(output_q) +
      static_cast<int64_t>(mn_idx) * groups_per_row * GROUP_SIZE +
      sf_k_idx * GROUP_SIZE + lane_id * VEC_SIZE;
  *reinterpret_cast<uint4*>(group_output) = packed_out;
}

// Public entry point: register-resident packed quant kernel.
// Constraints: group_size == 128 and bf16/fp16 input.
void per_token_group_quant_8bit_packed(const torch::stable::Tensor& input,
                                       torch::stable::Tensor& output_q,
                                       torch::stable::Tensor& output_s_packed,
                                       int64_t group_size, double eps,
                                       double min_8bit, double max_8bit) {
  STD_TORCH_CHECK(group_size == 128,
                  "per_token_group_quant_8bit_packed only supports "
                  "group_size==128, got ",
                  group_size, ".");
  const auto in_dtype = input.scalar_type();
  STD_TORCH_CHECK(
      in_dtype == torch::headeronly::ScalarType::Half ||
          in_dtype == torch::headeronly::ScalarType::BFloat16,
      "per_token_group_quant_8bit_packed only supports bf16/fp16 input.");

  STD_TORCH_CHECK(input.is_contiguous());
  STD_TORCH_CHECK(output_q.is_contiguous());

  const int64_t k = input.size(-1);
  STD_TORCH_CHECK(k % group_size == 0, "input last dim k=", k,
                  " is not divisible by group_size=", group_size, ".");

  const int64_t mn = input.numel() / k;
  const int64_t groups_per_row = k / group_size;
  const int64_t k_num_packed_sfk = (groups_per_row + 3) / 4;
  const int64_t tma_aligned_mn = ((mn + 3) / 4) * 4;

  // output_q may be allocated with extra padded mn rows (e.g.,
  // (tma_aligned_mn, k)) so the kernel can zero-fill them in-line and the
  // caller can use torch.empty instead of torch.zeros. The grid only covers
  // up to tma_aligned_mn, so we cap the extent there.
  const int64_t output_q_mn_actual = output_q.numel() / k;
  STD_TORCH_CHECK(output_q_mn_actual >= mn,
                  "output_q must have at least mn rows; got ",
                  output_q_mn_actual, " rows for mn=", mn, ".");
  const int64_t output_q_mn_extent =
      output_q_mn_actual < tma_aligned_mn ? output_q_mn_actual : tma_aligned_mn;

  STD_TORCH_CHECK(
      output_s_packed.scalar_type() == torch::headeronly::ScalarType::Int,
      "output_s_packed must be int32 for UE8M0-packed scales.");
  STD_TORCH_CHECK(output_s_packed.size(0) == mn &&
                      output_s_packed.size(1) == k_num_packed_sfk,
                  "output_s_packed shape must be [", mn, ", ", k_num_packed_sfk,
                  "]; got [", output_s_packed.size(0), ", ",
                  output_s_packed.size(1), "].");
  STD_TORCH_CHECK(output_s_packed.stride(0) == 1 &&
                      output_s_packed.stride(1) == tma_aligned_mn,
                  "output_s_packed strides must be [1, ", tma_aligned_mn,
                  "]; got [", output_s_packed.stride(0), ", ",
                  output_s_packed.stride(1), "].");

  cudaStream_t stream = get_current_cuda_stream();

  constexpr int THREADS_PER_GROUP = 8;
  const int64_t padded_groups_per_row = k_num_packed_sfk * 4;
  const int64_t num_scale_elems = mn + (k_num_packed_sfk - 1) * tma_aligned_mn;

  STD_TORCH_CHECK(padded_groups_per_row % 4 == 0,
                  "padded_groups_per_row=", padded_groups_per_row,
                  " is not a multiple of 4.");
  const int kx = GetGroupsPerBlockX(padded_groups_per_row);
  const int ry = 16 / kx;
  const int64_t blocks_x = padded_groups_per_row / kx;
  const int64_t blocks_y = (tma_aligned_mn + ry - 1) / ry;
  const int num_threads = (kx * ry) * THREADS_PER_GROUP;
  // CUDA caps grid.x and grid.y at 2^31 - 1; guard against pathological inputs.
  STD_TORCH_CHECK(blocks_x <= static_cast<int64_t>(INT32_MAX) &&
                      blocks_y <= static_cast<int64_t>(INT32_MAX),
                  "per_token_group_quant_8bit_packed grid too large: (",
                  blocks_x, ", ", blocks_y, ").");

  auto dst_type = output_q.scalar_type();

#define LAUNCH_REG_KERNEL_INST(T, DST_DTYPE, KX, RY)                         \
  do {                                                                       \
    dim3 grid(static_cast<unsigned int>(blocks_x),                           \
              static_cast<unsigned int>(blocks_y));                          \
    dim3 block(num_threads);                                                 \
    per_token_group_quant_8bit_packed_register_kernel<T, DST_DTYPE, 128, KX, \
                                                      RY>                    \
        <<<grid, block, 0, stream>>>(                                        \
            static_cast<const T*>(input.data_ptr()), output_q.data_ptr(),    \
            reinterpret_cast<unsigned int*>(output_s_packed.data_ptr()),     \
            static_cast<int>(padded_groups_per_row),                         \
            static_cast<int>(groups_per_row), static_cast<int>(mn),          \
            static_cast<int>(output_q_mn_extent),                            \
            static_cast<int>(tma_aligned_mn), num_scale_elems,               \
            static_cast<float>(eps), static_cast<float>(min_8bit),           \
            static_cast<float>(max_8bit));                                   \
  } while (0)

#define LAUNCH_REG_KERNEL(T, DST_DTYPE)                    \
  do {                                                     \
    if (kx == 16) {                                        \
      LAUNCH_REG_KERNEL_INST(T, DST_DTYPE, 16, 1);         \
    } else if (kx == 8) {                                  \
      LAUNCH_REG_KERNEL_INST(T, DST_DTYPE, 8, 2);          \
    } else if (kx == 4) {                                  \
      LAUNCH_REG_KERNEL_INST(T, DST_DTYPE, 4, 4);          \
    } else {                                               \
      STD_TORCH_CHECK(false, "Unsupported kx value ", kx); \
    }                                                      \
  } while (0)

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      input.scalar_type(), "per_token_group_quant_8bit_packed_register", ([&] {
        if (dst_type == torch::headeronly::ScalarType::Float8_e4m3fn) {
          LAUNCH_REG_KERNEL(scalar_t, __nv_fp8_e4m3);
        } else if (dst_type == torch::headeronly::ScalarType::Char) {
          LAUNCH_REG_KERNEL(scalar_t, int8_t);
        } else {
          STD_TORCH_CHECK(
              false,
              "per_token_group_quant_8bit_packed only supports FP8/INT8 "
              "outputs.");
        }
      }));

#undef LAUNCH_REG_KERNEL
#undef LAUNCH_REG_KERNEL_INST
}

void per_token_group_quant_fp8(const torch::stable::Tensor& input,
                               torch::stable::Tensor& output_q,
                               torch::stable::Tensor& output_s,
                               int64_t group_size, double eps, double fp8_min,
                               double fp8_max, bool scale_ue8m0,
                               bool dummy_is_scale_transposed = false,
                               bool dummy_is_tma_aligned = false) {
  per_token_group_quant_8bit(input, output_q, output_s, group_size, eps,
                             fp8_min, fp8_max, scale_ue8m0);
}
