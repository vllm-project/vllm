/**
 * Float Top-K kernel with multi-round radix select
 *
 * Based on sglang's algorithm but simplified for general use:
 * - Supports arbitrary k (not hardcoded to 2048)
 * - 4-round radix select for full 32-bit precision
 * - Handles float32 scores correctly
 *
 * Reference:
 * https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/elementwise/topk.cu
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int BLOCK_SIZE = 1024;
constexpr int RADIX = 256;

// Convert float to order-preserving uint32
__device__ __forceinline__ uint32_t float_to_ordered_uint32(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Extract 8-bit bucket from uint32 at given byte offset
__device__ __forceinline__ uint8_t extract_byte(uint32_t x, int byte_offset) {
  return static_cast<uint8_t>((x >> byte_offset) & 0xFF);
}

// Naive topk for length <= k
__device__ void naive_topk(int64_t* __restrict__ indices, int32_t length,
                           int32_t k) {
  const int tid = threadIdx.x;
  for (int i = tid; i < k; i += BLOCK_SIZE) {
    indices[i] = (i < length) ? i : -1;
  }
}

/**
 * Multi-round radix select top-k
 *
 * Uses 4 rounds of 8-bit radix select to achieve full 32-bit precision.
 * Each round narrows down the candidate set until exactly k elements remain.
 *
 * Histogram uses static shared memory (separate from candidate index buffers)
 * to avoid sentinel corruption during double-buffered cumsum.
 */
__global__ void float_topk_kernel(const float* __restrict__ input,  // [B, L]
                                  int64_t* __restrict__ indices,    // [B, k]
                                  int64_t input_stride, int32_t length,
                                  int32_t k) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;

  const float* in = input + bid * input_stride;
  int64_t* out = indices + bid * k;

  // Sentinel init (B+C): the radix-select paths below can leave some of the
  // k output slots unwritten when fewer than k candidates survive (e.g. a
  // truncated candidate buffer or duplicate-heavy score distributions). The
  // output tensor is allocated with torch::empty, so an unwritten slot would
  // otherwise carry stale garbage and silently corrupt downstream indexing
  // (observed as an illegal memory access in quest_map_back at large
  // bs*context). Pre-fill every slot with -1 so unwritten slots are a clean
  // sentinel that consumers can skip. This is a single coalesced pass of k
  // writes per block (k ~ hundreds) — negligible next to the radix select.
  for (int i = tid; i < k; i += BLOCK_SIZE) {
    out[i] = -1;
  }
  __syncthreads();

  // Early exit for length <= k
  if (length <= k) {
    naive_topk(out, length, k);
    return;
  }

  // Static shared: histogram double buffer with +1 sentinel slot
  __shared__ int s_histogram_buf[2][RADIX + 1];
  // Dynamic shared: candidate index double buffer
  extern __shared__ int s_input_idx[];

  __shared__ int s_counter;
  __shared__ int s_threshold_bin;
  __shared__ int s_num_input[2];
  __shared__ int s_last_remain;

  int* s_histogram = s_histogram_buf[0];
  const int smem_idx_size = (64 * 1024 / sizeof(int)) / 2;

  int topk_remain = k;

  // ===== Round 0: 8-bit coarse histogram (bits 24-31) =====
  if (tid < RADIX + 1) {
    s_histogram_buf[0][tid] = 0;
    s_histogram_buf[1][tid] = 0;
  }
  if (tid == 0) {
    s_counter = 0;
    s_num_input[0] = 0;
  }
  __syncthreads();

  // Build histogram
  for (int idx = tid; idx < length; idx += BLOCK_SIZE) {
    uint32_t ordered = float_to_ordered_uint32(in[idx]);
    uint8_t bin = extract_byte(ordered, 24);
    atomicAdd(&s_histogram_buf[0][bin], 1);
  }
  __syncthreads();

// Cumulative sum (suffix sum via parallel prefix)
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (tid < RADIX) {
      int j = 1 << i;
      int k_idx = i & 1;
      int value = s_histogram_buf[k_idx][tid];
      if (tid < RADIX - j) {
        value += s_histogram_buf[k_idx][tid + j];
      }
      s_histogram_buf[k_idx ^ 1][tid] = value;
    }
    __syncthreads();
  }

  // After 8 iterations, result is in buf[0]
  s_histogram = s_histogram_buf[0];

  // Find threshold bucket
  if (tid < RADIX && s_histogram[tid] > topk_remain &&
      s_histogram[tid + 1] <= topk_remain) {
    s_threshold_bin = tid;
  }
  __syncthreads();

  const int threshold_bin_r0 = s_threshold_bin;
  topk_remain -= s_histogram[threshold_bin_r0 + 1];

  // Collect bin > threshold
  if (topk_remain == 0) {
    for (int idx = tid; idx < length; idx += BLOCK_SIZE) {
      uint32_t ordered = float_to_ordered_uint32(in[idx]);
      uint8_t bin = extract_byte(ordered, 24);
      if (bin > threshold_bin_r0) {
        int pos = atomicAdd(&s_counter, 1);
        if (pos < k) out[pos] = idx;
      }
    }
    return;
  }

  // Reset histogram for next round
  if (tid < RADIX + 1) {
    s_histogram_buf[0][tid] = 0;
    s_histogram_buf[1][tid] = 0;
  }
  __syncthreads();

  // Collect bin > threshold and store bin == threshold
  for (int idx = tid; idx < length; idx += BLOCK_SIZE) {
    float val = in[idx];
    uint32_t ordered = float_to_ordered_uint32(val);
    uint8_t bin = extract_byte(ordered, 24);

    if (bin > threshold_bin_r0) {
      int pos = atomicAdd(&s_counter, 1);
      if (pos < k) out[pos] = idx;
    } else if (bin == threshold_bin_r0) {
      int pos = atomicAdd(&s_num_input[0], 1);
      if (pos < smem_idx_size) {
        s_input_idx[pos] = idx;
        uint8_t sub_bin = extract_byte(ordered, 16);
        atomicAdd(&s_histogram_buf[0][sub_bin], 1);
      }
    }
  }
  __syncthreads();

// ===== Rounds 1-3: Refine with remaining bytes =====
#pragma unroll
  for (int round = 1; round < 4; ++round) {
    const int r_idx = (round - 1) % 2;
    const int num_input = min(s_num_input[r_idx], smem_idx_size);
    const int byte_offset = 24 - round * 8;

// Cumulative sum
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (tid < RADIX) {
        int j = 1 << i;
        int k_idx = i & 1;
        int value = s_histogram_buf[k_idx][tid];
        if (tid < RADIX - j) {
          value += s_histogram_buf[k_idx][tid + j];
        }
        s_histogram_buf[k_idx ^ 1][tid] = value;
      }
      __syncthreads();
    }

    s_histogram = s_histogram_buf[0];

    // Find threshold
    if (tid < RADIX && s_histogram[tid] > topk_remain &&
        s_histogram[tid + 1] <= topk_remain) {
      s_threshold_bin = tid;
      s_last_remain = topk_remain - s_histogram[tid + 1];
    }
    if (tid == 0) {
      s_num_input[r_idx ^ 1] = 0;
    }
    __syncthreads();

    const int threshold_bin = s_threshold_bin;
    topk_remain -= s_histogram[threshold_bin + 1];

    if (topk_remain == 0) {
      for (int i = tid; i < num_input; i += BLOCK_SIZE) {
        int idx = s_input_idx[r_idx * smem_idx_size + i];
        uint32_t ordered = float_to_ordered_uint32(in[idx]);
        uint8_t bin = extract_byte(ordered, byte_offset);
        if (bin > threshold_bin) {
          int pos = atomicAdd(&s_counter, 1);
          if (pos < k) out[pos] = idx;
        }
      }
      return;
    }

    // Reset histogram
    if (tid < RADIX + 1) {
      s_histogram_buf[0][tid] = 0;
      s_histogram_buf[1][tid] = 0;
    }
    __syncthreads();

    // Collect and refine
    for (int i = tid; i < num_input; i += BLOCK_SIZE) {
      int idx = s_input_idx[r_idx * smem_idx_size + i];
      float val = in[idx];
      uint32_t ordered = float_to_ordered_uint32(val);
      uint8_t bin = extract_byte(ordered, byte_offset);

      if (bin > threshold_bin) {
        int pos = atomicAdd(&s_counter, 1);
        if (pos < k) out[pos] = idx;
      } else if (bin == threshold_bin) {
        if (round == 3) {
          // Last round: fill remaining slots
          int pos = atomicAdd(&s_last_remain, -1);
          if (pos > 0) {
            out[k - pos] = idx;
          }
        } else {
          int pos = atomicAdd(&s_num_input[r_idx ^ 1], 1);
          if (pos < smem_idx_size) {
            s_input_idx[(r_idx ^ 1) * smem_idx_size + pos] = idx;
            uint8_t sub_bin = extract_byte(ordered, byte_offset - 8);
            atomicAdd(&s_histogram_buf[0][sub_bin], 1);
          }
        }
      }
    }
    __syncthreads();
  }
}

}  // namespace

// =====================================================================
// Python Interface
// =====================================================================

torch::Tensor float_topk_cuda(torch::Tensor input, int64_t k) {
  TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
  TORCH_CHECK(input.dim() == 2, "input must be 2D [B, L]");
  TORCH_CHECK(k > 0, "k must be positive");

  const int B = input.size(0);
  const int L = input.size(1);

  TORCH_CHECK(k <= L, "k cannot exceed sequence length");

  torch::Tensor input_float;
  if (input.scalar_type() != torch::kFloat32) {
    input_float = input.to(torch::kFloat32).contiguous();
  } else {
    input_float = input.contiguous();
  }

  auto indices = torch::empty(
      {B, k},
      torch::TensorOptions().dtype(torch::kInt64).device(input.device()));

  c10::cuda::CUDAGuard device_guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  // Dynamic shared memory: candidate index double buffer (extern __shared__
  // s_input_idx). sm_80 default cap is 48KB; request 64KB before launch (same
  // pattern as collision.cu).
  constexpr size_t smem = 64 * 1024;
  auto err_attr = cudaFuncSetAttribute(
      float_topk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem));
  TORCH_CHECK(
      err_attr == cudaSuccess,
      "float_topk cudaFuncSetAttribute(MaxDynamicSharedMemorySize) failed: ",
      cudaGetErrorString(err_attr));

  float_topk_kernel<<<B, BLOCK_SIZE, smem, stream>>>(
      input_float.data_ptr<float>(), indices.data_ptr<int64_t>(),
      input_float.stride(0), L, k);

  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "float_topk kernel failed: ", cudaGetErrorString(err));

  return indices;
}

torch::Tensor float_topk_3d_cuda(torch::Tensor input, int64_t k) {
  TORCH_CHECK(input.dim() == 3, "input must be 3D [bs, kv_heads, kv_len]");

  const int bs = input.size(0);
  const int kv_heads = input.size(1);
  const int kv_len = input.size(2);

  auto input_2d = input.reshape({bs * kv_heads, kv_len});
  auto indices_2d = float_topk_cuda(input_2d, k);
  return indices_2d.reshape({bs, kv_heads, k});
}

#ifndef ZOOMKV_UNIFIED_EXTENSION
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("float_topk", &float_topk_cuda,
        "Float Top-K with multi-round radix select (2D input)",
        py::arg("input"), py::arg("k"));
  m.def("float_topk_3d", &float_topk_3d_cuda,
        "Float Top-K with multi-round radix select (3D input)",
        py::arg("input"), py::arg("k"));
}
#endif
