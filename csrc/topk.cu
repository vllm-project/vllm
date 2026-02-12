// Portions of this file are adapted from SGLang PR:
// https://github.com/sgl-project/sglang/pull/11194
// and
// https://github.com/sgl-project/sglang/pull/17747

#include "cuda_compat.h"
#include "dispatch_utils.h"

#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef USE_ROCM
  #include <cub/cub.cuh>
#else
  #include <hipcub/hipcub.hpp>
#endif

namespace vllm {

constexpr int TopK = 2048;              // DeepSeek V3 sparse attention top-k
constexpr int kThreadsPerBlock = 1024;  // Threads per block

// Shared memory budget
#if defined(USE_ROCM)
constexpr size_t kSmem = 48 * 1024;  // ROCm default: 48KB
#else
// Reduced from 128KB to 32KB to improve occupancy.
// Each radix pass needs at most ~TopK candidates in the threshold bin,
// so 4K entries per round (2 rounds = 8K entries = 32KB) is sufficient.
constexpr size_t kSmem = 8 * 1024 * sizeof(uint32_t);  // 32KB (bytes)
#endif

struct FastTopKParams {
  const float* __restrict__ input;         // [batch, seq_len] Logits
  const int32_t* __restrict__ row_starts;  // [batch] Offset into each row
                                           // (optional)
  int32_t* __restrict__ indices;           // [batch, TopK] Output top-k indices
  int32_t* __restrict__ lengths;           // [batch] Sequence lengths per row
  int64_t input_stride;                    // Stride between rows
};

__device__ __forceinline__ auto convert_to_uint32_v2(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                 : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ void naive_topk_cuda(const float* __restrict__ logits,
                                int32_t* __restrict__ output_indices,
                                int32_t seq_len) {
  const int thread_id = threadIdx.x;
  for (int i = thread_id; i < TopK; i += kThreadsPerBlock) {
    output_indices[i] = (i < seq_len) ? i : -1;
  }
}

// Adapted from:
// https://github.com/sgl-project/sglang/blob/v0.5.8/sgl-kernel/csrc/elementwise/topk.cu#L87
// by: DarkSharpness
// which at the same time is an optimized topk kernel copied from tilelang
// kernel
__device__ void fast_topk_cuda_tl(
    const float* __restrict__ logits,  // Input logits [seq_len]
    int* __restrict__ output_indices,  // Output top-k indices [TopK]
    int logits_offset,                 // Starting offset in logits array
    int seq_len)                       // Number of valid logits to process
{
  constexpr int RADIX = 256;
  constexpr int MAX_BUFFERED_ITEMS = kSmem / (2 * sizeof(int));

  alignas(128) __shared__ int shared_histogram[2][RADIX + 128];
  alignas(128) __shared__ int shared_output_count;
  alignas(128) __shared__ int shared_threshold_bin;
  alignas(128) __shared__ int shared_buffered_count[2];

  extern __shared__ int buffered_indices[][MAX_BUFFERED_ITEMS];

  const int thread_id = threadIdx.x;
  int remaining_k = TopK;

  // Pass 0: Build coarse 8-bit histogram using FP16 high bits
  if (thread_id < RADIX + 1) {
    shared_histogram[0][thread_id] = 0;
  }
  __syncthreads();

  for (int idx = thread_id; idx < seq_len; idx += kThreadsPerBlock) {
    const auto bin = convert_to_uint8(logits[idx + logits_offset]);
    ::atomicAdd(&shared_histogram[0][bin], 1);
  }
  __syncthreads();

  // Helper: Compute cumulative sum (suffix sum) over histogram using ping-pong
  // buffers
  auto compute_cumulative_sum = [&]() {
    static_assert(1 << 8 == RADIX,
                  "Radix must be 256 for 8 unrolled iterations");
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      if (C10_LIKELY(thread_id < RADIX)) {
        const int stride = 1 << i;
        const int src_buffer = i & 1;
        const int dst_buffer = src_buffer ^ 1;

        int value = shared_histogram[src_buffer][thread_id];
        if (thread_id < RADIX - stride) {
          value += shared_histogram[src_buffer][thread_id + stride];
        }
        shared_histogram[dst_buffer][thread_id] = value;
      }
      __syncthreads();
    }
  };

  compute_cumulative_sum();

  // Find threshold bin where cumsum crosses remaining_k
  if (thread_id < RADIX && shared_histogram[0][thread_id] > remaining_k &&
      shared_histogram[0][thread_id + 1] <= remaining_k) {
    shared_threshold_bin = thread_id;
    shared_buffered_count[0] = 0;
    shared_output_count = 0;
  }
  __syncthreads();

  const int threshold_bin = shared_threshold_bin;
  remaining_k -= shared_histogram[0][threshold_bin + 1];

  // Early exit if threshold bin perfectly matches remaining_k
  if (remaining_k == 0) {
    for (int idx = thread_id; idx < seq_len; idx += kThreadsPerBlock) {
      const int bin = convert_to_uint8(logits[idx + logits_offset]);
      if (bin > threshold_bin) {
        const int output_pos = ::atomicAdd(&shared_output_count, 1);
        output_indices[output_pos] = idx;
      }
    }
    __syncthreads();
    return;
  }

  // Prepare for refinement passes: Process threshold bin
  __syncthreads();
  if (thread_id < RADIX + 1) {
    shared_histogram[0][thread_id] = 0;
  }
  __syncthreads();

  // Scan all elements and:
  // 1. Write indices > threshold_bin to output
  // 2. Buffer indices == threshold_bin for refinement
  // 3. Build histogram for next refinement pass (fused optimization)
  for (int idx = thread_id; idx < seq_len; idx += kThreadsPerBlock) {
    const float logit_value = logits[idx + logits_offset];
    const int bin = convert_to_uint8(logit_value);

    if (bin > threshold_bin) {
      // in top-k, write to output
      const int output_pos = ::atomicAdd(&shared_output_count, 1);
      output_indices[output_pos] = idx;
    } else if (bin == threshold_bin) {
      // Candidate for top-k, needs refinement
      const int buffer_pos = ::atomicAdd(&shared_buffered_count[0], 1);
      if (C10_LIKELY(buffer_pos < MAX_BUFFERED_ITEMS)) {
        buffered_indices[0][buffer_pos] = idx;
        // Fused: Build histogram for next pass
        const uint32_t fp32_bits = convert_to_uint32_v2(logit_value);
        const int next_bin = (fp32_bits >> 24) & 0xFF;
        ::atomicAdd(&shared_histogram[0][next_bin], 1);
      }
    }
  }
  __syncthreads();

  // ============================================================================
  // Passes 1-4: Refine using 8-bit passes over FP32 bits
  // ============================================================================
  // FP32 bits [31:0] split into 4 bytes processed MSB-first:
  // Pass 1: bits [31:24], Pass 2: bits [23:16], Pass 3: bits [15:8], Pass 4:
  // bits [7:0]
#pragma unroll 4
  for (int pass = 0; pass < 4; ++pass) {
    __shared__ int shared_final_k;  // For final pass: remaining slots to fill
    const int src_buffer = pass % 2;
    const int dst_buffer = src_buffer ^ 1;

    // Clamp buffered count to prevent overflow
    const int raw_buffered = shared_buffered_count[src_buffer];
    const int num_buffered =
        (raw_buffered < MAX_BUFFERED_ITEMS) ? raw_buffered : MAX_BUFFERED_ITEMS;

    compute_cumulative_sum();

    // Find threshold bin for this pass
    if (thread_id < RADIX && shared_histogram[0][thread_id] > remaining_k &&
        shared_histogram[0][thread_id + 1] <= remaining_k) {
      shared_threshold_bin = thread_id;
      shared_buffered_count[dst_buffer] = 0;
      shared_final_k = remaining_k - shared_histogram[0][thread_id + 1];
    }
    __syncthreads();

    const int threshold_bin = shared_threshold_bin;
    remaining_k -= shared_histogram[0][threshold_bin + 1];

    // Bit offset for this pass: 24, 16, 8, 0
    const int bit_offset = 24 - pass * 8;

    // Early exit if threshold bin perfectly matches
    if (remaining_k == 0) {
      for (int i = thread_id; i < num_buffered; i += kThreadsPerBlock) {
        const int idx = buffered_indices[src_buffer][i];
        const uint32_t fp32_bits =
            convert_to_uint32_v2(logits[idx + logits_offset]);
        const int bin = (fp32_bits >> bit_offset) & 0xFF;
        if (bin > threshold_bin) {
          const int output_pos = ::atomicAdd(&shared_output_count, 1);
          output_indices[output_pos] = idx;
        }
      }
      __syncthreads();
      break;
    }

    // Continue refinement
    __syncthreads();
    if (thread_id < RADIX + 1) {
      shared_histogram[0][thread_id] = 0;
    }
    __syncthreads();

    for (int i = thread_id; i < num_buffered; i += kThreadsPerBlock) {
      const int idx = buffered_indices[src_buffer][i];
      const float logit_value = logits[idx + logits_offset];
      const uint32_t fp32_bits = convert_to_uint32_v2(logit_value);
      const int bin = (fp32_bits >> bit_offset) & 0xFF;

      if (bin > threshold_bin) {
        // Definitely in top-k
        const int output_pos = ::atomicAdd(&shared_output_count, 1);
        output_indices[output_pos] = idx;
      } else if (bin == threshold_bin) {
        if (pass == 3) {
          // Final pass (bits [7:0]): No more refinement possible
          // Fill remaining slots in reverse order to maintain descending order
          const int slot = ::atomicAdd(&shared_final_k, -1);
          if (slot > 0) {
            output_indices[TopK - slot] = idx;
          }
        } else {
          // Buffer for next pass and build next histogram
          const int buffer_pos =
              ::atomicAdd(&shared_buffered_count[dst_buffer], 1);
          if (C10_LIKELY(buffer_pos < MAX_BUFFERED_ITEMS)) {
            buffered_indices[dst_buffer][buffer_pos] = idx;
            // Fused: Build histogram for next pass
            const int next_bit_offset = bit_offset - 8;
            const int next_bin = (fp32_bits >> next_bit_offset) & 0xFF;
            ::atomicAdd(&shared_histogram[0][next_bin], 1);
          }
        }
      }
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) void topk_kernel(
    const FastTopKParams params) {
  const auto& [input, row_starts, indices, lengths, input_stride] = params;
  const uint64_t batch_idx = blockIdx.x;
  const int logits_offset = row_starts == nullptr ? 0 : row_starts[batch_idx];
  const int seq_len = lengths[batch_idx];
  int* output_indices = indices + batch_idx * TopK;
  const float* logits = input + batch_idx * input_stride;

  if (seq_len <= TopK) {
    // Shortcut: All elements are in top-k
    return naive_topk_cuda(logits, output_indices, seq_len);
  } else {
    return fast_topk_cuda_tl(logits, output_indices, logits_offset, seq_len);
  }
}

FastTopKParams get_params(
    const at::Tensor& score, const at::Tensor& lengths,
    std::optional<at::Tensor> row_starts_opt = std::nullopt,
    std::optional<at::Tensor> indices_opt = std::nullopt) {
  const int64_t batch_size = score.size(0);

  TORCH_CHECK(score.dim() == 2 && score.stride(1) == 1,
              "score must be 2D with contiguous rows");
  TORCH_CHECK(lengths.dim() == 1 && lengths.is_contiguous() &&
                  lengths.size(0) == batch_size,
              "lengths must be 1D contiguous with size matching batch");

  const int32_t* row_starts_ptr = nullptr;
  if (row_starts_opt.has_value()) {
    const auto& row_starts = *row_starts_opt;
    TORCH_CHECK(row_starts.dim() == 1 && row_starts.size(0) == batch_size,
                "row_starts must be 1D with size matching batch");
    row_starts_ptr = row_starts.data_ptr<int32_t>();
  }

  int32_t* indices_ptr = nullptr;
  if (indices_opt.has_value()) {
    const auto& indices = *indices_opt;
    TORCH_CHECK(indices.dim() == 2 && indices.is_contiguous() &&
                    indices.size(0) == batch_size && indices.size(1) == TopK,
                "indices must be 2D contiguous [batch, TopK]");
    indices_ptr = indices.data_ptr<int32_t>();
  }

  return FastTopKParams{
      .input = score.data_ptr<float>(),
      .row_starts = row_starts_ptr,
      .indices = indices_ptr,
      .lengths = lengths.data_ptr<int32_t>(),
      .input_stride = score.stride(0),
  };
}

template <auto* kernel_func, size_t smem_bytes>
void setup_kernel_smem_once() {
  static const cudaError_t result = []() -> cudaError_t {
#ifdef USE_ROCM
    auto func_ptr = reinterpret_cast<const void*>(kernel_func);
#else
    auto func_ptr = kernel_func;
#endif
    return cudaFuncSetAttribute(
        func_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  }();

  TORCH_CHECK(
      result == cudaSuccess,
      "Failed to set kernel shared memory limit: ", cudaGetErrorString(result));
}

}  // namespace vllm

void large_context_topk(
    const torch::Tensor& logits, torch::Tensor& indices,
    const torch::Tensor& seq_lens,
    c10::optional<torch::Tensor> row_starts = c10::nullopt) {
  TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(seq_lens.is_cuda(), "seq_lens must be a CUDA tensor");
  if (row_starts.has_value()) {
    TORCH_CHECK(row_starts->is_cuda(), "row_starts must be a CUDA tensor");
  }

  const auto params = vllm::get_params(logits, seq_lens, row_starts, indices);
  const int64_t batch_size = logits.size(0);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<uint32_t>(batch_size));
  const dim3 block(vllm::kThreadsPerBlock);

  vllm::setup_kernel_smem_once<vllm::topk_kernel, vllm::kSmem>();
  vllm::topk_kernel<<<grid, block, vllm::kSmem, stream>>>(params);

  const cudaError_t result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess,
              "large_context_topk kernel failed: ", cudaGetErrorString(result));
}