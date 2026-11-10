// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused vocab-parallel embedding lookup.
//
// With TP > 1 each rank owns a slice of the vocabulary, so the eager path is a
// range mask, an id shift, an int64 cast, a gather and a masked_fill_ of the
// gathered rows. This kernel does all of it in one pass: rows owned by another
// rank are written as zeros, which keeps the following all-reduce exact.
//
// The gather is a pure row copy, so the kernel is templated on a byte-vector
// type instead of the value type and works for any embedding dtype.

#include "torch_utils.h"

#include "ops.h"

#include <algorithm>
#include <cstdint>

namespace vllm::vocab_embedding {

constexpr int kBlockThreads = 256;
// Enough token-blocks to fill the device several times over; the token loop
// strides over the rest.
constexpr int kTokenWaves = 4;

template <typename idx_t, typename vec_t>
__global__ void vocab_parallel_embedding_kernel(
    vec_t* __restrict__ out,              // [num_tokens, vecs_per_row]
    const idx_t* __restrict__ input_ids,  // [num_tokens]
    const vec_t* __restrict__ weight,     // [num_rows, vecs_per_row]
    const int64_t num_tokens, const int64_t vecs_per_row,
    const int64_t org_vocab_start_index, const int64_t org_vocab_end_index,
    const int64_t added_vocab_start_index, const int64_t added_vocab_end_index,
    const int64_t added_row_offset) {
  const int64_t vec_begin =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t vec_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t token = blockIdx.y; token < num_tokens; token += gridDim.y) {
    const int64_t id = static_cast<int64_t>(input_ids[token]);
    const bool in_org = id >= org_vocab_start_index && id < org_vocab_end_index;
    const bool in_added =
        id >= added_vocab_start_index && id < added_vocab_end_index;
    vec_t* out_row = out + token * vecs_per_row;

    if (in_org || in_added) {
      const int64_t row =
          in_org ? id - org_vocab_start_index : id + added_row_offset;
      const vec_t* weight_row = weight + row * vecs_per_row;
      for (int64_t v = vec_begin; v < vecs_per_row; v += vec_stride) {
        out_row[v] = weight_row[v];
      }
    } else {
      // Token belongs to another rank's shard: contribute zeros.
      const vec_t zero = vec_t();
      for (int64_t v = vec_begin; v < vecs_per_row; v += vec_stride) {
        out_row[v] = zero;
      }
    }
  }
}

template <typename idx_t, typename vec_t>
void launch(torch::stable::Tensor& out, const torch::stable::Tensor& input_ids,
            const torch::stable::Tensor& weight, int64_t org_vocab_start_index,
            int64_t org_vocab_end_index, int64_t added_vocab_start_index,
            int64_t added_vocab_end_index, int64_t added_row_offset,
            int64_t num_tokens, int64_t row_bytes, cudaStream_t stream) {
  const int64_t vecs_per_row = row_bytes / static_cast<int64_t>(sizeof(vec_t));
  const int grid_x = static_cast<int>(
      std::min<int64_t>((vecs_per_row + kBlockThreads - 1) / kBlockThreads,
                        static_cast<int64_t>(kBlockThreads)));
  const int grid_y = static_cast<int>(std::min<int64_t>(
      num_tokens, static_cast<int64_t>(kTokenWaves) *
                      get_device_prop()->multiProcessorCount));

  vocab_parallel_embedding_kernel<idx_t, vec_t>
      <<<dim3(grid_x, grid_y), kBlockThreads, 0, stream>>>(
          reinterpret_cast<vec_t*>(out.mutable_data_ptr()),
          reinterpret_cast<const idx_t*>(input_ids.const_data_ptr()),
          reinterpret_cast<const vec_t*>(weight.const_data_ptr()), num_tokens,
          vecs_per_row, org_vocab_start_index, org_vocab_end_index,
          added_vocab_start_index, added_vocab_end_index, added_row_offset);
}

template <typename idx_t>
void dispatch_vec_width(
    torch::stable::Tensor& out, const torch::stable::Tensor& input_ids,
    const torch::stable::Tensor& weight, int64_t org_vocab_start_index,
    int64_t org_vocab_end_index, int64_t added_vocab_start_index,
    int64_t added_vocab_end_index, int64_t added_row_offset, int64_t num_tokens,
    int64_t row_bytes, cudaStream_t stream) {
  const uintptr_t addrs = reinterpret_cast<uintptr_t>(out.mutable_data_ptr()) |
                          reinterpret_cast<uintptr_t>(weight.const_data_ptr()) |
                          static_cast<uintptr_t>(row_bytes);

#define VLLM_LAUNCH_VOCAB_EMBEDDING(vec_t)                                  \
  launch<idx_t, vec_t>(out, input_ids, weight, org_vocab_start_index,       \
                       org_vocab_end_index, added_vocab_start_index,        \
                       added_vocab_end_index, added_row_offset, num_tokens, \
                       row_bytes, stream)

  if (addrs % 16 == 0) {
    VLLM_LAUNCH_VOCAB_EMBEDDING(uint4);
  } else if (addrs % 8 == 0) {
    VLLM_LAUNCH_VOCAB_EMBEDDING(uint2);
  } else if (addrs % 4 == 0) {
    VLLM_LAUNCH_VOCAB_EMBEDDING(uint32_t);
  } else if (addrs % 2 == 0) {
    VLLM_LAUNCH_VOCAB_EMBEDDING(uint16_t);
  } else {
    VLLM_LAUNCH_VOCAB_EMBEDDING(uint8_t);
  }
#undef VLLM_LAUNCH_VOCAB_EMBEDDING
}

}  // namespace vllm::vocab_embedding

void vocab_parallel_embedding(
    torch::stable::Tensor& out, const torch::stable::Tensor& input_ids,
    const torch::stable::Tensor& weight, int64_t org_vocab_start_index,
    int64_t org_vocab_end_index, int64_t num_org_vocab_padding,
    int64_t added_vocab_start_index, int64_t added_vocab_end_index) {
  STD_TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
  STD_TORCH_CHECK(out.dim() == 2, "out must be 2D");
  STD_TORCH_CHECK(out.size(1) == weight.size(1),
                  "out and weight must share the embedding dim");
  STD_TORCH_CHECK(out.size(0) == input_ids.numel(),
                  "out must have one row per input id");
  STD_TORCH_CHECK(out.scalar_type() == weight.scalar_type(),
                  "out and weight must have the same dtype");
  STD_TORCH_CHECK(input_ids.is_contiguous() && weight.is_contiguous() &&
                      out.is_contiguous(),
                  "vocab_parallel_embedding requires contiguous tensors");

  const int64_t num_tokens = input_ids.numel();
  if (num_tokens == 0) return;

  const int64_t row_bytes = weight.size(1) * weight.element_size();
  // Offset that maps an added-vocab (LoRA) id onto its local row, mirroring
  // get_masked_input_and_mask().
  const int64_t added_row_offset = org_vocab_end_index - org_vocab_start_index +
                                   num_org_vocab_padding -
                                   added_vocab_start_index;

  const torch::stable::accelerator::DeviceGuard device_guard(
      out.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  if (input_ids.scalar_type() == torch::headeronly::ScalarType::Long) {
    vllm::vocab_embedding::dispatch_vec_width<int64_t>(
        out, input_ids, weight, org_vocab_start_index, org_vocab_end_index,
        added_vocab_start_index, added_vocab_end_index, added_row_offset,
        num_tokens, row_bytes, stream);
  } else {
    STD_TORCH_CHECK(
        input_ids.scalar_type() == torch::headeronly::ScalarType::Int,
        "input_ids must be int32 or int64");
    vllm::vocab_embedding::dispatch_vec_width<int32_t>(
        out, input_ids, weight, org_vocab_start_index, org_vocab_end_index,
        added_vocab_start_index, added_vocab_end_index, added_row_offset,
        num_tokens, row_bytes, stream);
  }
}
