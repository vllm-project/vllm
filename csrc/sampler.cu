#include "dispatch_utils.h"

#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef USE_ROCM
  #include <cub/cub.cuh>
#else
  #include <hipcub/hipcub.hpp>
#endif

namespace vllm {

template <typename scalar_t>
__global__ void apply_repetition_penalties_kernel(
    scalar_t* __restrict__ logits,         // [num_seqs, vocab_size]
    const bool* __restrict__ prompt_mask,  // [num_seqs, vocab_size]
    const bool* __restrict__ output_mask,  // [num_seqs, vocab_size]
    const scalar_t* __restrict__ repetition_penalties,  // [num_seqs]
    const int num_seqs, const int vocab_size, const int tile_size) {
  // Each block handles one sequence and a tile of vocab
  const int seq_idx = blockIdx.x;
  if (seq_idx >= num_seqs) return;

  const int tile_start = blockIdx.y * tile_size;
  const int tile_end = min(tile_start + tile_size, vocab_size);

  // Load repetition penalty for this sequence
  const scalar_t penalty = repetition_penalties[seq_idx];

  // Each thread processes multiple vocab items within the tile
  for (int vocab_idx = tile_start + threadIdx.x; vocab_idx < tile_end;
       vocab_idx += blockDim.x) {
    const int64_t idx = static_cast<int64_t>(seq_idx) * vocab_size + vocab_idx;
    const bool is_repeated = prompt_mask[idx] || output_mask[idx];
    if (is_repeated) {
      scalar_t logit = logits[idx];
      if (logit > 0) {
        logits[idx] = logit / penalty;
      } else {
        logits[idx] = logit * penalty;
      }
    }
  }
}

}  // namespace vllm

void apply_repetition_penalties_(
    torch::Tensor& logits,             // [num_seqs, vocab_size], in-place
    const torch::Tensor& prompt_mask,  // [num_seqs, vocab_size]
    const torch::Tensor& output_mask,  // [num_seqs, vocab_size]
    const torch::Tensor& repetition_penalties) {  // [num_seqs]
  TORCH_CHECK(logits.is_contiguous());
  TORCH_CHECK(prompt_mask.is_contiguous());
  TORCH_CHECK(output_mask.is_contiguous());
  TORCH_CHECK(repetition_penalties.is_contiguous());

  int vocab_size = logits.size(-1);
  int num_seqs = logits.size(0);

  if (num_seqs == 0) return;

  // Get number of SMs on the current device
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                         logits.get_device());

  // Compute tile_num and tile_size
  int tile_num =
      std::min(vocab_size, std::max(1, (sms + num_seqs - 1) / num_seqs));
  int tile_size = (vocab_size + tile_num - 1) / tile_num;

  // Each block handles one sequence and a tile of vocab
  dim3 grid(num_seqs, tile_num);
  dim3 block(std::min(tile_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(logits));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      logits.scalar_type(), "apply_repetition_penalties_kernel", [&] {
        vllm::apply_repetition_penalties_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                logits.data_ptr<scalar_t>(), prompt_mask.data_ptr<bool>(),
                output_mask.data_ptr<bool>(),
                repetition_penalties.data_ptr<scalar_t>(), num_seqs, vocab_size,
                tile_size);
      });
}