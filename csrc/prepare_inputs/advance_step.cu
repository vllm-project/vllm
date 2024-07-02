/*
 * TODO: Add doc
 */

#include "advance_step.cuh"

namespace prepare_inputs {

__global__ void advance_step_kernel(
    int num_seqs, int32_t const* __restrict__ context_lens_ptr,
    int32_t const* __restrict__ seq_lens_ptr) {}

void advance_step(torch::Tensor& input_tokens,     // dtype: long
                  torch::Tensor& input_positions,  // dtype: long

                  torch::Tensor& context_lens,  // dtype: int
                  torch::Tensor& seq_lens       // dtype: int
) {
  int dev = input_tokens.get_device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev);

  int blocks = 1;  // TODO: FIX

  advance_step_kernel<<<blocks, max_threads, 0, stream>>>(
      1, reinterpret_cast<int32_t const*>(context_lens.data_ptr()),
      reinterpret_cast<int32_t const*>(seq_lens.data_ptr()));
}

}  // namespace prepare_inputs

void advance_step(torch::Tensor& input_tokens,     // dtype: long
                  torch::Tensor& input_positions,  // dtype: long

                  torch::Tensor& context_lens,  // dtype: int
                  torch::Tensor& seq_lens       // dtype: int
) {
  prepare_inputs::advance_step(input_tokens, input_positions, context_lens,
                               seq_lens);
}