#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace vllm {
__global__ void append_kernel(const int* __restrict__ row_indices,
                              const int* __restrict__ cu_num_appends,
                              const int* __restrict__ block_ids,
                              int* __restrict__ block_table,
                              int max_num_blocks_per_row) {
  int bid = blockIdx.x;
  int tgt_row = row_indices[2 * bid];
  int tgt_offset = row_indices[2 * bid + 1];

  int start = cu_num_appends[bid];
  int end = cu_num_appends[bid + 1];
  int length = end - start;
  int tid = threadIdx.x;
  int64_t offset = tgt_row * max_num_blocks_per_row + tgt_offset;
  for (int i = tid; i < length; i += blockDim.x) {
    block_table[offset + i] = block_ids[start + i];
  }
}

__global__ void move_kernel(const int* __restrict__ src_dst_n,
                            int* __restrict__ block_table,
                            int max_num_blocks_per_row) {
  int bid = blockIdx.x;
  int src_row = src_dst_n[3 * bid];
  int tgt_row = src_dst_n[3 * bid + 1];
  int num_blocks = src_dst_n[3 * bid + 2];

  int tid = threadIdx.x;
  for (int i = tid; i < num_blocks; i += blockDim.x) {
    block_table[tgt_row * max_num_blocks_per_row + i] =
        block_table[src_row * max_num_blocks_per_row + i];
  }
}
}  // namespace vllm

void block_table_appends(
    torch::Tensor& append_row_indices,
    torch::Tensor& append_row_indices_cpu,
    torch::Tensor& append_cumsums,
    torch::Tensor& append_cumsums_cpu,
    torch::Tensor& append_block_ids,
    torch::Tensor& append_block_ids_cpu,
    torch::Tensor& block_table,
    int64_t num_appends,
    int64_t total_num_append_blocks) {
  int* append_row_indices_ptr = append_row_indices.data_ptr<int>();
  const int* append_row_indices_cpu_ptr = append_row_indices_cpu.data_ptr<int>();
  int* append_cumsums_ptr = append_cumsums.data_ptr<int>();
  const int* append_cumsums_cpu_ptr = append_cumsums_cpu.data_ptr<int>();
  int* append_block_ids_ptr = append_block_ids.data_ptr<int>();
  const int* append_block_ids_cpu_ptr = append_block_ids_cpu.data_ptr<int>();
  int* block_table_ptr = block_table.data_ptr<int>();

  const at::cuda::OptionalCUDAGuard device_guard(device_of(block_table));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cudaMemcpyAsync(append_row_indices_ptr, append_row_indices_cpu_ptr,
                  num_appends * 2 * sizeof(int), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(append_cumsums_ptr, append_cumsums_cpu_ptr,
                  (num_appends + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(append_block_ids_ptr, append_block_ids_cpu_ptr,
                  total_num_append_blocks * sizeof(int), cudaMemcpyHostToDevice, stream);

  int64_t max_num_blocks_per_row = block_table.size(1);
  vllm::append_kernel<<<num_appends, 1024, 0, stream>>>(
      append_row_indices_ptr, append_cumsums_ptr, append_block_ids_ptr,
      block_table_ptr, max_num_blocks_per_row);
}

void block_table_moves(
    torch::Tensor& src_dst_n,
    torch::Tensor& src_dst_n_cpu,
    torch::Tensor& block_table,
    int64_t num_moves) {
  int* src_dst_n_ptr = src_dst_n.data_ptr<int>();
  const int* src_dst_n_cpu_ptr = src_dst_n_cpu.data_ptr<int>();
  int* block_table_ptr = block_table.data_ptr<int>();

  const at::cuda::OptionalCUDAGuard device_guard(device_of(block_table));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cudaMemcpyAsync(src_dst_n_ptr, src_dst_n_cpu_ptr,
                  num_moves * 3 * sizeof(int), cudaMemcpyHostToDevice, stream);

  int64_t max_num_blocks_per_row = block_table.size(1);
  vllm::move_kernel<<<num_moves, 1024, 0, stream>>>(
      src_dst_n_ptr, block_table_ptr, max_num_blocks_per_row);
}
