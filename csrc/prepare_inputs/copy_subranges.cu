#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace vllm {
__global__ void copy_subranges_kernel(const int* __restrict__ matrix_src,
                                      const int* __restrict__ matrix_diff,
                                      int* __restrict__ matrix_tgt, int64_t M) {
  int row_id = blockIdx.x;
  int row_offset = row_id * M;

  int start = matrix_diff[row_id * 2];
  int length = matrix_diff[row_id * 2 + 1];
  int end = start + length;
  int thread_idx = threadIdx.x;
  for (int i = start + thread_idx; i < end; i += blockDim.x) {
    int idx = row_offset + i;
    matrix_tgt[idx] = matrix_src[idx];
  }
}
}  // namespace vllm

void copy_subranges(torch::Tensor& matrix_src, torch::Tensor& matrix_diff,
                    torch::Tensor& matrix_tgt, int64_t n) {
  // NOTE(woosuk): Here, we skip most of the error checking to minimize the
  // CPU overheads. We assume that the caller will pass the correct inputs.

  // Check tensor properties
  // TORCH_CHECK(matrix_src.is_cuda(), "matrix_src must be a CUDA tensor");
  // TORCH_CHECK(matrix_diff.is_cuda(), "matrix_diff must be a CUDA tensor");
  // TORCH_CHECK(matrix_tgt.is_cuda(), "matrix_tgt must be a CUDA tensor");
  // TORCH_CHECK(matrix_src.is_contiguous(), "matrix_src must be contiguous");
  // TORCH_CHECK(matrix_diff.is_contiguous(), "matrix_diff must be contiguous");
  // TORCH_CHECK(matrix_tgt.is_contiguous(), "matrix_tgt must be contiguous");

  auto src_sizes = matrix_src.sizes();
  auto diff_sizes = matrix_diff.sizes();
  auto tgt_sizes = matrix_tgt.sizes();

  // TORCH_CHECK(src_sizes.size() == 2, "matrix_src must be 2D");
  // TORCH_CHECK(diff_sizes.size() == 2, "matrix_diff must be 2D");
  // TORCH_CHECK(tgt_sizes.size() == 2, "matrix_tgt must be 2D");

  int64_t N = src_sizes[0];
  int64_t M = src_sizes[1];

  // TORCH_CHECK(diff_sizes[0] == N, "matrix_diff first dim must match N");
  // TORCH_CHECK(diff_sizes[1] == 2, "matrix_diff second dim must be 2");
  // TORCH_CHECK(tgt_sizes[0] == N && tgt_sizes[1] == M,
  //             "matrix_tgt must have same shape as matrix_src");

  // TORCH_CHECK(n <= N, "n must be <= N");

  const int* d_matrix_src = matrix_src.data_ptr<int>();
  const int* d_matrix_diff = matrix_diff.data_ptr<int>();
  int* d_matrix_tgt = matrix_tgt.data_ptr<int>();

  // One thread block per row.
  int blocks = n;
  int threads;
  if (blocks < 128) {
    threads = 1024;
  } else if (blocks < 256) {
    threads = 512;
  } else if (blocks < 512) {
    threads = 256;
  } else {
    threads = 128;
  }
  const at::cuda::OptionalCUDAGuard device_guard(device_of(matrix_tgt));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vllm::copy_subranges_kernel<<<blocks, threads, 0, stream>>>(
      d_matrix_src, d_matrix_diff, d_matrix_tgt, M);
}
