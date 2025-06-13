#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>

static constexpr int default_threads = 256;
static constexpr int div_ceil(int a, int b) { return (a + b - 1) / b; }

// For a given "a" of size [M,K] performs a permutation of the K columns based
// on the given "perm" indices.
// Currently only supports 16bit types (since we permute half types)
__global__ void permute_cols_kernel(int4 const* __restrict__ a_int4_ptr,
                                    int const* __restrict__ perm_int_ptr,
                                    int4* __restrict__ out_int4_ptr, int size_m,
                                    int size_k, int block_rows) {
  int start_row = block_rows * blockIdx.x;
  int finish_row = start_row + block_rows;
  if (finish_row > size_m) {
    finish_row = size_m;
  }
  int cur_block_rows = std::max(finish_row - start_row, 0);

  int row_stride = size_k * sizeof(half) / 16;

  auto permute_row = [&](int row) {
    int iters = size_k / default_threads;
    int rest = size_k % default_threads;

    int offset = row * row_stride;

    half const* a_row_half = reinterpret_cast<half const*>(a_int4_ptr + offset);
    half* out_half = reinterpret_cast<half*>(out_int4_ptr + offset);

    int base_k = 0;

    for (int i = 0; i < iters; i++) {
      int cur_k = base_k + threadIdx.x;
      int src_pos = perm_int_ptr[cur_k];

      out_half[cur_k] = a_row_half[src_pos];

      base_k += default_threads;
    }

    if (rest) {
      if (threadIdx.x < rest) {
        int cur_k = base_k + threadIdx.x;
        int src_pos = perm_int_ptr[cur_k];

        out_half[cur_k] = a_row_half[src_pos];
      }
    }
  };

  for (int i = 0; i < cur_block_rows; i++) {
    int cur_row = start_row + i;
    if (cur_row < size_m) {
      permute_row(cur_row);
    }
  }
}

// More efficient version of A[..., perm]
//  taken from gptq_marlin.cu
torch::Tensor permute_cols(torch::Tensor const& A, torch::Tensor const& perm) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  auto dev = A.get_device();
  auto stream = at::cuda::getCurrentCUDAStream(dev);

  TORCH_CHECK(A.scalar_type() == at::kHalf || A.scalar_type() == at::kBFloat16,
              "Currently only 16bit types are supported");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(A.size(-1) % 8 == 0,
              "A columns must be a multiple of 8 (128bits)");
  auto A_2d = A.view({-1, A.size(-1)});

  torch::Tensor D = torch::empty_like(A);
  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  int block_rows = div_ceil(A_2d.size(0), sms);
  permute_cols_kernel<<<sms, default_threads, 0, stream>>>(
      reinterpret_cast<int4 const*>(A_2d.const_data_ptr()),
      perm.const_data_ptr<int>(), reinterpret_cast<int4*>(D.mutable_data_ptr()),
      A_2d.size(0), A_2d.size(1), block_rows);
  return D;
}