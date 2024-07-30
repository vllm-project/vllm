/*
 * Modified by Neural Magic
 * Adapted from https://github.com/Vahe1994/AQLM
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <iostream>
#include <cstdlib>

namespace vllm {
namespace aqlm {

__global__ void Code1x16MatVec(
    const int4* __restrict__ A, const int4* __restrict__ B,
    int4* __restrict__ C, const int4* __restrict__ codebook, const int prob_m,
    const int prob_k,
    const int4 codebook_a_sizes,  // cumulative sizes of A spanning each
                                  // codebook, at most 3 long.
    const int codebook_stride     // as int4.
) {
  int a_gl_stride = prob_k / 8 / 8;
  int a_gl_rd = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  bool pred = a_gl_rd < prob_m;

  if (pred) {
    // advance to the correct codebook, this easy because we only multiply one
    // column of the codebook.
    auto codebook_size = &codebook_a_sizes.x;
    while (a_gl_rd >= *codebook_size) {
      codebook += codebook_stride;
      ++codebook_size;
    }
  }

  int b_gl_rd = 0;
  int c_gl_wr = a_gl_rd;
  a_gl_rd = a_gl_stride * a_gl_rd + threadIdx.x % 32;
  int a_gl_end = a_gl_rd + a_gl_stride - threadIdx.x % 32;

  __shared__ int4 sh_b[32 * 9];
  float res = 0;

  int iters = (prob_k / 8 + 8 * 32 - 1) / (8 * 32);
  while (iters--) {
    // We pad shared memory to avoid bank conflicts during reads
    __syncthreads();
    for (int i = threadIdx.x; i < 32 * 8; i += blockDim.x) {
      if (b_gl_rd + i < prob_k / 8) sh_b[9 * (i / 8) + i % 8] = B[b_gl_rd + i];
    }
    __syncthreads();
    b_gl_rd += 32 * 8;

    int b_sh_rd = 9 * (threadIdx.x % 32);
    if (pred && a_gl_rd < a_gl_end) {
      const uint16_t* enc = reinterpret_cast<const uint16_t*>(&A[a_gl_rd]);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        uint32_t dec[4];
        // We bypass the L1 cache to avoid massive amounts of memory streaming
        // that doesn't actually help us; this brings > 2x speedup.
        asm volatile("ld.cg.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(dec[0]), "=r"(dec[1]), "=r"(dec[2]), "=r"(dec[3])
                     : "l"((void*)&codebook[enc[i]]));
        half2* a = reinterpret_cast<half2*>(&dec);
        half2* b = reinterpret_cast<half2*>(&sh_b[b_sh_rd]);
        half2 res2 = {};
#pragma unroll
        for (int j = 0; j < 4; j++) res2 = __hfma2(a[j], b[j], res2);
        res += __half2float(res2.x) + __half2float(res2.y);
        b_sh_rd++;
      }
      a_gl_rd += 32;
    }
  }

  if (pred) {
#pragma unroll
    for (int i = 16; i > 0; i /= 2) res += __shfl_down_sync(0xffffffff, res, i);
    if (threadIdx.x % 32 == 0)
      reinterpret_cast<__half*>(C)[c_gl_wr] = __float2half(res);
  }
}

__global__ void Code2x8MatVec(
    const int4* __restrict__ A, const int4* __restrict__ B,
    int4* __restrict__ C, const int4* __restrict__ codebook, int prob_m,
    int prob_k,
    const int4 codebook_a_sizes,  // cumulative sizes of A spanning each
                                  // codebook, at most 3 long.
    const int codebook_stride     // as int4.

) {
  int a_gl_stride = prob_k / 8 / 8;
  int a_gl_rd = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  bool pred = a_gl_rd < prob_m;

  if (pred) {
    // advance to the correct codebook, this easy because we only multiply one
    // column of the codebook.
    auto codebook_size = &codebook_a_sizes.x;
    while (a_gl_rd >= *codebook_size) {
      codebook += codebook_stride;
      ++codebook_size;
    }
  }

  int b_gl_rd = 0;
  int c_gl_wr = a_gl_rd;
  a_gl_rd = a_gl_stride * a_gl_rd + threadIdx.x % 32;
  int a_gl_end = a_gl_rd + a_gl_stride - threadIdx.x % 32;
  int lane = threadIdx.x % 8;

  extern __shared__ int4 sh[];
  int4* sh_b = sh;
  int4* sh_code = sh_b + 32 * 9;
  int4* sh_code0 = sh_code;
  int4* sh_code1 = sh_code + 256 * 8;

  for (int i = threadIdx.x; i < 2 * 256; i += blockDim.x) {
    int4 dec = codebook[i];
#pragma unroll
    for (int j = 0; j < 8; j++) sh_code[8 * i + (j + lane) % 8] = dec;
  }
  __syncthreads();

  float res = 0;

  int iters = (prob_k / 8 + 8 * 32 - 1) / (8 * 32);
  while (iters--) {
    // We pad shared memory to avoid bank conflicts during reads
    __syncthreads();
    for (int i = threadIdx.x; i < 32 * 8; i += blockDim.x) {
      if (b_gl_rd + i < prob_k / 8) sh_b[9 * (i / 8) + i % 8] = B[b_gl_rd + i];
    }
    __syncthreads();
    b_gl_rd += 32 * 8;

    int b_sh_rd = 9 * (threadIdx.x % 32);
    if (pred && a_gl_rd < a_gl_end) {
      const uint8_t* enc = reinterpret_cast<const uint8_t*>(&A[a_gl_rd]);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        half2* a0 =
            reinterpret_cast<half2*>(&sh_code0[8 * enc[2 * i + 0] + lane]);
        half2* a1 =
            reinterpret_cast<half2*>(&sh_code1[8 * enc[2 * i + 1] + lane]);
        half2* b = reinterpret_cast<half2*>(&sh_b[b_sh_rd]);
        half2 res2 = {};
#pragma unroll
        for (int j = 0; j < 4; j++)
          res2 = __hfma2(__hadd2(a0[j], a1[j]), b[j], res2);
        res += __half2float(res2.x) + __half2float(res2.y);
        b_sh_rd++;
      }
      a_gl_rd += 32;
    }
  }

  if (pred) {
#pragma unroll
    for (int i = 16; i > 0; i /= 2) res += __shfl_down_sync(0xffffffff, res, i);
    if (threadIdx.x % 32 == 0)
      reinterpret_cast<__half*>(C)[c_gl_wr] = __float2half(res);
  }
}

__global__ void Code1x16Dequant(
    const int4* __restrict__ A, int4* __restrict__ C,
    const int4* __restrict__ codebook, int prob_m, int prob_k,
    const int4 codebook_a_sizes,  // cumulative sizes of A spanning each
                                  // codebook, at most 3 long, sums to m.
    const int codebook_stride     // as int4
) {
  int a_gl_stride = prob_k / 8 / 8;
  int a_gl_rd = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  bool pred = a_gl_rd < prob_m;

  if (pred) {
    // advance to the correct codebook, this easy because we only multiply one
    // column of the codebook.
    auto codebook_size = &codebook_a_sizes.x;
    while (a_gl_rd >= *codebook_size) {
      codebook += codebook_stride;
      ++codebook_size;
    }
  }

  a_gl_rd = a_gl_stride * a_gl_rd + threadIdx.x % 32;
  int a_gl_end = a_gl_rd + a_gl_stride - threadIdx.x % 32;

  int c_gl_stride = prob_k / 8;
  int c_gl_wr = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  c_gl_wr = c_gl_stride * c_gl_wr + (threadIdx.x % 32) * 8;

  int iters = (prob_k / 8 - 1) / (8 * 32) + 1;
  while (iters--) {
    if (pred && a_gl_rd < a_gl_end) {
      const uint16_t* enc = reinterpret_cast<const uint16_t*>(&A[a_gl_rd]);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        int4 chunk;
        auto dec = reinterpret_cast<uint32_t*>(&chunk);
        // We bypass the L1 cache to avoid massive amounts of memory streaming
        // that doesn't actually help us; this brings > 2x speedup.
        asm volatile("ld.cg.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(dec[0]), "=r"(dec[1]), "=r"(dec[2]), "=r"(dec[3])
                     : "l"((void*)&codebook[enc[i]]));

        C[a_gl_rd * 8 + i] = chunk;
      }
    }
    a_gl_rd += 32;
  }
}

__global__ void Code2x8Dequant(
    const int4* __restrict__ A, int4* __restrict__ C,
    const int4* __restrict__ codebook, int prob_m, int prob_k,
    const int4
        codebook_a_sizes,  // cumulative sizes of A spanning each codebook, at
                           // most 3 long, corresponds to cols.
    const int codebook_stride  // as int4
) {
  int a_gl_stride = prob_k / 8 / 8;
  int a_gl_rd = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  bool pred = a_gl_rd < prob_m;

  if (pred) {
    // advance to the correct codebook, this easy because we only multiply one
    // column of the codebook.
    auto codebook_size = &codebook_a_sizes.x;
    while (a_gl_rd >= *codebook_size) {
      codebook += codebook_stride;
      ++codebook_size;
    }
  }

  a_gl_rd = a_gl_stride * a_gl_rd + threadIdx.x % 32;
  int a_gl_end = a_gl_rd + a_gl_stride - threadIdx.x % 32;
  int lane = threadIdx.x % 8;

  int c_gl_stride = prob_k / 8;
  int c_gl_wr = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  c_gl_wr = c_gl_stride * c_gl_wr + (threadIdx.x % 32) * 8;

  extern __shared__ int4 sh[];
  int4* sh_code = sh;
  int4* sh_code0 = sh_code;
  int4* sh_code1 = sh_code + 256 * 8;

  for (int i = threadIdx.x; i < 2 * 256; i += blockDim.x) {
    int4 dec = codebook[i];
#pragma unroll
    for (int j = 0; j < 8; j++) sh_code[8 * i + (j + lane) % 8] = dec;
  }
  __syncthreads();

  int iters = (prob_k / 8 - 1) / (8 * 32) + 1;
  while (iters--) {
    if (pred && a_gl_rd < a_gl_end) {
      const uint8_t* enc = reinterpret_cast<const uint8_t*>(&A[a_gl_rd]);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        int4 chunk;
        half2* a0 =
            reinterpret_cast<half2*>(&sh_code0[8 * enc[2 * i + 0] + lane]);
        half2* a1 =
            reinterpret_cast<half2*>(&sh_code1[8 * enc[2 * i + 1] + lane]);
#pragma unroll
        for (int j = 0; j < 4; j++)
          reinterpret_cast<half2*>(&chunk)[j] = __hadd2(a0[j], a1[j]);
        C[a_gl_rd * 8 + i] = chunk;
      }
    }
    a_gl_rd += 32;
  }
}

inline int ceildiv(int a, int b) { return (a + b - 1) / b; }

const int THREAD_M = 16;

void code1x16_matvec_cuda(const void* __restrict__ A,
                          const void* __restrict__ B, void* __restrict__ C,
                          const void* __restrict__ codebook, int prob_m,
                          int prob_k, const int4 codebook_a_sizes,
                          const int codebook_stride) {
  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);

  int blocks = ceildiv(prob_m, thread_m);
  int threads = 32 * thread_m;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  Code1x16MatVec<<<blocks, threads, 16 * 32 * 9, stream>>>(
      (const int4*)A, (const int4*)B, (int4*)C, (const int4*)codebook, prob_m,
      prob_k, codebook_a_sizes, codebook_stride);
}

void code2x8_matvec_cuda(const void* __restrict__ A, const void* __restrict__ B,
                         void* __restrict__ C,
                         const void* __restrict__ codebook, int prob_m,
                         int prob_k, const int4 codebook_a_sizes,
                         const int codebook_stride) {
  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);

  int blocks = ceildiv(prob_m, thread_m);
  int threads = 32 * thread_m;
  int shared = 16 * (2 * 256 * 8 + 32 * 9);
  cudaFuncSetAttribute(Code2x8MatVec,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  Code2x8MatVec<<<blocks, threads, shared, stream>>>(
      (const int4*)A, (const int4*)B, (int4*)C, (const int4*)codebook, prob_m,
      prob_k, codebook_a_sizes, codebook_stride);
}

void code1x16_dequant_cuda(
    const void* __restrict__ A, void* __restrict__ C,
    const void* __restrict__ codebook, int prob_m, int prob_k,
    const int4 codebook_a_sizes,  // cumulative sizes of A spanning each
                                  // codebook, at most 3 long.
    const int codebook_stride     // as int4.
) {
  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);

  int blocks = ceildiv(prob_m, thread_m);
  int threads = 32 * thread_m;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  Code1x16Dequant<<<blocks, threads, 0, stream>>>(
      (const int4*)A, (int4*)C, (const int4*)codebook, prob_m, prob_k,
      codebook_a_sizes,  // cumulative sizes of A spanning each codebook, at
                         // most 3 long.
      codebook_stride    // as int4.
  );
}

// Dequantizes the code and codebook into weights.
void code2x8_dequant_cuda(
    const void* __restrict__ A, void* __restrict__ C,
    const void* __restrict__ codebook, int prob_m, int prob_k,
    const int4
        codebook_a_sizes,  // cumulative sizes of A spanning each codebook, at
                           // most 3 long, corresponds to cols.
    const int codebook_stride  // as int4
) {
  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);

  int blocks = ceildiv(prob_m, thread_m);
  int threads = 32 * thread_m;
  int shared = 16 * (2 * 256 * 8 + 32 * 9);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  cudaFuncSetAttribute(Code2x8Dequant,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  Code2x8Dequant<<<blocks, threads, shared, stream>>>(
      (const int4*)A, (int4*)C, (const int4*)codebook, prob_m, prob_k,
      codebook_a_sizes, codebook_stride);
}

int codebook_stride(const torch::Tensor& codebooks) {
  return codebooks.stride(0) * codebooks.element_size() / sizeof(int4);
}

void code1x16_matvec(
    const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C,
    const torch::Tensor& codebook,
    const int4 codebook_a_sizes  // cumulative sizes of A spanning each
                                 // codebook, at most 3 long.
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  int prob_m = C.size(0);
  int prob_k = B.size(0);

  code1x16_matvec_cuda(A.data_ptr(), B.data_ptr(), C.data_ptr(),
                       codebook.data_ptr(), prob_m, prob_k, codebook_a_sizes,
                       codebook_stride(codebook));
}

torch::Tensor code1x16_matmat(const torch::Tensor& input,
                              const torch::Tensor& codes,
                              const torch::Tensor& codebooks,
                              const torch::Tensor& scales,
                              const int4 codebook_a_sizes,
                              const std::optional<torch::Tensor>& bias) {
  auto input_sizes = input.sizes();
  auto out_features = codes.size(0) * codebooks.size(2);
  auto flat_input = input.reshape({-1, input.size(-1)});
  auto flat_output = torch::empty(
      {flat_input.size(0), out_features},
      torch::TensorOptions().dtype(input.dtype()).device(input.device()));

  for (int i = 0; i < flat_input.size(0); ++i) {
    auto input_vec = flat_input.index({i});
    auto output_vec = flat_output.index({i});
    code1x16_matvec(codes.squeeze(2), input_vec, output_vec, codebooks,
                    codebook_a_sizes);
  }
  flat_output *= scales.flatten().unsqueeze(0);

  if (bias.has_value()) {
    flat_output += bias->unsqueeze(0);
  }

  auto output_sizes = input_sizes.vec();
  output_sizes.pop_back();
  output_sizes.push_back(-1);
  auto output = flat_output.reshape(output_sizes);
  return output;
}

void code2x8_matvec(const torch::Tensor& A, const torch::Tensor& B,
                    torch::Tensor& C, const torch::Tensor& codebook,
                    const int4 codebook_a_sizes) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  int prob_m = C.size(0);
  int prob_k = B.size(0);
  code2x8_matvec_cuda(A.data_ptr(), B.data_ptr(), C.data_ptr(),
                      codebook.data_ptr(), prob_m, prob_k, codebook_a_sizes,
                      2 * codebook_stride(codebook));
}

torch::Tensor code2x8_matmat(const torch::Tensor& input,
                             const torch::Tensor& codes,
                             const torch::Tensor& codebooks,
                             const torch::Tensor& scales,
                             const int4 codebook_a_sizes,
                             const std::optional<torch::Tensor>& bias) {
  auto input_sizes = input.sizes();
  auto out_features = codes.size(0) * codebooks.size(2);
  auto flat_input = input.reshape({-1, input.size(-1)});
  auto flat_output = torch::empty(
      {flat_input.size(0), out_features},
      torch::TensorOptions().dtype(input.dtype()).device(input.device()));

  for (int i = 0; i < flat_input.size(0); ++i) {
    auto input_vec = flat_input.index({i});
    auto output_vec = flat_output.index({i});
    code2x8_matvec(codes.squeeze(2), input_vec, output_vec, codebooks,
                   codebook_a_sizes);
  }
  flat_output *= scales.flatten().unsqueeze(0);
  if (bias.has_value()) {
    flat_output += bias->unsqueeze(0);
  }

  auto output_sizes = input_sizes.vec();
  output_sizes.pop_back();
  output_sizes.push_back(-1);
  auto output = flat_output.reshape(output_sizes);
  return output;
}

// Accumulate the partition sizes.
int4 accumulate_sizes(const torch::Tensor& codebook_partition_sizes) {
  int4 cumulative_sizes;
  auto cumulative_size = &cumulative_sizes.x;
  int i = 0;
  int last = 0;
  assert(codebook_partition_sizes.size(0) <= 4);
  for (; i < codebook_partition_sizes.size(0); ++i, ++cumulative_size) {
    *cumulative_size = codebook_partition_sizes[i].item<int>() + last;
    last = *cumulative_size;
  }
  // fill in the rest with unreachable.
  for (; i < 4; ++i, ++cumulative_size) {
    *cumulative_size = last * 10;
  }
  return cumulative_sizes;
}

}  // namespace aqlm
}  // namespace vllm

torch::Tensor aqlm_gemm(const torch::Tensor& input, const torch::Tensor& codes,
                        const torch::Tensor& codebooks,
                        const torch::Tensor& scales,
                        const torch::Tensor& codebook_partition_sizes,
                        const std::optional<torch::Tensor>& bias) {
  int4 cumulative_sizes =
      vllm::aqlm::accumulate_sizes(codebook_partition_sizes);

  int const nbooks = codebooks.size(0) / codebook_partition_sizes.size(0);
  int const entries = codebooks.size(1);

  if (nbooks == 1 && entries == (1 << 16)) {
    return vllm::aqlm::code1x16_matmat(input, codes, codebooks, scales,
                                       cumulative_sizes, bias);
  }
  if (nbooks == 2 && entries == (1 << 8)) {
    return vllm::aqlm::code2x8_matmat(input, codes, codebooks, scales,
                                      cumulative_sizes, bias);
  }

  TORCH_CHECK(false, "AQLM with ", nbooks, " codebooks and ", entries,
              " entries is not currently supported.")
  return {};
}

torch::Tensor aqlm_dequant(const torch::Tensor& codes,
                           const torch::Tensor& codebooks,
                           const torch::Tensor& codebook_partition_sizes) {
  int4 cumulative_sizes =
      vllm::aqlm::accumulate_sizes(codebook_partition_sizes);

  int const nbooks = codebooks.size(0) / codebook_partition_sizes.size(0);
  int const entries = codebooks.size(1);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(codes));
  int rows = codes.size(1);
  int cols = codes.size(0);

  auto in_features = codes.size(1) * 8;
  auto out_features = codes.size(0);

  assert(out_features = codebook_partition_sizes.sum().item<int>());

  auto weights = torch::empty({out_features, in_features},
                              torch::TensorOptions()
                                  .dtype(codebooks.dtype())
                                  .device(codebooks.device()));

  if (nbooks == 1 && entries == (1 << 16)) {
    vllm::aqlm::code1x16_dequant_cuda(codes.data_ptr(), weights.data_ptr(),
                                      codebooks.data_ptr(), out_features,
                                      in_features, cumulative_sizes,
                                      vllm::aqlm::codebook_stride(codebooks));

    // if you wanted to flip to scaling the weights, (though it's 30%-ish slower
    // and not consistent with gemv implementation.) weights *=
    // scales.index({"...", 0, 0});

    return weights;
  }

  if (nbooks == 2 && entries == (1 << 8)) {
    vllm::aqlm::code2x8_dequant_cuda(codes.data_ptr(), weights.data_ptr(),
                                     codebooks.data_ptr(), out_features,
                                     in_features, cumulative_sizes,
                                     vllm::aqlm::codebook_stride(codebooks));

    // if you wanted to flip to scaling the weights, (though it's 30%-ish slower
    // and not consistent with gemv implementation) weights *=
    // scales.index({"...", 0, 0});

    return weights;
  }

  TORCH_CHECK(false, "AQLM with ", nbooks, " codebooks and ", entries,
              " entries is not currently supported.")
  return {};
}
