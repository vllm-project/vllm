
#include "marlin.cuh"

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "libtorch_stable/torch_utils.h"

// for only non-zp format (like gptq)
__global__ void marlin_int4_fp8_preprocess_kernel_without_zp(
    // qweight: (size_k * size_n // 8,)
    const int32_t* __restrict__ qweight,
    // output: same shape with qweight
    int32_t* __restrict__ output) {
  int32_t val = qweight[blockIdx.x * 32 + threadIdx.x];
  int32_t new_val = 0;

#pragma unroll
  for (int32_t i = 0; i < 8; i++) {
    int32_t single_val = val & 0xF;
    single_val = single_val >= 8 ? single_val - 8 : 15 - single_val;
    new_val |= single_val << (i * 4);
    val >>= 4;
  }

  output[blockIdx.x * 32 + threadIdx.x] = new_val;
}

// for awq format only (with zp and with awq weight layout)
__global__ void marlin_int4_fp8_preprocess_kernel_awq(
    // AWQ qweight: (size_k, size_n // 8)
    const int32_t* __restrict__ qweight,
    // output: same shape with qweight
    int32_t* __restrict__ output,
    // AWQ zeros: (size_k // group_size, size_n // 8)
    const int32_t* __restrict__ qzeros, int32_t size_n, int32_t size_k,
    int32_t group_size) {
  // Thread mapping: threadIdx.x -> column dim (coalesced read within a row),
  // blockIdx.x -> row dim. Adjacent threads read consecutive int32 in the
  // same row (stride 1) instead of striding across rows (stride size_n/8).
  int col = blockIdx.y * 32 + threadIdx.x;
  if (col >= size_n / 8) return;
  (void)size_k;

  int32_t val = qweight[blockIdx.x * (size_n / 8) + col];
  int32_t zero = qzeros[blockIdx.x / group_size * (size_n / 8) + col];
  int32_t new_val = 0;

#pragma unroll
  for (int32_t i = 0; i < 8; i++) {
    int32_t single_val = val & 0xF;
    int32_t single_zero = zero & 0xF;

    single_val =
        single_val >= single_zero ? single_val - single_zero : 15 - single_val;
    new_val |= single_val << (i * 4);
    val >>= 4;
    zero >>= 4;
  }

  output[blockIdx.x * (size_n / 8) + col] = new_val;
}

torch::stable::Tensor marlin_int4_fp8_preprocess(
    torch::stable::Tensor& qweight,
    std::optional<torch::stable::Tensor> qzeros_or_none, bool inplace) {
  STD_TORCH_CHECK(qweight.is_cuda(), "qweight is not on GPU");
  STD_TORCH_CHECK(qweight.scalar_type() == torch::headeronly::ScalarType::Int,
                  "qweight.dtype != torch.int32");

  const int32_t device_index = qweight.get_device_index();
  torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const cudaStream_t stream = get_current_cuda_stream(device_index);

  torch::stable::Tensor output =
      inplace ? qweight : torch::stable::empty_like(qweight);

  if (!qzeros_or_none.has_value()) {
    STD_TORCH_CHECK(qweight.numel() * 8 % 256 == 0,
                    "qweight.numel() * 8 % 256 != 0");

    int blocks = qweight.numel() * 8 / 256;
    marlin_int4_fp8_preprocess_kernel_without_zp<<<blocks, 32, 0, stream>>>(
        reinterpret_cast<const int32_t*>(qweight.const_data_ptr()),
        reinterpret_cast<int32_t*>(output.mutable_data_ptr()));
  } else {
    int32_t size_k = qweight.size(0);
    int32_t size_n = qweight.size(1) * 8;
    torch::stable::Tensor qzeros = qzeros_or_none.value();

    STD_TORCH_CHECK(size_k % 32 == 0, "size_k % 32 != 0");
    STD_TORCH_CHECK(qzeros.is_cuda(), "qzeros is not on GPU");
    STD_TORCH_CHECK(qzeros.scalar_type() == torch::headeronly::ScalarType::Int,
                    "qweight.dtype != torch.int32");
    STD_TORCH_CHECK(qzeros.get_device_index() == device_index,
                    "qzeros is not on the same device with qweight");

    int32_t group_size = qweight.size(0) / qzeros.size(0);
    STD_TORCH_CHECK(qweight.size(1) == qzeros.size(1),
                    "qweight.size(1) != qzeros.size(1)");
    STD_TORCH_CHECK(qweight.size(0) % qzeros.size(0) == 0,
                    "qweight.size(0) % qzeros.size(0) != 0");
    STD_TORCH_CHECK(group_size % 8 == 0, "group_size % 8 != 0");

    dim3 blocks(size_k, (size_n / 8 + 31) / 32);
    marlin_int4_fp8_preprocess_kernel_awq<<<blocks, 32, 0, stream>>>(
        reinterpret_cast<const int32_t*>(qweight.const_data_ptr()),
        reinterpret_cast<int32_t*>(output.mutable_data_ptr()),
        reinterpret_cast<const int32_t*>(qzeros.const_data_ptr()), size_n,
        size_k, group_size);
  }

  return output;
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("marlin_int4_fp8_preprocess", TORCH_BOX(&marlin_int4_fp8_preprocess));
}
