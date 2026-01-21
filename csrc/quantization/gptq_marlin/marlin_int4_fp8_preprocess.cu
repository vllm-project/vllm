

#include "marlin.cuh"

#include "core/registration.h"

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
  int32_t val =
      qweight[(blockIdx.x * 32 + threadIdx.x) * size_n / 8 + blockIdx.y];
  int32_t zero =
      qzeros[(blockIdx.x * 32 + threadIdx.x) / group_size * size_n / 8 +
             blockIdx.y];
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

  output[(blockIdx.x * 32 + threadIdx.x) * size_n / 8 + blockIdx.y] = new_val;
}

torch::Tensor marlin_int4_fp8_preprocess(
    torch::Tensor& qweight, std::optional<torch::Tensor> qzeros_or_none,
    bool inplace) {
  TORCH_CHECK(qweight.device().is_cuda(), "qweight is not on GPU");
  TORCH_CHECK(qweight.scalar_type() == at::ScalarType::Int,
              "qweight.dtype != torch.int32");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));

  torch::Tensor output = inplace ? qweight : torch::empty_like(qweight);

  if (!qzeros_or_none.has_value()) {
    TORCH_CHECK(qweight.numel() * 8 % 256 == 0,
                "qweight.numel() * 8 % 256 != 0");

    int blocks = qweight.numel() * 8 / 256;
    marlin_int4_fp8_preprocess_kernel_without_zp<<<blocks, 32>>>(
        (const int32_t*)qweight.data_ptr(), (int32_t*)output.data_ptr());
  } else {
    int32_t size_k = qweight.size(0);
    int32_t size_n = qweight.size(1) * 8;
    torch::Tensor qzeros = qzeros_or_none.value();

    TORCH_CHECK(size_k % 32 == 0, "size_k % 32 != 0");
    TORCH_CHECK(qzeros.device().is_cuda(), "qzeros is not on GPU");
    TORCH_CHECK(qzeros.scalar_type() == at::ScalarType::Int,
                "qweight.dtype != torch.int32");
    TORCH_CHECK(device_of(qweight) == device_of(qzeros),
                "qzeros is not on the same device with qweight");

    int32_t group_size = qweight.size(0) / qzeros.size(0);
    TORCH_CHECK(qweight.size(1) == qzeros.size(1),
                "qweight.size(1) != qzeros.size(1)");
    TORCH_CHECK(qweight.size(0) % qzeros.size(0) == 0,
                "qweight.size(0) % qzeros.size(0) != 0");
    TORCH_CHECK(group_size % 8 == 0, "group_size % 8 != 0");

    dim3 blocks(size_k / 32, size_n / 8);
    marlin_int4_fp8_preprocess_kernel_awq<<<blocks, 32>>>(
        (const int32_t*)qweight.data_ptr(), (int32_t*)output.data_ptr(),
        (const int32_t*)qzeros.data_ptr(), size_n, size_k, group_size);
  }

  return output;
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("marlin_int4_fp8_preprocess", &marlin_int4_fp8_preprocess);
}
