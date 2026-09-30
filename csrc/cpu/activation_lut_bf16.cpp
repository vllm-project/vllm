#include "cpu_types.hpp"

#include <array>
#include <cstdint>
#include <mutex>
#include <string>

#include <ATen/ops/empty.h>
#include <ATen/ops/gelu.h>
#include <c10/util/BFloat16.h>

constexpr uint32_t ActivationLutSize = 1u << 16;

at::Tensor gelu_reference(const at::Tensor& x) { return at::gelu(x, "none"); }

void maybe_init_activation_lut_bf16(
    uint16_t* lut, std::once_flag& once,
    at::Tensor (*activation)(const at::Tensor&)) {
  std::call_once(once, [&]() {
    auto lut_input =
        at::empty({static_cast<int64_t>(ActivationLutSize)},
                  at::TensorOptions().device(at::kCPU).dtype(at::kFloat));
    auto* lut_input_ptr = lut_input.data_ptr<float>();
#pragma omp parallel for
    for (uint32_t i = 0; i < ActivationLutSize; ++i) {
      lut_input_ptr[i] = c10::detail::f32_from_bits(static_cast<uint16_t>(i));
    }

    auto lut_output = activation(lut_input);
    const auto* lut_output_ptr = lut_output.data_ptr<float>();
#pragma omp parallel for
    for (uint32_t i = 0; i < ActivationLutSize; ++i) {
      lut[i] = c10::detail::round_to_nearest_even(lut_output_ptr[i]);
    }
  });
}

void activation_lut_bf16(torch::Tensor& out, torch::Tensor& input,
                         const uint16_t* lut, const char* op_name) {
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, op_name,
              ": input must be bfloat16");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, op_name,
              ": out must be bfloat16");
  TORCH_CHECK(input.is_contiguous(), op_name, ": input must be contiguous");
  TORCH_CHECK(out.is_contiguous(), op_name, ": out must be contiguous");

  const auto* src =
      reinterpret_cast<const uint16_t*>(input.data_ptr<at::BFloat16>());
  auto* dst = reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>());
  const int64_t n = input.numel();

  CPU_KERNEL_GUARD_IN(activation_lut_bf16_impl)
#pragma omp parallel for
  for (int64_t i = 0; i < n; ++i) {
    dst[i] = lut[src[i]];
  }
  CPU_KERNEL_GUARD_OUT(activation_lut_bf16_impl)
}

void activation_lut_bf16(torch::Tensor& out, torch::Tensor& input,
                         const std::string& activation) {
  if (activation == "gelu") {
    static std::array<uint16_t, ActivationLutSize> lut{};
    static std::once_flag once;
    maybe_init_activation_lut_bf16(lut.data(), once, gelu_reference);
    activation_lut_bf16(out, input, lut.data(), "gelu_lut");
    return;
  }

  TORCH_CHECK(false, "Unsupported activation: ", activation);
}
