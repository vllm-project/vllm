// Entry point of the int4 skinny GEMM, forwarding to the translation unit that
// instantiates the kernels for this N_in. The per-lane units define the same
// function entry under a suffixed name (see skinny_gemms_int4_n<n>.cu), so the
// device code of each N_in is generated in its own unit instead of all five
// landing on one core.
#include <torch/all.h>

#include <optional>
#include <stdexcept>

torch::Tensor wvSplitK_int4_g_n1(
    const at::Tensor& in_a, const at::Tensor& in_b, const at::Tensor& in_scale,
    const std::optional<at::Tensor>& in_zero_points,
    const std::optional<at::Tensor>& in_bias, const int64_t CuCount,
    const int64_t group_size);
torch::Tensor wvSplitK_int4_g_n2(
    const at::Tensor& in_a, const at::Tensor& in_b, const at::Tensor& in_scale,
    const std::optional<at::Tensor>& in_zero_points,
    const std::optional<at::Tensor>& in_bias, const int64_t CuCount,
    const int64_t group_size);
torch::Tensor wvSplitK_int4_g_n3(
    const at::Tensor& in_a, const at::Tensor& in_b, const at::Tensor& in_scale,
    const std::optional<at::Tensor>& in_zero_points,
    const std::optional<at::Tensor>& in_bias, const int64_t CuCount,
    const int64_t group_size);
torch::Tensor wvSplitK_int4_g_n4(
    const at::Tensor& in_a, const at::Tensor& in_b, const at::Tensor& in_scale,
    const std::optional<at::Tensor>& in_zero_points,
    const std::optional<at::Tensor>& in_bias, const int64_t CuCount,
    const int64_t group_size);
torch::Tensor wvSplitK_int4_g_n5(
    const at::Tensor& in_a, const at::Tensor& in_b, const at::Tensor& in_scale,
    const std::optional<at::Tensor>& in_zero_points,
    const std::optional<at::Tensor>& in_bias, const int64_t CuCount,
    const int64_t group_size);

torch::Tensor wvSplitK_int4_g(const at::Tensor& in_a, const at::Tensor& in_b,
                              const at::Tensor& in_scale,
                              const std::optional<at::Tensor>& in_zero_points,
                              const std::optional<at::Tensor>& in_bias,
                              const int64_t CuCount, const int64_t group_size) {
  switch (in_b.size(0)) {
    case 1:
      return wvSplitK_int4_g_n1(in_a, in_b, in_scale, in_zero_points, in_bias,
                                CuCount, group_size);
    case 2:
      return wvSplitK_int4_g_n2(in_a, in_b, in_scale, in_zero_points, in_bias,
                                CuCount, group_size);
    case 3:
      return wvSplitK_int4_g_n3(in_a, in_b, in_scale, in_zero_points, in_bias,
                                CuCount, group_size);
    case 4:
      return wvSplitK_int4_g_n4(in_a, in_b, in_scale, in_zero_points, in_bias,
                                CuCount, group_size);
    case 5:
      return wvSplitK_int4_g_n5(in_a, in_b, in_scale, in_zero_points, in_bias,
                                CuCount, group_size);
    default:
      throw std::runtime_error("Unsupported N value: " +
                               std::to_string(in_b.size(0)));
  }
}
