// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <optional>
#include <tuple>

std::tuple<at::Tensor, at::Tensor> chunk_gated_delta_rule_cpu(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const at::Tensor& g, const at::Tensor& beta,
    const at::Tensor& initial_state, bool output_final_state,
    const at::Tensor& cu_seqlens, bool head_first, bool use_qk_l2norm_in_kernel,
    double eps) {
  TORCH_CHECK(false,
              "chunk_gated_delta_rule_cpu is not supported on the RISC-V "
              "P550 scalar CPU validation path.");
}

at::Tensor fused_sigmoid_gating_delta_rule_update_cpu(
    const at::Tensor& A_log, const at::Tensor& dt_bias, const at::Tensor& q,
    const at::Tensor& k, const at::Tensor& v, const at::Tensor& a,
    const at::Tensor& b, at::Tensor& initial_state_source,
    const at::Tensor& initial_state_indices, const at::Tensor& cu_seqlens,
    bool use_qk_l2norm_in_kernel, double softplus_beta,
    double softplus_threshold) {
  TORCH_CHECK(false,
              "fused_sigmoid_gating_delta_rule_update_cpu is not supported "
              "on the RISC-V P550 scalar CPU validation path.");
}

std::tuple<at::Tensor, at::Tensor> fused_gdn_gating_cpu(
    const at::Tensor& A_log, const at::Tensor& a, const at::Tensor& b,
    const at::Tensor& dt_bias) {
  TORCH_CHECK(false,
              "fused_gdn_gating_cpu is not supported on the RISC-V P550 "
              "scalar CPU validation path.");
}
