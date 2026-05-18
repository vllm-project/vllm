// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// CPU at::Tensor wrappers for Mamba decode-step kernels defined in
// mamba_kernels.hpp.

#include "cpu/mamba_kernels.hpp"

#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/util/Optional.h>

#include "cpu_types.hpp"

// ---------------------------------------------------------------------------
// causal_conv1d_update
// ---------------------------------------------------------------------------
at::Tensor causal_conv1d_update_cpu_impl(
    at::Tensor& x,  // modified in-place (re-typed to float32)
    at::Tensor& conv_state, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<std::string>& activation,
    const c10::optional<at::Tensor>& conv_state_indices,
    const c10::optional<at::Tensor>& query_start_loc, int64_t pad_slot_id) {
  
  bool do_silu = false;
  if (activation.has_value()) {
    const std::string& act = activation.value();
    do_silu = (act == "silu" || act == "swish");
  }

  // Causal conv still works in float32 for now (minimal overhead compared to SSM)
  at::Tensor x_f32 = x.to(at::kFloat).contiguous();
  at::Tensor state_f32 = conv_state.to(at::kFloat).contiguous();
  at::Tensor w_f32 = weight.to(at::kFloat).contiguous();
  at::Tensor bias_f32;
  if (bias.has_value() && bias.value().defined()) bias_f32 = bias.value().to(at::kFloat).contiguous();

  int64_t batch = x_f32.size(0);
  int64_t dim = x_f32.size(1);
  int64_t seqlen = (x_f32.dim() == 3) ? x_f32.size(2) : 1;
  int64_t width = w_f32.size(1);
  int64_t state_len = state_f32.size(2);

  at::Tensor out_f32 = at::empty_like(x_f32);

  const int32_t* cache_idx_ptr = nullptr;
  at::Tensor cache_idx_int;
  if (conv_state_indices.has_value()) {
    cache_idx_int = conv_state_indices.value().to(at::kInt).contiguous();
    cache_idx_ptr = cache_idx_int.data_ptr<int32_t>();
  }

  mamba_cpu::causal_conv1d_update_kernel(
      x_f32.data_ptr<float>(), state_f32.data_ptr<float>(),
      w_f32.data_ptr<float>(), bias_f32.defined() ? bias_f32.data_ptr<float>() : nullptr,
      out_f32.data_ptr<float>(), cache_idx_ptr, static_cast<int32_t>(pad_slot_id),
      batch, dim, seqlen, width, state_len, do_silu);

  conv_state.copy_(state_f32.to(conv_state.scalar_type()));
  return out_f32.to(x.scalar_type());
}

// ---------------------------------------------------------------------------
// selective_state_update
// ---------------------------------------------------------------------------
void selective_state_update_cpu_impl(
    at::Tensor& state,    // (nstates, nheads, dim, dstate)
    const at::Tensor& x,  // (N, nheads, dim)
    const at::Tensor& dt,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const c10::optional<at::Tensor>& D,
    const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& dt_bias,
    bool dt_softplus,
    const c10::optional<at::Tensor>& state_batch_indices,
    const c10::optional<at::Tensor>& dst_state_batch_indices,
    int64_t null_block_id,
    at::Tensor& out,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    const c10::optional<at::Tensor>& cu_seqlens
) {
  // Use state's dtype as the primary type to avoid expensive conversions
  // The kernel supports mixed types: state_t can be BFloat16 while input_t matches x
  at::ScalarType state_type = state.scalar_type();
  at::ScalarType input_type = x.scalar_type();
  
  // Only convert/contiguous if needed to minimize overhead
  auto ensure_type_and_contiguous = [input_type](const at::Tensor& t) -> at::Tensor {
    if (t.scalar_type() != input_type) {
      return t.to(input_type).contiguous();
    }
    return t.is_contiguous() ? t : t.contiguous();
  };
  
  at::Tensor dt_in = ensure_type_and_contiguous(dt);
  at::Tensor A_in = ensure_type_and_contiguous(A);
  at::Tensor B_in = ensure_type_and_contiguous(B);
  at::Tensor C_in = ensure_type_and_contiguous(C);
  
  at::Tensor D_in, z_in, dt_bias_in;
  if (D.has_value() && D.value().defined()) {
    D_in = ensure_type_and_contiguous(D.value());
  }
  if (z.has_value() && z.value().defined()) {
    z_in = ensure_type_and_contiguous(z.value());
  }
  if (dt_bias.has_value() && dt_bias.value().defined()) {
    dt_bias_in = ensure_type_and_contiguous(dt_bias.value());
  }

  int64_t nheads = state.size(1);
  int64_t dim = state.size(2);
  int64_t dstate = state.size(3);
  int64_t N = x.size(0);
  int64_t ngroups = B_in.size(1);

  // Strides
  int64_t stride_state_n = state.stride(0);
  int64_t stride_state_h = state.stride(1);
  int64_t stride_state_d = state.stride(2);
  int64_t stride_xdt_n = x.stride(0);
  int64_t stride_xdt_h = x.stride(1);
  int64_t stride_A_h = A_in.stride(0);
  int64_t stride_BC_n = B_in.stride(0);
  int64_t stride_BC_g = B_in.stride(1);
  int64_t stride_out_n = out.stride(0);
  int64_t stride_out_h = out.stride(1);
  int64_t stride_D_h = D_in.defined() ? D_in.stride(0) : 0;
  int64_t stride_dtbias_h = dt_bias_in.defined() ? dt_bias_in.stride(0) : 0;

  // Optional pointers - extract once
  auto get_int32_ptr = [](const c10::optional<at::Tensor>& opt) -> const int32_t* {
    return (opt.has_value() && opt.value().defined()) ? opt.value().data_ptr<int32_t>() : nullptr;
  };
  
  const int32_t* sbi_ptr = get_int32_ptr(state_batch_indices);
  const int32_t* dsbi_ptr = get_int32_ptr(dst_state_batch_indices);
  const int32_t* nat_ptr = get_int32_ptr(num_accepted_tokens);
  const int32_t* csl_ptr = get_int32_ptr(cu_seqlens);

  // Optimize output buffer: only use float32 if output type is not already float32
  // This avoids an extra copy when out is already float32
  bool need_out_conversion = (out.scalar_type() != at::kFloat);
  at::Tensor out_f32 = need_out_conversion ? at::empty_like(out, at::kFloat) : out;

  VLLM_DISPATCH_FLOATING_TYPES(state_type, "ssu_state", [&] {
    using state_t = scalar_t;
    VLLM_DISPATCH_FLOATING_TYPES(input_type, "ssu_input", [&] {
      using input_t = scalar_t;
      mamba_cpu::selective_state_update_kernel<state_t, input_t>(
          state.data_ptr<state_t>(), stride_state_n, stride_state_h, stride_state_d,
          x.data_ptr<input_t>(), dt_in.data_ptr<input_t>(), stride_xdt_n, stride_xdt_h,
          A_in.data_ptr<input_t>(), stride_A_h,
          B_in.data_ptr<input_t>(), C_in.data_ptr<input_t>(), stride_BC_n, stride_BC_g,
          D_in.defined() ? D_in.data_ptr<input_t>() : nullptr, stride_D_h,
          z_in.defined() ? z_in.data_ptr<input_t>() : nullptr,
          dt_bias_in.defined() ? dt_bias_in.data_ptr<input_t>() : nullptr, stride_dtbias_h,
          out_f32.data_ptr<float>(), stride_out_n, stride_out_h,
          sbi_ptr, dsbi_ptr, static_cast<int32_t>(null_block_id),
          nat_ptr, csl_ptr, N, nheads, ngroups, dim, dstate, dt_softplus);
    });
  });

  // Only copy back if we used a temporary buffer
  if (need_out_conversion) {
    out.copy_(out_f32);
  }
}
