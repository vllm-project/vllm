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
    at::Tensor& x, at::Tensor& conv_state, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<std::string>& activation,
    const c10::optional<at::Tensor>& conv_state_indices,
    const c10::optional<at::Tensor>& query_start_loc, int64_t pad_slot_id) {
  bool do_silu = false;
  if (activation.has_value()) {
    const std::string& act = activation.value();
    do_silu = (act == "silu" || act == "swish");
  }

  at::ScalarType dtype = x.scalar_type();

  // Input x: contiguous in native dtype.
  at::Tensor x_c = x.is_contiguous() ? x : x.contiguous();

  // conv_state: NEVER copy the full paged tensor just for layout reasons.
  // If the dtype matches we work directly on conv_state (contiguous or not)
  // by extracting strides and passing them to the kernel.
  // Only a dtype-conversion copy is made when types differ (rare for BF16).
  bool state_type_ok = (conv_state.scalar_type() == dtype);
  at::Tensor state_c = state_type_ok ? conv_state : conv_state.to(dtype);
  // state_c and conv_state may be non-contiguous — that is intentional.

  // Weight: coerce to same dtype if needed (should match in practice)
  at::Tensor w_c =
      (weight.scalar_type() != dtype)
          ? weight.to(dtype).contiguous()
          : (weight.is_contiguous() ? weight : weight.contiguous());

  // Bias stays float32 (small scalar, used only for fp32 accumulation)
  at::Tensor bias_f32;
  if (bias.has_value() && bias.value().defined())
    bias_f32 = bias.value().to(at::kFloat).contiguous();

  int64_t batch = x_c.size(0);
  int64_t dim = x_c.size(1);
  int64_t seqlen = (x_c.dim() == 3) ? x_c.size(2) : 1;
  int64_t width = w_c.size(1);
  int64_t state_len = state_c.size(2);

  // Extract strides — works for contiguous AND non-contiguous (transposed)
  // state. stride(0): between cache slots (e.g. num_slots × dim × width-1 in
  // contiguous) stride(1): between conv channels (dim stride) stride(2):
  // between state elements (=1 when contiguous, =dim when transposed)
  int64_t stride_s_slot = state_c.stride(0);
  int64_t stride_s_dim = state_c.stride(1);
  int64_t stride_s_state = state_c.stride(2);

  at::Tensor out = x_c.clone();  // native dtype, no float32 alloc

  const int32_t* cache_idx_ptr = nullptr;
  at::Tensor cache_idx_int;
  if (conv_state_indices.has_value()) {
    cache_idx_int = conv_state_indices.value().to(at::kInt).contiguous();
    cache_idx_ptr = cache_idx_int.data_ptr<int32_t>();
  }

  VLLM_DISPATCH_FLOATING_TYPES(dtype, "causal_conv1d_update", [&] {
    mamba_cpu::causal_conv1d_update_kernel<scalar_t>(
        x_c.data_ptr<scalar_t>(), state_c.data_ptr<scalar_t>(), stride_s_slot,
        stride_s_dim, stride_s_state, w_c.data_ptr<scalar_t>(),
        bias_f32.defined() ? bias_f32.data_ptr<float>() : nullptr,
        out.data_ptr<scalar_t>(), cache_idx_ptr,
        static_cast<int32_t>(pad_slot_id), batch, dim, seqlen, width, state_len,
        do_silu);
  });

  // Write back only when a type-conversion copy was made.
  // Layout-only non-contiguity is handled via strides above — no copy needed.
  if (!state_type_ok) conv_state.copy_(state_c);

  return out;
}

// ---------------------------------------------------------------------------
// selective_state_update
// ---------------------------------------------------------------------------
void selective_state_update_cpu_impl(
    at::Tensor& state,    // (nstates, nheads, dim, dstate)
    const at::Tensor& x,  // (N, nheads, dim)
    const at::Tensor& dt, const at::Tensor& A, const at::Tensor& B,
    const at::Tensor& C, const c10::optional<at::Tensor>& D,
    const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& dt_bias, bool dt_softplus,
    const c10::optional<at::Tensor>& state_batch_indices,
    const c10::optional<at::Tensor>& dst_state_batch_indices,
    int64_t null_block_id, at::Tensor& out,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    const c10::optional<at::Tensor>& cu_seqlens) {
  at::ScalarType state_type = state.scalar_type();
  at::ScalarType input_type = x.scalar_type();

  // x, B, C must be contiguous and match input_type
  auto ensure_input = [input_type](const at::Tensor& t) -> at::Tensor {
    at::Tensor r = (t.scalar_type() != input_type) ? t.to(input_type) : t;
    return r.is_contiguous() ? r : r.contiguous();
  };
  at::Tensor x_in = ensure_input(x);
  at::Tensor B_in = ensure_input(B);
  at::Tensor C_in = ensure_input(C);
  at::Tensor z_in;
  if (z.has_value() && z.value().defined()) z_in = ensure_input(z.value());

  // A, D, dt_bias are float32 model parameters that arrive here as expanded
  // tensors, e.g. A is (nheads, head_dim, dstate) with strides (1, 0, 0).
  // We need just the scalar value per head as a (nheads,) 1-D array so that
  // A_ptr[h] in the kernel correctly reads head h's value.
  //
  // Strategy: peel trailing expanded (stride=0) dims via .select(), which is
  // a zero-copy view.  For A: (nheads, head_dim, dstate) strides (1,0,0)
  //   → .select(2,0) → (nheads, head_dim) strides (1,0)
  //   → .select(1,0) → (nheads,)          stride  (1,)   ← contiguous, free.
  // No allocation, no type conversion (A is already float32).
  auto to_per_head_1d_f32 = [](const at::Tensor& t) -> at::Tensor {
    at::Tensor r = t;
    // Peel trailing dimensions that are broadcast (stride=0 or size=1)
    while (r.dim() > 1) r = r.select(r.dim() - 1, 0);
    if (r.scalar_type() != at::kFloat) r = r.to(at::kFloat);
    return r.is_contiguous() ? r : r.contiguous();
  };

  at::Tensor A_f32 = to_per_head_1d_f32(A);  // (nheads,) float32
  at::Tensor D_f32, dt_bias_f32;
  if (D.has_value() && D.value().defined())
    D_f32 = to_per_head_1d_f32(D.value());
  if (dt_bias.has_value() && dt_bias.value().defined())
    dt_bias_f32 = to_per_head_1d_f32(dt_bias.value());

  // dt: reduce (N, nheads, head_dim) expanded tensor → (N, nheads) BEFORE
  // the type conversion so we convert head_dim x fewer elements.
  at::Tensor dt_f32;
  {
    // If dt was expanded to (N, nheads, head_dim) with stride-0 in dim 2,
    // take a zero-copy view of index 0 along that dim first.
    at::Tensor t2 = (dt.dim() == 3) ? dt.select(2, 0) : dt;  // (N, nheads)
    at::Tensor t3 = (t2.scalar_type() != at::kFloat) ? t2.to(at::kFloat) : t2;
    dt_f32 = t3.is_contiguous() ? t3 : t3.contiguous();
  }

  int64_t nheads = state.size(1);
  int64_t dim = state.size(2);
  int64_t dstate = state.size(3);
  int64_t N = (cu_seqlens.has_value() && cu_seqlens.value().defined())
                  ? cu_seqlens.value().size(0) - 1
                  : x_in.size(0);
  int64_t ngroups = B_in.size(1);

  // Strides
  int64_t stride_state_n = state.stride(0);
  int64_t stride_state_h = state.stride(1);
  int64_t stride_state_d = state.stride(2);
  int64_t stride_x_n = x_in.stride(0);
  int64_t stride_x_h = x_in.stride(1);
  int64_t stride_dt_n = dt_f32.stride(0);  // dt is (N, nheads)
  int64_t stride_BC_n = B_in.stride(0);
  int64_t stride_BC_g = B_in.stride(1);
  int64_t stride_out_n = out.stride(0);
  int64_t stride_out_h = out.stride(1);

  // Optional index pointers
  auto get_int32_ptr =
      [](const c10::optional<at::Tensor>& opt) -> const int32_t* {
    return (opt.has_value() && opt.value().defined())
               ? opt.value().data_ptr<int32_t>()
               : nullptr;
  };
  const int32_t* sbi_ptr = get_int32_ptr(state_batch_indices);
  const int32_t* dsbi_ptr = get_int32_ptr(dst_state_batch_indices);
  const int32_t* nat_ptr = get_int32_ptr(num_accepted_tokens);
  const int32_t* csl_ptr = get_int32_ptr(cu_seqlens);

  // Dispatch on (state_t, input_t, out_t): write directly into `out`
  // without any intermediate float32 buffer.
  VLLM_DISPATCH_FLOATING_TYPES(state_type, "ssu_state", [&] {
    using state_t = scalar_t;
    VLLM_DISPATCH_FLOATING_TYPES(input_type, "ssu_input", [&] {
      using input_t = scalar_t;
      VLLM_DISPATCH_FLOATING_TYPES(out.scalar_type(), "ssu_out", [&] {
        using out_t = scalar_t;
        mamba_cpu::selective_state_update_kernel<state_t, input_t, out_t>(
            state.data_ptr<state_t>(), stride_state_n, stride_state_h,
            stride_state_d, x_in.data_ptr<input_t>(), stride_x_n, stride_x_h,
            dt_f32.data_ptr<float>(), stride_dt_n, A_f32.data_ptr<float>(),
            B_in.data_ptr<input_t>(), C_in.data_ptr<input_t>(), stride_BC_n,
            stride_BC_g, D_f32.defined() ? D_f32.data_ptr<float>() : nullptr,
            z_in.defined() ? z_in.data_ptr<input_t>() : nullptr,
            dt_bias_f32.defined() ? dt_bias_f32.data_ptr<float>() : nullptr,
            out.data_ptr<out_t>(), stride_out_n, stride_out_h, sbi_ptr,
            dsbi_ptr, static_cast<int32_t>(null_block_id), nat_ptr, csl_ptr, N,
            nheads, ngroups, dim, dstate, dt_softplus);
      });
    });
  });
}

// ---------------------------------------------------------------------------
// mamba_chunk_scan_fwd_cpu
// ---------------------------------------------------------------------------
void mamba_chunk_scan_fwd_cpu_impl(
    at::Tensor& out,  // [seqlen, nheads, headdim] — pre-allocated by caller
    at::Tensor&
        final_states,     // [batch, nheads, headdim, dstate] float32 contiguous
    const at::Tensor& x,  // [seqlen, nheads, headdim]
    const at::Tensor&
        dt,  // [seqlen, nheads] float32 (preprocessed: bias+softplus+clamp)
    const at::Tensor& A,                 // [nheads] float32
    const at::Tensor& B,                 // [seqlen, ngroups, dstate]
    const at::Tensor& C,                 // [seqlen, ngroups, dstate]
    const c10::optional<at::Tensor>& D,  // [nheads] float32 (optional)
    const c10::optional<at::Tensor>& z,  // [seqlen, nheads, headdim] (optional)
    const at::Tensor& cu_seqlens         // [batch+1] int32
) {
  const at::ScalarType input_type = x.scalar_type();

  auto ensure_contig = [input_type](const at::Tensor& t) -> at::Tensor {
    at::Tensor r = (t.scalar_type() != input_type) ? t.to(input_type) : t;
    return r.is_contiguous() ? r : r.contiguous();
  };
  at::Tensor x_in = ensure_contig(x);
  at::Tensor B_in = ensure_contig(B);
  at::Tensor C_in = ensure_contig(C);
  at::Tensor z_in;
  if (z.has_value() && z.value().defined()) z_in = ensure_contig(z.value());

  // A and D are float32 model parameters, potentially broadcast-expanded.
  // Strip trailing broadcast dims to get a contiguous (nheads,) array.
  auto to_per_head_f32 = [](const at::Tensor& t) -> at::Tensor {
    at::Tensor r = t;
    while (r.dim() > 1) r = r.select(r.dim() - 1, 0);
    if (r.scalar_type() != at::kFloat) r = r.to(at::kFloat);
    return r.is_contiguous() ? r : r.contiguous();
  };
  at::Tensor A_f32 = to_per_head_f32(A);
  at::Tensor D_f32;
  if (D.has_value() && D.value().defined()) D_f32 = to_per_head_f32(D.value());

  // dt: [seqlen, nheads] float32 — caller has applied bias+softplus+clamp in
  // Python.
  at::Tensor dt_c = dt.is_contiguous() ? dt : dt.contiguous();
  if (dt_c.scalar_type() != at::kFloat) dt_c = dt_c.to(at::kFloat);

  at::Tensor cu_int = cu_seqlens.to(at::kInt).contiguous();

  const int64_t batch = final_states.size(0);
  const int64_t nheads = final_states.size(1);
  const int64_t headdim = final_states.size(2);
  const int64_t dstate = final_states.size(3);
  const int64_t ngroups = B_in.size(1);

  TORCH_CHECK(final_states.is_contiguous(),
              "mamba_chunk_scan_fwd_cpu: final_states must be contiguous");
  TORCH_CHECK(out.is_contiguous(),
              "mamba_chunk_scan_fwd_cpu: out must be contiguous (writes via "
              "raw data_ptr)");

  VLLM_DISPATCH_FLOATING_TYPES(input_type, "mamba_chunk_scan_fwd_cpu", [&] {
    mamba_cpu::mamba_chunk_scan_fwd_kernel<scalar_t>(
        final_states.data_ptr<float>(), x_in.data_ptr<scalar_t>(),
        dt_c.data_ptr<float>(), A_f32.data_ptr<float>(),
        B_in.data_ptr<scalar_t>(), C_in.data_ptr<scalar_t>(),
        D_f32.defined() ? D_f32.data_ptr<float>() : nullptr,
        z_in.defined() ? z_in.data_ptr<scalar_t>() : nullptr,
        out.data_ptr<scalar_t>(), cu_int.data_ptr<int32_t>(), batch, nheads,
        ngroups, headdim, dstate);
  });
}
