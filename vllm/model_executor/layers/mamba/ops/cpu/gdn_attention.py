# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
import torch.nn.functional as F

import vllm._custom_ops as ops
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d import (
    causal_conv1d_fn_cpu as causal_conv1d_torch,
)
from vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d import (
    causal_conv1d_update_cpu,
    causal_conv1d_update_torch,
)
from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

_CPU_GDN_ATTENTION_OPS_REGISTERED = False


def _cpu_gdn_layout_error(
    branch: str,
    layout: str,
    name: str,
    tensor: torch.Tensor,
    reason: str,
) -> RuntimeError:
    return RuntimeError(
        f"CPU GDN {branch} {layout} layout invalid for {name}: {reason}; "
        f"shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}"
    )


def _prepare_cpu_gdn_recurrent_inputs(
    branch: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Validate and materialize the CPU recurrent-kernel arguments."""
    recurrent_inputs = {
        "q": (q, q.dtype),
        "k": (k, q.dtype),
        "v": (v, q.dtype),
        "a": (a, q.dtype),
        "b": (b, q.dtype),
        "dt_bias": (dt_bias, q.dtype),
        "state_indices": (state_indices, torch.int32),
        "cu_seqlens": (cu_seqlens, torch.int32),
    }
    if num_accepted_tokens is not None:
        recurrent_inputs["num_accepted_tokens"] = (
            num_accepted_tokens,
            torch.int32,
        )

    for name, (tensor, expected_dtype) in recurrent_inputs.items():
        if tensor.device.type != "cpu":
            raise _cpu_gdn_layout_error(
                branch,
                "recurrent-input",
                name,
                tensor,
                "expected a CPU tensor",
            )
        if tensor.dtype != expected_dtype:
            raise _cpu_gdn_layout_error(
                branch,
                "recurrent-input",
                name,
                tensor,
                f"expected dtype {expected_dtype}, got {tensor.dtype}",
            )
        if tensor.ndim == 0:
            raise _cpu_gdn_layout_error(
                branch,
                "recurrent-input",
                name,
                tensor,
                "expected at least one dimension",
            )

    materialized = []
    for name, (tensor, _) in recurrent_inputs.items():
        if tensor.stride(-1) != 1:
            tensor = tensor.clone(memory_format=torch.contiguous_format)
        else:
            tensor = tensor.contiguous()
        if tensor.stride(-1) != 1:
            raise _cpu_gdn_layout_error(
                branch,
                "recurrent-input",
                name,
                tensor,
                "expected stride(-1) == 1 after materialization",
            )
        materialized.append(tensor)

    q, k, v, a, b, dt_bias, state_indices, cu_seqlens = materialized[:8]
    num_accepted_tokens = None if num_accepted_tokens is None else materialized[8]

    return (
        q,
        k,
        v,
        a,
        b,
        dt_bias,
        state_indices,
        cu_seqlens,
        num_accepted_tokens,
    )


def _validate_cpu_gdn_recurrent_state(state: torch.Tensor, branch: str) -> torch.Tensor:
    """Validate a writable recurrent state view without copying it."""
    if state.device.type != "cpu":
        raise _cpu_gdn_layout_error(
            branch,
            "recurrent-state",
            "initial_state_source",
            state,
            "expected a CPU tensor",
        )
    if state.ndim != 4:
        raise _cpu_gdn_layout_error(
            branch,
            "recurrent-state",
            "initial_state_source",
            state,
            "expected rank 4",
        )

    expected_inner_strides = (
        state.size(-2) * state.size(-1),
        state.size(-1),
        1,
    )
    if tuple(state.stride())[-3:] != expected_inner_strides:
        raise _cpu_gdn_layout_error(
            branch,
            "recurrent-state",
            "initial_state_source",
            state,
            "expected dense [head_dim, v_head_dim] inner strides",
        )
    return state


def cpu_gdn_attention_core(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """CPU custom op for the core GDN attention computation."""
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]

    attn_metadata = forward_context.attn_metadata

    if attn_metadata is None:
        return

    assert isinstance(attn_metadata, dict)
    attn_metadata_i = attn_metadata[layer.prefix]
    assert isinstance(attn_metadata_i, GDNAttentionMetadata)

    if attn_metadata_i.num_actual_tokens == 0:
        return

    assert mixed_qkv.dtype == torch.bfloat16, "CPU GDN attention requires BF16."

    # The conv-state cache temporal dim is ``width - 1`` when speculative
    # decoding is disabled, and ``width - 1 + num_spec`` when it is enabled
    # (vLLM allocates a wider rolling buffer to support draft-token rollback).
    width = layer.conv1d.weight.size(-1)
    conv_cache = layer.kv_cache[0]
    if is_conv_state_dim_first():
        # DS layout: (num_slots, dim, state_len)
        state_len = conv_cache.size(-1)
    else:
        # SD layout: (num_slots, state_len, dim)
        state_len = conv_cache.size(-2)

    spec_decode_cache = state_len > (width - 1)

    if not spec_decode_cache:
        # ===== Fast path: speculative decoding NOT configured =====
        # conv-state is exactly (..., dim, width-1); use the original AMX /
        # torch implementation unchanged.
        _cpu_gdn_attention_nonspec(
            layer, attn_metadata_i, mixed_qkv, b, a, core_attn_out
        )
        return

    # ===== Speculative-decoding-aware path (wide rolling conv buffer) =====
    _cpu_gdn_attention_spec_aware(
        layer, attn_metadata_i, mixed_qkv, b, a, core_attn_out, width, state_len
    )


# ---------------------------------------------------------------------------
# Non-speculative implementation: conv-state is exactly width-1.
# ---------------------------------------------------------------------------
def _cpu_gdn_attention_nonspec(
    layer,
    attn_metadata_i: GDNAttentionMetadata,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
) -> None:
    assert (
        attn_metadata_i.spec_sequence_masks is None
        and attn_metadata_i.num_accepted_tokens is None
    ), "speculative decode not supported without a wide conv-state cache."

    state_indices_tensor = attn_metadata_i.non_spec_state_indices_tensor
    query_start_loc = attn_metadata_i.non_spec_query_start_loc
    assert state_indices_tensor is not None
    assert query_start_loc is not None

    # C++ conv (conv.cpp) uses VDPBF16PS, not AMX tiles, so it runs on any
    # AVX-512BF16 CPU; weight is VNNI-packed on this same predicate at load time.
    # The C++ kernel accepts both physical state layouts.  SD is exposed as
    # the kernel's logical [dim, state_len] view; DS is already in that view.
    use_cpp_conv = torch.cpu._is_avx512_bf16_supported()

    conv_state = layer.kv_cache[0]
    if use_cpp_conv:
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(1, 2)
    else:
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        conv_weights = _unpacked_conv_weight(layer)

    # [num_allocated_slots, num_v_heads / tp_size, v_dim, k_dim]
    ssm_state = _ssm_state_view(layer, "nonspec")
    mixed_qkv = mixed_qkv.contiguous()
    a = a.contiguous()
    b = b.contiguous()

    num_decodes = attn_metadata_i.num_decodes
    num_decode_tokens = attn_metadata_i.num_decode_tokens
    num_prefills = attn_metadata_i.num_prefills
    num_prefill_tokens = attn_metadata_i.num_prefill_tokens

    # all decode requests (batched)
    if num_decodes > 0:
        decode_mixed_qkv = mixed_qkv[:num_decode_tokens]
        decode_b = b[:num_decode_tokens].contiguous()
        decode_a = a[:num_decode_tokens].contiguous()
        decode_state_indices = state_indices_tensor[:num_decodes].contiguous()
        decode_query_start_loc = query_start_loc[: num_decodes + 1].contiguous()
        if use_cpp_conv:
            decode_mixed_qkv = ops.causal_conv1d_update_cpu(
                x=decode_mixed_qkv,
                conv_states=conv_state,
                weight=layer.conv1d.weight,
                bias=layer.conv1d.bias,
                silu_activation=(layer.activation == "silu"),
                conv_state_indices=decode_state_indices,
                is_vnni=True,
            )
        else:
            if current_platform.get_cpu_architecture() == CpuArchEnum.ARM:
                decode_conv_state = conv_state[decode_state_indices].contiguous()
                decode_mixed_qkv = causal_conv1d_update_torch(
                    x=decode_mixed_qkv.unsqueeze(-1),
                    conv_state=decode_conv_state,
                    weight=conv_weights,
                    bias=layer.conv1d.bias,
                    activation=layer.activation,
                ).squeeze(-1)
                conv_state[decode_state_indices] = decode_conv_state
            else:
                decode_mixed_qkv = causal_conv1d_update_cpu(
                    x=decode_mixed_qkv,
                    conv_state=conv_state,
                    weight=conv_weights,
                    bias=layer.conv1d.bias,
                    activation=layer.activation,
                    conv_state_indices=decode_state_indices,
                )

        query, key, value = layer.rearrange_mixed_qkv(decode_mixed_qkv)
        (
            query,
            key,
            value,
            decode_a,
            decode_b,
            decode_dt_bias,
            decode_state_indices,
            decode_query_start_loc,
            _,
        ) = _prepare_cpu_gdn_recurrent_inputs(
            "nonspec.decode",
            query,
            key,
            value,
            decode_a,
            decode_b,
            layer.dt_bias,
            decode_state_indices,
            decode_query_start_loc,
        )

        attn_out = ops.fused_sigmoid_gating_delta_rule_update_cpu(
            A_log=layer.A_log.contiguous(),
            dt_bias=decode_dt_bias,
            q=query,
            k=key,
            v=value,
            a=decode_a,
            b=decode_b,
            initial_state_source=ssm_state,
            initial_state_indices=decode_state_indices,
            cu_seqlens=decode_query_start_loc,
            use_qk_l2norm_in_kernel=True,
        )
        core_attn_out[:num_decode_tokens] = attn_out.squeeze(1)

    # all prefill requests: (varlen) currently naively loops over sequences
    if num_prefills > 0:
        has_initial_state = attn_metadata_i.has_initial_state
        assert has_initial_state is not None

        prefill_token_start = num_decode_tokens
        prefill_token_end = prefill_token_start + num_prefill_tokens
        prefill_mixed_qkv = mixed_qkv[prefill_token_start:prefill_token_end]
        prefill_b = b[prefill_token_start:prefill_token_end].contiguous()
        prefill_a = a[prefill_token_start:prefill_token_end].contiguous()
        prefill_state_indices = state_indices_tensor[
            num_decodes : num_decodes + num_prefills
        ].contiguous()
        prefill_query_start_loc = (
            query_start_loc[num_decodes : num_decodes + num_prefills + 1]
            - num_decode_tokens
        ).contiguous()
        prefill_has_initial_state = has_initial_state[
            num_decodes : num_decodes + num_prefills
        ]

        if use_cpp_conv:
            prefill_mixed_qkv = ops.causal_conv1d_fwd_cpu(
                x=prefill_mixed_qkv.transpose(0, 1),
                weight=layer.conv1d.weight,
                bias=layer.conv1d.bias,
                conv_states=conv_state,
                query_start_loc=prefill_query_start_loc,
                cache_indices=prefill_state_indices,
                has_initial_state=prefill_has_initial_state,
                silu_activation=layer.activation == "silu",
                is_vnni=True,
            ).transpose(0, 1)
        else:
            prefill_mixed_qkv = causal_conv1d_torch(
                x=prefill_mixed_qkv.transpose(0, 1),
                weight=conv_weights,
                bias=layer.conv1d.bias,
                conv_states=conv_state,
                query_start_loc=prefill_query_start_loc,
                cache_indices=prefill_state_indices,
                has_initial_state=prefill_has_initial_state,
                activation=layer.activation,
            ).transpose(0, 1)

        query, key, value = layer.rearrange_mixed_qkv(prefill_mixed_qkv)
        (
            query,
            key,
            value,
            prefill_a,
            prefill_b,
            prefill_dt_bias,
            prefill_state_indices,
            prefill_query_start_loc,
            _,
        ) = _prepare_cpu_gdn_recurrent_inputs(
            "nonspec.prefill",
            query,
            key,
            value,
            prefill_a,
            prefill_b,
            layer.dt_bias,
            prefill_state_indices,
            prefill_query_start_loc,
        )
        g, beta = ops.fused_gdn_gating_cpu(
            A_log=layer.A_log.contiguous(),
            a=prefill_a,
            b=prefill_b,
            dt_bias=prefill_dt_bias,
        )

        # zero pool slots for sequences without a prior state; the kernel
        # gathers/mutates ssm_state in place via prefill_state_indices, so no
        # separate scatter-back is needed after the call.
        no_initial_state = prefill_state_indices[~prefill_has_initial_state]
        if no_initial_state.numel() > 0:
            ssm_state[no_initial_state] = 0
        attn_out, _ = ops.chunk_gated_delta_rule_cpu(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            initial_state=ssm_state,
            output_final_state=True,
            cu_seqlens=prefill_query_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            initial_state_indices=prefill_state_indices,
        )
        core_attn_out[prefill_token_start:prefill_token_end] = attn_out.squeeze(0)


# ---------------------------------------------------------------------------
# Speculative-decoding-aware implementation (wide rolling conv buffer).
#
# Mirrors the GPU `_forward_core` spec path using torch + the existing CPU
# kernels. The conv-state is a per-sequence rolling buffer of length
# ``state_len = width - 1 + num_spec`` (single cache slot, column 0 of the
# spec block table), and the SSM recurrent state is stored multi-slot: the
# state *after* draft token ``t`` lives in block-table column ``t``, so the
# next step can resume from the slot of the last accepted token.
# ---------------------------------------------------------------------------
def _conv_buffer_view(layer) -> torch.Tensor:
    """Return the conv-state cache as (num_slots, dim, state_len)."""
    conv_cache = layer.kv_cache[0]
    if is_conv_state_dim_first():
        return conv_cache  # (num_slots, dim, state_len)
    return conv_cache.transpose(-1, -2)  # view -> (num_slots, dim, state_len)


def _ssm_state_view(layer, branch: str) -> torch.Tensor:
    raw_state = layer.kv_cache[1]
    if raw_state.ndim != 4:
        raise _cpu_gdn_layout_error(
            branch,
            "recurrent-state",
            "kv_cache[1]",
            raw_state,
            "expected raw shape [slots, heads, value_dim, key_dim]",
        )

    value_dim = raw_state.size(-2)
    key_dim = raw_state.size(-1)
    expected_raw_inner_strides = (
        raw_state.size(-2) * raw_state.size(-1),
        raw_state.size(-1),
        1,
    )
    if tuple(raw_state.stride())[-3:] != expected_raw_inner_strides:
        raise _cpu_gdn_layout_error(
            branch,
            "recurrent-state",
            "kv_cache[1]",
            raw_state,
            "expected dense raw [value_dim, key_dim] inner strides",
        )

    state = raw_state.as_strided(
        size=(raw_state.size(0), raw_state.size(1), key_dim, value_dim),
        stride=(raw_state.stride(0), raw_state.stride(1), value_dim, 1),
    )
    return _validate_cpu_gdn_recurrent_state(state, branch)


def _unpacked_conv_weight(layer) -> torch.Tensor:
    """Return the plain (dim, width) conv weight.

    On AMX the conv1d weight is VNNI-packed in place at load time (only usable
    by the AMX C++ kernel), so the torch spec-decode path relies on the
    un-packed copy stashed by ``dispatch_cpu_unquantized_gemm``.
    """
    w = getattr(layer.conv1d, "_cpu_unpacked_conv_weight", None)
    if w is not None:
        return w
    w = layer.conv1d.weight
    if w.dim() == 3:
        return w.view(w.size(0), w.size(2))
    return w


def _cpu_gdn_attention_spec_aware(
    layer,
    attn_metadata_i: GDNAttentionMetadata,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    width: int,
    state_len: int,
) -> None:
    spec_sequence_masks = attn_metadata_i.spec_sequence_masks
    conv_buf = _conv_buffer_view(layer)  # (num_slots, dim, state_len)
    ssm_state = _ssm_state_view(layer, "spec-aware")

    if spec_sequence_masks is None:
        # No spec sequences in this batch (e.g. the prompt prefill step while
        # speculative decoding is configured). Process as ordinary
        # prefill/decode while touching only the first ``width-1`` columns of
        # the wide buffer, leaving the rolling history untouched.
        _spec_aware_nonspec(
            layer,
            attn_metadata_i,
            mixed_qkv.contiguous(),
            b.contiguous(),
            a.contiguous(),
            core_attn_out,
            conv_buf,
            ssm_state,
            width,
        )
        return

    # Split spec vs non-spec tokens.
    spec_token_indx = attn_metadata_i.spec_token_indx
    non_spec_token_indx = attn_metadata_i.non_spec_token_indx
    num_prefills = attn_metadata_i.num_prefills
    num_decodes = attn_metadata_i.num_decodes

    if num_prefills == 0 and num_decodes == 0:
        mixed_qkv_spec = mixed_qkv.contiguous()
        b_spec = b.contiguous()
        a_spec = a.contiguous()
        spec_out_indx = None
    else:
        assert spec_token_indx is not None
        mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
        b_spec = b.index_select(0, spec_token_indx)
        a_spec = a.index_select(0, spec_token_indx)
        spec_out_indx = spec_token_indx

    spec_out = _spec_forward(
        layer,
        attn_metadata_i,
        mixed_qkv_spec,
        b_spec,
        a_spec,
        conv_buf,
        ssm_state,
        width,
        state_len,
    )

    # Process the (rare) non-spec subset (prefill) tokens, if any.
    nonspec_out = None
    if (num_prefills > 0 or num_decodes > 0) and non_spec_token_indx is not None:
        mixed_qkv_ns = mixed_qkv.index_select(0, non_spec_token_indx)
        b_ns = b.index_select(0, non_spec_token_indx)
        a_ns = a.index_select(0, non_spec_token_indx)
        nonspec_out = _spec_aware_nonspec_subset(
            layer, attn_metadata_i, mixed_qkv_ns, b_ns, a_ns, conv_buf, ssm_state, width
        )

    # Scatter outputs back to their token positions.
    if spec_out_indx is None:
        core_attn_out[: spec_out.size(0)] = spec_out
    else:
        core_attn_out.index_copy_(0, spec_out_indx, spec_out)
        if nonspec_out is not None:
            core_attn_out.index_copy_(0, non_spec_token_indx, nonspec_out)


def _spec_forward(
    layer,
    attn_metadata_i: GDNAttentionMetadata,
    mixed_qkv_spec: torch.Tensor,
    b_spec: torch.Tensor,
    a_spec: torch.Tensor,
    conv_buf: torch.Tensor,
    ssm_state: torch.Tensor,
    width: int,
    state_len: int,
) -> torch.Tensor:
    """Run the GDN core for the multi-query (speculative) tokens.

    Returns the core attention output for the spec tokens, in the same token
    order as ``mixed_qkv_spec`` (i.e. ordered by ``spec_query_start_loc``).
    """
    num_spec_decodes = attn_metadata_i.num_spec_decodes
    spec_state_indices = attn_metadata_i.spec_state_indices_tensor
    spec_qsl = attn_metadata_i.spec_query_start_loc
    num_accepted = attn_metadata_i.num_accepted_tokens
    assert spec_state_indices is not None
    assert spec_qsl is not None
    assert num_accepted is not None

    mixed_qkv_spec = mixed_qkv_spec.contiguous()
    a_spec = a_spec.contiguous()
    b_spec = b_spec.contiguous()
    spec_state_indices = (
        spec_state_indices[:num_spec_decodes].to("cpu", torch.int32).contiguous()
    )
    spec_qsl_cpu = spec_qsl[: num_spec_decodes + 1].to("cpu", torch.int32).contiguous()
    num_accepted = num_accepted[:num_spec_decodes].to("cpu", torch.int32).contiguous()
    seq_starts = spec_qsl_cpu[:-1]
    seq_lens = spec_qsl_cpu[1:] - spec_qsl_cpu[:-1]

    # ---- 1. Convolution (per-sequence rolling buffer) ----
    dim = mixed_qkv_spec.size(-1)
    bias = layer.conv1d.bias
    silu = layer.activation == "silu"

    can_use_native_conv = (
        torch.cpu._is_avx512_bf16_supported()
        and width == 4
        and num_spec_decodes > 0
        and bool(torch.all(seq_lens == seq_lens[0]).item())
        and int(seq_lens[0].item()) > 0
    )
    if can_use_native_conv:
        q_i = int(seq_lens[0].item())
        conv_out = ops.causal_conv1d_update_cpu(
            x=mixed_qkv_spec.view(num_spec_decodes, q_i, dim),
            conv_states=conv_buf,
            weight=layer.conv1d.weight,
            bias=bias,
            silu_activation=silu,
            conv_state_indices=spec_state_indices[:, 0].contiguous(),
            is_vnni=True,
            num_accepted_tokens=num_accepted,
        ).view_as(mixed_qkv_spec)
    else:
        w = _unpacked_conv_weight(layer).unsqueeze(1)
        col0 = spec_state_indices[:num_spec_decodes, 0]
        num_acc_cpu = num_accepted
        conv_out = torch.empty_like(mixed_qkv_spec)
        for i in range(num_spec_decodes):
            q_i = int(seq_lens[i].item())
            if q_i == 0:
                continue
            start = int(seq_starts[i].item())
            slot0 = int(col0[i].item())
            offset = int(num_acc_cpu[i].item()) - 1
            B = conv_buf[slot0]  # (dim, state_len)
            x_seq = mixed_qkv_spec[start : start + q_i].transpose(0, 1).to(B.dtype)
            prior = B[:, offset : offset + (width - 1)]
            conv_in = torch.cat([prior, x_seq], dim=-1).unsqueeze(0)
            out = F.conv1d(conv_in, w, bias, groups=dim)[0]  # (dim, q_i)
            if silu:
                out = F.silu(out)
            conv_out[start : start + q_i] = out.transpose(0, 1).to(conv_out.dtype)
            # Roll the buffer: drop the accepted history from the front, append
            # the new draft tokens, keep total length == state_len.
            keep = B[:, offset + 1 : offset + 1 + (state_len - q_i)]
            new_B = torch.cat([keep, x_seq], dim=-1)
            B.copy_(new_B)

    # ---- 2. Recurrent (multi-slot SSM state) ----
    # Single fused kernel call: it runs the recurrence over each sequence's
    # draft tokens internally, resumes from slot ``num_accepted-1`` and stores
    # the state after token ``t`` into slot ``t`` (rollback for the next step).
    query, key, value = layer.rearrange_mixed_qkv(conv_out)
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    query = query.squeeze(0)
    key = key.squeeze(0)
    value = value.squeeze(0)
    if a_spec.numel() == 0 or b_spec.numel() == 0:
        # Keep metadata-only callers compatible; real spec requests always
        # provide non-empty recurrent inputs and take the validated path.
        return ops.fused_sigmoid_gating_delta_rule_update_spec_cpu(
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            q=query,
            k=key,
            v=value,
            a=a_spec,
            b=b_spec,
            initial_state_source=ssm_state,
            spec_state_indices=spec_state_indices,
            num_accepted_tokens=num_accepted,
            cu_seqlens=spec_qsl_cpu,
            use_qk_l2norm_in_kernel=True,
        )
    ssm_state = _validate_cpu_gdn_recurrent_state(ssm_state, "spec.forward")
    (
        query,
        key,
        value,
        a_spec,
        b_spec,
        dt_bias,
        spec_idx,
        cu,
        num_acc,
    ) = _prepare_cpu_gdn_recurrent_inputs(
        "spec.forward",
        query,
        key,
        value,
        a_spec,
        b_spec,
        layer.dt_bias,
        spec_state_indices,
        spec_qsl_cpu,
        num_accepted,
    )
    out_spec = ops.fused_sigmoid_gating_delta_rule_update_spec_cpu(
        A_log=layer.A_log.contiguous(),
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        a=a_spec,
        b=b_spec,
        initial_state_source=ssm_state,
        spec_state_indices=spec_idx,
        num_accepted_tokens=num_acc,
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=True,
    )
    return out_spec


def _spec_aware_nonspec(
    layer,
    attn_metadata_i: GDNAttentionMetadata,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_buf: torch.Tensor,
    ssm_state: torch.Tensor,
    width: int,
) -> None:
    """Non-spec prefill/decode with a wide conv buffer."""
    ssm_state = _validate_cpu_gdn_recurrent_state(ssm_state, "wide-nonspec")
    state_indices_tensor = attn_metadata_i.non_spec_state_indices_tensor
    query_start_loc = attn_metadata_i.non_spec_query_start_loc
    assert state_indices_tensor is not None
    assert query_start_loc is not None
    state_indices_tensor = state_indices_tensor.contiguous()

    use_native_conv = torch.cpu._is_avx512_bf16_supported()

    if not use_native_conv:
        conv_weights = _unpacked_conv_weight(layer)

    num_decodes = attn_metadata_i.num_decodes
    num_decode_tokens = attn_metadata_i.num_decode_tokens
    num_prefills = attn_metadata_i.num_prefills
    num_prefill_tokens = attn_metadata_i.num_prefill_tokens

    if num_decodes > 0:
        decode_mixed_qkv = mixed_qkv[:num_decode_tokens].contiguous()
        decode_b = b[:num_decode_tokens]
        decode_a = a[:num_decode_tokens]
        decode_state_indices = state_indices_tensor[:num_decodes].contiguous()
        decode_query_start_loc = query_start_loc[: num_decodes + 1].contiguous()
        if use_native_conv:
            decode_mixed_qkv = ops.causal_conv1d_update_cpu(
                x=decode_mixed_qkv,
                conv_states=conv_buf,
                weight=layer.conv1d.weight,
                bias=layer.conv1d.bias,
                silu_activation=layer.activation == "silu",
                conv_state_indices=decode_state_indices,
                is_vnni=True,
            )
        else:
            # Only the first ``width-1`` columns hold the real conv state.
            conv_state_view = conv_buf[:, :, : width - 1]
            if current_platform.get_cpu_architecture() == CpuArchEnum.ARM:
                decode_conv_state = conv_state_view[decode_state_indices].contiguous()
                decode_mixed_qkv = causal_conv1d_update_torch(
                    x=decode_mixed_qkv.unsqueeze(-1),
                    conv_state=decode_conv_state,
                    weight=conv_weights,
                    bias=layer.conv1d.bias,
                    activation=layer.activation,
                ).squeeze(-1)
                conv_state_view[decode_state_indices] = decode_conv_state
            else:
                decode_mixed_qkv = causal_conv1d_update_cpu(
                    x=decode_mixed_qkv,
                    conv_state=conv_state_view,
                    weight=conv_weights,
                    bias=layer.conv1d.bias,
                    activation=layer.activation,
                    conv_state_indices=decode_state_indices,
                )

        query, key, value = layer.rearrange_mixed_qkv(decode_mixed_qkv)
        (
            query,
            key,
            value,
            decode_a,
            decode_b,
            decode_dt_bias,
            decode_state_indices,
            decode_query_start_loc,
            _,
        ) = _prepare_cpu_gdn_recurrent_inputs(
            "wide-nonspec.decode",
            query,
            key,
            value,
            decode_a,
            decode_b,
            layer.dt_bias,
            decode_state_indices,
            decode_query_start_loc,
        )
        attn_out = ops.fused_sigmoid_gating_delta_rule_update_cpu(
            A_log=layer.A_log.contiguous(),
            dt_bias=decode_dt_bias,
            q=query,
            k=key,
            v=value,
            a=decode_a,
            b=decode_b,
            initial_state_source=ssm_state,
            initial_state_indices=decode_state_indices,
            cu_seqlens=decode_query_start_loc,
            use_qk_l2norm_in_kernel=True,
        )
        core_attn_out[:num_decode_tokens] = attn_out.squeeze(1)

    if num_prefills > 0:
        has_initial_state = attn_metadata_i.has_initial_state
        assert has_initial_state is not None
        prefill_token_start = num_decode_tokens
        prefill_token_end = prefill_token_start + num_prefill_tokens
        prefill_mixed_qkv = mixed_qkv[
            prefill_token_start:prefill_token_end
        ].contiguous()
        prefill_b = b[prefill_token_start:prefill_token_end]
        prefill_a = a[prefill_token_start:prefill_token_end]
        prefill_state_indices = state_indices_tensor[
            num_decodes : num_decodes + num_prefills
        ].contiguous()
        prefill_query_start_loc = (
            query_start_loc[num_decodes : num_decodes + num_prefills + 1]
            - num_decode_tokens
        ).contiguous()
        prefill_has_initial_state = has_initial_state[
            num_decodes : num_decodes + num_prefills
        ]
        if use_native_conv:
            prefill_mixed_qkv = ops.causal_conv1d_fwd_cpu(
                x=prefill_mixed_qkv.transpose(0, 1),
                weight=layer.conv1d.weight,
                bias=layer.conv1d.bias,
                conv_states=conv_buf,
                query_start_loc=prefill_query_start_loc,
                cache_indices=prefill_state_indices,
                has_initial_state=prefill_has_initial_state,
                silu_activation=layer.activation == "silu",
                is_vnni=True,
            ).transpose(0, 1)
        else:
            prefill_mixed_qkv = causal_conv1d_torch(
                x=prefill_mixed_qkv.transpose(0, 1),
                weight=conv_weights,
                bias=layer.conv1d.bias,
                conv_states=conv_buf,
                query_start_loc=prefill_query_start_loc,
                cache_indices=prefill_state_indices,
                has_initial_state=prefill_has_initial_state,
                activation=layer.activation,
            ).transpose(0, 1)

        query, key, value = layer.rearrange_mixed_qkv(prefill_mixed_qkv)
        (
            query,
            key,
            value,
            prefill_a,
            prefill_b,
            prefill_dt_bias,
            prefill_state_indices,
            prefill_query_start_loc,
            _,
        ) = _prepare_cpu_gdn_recurrent_inputs(
            "wide-nonspec.prefill",
            query,
            key,
            value,
            prefill_a,
            prefill_b,
            layer.dt_bias,
            prefill_state_indices,
            prefill_query_start_loc,
        )
        g, beta = ops.fused_gdn_gating_cpu(
            A_log=layer.A_log.contiguous(),
            a=prefill_a,
            b=prefill_b,
            dt_bias=prefill_dt_bias,
        )
        # zero pool slots for sequences without a prior state; the kernel
        # gathers/mutates ssm_state in place via prefill_state_indices, so no
        # separate scatter-back is needed after the call.
        no_initial_state = prefill_state_indices[~prefill_has_initial_state]
        if no_initial_state.numel() > 0:
            ssm_state[no_initial_state] = 0
        attn_out, _ = ops.chunk_gated_delta_rule_cpu(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            initial_state=ssm_state,
            output_final_state=True,
            cu_seqlens=prefill_query_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            initial_state_indices=prefill_state_indices,
        )
        core_attn_out[prefill_token_start:prefill_token_end] = attn_out.squeeze(0)


def _spec_aware_nonspec_subset(
    layer,
    attn_metadata_i: GDNAttentionMetadata,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    conv_buf: torch.Tensor,
    ssm_state: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """Process non-spec (prefill) tokens that coexist with spec sequences.

    Returns outputs ordered like ``non_spec_token_indx``.
    """
    has_initial_state = attn_metadata_i.has_initial_state
    prefill_state_indices = attn_metadata_i.prefill_state_indices
    prefill_qsl = attn_metadata_i.prefill_query_start_loc
    assert prefill_state_indices is not None and prefill_qsl is not None
    assert has_initial_state is not None
    ssm_state = _validate_cpu_gdn_recurrent_state(ssm_state, "wide-nonspec-subset")
    mixed_qkv = mixed_qkv.contiguous()
    a = a.contiguous()
    b = b.contiguous()
    prefill_state_indices = prefill_state_indices.contiguous()
    prefill_qsl = prefill_qsl.contiguous()

    use_native_conv = torch.cpu._is_avx512_bf16_supported()

    if use_native_conv:
        conv_out = ops.causal_conv1d_fwd_cpu(
            x=mixed_qkv.transpose(0, 1),
            weight=layer.conv1d.weight,
            bias=layer.conv1d.bias,
            conv_states=conv_buf,
            query_start_loc=prefill_qsl,
            cache_indices=prefill_state_indices,
            has_initial_state=has_initial_state,
            silu_activation=layer.activation == "silu",
            is_vnni=True,
        ).transpose(0, 1)
    else:
        conv_weights = _unpacked_conv_weight(layer)
        conv_out = causal_conv1d_torch(
            x=mixed_qkv.transpose(0, 1),
            weight=conv_weights,
            bias=layer.conv1d.bias,
            conv_states=conv_buf,
            query_start_loc=prefill_qsl,
            cache_indices=prefill_state_indices,
            has_initial_state=has_initial_state,
            activation=layer.activation,
        ).transpose(0, 1)

    query, key, value = layer.rearrange_mixed_qkv(conv_out)
    (
        query,
        key,
        value,
        a,
        b,
        dt_bias,
        prefill_state_indices,
        prefill_qsl,
        _,
    ) = _prepare_cpu_gdn_recurrent_inputs(
        "wide-nonspec-subset",
        query,
        key,
        value,
        a,
        b,
        layer.dt_bias,
        prefill_state_indices,
        prefill_qsl,
    )
    g, beta = ops.fused_gdn_gating_cpu(
        A_log=layer.A_log.contiguous(),
        a=a,
        b=b,
        dt_bias=dt_bias,
    )
    # zero pool slots for sequences without a prior state; the kernel
    # gathers/mutates ssm_state in place via prefill_state_indices, so no
    # separate scatter-back is needed after the call.
    no_initial_state = prefill_state_indices[~has_initial_state]
    if no_initial_state.numel() > 0:
        ssm_state[no_initial_state] = 0
    attn_out, _ = ops.chunk_gated_delta_rule_cpu(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        initial_state=ssm_state,
        output_final_state=True,
        cu_seqlens=prefill_qsl,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
        initial_state_indices=prefill_state_indices,
    )
    return attn_out.squeeze(0)


def cpu_gdn_attention_core_fake(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Fake implementation for torch.compile."""
    return


def register_cpu_gdn_attention_ops() -> None:
    global _CPU_GDN_ATTENTION_OPS_REGISTERED
    if _CPU_GDN_ATTENTION_OPS_REGISTERED:
        return

    direct_register_custom_op(
        op_name="cpu_gdn_attention_core",
        op_func=cpu_gdn_attention_core,
        mutates_args=["core_attn_out"],
        fake_impl=cpu_gdn_attention_core_fake,
    )
    _CPU_GDN_ATTENTION_OPS_REGISTERED = True
