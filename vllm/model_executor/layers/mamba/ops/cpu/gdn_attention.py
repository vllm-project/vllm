# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

import vllm._custom_ops as ops
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d import (
    causal_conv1d_fn_cpu,
    causal_conv1d_update_cpu,
    resolve_cpu_conv_weights,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

_CPU_GDN_ATTENTION_OPS_REGISTERED = False


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
    conv_weight, native_weight = resolve_cpu_conv_weights(layer.conv1d)

    if not spec_decode_cache:
        # Conv-state is exactly (..., dim, width - 1).
        _cpu_gdn_attention_nonspec(
            layer,
            attn_metadata_i,
            mixed_qkv,
            b,
            a,
            core_attn_out,
            conv_weight,
            native_weight,
        )
        return

    # ===== Speculative-decoding-aware path (wide rolling conv buffer) =====
    _cpu_gdn_attention_spec_aware(
        layer,
        attn_metadata_i,
        mixed_qkv,
        b,
        a,
        core_attn_out,
        width,
        conv_weight,
        native_weight,
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
    conv_weight: torch.Tensor,
    native_weight: torch.Tensor | None,
) -> None:
    assert (
        attn_metadata_i.spec_sequence_masks is None
        and attn_metadata_i.num_accepted_tokens is None
    ), "speculative decode not supported without a wide conv-state cache."

    state_indices_tensor = attn_metadata_i.non_spec_state_indices_tensor
    query_start_loc = attn_metadata_i.non_spec_query_start_loc
    assert state_indices_tensor is not None
    assert query_start_loc is not None

    conv_state = _conv_buffer_view(layer)

    # [num_allocated_slots, num_v_heads / tp_size, v_dim, k_dim]
    ssm_state = layer.kv_cache[1]
    mixed_qkv = mixed_qkv.contiguous()
    a = a.contiguous()
    b = b.contiguous()

    num_allocated_slots, head_num, v_dim, k_dim = ssm_state.size()
    ssm_state = ssm_state.view(num_allocated_slots, head_num, k_dim, v_dim)

    num_decodes = attn_metadata_i.num_decodes
    num_decode_tokens = attn_metadata_i.num_decode_tokens
    num_prefills = attn_metadata_i.num_prefills
    num_prefill_tokens = attn_metadata_i.num_prefill_tokens

    # all decode requests (batched)
    if num_decodes > 0:
        decode_mixed_qkv = mixed_qkv[:num_decode_tokens]
        decode_b = b[:num_decode_tokens]
        decode_a = a[:num_decode_tokens]
        decode_state_indices = state_indices_tensor[:num_decodes]
        decode_mixed_qkv = causal_conv1d_update_cpu(
            x=decode_mixed_qkv,
            conv_state=conv_state,
            weight=conv_weight,
            bias=layer.conv1d.bias,
            activation=layer.activation,
            conv_state_indices=decode_state_indices,
            native_weight=native_weight,
        )

        query, key, value = layer.rearrange_mixed_qkv(decode_mixed_qkv)

        attn_out = ops.fused_sigmoid_gating_delta_rule_update_cpu(
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            q=query,
            k=key,
            v=value,
            a=decode_a,
            b=decode_b,
            initial_state_source=ssm_state,
            initial_state_indices=decode_state_indices,
            cu_seqlens=query_start_loc[: num_decodes + 1],
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
        prefill_b = b[prefill_token_start:prefill_token_end]
        prefill_a = a[prefill_token_start:prefill_token_end]
        prefill_state_indices = state_indices_tensor[
            num_decodes : num_decodes + num_prefills
        ]
        prefill_query_start_loc = (
            query_start_loc[num_decodes : num_decodes + num_prefills + 1]
            - num_decode_tokens
        )
        prefill_has_initial_state = has_initial_state[
            num_decodes : num_decodes + num_prefills
        ]

        prefill_mixed_qkv = causal_conv1d_fn_cpu(
            x=prefill_mixed_qkv.transpose(0, 1),
            weight=conv_weight,
            bias=layer.conv1d.bias,
            conv_states=conv_state,
            query_start_loc=prefill_query_start_loc,
            cache_indices=prefill_state_indices,
            has_initial_state=prefill_has_initial_state,
            activation=layer.activation,
            native_weight=native_weight,
        ).transpose(0, 1)

        query, key, value = layer.rearrange_mixed_qkv(prefill_mixed_qkv)
        g, beta = ops.fused_gdn_gating_cpu(
            A_log=layer.A_log, a=prefill_a, b=prefill_b, dt_bias=layer.dt_bias
        )

        initial_state = ssm_state[prefill_state_indices]
        initial_state[~prefill_has_initial_state, ...] = 0
        attn_out, last_recurrent_state = ops.chunk_gated_delta_rule_cpu(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=prefill_query_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        ssm_state[prefill_state_indices] = last_recurrent_state.to(
            ssm_state.dtype, copy=False
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


def _ssm_state_view(layer) -> torch.Tensor:
    ssm_state = layer.kv_cache[1]
    num_slots, head_num, v_dim, k_dim = ssm_state.size()
    return ssm_state.view(num_slots, head_num, k_dim, v_dim)


def _cpu_gdn_attention_spec_aware(
    layer,
    attn_metadata_i: GDNAttentionMetadata,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    width: int,
    conv_weight: torch.Tensor,
    native_weight: torch.Tensor | None,
) -> None:
    spec_sequence_masks = attn_metadata_i.spec_sequence_masks
    conv_buf = _conv_buffer_view(layer)  # (num_slots, dim, state_len)
    ssm_state = _ssm_state_view(layer)

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
            conv_weight,
            native_weight,
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
        conv_weight,
        native_weight,
    )

    # Process the (rare) non-spec subset (prefill) tokens, if any.
    nonspec_out = None
    if (num_prefills > 0 or num_decodes > 0) and non_spec_token_indx is not None:
        mixed_qkv_ns = mixed_qkv.index_select(0, non_spec_token_indx)
        b_ns = b.index_select(0, non_spec_token_indx)
        a_ns = a.index_select(0, non_spec_token_indx)
        nonspec_out = _spec_aware_nonspec_subset(
            layer,
            attn_metadata_i,
            mixed_qkv_ns,
            b_ns,
            a_ns,
            conv_buf,
            ssm_state,
            width,
            conv_weight,
            native_weight,
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
    conv_weight: torch.Tensor,
    native_weight: torch.Tensor | None,
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

    conv_out = causal_conv1d_update_cpu(
        x=mixed_qkv_spec,
        conv_state=conv_buf,
        weight=conv_weight,
        bias=layer.conv1d.bias,
        activation=layer.activation,
        conv_state_indices=spec_state_indices[:num_spec_decodes, 0]
        .to("cpu", torch.int32)
        .contiguous(),
        query_start_loc=spec_qsl[: num_spec_decodes + 1].to("cpu", torch.int32),
        num_accepted_tokens=num_accepted[:num_spec_decodes]
        .to("cpu", torch.int32)
        .contiguous(),
        native_weight=native_weight,
    )

    # ---- 2. Recurrent (multi-slot SSM state) ----
    # Single fused kernel call: it runs the recurrence over each sequence's
    # draft tokens internally, resumes from slot ``num_accepted-1`` and stores
    # the state after token ``t`` into slot ``t`` (rollback for the next step).
    query, key, value = layer.rearrange_mixed_qkv(conv_out)
    query = query.squeeze(0)
    key = key.squeeze(0)
    value = value.squeeze(0)
    spec_idx = spec_state_indices[:num_spec_decodes].to(torch.int32).contiguous()
    num_acc = num_accepted[:num_spec_decodes].to(torch.int32).contiguous()
    cu = spec_qsl[: num_spec_decodes + 1].to(torch.int32).contiguous()
    out_spec = ops.fused_sigmoid_gating_delta_rule_update_spec_cpu(
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
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
    conv_weight: torch.Tensor,
    native_weight: torch.Tensor | None,
) -> None:
    """Non-spec prefill/decode with a wide conv buffer."""
    state_indices_tensor = attn_metadata_i.non_spec_state_indices_tensor
    query_start_loc = attn_metadata_i.non_spec_query_start_loc
    assert state_indices_tensor is not None
    assert query_start_loc is not None
    state_indices_tensor = state_indices_tensor.contiguous()

    num_decodes = attn_metadata_i.num_decodes
    num_decode_tokens = attn_metadata_i.num_decode_tokens
    num_prefills = attn_metadata_i.num_prefills
    num_prefill_tokens = attn_metadata_i.num_prefill_tokens

    if num_decodes > 0:
        decode_mixed_qkv = mixed_qkv[:num_decode_tokens]
        decode_b = b[:num_decode_tokens]
        decode_a = a[:num_decode_tokens]
        decode_state_indices = state_indices_tensor[:num_decodes]
        decode_mixed_qkv = causal_conv1d_update_cpu(
            x=decode_mixed_qkv,
            conv_state=conv_buf[:, :, : width - 1],
            weight=conv_weight,
            bias=layer.conv1d.bias,
            activation=layer.activation,
            conv_state_indices=decode_state_indices,
            native_weight=native_weight,
        )

        query, key, value = layer.rearrange_mixed_qkv(decode_mixed_qkv)
        attn_out = ops.fused_sigmoid_gating_delta_rule_update_cpu(
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            q=query,
            k=key,
            v=value,
            a=decode_a,
            b=decode_b,
            initial_state_source=ssm_state,
            initial_state_indices=decode_state_indices,
            cu_seqlens=query_start_loc[: num_decodes + 1],
            use_qk_l2norm_in_kernel=True,
        )
        core_attn_out[:num_decode_tokens] = attn_out.squeeze(1)

    if num_prefills > 0:
        has_initial_state = attn_metadata_i.has_initial_state
        assert has_initial_state is not None
        prefill_token_start = num_decode_tokens
        prefill_token_end = prefill_token_start + num_prefill_tokens
        prefill_mixed_qkv = mixed_qkv[prefill_token_start:prefill_token_end]
        prefill_b = b[prefill_token_start:prefill_token_end]
        prefill_a = a[prefill_token_start:prefill_token_end]
        prefill_state_indices = state_indices_tensor[
            num_decodes : num_decodes + num_prefills
        ]
        prefill_query_start_loc = (
            query_start_loc[num_decodes : num_decodes + num_prefills + 1]
            - num_decode_tokens
        )
        prefill_has_initial_state = has_initial_state[
            num_decodes : num_decodes + num_prefills
        ]
        prefill_mixed_qkv = causal_conv1d_fn_cpu(
            x=prefill_mixed_qkv.transpose(0, 1),
            weight=conv_weight,
            bias=layer.conv1d.bias,
            conv_states=conv_buf,
            query_start_loc=prefill_query_start_loc,
            cache_indices=prefill_state_indices,
            has_initial_state=prefill_has_initial_state,
            activation=layer.activation,
            native_weight=native_weight,
        ).transpose(0, 1)

        query, key, value = layer.rearrange_mixed_qkv(prefill_mixed_qkv)
        g, beta = ops.fused_gdn_gating_cpu(
            A_log=layer.A_log, a=prefill_a, b=prefill_b, dt_bias=layer.dt_bias
        )
        initial_state = ssm_state[prefill_state_indices]
        initial_state[~prefill_has_initial_state, ...] = 0
        attn_out, last_recurrent_state = ops.chunk_gated_delta_rule_cpu(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=prefill_query_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        ssm_state[prefill_state_indices] = last_recurrent_state.to(
            ssm_state.dtype, copy=False
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
    conv_weight: torch.Tensor,
    native_weight: torch.Tensor | None,
) -> torch.Tensor:
    """Process non-spec (prefill) tokens that coexist with spec sequences.

    Returns outputs ordered like ``non_spec_token_indx``.
    """
    has_initial_state = attn_metadata_i.has_initial_state
    prefill_state_indices = attn_metadata_i.prefill_state_indices
    prefill_qsl = attn_metadata_i.prefill_query_start_loc
    assert prefill_state_indices is not None and prefill_qsl is not None
    assert has_initial_state is not None
    prefill_state_indices = prefill_state_indices.contiguous()

    conv_out = causal_conv1d_fn_cpu(
        x=mixed_qkv.transpose(0, 1),
        weight=conv_weight,
        bias=layer.conv1d.bias,
        conv_states=conv_buf,
        query_start_loc=prefill_qsl,
        cache_indices=prefill_state_indices,
        has_initial_state=has_initial_state,
        activation=layer.activation,
        native_weight=native_weight,
    ).transpose(0, 1)

    query, key, value = layer.rearrange_mixed_qkv(conv_out)
    g, beta = ops.fused_gdn_gating_cpu(
        A_log=layer.A_log, a=a, b=b, dt_bias=layer.dt_bias
    )
    initial_state = ssm_state[prefill_state_indices]
    initial_state[~has_initial_state, ...] = 0
    attn_out, last_recurrent_state = ops.chunk_gated_delta_rule_cpu(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=prefill_qsl,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    ssm_state[prefill_state_indices] = last_recurrent_state.to(
        ssm_state.dtype, copy=False
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
