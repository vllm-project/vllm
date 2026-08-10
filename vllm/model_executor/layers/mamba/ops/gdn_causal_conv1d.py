# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn as official_causal_conv1d_fn,
)
from vllm.model_executor.layers.mamba.ops.gdn_causal_conv1d_apc import (
    apc_causal_conv1d,
)
from vllm.model_executor.layers.mamba.ops.gdn_causal_conv1d_generic import (
    generic_causal_conv1d,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

_IS_SM103 = current_platform.get_device_capability() == (10, 3)
_HAS_SM103_KERNEL = hasattr(torch.ops, "_C") and hasattr(
    torch.ops._C, "gdn_causal_conv1d_sm103"
)


def fast_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor | None:
    """Run the guarded SM103 single-sequence kernel, or return ``None``."""
    eligible = (
        _IS_SM103
        and _HAS_SM103_KERNEL
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.stride(0) == 1
        and x.stride(1) >= x.shape[0]
        and x.stride(1) % 4 == 0
        and x.storage_offset() % 4 == 0
        and x.shape[0] % 4 == 0
        and x.shape[1] >= 3
        and weight.dtype == x.dtype
        and weight.device == x.device
        and weight.shape == (x.shape[0], 4)
        and weight.stride(1) == 1
        and (
            bias is None
            or (
                bias.dtype == x.dtype
                and bias.device == x.device
                and bias.shape == (x.shape[0],)
                and bias.stride(0) == 1
            )
        )
        and conv_states.dtype == x.dtype
        and conv_states.device == x.device
        and conv_states.ndim == 3
        and conv_states.shape[1:] == (x.shape[0], 3)
        and cache_indices.dtype == torch.int32
        and cache_indices.device == x.device
        and cache_indices.numel() == 1
        and has_initial_state.dtype == torch.bool
        and has_initial_state.device == x.device
        and has_initial_state.numel() == 1
    )
    if not eligible:
        return None
    has_bias = bias is not None
    bias_arg = (
        bias if bias is not None else torch.empty(1, device=x.device, dtype=x.dtype)
    )
    return ops.gdn_causal_conv1d_sm103(
        x,
        weight,
        bias_arg,
        conv_states,
        cache_indices,
        has_initial_state,
        has_bias,
    )


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    null_block_id: int = NULL_BLOCK_ID,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    num_computed_tokens: torch.Tensor | None = None,
    block_size_to_align=0,
    metadata=None,
    validate_data=False,
):
    """Dispatch profitable long GDN prefills to optimized kernels."""
    if isinstance(activation, bool) and activation:
        activation = "silu"

    if validate_data:
        assert x.dim() == 2
        assert query_start_loc.dim() == 1
        assert x.stride(0) == 1 or x.stride(1) == 1
        if bias is not None:
            assert bias.dim() == 1 and bias.size(0) == x.size(0)
        if cache_indices is not None:
            assert cache_indices.dim() in (1, 2)
        if has_initial_state is not None:
            assert has_initial_state.size(0) == query_start_loc.size(0) - 1
        assert weight.stride(1) == 1
        assert weight.size(0) == x.size(0)
        if block_size_to_align is not None and block_size_to_align > 0:
            assert block_size_to_align % 8 == 0

    original_x_dtype = x.dtype
    conv_x = x.to(conv_states.dtype)
    simple = (
        cache_indices is not None
        and has_initial_state is not None
        and query_start_loc.numel() == 2
        and block_idx_first_scheduled_token is None
        and block_idx_last_scheduled_token is None
        and activation in ("silu", "swish")
        and conv_x.shape[1] >= 1024
    )
    if simple:
        output = fast_causal_conv1d(
            conv_x,
            weight,
            bias,
            conv_states,
            cache_indices,
            has_initial_state,
        )
        if output is not None:
            return output.to(original_x_dtype)

    apc_requested = any(
        value is not None
        for value in (
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
            initial_state_idx,
            num_computed_tokens,
        )
    )
    batch = query_start_loc.numel() - 1
    use_tiled = (
        _IS_SM103
        and conv_x.is_cuda
        and conv_x.shape[1] >= 1024
        and conv_x.shape[0] <= 4096
        and batch <= 32
        and conv_x.shape[1] // max(batch, 1) >= 128
    )
    if not apc_requested and use_tiled:
        generated_cache_indices = cache_indices is None
        if cache_indices is None:
            cache_indices = torch.arange(batch, dtype=torch.int32, device=x.device)
        if has_initial_state is None:
            has_initial_state = torch.zeros(batch, dtype=torch.bool, device=x.device)
        return generic_causal_conv1d(
            conv_x,
            weight,
            bias,
            conv_states,
            query_start_loc,
            cache_indices,
            has_initial_state,
            activation,
            pad_slot_id,
            pad_slot_id - 1 if generated_cache_indices else null_block_id,
        ).to(original_x_dtype)

    if not apc_requested or not use_tiled:
        return official_causal_conv1d_fn(
            x,
            weight,
            bias,
            conv_states,
            query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation=activation,
            pad_slot_id=pad_slot_id,
            null_block_id=null_block_id,
            block_idx_first_scheduled_token=block_idx_first_scheduled_token,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            initial_state_idx=initial_state_idx,
            num_computed_tokens=num_computed_tokens,
            block_size_to_align=block_size_to_align,
            metadata=metadata,
            validate_data=validate_data,
        )

    assert cache_indices is not None
    assert has_initial_state is not None
    assert block_idx_first_scheduled_token is not None
    assert block_idx_last_scheduled_token is not None
    assert initial_state_idx is not None
    assert num_computed_tokens is not None
    return apc_causal_conv1d(
        conv_x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation,
        pad_slot_id,
        null_block_id,
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        initial_state_idx,
        num_computed_tokens,
        block_size_to_align,
    ).to(original_x_dtype)
