# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm._custom_ops import causal_conv1d_update_cpu_vec
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID


def causal_conv1d_fn_cpu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    **kwargs,
) -> torch.Tensor:
    """CPU implementation for causal_conv1d_fwd."""
    if isinstance(activation, bool) and activation:
        activation = "silu"
    elif isinstance(activation, bool):
        activation = None

    original_x_dtype = x.dtype
    x = x.to(conv_states.dtype)

    out = torch.empty_like(x)
    state_len = weight.shape[1] - 1
    assert activation in {None, "silu", "swish"}

    seq_begin_end_idx = [
        (int(query_start_loc[idx].item()), int(query_start_loc[idx + 1].item()))
        for idx in range(query_start_loc.shape[0] - 1)
    ]
    weight = weight.unsqueeze(1)

    for seq_idx, (bos, eos) in enumerate(seq_begin_end_idx):
        if bos == eos:
            continue

        slot = (
            int(cache_indices[seq_idx].item()) if cache_indices is not None else seq_idx
        )

        if slot == pad_slot_id:
            continue

        seq_x = x[:, bos:eos].unsqueeze(0)

        if has_initial_state is not None and bool(has_initial_state[seq_idx].item()):
            initial_state = conv_states[slot, :, :state_len].unsqueeze(0)
        else:
            initial_state = torch.zeros(
                1,
                weight.shape[0],
                state_len,
                device=seq_x.device,
                dtype=seq_x.dtype,
            )

        conv_input = torch.cat([initial_state, seq_x], dim=-1).to(weight.dtype)
        seq_out = F.conv1d(
            conv_input,
            weight,
            bias,
            padding=0,
            groups=weight.shape[0],
        )
        seq_out = seq_out[..., -seq_x.shape[-1] :].to(dtype=x.dtype)

        if activation in ("silu", "swish"):
            seq_out = F.silu(seq_out)

        out[:, bos:eos] = seq_out.squeeze(0)
        conv_states[slot, :, :state_len].copy_(conv_input[..., -state_len:].squeeze(0))

    return out.to(original_x_dtype)


def causal_conv1d_update_cpu(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    pad_slot_id: int | None = None,
    **kwargs,
) -> torch.Tensor:
    """CPU implementation for causal_conv1d_update."""
    if isinstance(activation, bool):
        activation = "silu" if activation else None

    if pad_slot_id is None:
        pad_slot_id = kwargs.get("null_block_id", NULL_BLOCK_ID)
        if pad_slot_id is None:
            pad_slot_id = NULL_BLOCK_ID

    return causal_conv1d_update_cpu_vec(
        x,
        conv_state,
        weight,
        bias,
        activation,
        conv_state_indices,
        query_start_loc,
        pad_slot_id,
    )


def causal_conv1d_update_torch(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """
    Pure PyTorch fallback for causal_conv1d_update.
    Currently used as a fallback for Arm (aarch64) to leverage
    oneDNN/ACL F.conv1d kernels for batched decoding.
    """
    assert activation in {None, "silu", "swish"}

    _, dim, seq_len = x.shape
    state_len = conv_state.shape[-1]

    x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
    conv_state.copy_(x_new[:, :, -state_len:])

    out = F.conv1d(
        x_new,
        weight.unsqueeze(1),
        bias,
        padding=0,
        groups=dim,
    )[:, :, -seq_len:]
    if activation in ("silu", "swish"):
        out = F.silu(out)
    return out
