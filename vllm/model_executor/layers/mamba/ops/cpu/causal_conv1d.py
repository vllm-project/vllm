# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
import torch.nn.functional as F

import vllm._custom_ops as ops
from vllm.platforms import CpuArchEnum, current_platform
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID


def resolve_cpu_conv_weights(
    conv: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return plain fallback weights and an optional packed native weight."""
    plain_weight = getattr(conv, "_cpu_unpacked_conv_weight", None)
    if plain_weight is None:
        plain_weight = conv.weight.flatten(start_dim=1)
        return plain_weight, None
    return plain_weight, conv.weight


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
    native_weight: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """CPU implementation for causal_conv1d_fwd."""
    if isinstance(activation, bool) and activation:
        activation = "silu"
    elif isinstance(activation, bool):
        activation = None

    if native_weight is not None and _can_use_native_fwd(x, conv_states):
        return ops.causal_conv1d_fwd_cpu(
            x=x,
            weight=native_weight,
            bias=bias,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            silu_activation=activation in ("silu", "swish"),
            is_vnni=True,
        )

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
    num_accepted_tokens: torch.Tensor | None = None,
    native_weight: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """CPU implementation for causal_conv1d_update."""
    if isinstance(activation, bool):
        activation = "silu" if activation else None

    native_x = _prepare_native_update_input(
        x=x,
        weight=weight,
        conv_state=conv_state,
        conv_state_indices=conv_state_indices,
        query_start_loc=query_start_loc,
        num_accepted_tokens=num_accepted_tokens,
        native_weight=native_weight,
    )
    if native_x is not None:
        return ops.causal_conv1d_update_cpu(
            x=native_x,
            conv_states=conv_state,
            weight=native_weight,
            bias=bias,
            silu_activation=activation in ("silu", "swish"),
            conv_state_indices=conv_state_indices,
            is_vnni=True,
            num_accepted_tokens=num_accepted_tokens,
        )

    if num_accepted_tokens is not None:
        if query_start_loc is None:
            raise ValueError(
                "query_start_loc is required for accepted-token causal "
                "convolution fallback."
            )
        return _causal_conv1d_update_ragged_torch(
            x=x,
            conv_state=conv_state,
            weight=weight,
            bias=bias,
            activation=activation,
            conv_state_indices=conv_state_indices,
            query_start_loc=query_start_loc,
            pad_slot_id=_resolve_pad_slot_id(pad_slot_id, kwargs),
            num_accepted_tokens=num_accepted_tokens,
        )

    if _can_use_arm_torch_update(x, conv_state, weight):
        return _causal_conv1d_update_arm_torch(
            x=x,
            conv_state=conv_state,
            weight=weight,
            bias=bias,
            activation=activation,
            conv_state_indices=conv_state_indices,
        )

    return ops.causal_conv1d_update_cpu_vec(
        x,
        conv_state,
        weight,
        bias,
        activation,
        conv_state_indices,
        query_start_loc,
        _resolve_pad_slot_id(pad_slot_id, kwargs),
    )


def _can_use_native_conv(conv_state: torch.Tensor) -> bool:
    return (
        torch.cpu._is_amx_tile_supported()
        and conv_state.dim() == 3
        and conv_state.stride(-2) == 1
        and conv_state.stride(-1) == conv_state.size(1)
    )


def _can_use_native_fwd(x: torch.Tensor, conv_state: torch.Tensor) -> bool:
    return (
        _can_use_native_conv(conv_state)
        and x.dim() == 2
        and x.stride(-2) == 1
        and x.stride(-1) == x.size(-2)
    )


def _prepare_native_update_input(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    conv_state_indices: torch.Tensor | None,
    query_start_loc: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    native_weight: torch.Tensor | None,
) -> torch.Tensor | None:
    if native_weight is None or not _can_use_native_conv(conv_state):
        return None
    if num_accepted_tokens is None:
        return x
    if x.dim() == 3:
        return x
    if (
        x.dim() != 2
        or query_start_loc is None
        or conv_state_indices is None
        or weight.size(1) != 4
    ):
        return None

    num_sequences = query_start_loc.numel() - 1
    if (
        num_sequences <= 0
        or conv_state_indices.numel() != num_sequences
        or num_accepted_tokens.numel() != num_sequences
        or int(query_start_loc[0].item()) != 0
        or int(query_start_loc[-1].item()) != x.size(0)
    ):
        return None

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    if int(query_lens[0].item()) <= 0 or not bool(
        torch.all(query_lens == query_lens[0]).item()
    ):
        return None
    return x.view(num_sequences, int(query_lens[0].item()), x.size(1))


def _can_use_arm_torch_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    return (
        current_platform.get_cpu_architecture() == CpuArchEnum.ARM
        and x.dim() == 2
        and x.size(1) == weight.size(0)
        and conv_state.dim() == 3
    )


def _causal_conv1d_update_arm_torch(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    conv_state_indices: torch.Tensor | None,
) -> torch.Tensor:
    if conv_state_indices is None:
        conv_state_indices = torch.arange(x.size(0), device=x.device)
    state = conv_state[conv_state_indices].contiguous()
    out = causal_conv1d_update_torch(
        x=x.unsqueeze(-1),
        conv_state=state,
        weight=weight,
        bias=bias,
        activation=activation,
    ).squeeze(-1)
    conv_state[conv_state_indices] = state
    return out


def _resolve_pad_slot_id(pad_slot_id: int | None, kwargs: dict[str, int | None]) -> int:
    if pad_slot_id is not None:
        return pad_slot_id
    null_block_id = kwargs.get("null_block_id", NULL_BLOCK_ID)
    return NULL_BLOCK_ID if null_block_id is None else int(null_block_id)


def _causal_conv1d_update_ragged_torch(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    conv_state_indices: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    pad_slot_id: int,
    num_accepted_tokens: torch.Tensor,
) -> torch.Tensor:
    """Apply causal convolution to flat, variable-length decode queries."""
    if x.dim() != 2:
        raise ValueError("ragged causal convolution expects x with shape [T, D]")
    if conv_state.dim() != 3:
        raise ValueError("ragged causal convolution expects conv_state with rank 3")
    if weight.dim() != 2:
        raise ValueError("ragged causal convolution expects weight with rank 2")
    if activation not in {None, "silu", "swish"}:
        raise ValueError(f"unsupported activation: {activation}")

    num_sequences = query_start_loc.numel() - 1
    if num_sequences <= 0:
        raise ValueError("query_start_loc must describe at least one sequence")
    if conv_state_indices is None or conv_state_indices.numel() != num_sequences:
        raise ValueError("conv_state_indices must match query_start_loc")
    if num_accepted_tokens.numel() != num_sequences:
        raise ValueError("num_accepted_tokens must match query_start_loc")
    if int(query_start_loc[0].item()) != 0 or int(query_start_loc[-1].item()) != x.size(
        0
    ):
        raise ValueError("query_start_loc must span x")

    state_len = conv_state.size(-1)
    width = weight.size(1)
    history_len = width - 1
    if conv_state.size(1) != x.size(1) or weight.size(0) != x.size(1):
        raise ValueError("incompatible convolution dimensions")
    if history_len <= 0 or state_len < history_len:
        raise ValueError("invalid convolution state length")

    out = torch.empty_like(x)
    for seq_idx in range(num_sequences):
        begin = int(query_start_loc[seq_idx].item())
        end = int(query_start_loc[seq_idx + 1].item())
        if begin > end:
            raise ValueError("query_start_loc must be nondecreasing")
        if begin == end:
            continue

        slot = int(conv_state_indices[seq_idx].item())
        if slot == pad_slot_id:
            out[begin:end].zero_()
            continue
        if slot < 0 or slot >= conv_state.size(0):
            raise ValueError("conv_state_indices contains an invalid slot")

        seq_len = end - begin
        num_accepted = int(num_accepted_tokens[seq_idx].item())
        if not 1 <= num_accepted <= seq_len or seq_len > state_len:
            raise ValueError("invalid accepted-token count or query length")

        state = conv_state[slot]
        x_seq = x[begin:end].transpose(0, 1).to(state.dtype)
        offset = num_accepted - 1
        prior = state[:, offset : offset + history_len]
        keep = state[:, offset + 1 : offset + 1 + (state_len - seq_len)]
        conv_in = torch.cat([prior, x_seq], dim=-1).unsqueeze(0).to(weight.dtype)
        seq_out = F.conv1d(
            conv_in, weight.unsqueeze(1), bias, padding=0, groups=x.size(1)
        )[0]
        if activation in ("silu", "swish"):
            seq_out = F.silu(seq_out)
        out[begin:end] = seq_out.transpose(0, 1).to(out.dtype)
        state.copy_(torch.cat([keep, x_seq], dim=-1))

    return out


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
