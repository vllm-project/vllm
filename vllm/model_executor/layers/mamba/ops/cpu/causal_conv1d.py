# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm.v1.attention.backends.utils import PAD_SLOT_ID


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

    original_x_dtype = x.dtype
    x = x.to(conv_states.dtype)

    dim, cu_seqlen = x.shape
    _, width = weight.shape
    state_len = width - 1

    out = torch.zeros_like(x)

    batch = query_start_loc.size(0) - 1

    for b in range(batch):
        seq_start = query_start_loc[b].item()
        seq_end = query_start_loc[b + 1].item()
        seq_len = seq_end - seq_start

        if seq_len == 0:
            continue

        cache_idx = cache_indices[b].item() if cache_indices is not None else b

        if cache_idx == pad_slot_id:
            continue

        x_seq = x[:, seq_start:seq_end]  # (dim, seq_len)

        if has_initial_state is not None and has_initial_state[b]:
            state = conv_states[cache_idx].clone()  # (dim, state_len)
        else:
            state = torch.zeros((dim, state_len), dtype=x.dtype, device=x.device)

        for t in range(seq_len):
            x_t = x_seq[:, t]  # (dim,)

            window = torch.cat([state, x_t.unsqueeze(1)], dim=1)  # (dim, width)
            val = (window * weight).sum(dim=1)  # (dim,)

            if bias is not None:
                val = val + bias
            if activation in ["silu", "swish"]:
                val = val * torch.sigmoid(val)

            out[:, seq_start + t] = val

            if state_len > 1:
                state[:, :-1] = state[:, 1:].clone()
            state[:, -1] = x_t

        conv_states[cache_idx].copy_(state)

    return out.to(original_x_dtype)


def causal_conv1d_update_cpu(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
    **kwargs,
) -> torch.Tensor:
    """CPU implementation for causal_conv1d_update."""
    if isinstance(activation, bool):
        activation = "silu" if activation else None

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    _, width = weight.shape
    state_len = width - 1

    if query_start_loc is None and x.dim() == 2:
        x = x.unsqueeze(-1)
        unsqueeze = True
    else:
        unsqueeze = False

    if query_start_loc is None:
        batch, dim, seqlen = x.shape

        if conv_state_indices is not None:
            cache_idxs = conv_state_indices.flatten()
            valid_mask = cache_idxs != pad_slot_id
        else:
            cache_idxs = torch.arange(batch, device=x.device)
            valid_mask = torch.ones(batch, dtype=torch.bool, device=x.device)

        for t in range(seqlen):
            x_t = x[:, :, t].clone()

            states = conv_state[cache_idxs]

            windows = torch.cat([states, x_t.unsqueeze(-1)], dim=-1)

            val = (windows * weight.unsqueeze(0)).sum(dim=-1)
            if bias is not None:
                val = val + bias.unsqueeze(0)

            if activation in ["silu", "swish"]:
                val = val * torch.sigmoid(val)

            val = val * valid_mask.unsqueeze(-1).to(val.dtype)
            x[:, :, t] = val

            new_state = torch.cat([states[:, :, 1:], x_t.unsqueeze(-1)], dim=-1)
            conv_state[cache_idxs[valid_mask]] = new_state[valid_mask]

        out = x
        if unsqueeze:
            out = out.squeeze(-1)
        return out.to(original_x_dtype)

    assert conv_state_indices is not None
    assert query_start_loc is not None
    batch = conv_state_indices.size(0)
    out = x.clone()

    for b in range(batch):
        cache_idx = conv_state_indices[b].item()
        if cache_idx == pad_slot_id:
            continue

        seq_start = query_start_loc[b].item()
        seq_end = query_start_loc[b + 1].item()
        seqlen_b = seq_end - seq_start

        if seqlen_b == 0:
            continue

        local_state = conv_state[cache_idx].clone()

        for t in range(seqlen_b):
            x_t = x[seq_start + t, :]

            window = torch.cat([local_state, x_t.unsqueeze(-1)], dim=-1)
            val = (window * weight).sum(dim=-1)
            if bias is not None:
                val = val + bias
            if activation in ["silu", "swish"]:
                val = val * torch.sigmoid(val)

            out[seq_start + t, :] = val

            if state_len > 1:
                local_state[:, :-1] = local_state[:, 1:].clone()
            local_state[:, -1] = x_t

        conv_state[cache_idx] = local_state

    return out.to(original_x_dtype)



def causal_conv1d_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    activation: str | None = "silu",
) -> torch.Tensor:
    out = torch.empty_like(x)
    state_len = weight.shape[1] - 1
    assert activation in {None, "silu", "swish"}

    seq_begin_end_idx = [
        (int(query_start_loc[idx].item()), int(query_start_loc[idx + 1].item()))
        for idx in range(query_start_loc.shape[0] - 1)
    ]
    weight = weight.unsqueeze(1)
    for seq_idx, (bos, eos) in enumerate(seq_begin_end_idx):
        slot = int(cache_indices[seq_idx].item())

        seq_x = x[:, bos:eos].unsqueeze(0)
        if bool(has_initial_state[seq_idx].item()):
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

    return out


def causal_conv1d_update_torch(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
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
