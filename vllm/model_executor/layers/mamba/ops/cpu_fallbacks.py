# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID


def _causal_conv1d_fn_cpu(
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
    """Pure PyTorch CPU fallback for causal_conv1d_fwd."""
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

        if seq_len >= state_len:
            conv_states[cache_idx, :, :state_len] = x_seq[:, -state_len:]
        else:
            conv_states[cache_idx, :, : state_len - seq_len] = conv_states[
                cache_idx, :, seq_len:state_len
            ].clone()
            conv_states[cache_idx, :, state_len - seq_len :] = x_seq

    return out.to(original_x_dtype)


def _causal_conv1d_update_cpu(
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
    """Pure PyTorch CPU fallback for causal_conv1d_update (decode path)."""
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


def _selective_state_update_cpu(
    state,
    x,
    dt,
    A,
    B,
    C,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    state_batch_indices=None,
    dst_state_batch_indices=None,
    null_block_id=NULL_BLOCK_ID,
    out=None,
    num_accepted_tokens=None,
    cu_seqlens=None,
    is_blackwell=False,
    enable_stochastic_rounding=False,
    cache_philox_rounds=0,
    **kwargs,
):
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    if out.dim() == 2:
        out = out.unsqueeze(1)
    if state_batch_indices is not None and state_batch_indices.dim() == 1:
        state_batch_indices = state_batch_indices.unsqueeze(1)
    if dst_state_batch_indices is not None and dst_state_batch_indices.dim() == 1:
        dst_state_batch_indices = dst_state_batch_indices.unsqueeze(1)

    _, nheads, dim, dstate = state.shape
    batch = x.shape[0]
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else batch

    ngroups = B.shape[1]
    nheads_ngroups_ratio = nheads // ngroups

    for seq_idx in range(N):
        if cu_seqlens is not None:
            bos = cu_seqlens[seq_idx].item()
            seq_len = cu_seqlens[seq_idx + 1].item() - bos
        else:
            bos = seq_idx
            seq_len = 1

        if state_batch_indices is not None:
            state_idx = state_batch_indices[seq_idx, 0].item()
            if state_idx == null_block_id:
                continue
        else:
            state_idx = seq_idx

        if num_accepted_tokens is None:
            if dst_state_batch_indices is not None:
                dst_idx = dst_state_batch_indices[seq_idx, 0].item()
            else:
                dst_idx = state_idx

        s = state[state_idx].float()

        for t in range(seq_len):
            token_idx = bos + t

            x_val = x[token_idx].float()
            dt_val = dt[token_idx].float()

            if dt_bias is not None:
                dt_val = dt_val + dt_bias.float()
            if dt_softplus:
                dt_val = torch.nn.functional.softplus(dt_val)

            A_val = A.float()

            B_val = B[token_idx].float()
            B_expanded = B_val.repeat_interleave(nheads_ngroups_ratio, dim=0)
            C_val = C[token_idx].float()
            C_expanded = C_val.repeat_interleave(nheads_ngroups_ratio, dim=0)

            dA = torch.exp(A_val * dt_val.unsqueeze(-1))
            dBx = B_expanded.unsqueeze(1) * (x_val * dt_val).unsqueeze(-1)
            s = s * dA + dBx

            if num_accepted_tokens is not None:
                token_dst_idx = dst_state_batch_indices[seq_idx, t].item()
                if token_dst_idx != null_block_id:
                    state[token_dst_idx] = s.to(state.dtype)

            out_val = (s * C_expanded.unsqueeze(1)).sum(dim=-1)

            if D is not None:
                out_val = out_val + x_val * D.float()

            if z is not None:
                z_val = z[token_idx].float()
                out_val = out_val * z_val * torch.sigmoid(z_val)

            out[token_idx] = out_val.to(out.dtype)

        if num_accepted_tokens is None and dst_idx != null_block_id:
            state[dst_idx] = s.to(state.dtype)


def _mamba_chunk_scan_combined_fwd_cpu(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    out,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    return_intermediate_states=False,
    seq_idx=None,
    cu_seqlens=None,
    cu_chunk_seqlens=None,
    last_chunk_indices=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    state_dtype=None,
    **kwargs,
):
    seqlen, nheads, headdim = x.shape
    _, ngroups, dstate = B.shape
    nheads_per_group = nheads // ngroups

    assert cu_seqlens is not None
    batch = cu_seqlens.size(0) - 1

    dt_f = dt.float()
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.float().unsqueeze(0)
    if dt_softplus:
        dt_f = torch.nn.functional.softplus(dt_f)
    if dt_limit[0] > 0.0 or dt_limit[1] < float("inf"):
        dt_f = dt_f.clamp(min=dt_limit[0], max=dt_limit[1])

    all_states = torch.zeros(
        batch, nheads, headdim, dstate, dtype=torch.float32, device=x.device
    )

    for b_idx in range(batch):
        seq_start = cu_seqlens[b_idx].item()
        seq_end = cu_seqlens[b_idx + 1].item()

        if initial_states is not None:
            state = initial_states[b_idx].float()
        else:
            state = torch.zeros(
                nheads, headdim, dstate, dtype=torch.float32, device=x.device
            )

        for t in range(seq_start, seq_end):
            x_t = x[t].float()
            dt_t = dt_f[t]
            A_val = A.float()

            dA = torch.exp(A_val * dt_t).unsqueeze(-1).unsqueeze(-1)

            B_expanded = B[t].float().repeat_interleave(nheads_per_group, dim=0)
            C_expanded = C[t].float().repeat_interleave(nheads_per_group, dim=0)

            xdt = x_t * dt_t.unsqueeze(-1)
            dBx = xdt.unsqueeze(-1) * B_expanded.unsqueeze(1)
            state = state * dA + dBx

            y = (state * C_expanded.unsqueeze(1)).sum(dim=-1)

            if D is not None:
                y = (
                    y + x_t * D.float().unsqueeze(-1)
                    if D.dim() == 1
                    else y + x_t * D.float()
                )

            if z is not None:
                z_t = z[t].float()
                y = y * z_t * torch.sigmoid(z_t)

            out[t] = y.to(out.dtype)

        all_states[b_idx] = state.to(all_states.dtype)

    out_dtype = state_dtype if state_dtype is not None else x.dtype
    all_states = all_states.to(out_dtype)

    return all_states
