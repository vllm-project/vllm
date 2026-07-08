# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm._custom_ops as ops
from vllm.v1.attention.backends.utils import PAD_SLOT_ID


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

    assert cu_seqlens is not None
    batch = cu_seqlens.size(0) - 1

    # Preprocess dt: apply bias, softplus, and clamp.
    # Kept in Python to avoid duplicating this logic in C++.
    dt_f = dt.float()
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.float().unsqueeze(0)
    if dt_softplus:
        dt_f = torch.nn.functional.softplus(dt_f)
    if dt_limit[0] > 0.0 or dt_limit[1] < float("inf"):
        dt_f = dt_f.clamp(min=dt_limit[0], max=dt_limit[1])

    # Allocate state output buffer (float32, contiguous — required by kernel).
    all_states = torch.zeros(
        batch, nheads, headdim, dstate, dtype=torch.float32, device=x.device
    )
    if initial_states is not None:
        all_states.copy_(initial_states.float())

    # Use the C++ kernel when available (CPU build with mamba kernels compiled).
    # Falls back to the pure-Python loop below if the op is not registered.
    _use_cpp_kernel = hasattr(torch.ops._C, "mamba_chunk_scan_fwd_cpu")

    if _use_cpp_kernel:
        # out must be contiguous — the C++ kernel writes via raw data_ptr().
        # Creating a contiguous copy here would discard the results, so we
        # require the caller to pass a contiguous tensor.
        assert out.is_contiguous(), (
            "_mamba_chunk_scan_combined_fwd_cpu: `out` must be "
            "pre-allocated as a contiguous tensor"
        )

        # D: strip trailing dims that are broadcast (stride == 0) so the
        # kernel sees a (nheads,) float32 array.  Only peel dims whose
        # stride is 0 (broadcast-expanded); genuine multi-dim D is passed
        # as-is and flattened by the C++ wrapper.
        D_1d = None
        if D is not None:
            d = D.float()
            while d.dim() > 1 and d.stride(-1) == 0:
                d = d.squeeze(-1)
            D_1d = d.contiguous()

        ops.mamba_chunk_scan_fwd_cpu(
            out,
            all_states,
            x,
            dt_f,
            A,
            B,
            C,
            D_1d,
            z,
            cu_seqlens.to(torch.int32),
        )
    else:
        # Pure-Python fallback (no compiled extension available).
        for b_idx in range(batch):
            seq_start = cu_seqlens[b_idx].item()
            seq_end = cu_seqlens[b_idx + 1].item()

            # Note: basic indexing returns a view, but arithmetic ops
            # below (state * dA + dBx) produce new tensors, so state is
            # reassigned each token. all_states[b_idx] is committed at line 416.
            state = all_states[b_idx].clone()  # local working copy

            for t in range(seq_start, seq_end):
                x_t = x[t].float()
                dt_t = dt_f[t]
                A_val = A.float()

                dA = torch.exp(A_val * dt_t).unsqueeze(-1).unsqueeze(-1)

                B_expanded = B[t].float().repeat_interleave(nheads // ngroups, dim=0)
                C_expanded = C[t].float().repeat_interleave(nheads // ngroups, dim=0)

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
