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
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    num_computed_tokens: torch.Tensor | None = None,
    block_size_to_align: int | None = None,
    **kwargs,
) -> torch.Tensor:
    """CPU implementation for causal_conv1d_fwd.

    Supports both the legacy layout (1-D ``cache_indices``: one state slot per
    sequence) and the mamba-prefix-caching layout (2-D ``cache_indices``: a
    per-sequence block table). In the block-table layout, mirroring the GPU
    kernel contract documented in ``mamba_mixer2.conv_ssm_forward``:

      * the initial state is read from
        ``cache_indices[seq, initial_state_idx[seq]]``,
      * the end-of-step state is written to
        ``cache_indices[seq, block_idx_last_scheduled_token[seq]]``,
      * an additional snapshot is written at every ``block_size_to_align``
        token boundary crossed by this step, into that boundary's block slot,
        so prefix-cache hits can restore conv state at any block boundary.
    """
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

    block_table_mode = cache_indices is not None and cache_indices.dim() == 2

    for seq_idx, (bos, eos) in enumerate(seq_begin_end_idx):
        if bos == eos:
            continue

        if cache_indices is None:
            slot = seq_idx
            block_row = None
        elif block_table_mode:
            block_row = cache_indices[seq_idx]
            last_block = (
                int(block_idx_last_scheduled_token[seq_idx].item())
                if block_idx_last_scheduled_token is not None
                else 0
            )
            slot = int(block_row[last_block].item())
        else:
            block_row = None
            slot = int(cache_indices[seq_idx].item())

        # Block 0 is the reserved null block only in the block-table layout;
        # in the legacy layout slot 0 is a valid state slot.
        if slot == pad_slot_id or (block_table_mode and slot == NULL_BLOCK_ID):
            continue

        seq_x = x[:, bos:eos].unsqueeze(0)

        if has_initial_state is not None and bool(has_initial_state[seq_idx].item()):
            if block_table_mode and initial_state_idx is not None:
                init_slot = int(block_row[int(initial_state_idx[seq_idx].item())].item())
            else:
                init_slot = slot
            initial_state = conv_states[init_slot, :, :state_len].unsqueeze(0)
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

        # Aligned boundary snapshots (block-table mode only). A boundary at
        # global token position P (multiple of block_size_to_align) crossed by
        # this step gets the conv window ending at P: conv_input[t : t+state_len]
        # where t is P local to this step's tokens.
        if (
            block_table_mode
            and block_size_to_align is not None
            and block_size_to_align > 0
        ):
            computed = (
                int(num_computed_tokens[seq_idx].item())
                if num_computed_tokens is not None
                else 0
            )
            seqlen = eos - bos
            first_boundary = (
                (computed + block_size_to_align) // block_size_to_align
            ) * block_size_to_align
            for pos in range(first_boundary, computed + seqlen, block_size_to_align):
                t_local = pos - computed
                boundary_block = pos // block_size_to_align - 1
                if 0 <= boundary_block < block_row.shape[0]:
                    b_slot = int(block_row[boundary_block].item())
                    if b_slot not in (pad_slot_id, NULL_BLOCK_ID) and b_slot >= 0:
                        conv_states[b_slot, :, :state_len].copy_(
                            conv_input[..., t_local : t_local + state_len]
                            .squeeze(0)
                            .to(conv_states.dtype)
                        )

        conv_states[slot, :, :state_len].copy_(
            conv_input[..., -state_len:].squeeze(0)
        )

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
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """CPU implementation for causal_conv1d_update."""
    if isinstance(activation, bool):
        activation = "silu" if activation else None

    if pad_slot_id is None:
        pad_slot_id = kwargs.get("null_block_id", NULL_BLOCK_ID)
        if pad_slot_id is None:
            pad_slot_id = NULL_BLOCK_ID

    if (
        conv_state_indices is not None
        and conv_state_indices.dim() == 2
        and block_idx_last_scheduled_token is None
    ):
        # Without prefix caching each sequence owns exactly one mamba state
        # block, so the block table is (batch, 1): flatten to the legacy
        # one-slot-per-sequence layout the C++ vec kernel expects.
        conv_state_indices = conv_state_indices[:, 0].contiguous()

    if conv_state_indices is not None and conv_state_indices.dim() == 2:
        # Mamba prefix caching: a per-sequence BLOCK TABLE plus pointers.
        # The C++ vec kernel only understands one slot per sequence, so
        # resolve the table here: the running conv state is read from the
        # ``initial_state_idx`` block and lives at the
        # ``block_idx_last_scheduled_token`` block afterwards. On a block
        # transition (read != write slot), migrate the state first, then
        # let the kernel read+write in place at the write slot.
        write_idx = (
            conv_state_indices.gather(
                1, block_idx_last_scheduled_token.to(torch.int64).unsqueeze(1)
            )
            .squeeze(1)
            .to(torch.int32)
        )
        if initial_state_idx is not None:
            read_idx = (
                conv_state_indices.gather(
                    1, initial_state_idx.to(torch.int64).unsqueeze(1)
                )
                .squeeze(1)
                .to(torch.int32)
            )
            moved = (read_idx != write_idx) & (write_idx != pad_slot_id)
            for s in torch.nonzero(moved).flatten().tolist():
                r, w = int(read_idx[s].item()), int(write_idx[s].item())
                if r != pad_slot_id and r != NULL_BLOCK_ID and r >= 0:
                    conv_state[w].copy_(conv_state[r])
        conv_state_indices = write_idx

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
