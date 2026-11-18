# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba2 prefill SSD scan through FlashInfer's fused SSD (Cake backend).

Runs the chunked scan of `mamba_chunk_scan_combined_varlen` in two kernel
launches instead of five, with the packed-varlen `cu_seqlens` form (the
kernels derive their chunk metadata on the device).
"""

import torch

from vllm.model_executor.layers.mamba.ops.scatter_states import scatter_states

# Shapes compiled into FlashInfer's Cake SSD programs.
FLASHINFER_SSD_HEAD_DIM = 64
FLASHINFER_SSD_STATE_SIZE = 128


def mamba_chunk_scan_flashinfer(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    initial_states: torch.Tensor,
    checkpoint_token_indices: torch.Tensor | None = None,
    checkpoint_state_slots: torch.Tensor | None = None,
    checkpoint_states: torch.Tensor | None = None,
) -> None:
    """Scan the prefill tokens, write each sequence's final state, and
    optionally capture a checkpoint state inside each sequence.

    x: (seqlen, nheads, headdim) bf16, may be a strided view
    dt: (seqlen, nheads)
    A, dt_bias: (nheads,) fp32
    B, C: (seqlen, ngroups, dstate) bf16, may be strided views
    D: (nheads,)
    cu_seqlens: (batch + 1,) int32, starting at 0
    out: (seqlen, nheads, headdim) bf16, contiguous; written in place
    ssm_state: (num_slots, nheads, headdim, dstate) state cache
    state_indices: (batch,) cache slot of each sequence
    initial_states: (batch, nheads, headdim, dstate), zeros for sequences
        without a cached state; FlashInfer takes the state dtype from it.
    checkpoint_token_indices: (batch,) int32 packed token whose preceding
        state each sequence checkpoints, -1 for none
    checkpoint_state_slots: (batch,) int32 row of `checkpoint_states` each
        sequence writes
    checkpoint_states: (num_checkpoints, nheads, headdim, dstate); written in
        place
    """
    from flashinfer.mamba import ssd_combined_fwd

    _, final_states = ssd_combined_fwd(
        x.unsqueeze(0),
        dt.unsqueeze(0),
        A,
        B.unsqueeze(0),
        C.unsqueeze(0),
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        initial_states=initial_states,
        cu_seqlens=cu_seqlens,
        checkpoint_token_indices=checkpoint_token_indices,
        checkpoint_state_slots=checkpoint_state_slots,
        checkpoint_states=checkpoint_states,
        out=out.unsqueeze(0),
    )
    scatter_states(ssm_state, final_states, state_indices)
