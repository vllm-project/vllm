# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba2 prefill SSD scan through FlashInfer's fused SSD (Cake backend).

Runs the chunked scan of ``mamba_chunk_scan_combined_varlen`` in two kernel
launches instead of five, with the packed-varlen ``cu_seqlens`` form (the
kernels derive their chunk metadata on the device).
"""

from functools import lru_cache

import torch

from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.model_executor.layers.mamba.ops.scatter_states import scatter_states

# Shapes compiled into FlashInfer's Cake SSD programs.
FLASHINFER_SSD_HEAD_DIM = 64
FLASHINFER_SSD_STATE_SIZE = 128


@lru_cache(maxsize=16)
def get_ssd_runner(
    device_index: int,
    stream_id: int,
    nheads: int,
    ngroups: int,
    state_dtype: torch.dtype,
    has_initial_states: bool,
):
    # Runner workspaces are mutable, so like flashinfer.mamba.ssd_combined_fwd
    # keep one runner per stream.
    from flashinfer.mamba import SSDCombined

    with torch.accelerator.device_index(device_index):
        return SSDCombined(
            chunk_size=128,
            nheads=nheads,
            headdim=FLASHINFER_SSD_HEAD_DIM,
            dstate=FLASHINFER_SSD_STATE_SIZE,
            ngroups=ngroups,
            io_dtype=torch.bfloat16,
            state_dtype=state_dtype,
            has_d=True,
            d_has_hdim=False,
            has_initial_states=has_initial_states,
            has_varlen=True,
            has_z=False,
            backend="cake",
        )


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
    has_initial_states: torch.Tensor | None,
) -> None:
    """Scan the prefill tokens and write each sequence's final state.

    x: (seqlen, nheads, headdim) bf16, may be a strided view
    dt: (seqlen, nheads)
    A, dt_bias: (nheads,) fp32
    B, C: (seqlen, ngroups, dstate) bf16, may be strided views
    D: (nheads,) bf16
    cu_seqlens: (batch + 1,) int32, starting at 0
    out: (seqlen, nheads, headdim) bf16, contiguous; written in place
    ssm_state: (num_slots, nheads, headdim, dstate) state cache
    state_indices: (batch,) cache slot of each sequence
    has_initial_states: (batch,) bool, or None when no sequence has a state
    """
    initial_states = (
        gather_initial_states(ssm_state, state_indices, has_initial_states)
        if has_initial_states is not None
        else None
    )
    runner = get_ssd_runner(
        x.device.index,
        torch.accelerator.current_stream(x.device).stream_id,
        x.size(1),
        B.size(1),
        ssm_state.dtype,
        initial_states is not None,
    )
    _, final_states = runner.run(
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
        out=out.unsqueeze(0),
        return_final_states=True,
    )
    scatter_states(ssm_state, final_states, state_indices)
