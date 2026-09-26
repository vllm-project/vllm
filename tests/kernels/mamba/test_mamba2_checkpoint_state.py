# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An SSD intermediate state equals an independent run over the same prefix.

This is the central claim of internal prefill checkpoints: the state the SSD
scan already materializes at the checkpoint's chunk boundary is exactly the
state a fresh prefill over `[0, checkpoint)` would end with, so it can be
cached and later resumed from. The last test covers the other half: the
shared exporter moving that state, and the conv window, into the paged cache.
"""

import pytest
import torch

from tests.kernels.mamba.test_mamba_ssm_ssd import generate_random_inputs
from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointExporter,
    MambaPrefillCheckpointMetadata,
)
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

DEVICE = current_platform.device_type
compute_chunk_metadata = BaseMambaAttentionMetadataBuilder._compute_chunk_metadata


def _run(inputs, seqlen, chunk_size, checkpoint_offset):
    """Run one prefill of `seqlen` tokens, optionally exporting a checkpoint.

    Returns (states, checkpoint_chunk_index). `states` holds every chunk's
    state when a checkpoint was requested, else the per-sequence final state.
    """
    A, dt, X, B, C = inputs
    offsets = [checkpoint_offset] if checkpoint_offset else None
    cu, seq_idx, last, ckpt = compute_chunk_metadata(
        chunk_size,
        1,
        torch.tensor([0]),
        torch.tensor([0, seqlen]),
        checkpoint_offsets_p=offsets,
    )

    x = X[:seqlen]
    out = torch.empty_like(x)
    states = mamba_chunk_scan_combined_varlen(
        x,
        dt[:seqlen],
        A,
        B[:seqlen],
        C[:seqlen],
        chunk_size,
        cu_seqlens=torch.tensor([0, seqlen], dtype=torch.int32, device=DEVICE),
        cu_chunk_seqlens=torch.tensor(cu, dtype=torch.int32, device=DEVICE),
        last_chunk_indices=torch.tensor(last, dtype=torch.int32, device=DEVICE),
        seq_idx=torch.tensor(seq_idx, dtype=torch.int32, device=DEVICE),
        out=out,
        D=None,
        return_intermediate_states=checkpoint_offset > 0,
        state_dtype=torch.float32,
    )
    return states, ckpt[0]


@pytest.mark.parametrize("checkpoint_offset", [32, 48, 96])
def test_checkpoint_state_matches_an_independent_prefix_run(checkpoint_offset):
    """states[checkpoint_chunk_idx] == the final state of a prefix-only run.

    48 is deliberately not a multiple of chunk_size: align-mode block sizes
    carry no chunk_size factor, so off-grid checkpoints are the common case
    and are what the chunk split exists to support.
    """
    seqlen, chunk_size, n_heads, d_head = 128, 32, 4, 32
    inputs = generate_random_inputs(1, seqlen, n_heads, d_head, torch.float32)
    inputs = tuple(t.squeeze(0) if t.dim() > 1 else t for t in inputs)

    states, ckpt_idx = _run(inputs, seqlen, chunk_size, checkpoint_offset)
    assert ckpt_idx >= 0, "the checkpoint did not land on a chunk boundary"
    checkpoint_state = states[ckpt_idx]

    prefix_final, _ = _run(inputs, checkpoint_offset, chunk_size, 0)

    torch.testing.assert_close(
        checkpoint_state.to(torch.float32),
        prefix_final[0].to(torch.float32),
        atol=8e-3,
        rtol=5e-3,
    )


def test_checkpoint_does_not_perturb_the_final_state():
    """Exporting a checkpoint must not change what the prefill produces.

    The chunk split is an extra boundary, not a different computation, so the
    sequence's final state has to match a run that took no checkpoint.
    """
    seqlen, chunk_size, n_heads, d_head = 128, 32, 4, 32
    inputs = generate_random_inputs(1, seqlen, n_heads, d_head, torch.float32)
    inputs = tuple(t.squeeze(0) if t.dim() > 1 else t for t in inputs)

    with_ckpt, ckpt_idx = _run(inputs, seqlen, chunk_size, 48)
    without_ckpt, _ = _run(inputs, seqlen, chunk_size, 0)

    torch.testing.assert_close(
        with_ckpt[-1].to(torch.float32),
        without_ckpt[0].to(torch.float32),
        atol=8e-3,
        rtol=5e-3,
    )


@pytest.mark.parametrize("dim_first", [True, False])
def test_exporter_stores_the_checkpoint_selected_from_all_chunk_states(dim_first):
    """Mamba2 hands the exporter every chunk state plus a row id per request.

    Row 0 checkpoints 32 tokens in, its state being chunk 3 rather than the
    program's own row, so the store only passes if the indirection is used.
    Row 1 declines and must write nothing. The conv input is a column slice of
    a wider projection, the SD conv layout is a transposed view, and SSM rows
    are padded pages, matching the strides the mixer passes.
    """
    dim, state_len, heads, head_dim, d_state = 16, 4, 2, 4, 8
    projection = torch.randn(64, dim + 8, device=DEVICE)
    conv_input = projection[:, :dim]
    if dim_first:
        conv_state = torch.zeros(3, dim, state_len, device=DEVICE)
    else:
        conv_state = torch.zeros(3, state_len, dim, device=DEVICE).transpose(-1, -2)
    varlen_states = torch.randn(5, heads, head_dim, d_state, device=DEVICE)
    # Blocks are pages holding more than the SSM state, so rows are padded.
    row = heads * head_dim * d_state
    ssm_storage = torch.zeros(3, row + 8, device=DEVICE)
    ssm_state = ssm_storage[:, :row].view(3, heads, head_dim, d_state)
    checkpoint = MambaPrefillCheckpointMetadata(
        checkpoint_offsets=torch.tensor([32, 0], dtype=torch.int32, device=DEVICE),
        state_indices=torch.tensor(
            [2, NULL_BLOCK_ID], dtype=torch.int32, device=DEVICE
        ),
        offsets=[32, 0],
    )

    MambaPrefillCheckpointExporter().export(
        checkpoint,
        conv_input=conv_input,
        conv_state=conv_state,
        recurrent_checkpoint=varlen_states,
        recurrent_state=ssm_state,
        cu_seqlens=torch.tensor([0, 40, 64], dtype=torch.int32, device=DEVICE),
        recurrent_row_ids=torch.tensor([3, 0], device=DEVICE),
    )

    exact = dict(rtol=0, atol=0)
    torch.testing.assert_close(
        conv_state[2], conv_input[32 - state_len : 32].T, **exact
    )
    torch.testing.assert_close(ssm_state[2], varlen_states[3], **exact)
    assert not conv_state[:2].any() and not ssm_storage[:2].any()
    assert not ssm_storage[:, row:].any(), "wrote into the page padding"
