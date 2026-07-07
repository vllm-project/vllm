# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the fused dcp_split_q Triton kernel.

Tests correctness against the reference PyTorch implementation that was
previously inlined in FlashInferMLAImpl._split_q_for_dcp.
"""

import pytest
import torch

from vllm.v1.attention.ops.dcp_split_q import dcp_split_q


def _reference_dcp_split_q(
    global_seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    num_decodes: int,
    tokens_per_req: int,
    dcp_world_size: int,
    dcp_rank: int,
    interleave: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference."""
    num_decode_tokens = num_decodes * tokens_per_req

    offsets = torch.arange(
        1 - tokens_per_req,
        1,
        device=global_seq_lens.device,
        dtype=global_seq_lens.dtype,
    )
    per_token_global = (global_seq_lens[:num_decodes].unsqueeze(1) + offsets).flatten()

    virtual = dcp_world_size * interleave
    base = per_token_global // virtual * interleave
    remainder = (
        (per_token_global - base * dcp_world_size - dcp_rank * interleave)
        .clamp(min=0)
        .clamp(max=interleave)
    )
    ref_seq_lens = base + remainder

    ref_block_table = (
        block_table[:num_decodes]
        .unsqueeze(1)
        .expand(-1, tokens_per_req, -1)
        .reshape(num_decode_tokens, -1)
    )

    return ref_seq_lens, ref_block_table


@pytest.mark.parametrize("num_decodes", [1, 7, 64, 256])
@pytest.mark.parametrize("tokens_per_req", [2, 4, 8])
@pytest.mark.parametrize("dcp_world_size", [2, 4])
@pytest.mark.parametrize("interleave", [1, 2])
def test_dcp_split_q_seq_lens(
    num_decodes: int,
    tokens_per_req: int,
    dcp_world_size: int,
    interleave: int,
) -> None:
    device = "cuda"
    ncols = 128
    max_seq_len = 4096

    global_seq_lens = torch.randint(
        tokens_per_req,
        max_seq_len,
        (num_decodes,),
        device=device,
        dtype=torch.int32,
    )
    block_table = torch.randint(
        0,
        1000,
        (num_decodes, ncols),
        device=device,
        dtype=torch.int32,
    )

    for dcp_rank in range(dcp_world_size):
        ref_sl, ref_bt = _reference_dcp_split_q(
            global_seq_lens,
            block_table,
            num_decodes,
            tokens_per_req,
            dcp_world_size,
            dcp_rank,
            interleave,
        )

        out_sl, out_bt = dcp_split_q(
            global_seq_lens=global_seq_lens,
            block_table=block_table,
            num_decodes=num_decodes,
            tokens_per_req=tokens_per_req,
            dcp_world_size=dcp_world_size,
            dcp_rank=dcp_rank,
            interleave=interleave,
        )

        torch.testing.assert_close(out_sl, ref_sl, msg=f"{dcp_rank=}")
        torch.testing.assert_close(out_bt, ref_bt, msg=f"{dcp_rank=}")


def test_dcp_split_q_with_preallocated_buffers() -> None:
    device = "cuda"
    num_decodes = 32
    tokens_per_req = 4
    dcp_world_size = 2
    dcp_rank = 1
    interleave = 1
    ncols = 64
    total = num_decodes * tokens_per_req

    global_seq_lens = torch.randint(
        tokens_per_req,
        2048,
        (num_decodes,),
        device=device,
        dtype=torch.int32,
    )
    block_table = torch.randint(
        0,
        500,
        (num_decodes, ncols),
        device=device,
        dtype=torch.int32,
    )

    out_sl = torch.empty(total, device=device, dtype=torch.int32)
    out_bt = torch.empty(total, ncols, device=device, dtype=torch.int32)

    dcp_split_q(
        global_seq_lens=global_seq_lens,
        block_table=block_table,
        num_decodes=num_decodes,
        tokens_per_req=tokens_per_req,
        dcp_world_size=dcp_world_size,
        dcp_rank=dcp_rank,
        interleave=interleave,
        out_seq_lens=out_sl,
        out_block_table=out_bt,
    )

    ref_sl, ref_bt = _reference_dcp_split_q(
        global_seq_lens,
        block_table,
        num_decodes,
        tokens_per_req,
        dcp_world_size,
        dcp_rank,
        interleave,
    )

    torch.testing.assert_close(out_sl, ref_sl)
    torch.testing.assert_close(out_bt, ref_bt)


def test_dcp_split_q_monotonic_seq_lens() -> None:
    """Within each request, per-token local seq_lens must be non-decreasing."""
    device = "cuda"
    num_decodes = 128
    tokens_per_req = 4
    dcp_world_size = 2
    interleave = 1
    ncols = 32

    global_seq_lens = torch.randint(
        tokens_per_req,
        8192,
        (num_decodes,),
        device=device,
        dtype=torch.int32,
    )
    block_table = torch.zeros(
        num_decodes,
        ncols,
        device=device,
        dtype=torch.int32,
    )

    for dcp_rank in range(dcp_world_size):
        out_sl, _ = dcp_split_q(
            global_seq_lens=global_seq_lens,
            block_table=block_table,
            num_decodes=num_decodes,
            tokens_per_req=tokens_per_req,
            dcp_world_size=dcp_world_size,
            dcp_rank=dcp_rank,
            interleave=interleave,
        )

        per_req = out_sl.view(num_decodes, tokens_per_req)
        diffs = per_req[:, 1:] - per_req[:, :-1]
        assert (diffs >= 0).all(), (
            f"Non-monotonic local seq_lens for {dcp_rank=}: "
            f"found negative diffs {diffs[diffs < 0].tolist()}"
        )


def test_dcp_split_q_sum_across_ranks() -> None:
    """Sum of local seq_lens across all DCP ranks should equal global seq_len
    for the last token (offset=0) in each request."""
    device = "cuda"
    num_decodes = 64
    tokens_per_req = 4
    dcp_world_size = 4
    interleave = 2
    ncols = 16

    global_seq_lens = torch.randint(
        tokens_per_req,
        4096,
        (num_decodes,),
        device=device,
        dtype=torch.int32,
    )
    block_table = torch.zeros(
        num_decodes,
        ncols,
        device=device,
        dtype=torch.int32,
    )

    rank_locals = []
    for dcp_rank in range(dcp_world_size):
        out_sl, _ = dcp_split_q(
            global_seq_lens=global_seq_lens,
            block_table=block_table,
            num_decodes=num_decodes,
            tokens_per_req=tokens_per_req,
            dcp_world_size=dcp_world_size,
            dcp_rank=dcp_rank,
            interleave=interleave,
        )
        per_req = out_sl.view(num_decodes, tokens_per_req)
        rank_locals.append(per_req[:, -1])

    total = torch.stack(rank_locals).sum(dim=0)
    torch.testing.assert_close(total.to(torch.int32), global_seq_lens[:num_decodes])
