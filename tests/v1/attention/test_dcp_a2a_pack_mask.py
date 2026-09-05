# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The empty-shard mask fused into the DCP A2A pack kernel must be bit-exact
with the eager ``mask_dcp_empty_shards_`` pass it replaces."""

import pytest
import torch

from vllm.v1.attention.ops.dcp import (
    _dcp_a2a_lse_pack_dim,
    _dcp_a2a_pack_send,
    mask_dcp_empty_shards_,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU for the Triton pack kernel"
)


def _pack(out, lse, world_size, h_per_rank, head_dim, seq_lens, query_start_loc):
    lse_pack_dim = _dcp_a2a_lse_pack_dim(out.dtype)
    send = torch.zeros(
        (world_size, out.shape[0], h_per_rank, head_dim + lse_pack_dim),
        device=out.device,
        dtype=out.dtype,
    )
    _dcp_a2a_pack_send(
        out,
        lse,
        send,
        world_size,
        h_per_rank,
        head_dim,
        lse_pack_dim,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
    )
    return send


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("world_size,h_per_rank", [(8, 2), (8, 16), (4, 4), (2, 1)])
@pytest.mark.parametrize("tokens_per_req", [1, 2, 3])
@pytest.mark.parametrize("num_pad_rows", [0, 2])
@pytest.mark.parametrize(
    "seq_lens_list",
    [
        [16, 16, 16, 16],  # no empty shards (the long-context decode case)
        [16, 0, 16, 0],  # interior empty shards
        [0, 0, 0, 0],  # every shard empty
        [16, 16, 0, 0],  # trailing empty shards, i.e. cudagraph padding
        [0, 16, 16, 16],  # leading empty shard
        [16] * 52,  # the profiled conc-52 decode batch
        [7],  # single request
    ],
)
def test_fused_mask_matches_eager(
    dtype, world_size, h_per_rank, tokens_per_req, num_pad_rows, seq_lens_list
):
    torch.manual_seed(0)
    device = "cuda"
    head_dim = 512
    num_seqs = len(seq_lens_list)
    # tokens_per_req > 1 is the MTP / multi-token-verify shape.
    num_rows = num_seqs * tokens_per_req + num_pad_rows
    num_heads = world_size * h_per_rank

    out = torch.randn(num_rows, num_heads, head_dim, device=device, dtype=dtype)
    lse = torch.randn(num_rows, num_heads, device=device, dtype=torch.float32)
    seq_lens = torch.tensor(seq_lens_list, device=device, dtype=torch.int32)
    query_start_loc = torch.arange(
        0,
        (num_seqs + 1) * tokens_per_req,
        tokens_per_req,
        device=device,
        dtype=torch.int32,
    )  # rows past query_start_loc[-1] are cudagraph padding

    # Reference: mask eagerly, then pack with masking disabled.
    ref_lse = lse.clone()
    mask_dcp_empty_shards_(ref_lse, seq_lens, query_start_loc)
    expected = _pack(out, ref_lse, world_size, h_per_rank, head_dim, None, None)

    # Under test: pack with the mask fused in.
    actual = _pack(
        out, lse.clone(), world_size, h_per_rank, head_dim, seq_lens, query_start_loc
    )

    # Compare bit patterns, not values: for a 2-element LSE pack the high half
    # of -inf bitcasts to a NaN payload, which never compares equal to itself.
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), (
        "fused mask is not bit-exact with the eager mask"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seed", range(8))
def test_fused_mask_matches_eager_ragged(dtype, seed):
    """Non-uniform per-request query lengths.

    The uniform-stride cases above never exercise an unequal gap between
    consecutive ``query_start_loc`` entries, which is what the in-kernel
    boundary search has to get right. MTP with partial acceptance produces
    exactly this: some requests contribute one row, others several, in the
    same batch.
    """
    torch.manual_seed(seed)
    device = "cuda"
    world_size, h_per_rank, head_dim = 8, 2, 512
    num_seqs = 17

    query_lens = torch.randint(1, 5, (num_seqs,), dtype=torch.int32)
    query_start_loc = torch.zeros(num_seqs + 1, dtype=torch.int32)
    query_start_loc[1:] = torch.cumsum(query_lens, 0)
    # Mix real, empty, and (via padding rows) out-of-range shards.
    seq_lens = torch.randint(0, 2, (num_seqs,), dtype=torch.int32) * 16
    num_rows = int(query_start_loc[-1]) + 3

    query_start_loc = query_start_loc.to(device)
    seq_lens = seq_lens.to(device)
    num_heads = world_size * h_per_rank
    out = torch.randn(num_rows, num_heads, head_dim, device=device, dtype=dtype)
    lse = torch.randn(num_rows, num_heads, device=device, dtype=torch.float32)

    ref_lse = lse.clone()
    mask_dcp_empty_shards_(ref_lse, seq_lens, query_start_loc)
    expected = _pack(out, ref_lse, world_size, h_per_rank, head_dim, None, None)
    actual = _pack(
        out, lse.clone(), world_size, h_per_rank, head_dim, seq_lens, query_start_loc
    )

    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), (
        f"ragged query_start_loc mismatch: query_lens={query_lens.tolist()}, "
        f"seq_lens={seq_lens.tolist()}"
    )


def test_mask_disabled_is_unmasked():
    """Passing no seq_lens/query_start_loc must leave the LSE untouched."""
    torch.manual_seed(0)
    device = "cuda"
    world_size, h_per_rank, head_dim = 8, 2, 512
    out = torch.randn(
        4, world_size * h_per_rank, head_dim, device=device, dtype=torch.bfloat16
    )
    lse = torch.randn(4, world_size * h_per_rank, device=device, dtype=torch.float32)

    packed = _pack(out, lse, world_size, h_per_rank, head_dim, None, None)
    assert torch.isfinite(packed.float()).all()
