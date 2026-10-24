# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 CSA summary and tail-ring kernels on block-strided KV views.

With a block-outer KV layout, each layer's pages sit inside a block shared
with every other layer, so the summary and ring views are strided by the
whole block. Every update and read path must give the same bytes as on dense
pages.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Step5 kernels require a GPU"
)

BLOCK_SIZE = 16
BLOCKS = 12
RING_ROWS = 12
SUMMARY_BYTES = BLOCK_SIZE // 8 * 256
# Another layer's page, the ring page, then the summary page.
BLOCK_STRIDE = 2048 + RING_ROWS * 1024 + SUMMARY_BYTES + 1536
TABLE = [3, 7, 1, 9, 4, 10]
RING_BLOCK = 5


class Cache:
    def __init__(self, strided: bool):
        if strided:
            data = torch.zeros((BLOCKS, BLOCK_STRIDE), dtype=torch.uint8, device="cuda")
            ring = data[:, 2048 : 2048 + RING_ROWS * 1024].view(torch.bfloat16)
            self.mean = data[:, 2048 + RING_ROWS * 1024 :][:, :SUMMARY_BYTES].view(
                BLOCKS, BLOCK_SIZE // 8, 256
            )
        else:
            ring = torch.zeros(
                (BLOCKS, RING_ROWS * 512), dtype=torch.bfloat16, device="cuda"
            )
            self.mean = torch.zeros(
                (BLOCKS, BLOCK_SIZE // 8, 256), dtype=torch.uint8, device="cuda"
            )
        self.rows_per_block = ring.stride(0) // 512
        self.rows = ring.as_strided(
            ((BLOCKS - 1) * self.rows_per_block + RING_ROWS, 512), (512, 1)
        )

    def ring_metadata(self, positions, seq_len):
        base = torch.full_like(positions, RING_BLOCK * self.rows_per_block)
        keep = positions >= seq_len - RING_ROWS
        slots = torch.where(keep, base + positions % RING_ROWS, -1)
        return base, slots

    def ring_rows(self):
        start = RING_BLOCK * self.rows_per_block
        return self.rows[start : start + RING_ROWS]


def _slots(positions):
    table = torch.tensor(TABLE, device="cuda")
    return table[positions // BLOCK_SIZE] * BLOCK_SIZE + positions % BLOCK_SIZE


def _run_updates(cache: Cache, key, z):
    from vllm.models.step5.nvidia.ops.summary_update import (
        build_prefill_groups,
        update_decode,
        update_prefill,
        update_verify,
    )

    # A 21-token prefill chunk, then a 22-token chunk that resumes the open
    # region from the ring.
    for start, end in ((0, 21), (21, 43)):
        positions = torch.arange(start, end, device="cuda")
        slots = _slots(positions)
        owners = torch.zeros_like(positions)
        query_start = torch.tensor([0, end - start], device="cuda")
        metadata = build_prefill_groups(slots, positions, owners, query_start)
        base, ring_slots = cache.ring_metadata(positions, end)
        update_prefill(
            key[start:end, None],
            z[start:end, None],
            metadata,
            positions,
            base,
            ring_slots,
            cache.mean,
            cache.rows,
            RING_ROWS,
        )
    # Decode through a region boundary.
    for pos in range(43, 52):
        positions = torch.tensor([pos], device="cuda")
        base, _ = cache.ring_metadata(positions, pos + 1)
        update_decode(
            key[pos : pos + 1],
            z[pos : pos + 1],
            _slots(positions),
            positions,
            base,
            cache.mean,
            cache.rows,
            RING_ROWS,
        )
    # One MTP verify step of four rows that closes a region.
    positions = torch.arange(52, 56, device="cuda")
    base, _ = cache.ring_metadata(positions, 56)
    update_verify(
        key[52:56, None],
        z[52:56, None],
        _slots(positions),
        torch.tensor([0, 4], device="cuda"),
        torch.tensor([56], device="cuda"),
        base,
        cache.mean,
        cache.rows,
        RING_ROWS,
    )


def test_summary_updates_match_on_strided_blocks():
    torch.manual_seed(0)
    key = torch.randn((56, 256), device="cuda").to(torch.bfloat16)
    z = torch.randn((56, 256), device="cuda").to(torch.bfloat16)
    dense, strided = Cache(strided=False), Cache(strided=True)
    _run_updates(dense, key, z)
    _run_updates(strided, key, z)

    assert torch.equal(dense.mean, strided.mean)
    assert dense.mean[TABLE[:3]].any(), "no region was summarized"
    assert torch.equal(dense.ring_rows(), strided.ring_rows())


def test_summary_reads_match_on_strided_blocks():
    from vllm.models.step5.nvidia.ops.sparse_indexer import pack_summary_pages
    from vllm.models.step5.nvidia.ops.sparse_indexer_scores import (
        decode_summary_scores,
    )

    torch.manual_seed(0)
    dense, strided = Cache(strided=False), Cache(strided=True)
    values = torch.randint(0, 120, dense.mean.shape, device="cuda").to(torch.uint8)
    dense.mean.copy_(values)
    strided.mean.copy_(values)
    table = torch.tensor([TABLE, TABLE[::-1]], dtype=torch.int32, device="cuda")
    regions = len(TABLE) * BLOCK_SIZE // 8

    packed_dense, pages = pack_summary_pages(dense.mean, table, regions)
    packed_strided, _ = pack_summary_pages(strided.mean, table, regions)
    assert pages > 0 and torch.equal(packed_dense, packed_strided)

    query = torch.randn((2, 1, 8, 256), device="cuda").to(torch.float8_e4m3fn)
    weights = torch.rand((2, 1, 8), device="cuda")
    slots = torch.tensor([0, 0], device="cuda")
    seq_lens = torch.tensor([regions * 8, regions * 8 - 5], device="cuda")
    scores_dense, meta_dense = decode_summary_scores(
        query, weights, dense.mean, table, slots, seq_lens, regions, num_sms=4
    )
    scores_strided, _ = decode_summary_scores(
        query, weights, strided.mean, table, slots, seq_lens, regions, num_sms=4
    )
    for row, length in enumerate(meta_dense.lengths.tolist()):
        assert torch.equal(scores_dense[row, :length], scores_strided[row, :length])
