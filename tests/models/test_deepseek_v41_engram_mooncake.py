# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v4_1.nvidia.engram_mooncake import (
    MooncakeEngramBackend,
    _consume_previous_lookup,
    _dequant_packed_engram_rows,
    _LayerBuffers,
    _LookupSlot,
    engram_head_shard,
)


class _Event:
    def synchronize(self):
        pass


class _Table:
    def __init__(self):
        self.calls = []

    def lookup_many_into(self, layer_ids, row_ids, outputs):
        self.calls.append((list(layer_ids), [ids.copy() for ids in row_ids]))
        for ids, output in zip(row_ids, outputs):
            output[...] = ids[..., None]


def _layer(layer_id, hash_index, offsets):
    tokens, heads, row_bytes = 4, len(offsets), 4
    return _LayerBuffers(
        embedding=lambda: None,
        model_layer_id=layer_id,
        store_layer_id=layer_id * 2,
        layer_hash_index=hash_index,
        head_start=0,
        head_sizes=(10,) * heads,
        global_offsets=np.asarray(offsets, dtype=np.int64),
        row_bytes=row_bytes,
        max_tokens=tokens,
        host_ids=[torch.empty(tokens, heads, dtype=torch.int32)],
        local_ids=[np.empty((tokens, heads), dtype=np.int64)],
        host_rows=[torch.full((tokens, heads, row_bytes), 255, dtype=torch.uint8)],
    )


def test_engram_head_shard_uses_only_rank_owned_heads():
    sizes = tuple(range(11, 21))
    start, local, offsets = engram_head_shard(sizes, num_shards=3, shard_rank=1)
    assert start == 4
    assert local == sizes[4:8]
    assert offsets == tuple(np.cumsum((0, *sizes[:-1]))[4:8])


def test_lookup_batches_layers_and_zeros_dead_and_trailing_rows():
    first = _layer(1, 0, (0, 10))
    second = _layer(14, 1, (0, 10))
    first.host_ids[0][...] = torch.tensor(
        [[1, 12], [-1, 15], [-1, -1], [-1, -1]], dtype=torch.int32
    )
    second.host_ids[0][...] = torch.tensor(
        [[3, 19], [4, -1], [-1, -1], [-1, -1]], dtype=torch.int32
    )
    table = _Table()
    backend = MooncakeEngramBackend.__new__(MooncakeEngramBackend)
    backend._layers = {0: first, 1: second}
    backend._slots = [SimpleNamespace(copied=_Event())]
    backend._table = table

    assert backend._lookup(0, 4) == 2
    assert len(table.calls) == 1
    layer_ids, row_ids = table.calls[0]
    assert layer_ids == [2, 28]
    np.testing.assert_array_equal(row_ids[0], [[[1, 2], [0, 5]]])
    np.testing.assert_array_equal(row_ids[1], [[[3, 9], [4, 0]]])
    assert not first.host_rows[0][1, 0].any()
    assert not second.host_rows[0][1, 1].any()
    assert not first.host_rows[0][2:].any()
    assert not second.host_rows[0][2:].any()


def test_failed_lookup_is_consumed_once_and_allows_retry():
    failed = Future()
    failed.set_exception(RuntimeError("lookup failed"))
    slot = _LookupSlot(copied=_Event(), future=failed)

    with pytest.raises(RuntimeError, match="lookup failed"):
        _consume_previous_lookup(slot)
    assert slot.future is None
    _consume_previous_lookup(slot)


def test_packed_row_dequant_matches_torch():
    if not torch.cuda.is_available():
        return
    tokens, actual_heads, part_heads, dim, block = 3, 2, 3, 256, 32
    row_bytes = dim + dim // block
    generator = torch.Generator().manual_seed(7)
    weight = torch.randint(
        0, 255, (tokens, actual_heads, dim), generator=generator, dtype=torch.uint8
    )
    weight[weight == 0x7F] = 0x7E
    scales = torch.randint(
        120,
        134,
        (tokens, actual_heads, dim // block),
        generator=generator,
        dtype=torch.uint8,
    )
    packed = torch.cat((weight, scales), dim=-1).cuda()
    output = torch.zeros(tokens, part_heads, dim, dtype=torch.bfloat16, device="cuda")
    num_rows = tokens * actual_heads
    _dequant_packed_engram_rows[((num_rows + 15) // 16,)](
        packed,
        output,
        num_rows,
        ACTUAL_HEADS=actual_heads,
        PART_HEADS=part_heads,
        ROW_BYTES=row_bytes,
        DIM=dim,
        QUANT_BLOCK=block,
        BLOCK_R=16,
    )
    expected = (
        weight.view(torch.float8_e4m3fn).float().unflatten(-1, (-1, block))
        * scales.view(torch.float8_e8m0fnu).float().unsqueeze(-1)
    ).flatten(-2)
    torch.testing.assert_close(output[:, :actual_heads].cpu(), expected.bfloat16())
    assert not output[:, actual_heads:].count_nonzero()
