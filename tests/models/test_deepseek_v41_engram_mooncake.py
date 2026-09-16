# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v4_1.nvidia import engram_mooncake
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

    def lookup_many_into_registered(
        self, layer_ids, row_ids, output_addresses, output_sizes
    ):
        self.calls.append(
            (
                list(layer_ids),
                [ids.copy() for ids in row_ids],
                list(output_addresses),
                list(output_sizes),
            )
        )


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
        host_dead=[torch.empty(tokens, heads, dtype=torch.bool)],
        device_packed=[torch.empty(tokens, heads, row_bytes, dtype=torch.uint8)],
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
    backend._slots = [SimpleNamespace(ids_ready=_Event(), rows_consumed=_Event())]
    backend._table = table

    assert backend._lookup(0, 4) == 2
    assert len(table.calls) == 1
    layer_ids, row_ids, output_addresses, output_sizes = table.calls[0]
    assert layer_ids == [2, 28]
    np.testing.assert_array_equal(row_ids[0], [[[1, 2], [0, 5]]])
    np.testing.assert_array_equal(row_ids[1], [[[3, 9], [4, 0]]])
    assert output_addresses == [
        first.device_packed[0].data_ptr(),
        second.device_packed[0].data_ptr(),
    ]
    assert output_sizes == [2 * 2 * 4, 2 * 2 * 4]
    torch.testing.assert_close(
        first.host_dead[0],
        torch.tensor([[False, False], [True, False], [True, True], [True, True]]),
    )
    torch.testing.assert_close(
        second.host_dead[0],
        torch.tensor([[False, False], [False, True], [True, True], [True, True]]),
    )


def test_failed_lookup_is_consumed_once_and_allows_retry():
    failed = Future()
    failed.set_exception(RuntimeError("lookup failed"))
    slot = _LookupSlot(ids_ready=_Event(), rows_consumed=_Event(), future=failed)

    with pytest.raises(RuntimeError, match="lookup failed"):
        _consume_previous_lookup(slot)
    assert slot.future is None
    _consume_previous_lookup(slot)


@pytest.mark.parametrize(
    "active_rows, needs_flush, expected_flushes",
    [(2, True, 1), (0, True, 0), (2, False, 0)],
)
def test_wait_flushes_gpudirect_writes_once(
    monkeypatch, active_rows, needs_flush, expected_flushes
):
    future = Future()
    future.set_result(active_rows)
    slot = _LookupSlot(
        ids_ready=_Event(),
        rows_consumed=_Event(),
        future=future,
        writes_flushed=False,
    )
    backend = MooncakeEngramBackend.__new__(MooncakeEngramBackend)
    backend._slots = [slot]
    backend._needs_gpudirect_flush = needs_flush
    flushes = []
    monkeypatch.setattr(engram_mooncake, "dbo_current_ubatch_id", lambda: 0)
    monkeypatch.setattr(
        engram_mooncake, "_flush_gpudirect_writes", lambda: flushes.append(True)
    )

    backend.wait()
    backend.wait()

    assert len(flushes) == expected_flushes


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
    dead = torch.zeros(tokens, actual_heads, dtype=torch.bool, device="cuda")
    dead[1, 0] = True
    output = torch.zeros(tokens, part_heads, dim, dtype=torch.bfloat16, device="cuda")
    num_rows = tokens * actual_heads
    _dequant_packed_engram_rows[((num_rows + 15) // 16,)](
        packed,
        dead,
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
    assert not output[1, 0].count_nonzero()
    expected[1, 0].zero_()
    torch.testing.assert_close(output[:, :actual_heads].cpu(), expected.bfloat16())
    assert not output[:, actual_heads:].count_nonzero()
