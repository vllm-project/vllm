# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP query-row sharding of the DSA indexer prefill (see PrefillRowShard):
each rank scores a contiguous slice; the group exchanges int32s per row."""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer
import vllm.v1.attention.backends.mla.indexer as indexer
from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.sparse_attn_indexer import PrefillRowShard


def _chunk(start: int, end: int) -> SimpleNamespace:
    """Chunk over rows [start, end); bounds derive from the global row."""
    n = end - start
    return SimpleNamespace(
        token_start=start,
        token_end=end,
        cu_seqlen_ks=torch.zeros(n, dtype=torch.int32),
        cu_seqlen_ke=((torch.arange(n) * 7 + start) % 65).int(),
    )


@pytest.mark.parametrize("world", [2, 3, 8])
def test_shard_narrowing_covers_every_row_once(world: int) -> None:
    """Narrowed windows tile the prefill rows exactly once, carry the causal
    bounds of those global rows, and skip chunks foreign to this rank."""
    decode, rows = 5, 997
    edges = [decode, decode + 330, decode + 337, decode + rows]
    chunks = [_chunk(lo, hi) for lo, hi in zip(edges, edges[1:])]
    sizes = [rows // world + int(r < rows % world) for r in range(world)]
    covered = []
    for rank in range(world):
        shard = PrefillRowShard(decode, sizes, rank)
        for chunk in chunks:
            window = shard.narrow(chunk)
            if window is None:
                continue
            start, end, ks, ke = window
            lo, hi = start - chunk.token_start, end - chunk.token_start
            torch.testing.assert_close(ks, chunk.cu_seqlen_ks[lo:hi])
            torch.testing.assert_close(ke, chunk.cu_seqlen_ke[lo:hi])
            covered.append((start, end))
    covered.sort()
    # Windows must be disjoint, contiguous and exactly cover the prefill rows.
    assert [s for s, _ in covered] == [decode, *[e for _, e in covered][:-1]]
    assert covered[-1][1] == decode + rows


def test_shard_exchange_reassembles_rows_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-place exchange passes the buffer's own full-width prefill rows
    as gather destination and local slice as source — both contiguous, which
    is what lets PyNCCL skip the output allocation — and writes land only on
    real prefill rows: leading decode rows and trailing CUDA-graph padding
    stay untouched, with the tile-padded width exchanged."""
    decode, pad, topk, kpool = 5, 3, 8, 4
    width = topk + kpool - 1
    padded_width = (width + 127) // 128 * 128
    sizes = [17, 9, 21]
    buffer = torch.full(
        (decode + sum(sizes) + pad, padded_width), -7, dtype=torch.int32
    )
    gathered = torch.arange(sum(sizes) * padded_width, dtype=torch.int32).reshape(
        -1, padded_width
    )
    calls = []

    def fake_group():
        def all_gatherv(output_tensor, input_tensor, sizes):
            calls.append((output_tensor, input_tensor, sizes))
            output_tensor.copy_(gathered)

        return SimpleNamespace(
            device_communicator=SimpleNamespace(
                pynccl_comm=SimpleNamespace(disabled=False, all_gatherv=all_gatherv)
            )
        )

    monkeypatch.setattr(sparse_indexer, "get_tp_group", fake_group)
    for rank in range(len(sizes)):
        PrefillRowShard(decode, sizes, rank).exchange_topk(buffer)
        dst, src, arg_sizes = calls[-1]
        start = decode + sum(sizes[:rank])
        assert arg_sizes == sizes
        assert dst.shape == (sum(sizes), padded_width) and dst.is_contiguous()
        assert src.data_ptr() == buffer[start].data_ptr()
        assert src.shape == (sizes[rank], padded_width) and src.is_contiguous()
    torch.testing.assert_close(buffer[decode : decode + sum(sizes)], gathered)
    assert torch.all(buffer[:decode] == -7), "decode rows stay outside"
    assert torch.all(buffer[decode + sum(sizes) :] == -7), "padding untouched"


@pytest.mark.parametrize("tp_size", [2, 4])
def test_balanced_row_shard_partitions_and_declines(tp_size: int) -> None:
    """The split covers every row once with >= 1 row per rank and declines
    below the activation floor (or on a single rank)."""
    n = indexer.MIN_TP_SHARD_ROWS_PER_RANK * tp_size
    shapes = [
        ([4 * n], [4 * n]),  # fresh prompt: the full causal cost ramp
        # uneven requests at different context depths
        ([2 * n, 7, 4 * n, n // 3 + 5], [2 * n, 9007, 4 * n + 500, n // 3 + 60005]),
    ]
    ratio = 4
    for query_lens, seq_lens in shapes:
        sizes = indexer.balanced_prefill_row_shard(
            torch.tensor(seq_lens, dtype=torch.int32),
            torch.tensor(query_lens, dtype=torch.int32),
            ratio,
            tp_size,
        )
        assert sizes is not None and len(sizes) == tp_size
        assert sum(sizes) == sum(query_lens) and min(sizes) >= 1
    lens = torch.tensor([n - 1], dtype=torch.int32)
    assert indexer.balanced_prefill_row_shard(lens, lens, ratio, tp_size) is None
    assert indexer.balanced_prefill_row_shard(lens, lens, ratio, 1) is None


def test_row_sharding_gate_rejects_unsupported_configurations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate is the whole safety envelope; nothing else guards the exchange."""
    monkeypatch.setattr(indexer.current_platform, "is_cuda", lambda: True)

    def supported(**kwargs):
        mode = kwargs.pop("cudagraph_mode", CUDAGraphMode.PIECEWISE)
        cfg = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=mode))
        args = {"dcp_world_size": 1, "use_pcp": False, "tp_size": 4, **kwargs}
        return indexer.tp_prefill_row_sharding_supported(cfg, **args)

    assert supported()
    for bad in ({"tp_size": 1}, {"dcp_world_size": 2}, {"use_pcp": True}):
        assert not supported(**bad)
    for name in (
        "VLLM_DISABLE_PYNCCL",
        "VLLM_USE_NCCL_SYMM_MEM",
        "VLLM_BATCH_INVARIANT",
    ):
        with monkeypatch.context() as m:
            m.setattr(indexer.envs, name, True)
            assert not supported()
    # FULL captures mixed batches whole, so the exchange must stay out.
    assert not supported(cudagraph_mode=CUDAGraphMode.FULL)
    assert supported(cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE)
