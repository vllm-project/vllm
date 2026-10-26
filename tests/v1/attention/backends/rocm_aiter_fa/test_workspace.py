# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Buffers ROCM_AITER_FA allocates, and that each is used by its config.

Persistent builder buffers must exist only for configs that read them and be
reused across builds; per-forward allocations are pinned to their formula so a
change in their size is a visible diff rather than a silent memory regression.
"""

import pytest
import torch

from tests.v1.attention.backends.rocm_aiter_fa.fa_harness import (
    GATHER,
    FaCase,
    blocks_for,
    set_shuffle,
)
from tests.v1.attention.model_shapes import (
    MUSE_SPARK_FULL,
    MUSE_SPARK_SLIDING,
    QWEN3_235B,
)
from vllm.v1.attention.backends.rocm_aiter_fa import (
    _CP_TOKENS_PER_ITER_ROCM,
    _PARTITION_SIZE_ROCM,
    AsmKvScaleTables,
)


def _nbytes(t: torch.Tensor | None) -> int:
    return 0 if t is None else t.numel() * t.element_size()


@pytest.mark.parametrize(
    "kwargs, swa, sharing",
    [
        ({}, False, False),
        ({"sliding_window": 256}, True, False),
        ({"kv_sharing": True}, False, True),
        ({"sliding_window": 256, "kv_sharing": True}, True, True),
    ],
    ids=["plain", "swa", "kv-sharing", "swa+kv-sharing"],
)
def test_persistent_buffers_only_for_configs_that_read_them(kwargs, swa, sharing):
    shape = MUSE_SPARK_SLIDING
    case = FaCase(shape, max_num_seqs=64, max_num_batched_tokens=4096, **kwargs).setup(
        64
    )
    b = case.builder
    assert (b._swa_block_table is not None) == swa
    assert (b._swa_seq_lens is not None) == swa
    assert (b.kv_sharing_workspace is not None) == sharing
    if swa:
        window_blocks = -(-256 // 16) + 1
        assert b._swa_block_table.shape == (64, window_blocks)
    if sharing:
        assert b.kv_sharing_workspace.shape == (
            2,
            4096,
            shape.num_kv_heads,
            shape.head_size,
        )
    # Every builder holds the chunked-context gather buffer.
    assert b.extend_workspace.shape == (
        2,
        _CP_TOKENS_PER_ITER_ROCM,
        shape.num_kv_heads,
        shape.head_size,
    )


def test_kv_update_bypass_layer_gets_sharing_workspace(monkeypatch):
    """Muse Spark's fused qk-norm+rope+KV insert bypasses the KV write and
    reads new K/V back from the cache like a KV-sharing layer."""
    set_shuffle(monkeypatch, False)
    case = FaCase(MUSE_SPARK_FULL)
    case.setup(32)
    assert case.builder.kv_sharing_workspace is None
    case.layers["target"].kv_cache_update_bypass = True
    from vllm.v1.attention.backends.rocm_aiter_fa import (
        AiterFlashAttentionMetadataBuilder,
    )

    builder = AiterFlashAttentionMetadataBuilder(
        case.builder.kv_cache_spec, ["target"], case.config, case.kv_cache.device
    )
    assert builder.kv_sharing_workspace is not None


def test_extend_gathers_into_the_persistent_workspace(spy):
    seq_lens = [400, 80]
    case = FaCase(QWEN3_235B).setup(blocks_for(seq_lens, 16))
    batch = case.batch([40, 16], seq_lens)
    spy.reset()
    case.run(batch, case.query(batch.num_tokens))
    (gather,) = spy.kwargs_of(GATHER)
    assert gather["key"].data_ptr() == case.builder.extend_workspace[0].data_ptr()
    assert gather["value"].data_ptr() == case.builder.extend_workspace[1].data_ptr()


def _decode_workspace_bytes(num_seqs, num_heads, head_size, max_seq_len, elem):
    partitions = -(-max_seq_len // _PARTITION_SIZE_ROCM)
    return (
        num_seqs * num_heads * partitions * head_size * elem
        + 2 * num_seqs * num_heads * partitions * 4
    )


@pytest.mark.parametrize("max_seq_len", [512, 4096])
def test_decode_transient_allocation_matches_formula(max_seq_len, monkeypatch):
    """paged_attention_v1's partition workspace is allocated per forward and
    sized by metadata.max_seq_len (max_model_len during graph capture)."""
    set_shuffle(monkeypatch, False)
    num_seqs = 16
    seq_lens = [max_seq_len] * num_seqs
    case = FaCase(QWEN3_235B, max_model_len=max_seq_len).setup(blocks_for(seq_lens, 16))
    batch = case.batch([1] * num_seqs, seq_lens)
    query = case.query(num_seqs)
    metadata = case.build(batch)
    case.run(batch, query, metadata)  # JIT warm-up
    torch.accelerator.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    case.run(batch, query, metadata)
    torch.accelerator.synchronize()
    transient = torch.cuda.max_memory_allocated() - base
    expected = _decode_workspace_bytes(
        num_seqs, QWEN3_235B.num_heads, QWEN3_235B.head_size, max_seq_len, 2
    )
    # Output/key/value buffers of run() account for the rest.
    io_bytes = 3 * _nbytes(query)
    assert expected <= transient <= expected + io_bytes + (1 << 20)


def test_asm_scale_tables_are_shared_by_key():
    tables = AsmKvScaleTables()
    dev = torch.device("cuda")
    a = tables.get(dev, 128, 4, 16, 0.5)
    assert tables.get(dev, 128, 4, 16, 0.5) is a
    assert tables.get(dev, 128, 4, 16, 0.25) is not a
    assert tables.get(dev, 256, 4, 16, 0.5) is not a
    assert len(tables.tables) == 3
