# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AiterFlashAttentionMetadataBuilder: every field against an independent value.

Covers the decode/extend/prefill split, context chunking for extends, the
sliding-window decode tables, the capture and drafting builders, persistent
buffers and the no-GPU-sync guarantee of decode-only builds.
"""

from contextlib import contextmanager

import pytest
import torch

from tests.v1.attention.backends.rocm_aiter_fa.fa_harness import (
    DEVICE,
    FaCase,
    blocks_for,
)
from tests.v1.attention.model_shapes import (
    MUSE_SPARK_SLIDING,
    QWEN3_235B,
)
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backends.rocm_aiter_fa import (
    _CP_TOKENS_PER_ITER_ROCM,
    AiterFlashAttentionBackend,
    AiterFlashAttentionMetadataBuilder,
)

# (query_lens, seq_lens) in reorder order: decodes, extends, prefills.
BATCHES = {
    "decode": ([1, 1, 1], [17, 300, 4096]),
    "prefill": ([64, 7], [64, 7]),
    "extend": ([40, 3], [200, 19]),
    "mixed": ([1, 1, 40, 3, 64], [100, 300, 200, 19, 64]),
}


@contextmanager
def no_gpu_sync():
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(previous)


def _case(shape=QWEN3_235B, num_blocks=2048, **kwargs) -> FaCase:
    return FaCase(shape, **kwargs).setup(num_blocks)


@pytest.mark.parametrize("name", list(BATCHES))
def test_phase_split_and_per_phase_fields(name):
    query_lens, seq_lens = BATCHES[name]
    case = _case()
    batch = case.batch(query_lens, seq_lens)
    md = case.build(batch)

    kinds = [
        "decode" if q == 1 else "prefill" if q == s else "extend"
        for q, s in zip(query_lens, seq_lens)
    ]
    n_dec, n_ext, n_pre = (kinds.count(k) for k in ("decode", "extend", "prefill"))
    assert (md.num_decodes, md.num_extends, md.num_prefills) == (n_dec, n_ext, n_pre)
    assert md.num_decode_tokens == n_dec
    assert md.num_extend_tokens == sum(query_lens[n_dec : n_dec + n_ext])
    assert md.num_actual_tokens == sum(query_lens)
    assert (md.decode_metadata is None) == (n_dec == 0)
    assert (md.extend_metadata is None) == (n_ext == 0)
    assert (md.prefill_metadata is None) == (n_pre == 0)

    def rebased(lens):
        return torch.tensor([0, *lens], dtype=torch.int32).cumsum(0).tolist()

    if n_pre:
        pre_q = query_lens[n_dec + n_ext :]
        pm = md.prefill_metadata
        assert pm.query_start_loc.tolist() == rebased(pre_q)
        assert pm.max_query_len == max(pre_q)
        assert pm.max_seq_len == max(seq_lens[n_dec + n_ext :])
    if n_ext:
        ext_q = query_lens[n_dec : n_dec + n_ext]
        em = md.extend_metadata
        assert em.query_start_loc.tolist() == rebased(ext_q)
        assert em.max_query_len == max(ext_q)
        assert em.max_seq_len == max(seq_lens[n_dec : n_dec + n_ext])
    if n_dec:
        assert md.decode_metadata.max_query_len == 1
        # No sliding window: no trimmed tables.
        assert md.decode_metadata.swa_block_table is None


def _expected_chunks(context_lens: list[int], num_extends: int):
    per_chunk = _CP_TOKENS_PER_ITER_ROCM // num_extends
    num_chunks = -(-max(context_lens) // per_chunk)
    chunks = []
    for c in range(num_chunks):
        lens = [
            max(0, min(ctx, (c + 1) * per_chunk) - c * per_chunk)
            for ctx in context_lens
        ]
        chunks.append(lens)
    return per_chunk, chunks


@pytest.mark.parametrize(
    "query_lens, seq_lens",
    [
        ([16], [16 + 1]),  # one context token
        ([16], [16 + _CP_TOKENS_PER_ITER_ROCM]),  # exactly one chunk
        ([16], [16 + _CP_TOKENS_PER_ITER_ROCM + 1]),  # spills into a second
        ([8, 8, 8], [8 + 30000, 8 + 5, 8 + 12000]),  # uneven, several chunks
    ],
    ids=["ctx1", "one-chunk", "chunk-plus-one", "uneven"],
)
def test_extend_context_chunking(query_lens, seq_lens):
    case = _case(num_blocks=blocks_for(seq_lens, 16), max_model_len=65536)
    md = case.build(case.batch(query_lens, seq_lens))
    ctx = [s - q for q, s in zip(query_lens, seq_lens)]
    per_chunk, chunks = _expected_chunks(ctx, len(query_lens))
    cm = md.extend_metadata.chunk_context_metadata
    assert cm.num_chunks == len(chunks)
    assert cm.workspace.shape[1] == _CP_TOKENS_PER_ITER_ROCM
    for c, lens in enumerate(chunks):
        assert cm.chunk_starts[c].tolist() == [c * per_chunk] * len(ctx)
        cu = torch.tensor([0, *lens]).cumsum(0).tolist()
        assert cm.cu_seq_lens_chunk[c].tolist() == cu
        assert cm.total_token_per_batch[c] == sum(lens)
        assert cm.max_seq_lens[c] == max(lens)
        # token_to_batch maps every gathered token to its request.
        owners = torch.repeat_interleave(torch.arange(len(lens)), torch.tensor(lens))
        assert cm.token_to_batch[c, : sum(lens)].tolist() == owners.tolist()
        assert sum(lens) <= _CP_TOKENS_PER_ITER_ROCM


@pytest.mark.parametrize("tail_seq", [0, 700], ids=["pad-seq0", "kv-load"])
@pytest.mark.parametrize(
    "query_lens, seq_lens",
    [
        ([16], [16 + 40]),  # under one chunk
        ([16], [16 + 96]),  # exactly one chunk
        ([8, 24], [8 + 400, 24 + 250]),  # many, uneven
    ],
    ids=["under-one-chunk", "exactly-one-chunk", "many-uneven"],
)
def test_chunk_planner_skips_rows_without_queries(
    monkeypatch, query_lens, seq_lens, tail_seq
):
    """A row with no query tokens (CUDA graph padding, or a request whose KV an
    async connector is loading, vllm-project/vllm#51486) gets no context: no
    chunk may carry only such rows (vllm-project/vllm#55695, #51489)."""
    from vllm.v1.attention.backends import rocm_aiter_fa

    monkeypatch.setattr(rocm_aiter_fa, "_CP_TOKENS_PER_ITER_ROCM", 96)
    q_all = query_lens + [0, 0]
    s_all = seq_lens + [tail_seq, tail_seq // 2]
    case = _case(num_blocks=blocks_for(s_all, 16))
    md = case.build(case.batch(q_all, s_all))
    em = md.extend_metadata
    assert em is not None
    cm = em.chunk_context_metadata
    rows = range(md.num_decodes, md.num_decodes + md.num_extends)
    real = [i - md.num_decodes for i in rows if q_all[i] > 0]
    empty = [i - md.num_decodes for i in rows if q_all[i] == 0]
    per_chunk = 96 // md.num_extends
    ctx = [s_all[md.num_decodes + i] - q_all[md.num_decodes + i] for i in real]
    assert cm.num_chunks == -(-max(ctx) // per_chunk)
    for c in range(cm.num_chunks):
        lens = cm.cu_seq_lens_chunk[c].diff().tolist()
        assert all(lens[i] == 0 for i in empty), (c, lens)
        assert sum(lens[i] for i in real) > 0, (c, lens)


@pytest.mark.parametrize("tail_seq", [0, 700], ids=["pad-seq0", "kv-load"])
def test_sliding_window_extend_skips_rows_without_queries(tail_seq):
    """The sliding-window extend gathers window + query tokens per row; a row
    with no query tokens must gather none."""
    window = 64
    q_all, s_all = [8, 24, 0, 0], [8 + 400, 24 + 30, tail_seq, tail_seq // 2]
    case = _case(
        shape=MUSE_SPARK_SLIDING,
        sliding_window=window,
        num_blocks=blocks_for(s_all, 16),
    )
    md = case.build(case.batch(q_all, s_all))
    swa = md.extend_metadata.chunk_context_metadata.swa_metadata
    assert swa is not None
    lens = swa.swa_cu_seqlens.diff().tolist()
    rows = range(md.num_decodes, md.num_decodes + md.num_extends)
    expected = [min(s_all[i], q_all[i] + window) if q_all[i] else 0 for i in rows]
    assert lens == expected
    assert swa.swa_total_tokens == sum(expected)


@pytest.mark.parametrize("block_size", [16, 32])
def test_sliding_window_decode_tables(block_size):
    window = MUSE_SPARK_SLIDING.sliding_window
    # Below, at, and just past the window and block edges.
    seq_lens = [
        1,
        window - 1,
        window,
        window + 1,
        window + block_size,
        window + block_size + 1,
        3 * window + 7,
    ]
    case = _case(
        shape=MUSE_SPARK_SLIDING,
        sliding_window=window,
        block_size=block_size,
        num_blocks=blocks_for(seq_lens, block_size),
    )
    batch = case.batch([1] * len(seq_lens), seq_lens)
    md = case.build(batch)
    dm = md.decode_metadata
    assert dm.swa_block_table is not None
    table = batch.common.block_table_tensor
    for i, s in enumerate(seq_lens):
        first = max(s - window, 0) // block_size
        kept = dm.swa_seq_lens[i].item()
        assert kept == s - first * block_size
        n_blocks = -(-kept // block_size)
        assert dm.swa_block_table[i, :n_blocks].tolist() == (
            table[i, first : first + n_blocks].tolist()
        )
    assert dm.swa_max_seq_len == min(max(seq_lens), window + block_size - 1)


def test_sliding_window_tables_are_persistent_and_bounded():
    window = 256
    case = _case(shape=MUSE_SPARK_SLIDING, sliding_window=window, max_num_seqs=4)
    first = case.build(case.batch([1, 1], [300, 600])).decode_metadata
    second = case.build(case.batch([1, 1, 1], [1000, 20, 700])).decode_metadata
    assert first.swa_block_table.data_ptr() == second.swa_block_table.data_ptr()
    assert first.swa_seq_lens.data_ptr() == second.swa_seq_lens.data_ptr()
    # More decodes than the persistent table holds: fall back, don't overflow.
    over = case.build(case.batch([1] * 5, [300] * 5)).decode_metadata
    assert over.swa_block_table is None


def test_spec_decode_rows_skip_sliding_window_tables():
    case = _case(shape=MUSE_SPARK_SLIDING, sliding_window=256)
    case.builder.reorder_batch_threshold = 4
    md = case.build(case.batch([4, 4], [300, 600]))
    assert md.num_decodes == 2 and md.decode_metadata.max_query_len == 4
    assert md.decode_metadata.swa_block_table is None


@pytest.mark.parametrize("sliding_window", [None, 256])
def test_decode_only_build_does_not_sync(sliding_window):
    case = _case(shape=MUSE_SPARK_SLIDING, sliding_window=sliding_window)
    batch = case.batch([1] * 16, [37 * i + 5 for i in range(16)])
    case.build(batch)  # warm up lazy allocations
    with no_gpu_sync():
        md = case.build(batch)
        md_capture = case.builder.build_for_cudagraph_capture(batch.common)
        md_draft = case.builder.build_for_drafting(batch.common, draft_index=0)
    assert md_capture.num_decodes == md.num_decodes == 16
    assert md_draft.num_decodes == 16 and md_draft.extend_metadata is None


def test_capture_build_matches_regular_build():
    case = _case()
    batch = case.batch([1] * 8, [64 * (i + 1) for i in range(8)])
    a = case.build(batch)
    b = case.builder.build_for_cudagraph_capture(batch.common)
    for field in (
        "num_decodes",
        "num_decode_tokens",
        "num_prefills",
        "num_extends",
        "max_seq_len",
        "num_actual_tokens",
    ):
        assert getattr(a, field) == getattr(b, field), field
    assert a.decode_metadata.max_query_len == b.decode_metadata.max_query_len


def test_drafting_build_treats_every_row_as_decode():
    case = _case()
    batch = case.batch([3, 3, 3], [40, 90, 500])
    md = case.builder.build_for_drafting(batch.common, draft_index=1)
    assert (md.num_decodes, md.num_decode_tokens) == (3, 9)
    assert md.prefill_metadata is None and md.extend_metadata is None
    assert md.decode_metadata.max_query_len == 3
    assert md.block_table is batch.common.block_table_tensor


def test_builder_rejects_two_sliding_windows():
    case = _case(shape=MUSE_SPARK_SLIDING, sliding_window=128)
    shape = MUSE_SPARK_SLIDING
    with set_current_vllm_config(case.config), torch.device(DEVICE):
        Attention(
            shape.num_heads,
            shape.head_size,
            case.scale,
            shape.num_kv_heads,
            cache_config=case.config.cache_config,
            per_layer_sliding_window=512,
            prefix="other_window",
            attn_backend=AiterFlashAttentionBackend,
        )
        spec = case.builder.kv_cache_spec
        with pytest.raises(AssertionError, match="one valid sliding window"):
            AiterFlashAttentionMetadataBuilder(
                spec, ["target", "other_window"], case.config, DEVICE
            )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_asm_scale_tables_only_for_fp8_shuffle(monkeypatch, kv_cache_dtype, shuffle):
    case = _case(kv_cache_dtype=kv_cache_dtype)
    md = case.build(case.batch([1, 1], [40, 80]))
    expect_tables = shuffle and kv_cache_dtype == "fp8"
    assert (md.k_scale is not None) == expect_tables
    assert (md.v_scale is not None) == expect_tables
    if expect_tables:
        table = md.k_scale["target"]
        assert table.shape == (case.num_blocks, QWEN3_235B.num_kv_heads, 16)
        assert torch.all(table == case.layer._k_scale_float)


def test_builder_ignores_layers_outside_its_group():
    """A builder serves only its KV-cache group: a sliding-window layer in
    another group must not give this group window tables."""
    case = _case(shape=MUSE_SPARK_SLIDING)
    shape = MUSE_SPARK_SLIDING
    with set_current_vllm_config(case.config), torch.device(DEVICE):
        Attention(
            shape.num_heads,
            shape.head_size,
            case.scale,
            shape.num_kv_heads,
            cache_config=case.config.cache_config,
            per_layer_sliding_window=256,
            prefix="other_group",
            attn_backend=AiterFlashAttentionBackend,
        )
        builder = AiterFlashAttentionMetadataBuilder(
            case.builder.kv_cache_spec, ["target"], case.config, DEVICE
        )
    assert builder.aot_sliding_window is None
    assert builder._swa_block_table is None


def test_shuffle_layout_refuses_multi_token_decode(monkeypatch):
    from types import SimpleNamespace

    from tests.v1.attention.backends.rocm_aiter_fa.fa_harness import set_shuffle

    case = _case()
    set_shuffle(monkeypatch, True)
    case.config.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    monkeypatch.setattr(
        "vllm.v1.attention.backend.max_decode_query_len", lambda config: 4
    )
    with pytest.raises(ValueError, match="multi-token decode"):
        AiterFlashAttentionMetadataBuilder(
            case.builder.kv_cache_spec, ["target"], case.config, DEVICE
        )
