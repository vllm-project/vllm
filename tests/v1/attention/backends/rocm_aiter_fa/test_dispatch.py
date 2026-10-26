# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every AiterFlashAttentionImpl.forward branch: kernel taken and output.

Dispatch table (rocm_aiter_fa.py, AiterFlashAttentionImpl.forward):

| Row | Phase   | Condition                         | Kernels                        |
| --- | ------- | --------------------------------- | ------------------------------ |
| P1  | prefill | always                            | varlen (causal from metadata)  |
| E1  | extend  | no sliding window                 | varlen suffix; per context     |
|     |         |                                   | chunk gather + varlen; merge   |
| E2  | extend  | sliding window                    | gather window + varlen         |
| D1  | decode  | q=1, packed cache, head >= 64     | paged_attention_v1             |
| D2  | decode  | q=1, sliding window, no           | paged_attention_v1 over the    |
|     |         | sinks/alibi                       | window-trimmed block table     |
| D3  | decode  | sliding window with alibi         | unified_attention              |
| D4  | decode  | sinks                             | unified_attention              |
| D5  | decode  | q>1 (spec decode), causal         | unified_attention              |
| D6  | decode  | non-causal                        | mha_v3 flash_attn_with_kvcache |
| D7  | decode  | shuffle layout                    | paged_attention_common         |
| D8  | decode  | head < 64                         | unified_attention              |
| U1  | any     | soft cap, sinks or ALiBi, with    | unified_attention for the      |
|     |         | prefill or extend rows (causal)   | whole batch                    |
| S1  | any     | KV-sharing / update-bypass layer  | gather new K/V from the cache  |
|     |         | with prefill or extend rows       | before prefill/extend          |
| X1  | -       | attn_metadata is None (profiling) | none, output zeroed            |
| X2  | -       | output_scale given                | NotImplementedError            |
| X3  | decode  | shuffle + window / sinks / q>1    | assertion                      |
"""

from collections import Counter
from dataclasses import dataclass, field, replace

import pytest
import torch

from tests.v1.attention.backends.rocm_aiter_fa.fa_harness import (
    DEVICE,
    GATHER,
    MERGE,
    MHA_V3,
    PAGED_COMMON,
    PAGED_V1,
    UNIFIED,
    VARLEN,
    FaCase,
    blocks_for,
)
from tests.v1.attention.model_shapes import (
    MUSE_SPARK_FULL,
    MUSE_SPARK_SLIDING,
    QWEN3_8_27B,
    QWEN3_235B,
    AttnShape,
)
from vllm.v1.attention.backends.rocm_aiter_fa import _CP_TOKENS_PER_ITER_ROCM

SMALL_WINDOW = 64


@dataclass
class Row:
    id: str
    query_lens: list[int]
    seq_lens: list[int]
    kernels: dict[str, int]
    shape: AttnShape = QWEN3_235B
    case: dict = field(default_factory=dict)
    shuffle: bool = False
    causal: bool = True
    decode_threshold: int = 1


ROWS = [
    Row("P1-prefill", [33, 64], [33, 64], {VARLEN: 1}),
    Row("P1-prefill-noncausal", [33, 64], [33, 64], {VARLEN: 1}, causal=False),
    Row("E1-extend-one-chunk", [40, 3], [200, 19], {VARLEN: 2, GATHER: 1, MERGE: 1}),
    Row(
        "E1-extend-two-chunks",
        [16],
        [16 + _CP_TOKENS_PER_ITER_ROCM + 100],
        {VARLEN: 3, GATHER: 2, MERGE: 2},
    ),
    Row(
        "E2-extend-swa",
        [40, 3],
        [400, 19],
        {VARLEN: 1, GATHER: 1},
        shape=MUSE_SPARK_SLIDING,
        case={"sliding_window": SMALL_WINDOW},
    ),
    Row("D1-decode", [1, 1, 1], [17, 300, 1025], {PAGED_V1: 1}),
    Row(
        "D2-decode-swa",
        [1, 1, 1],
        [17, 300, 1025],
        {PAGED_V1: 1},
        shape=MUSE_SPARK_SLIDING,
        case={"sliding_window": SMALL_WINDOW},
    ),
    Row(
        "D3-decode-swa-alibi",
        [1, 1],
        [17, 300],
        {UNIFIED: 1},
        shape=MUSE_SPARK_SLIDING,
        case={"sliding_window": SMALL_WINDOW, "use_alibi": True},
    ),
    Row("D4-decode-sinks", [1, 1], [17, 300], {UNIFIED: 1}, case={"use_sinks": True}),
    Row("D5-decode-spec", [4, 4, 2], [40, 300, 77], {UNIFIED: 1}, decode_threshold=4),
    Row(
        "D6-decode-noncausal",
        [4, 4],
        [40, 300],
        {MHA_V3: 1},
        causal=False,
        decode_threshold=4,
    ),
    Row(
        "D7-decode-shuffle", [1, 1, 1], [17, 300, 1025], {PAGED_COMMON: 1}, shuffle=True
    ),
    Row(
        "D8-decode-head32",
        [1, 1],
        [17, 300],
        {UNIFIED: 1},
        shape=AttnShape("synthetic", 8, 2, 32),
    ),
    Row(
        "mixed",
        [1, 1, 40, 3, 64],
        [100, 300, 200, 19, 64],
        {PAGED_V1: 1, VARLEN: 3, GATHER: 1, MERGE: 1},
    ),
    Row(
        "mixed-swa",
        [1, 1, 40, 64],
        [100, 300, 200, 64],
        {PAGED_V1: 1, VARLEN: 2, GATHER: 1},
        shape=MUSE_SPARK_SLIDING,
        case={"sliding_window": SMALL_WINDOW},
    ),
    Row(
        "mixed-shuffle",
        [1, 1, 40, 64],
        [100, 300, 200, 64],
        {PAGED_COMMON: 1, VARLEN: 3, GATHER: 1, MERGE: 1},
        shuffle=True,
    ),
    Row(
        "S1-shared-prefill",
        [33, 64],
        [33, 64],
        {GATHER: 1, VARLEN: 1},
        case={"kv_sharing": True},
    ),
    Row(
        "S1-shared-mixed",
        [1, 40, 64],
        [100, 200, 64],
        {GATHER: 2, VARLEN: 3, MERGE: 1, PAGED_V1: 1},
        case={"kv_sharing": True},
    ),
    Row(
        "S1-shared-extend-swa",
        [40, 3],
        [400, 19],
        {GATHER: 2, VARLEN: 1},
        shape=MUSE_SPARK_SLIDING,
        case={"sliding_window": SMALL_WINDOW, "kv_sharing": True},
    ),
    Row(
        "S1-shared-mixed-swa",
        [1, 40, 64],
        [100, 200, 64],
        {GATHER: 2, VARLEN: 2, PAGED_V1: 1},
        shape=MUSE_SPARK_SLIDING,
        case={"sliding_window": SMALL_WINDOW, "kv_sharing": True},
    ),
    Row(
        "S1-shared-mixed-shuffle",
        [1, 40, 64],
        [100, 200, 64],
        {GATHER: 2, VARLEN: 3, MERGE: 1, PAGED_COMMON: 1},
        shuffle=True,
        case={"kv_sharing": True},
    ),
]


def _run_row(row: Row, spy, monkeypatch, kv_cache_dtype: str = "auto"):
    from tests.v1.attention.backends.rocm_aiter_fa.fa_harness import set_shuffle

    set_shuffle(monkeypatch, row.shuffle)
    case = FaCase(row.shape, kv_cache_dtype=kv_cache_dtype, **row.case).setup(
        blocks_for(row.seq_lens, 16)
    )
    case.builder.reorder_batch_threshold = row.decode_threshold
    batch = case.batch(row.query_lens, row.seq_lens, causal=row.causal)
    query = case.query(batch.num_tokens)
    metadata = case.build(batch)
    cache_before = case.kv_cache.view(torch.uint8).clone()
    spy.reset()
    output = case.run(batch, query, metadata)
    if case.kv_sharing:
        # A sharing layer reads the target's cache and must never write it.
        torch.testing.assert_close(
            case.kv_cache.view(torch.uint8), cache_before, atol=0, rtol=0
        )
    return case, batch, query, output


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r.id)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_dispatch_row(row, kv_cache_dtype, spy, monkeypatch):
    case, batch, query, output = _run_row(row, spy, monkeypatch, kv_cache_dtype)
    assert Counter(spy.names()) == Counter(row.kernels), spy.names()
    case.check(batch, query, output)


@pytest.mark.parametrize(
    "query_lens, seq_lens, kernels, window",
    [
        ([1, 1], [17, 300], {PAGED_V1: 1}, None),
        ([33, 64], [33, 64], {UNIFIED: 1}, None),
        ([40, 3], [200, 19], {UNIFIED: 1}, None),
        ([40, 3], [400, 19], {UNIFIED: 1}, SMALL_WINDOW),
    ],
    ids=["decode", "prefill", "extend", "extend-swa"],
)
def test_logits_soft_cap_every_phase(spy, query_lens, seq_lens, kernels, window):
    """Gemma-2/Grok-style attention soft cap must apply in every phase."""
    case = FaCase(
        QWEN3_235B if window is None else MUSE_SPARK_SLIDING,
        logits_soft_cap=5.0,
        sliding_window=window,
    ).setup(blocks_for(seq_lens, 16))
    batch = case.batch(query_lens, seq_lens)
    # Large logits so the cap changes the result.
    query = case.query(batch.num_tokens) * 8
    spy.reset()
    output = case.run(batch, query)
    assert Counter(spy.names()) == Counter(kernels)
    case.check(batch, query, output)


@pytest.mark.parametrize(
    "row", [r for r in ROWS if r.shape is QWEN3_235B], ids=lambda r: r.id
)
@pytest.mark.parametrize(
    "shape", [MUSE_SPARK_FULL, QWEN3_8_27B], ids=lambda s: f"d{s.head_size}"
)
def test_dispatch_row_other_head_sizes(row, shape, spy, monkeypatch):
    """The head-128 dispatch rows again at head 64 and 256."""
    row = replace(row, shape=shape)
    case, batch, query, output = _run_row(row, spy, monkeypatch)
    assert Counter(spy.names()) == Counter(row.kernels), spy.names()
    case.check(batch, query, output)


def test_decode_swa_uses_trimmed_table(spy, monkeypatch):
    row = next(r for r in ROWS if r.id == "D2-decode-swa")
    case, batch, _, _ = _run_row(row, spy, monkeypatch)
    md = case.build(batch).decode_metadata
    (args,) = spy.args_of(PAGED_V1)
    block_table, seq_lens, max_seq_len = args[6], args[8], args[9]
    assert block_table.data_ptr() == md.swa_block_table.data_ptr()
    assert seq_lens.data_ptr() == md.swa_seq_lens.data_ptr()
    assert max_seq_len == md.swa_max_seq_len < max(row.seq_lens)


def test_extend_swa_gathers_only_the_window(spy, monkeypatch):
    row = next(r for r in ROWS if r.id == "E2-extend-swa")
    _, batch, _, _ = _run_row(row, spy, monkeypatch)
    (gather,) = spy.kwargs_of(GATHER)
    expected = sum(
        min(s, q + SMALL_WINDOW) for q, s in zip(row.query_lens, row.seq_lens)
    )
    assert gather["total_tokens"] == expected


def test_profiling_run_zeroes_output_without_kernels(spy):
    case = FaCase(QWEN3_235B).setup(16)
    query = case.query(8)
    output = torch.full_like(query, 7.0)
    spy.reset()
    case.layer.impl.forward(case.layer, query, None, None, case.kv_cache, None, output)
    assert spy.names() == []
    assert torch.count_nonzero(output) == 0


def test_fused_output_quant_is_rejected():
    case = FaCase(QWEN3_235B).setup(16)
    batch = case.batch([4], [4])
    query = case.query(4)
    with pytest.raises(NotImplementedError, match="fused output quantization"):
        case.layer.impl.forward(
            case.layer,
            query,
            query[:, :4],
            query[:, :4],
            case.kv_cache,
            case.build(batch),
            torch.empty_like(query),
            output_scale=torch.ones(1, device=query.device),
        )


@pytest.mark.parametrize(
    "case_kwargs, query_lens, threshold",
    [
        ({"sliding_window": SMALL_WINDOW}, [1, 1], 1),
        ({}, [4, 4], 4),
    ],
    ids=["swa", "spec"],
)
def test_shuffle_rejects_unsupported_decode(
    monkeypatch, case_kwargs, query_lens, threshold
):
    from tests.v1.attention.backends.rocm_aiter_fa.fa_harness import set_shuffle

    set_shuffle(monkeypatch, True)
    seq_lens = [300, 500]
    case = FaCase(MUSE_SPARK_SLIDING, **case_kwargs).setup(blocks_for(seq_lens, 16))
    case.builder.reorder_batch_threshold = threshold
    batch = case.batch(query_lens, seq_lens)
    with pytest.raises(AssertionError, match="Shuffle KV cache layout"):
        case.run(batch, case.query(batch.num_tokens))


@pytest.mark.parametrize(
    "feature",
    [{"use_sinks": True}, {"use_alibi": True}, {"logits_soft_cap": 5.0}],
    ids=["sinks", "alibi", "softcap"],
)
def test_shuffle_refuses_features_it_cannot_decode(feature, monkeypatch):
    from tests.v1.attention.backends.rocm_aiter_fa.fa_harness import set_shuffle

    set_shuffle(monkeypatch, True)
    with pytest.raises(NotImplementedError, match="shuffle KV cache layout"):
        FaCase(QWEN3_235B, **feature).setup(16)


def test_padded_tokens_are_not_written():
    """CUDA-graph padding: rows past num_actual_tokens stay untouched."""
    case = FaCase(QWEN3_8_27B).setup(64)
    batch = case.batch([1, 1, 1], [40, 90, 500], num_padded_tokens=8)
    query = case.query(8)
    output = case.run(batch, query)
    case.check(batch, query, output)
    assert torch.isnan(output[batch.num_tokens :]).all()


@pytest.mark.parametrize(
    "query_lens, seq_lens",
    [
        # Model runner v1 order: decode, short extend, long extend, prefill.
        ([1, 3, 40, 5], [100, 19, 200, 5]),
        # Model runner v2 order: decodes first, then by scheduled token count,
        # so a short prefill can precede a longer extend.
        ([1, 3, 5, 40], [100, 19, 5, 200]),
    ],
    ids=["runner-v1-order", "runner-v2-order"],
)
def test_batch_order_from_each_model_runner(query_lens, seq_lens):
    case = FaCase(QWEN3_235B).setup(blocks_for(seq_lens, 16))
    batch = case.batch(query_lens, seq_lens)
    query = case.query(batch.num_tokens)
    case.check(batch, query, case.run(batch, query))


FEATURES = {
    "alibi": {"use_alibi": True},
    "sinks": {"use_sinks": True},
    "softcap": {"logits_soft_cap": 5.0},
    "window": {"sliding_window": SMALL_WINDOW},
    "window+sinks": {"sliding_window": SMALL_WINDOW, "use_sinks": True},
    "window+alibi": {"sliding_window": SMALL_WINDOW, "use_alibi": True},
    "window+softcap": {"sliding_window": SMALL_WINDOW, "logits_soft_cap": 5.0},
}
# (query_lens, seq_lens, decode threshold) per phase, contexts past the window.
PHASE_BATCHES = {
    "prefill": ([33, 130], [33, 130], 1),
    "extend": ([40, 3], [400, 19], 1),
    "decode": ([1, 1, 1], [17, 300, 1025], 1),
    "spec-decode": ([4, 4, 2], [40, 300, 77], 4),
    "mixed": ([1, 1, 40, 130], [100, 300, 200, 130], 1),
}


# One model shape per supported head size: 64, 128, 256.
HEAD_SIZE_SHAPES = [MUSE_SPARK_FULL, QWEN3_235B, QWEN3_8_27B]


@pytest.mark.parametrize("phase", list(PHASE_BATCHES))
@pytest.mark.parametrize("feature", list(FEATURES))
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("shape", HEAD_SIZE_SHAPES, ids=lambda s: f"d{s.head_size}")
def test_feature_in_every_phase(shape, feature, phase, kv_cache_dtype):
    """Every attention feature the backend declares gives the right output in
    every phase, i.e. through every kernel that phase can dispatch to, for
    every supported head size."""
    query_lens, seq_lens, threshold = PHASE_BATCHES[phase]
    case = FaCase(shape, kv_cache_dtype=kv_cache_dtype, **FEATURES[feature]).setup(
        blocks_for(seq_lens, 16)
    )
    case.builder.reorder_batch_threshold = threshold
    batch = case.batch(query_lens, seq_lens)
    # Larger logits so the soft cap and ALiBi slopes matter; not with an fp8
    # cache, whose decode error grows with the logits (see fa_harness).
    gain = 1 if kv_cache_dtype == "fp8" else 4
    query = case.query(batch.num_tokens) * gain
    case.check(batch, query, case.run(batch, query))


@pytest.mark.parametrize(
    "query_lens, seq_lens, threshold",
    [
        ([33, 64], [33, 64], 1),
        ([40, 3], [200, 19], 1),
        ([4, 4], [40, 300], 4),
        ([4, 40, 64], [100, 200, 64], 4),
    ],
    ids=["prefill", "extend", "decode", "mixed"],
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_non_causal_in_every_phase(query_lens, seq_lens, threshold, kv_cache_dtype):
    """Non-causal batches (DSpark-style drafts) attend over the whole
    sequence in every phase."""
    case = FaCase(QWEN3_235B, kv_cache_dtype=kv_cache_dtype).setup(
        blocks_for(seq_lens, 16)
    )
    case.builder.reorder_batch_threshold = threshold
    batch = case.batch(query_lens, seq_lens, causal=False)
    query = case.query(batch.num_tokens)
    case.check(batch, query, case.run(batch, query))


def test_kv_update_bypass_layer_reads_new_tokens_from_cache(spy):
    """A layer whose KV write is done by a fused model op (no KV sharing)
    gets key/value None and must gather its new tokens from the cache."""
    q_lens, s_lens = [1, 40, 64], [100, 200, 64]
    case = FaCase(MUSE_SPARK_SLIDING, sliding_window=SMALL_WINDOW)
    case.setup(blocks_for(s_lens, 16))
    case.layers["target"].kv_cache_update_bypass = True
    from vllm.v1.attention.backends.rocm_aiter_fa import (
        AiterFlashAttentionMetadataBuilder,
    )

    case.builder = AiterFlashAttentionMetadataBuilder(
        case.builder.kv_cache_spec, ["target"], case.config, DEVICE
    )
    batch = case.batch(q_lens, s_lens)
    query = case.query(batch.num_tokens)
    output = torch.full_like(query, float("nan"))
    spy.reset()
    case.layer.impl.forward(
        case.layer, query, None, None, case.kv_cache, case.build(batch), output
    )
    assert spy.names().count(GATHER) == 2, spy.names()
    case.check(batch, query, output)


def test_sliding_window_decode_past_table_capacity(spy):
    """More decodes than the persistent window tables hold: the trimmed-table
    kernel is skipped and the result must stay correct."""
    s_lens = [300 + 37 * i for i in range(6)]
    case = FaCase(MUSE_SPARK_SLIDING, sliding_window=SMALL_WINDOW, max_num_seqs=4)
    case.setup(blocks_for(s_lens, 16))
    batch = case.batch([1] * len(s_lens), s_lens)
    query = case.query(batch.num_tokens)
    spy.reset()
    output = case.run(batch, query)
    assert spy.names() == [UNIFIED]
    case.check(batch, query, output)


def test_drafting_metadata_runs_forward():
    """build_for_drafting skips the phase split; forward on it must match a
    regular build for the uniform decode batches drafting produces."""
    q_lens, s_lens = [3, 3, 3], [40, 90, 500]
    case = FaCase(QWEN3_235B).setup(blocks_for(s_lens, 16))
    case.builder.reorder_batch_threshold = 3
    batch = case.batch(q_lens, s_lens)
    query = case.query(batch.num_tokens)
    metadata = case.builder.build_for_drafting(batch.common, draft_index=0)
    case.check(batch, query, case.run(batch, query, metadata))
