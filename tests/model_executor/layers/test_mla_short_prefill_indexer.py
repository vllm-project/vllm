# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer
from vllm.config import CUDAGraphMode
from vllm.models.deepseek_v32 import attention as deepseek_v32_attention
from vllm.models.deepseek_v32.attention import DeepseekV32Attention
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata

INDEXER_LAYER = "model.layers.0.self_attn.indexer.k_cache"
MLA_LAYER = "model.layers.0.self_attn.attn"


def make_indexer_metadata(
    *,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
    num_prefills: int = 1,
    num_prefill_tokens: int = 1,
    slot_mapping: torch.Tensor | None = None,
) -> DeepseekV32IndexerMetadata:
    if slot_mapping is None:
        slot_mapping = torch.zeros(num_prefill_tokens, dtype=torch.long)
    return DeepseekV32IndexerMetadata(
        seq_lens=torch.empty(0, dtype=torch.int32),
        max_seq_len=2048,
        slot_mapping=slot_mapping,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        prefill=SimpleNamespace(chunks=[]) if num_prefills else None,
    )


def make_mla_metadata(*, use_dense_mha: bool = True, num_decode_tokens: int = 0):
    return SimpleNamespace(
        num_decode_tokens=num_decode_tokens,
        prefill=SimpleNamespace(use_dense_mha=use_dense_mha),
    )


@pytest.mark.parametrize(
    "batch_kind",
    [
        "short",
        "threshold_mismatch",
        "force_mqa",
        "mla_decode",
        "capture",
        "full",
    ],
)
def test_short_prefill_updates_k_cache_before_scoring_decision(
    monkeypatch: pytest.MonkeyPatch,
    batch_kind: str,
):
    slot_mapping = torch.tensor([63, 64, 127, 128, -1])
    mla_num_decode_tokens = 1 if batch_kind == "mla_decode" else 0
    runtime_mode = (
        CUDAGraphMode.FULL if batch_kind == "full" else CUDAGraphMode.PIECEWISE
    )
    should_skip = batch_kind in ("short", "threshold_mismatch")
    num_decodes = int(batch_kind == "threshold_mismatch")
    num_decode_tokens = 3 if batch_kind == "threshold_mismatch" else 0
    num_prefills = 0 if batch_kind == "threshold_mismatch" else 2
    num_prefill_tokens = 0 if batch_kind == "threshold_mismatch" else 5
    if batch_kind == "threshold_mismatch":
        # With MTP=3 the indexer threshold is four. A main MLA backend whose
        # threshold is one (for example FlashMLA under DCP) still routes this
        # three-token extend through dense prefill attention.
        slot_mapping = slot_mapping[:3]
    indexer_metadata = make_indexer_metadata(
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        slot_mapping=slot_mapping,
    )
    if indexer_metadata.num_decodes:
        indexer_metadata.decode = object()
    mla_metadata = make_mla_metadata(
        use_dense_mha=batch_kind != "force_mqa",
        num_decode_tokens=mla_num_decode_tokens,
    )

    observed: dict[str, object] = {}

    monkeypatch.setattr(
        sparse_indexer,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={
                INDEXER_LAYER: indexer_metadata,
                MLA_LAYER: mla_metadata,
            },
            cudagraph_runtime_mode=runtime_mode,
        ),
    )
    monkeypatch.setattr(
        sparse_indexer.current_platform, "fp8_dtype", lambda: torch.float16
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: batch_kind == "capture",
    )

    def record_cache_update(k, kv_cache, slots, block_size, scale_fmt):
        observed.update(k=k.clone(), slots=slots)

    monkeypatch.setattr(
        sparse_indexer.ops, "indexer_k_quant_and_cache", record_cache_update
    )

    class ScoringReached(Exception):
        pass

    def scoring_trigger():
        if should_skip:
            pytest.fail("short dense-MHA prefill must not enter indexer scoring")
        raise ScoringReached

    def scoring_decode(*args):
        raise ScoringReached

    monkeypatch.setattr(sparse_indexer, "current_workspace_manager", scoring_trigger)
    monkeypatch.setattr(
        sparse_indexer,
        "kv_cache_as_quant_view",
        scoring_decode,
    )

    hidden_states = torch.full((7, 1), float("inf"))
    k = torch.arange(28, dtype=torch.float32).reshape(7, 4)
    topk_indices = torch.full((7, 2048), 17, dtype=torch.int32)

    def run_indexer():
        assert DeepseekV32Attention.supports_dense_mha_prefill
        return sparse_indexer.sparse_attn_indexer(
            hidden_states,
            INDEXER_LAYER,
            torch.empty(1),
            torch.full((7, 1), float("inf")),
            None,
            k,
            torch.full((7, 1), float("inf")),
            128,
            "ue8m0",
            2048,
            4,
            4096,
            4096,
            topk_indices,
            False,
            False,
            MLA_LAYER,
        )

    if should_skip:
        assert run_indexer() is topk_indices
        assert torch.all(topk_indices == 17)
    else:
        with pytest.raises(ScoringReached):
            run_indexer()
        assert torch.all(topk_indices == -1)

    # K cache is always updated before the scoring decision.
    torch.testing.assert_close(observed["k"], k[: slot_mapping.numel()])
    assert observed["slots"] is slot_mapping


def test_skipped_k_cache_insert_accepts_no_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexer_metadata = make_indexer_metadata(
        num_prefills=0,
        num_prefill_tokens=0,
        slot_mapping=torch.empty(0, dtype=torch.long),
    )
    monkeypatch.setattr(
        sparse_indexer,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={INDEXER_LAYER: indexer_metadata},
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        ),
    )
    monkeypatch.setattr(
        sparse_indexer.current_platform, "fp8_dtype", lambda: torch.float16
    )

    topk_indices = torch.full((1, 2048), 17, dtype=torch.int32)
    result = sparse_indexer.sparse_attn_indexer(
        torch.empty(1, 1),
        INDEXER_LAYER,
        torch.empty(1),
        torch.empty(1, 1),
        None,
        None,
        torch.empty(1, 1),
        128,
        "ue8m0",
        2048,
        4,
        4096,
        4096,
        topk_indices,
        True,
        False,
        "",
    )

    assert result is topk_indices
    assert torch.all(topk_indices == -1)


@pytest.mark.parametrize("fp8_query", [False, True])
def test_deepseek_v32_dispatches_selected_mha(
    monkeypatch: pytest.MonkeyPatch,
    fp8_query: bool,
) -> None:
    attn_metadata = SimpleNamespace(num_actual_tokens=2)
    kv_cache = torch.empty(1)
    monkeypatch.setattr(
        deepseek_v32_attention,
        "get_attention_context",
        lambda _: (attn_metadata, None, kv_cache, None),
    )

    observed = {}

    def record_forward_impl(*args):
        observed["args"] = args

    layer = SimpleNamespace(
        indexer=None,
        skip_topk=False,
        layer_name=MLA_LAYER,
        use_pcp=False,
        _fp8_query=fp8_query,
        _use_sparse_mha=lambda _: True,
        rotary_emb=lambda _positions, q: (q + 1, None),
        forward_impl=record_forward_impl,
    )
    q_nope = torch.randn(2, 1, 2)
    q_pe = torch.randn(2, 1, 2)
    mqa_q = torch.randn(2, 1, 2)
    mha_q = torch.cat((q_nope, q_pe + 1 if fp8_query else mqa_q), dim=-1)
    kv_c = torch.empty(2, 2)
    k_pe = torch.empty(2, 2)
    output = torch.empty(2, 2)

    DeepseekV32Attention._sparse_indexer_and_attn(
        layer,
        torch.arange(2),
        torch.empty(2, 2),
        q_nope,
        q_pe,
        None,
        None,
        None,
        kv_c,
        k_pe,
        torch.empty(2, 1, 2),
        mqa_q,
        output,
    )

    expected_args = (
        mha_q,
        kv_c,
        kv_cache,
        attn_metadata,
        output,
    )
    actual_args = observed["args"]
    torch.testing.assert_close(actual_args[0], mha_q)
    assert actual_args[2].shape == (2, 1, 2)
    assert actual_args[2].data_ptr() == k_pe.data_ptr()
    assert all(
        actual is expected
        for actual, expected in zip(
            actual_args[1:2] + actual_args[3:],
            expected_args[1:],
        )
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_select_candidate_blocks_tolerates_empty_rows():
    """Full-cudagraph decode pads the batch with seq_len-0 rows. The newest
    block pin must not index -1 for them (device-side assert); they select
    no candidate blocks and real rows still pin their newest block."""
    block_size, topk_blocks = 8, 3
    logits = torch.zeros(3, 64, device="cuda")
    logits[0, 3] = 5.0  # block 0 scores highest for row 0
    logits[2, 9] = 5.0  # block 1 scores highest for row 2
    row_ks = torch.zeros(3, dtype=torch.int64, device="cuda")
    row_ke = torch.tensor([40, 0, 17], device="cuda")
    out = torch.empty(3, topk_blocks, dtype=torch.int32, device="cuda")

    sparse_indexer._select_candidate_blocks(
        logits, row_ks, row_ke, topk_blocks, block_size, out
    )

    assert out[1].tolist() == [-1, -1, -1]
    assert out[0, 0].item() == 4 and 0 in out[0].tolist()  # newest block pinned
    assert out[2, 0].item() == 2 and 1 in out[2].tolist()
    assert (out[0] >= 0).all() and (out[2, :2] >= 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "width,block_size,k,decode",
    [
        (73, 8, 16, False),
        (97, 3, 7, False),
        (32768, 8, 2048, False),
        (32768, 8, 2048, True),
    ],
)
def test_candidate_kernels_preserve_packed_bounds_and_padding(
    width, block_size, k, decode
):
    """Preserve top-k ties, newest blocks, empty rows and candidate clamping."""
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        apply_candidate_mask,
        select_candidate_blocks,
    )

    torch.manual_seed(42)
    rows = 6
    logits = torch.randn(rows, width * 2, device="cuda")[:, ::2]
    logits[0, :16] = 0
    logits[2, 5] = float("nan")
    starts = torch.tensor([0, 5, 3, 0, 7, 1], device="cuda", dtype=torch.int32)
    ends = torch.tensor([0, width, width - 1, 1, 11, width], device="cuda")
    repeat = 1
    if decode:
        starts = None
        ends = torch.tensor([0, 1, width], device="cuda", dtype=torch.int32)
        repeat = 2
    ks = torch.zeros(rows, device="cuda", dtype=torch.int64) if decode else starts
    ke = ends.repeat_interleave(repeat)
    cols = torch.arange(width, device="cuda")
    valid = (cols >= ks[:, None]) & (cols < ke[:, None])
    scores = logits.masked_fill(~valid, -torch.inf)
    blocks = ((cols - ks[:, None]) // block_size).clamp(min=0).long()
    nblocks = (width + block_size - 1) // block_size
    reduced = logits.new_full((rows, nblocks), -torch.inf)
    reduced.scatter_reduce_(1, blocks, scores, reduce="amax", include_self=True)
    lengths = ke - ks
    last = ((lengths - 1) // block_size).clamp(min=0).long()
    reduced.scatter_(
        1, last[:, None], torch.where(lengths > 0, torch.inf, -torch.inf)[:, None]
    )
    top = reduced.topk(min(k, nblocks), dim=-1)
    expected = torch.full((rows, k), -1, device="cuda", dtype=torch.int32)
    expected[:, : top.indices.shape[1]] = torch.where(
        top.values > -torch.inf, top.indices, -1
    ).int()
    actual = torch.empty(rows, k * 2, device="cuda", dtype=torch.int32)[:, ::2]
    select_candidate_blocks(logits, starts, ends, k, block_size, actual, repeat)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    candidates = actual.clone()
    candidates[:, 0] = nblocks + 10
    candidates[:, 1] = 0
    candidates[:, 2] = 0
    positions = ks[:, None, None] + candidates.long()[:, :, None] * block_size
    positions = positions + torch.arange(block_size, device="cuda")
    keep = torch.zeros(rows, width, device="cuda", dtype=torch.int8)
    keep.scatter_reduce_(
        1,
        positions.clamp(0, width - 1).reshape(rows, -1),
        (candidates >= 0)[:, :, None]
        .expand(-1, -1, block_size)
        .reshape(rows, -1)
        .to(torch.int8),
        reduce="amax",
        include_self=True,
    )
    reference = logits.masked_fill((keep == 0) | ~valid, -torch.inf)
    apply_candidate_mask(logits, starts, ends, candidates, block_size, repeat)
    torch.testing.assert_close(logits, reference, rtol=0, atol=0, equal_nan=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        apply_candidate_mask(logits, starts, ends, candidates, block_size, repeat)
    for block in (1, -1):
        candidates.fill_(-1)
        candidates[:, 0] = block
        logits.fill_(3.0)
        graph.replay()
        keep = valid & (block >= 0) & ((cols - ks[:, None]) // block_size == block)
        reference = torch.where(keep, 3.0, -torch.inf)
        torch.testing.assert_close(logits, reference, rtol=0, atol=0)
