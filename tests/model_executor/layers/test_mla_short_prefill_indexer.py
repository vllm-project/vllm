# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer
import vllm.models.deepseek_v32.attention as deepseek_attention
from vllm.config import CUDAGraphMode
from vllm.models.deepseek_v32 import attention as deepseek_v32_attention
from vllm.models.deepseek_v32.attention import DeepseekV32Attention
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata

INDEXER_LAYER = "model.layers.0.self_attn.indexer.k_cache"
MLA_LAYER = "model.layers.0.self_attn.attn"


def test_sparse_attention_refreshes_batch_state_inside_eager_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = SimpleNamespace(num_actual_tokens=1)

    class BatchStateRefreshed(Exception):
        pass

    def refresh_batch_state(attn_metadata):
        assert attn_metadata is metadata
        raise BatchStateRefreshed

    layer = SimpleNamespace(
        indexer=None,
        skip_topk=True,
        layer_name=MLA_LAYER,
        impl=SimpleNamespace(
            prepare_for_batch=refresh_batch_state,
            record_logical_topk_ready=lambda: None,
        ),
    )
    monkeypatch.setattr(
        deepseek_attention,
        "get_attention_context",
        lambda layer_name: (metadata, None, torch.empty(1), None),
    )

    with pytest.raises(BatchStateRefreshed):
        DeepseekV32Attention._sparse_indexer_and_attn(
            layer,
            torch.empty(1, dtype=torch.long),
            torch.empty(1, 1),
            torch.empty(1, 1, 1),
            torch.empty(1, 1, 1),
            None,
            None,
            None,
            None,
            None,
            torch.empty(1, 1, 1),
            torch.empty(1, 1, 1),
            torch.empty(1, 1),
        )


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
        impl=SimpleNamespace(
            record_logical_topk_ready=lambda: None,
            prepare_for_batch=lambda _: None,
        ),
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


# --- DeepGEMM sparse-MQA indexer path (V4.1 two-level candidate filtering) ---


def _make_candidates(lens: torch.Tensor, num_candidates: int, cbk: int):
    """Production-faithful candidates: top min(avail, K) blocks per row —
    full coverage for lens <= K*cbk, a fixed-size subset beyond that."""
    avail = ((lens.cpu() + cbk - 1) // cbk).tolist()
    rows = []
    for a in avail:
        n = min(a, num_candidates)
        picks = torch.randperm(a)[:n].sort().values[:num_candidates].tolist()
        rows.append(picks + [-1] * (num_candidates - len(picks)))
    return torch.tensor(rows, dtype=torch.int32)


def _candidate_token_mask(
    candidates: torch.Tensor, cbk: int, total: int, ks: torch.Tensor | None = None
):
    """[rows, total] bool mask of the candidate-covered token positions.

    ``ks`` converts request-local candidate blocks to absolute positions
    (prefill's packed workspace); pass None when already request-local.
    """
    rows = candidates.shape[0]
    device = candidates.device
    # Sentinel block keeps -1 padding from scattering onto real columns.
    width = max(total, (candidates.shape[1] + 1) * cbk)
    if ks is not None:
        width += int(ks.max()) + cbk
    in_cand = torch.zeros(rows, width, dtype=torch.bool, device=device)
    block_id = torch.where(candidates >= 0, candidates.long(), candidates.shape[1])
    tok = block_id.unsqueeze(2) * cbk + torch.arange(cbk, device=device)
    if ks is not None:
        tok = tok + ks.unsqueeze(1).unsqueeze(2)
    keep = (candidates >= 0).unsqueeze(2).expand(-1, -1, cbk)
    in_cand.scatter_(1, tok.flatten(1), keep.flatten(1))
    return in_cand[:, :total]


def _topk_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    total, inter = 0, 0
    for ra, rb in zip(a.cpu(), b.cpu()):
        sa, sb = set(ra[ra >= 0].tolist()), set(rb[rb >= 0].tolist())
        total += len(sb)
        inter += len(sa & sb)
    return inter / max(total, 1)


def test_candidate_blocks_to_sparse_indices_math():
    """Candidate->sparse-block expansion: ks anchoring, range filtering,
    repeat-last padding, and the per-row valid column count (``end``)."""
    from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
        candidate_blocks_to_sparse_indices,
    )

    # Rows cover: exactly one block, an unaligned ks with a partial tail
    # block, an empty row, duplicates / -1 padding / out-of-range garbage
    # (warmup runs touch uninitialized buffers), and int32-overflow junk.
    ks = torch.tensor([64, 13, 100, 8, 0], dtype=torch.int32)
    ke = torch.tensor([72, 30, 100, 40, 1000], dtype=torch.int32)
    candidates = torch.tensor(
        [
            [0, -1, -1, -1],
            [2, 0, -1, -1],
            [3, -1, -1, -1],
            [1, 1, 7, 0],
            [2**30, -(2**30), 0, -1],
        ],
        dtype=torch.int32,
    )
    indices, end = candidate_blocks_to_sparse_indices(
        candidates, ks, ke, candidate_block_size=8, sparse_block_kv=8
    )
    assert indices.tolist() == [
        [8, 8, 8, 8],
        [1, 3, 3, 3],
        [12, 12, 12, 12],
        [1, 2, 2, 2],
        [0, 0, 0, 0],
    ]
    # end: row 1's last block covers [29, 37) but ke = 30 -> one valid token;
    # row 3 keeps the duplicated block (production candidates are unique).
    assert end.tolist() == [8, 9, 0, 24, 8]

    # Wider candidate blocks expand into multiple sparse blocks.
    idx2, end2 = candidate_blocks_to_sparse_indices(
        torch.tensor([[1, 0]]),
        torch.tensor([0]),
        torch.tensor([48]),
        candidate_block_size=16,
        sparse_block_kv=8,
    )
    assert idx2.tolist() == [[0, 1, 2, 3]]
    assert end2.tolist() == [32]


def _skip_unless_sm100_deep_gemm():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 required")
    deep_gemm = pytest.importorskip("deep_gemm")
    if not hasattr(deep_gemm, "fp8_fp4_sparse_mqa_logits"):
        pytest.skip("DeepGEMM >= 2.8 required for sparse MQA logits")
    return deep_gemm


def _quant_fp4(x: torch.Tensor):
    from deep_gemm.utils import per_token_cast_to_fp4

    return per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)


@pytest.mark.parametrize("capture", [False, True])
def test_sparse_mqa_logits_prefill_matches_dense_masked(capture: bool):
    """Sparse-MQA candidate path selects the same top-k as the production
    dense logits + candidate-mask path (incl. under CUDA graph capture)."""
    deep_gemm = _skip_unless_sm100_deep_gemm()
    from vllm import _custom_ops as ops  # noqa: F401  (registers _C topk ops)
    from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
        sparse_mqa_logits_prefill_chunk,
    )

    torch.manual_seed(0)
    rows, num_heads, head_dim = 37, 32, 128
    cbk, topk, num_candidates = 8, 512, 2048  # V4.1-Flash production values
    lens = torch.randint(1, 30000, (rows,), dtype=torch.int32)
    lens[0], lens[1], lens[2] = 1, 5, 2048  # single token, sub-block, long
    ks = torch.zeros(rows, dtype=torch.int32)
    ks[1:] = lens.cumsum(0)[:-1]
    ks, ke, lens = ks.cuda(), (ks + lens).cuda(), lens.cuda()
    total = int(lens.sum())
    q_fp, q_sf = _quant_fp4(
        torch.randn(rows * num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    )
    k_fp, k_sf = _quant_fp4(
        torch.randn(total, head_dim, device="cuda", dtype=torch.bfloat16)
    )
    weights = torch.randn(rows, num_heads, device="cuda", dtype=torch.bfloat16)
    candidates = _make_candidates(lens, num_candidates, cbk).cuda()

    # Reference: production dense path (bf16 logits masked to candidates,
    # then the same radix top-k kernel the dense path uses).
    dense = deep_gemm.fp8_fp4_mqa_logits(
        (
            q_fp.view(torch.int8).view(rows, num_heads, head_dim // 2),
            q_sf.view(rows, num_heads),
        ),
        (k_fp.view(torch.int8), k_sf.view(total)),
        weights,
        ks,
        ke,
        clean_logits=False,
        logits_dtype=torch.bfloat16,
    )
    cols = torch.arange(total, device="cuda")
    in_range = (cols - ks.unsqueeze(1) >= 0) & (
        cols - ks.unsqueeze(1) < lens.unsqueeze(1)
    )
    masked = dense.masked_fill(
        ~(in_range & _candidate_token_mask(candidates, cbk, total, ks)),
        float("-inf"),
    ).float()
    ref = torch.empty(rows, topk, dtype=torch.int32, device="cuda")
    ops.top_k_per_row_prefill(masked, ks, ke, ref, rows, masked.stride(0), 1, topk)

    out = torch.full((rows, topk), -1, dtype=torch.int32, device="cuda")

    def run_sparse():
        sparse_mqa_logits_prefill_chunk(
            q_fp.view(torch.int8).view(rows, num_heads, head_dim // 2),
            q_sf.view(rows, num_heads),
            k_fp.view(torch.int8),
            k_sf.view(total),
            weights,
            ks,
            ke,
            candidates,
            cbk,
            topk,
            out,
        )

    if capture:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            run_sparse()  # warm up JIT + the DeepGEMM workspace pre-capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                run_sparse()
        out.fill_(-1)
        graph.replay()
    else:
        run_sparse()

    overlap = _topk_overlap(out, ref)
    print(f"prefill top-{topk} overlap vs dense masked path: {overlap:.4f}")
    assert overlap >= 0.99


def _build_fp4_paged_cache(req_lens: torch.Tensor, page_kv: int, head_dim: int):
    """MXFP4 paged cache in DeepGEMM's fused layout: per page, all tokens'
    fp4 data first, then their packed UE8M0 scales."""
    device = req_lens.device
    pages_per = (req_lens + page_kv - 1) // page_kv
    num_pages, max_pages = int(pages_per.sum()), int(pages_per.max())
    kv_cache = torch.zeros(
        num_pages, page_kv, 1, head_dim // 2 + 4, dtype=torch.uint8, device=device
    )
    block_table = torch.zeros(
        len(req_lens), max_pages, dtype=torch.int32, device=device
    )
    page_pool = torch.randperm(num_pages, device=device)
    offset = 0
    for b in range(len(req_lens)):
        n = int(pages_per[b])
        block_table[b, :n] = page_pool[offset : offset + n].to(torch.int32)
        offset += n
        k_fp, k_sf = _quant_fp4(
            torch.randn(int(req_lens[b]), head_dim, device=device, dtype=torch.bfloat16)
        )
        rows68 = torch.zeros(n * page_kv, 68, dtype=torch.uint8, device=device)
        rows68[: int(req_lens[b]), : head_dim // 2] = k_fp.view(torch.uint8)
        rows68[: int(req_lens[b]), head_dim // 2 :] = k_sf.view(torch.uint8).view(-1, 4)
        by_page = rows68.view(n, page_kv, 68)
        split = torch.cat(
            [by_page[..., :64].reshape(n, -1), by_page[..., 64:].reshape(n, -1)], dim=1
        ).view(n, page_kv, 1, head_dim // 2 + 4)
        kv_cache[block_table[b, :n].long()] = split
    return kv_cache, block_table


@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("capture", [False, True])
def test_sparse_mqa_logits_paged_matches_dense_masked(next_n: int, capture: bool):
    """Paged sparse-MQA decode path, incl. MTP rows (next_n > 1 flattened to
    one query row each) and CUDA graph capture/replay."""
    deep_gemm = _skip_unless_sm100_deep_gemm()
    from vllm import _custom_ops as _ops  # noqa: F401  (registers _C topk ops)
    from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
        sparse_mqa_logits_paged_decode,
    )

    torch.manual_seed(0)
    batch, num_heads, head_dim, page_kv = 9, 32, 128, 128
    cbk, topk, num_candidates = 8, 512, 2048
    rows = batch * next_n
    req_lens = torch.randint(1, 30000, (batch,), dtype=torch.int32, device="cuda")
    req_lens[0], req_lens[1] = 1, 30000  # tiny row + a filtered (subset) row
    # Row (b, j) sees L_b - next_n + j + 1 tokens (native MTP layout).
    row_lens = (
        (req_lens.unsqueeze(1) - next_n + torch.arange(1, next_n + 1, device="cuda"))
        .clamp(min=0)
        .to(torch.int32)
    )
    kv_cache, block_table = _build_fp4_paged_cache(req_lens, page_kv, head_dim)
    assert kv_cache.stride(0) % 512 == 0  # the paged sparse kernel requires it

    q_fp, q_sf = _quant_fp4(
        torch.randn(rows * num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    )
    q = q_fp.view(torch.int8).view(batch, next_n, num_heads, head_dim // 2)
    q_scale = q_sf.view(batch, next_n, num_heads)
    weights = torch.randn(rows, num_heads, device="cuda", dtype=torch.bfloat16)
    candidates = _make_candidates(row_lens.flatten(), num_candidates, cbk).cuda()

    def reference_topk():
        context_lens_2d = row_lens.view(batch, next_n)
        sched = deep_gemm.get_paged_mqa_logits_metadata(
            context_lens_2d, page_kv, deep_gemm.get_num_sms(), None
        )
        logits = deep_gemm.fp8_fp4_paged_mqa_logits(
            (q, q_scale),
            kv_cache,
            weights,
            context_lens_2d,
            block_table,
            sched,
            int(req_lens.max()),
            False,
            torch.bfloat16,
            None,
        )
        lens_flat = row_lens.flatten()
        cols = torch.arange(logits.shape[1], device="cuda")
        masked = logits.masked_fill(
            ~(
                (cols.unsqueeze(0) < lens_flat.unsqueeze(1))
                & _candidate_token_mask(candidates, cbk, logits.shape[1])
            ),
            float("-inf"),
        ).float()
        ref = torch.empty(rows, topk, dtype=torch.int32, device="cuda")
        workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
        torch.ops._C.persistent_topk(
            masked, lens_flat, ref, workspace, topk, masked.shape[1]
        )
        return ref, logits

    expected, dense_logits = reference_topk()
    out = torch.full((rows, topk), -1, dtype=torch.int32, device="cuda")

    def run_sparse(o: torch.Tensor):
        assert sparse_mqa_logits_paged_decode(
            q,
            q_scale,
            kv_cache,
            weights,
            row_lens.view(batch, next_n),
            block_table,
            None,
            candidates,
            cbk,
            topk,
            o,
        )

    if capture:
        # persistent_topk's tie-breaking among equal values is
        # order-nondeterministic; compare the deterministic selected *value*
        # multisets (the dense and sparse logits are bitwise-equal there).
        eager_out = torch.full((rows, topk), -1, dtype=torch.int32, device="cuda")
        run_sparse(eager_out)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            run_sparse(out)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                run_sparse(out)
        out.fill_(-1)
        graph.replay()
        for r in range(rows):
            for sel in (out, eager_out):
                pass
            vc = dense_logits[r][out[r][out[r] >= 0].long()].sort().values
            ve = dense_logits[r][eager_out[r][eager_out[r] >= 0].long()].sort().values
            assert torch.equal(vc, ve), f"row {r}: replay selected different values"
    else:
        run_sparse(out)
        overlap = _topk_overlap(out, expected)
        print(f"paged top-{topk} overlap vs dense masked path: {overlap:.4f}")
        assert overlap >= 0.99


def test_sparse_mqa_logits_accuracy_vs_torch():
    """Sparse-kernel numerics vs a PyTorch fp32 reference on the dequantized
    MXFP4 inputs, with the dense kernel as a context point."""
    deep_gemm = _skip_unless_sm100_deep_gemm()
    from deep_gemm.utils import cast_back_from_fp4

    from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
        candidate_blocks_to_sparse_indices,
        sparse_topk_remap,
    )
    from vllm.utils.deep_gemm import (
        fp8_fp4_sparse_mqa_logits,
        get_sparse_mqa_logits_metadata,
    )

    torch.manual_seed(0)
    rows, num_heads, head_dim = 17, 32, 128
    sbk, num_candidates = 8, 2048
    lens = torch.randint(1, 1500, (rows,), dtype=torch.int32)
    lens[0] = 1
    ks = torch.zeros(rows, dtype=torch.int32)
    ks[1:] = lens.cumsum(0)[:-1]
    ks, ke, lens = ks.cuda(), (ks + lens).cuda(), lens.cuda()
    total = int(lens.sum())
    q_fp, q_sf = _quant_fp4(
        torch.randn(rows * num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    )
    k_fp, k_sf = _quant_fp4(
        torch.randn(total, head_dim, device="cuda", dtype=torch.bfloat16)
    )
    weights = torch.randn(rows, num_heads, device="cuda", dtype=torch.bfloat16)
    candidates = _make_candidates(lens, num_candidates, sbk).cuda()

    sparse_idx, end = candidate_blocks_to_sparse_indices(candidates, ks, ke, sbk, sbk)
    metadata = get_sparse_mqa_logits_metadata(
        ks, ke, total, sparse_idx, torch.int8, sbk
    )
    logits = fp8_fp4_sparse_mqa_logits(
        (
            q_fp.view(torch.int8).view(rows, num_heads, head_dim // 2),
            q_sf.view(rows, num_heads),
        ),
        (k_fp.view(torch.int8), k_sf.view(total)),
        weights,
        metadata,
        sparse_idx.shape[1],
        sbk,
    )

    # PyTorch fp32 reference on the dequantized inputs.
    q_deq = cast_back_from_fp4(q_fp, q_sf, gran_k=32, use_packed_ue8m0=True)
    k_deq = cast_back_from_fp4(k_fp, k_sf, gran_k=32, use_packed_ue8m0=True)
    dots = torch.einsum(
        "rhd,td->rht",
        q_deq.view(rows, num_heads, head_dim).float(),
        k_deq.float(),
    )
    ref = (dots.relu() * weights.float().unsqueeze(-1)).sum(1)  # [rows, total]

    # Compare on valid sparse columns (absolute packed-workspace positions).
    width = sparse_idx.shape[1] * sbk
    cols = torch.arange(width, device="cuda")
    valid = cols.unsqueeze(0) < end.unsqueeze(1).long()
    pos = (
        sparse_idx.repeat_interleave(sbk, dim=1).long() * sbk
        + (ks % sbk).unsqueeze(1)
        + (cols % sbk).unsqueeze(0)
    ).clamp(min=0, max=total - 1)
    abs_err = (logits.float() - ref.gather(1, pos)).abs()[valid]
    row_scale = ref.abs().amax(dim=1)
    norm_err = ((logits.float() - ref.gather(1, pos)).abs() / row_scale.unsqueeze(1))[
        valid
    ]
    print(
        f"sparse vs torch fp32: max_abs={abs_err.max().item():.4f}; "
        f"rowmax-normalized p50={norm_err.median().item():.5f} "
        f"p99={norm_err.quantile(0.99).item():.5f} max={norm_err.max().item():.5f}"
    )
    # bf16-level numerics: far below the value gaps that change top-k selection.
    assert norm_err.quantile(0.99).item() < 1e-2
    assert norm_err.max().item() < 2e-2

    # Selection quality vs the fp32 reference restricted to the candidates,
    # with the dense kernel as a context point.
    topk = 64
    ref_masked = ref.masked_fill(
        ~_candidate_token_mask(candidates, sbk, total, ks), float("-inf")
    )
    ref_top = ref_masked.topk(topk, dim=1)
    ref_idx = torch.where(
        ref_top.values == float("-inf"), -1, ref_top.indices - ks.unsqueeze(1).long()
    ).to(torch.int32)
    out = torch.full((rows, topk), -1, dtype=torch.int32, device="cuda")
    sparse_topk_remap(logits, sparse_idx, end, ks, sbk, topk, out, decode=False)
    dense_masked = deep_gemm.fp8_fp4_mqa_logits(
        (
            q_fp.view(torch.int8).view(rows, num_heads, head_dim // 2),
            q_sf.view(rows, num_heads),
        ),
        (k_fp.view(torch.int8), k_sf.view(total)),
        weights,
        ks,
        ke,
        clean_logits=False,
        logits_dtype=torch.bfloat16,
    ).masked_fill(~_candidate_token_mask(candidates, sbk, total, ks), float("-inf"))
    dense_top = dense_masked.topk(topk, dim=1)
    dense_idx = torch.where(
        dense_top.values == float("-inf"),
        -1,
        dense_top.indices - ks.unsqueeze(1).long(),
    ).to(torch.int32)
    sparse_ov = _topk_overlap(out, ref_idx)
    dense_ov = _topk_overlap(dense_idx, ref_idx)
    print(
        f"top-{topk} overlap vs torch fp32: "
        f"sparse={sparse_ov:.4f}, dense={dense_ov:.4f}"
    )
    # The sparse path uses vllm's radix top-k while both references here use
    # torch.topk; tie-break order alone costs a few percent at bf16 value
    # granularity, so the meaningful guard is parity with the dense kernel.
    assert sparse_ov >= 0.95
    assert sparse_ov >= dense_ov - 0.02
