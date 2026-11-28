# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dtype-aware prefill logits budget of the DSA indexer, and in-place BF16
ranking of the dense prefill logits (DeepSelect) against an FP32 reference."""

import pytest
import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.model_executor.layers import indexer_topk
from vllm.model_executor.layers.indexer_topk import (
    SparseIndexerTopk,
    deep_select_topk,
)
from vllm.model_executor.layers.sparse_attn_indexer import as_fp32_logits
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
    dense_logits_rows,
)

MiB = 1024 * 1024
BUDGET = 512 * MiB

requires_deep_select = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform.is_device_capability_family(100)
    or not indexer_topk._HAS_DEEP_SELECT_C,
    reason="Requires SM100 and vllm._deepselect_C",
)


@pytest.fixture
def sm100(monkeypatch):
    """Pretend to run on SM100 with the DeepSelect extension, so the budget
    math is testable on any machine."""
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(indexer_topk, "_HAS_DEEP_SELECT_C", True)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda capability, device_id=0: capability == 100,
    )


# ---------------------------------------------------------------------------
# Budget math
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend,dcp,expected",
    [
        ("auto", 1, True),
        ("deep_select", 1, True),
        ("auto", 2, False),  # the DCP merge reads FP32 logits
        ("persistent", 1, False),
        ("flashinfer_gvr2", 1, False),
        ("cooperative", 1, False),
    ],
)
def test_prefill_ranks_bf16(sm100, backend, dcp, expected):
    topk = SparseIndexerTopk(backend)
    assert topk.prefill_ranks_bf16(torch.bfloat16, dcp) is expected
    assert not topk.prefill_ranks_bf16(torch.float32, dcp)


def test_prefill_ranks_bf16_needs_deep_select(sm100, monkeypatch):
    monkeypatch.setattr(indexer_topk, "_HAS_DEEP_SELECT_C", False)
    assert not SparseIndexerTopk("auto").prefill_ranks_bf16(torch.bfloat16, 1)
    monkeypatch.setattr(indexer_topk, "_HAS_DEEP_SELECT_C", True)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda capability, device_id=0: capability == 90,
    )
    assert not SparseIndexerTopk("auto").prefill_ranks_bf16(torch.bfloat16, 1)


def test_prefill_ranks_bf16_batch_invariant(sm100, monkeypatch):
    # Batch-invariant mode keeps "auto" on the FP32 per-row top-k.
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    auto = SparseIndexerTopk("auto")
    assert not auto.prefill_ranks_bf16(torch.bfloat16, 1)
    assert auto.prefill_logit_bytes(torch.bfloat16, 1) == 6
    assert SparseIndexerTopk("deep_select").prefill_ranks_bf16(torch.bfloat16, 1)


@pytest.mark.parametrize(
    "dtype,backend,dcp,expected",
    [
        # FP32: DeepGEMM's output is what top-k reads.
        (torch.float32, "auto", 1, 4),
        (torch.float32, "persistent", 2, 4),
        # BF16 ranked in place.
        (torch.bfloat16, "auto", 1, 2),
        (torch.bfloat16, "deep_select", 1, 2),
        # BF16 output plus its FP32 copy.
        (torch.bfloat16, "auto", 2, 6),
        (torch.bfloat16, "persistent", 1, 6),
        (torch.bfloat16, "flashinfer_gvr2", 1, 6),
    ],
)
def test_prefill_logit_bytes(sm100, dtype, backend, dcp, expected):
    assert SparseIndexerTopk(backend).prefill_logit_bytes(dtype, dcp) == expected


def test_prefill_logit_bytes_without_deep_select(sm100, monkeypatch):
    monkeypatch.setattr(indexer_topk, "_HAS_DEEP_SELECT_C", False)
    auto = SparseIndexerTopk("auto")
    assert auto.prefill_logit_bytes(torch.bfloat16, 1) == 6
    assert auto.prefill_logit_bytes(torch.float32, 1) == 4


@pytest.mark.parametrize(
    "width,logit_bytes,expected",
    [
        # 1M tokens at compress ratio 4.
        (262144, 4, 512),
        (262144, 2, 1024),
        (262144, 6, 341),
        (131072, 2, 2048),
        # One row is always allowed, even past the budget.
        (BUDGET, 2, 1),
    ],
)
def test_dense_logits_rows(width, logit_bytes, expected):
    rows = dense_logits_rows(width, BUDGET, logit_bytes)
    assert rows == expected
    assert rows == 1 or rows * width * logit_bytes <= BUDGET
    assert rows == 1 or (rows + 1) * width * logit_bytes > BUDGET


def test_dense_logits_rows_default_is_fp32():
    assert dense_logits_rows(262144, BUDGET) == dense_logits_rows(262144, BUDGET, 4)


def _chunk_logits_bytes(chunks, seq_lens, query_lens, logit_bytes):
    """Peak logits bytes of each planned (request, query) sub-chunk."""
    out = []
    for req, q in chunks:
        n = int(seq_lens[req].sum())
        out.append((q.stop - q.start) * n * logit_bytes)
    return out


@pytest.mark.parametrize("logit_bytes", [2, 4, 6])
@pytest.mark.parametrize("seed", range(4))
def test_split_prefill_chunks_respect_budget(logit_bytes, seed):
    """Every chunk's logits fit the budget at its bytes per element, and every
    query row of every request is planned exactly once."""
    g = torch.Generator().manual_seed(seed)
    n = 24
    seq_lens = torch.randint(1, 40_000, (n,), generator=g, dtype=torch.int32)
    query_lens = torch.minimum(
        torch.randint(1, 4096, (n,), generator=g, dtype=torch.int32), seq_lens
    )
    budget = 64 * MiB
    chunks = DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks(
        seq_lens, query_lens, 1 << 30, budget, logit_bytes=logit_bytes
    )
    for size in _chunk_logits_bytes(chunks, seq_lens, query_lens, logit_bytes):
        assert size <= budget
    covered = torch.zeros(n, dtype=torch.int64)
    for req, q in chunks:
        if req.stop - req.start == 1:
            covered[req.start] += q.stop - q.start
        else:
            assert q.start == 0
            covered[req] += query_lens[req].to(torch.int64)
    assert torch.equal(covered, query_lens.to(torch.int64))


def test_split_prefill_chunks_bf16_in_place_halves_calls():
    """A 1M-token request (262144 compressed keys) with an 8K-token chunk:
    BF16 in place takes half the logits calls of FP32 and a third of BF16
    with its FP32 copy."""
    seq_lens = torch.tensor([262144])
    query_lens = torch.tensor([8192])

    def calls(logit_bytes):
        return len(
            DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks(
                seq_lens, query_lens, 1 << 30, BUDGET, logit_bytes=logit_bytes
            )
        )

    assert calls(4) == 16  # 512 rows per call
    assert calls(2) == 8  # 1024 rows
    assert calls(6) == 25  # 341 rows
    # The default is the FP32 budget.
    assert (
        len(
            DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks(
                seq_lens, query_lens, 1 << 30, BUDGET
            )
        )
        == 16
    )


def test_split_prefill_chunks_packs_by_logit_bytes():
    """Requests pack until M * N * logit_bytes would exceed the budget."""
    seq_lens = torch.tensor([10, 10, 10])
    query_lens = torch.tensor([5, 5, 5])
    split = DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks
    # 1000 B: 250 FP32 elements (two requests: 200), 500 BF16 (three: 450).
    assert split(seq_lens, query_lens, 100, 1000, logit_bytes=4) == [
        (slice(0, 2), slice(0, 10)),
        (slice(2, 3), slice(0, 5)),
    ]
    assert split(seq_lens, query_lens, 100, 1000, logit_bytes=2) == [
        (slice(0, 3), slice(0, 15)),
    ]
    # 166 BF16+copy elements: one request (50) per chunk; two need 200.
    assert split(seq_lens, query_lens, 100, 1000, logit_bytes=6) == [
        (slice(0, 1), slice(0, 5)),
        (slice(1, 2), slice(0, 5)),
        (slice(2, 3), slice(0, 5)),
    ]


# ---------------------------------------------------------------------------
# In-place BF16 ranking vs the FP32 reference
# ---------------------------------------------------------------------------


def _aligned_bf16(rows: int, width: int, device="cuda") -> torch.Tensor:
    """[rows, width] BF16 view with DeepGEMM's 1024-byte row stride."""
    stride = (width + 511) // 512 * 512
    return torch.empty(rows, stride, dtype=torch.bfloat16, device=device)[:, :width]


def _fill(logits: torch.Tensor, case: str, gen: torch.Generator) -> None:
    rows, width = logits.shape
    if case == "random":
        vals = torch.randn(rows, width, generator=gen, device=logits.device)
    elif case == "few_levels":
        # Eight distinct values: thousands of ties at the top-k boundary.
        vals = torch.randint(
            0, 8, (rows, width), generator=gen, device=logits.device
        ).float()
    elif case == "constant":
        vals = torch.full((rows, width), 1.5, device=logits.device)
    elif case == "relu":
        # DeepGEMM's sum of ReLUs: many exact zeros, few positive scores.
        vals = torch.relu(
            torch.randn(rows, width, generator=gen, device=logits.device) - 2.0
        )
    elif case == "bf16_neighbors":
        # 64 adjacent BF16 values in [1, 1.5): dense ties one ULP apart.
        base = torch.randint(
            0, 64, (rows, width), generator=gen, device=logits.device
        ).float()
        vals = 1.0 + base * 2.0**-7
    else:
        raise AssertionError(case)
    logits.copy_(vals.to(torch.bfloat16))


def _check_exact(idx, logits_fp32, row_ends, topk):
    """The selected indices are valid and distinct, and their FP32 scores are
    exactly the FP32 reference top-k scores (any tie order)."""
    rows, width = logits_fp32.shape
    cols = torch.arange(width, device=logits_fp32.device)
    masked = logits_fp32.masked_fill(cols[None] >= row_ends[:, None], -torch.inf)
    k = min(topk, width)
    ref = masked.topk(k, dim=1).values
    visible = row_ends.clamp(max=topk)
    valid = idx >= 0
    assert torch.equal(valid.sum(1), visible.to(valid.sum(1).dtype)), "valid count"
    assert (idx[~valid] == -1).all(), "padding must be -1"
    assert ((~valid) | (idx < row_ends[:, None])).all(), "index past row end"
    srt = idx.masked_fill(~valid, -1).sort(dim=1).values
    assert ((srt[:, 1:] < 0) | (srt.diff(dim=1) > 0)).all(), "duplicate index"
    chosen = masked.gather(1, idx.clamp(min=0).long()).masked_fill(~valid, -torch.inf)
    got = chosen.sort(dim=1, descending=True).values[:, :k]
    assert torch.equal(got, ref), "selected scores differ from the FP32 top-k"


@requires_deep_select
@pytest.mark.parametrize("topk", [512, 1024, 2048])
@pytest.mark.parametrize(
    "case", ["random", "few_levels", "constant", "relu", "bf16_neighbors"]
)
@pytest.mark.parametrize("width", [4096, 65536, 262144])
def test_bf16_in_place_matches_fp32_reference(topk, case, width):
    rows = 32
    gen = torch.Generator(device="cuda").manual_seed(width + topk)
    logits = _aligned_bf16(rows, width)
    _fill(logits, case, gen)
    # Ragged causal rows, some shorter than top-k, one empty, one full.
    row_ends = torch.randint(
        1, width + 1, (rows,), generator=gen, device="cuda", dtype=torch.int32
    )
    row_ends[0], row_ends[1], row_ends[2] = 0, min(topk // 2, width), width
    fp32 = as_fp32_logits(logits, row_ends)

    # This PR: DeepSelect on the BF16 logits in place.
    out = torch.full((rows, topk), -7, dtype=torch.int32, device="cuda")
    deep_select_topk(logits, topk, end=row_ends, output_idx=out)
    _check_exact(out, fp32.float(), row_ends, topk)

    # main: top_k_per_row_prefill on the FP32 copy, same check.
    ref = torch.full((rows, topk), -7, dtype=torch.int32, device="cuda")
    ops.top_k_per_row_prefill(
        fp32,
        torch.zeros_like(row_ends),
        row_ends,
        ref,
        rows,
        fp32.stride(0),
        fp32.stride(1),
        topk,
    )
    _check_exact(ref, fp32.float(), row_ends, topk)


@requires_deep_select
@pytest.mark.parametrize("width", [65536, 262144])
def test_deepgemm_bf16_logits_ranked_in_place(width):
    """DeepGEMM's BF16 prefill logits keep a row stride DeepSelect accepts,
    and ranking them in place selects the FP32 copy's top-k scores."""
    from vllm.model_executor.layers.indexer_topk import is_deep_select_supported
    from vllm.utils.deep_gemm import (
        fp8_fp4_mqa_logits,
        mqa_logits_mx_inputs,
        mqa_logits_use_mx_scales,
    )

    if not mqa_logits_use_mx_scales():
        pytest.skip("BF16 MQA logits need the SM100 MX-scale path")
    topk, rows = 2048, 64
    torch.manual_seed(width)
    q = torch.randn(rows, 32, 128, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn(width, 128, device="cuda").to(torch.float8_e4m3fn)
    k_scale = torch.full((width,), 127 * 0x01010101, device="cuda", dtype=torch.int32)
    weights = torch.randn(rows, 32, device="cuda") / 32
    # Causal: the last rows see every key.
    ends = torch.arange(width - rows + 1, width + 1, device="cuda", dtype=torch.int32)
    starts = torch.zeros_like(ends)
    q_scale, w = mqa_logits_mx_inputs(q, None, weights, torch.bfloat16)
    logits = fp8_fp4_mqa_logits(
        (q, q_scale),
        (k, k_scale),
        w,
        starts,
        ends,
        width,
        logits_dtype=torch.bfloat16,
    )
    assert logits.dtype == torch.bfloat16
    assert is_deep_select_supported(logits, topk)
    row_ends = ends - starts
    fp32 = as_fp32_logits(logits, row_ends)
    out = torch.full((rows, topk), -7, dtype=torch.int32, device="cuda")
    deep_select_topk(logits, topk, end=row_ends, output_idx=out)
    _check_exact(out, fp32.float(), row_ends, topk)
