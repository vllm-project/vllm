# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill top-k of the DSA sparse attention indexer.

Prefill selects over a chunk of query rows (thousands, eager, never in a CUDA
graph) whose lengths grow causally across the chunk, from the dense logits
of one `fp8_mqa_logits` call. Rows start at column 0 (DeepGEMM's compressed
logits) or mid-row (XPU, AITER), and indices are relative to the row start.
Every kernel must return an exact top-k (`check_topk`).

- Matrix: every prefill kernel x input dtype x chunk shapes x every k it takes
  x row layouts (causal, ragged, offset) x adversarial score distributions.
- `SparseIndexerTopk.forward_prefill`: backend resolution and results.
- ROCm: the AITER backend.
"""

import pytest
import torch
from vllm.utils.torch_utils import set_random_seed

from tests.kernels.indexer_topk_utils import (
    ANY_K,
    BF16,
    DEVICE,
    DTYPE_NAMES,
    F32,
    KINDS,
    NAN_KINDS,
    TOP_KS,
    check_topk,
    distinct_logits,
    enable_aiter_topk,
    make_logits,
    make_topk,
    ragged_lengths,
    rand,
    requires_cuda_alike,
    requires_flashinfer_gvr2,
    requires_rocm,
    requires_sm100,
)

# The per-test gc.collect() and allocator flush cost more than these tests.
pytestmark = pytest.mark.skip_global_cleanup


# ---------------------------------------------------------------------------
# Prefill kernels
# ---------------------------------------------------------------------------


def prefill_rows(
    layout: str, rows: int, width: int, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """(starts, lengths) of a prefill chunk: row r selects over
    logits[r, starts[r]:starts[r] + lengths[r]].

    causal  the chunk's queries are the last `rows` tokens of a `width`
            token context: row r sees width - rows + r + 1 keys (at least 0).
    ragged  path and alignment boundaries (`ragged_lengths`), from column 0.
    offset  packed sequences: rows start mid-row (XPU and AITER logits keep
            absolute columns); row 0 still reaches the width.
    """
    starts = torch.zeros(rows, dtype=torch.int32, device=DEVICE)
    if layout == "causal":
        r = torch.arange(rows, dtype=torch.int32, device=DEVICE)
        return starts, (width - rows + 1 + r).clamp(min=0)
    if layout == "ragged":
        return starts, ragged_lengths(rows, width, k)
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    starts.random_(0, width // 2, generator=gen)
    lengths = (rand(1, rows, gen)[0] * (width - starts)).int()
    lengths[0] = width - int(starts[0])
    return starts, lengths


def prefill_logits(
    kind: str,
    dtype: torch.dtype,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    width: int,
    k: int,
) -> torch.Tensor:
    """`make_logits` with NaN outside every row's span."""
    logits = make_logits(kind, starts.numel(), width, k, dtype, starts + lengths)
    cols = torch.arange(width, device=DEVICE)
    return logits.masked_fill_(cols < starts[:, None], float("nan"))


def run_prefill(
    backend: str,
    logits: torch.Tensor,
    starts: torch.Tensor | None,
    lengths: torch.Tensor,
    out: torch.Tensor,
    k: int,
) -> None:
    """One prefill top-k kernel; indices are relative to each row's start
    (`starts` None: every row starts at column 0)."""
    if backend == "per_row":
        zeros = torch.zeros_like(lengths)
        torch.ops._C.top_k_per_row_prefill(
            logits,
            zeros if starts is None else starts,
            lengths if starts is None else starts + lengths,
            out,
            out.shape[0],
            *logits.stride(),
            k,
        )
    elif backend == "deep_select":
        from vllm.model_executor.layers.indexer_topk import deep_select_topk

        assert starts is None
        deep_select_topk(logits, k, end=lengths, output_idx=out)
    elif backend == "flashinfer_gvr2":
        from flashinfer.topk_varlen.kernels.gvr2_topk_host import run_varlen

        assert starts is None
        run_varlen(
            logits,
            None,
            lengths,
            out,
            next_n=1,
            compress_ratio=1,
            max_seq_len=logits.shape[1],
            top_k=k,
        )
    else:
        raise ValueError(f"unknown prefill top-k backend: {backend}")


# backend: (availability mark, dtypes it reads, (rows, width) chunk shapes,
# k values, row layouts).
PREFILL_BACKENDS = {
    # FP32 logits (BF16 logits are upcast first in production); any k.
    "per_row": (
        requires_cuda_alike,
        (F32,),
        [(1, 8192), (32, 20000), (2050, 20000), (1024, 65536)],
        ANY_K,
        ("causal", "ragged", "offset"),
    ),
    # Ranks BF16 logits in place when sparse_indexer_topk_backend=deep_select.
    "deep_select": (
        requires_sm100,
        (F32, BF16),
        [(256, 65536), (2050, 20000), (1024, 32768)],
        (*ANY_K, 4096),
        ("causal", "ragged"),
    ),
    "flashinfer_gvr2": (
        requires_flashinfer_gvr2,
        (F32,),
        [(256, 65536), (2050, 20000), (1024, 32768)],
        TOP_KS,
        ("causal", "ragged"),
    ),
}

PREFILL_CASES = [
    pytest.param(
        backend,
        dtype,
        rows,
        width,
        k,
        layout,
        marks=mark,
        id=f"{backend}-{DTYPE_NAMES[dtype]}-{rows}x{width}-k{k}-{layout}",
    )
    for backend, (mark, dtypes, shapes, ks, layouts) in PREFILL_BACKENDS.items()
    for dtype in dtypes
    for rows, width in shapes
    for k in ks
    for layout in layouts
]


def known_prefill_failure(backend: str, kind: str, layout: str):
    """The xfail mark of a matrix case a kernel is known to get wrong, or None.

    Strict where the kernel fails on every such input; not strict where
    whether the wrong keys reach the top-k depends on the data.
    """
    nan = kind in NAN_KINDS
    if backend == "deep_select" and nan:
        reason, strict = "DeepSelect writes uninitialized indices for NaN rows", True
    elif backend == "deep_select" and kind == "few_finite" and layout == "ragged":
        reason = "DeepSelect selects past the row length with < k finite scores"
        strict = True
    elif backend == "per_row" and kind == "nan_flood":
        reason, strict = "top_k_per_row_prefill duplicates indices among NaNs", False
    elif backend == "flashinfer_gvr2" and nan:
        reason, strict = "gvr_2 ranks NaN below finite scores", False
    else:
        return None
    return pytest.mark.xfail(reason=reason, strict=strict)


@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize(
    ("backend", "dtype", "rows", "width", "k", "layout"), PREFILL_CASES
)
@torch.inference_mode()
def test_prefill_topk_exact(
    backend: str,
    dtype: torch.dtype,
    rows: int,
    width: int,
    k: int,
    layout: str,
    kind: str,
    request: pytest.FixtureRequest,
) -> None:
    if mark := known_prefill_failure(backend, kind, layout):
        request.applymarker(mark)
    starts, lengths = prefill_rows(layout, rows, width, k)
    logits = prefill_logits(kind, dtype, starts, lengths, width, k)
    out = torch.full((rows, k), -2, dtype=torch.int32, device=DEVICE)
    run_prefill(
        backend, logits, starts if layout == "offset" else None, lengths, out, k
    )
    check_topk(logits, out, lengths, starts)


# Long-context chunks (128K-1M keys): the widths where width-gated prefill
# kernels switch paths (64K and 256K for an FP32 DeepSelect prefill), with the
# kinds that matter at that size (``LONG_KINDS`` and causal-tail scores).
WIDE_PREFILL_SHAPES = [(256, 131072), (128, 262144), (64, 262145), (32, 524288)]
WIDE_PREFILL_SHAPES += [(16, 1048576)]
WIDE_PREFILL_CASES = [
    pytest.param(
        backend,
        dtype,
        rows,
        width,
        k,
        marks=mark,
        id=f"{backend}-{DTYPE_NAMES[dtype]}-{rows}x{width}-k{k}",
    )
    for backend, (mark, dtypes, _, _, _) in PREFILL_BACKENDS.items()
    for dtype in dtypes
    for rows, width in WIDE_PREFILL_SHAPES
    for k in (512, 2048)
]


@pytest.mark.parametrize("layout", ["causal", "ragged"])
@pytest.mark.parametrize(
    "kind", ["gaussian", "rising", "recent", "sample_trap", "block"]
)
@pytest.mark.parametrize(("backend", "dtype", "rows", "width", "k"), WIDE_PREFILL_CASES)
@torch.inference_mode()
def test_prefill_topk_wide(
    backend: str,
    dtype: torch.dtype,
    rows: int,
    width: int,
    k: int,
    kind: str,
    layout: str,
) -> None:
    starts, lengths = prefill_rows(layout, rows, width, k)
    logits = prefill_logits(kind, dtype, starts, lengths, width, k)
    out = torch.full((rows, k), -2, dtype=torch.int32, device=DEVICE)
    run_prefill(backend, logits, None, lengths, out, k)
    check_topk(logits, out, lengths, starts)


# ---------------------------------------------------------------------------
# SparseIndexerTopk
# ---------------------------------------------------------------------------


@requires_cuda_alike
@pytest.mark.parametrize("k", ANY_K)
@pytest.mark.parametrize("with_starts", [False, True])
@pytest.mark.parametrize("rows", [8, 1024])
@pytest.mark.parametrize(
    "backend",
    [
        "auto",
        "per_row",
        pytest.param("flashinfer_gvr2", marks=requires_flashinfer_gvr2),
    ],
)
@torch.inference_mode()
def test_sparse_indexer_prefill_topk(
    backend: str, rows: int, with_starts: bool, k: int
) -> None:
    """gvr_2 takes prefill chunks of at least GVR2_PREFILL_MIN_ROWS rows that
    start at column 0, when requested explicitly and its constraints hold;
    everything else falls back to top_k_per_row_prefill. Either way the
    indices are exact and relative to each row's start."""
    from vllm.model_executor.layers.indexer_topk import GVR2_PREFILL_MIN_ROWS

    starts, lengths = prefill_rows("offset" if with_starts else "ragged", rows, 8192, k)
    logits = prefill_logits("gaussian", F32, starts, lengths, 8192, k)
    row_starts = starts if with_starts else None
    topk = make_topk(backend)
    takes_gvr2 = (
        backend == "flashinfer_gvr2"
        and rows >= GVR2_PREFILL_MIN_ROWS
        and not with_starts
        and k in TOP_KS
    )
    resolved = topk.resolve_prefill_backend(logits, k, row_starts)
    assert resolved == ("flashinfer_gvr2" if takes_gvr2 else "per_row")
    out = torch.full((rows, k), -2, dtype=torch.int32, device=DEVICE)
    topk.forward_prefill(logits, row_starts, starts + lengths, out, k)
    check_topk(logits, out, lengths, starts)


# ---------------------------------------------------------------------------
# ROCm: AITER
# ---------------------------------------------------------------------------


@requires_rocm
@torch.inference_mode()
def test_aiter_c4a_prefill_topk_returns_sequence_local_indices() -> None:
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _launch_aiter_top_k_per_row_prefill,
    )

    row_starts = torch.tensor([1, 6, 20], dtype=torch.int32, device=DEVICE)
    row_ends = torch.tensor([3, 18, 23], dtype=torch.int32, device=DEVICE)
    logits = torch.arange(96, dtype=torch.float32, device=DEVICE).reshape(3, 32)
    indices = torch.empty((3, 4), dtype=torch.int32, device=DEVICE)
    if not rocm_aiter_ops.is_indexer_top_k_supported(
        is_prefill=True, compress_ratio=4, num_rows=logits.shape[0]
    ):
        pytest.skip("AITER top-k is unavailable")
    # The launcher fills `indices` in place; the test pins that it returns nothing.
    assert (
        _launch_aiter_top_k_per_row_prefill(  # type: ignore[func-returns-value]
            logits, row_starts, row_ends, indices, 4
        )
        is None
    )
    check_topk(logits, indices, row_ends - row_starts, row_starts)


@requires_rocm
@torch.inference_mode()
def test_aiter_dsa_prefill_topk_matches_per_row() -> None:
    """DeepSeek-V3.2 / GLM-5 prefill (k=2048): two sequences packed in one
    chunk; AITER's and the in-tree kernel's localized indices are exact."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _launch_aiter_top_k_per_row_prefill,
    )

    top_k = 2048
    if not rocm_aiter_ops.is_indexer_top_k_enabled():
        pytest.skip("AITER top-k is unavailable")
    positions = [(0, p) for p in (0, 1, 2046, 2047, 2048, 4999)]
    positions += [(5000, p) for p in (0, 2500, 2999)]
    starts = torch.tensor([s for s, _ in positions], dtype=torch.int32, device=DEVICE)
    ends = torch.tensor(
        [s + p + 1 for s, p in positions], dtype=torch.int32, device=DEVICE
    )
    rows = len(positions)
    set_random_seed(0)
    logits = distinct_logits(rows, 8000)
    native = torch.full((rows, top_k), -777, dtype=torch.int32, device=DEVICE)
    torch.ops._C.top_k_per_row_prefill(
        logits, starts, ends, native, rows, *logits.stride(), top_k
    )
    aiter = torch.full_like(native, -777)
    _launch_aiter_top_k_per_row_prefill(logits, starts, ends, aiter, top_k)
    for indices in (native, aiter):
        check_topk(logits, indices, ends - starts, starts)


def test_aiter_c4a_prefill_topk_kernel_selection(monkeypatch) -> None:
    from vllm._aiter_ops import rocm_aiter_ops

    enable_aiter_topk(monkeypatch)
    is_supported = rocm_aiter_ops.is_indexer_top_k_supported
    assert is_supported(is_prefill=True, compress_ratio=4, num_rows=1) is True
    assert is_supported(is_prefill=True, compress_ratio=1, num_rows=1) is False

    enable_aiter_topk(monkeypatch, enabled=False)
    assert is_supported(is_prefill=True, compress_ratio=4, num_rows=1) is False


def test_aiter_dsa_prefill_topk_kernel_selection(monkeypatch) -> None:
    """The uncompressed DSA indexer takes AITER for k=2048 on prefill chunks
    whose rows reach 32K tokens; shorter chunks stay on the in-tree kernel."""
    from vllm._aiter_ops import rocm_aiter_ops

    enable_aiter_topk(monkeypatch)
    is_supported = rocm_aiter_ops.is_indexer_top_k_supported

    kwargs = dict(is_prefill=True, num_rows=16_384, max_valid_seq_len=32_768)
    assert is_supported(compress_ratio=1, topk_tokens=2048, **kwargs) is True
    assert is_supported(compress_ratio=1, topk_tokens=1024, **kwargs) is False
    assert is_supported(compress_ratio=1, **kwargs) is False
    for max_valid_seq_len in (None, 32_767):
        assert (
            is_supported(
                is_prefill=True,
                compress_ratio=1,
                num_rows=1,
                max_valid_seq_len=max_valid_seq_len,
                topk_tokens=2048,
            )
            is False
        )

    enable_aiter_topk(monkeypatch, enabled=False)
    assert is_supported(compress_ratio=1, topk_tokens=2048, **kwargs) is False
