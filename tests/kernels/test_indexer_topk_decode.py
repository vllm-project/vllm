# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode top-k of the DSA sparse attention indexer.

Decode selects over one row per query token (`next_n` per request under
MTP), with per-row lengths, inside CUDA graphs, from FP32 or BF16 logits.
Every kernel must return an exact top-k (`check_topk`).

- Matrix: every decode kernel x input dtype x shapes that reach its internal
  paths x every k it takes x adversarial score distributions (`KINDS`).
- Kernel invariants: CUDA-graph replay with new scores and lengths, unaligned
  strides, host-side length bounds, workspace reuse, preallocated outputs.
- `top_k_per_row_decode` with MTP row ends.
- `SparseIndexerTopk`: every backend through the production path (BF16
  upcast when the backend cannot read BF16), backend resolution, decode
  chunking.
- ROCm: `top_k_per_row_decode` on gfx950 and the AITER backend.
"""

import functools

import pytest
import torch
from vllm.utils.torch_utils import set_random_seed

from tests.kernels.indexer_topk_utils import (
    ANY_K,
    BF16,
    DEVICE,
    DTYPE_NAMES,
    F32,
    HAS_COOPERATIVE,
    IS_SM100,
    KINDS,
    LONG_KINDS,
    NAN_KINDS,
    RADIX_TOPK_WORKSPACE_SIZE,
    TOP_KS,
    check_topk,
    distinct_logits,
    enable_aiter_topk,
    has_flashinfer_gvr2,
    has_flashinfer_topk,
    long_lengths,
    make_logits,
    make_topk,
    max_smem_per_block,
    mixed_lengths,
    patch_aiter_topk,
    planted,
    ragged_lengths,
    randn,
    requires_bf16_topk_smem,
    requires_cooperative,
    requires_cuda,
    requires_cuda_alike,
    requires_flashinfer,
    requires_flashinfer_gvr2,
    requires_gfx950,
    requires_rocm,
    requires_sm100,
    shared_logits,
)

# The per-test gc.collect() and allocator flush cost more than these tests.
pytestmark = pytest.mark.skip_global_cleanup


# ---------------------------------------------------------------------------
# Decode kernels
# ---------------------------------------------------------------------------


@functools.cache
def _workspace() -> torch.Tensor:
    return torch.empty(RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=DEVICE)


def per_row_decode(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    k: int,
    next_n: int = 1,
) -> None:
    torch.ops._C.top_k_per_row_decode(
        logits, next_n, seq_lens, out, out.shape[0], *logits.stride(), k
    )


def run_decode(
    backend: str,
    logits: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
    k: int,
    max_seq_len: int,
) -> None:
    """One decode top-k kernel: row r selects over logits[r, :lengths[r]]."""
    if backend == "per_row":
        per_row_decode(logits, lengths, out, k)
    elif backend in ("cooperative", "persistent"):
        op = getattr(torch.ops._C, f"{backend}_topk")
        op(logits, lengths, out, _workspace(), k, max_seq_len)
    elif backend == "deep_select":
        from vllm.model_executor.layers.indexer_topk import deep_select_topk

        deep_select_topk(logits, k, end=lengths.reshape(-1), output_idx=out)
    elif backend == "flashinfer":
        from flashinfer.topk import top_k_ragged_transform

        lengths = lengths.reshape(-1)
        out.copy_(top_k_ragged_transform(logits, torch.zeros_like(lengths), lengths, k))
    elif backend == "flashinfer_gvr2":
        from flashinfer.topk_varlen.kernels.gvr2_topk_host import run_varlen

        run_varlen(
            logits,
            None,
            lengths.reshape(-1),
            out,
            next_n=1,
            compress_ratio=1,
            max_seq_len=logits.shape[1],
            top_k=k,
        )
    elif backend == "decode_topk":
        from vllm.models.deepseek_v41.nvidia.specialized_decode.attention import (
            decode_topk_cute,
        )

        assert decode_topk_cute.supports_decode_topk(logits, out.shape[0], k, False)
        decode_topk_cute.decode_topk(logits, lengths.reshape(-1), out, False)
    else:
        raise ValueError(f"unknown decode top-k backend: {backend}")


# backend: (availability mark, dtypes it reads, (rows, width) shapes that reach
# its paths[, k values; TOP_KS by default]).
DECODE_BACKENDS = {
    "per_row": (
        requires_cuda_alike,
        (F32,),
        [(1, 8192), (6, 65536), (64, 262144)],
        ANY_K,
    ),
    # Clusters of 16 (<= 4 rows on SM100), 8, 4 and 2 CTAs; the register,
    # cluster and streaming (two-pass) paths.
    "cooperative": (
        requires_cooperative,
        (F32, BF16),
        [(1, 262144), (8, 65536), (33, 131072), (64, 32768), (2, 1048576)],
    ),
    # Radix (<= 32 rows), FilteredTopK (33-64 rows, or more with short rows)
    # and sampled_topk (> 64 rows once max_seq_len reaches 98304 for k=512,
    # 65536 otherwise).
    "persistent": (
        requires_cuda_alike,
        (F32, BF16),
        [
            (4, 65536),
            (2, 1048576),
            (48, 65536),
            (80, 16384),
            (65, 98307),
            (128, 131072),
            (512, 8192),
        ],
    ),
    "deep_select": (
        requires_sm100,
        (F32, BF16),
        [(1, 262144), (16, 65536), (128, 131072)],
        (*ANY_K, 4096),
    ),
    "flashinfer": (
        requires_flashinfer,
        (F32,),
        [(1, 262144), (16, 65536), (128, 32768)],
        ANY_K,
    ),
    "flashinfer_gvr2": (
        requires_flashinfer_gvr2,
        (F32,),
        [(1, 262144), (16, 65536), (128, 32768)],
    ),
    "decode_topk": (requires_sm100, (BF16,), [(1, 262144), (12, 65536), (64, 32768)]),
}


def dtype_marks(backend: str, dtype: torch.dtype) -> list:
    if backend == "persistent" and dtype == BF16:
        return [requires_bf16_topk_smem]
    return []


DECODE_CASES = [
    pytest.param(
        backend,
        dtype,
        rows,
        width,
        k,
        marks=[mark, *dtype_marks(backend, dtype)],
        id=f"{backend}-{DTYPE_NAMES[dtype]}-{rows}x{width}-k{k}",
    )
    for backend, (mark, dtypes, shapes, *ks) in DECODE_BACKENDS.items()
    for dtype in dtypes
    for rows, width in shapes
    for k in (ks[0] if ks else TOP_KS)
]


def known_failure(backend: str, rows: int, width: int, kind: str):
    """The xfail mark of a matrix case a kernel is known to get wrong, or None.

    Strict where the kernel fails on every such input; not strict where
    whether the wrong keys reach the top-k depends on the data.
    """
    nan = kind in NAN_KINDS
    if backend == "deep_select" and nan:
        reason, strict = "DeepSelect writes uninitialized indices for NaN rows", True
    elif backend == "deep_select" and kind == "few_finite" and rows > 1:
        reason = "DeepSelect selects past the row length with < k finite scores"
        strict = True
    elif (
        backend == "per_row"
        and kind in ("few_finite", "infs", "nan_neg")
        and width > 65536
    ):
        reason = (
            "top_k_per_row_decode mis-selects short rows holding -inf or a "
            "sign-set NaN on wide logits"
        )
        strict = True
    elif backend in ("flashinfer", "decode_topk") and kind == "nan_neg":
        # #66839 keyed sign-set NaN above +inf in the in-tree kernels only.
        reason, strict = f"{backend} ranks a sign-set NaN below +inf", True
    elif backend == "per_row" and kind == "nan_flood":
        reason, strict = "top_k_per_row_decode duplicates indices among NaNs", False
    elif backend == "flashinfer_gvr2" and nan:
        reason, strict = "gvr_2 ranks NaN below finite scores", False
    else:
        return None
    return pytest.mark.xfail(reason=reason, strict=strict)


@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize(("backend", "dtype", "rows", "width", "k"), DECODE_CASES)
@torch.inference_mode()
def test_decode_topk_exact(
    backend: str,
    dtype: torch.dtype,
    rows: int,
    width: int,
    kind: str,
    k: int,
    request: pytest.FixtureRequest,
) -> None:
    if mark := known_failure(backend, rows, width, kind):
        request.applymarker(mark)
    lengths = ragged_lengths(rows, width, k)
    logits = make_logits(kind, rows, width, k, dtype, lengths)
    out = torch.full((rows, k), -2, dtype=torch.int32, device=DEVICE)
    run_decode(backend, logits, lengths, out, k, width)
    check_topk(logits, out, lengths)


@pytest.mark.parametrize(("backend", "dtype", "rows", "width", "k"), DECODE_CASES)
@torch.inference_mode()
def test_decode_topk_graph_replay(
    backend: str, dtype: torch.dtype, rows: int, width: int, k: int
) -> None:
    """A captured launch stays exact when replayed on new scores and lengths.

    Captured on a side stream after one eager run there, since gvr_2 keeps a
    per-stream scratch buffer that it can only allocate outside capture.
    """
    lengths = ragged_lengths(rows, width, k)
    logits = make_logits("gaussian", rows, width, k, dtype, lengths)
    out = torch.full((rows, k), -2, dtype=torch.int32, device=DEVICE)
    # The workspace kernels also take (batch, next_n) lengths.
    if backend in ("cooperative", "persistent") and rows % 4 == 0:
        lengths = lengths.view(rows // 4, 4)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run_decode(backend, logits, lengths, out, k, width)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run_decode(backend, logits, lengths, out, k, width)

    sampled = rows > 64 and width >= (98304 if k == 512 else 65536)
    for step, kind in enumerate(["ramp", "ties", "rising"]):
        new = ragged_lengths(rows, width, k).roll(step + 1)
        if backend == "persistent":
            # Negative lengths count as 0; sampled_topk also clamps lengths
            # past max_seq_len.
            new[1::7] = -1
            if sampled:
                new[2::7] = width + 17
        lengths.copy_(new.view_as(lengths))
        logits.copy_(make_logits(kind, rows, width, k, dtype, new, seed=step))
        out.fill_(-2)
        graph.replay()
        check_topk(logits, out, new)


@pytest.mark.parametrize("extra", [1, 2, 3, 4])
@pytest.mark.parametrize("num_finite", [0, 100])
@pytest.mark.parametrize("fill", [float("-inf"), -1e30])
@pytest.mark.parametrize(
    ("backend", "dtype", "k"),
    [pytest.param("per_row", F32, k, marks=requires_cuda_alike) for k in ANY_K]
    + [
        pytest.param(
            "persistent",
            dtype,
            k,
            marks=[requires_cuda_alike, *dtype_marks("persistent", dtype)],
        )
        for dtype in (F32, BF16)
        for k in TOP_KS
    ],
)
@torch.inference_mode()
def test_decode_topk_unaligned_stride(
    backend: str, dtype: torch.dtype, k: int, extra: int, num_finite: int, fill: float
) -> None:
    """Rows of k + extra scores at an odd stride, with finite padding past
    each row: the vector tail lanes past the row must never be selected. With
    fewer than k scores above `fill`, the threshold lands in its bin."""
    rows, length = 4, k + extra
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    logits = torch.zeros(rows, length + 8, device=DEVICE)
    finite = randn(rows, num_finite, gen)
    logits[:, :length] = planted(
        torch.full((rows, length), fill, device=DEVICE), gen, finite
    )
    logits = logits.to(dtype)
    lengths = torch.full((rows,), length, dtype=torch.int32, device=DEVICE)
    out = torch.full((rows, k), -2, dtype=torch.int32, device=DEVICE)
    run_decode(backend, logits, lengths, out, k, length)
    check_topk(logits, out, lengths)


@requires_cuda_alike
@pytest.mark.parametrize("k", TOP_KS)
@pytest.mark.parametrize(
    "dtype", [F32, pytest.param(BF16, marks=requires_bf16_topk_smem)]
)
@pytest.mark.parametrize(
    ("rows", "width", "bound"),
    [(33, 1025, 1025), (128, 8192, 1 << 20), (128, 262144, 1 << 20)],
)
@torch.inference_mode()
def test_persistent_topk_host_bound(
    rows: int, width: int, bound: int, dtype: torch.dtype, k: int
) -> None:
    """persistent_topk's max_seq_len is a host-side upper bound on the row
    lengths (it gates sampled_topk), which may exceed the logits width."""
    stride = (width + 7) // 8 * 8
    logits = torch.empty_strided((rows, width), (stride, 1), dtype=dtype, device=DEVICE)
    lengths = ragged_lengths(rows, width, k)
    logits.copy_(make_logits("gaussian", rows, width, k, dtype, lengths))
    out = torch.full((rows, k), -2, dtype=torch.int32, device=DEVICE)
    run_decode("persistent", logits, lengths, out, k, bound)
    check_topk(logits, out, lengths)


@requires_cooperative
@pytest.mark.parametrize("k", TOP_KS)
@torch.inference_mode()
def test_cooperative_topk_tie_workspace_is_per_row(k: int) -> None:
    """Row 1's exact ties must not overwrite row 0's tie workspace."""
    width = 65536
    lengths = torch.tensor([40960, 65536], dtype=torch.int32, device=DEVICE)
    logits = torch.full((2, width), float("-inf"), device=DEVICE)
    logits[0, :2048] = -10.0
    logits[0, 2048:40960] = 1.0
    logits[1] = 2.0
    out = torch.full((2, k), -2, dtype=torch.int32, device=DEVICE)
    run_decode("cooperative", logits, lengths, out, k, width)
    check_topk(logits, out, lengths)


@requires_cuda_alike
@pytest.mark.parametrize("k", TOP_KS)
@torch.inference_mode()
def test_persistent_topk_reused_group_after_short_row(k: int) -> None:
    """A short row must not advance a reused CTA group's histogram ring."""
    long_len, radix = 32769, 256
    props = torch.cuda.get_device_properties(0)
    max_smem = max_smem_per_block()
    if max_smem <= props.shared_memory_per_multiprocessor // 2:
        pytest.skip("Cannot force one persistent_topk CTA per SM")
    fixed_smem = ((radix + radix + 5) * 4 + 15) & ~15
    max_chunk = ((max_smem - fixed_smem) // 4 // 4) * 4
    ctas_per_group = max(
        (props.multi_processor_count - 1 + 9) // 10,
        (long_len + max_chunk - 1) // max_chunk,
    )
    if ctas_per_group >= props.multi_processor_count:
        pytest.skip("Not enough SMs to construct a reused CTA group")
    stride = ctas_per_group * max_chunk
    groups = max(1, (props.multi_processor_count - 1) // ctas_per_group)
    lengths = torch.full((3 * groups,), k, dtype=torch.int32, device=DEVICE)
    lengths[0] = lengths[2 * groups] = long_len
    lengths[groups] = long_len - 1
    set_random_seed(0)
    logits = torch.randn(3 * groups, stride, device=DEVICE)
    out = torch.full((3 * groups, k), -2, dtype=torch.int32, device=DEVICE)
    run_decode("persistent", logits, lengths, out, k, stride)
    check_topk(logits, out, lengths)


@requires_sm100
@pytest.mark.parametrize("k", (*ANY_K, 4096))
@torch.inference_mode()
def test_deep_select_topk_preallocated_output(k: int) -> None:
    """DeepSelect writes into a slice of a wider buffer (vLLM's
    topk_indices_buffer), without `end`, and nowhere else."""
    from vllm.model_executor.layers.indexer_topk import deep_select_topk

    rows, width = 8, 131072
    lengths = torch.full((rows,), width, dtype=torch.int32, device=DEVICE)
    logits = make_logits("gaussian", rows, width, k, F32, lengths)
    buffer = torch.full((rows, k + 64), -2, dtype=torch.int32, device=DEVICE)
    out = buffer[:, :k]
    assert deep_select_topk(logits, k, output_idx=out).data_ptr() == out.data_ptr()
    assert torch.all(buffer[:, k:] == -2), "kernel wrote out of bounds"
    check_topk(logits, out, lengths)


@requires_cuda_alike
@pytest.mark.parametrize("k", ANY_K)
@pytest.mark.parametrize("kind", ["gaussian", "lsb10"])
@pytest.mark.parametrize("per_row_lens", [False, True])
@pytest.mark.parametrize(
    ("batch", "next_n", "width"),
    [(1, 1, 20000), (2048, 1, 20000), (2, 8, 20000), (512, 8, 20000), (2, 2, 300000)],
)
@torch.inference_mode()
def test_top_k_per_row_decode(
    batch: int, next_n: int, width: int, per_row_lens: bool, kind: str, k: int
) -> None:
    """MTP rows: seq_lens is (batch,) with row j of a request ending at
    seq_len - next_n + j + 1 (clamped at 0), or (batch, next_n) per-row
    lengths."""
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    seq_lens = torch.randint(
        0, width + 1, (batch,), generator=gen, device=DEVICE, dtype=torch.int32
    )
    offsets = torch.arange(next_n, dtype=torch.int32, device=DEVICE)
    row_ends = (seq_lens[:, None] - next_n + 1 + offsets).clamp(min=0)
    rows = batch * next_n
    logits = make_logits(kind, rows, width, k, F32, row_ends)
    out = torch.full((rows, k), -2, dtype=torch.int32, device=DEVICE)
    per_row_decode(logits, row_ends if per_row_lens else seq_lens, out, k, next_n)
    check_topk(logits, out, row_ends)


# ---------------------------------------------------------------------------
# Long rows (512K-1M): launch-shape boundaries and ragged batches
# ---------------------------------------------------------------------------

# Row counts on both sides of each launch-shape switch, with 512K-1M rows.
# cooperative_topk: 16-CTA clusters up to 4 rows (SM100), 8 up to 8, 4 up to
# 33, then 2-CTA clusters (34 rows is a measured perf cliff) up to its 64-row
# limit. persistent_topk: radix up to 32 rows, FilteredTopK up to 64, then
# sampled_topk (rows > 64 with max_seq_len >= 98304 for k=512, else 65536).
# 15/16, 22/23 and 49/50 are cluster-wave edges on 148-SM parts.
BOUNDARY_ROWS = (4, 5, 8, 9, 15, 16, 22, 23, 32, 33, 34, 49, 50, 64, 65)
# (dtype, k): GLM-5.x (FP32 or BF16 logits, k=2048) and DSv4.1 (BF16, k=512).
LONG_DTYPE_K = [
    pytest.param(F32, 2048, id="fp32-k2048"),
    pytest.param(BF16, 2048, id="bf16-k2048", marks=requires_bf16_topk_smem),
    pytest.param(BF16, 512, id="bf16-k512", marks=requires_bf16_topk_smem),
]
LONG_WIDTH_KINDS = [(524288, "gaussian"), (524288, "recent")] + [
    (1048576, kind) for kind in LONG_KINDS
]
WORKSPACE_BACKENDS = [
    pytest.param("cooperative", marks=requires_cooperative),
    pytest.param("persistent", marks=requires_cuda_alike),
]


# The data parameters vary slowest (bottom decorators), so consecutive tests
# share one input (shared_logits) across backends and row counts.
@pytest.mark.parametrize("rows", BOUNDARY_ROWS)
@pytest.mark.parametrize("backend", WORKSPACE_BACKENDS)
@pytest.mark.parametrize(("dtype", "k"), LONG_DTYPE_K)
@pytest.mark.parametrize(("width", "kind"), LONG_WIDTH_KINDS)
@torch.inference_mode()
def test_decode_topk_long_row_launch_boundaries(
    width: int, kind: str, dtype: torch.dtype, k: int, backend: str, rows: int
) -> None:
    """Long rows on both sides of every launch-shape switch of the workspace
    kernels, including the 2-CTA clusters and the > 64-row sampled path."""
    if backend == "cooperative" and rows > 64:
        pytest.skip("cooperative_topk takes at most 64 rows")
    lengths = long_lengths(max(BOUNDARY_ROWS), width)
    logits = shared_logits(kind, lengths.numel(), width, k, dtype, lengths)[:rows]
    lengths = lengths[:rows]
    out = torch.full((rows, k), -2, dtype=torch.int32, device=DEVICE)
    run_decode(backend, logits, lengths, out, k, int(lengths.max()))
    check_topk(logits, out, lengths)


def _ragged_long_batches() -> list:
    """(id, lengths) of ragged 1M-wide decode batches: three rows at and next
    to a length threshold of split-row / dynamic-split kernels (448K, 512K,
    640K) among short rows, and 1-8 long rows among 24-48 short rows."""
    batches = []
    for t in (458752, 524288, 655360):
        for short in (21, 45):
            batches.append(
                (f"t{t}-s{short}", lambda t=t, s=short: ([t - 1, t, t + 1], s))
            )
    for long in (1, 2, 4, 8):
        for short in (24, 48):
            batches.append(
                (f"l{long}-s{short}", lambda n=long, s=short: ([1040001] * n, s))
            )
    return [pytest.param(make, id=name) for name, make in batches]


RAGGED_LONG_BACKENDS = [
    pytest.param("cooperative", marks=requires_cooperative),
    pytest.param("persistent", marks=requires_cuda_alike),
    pytest.param("deep_select", marks=requires_sm100),
    pytest.param("flashinfer_gvr2", marks=requires_flashinfer_gvr2),
    pytest.param("decode_topk", marks=requires_sm100),
]


@pytest.mark.parametrize("backend", RAGGED_LONG_BACKENDS)
@pytest.mark.parametrize(
    ("dtype", "k"),
    [
        pytest.param(F32, 2048, id="fp32-k2048"),
        pytest.param(BF16, 512, id="bf16-k512", marks=requires_bf16_topk_smem),
    ],
)
@pytest.mark.parametrize("kind", ["gaussian", "recent", "sample_trap"])
@pytest.mark.parametrize("batch", _ragged_long_batches())
@torch.inference_mode()
def test_decode_topk_ragged_long_rows(
    batch, kind: str, dtype: torch.dtype, k: int, backend: str
) -> None:
    """A few long rows set the cost of a batch of short ones; rows exactly
    at a split threshold and one column either side must stay exact. The
    logits are as wide as max_model_len (1M)."""
    width = 1048576
    long, short = batch()
    lengths = mixed_lengths(long, short, k)
    rows = lengths.numel()
    if backend == "flashinfer_gvr2" and dtype != F32:
        pytest.skip("gvr_2 reads FP32 logits")
    if backend == "decode_topk" and dtype != BF16:
        pytest.skip("decode_topk reads BF16 logits")
    logits = shared_logits(kind, rows, width, k, dtype, lengths)
    if backend == "decode_topk":
        from vllm.models.deepseek_v41.nvidia.specialized_decode.attention import (
            decode_topk_cute,
        )

        if not decode_topk_cute.supports_decode_topk(logits, rows, k, False):
            pytest.skip("decode_topk does not take this shape")
    out = torch.full((rows, k), -2, dtype=torch.int32, device=DEVICE)
    run_decode(backend, logits, lengths, out, k, int(lengths.max()))
    check_topk(logits, out, lengths)


# ---------------------------------------------------------------------------
# SparseIndexerTopk
# ---------------------------------------------------------------------------


SPARSE_INDEXER_BACKENDS = {
    "auto": (requires_cuda_alike, ANY_K),
    "torch": (requires_cuda_alike, ANY_K),
    "per_row": (requires_cuda_alike, ANY_K),
    "persistent": (requires_cuda_alike, TOP_KS),
    "cooperative": (requires_cooperative, TOP_KS),
    "flashinfer": (requires_flashinfer, ANY_K),
    "flashinfer_gvr2": (requires_flashinfer_gvr2, TOP_KS),
    "deep_select": (requires_sm100, (*ANY_K, 4096)),
}


SPARSE_INDEXER_CASES = [
    pytest.param(backend, k, marks=mark, id=f"{backend}-k{k}")
    for backend, (mark, ks) in SPARSE_INDEXER_BACKENDS.items()
    for k in ks
]


@requires_cuda_alike
@pytest.mark.parametrize("dtype", [F32, BF16])
@pytest.mark.parametrize(
    ("next_n", "per_request_lens"), [(1, False), (4, False), (4, True)]
)
@pytest.mark.parametrize(("backend", "k"), SPARSE_INDEXER_CASES)
@torch.inference_mode()
def test_sparse_indexer_decode_topk(
    backend: str,
    k: int,
    next_n: int,
    per_request_lens: bool,
    dtype: torch.dtype,
    workspace_init,
) -> None:
    """Every backend through the production path: BF16 logits go to the
    kernel as-is when the backend reads BF16, otherwise through
    as_fp32_logits (which leaves the columns past each row uninitialized).
    seq_lens is (batch, next_n) per-row lengths, or (batch,) request lengths
    whose derived row ends clamp at 0 (every backend expands them)."""
    from vllm.model_executor.layers.sparse_attn_indexer import as_fp32_logits

    width = 16384
    seq_lens = torch.tensor([2, 3000, 9001, width], dtype=torch.int32, device=DEVICE)
    offsets = torch.arange(next_n, dtype=torch.int32, device=DEVICE)
    row_ends = (seq_lens[:, None] - next_n + 1 + offsets).clamp(min=0).flatten()
    rows = row_ends.numel()
    logits = make_logits("gaussian", rows, width, k, dtype, row_ends)
    topk = make_topk(backend)
    x = logits
    if dtype == BF16 and not topk.takes_bf16_logits(logits, k, rows):
        x = as_fp32_logits(logits, row_ends)
    out = torch.full((rows, k), -2, dtype=torch.int32, device=DEVICE)
    lens = seq_lens if per_request_lens else row_ends.view(-1, next_n)
    topk(x, lens, next_n, out, k, int(row_ends.max()))
    check_topk(logits, out, row_ends)


# MTP batches around cooperative_topk's 64-row limit at 1M: "auto" switches
# to persistent_topk past it, which takes sampled_topk there.
# Data parameters at the bottom: consecutive tests share one input.
@pytest.mark.parametrize("per_request_lens", [False, True])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("auto", marks=requires_cuda_alike),
        pytest.param("persistent", marks=requires_cuda_alike),
    ],
)
@pytest.mark.parametrize(
    ("batch", "next_n"),
    [
        pytest.param(16, 4, id="64rows"),
        pytest.param(13, 5, id="65rows"),
        pytest.param(33, 2, id="66rows"),
        pytest.param(17, 4, id="68rows"),
    ],
)
@pytest.mark.parametrize(("dtype", "k"), LONG_DTYPE_K)
@pytest.mark.parametrize("kind", ["gaussian", "recent", "sample_trap"])
@torch.inference_mode()
def test_sparse_indexer_mtp_long_rows(
    kind: str,
    dtype: torch.dtype,
    k: int,
    backend: str,
    per_request_lens: bool,
    batch: int,
    next_n: int,
    workspace_init,
) -> None:
    """MTP decode at 1M context through SparseIndexerTopk (production BF16
    handling), across the 64-row switch to sampled_topk."""
    from vllm.model_executor.layers.indexer_topk import AUTO_COOPERATIVE_MAX_ROWS
    from vllm.model_executor.layers.sparse_attn_indexer import as_fp32_logits

    width = 1048576
    gen = torch.Generator().manual_seed(batch)
    seq_lens = (1040001 - torch.randint(0, 65536, (batch,), generator=gen)).to(
        dtype=torch.int32, device=DEVICE
    )
    offsets = torch.arange(next_n, dtype=torch.int32, device=DEVICE)
    row_ends = (seq_lens[:, None] - next_n + 1 + offsets).flatten()
    rows = row_ends.numel()
    logits = shared_logits(kind, rows, width, k, dtype, row_ends)
    topk = make_topk(backend)
    resolved = topk.resolve_backend(logits, k, rows)
    if backend == "auto" and HAS_COOPERATIVE:
        expected = "cooperative" if rows <= AUTO_COOPERATIVE_MAX_ROWS else "persistent"
        assert resolved == expected
    x = logits
    if dtype == BF16 and not topk.takes_bf16_logits(logits, k, rows):
        x = as_fp32_logits(logits, row_ends)
    out = torch.full((rows, k), -2, dtype=torch.int32, device=DEVICE)
    lens = seq_lens.view(-1, 1) if per_request_lens else row_ends.view(-1, next_n)
    topk(x, lens, next_n, out, k, int(row_ends.max()))
    check_topk(logits, out, row_ends)


@requires_cuda
@torch.inference_mode()
def test_sparse_indexer_topk_backend_resolution() -> None:
    """Auto heuristic chain and fail-fast validation of explicit backends."""
    from vllm.model_executor.layers.indexer_topk import AUTO_COOPERATIVE_MAX_ROWS

    # 1024B-aligned stride: 131072 * 4 bytes.
    logits = torch.randn(64, 131072, dtype=torch.float32, device=DEVICE)
    # 16B- but not 1024B-aligned rows: cooperative yes, DeepSelect no.
    unaligned_logits = torch.randn(128, 131076, device=DEVICE)[:, :131072]

    def resolve(backend, t=None, k=2048, num_rows=64) -> str:
        return make_topk(backend).resolve_backend(
            logits if t is None else t, k, num_rows
        )

    if has_flashinfer_gvr2():
        from vllm.model_executor.layers.sparse_attn_indexer import as_fp32_logits

        assert resolve("flashinfer_gvr2", num_rows=1) == "flashinfer_gvr2"
        with pytest.raises(RuntimeError, match="topk_tokens must be in"):
            resolve("flashinfer_gvr2", k=3000)
        with pytest.raises(RuntimeError, match="fp32"):
            resolve("flashinfer_gvr2", logits.to(torch.bfloat16))
        # Row pitch not a multiple of 4 (float4 loads).
        odd_pitch = torch.randn(64, 131074, device=DEVICE)[:, :131072]
        with pytest.raises(RuntimeError, match="multiple of 4"):
            resolve("flashinfer_gvr2", odd_pitch)

        # A strided view whose last row ends at its width cannot be read as
        # [rows, stride]; as_fp32_logits output always can.
        def short_tail(dtype):
            return torch.empty_strided(
                (64, 131072), (131328, 1), dtype=dtype, device=DEVICE
            )

        with pytest.raises(RuntimeError, match="full stride"):
            resolve("flashinfer_gvr2", short_tail(torch.float32))
        upcast = as_fp32_logits(
            short_tail(torch.bfloat16),
            torch.full((64,), 131072, dtype=torch.int32, device=DEVICE),
        )
        assert upcast.stride(0) == 131328
        assert resolve("flashinfer_gvr2", upcast) == "flashinfer_gvr2"
    else:
        with pytest.raises(RuntimeError, match="gvr_2|SM100|CUDA"):
            resolve("flashinfer_gvr2")

    # "auto" is the chain cooperative -> persistent -> per_row, switching at
    # cooperative_topk's row limit. It never selects an opt-in backend.
    if HAS_COOPERATIVE:
        for t in (logits, unaligned_logits):
            assert resolve("auto", t, 512, AUTO_COOPERATIVE_MAX_ROWS) == "cooperative"
        assert resolve("auto", k=512, num_rows=8) == "cooperative"
        assert resolve("auto", k=512, num_rows=AUTO_COOPERATIVE_MAX_ROWS + 1) == (
            "persistent"
        )
        assert resolve("auto", k=512, num_rows=128) == "persistent"
    assert resolve("auto", unaligned_logits, num_rows=128) == "persistent"
    # k outside {512, 1024, 2048} -> per_row.
    assert resolve("auto", k=5000) == "per_row"

    # Explicit values are returned as-is when constraints are met.
    for backend in ("per_row", "torch", "persistent"):
        assert resolve(backend) == backend
    if HAS_COOPERATIVE:
        assert resolve("cooperative") == "cooperative"
    if has_flashinfer_topk():
        assert resolve("flashinfer") == "flashinfer"
    if IS_SM100:
        # Explicit deep_select has no row-count requirement.
        assert resolve("deep_select", num_rows=1) == "deep_select"
        with pytest.raises(RuntimeError, match="constraints"):
            resolve("deep_select", unaligned_logits)

    # Fail-fast on unmet constraints.
    with pytest.raises(RuntimeError, match="num_rows must be <= 64"):
        resolve("cooperative", num_rows=128)
    for backend in ("persistent", "cooperative"):
        with pytest.raises(RuntimeError, match="topk_tokens must be in"):
            resolve(backend, k=3000)
    if not IS_SM100:
        with pytest.raises(RuntimeError, match="SM100a/SM103a"):
            resolve("deep_select")

    def takes_bf16(backend: str, t: torch.Tensor) -> bool:
        return make_topk(backend).takes_bf16_logits(t, 2048, 64)

    # cooperative_topk reads BF16 rows as-is when they are 16B-aligned.
    assert takes_bf16("cooperative", logits.to(torch.bfloat16)) == HAS_COOPERATIVE
    # Row stride 131076 is 8B- but not 16B-aligned in BF16.
    bf16_unaligned = torch.randn(64, 131076, device=DEVICE).bfloat16()[:, :131072]
    assert not takes_bf16("cooperative", bf16_unaligned)


@requires_cuda
@pytest.mark.parametrize("k", TOP_KS)
@pytest.mark.parametrize("backend", ["auto", "persistent"])
@pytest.mark.parametrize("rows", [8, 64, 65, 128])
@torch.inference_mode()
def test_sparse_indexer_resolved_persistent_bf16(
    backend: str, rows: int, k: int, workspace_init
) -> None:
    """Backend selection and exact scores across the BF16 row boundary."""
    set_random_seed(0)
    width = 32768
    logits = torch.empty_strided(
        (rows, width), (33792, 1), dtype=torch.bfloat16, device=DEVICE
    ).normal_()
    fp32 = logits.float()
    ends = torch.full((rows, 1), width, dtype=torch.int32, device=DEVICE)
    selector = make_topk(backend)
    resolved = selector.resolve_backend(logits, k, rows)
    assert resolved == (
        "cooperative"
        if backend == "auto" and rows <= 64 and HAS_COOPERATIVE
        else "persistent"
    )
    assert selector.resolve_backend(fp32, k, rows) == resolved
    takes_bf16 = selector.takes_bf16_logits(logits, k, rows)
    assert takes_bf16 == (resolved == "cooperative" or IS_SM100)
    assert not selector.takes_bf16_logits(fp32, k, rows)
    assert not selector.takes_bf16_logits(logits[:, ::2], k, rows)
    assert not selector.takes_bf16_logits(logits, 256, rows)
    for x in (logits if takes_bf16 else fp32, fp32):
        out = torch.full((rows, k), -2, dtype=torch.int32, device=DEVICE)
        selector(x, ends, 1, out, k, width)
        check_topk(fp32, out, ends)


@pytest.mark.parametrize("itemsize", [2, 4])
@pytest.mark.parametrize(
    "rows,capacity",
    [(33, 32), (65, 64), (96, 64), (1025, 512), (8192, 1536), (8193, 2048)],
)
def test_decode_chunk_ranges_cover_rows_with_bounded_balanced_slices(
    rows, capacity, itemsize
):
    """No query is lost/duplicated; small tails share work with preceding chunks.
    The budget counts BF16 or FP32 logits bytes."""
    from vllm.model_executor.layers.sparse_attn_indexer import _decode_chunk_ranges

    width = 1024
    ranges = _decode_chunk_ranges(rows, width, capacity * width * itemsize, itemsize)
    assert ranges[0][0] == 0 and ranges[-1][1] == rows
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    lengths = [end - start for start, end in ranges]
    assert all(0 < n <= capacity for n in lengths)
    assert all(start % 16 == 0 for start, _ in ranges)
    assert max(lengths) - min(lengths) < 32
    assert lengths == sorted(lengths, reverse=True)


def test_decode_chunk_budget_disabled_fitting_and_invalid():
    from vllm.model_executor.layers.sparse_attn_indexer import _decode_chunk_ranges

    assert _decode_chunk_ranges(8192, 1024, 0, 2) == []
    assert _decode_chunk_ranges(32, 1024, 65536, 2) == []
    with pytest.raises(ValueError):
        _decode_chunk_ranges(32, 1024, -1, 2)
    with pytest.raises(ValueError, match="16 logits rows"):
        _decode_chunk_ranges(32, 1024, 1024, 2)


# ---------------------------------------------------------------------------
# ROCm: top_k_per_row_decode on gfx950
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_size,num_speculative_tokens,min_seq_len,max_seq_len",
    [
        pytest.param(1, 2, 125_000, 250_000, id="eight-splits"),
        pytest.param(7, 4, 125_000, 250_000, id="four-splits"),
        pytest.param(16, 4, 2_500, 2_501, id="dynamic-short-rows"),
        pytest.param(13, 4, 25_000, 25_001, id="dynamic-three-splits"),
        pytest.param(16, 4, 65_535, 65_536, id="dynamic-boundary-mixed"),
        pytest.param(16, 4, 125_000, 125_001, id="dynamic-two-splits"),
        pytest.param(64, 3, 125_000, 125_001, id="dynamic-two-splits-max-rows"),
        pytest.param(64, 4, 125_000, 125_001, id="baseline-fallback"),
    ],
)
@requires_gfx950
@torch.inference_mode()
def test_top_k_per_row_decode_gfx950_long_c4a(
    batch_size: int,
    num_speculative_tokens: int,
    min_seq_len: int,
    max_seq_len: int,
) -> None:
    query_tokens = num_speculative_tokens + 1
    num_rows = batch_size * query_tokens
    offsets = torch.arange(num_rows, dtype=torch.int32, device=DEVICE)
    seq_lens = min_seq_len + offsets * (max_seq_len - min_seq_len) // max(
        num_rows - 1, 1
    )
    seq_lens = seq_lens.reshape(batch_size, query_tokens)
    logits = torch.randn(num_rows, 262_144, device=DEVICE)
    indices = torch.empty((num_rows, 1024), dtype=torch.int32, device=DEVICE)
    per_row_decode(logits, seq_lens, indices, 1024, query_tokens)
    check_topk(logits, indices, seq_lens)


@requires_gfx950
@torch.inference_mode()
def test_top_k_per_row_decode_gfx950_long_c4a_1d_seq_lens() -> None:
    batch_size, query_tokens, top_k = 20, 4, 1024
    seq_lens = torch.linspace(
        125_003, 250_000, batch_size, dtype=torch.int32, device=DEVICE
    )
    row_offsets = torch.arange(query_tokens, dtype=torch.int32, device=DEVICE)
    row_ends = (seq_lens[:, None] - query_tokens + row_offsets + 1).reshape(-1)
    logits = torch.randn(batch_size * query_tokens, 262_144, device=DEVICE)
    indices = torch.empty((logits.shape[0], top_k), dtype=torch.int32, device=DEVICE)
    per_row_decode(logits, seq_lens, indices, top_k, query_tokens)
    check_topk(logits, indices, row_ends)


@requires_gfx950
@torch.inference_mode()
def test_top_k_per_row_decode_gfx950_k512_1d_seq_lens() -> None:
    """Honor causal offsets and exact compressed lengths, including short rows."""
    next_n, width, top_k = 6, 16_384, 512
    lengths = torch.tensor(
        [0, 1, 511, 512, 513, 1023, 1024, 16_000], dtype=torch.int32, device=DEVICE
    )
    offsets = torch.arange(next_n, dtype=torch.int32, device=DEVICE)
    row_ends = (lengths[:, None] - next_n + offsets + 1).clamp_min(0).flatten()
    rows = row_ends.numel()
    storage = torch.full((rows, width + 8), 1e20, device=DEVICE)
    logits = storage[:, 1 : width + 1]
    logits[:, :16_000] = torch.arange(16_000, dtype=torch.float32, device=DEVICE)
    indices = torch.full((rows, top_k), -777, dtype=torch.int32, device=DEVICE)
    per_row_decode(logits, lengths, indices, top_k, next_n)
    check_topk(logits, indices, row_ends)


@pytest.mark.parametrize(
    ("rows", "row_length"),
    [
        pytest.param(128, 65_536, id="single-before-64k"),
        pytest.param(128, 65_537, id="four-splits-after-64k"),
        pytest.param(129, 262_144, id="single-before-three-split-length"),
        pytest.param(129, 262_145, id="three-splits-after-256k"),
        pytest.param(160, 262_145, id="three-splits-upper-row-bound"),
        pytest.param(161, 262_144, id="single-before-two-split-length"),
        pytest.param(161, 262_145, id="two-splits-after-256k"),
        pytest.param(256, 262_145, id="two-splits-upper-row-bound"),
        pytest.param(257, 262_145, id="large-row-single-block"),
        pytest.param(384, 262_145, id="optimized-row-limit"),
        pytest.param(385, 262_145, id="native-fallback"),
    ],
)
@requires_gfx950
@torch.inference_mode()
def test_top_k_per_row_decode_gfx950_k512_split_policy_boundaries(
    rows: int, row_length: int
) -> None:
    width, top_k = 262_145, 512
    logits = torch.arange(width, dtype=torch.float32, device=DEVICE).expand(rows, -1)
    lengths = torch.full((rows,), row_length, dtype=torch.int32, device=DEVICE)
    indices = torch.empty((rows, top_k), dtype=torch.int32, device=DEVICE)
    per_row_decode(logits, lengths, indices, top_k)
    expected = torch.arange(
        row_length - top_k, row_length, dtype=torch.int32, device=DEVICE
    ).expand_as(indices)
    assert torch.equal(indices.sort(1).values, expected)


@requires_gfx950
@torch.inference_mode()
def test_top_k_per_row_decode_gfx950_k512_masked_graph_replay() -> None:
    """Preserve visible top-k scores when masks and lengths change on replay."""
    rows, next_n, width, top_k = 24, 6, 262_145, 512
    logits = torch.full((rows, width), 1e20, device=DEVICE)
    lengths = torch.full(
        (rows // next_n, next_n), 70_000, dtype=torch.int32, device=DEVICE
    )
    indices = torch.full((rows, top_k), -777, dtype=torch.int32, device=DEVICE)
    logits[:, :70_000] = float("-inf")
    bin_sizes = [128, 129, 1024, 1025]
    for row in range(rows):
        pattern = row % 9
        if pattern < 4:
            count = [0, 1, 511, 513][pattern]
            logits[row, :count] = torch.arange(count, device=DEVICE).float()
        elif pattern == 4:
            logits[row, :511] = 0
            logits[row, 511] = torch.finfo(torch.float32).min
        else:
            count = bin_sizes[pattern - 5]
            higher = max(0, top_k - count // 2)
            logits[row, :higher] = 2
            logits[row, higher : higher + count] = 1
            logits[row, higher] = 1 + 2**-23

    def run() -> None:
        per_row_decode(logits, lengths, indices, top_k, next_n)

    run()
    check_topk(logits, indices, lengths)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    indices.fill_(-777)
    graph.replay()
    check_topk(logits, indices, lengths)

    bounds = [0, 511, 512, 513, 16_384, 16_385, 131_072, 131_073, 262_144, 262_145]
    row_bounds = torch.tensor(bounds, dtype=torch.int32, device=DEVICE)
    repeats = (rows + len(bounds) - 1) // len(bounds)
    lengths.flatten().copy_(row_bounds.repeat(repeats)[:rows])
    logits[:] = torch.arange(width, dtype=torch.float32, device=DEVICE)
    indices.fill_(-777)
    graph.replay()
    check_topk(logits, indices, lengths)


# ---------------------------------------------------------------------------
# ROCm: AITER
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_speculative_tokens", [2, 3, 4])
@requires_rocm
@torch.inference_mode()
def test_aiter_c4a_decode_topk_uses_exact_mtp_lengths(
    num_speculative_tokens: int,
) -> None:
    from vllm._aiter_ops import rocm_aiter_ops

    query_tokens = num_speculative_tokens + 1
    offsets = torch.arange(query_tokens, dtype=torch.int32, device=DEVICE)
    final_uncompressed_lens = torch.tensor([43, 59], dtype=torch.int32, device=DEVICE)
    seq_lens = (final_uncompressed_lens[:, None] - query_tokens + offsets + 1) // 4
    seq_lens[0, 0] = 0
    num_rows = seq_lens.numel()
    logits = torch.arange(num_rows * 16, dtype=torch.float32, device=DEVICE).reshape(
        num_rows, 16
    )
    indices = torch.empty((num_rows, 4), dtype=torch.int32, device=DEVICE)
    if not rocm_aiter_ops.is_indexer_top_k_supported(
        is_prefill=False,
        compress_ratio=4,
        num_rows=num_rows,
        max_valid_seq_len=seq_lens.max().item(),
    ):
        pytest.skip("AITER top-k is unavailable")
    assert (
        rocm_aiter_ops.indexer_top_k_decode(logits, 1, seq_lens.reshape(-1), indices, 4)
        is None
    )
    check_topk(logits, indices, seq_lens)


@requires_rocm
@torch.inference_mode()
def test_persistent_topk_on_rocm_paged_mqa_logits(workspace_init) -> None:
    """rocm_fp8_paged_mqa_logits returns its logits in the shared workspace,
    where persistent_topk's multi-CTA launch also clears its row states."""
    from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk
    from vllm.platforms import current_platform
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        rocm_fp8_paged_mqa_logits,
    )

    set_random_seed(0)
    fp8 = current_platform.fp8_dtype()
    batch_size, num_heads, head_dim, block_size, top_k = 4, 32, 128, 64, 2048
    seq_len = 81920
    num_pages = seq_len // block_size
    values = torch.randn(batch_size * num_pages, block_size * head_dim, device=DEVICE)
    scales = torch.rand(batch_size * num_pages, block_size, device=DEVICE) + 0.5
    kv_cache = torch.cat(
        [values.to(fp8).view(torch.uint8), scales.view(torch.uint8)], 1
    ).view(-1, block_size, 1, head_dim + 4)
    q = torch.randn(batch_size, 1, num_heads, head_dim, device=DEVICE).to(fp8)
    weights = torch.randn(batch_size, num_heads, device=DEVICE)
    seq_lens = torch.full((batch_size, 1), seq_len, dtype=torch.int32, device=DEVICE)
    block_tables = torch.arange(
        batch_size * num_pages, dtype=torch.int32, device=DEVICE
    ).view(batch_size, num_pages)

    logits = rocm_fp8_paged_mqa_logits(
        q, kv_cache, weights, seq_lens, block_tables, None, seq_len, compress_ratio=4
    )
    expected = logits.clone()
    indices = torch.empty((batch_size, top_k), dtype=torch.int32, device=DEVICE)
    SparseIndexerTopk("persistent")(logits, seq_lens, 1, indices, top_k, seq_len)
    torch.accelerator.synchronize()
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    check_topk(expected, indices, seq_lens)


@requires_gfx950
@torch.inference_mode()
def test_aiter_decode_topk_matches_per_row_under_mtp() -> None:
    """The aiter backend expands (B, 1) seq_lens into per-row ends before
    launching, so next_n > 1 must agree with the in-tree kernel, including
    the rows that clamp to a zero-length prefix."""
    from vllm.model_executor.layers.indexer_topk import get_indexer_topk

    next_n, top_k, num_columns = 4, 512, 1024
    seq_lens = torch.tensor([[2], [513], [700]], dtype=torch.int32, device=DEVICE)
    num_rows = seq_lens.shape[0] * next_n
    set_random_seed(0)
    logits = distinct_logits(num_rows, num_columns)
    row_ends = (
        (seq_lens - next_n + 1 + torch.arange(next_n, device=DEVICE))
        .clamp_(min=0)
        .reshape(-1)
    )
    assert row_ends[:next_n].tolist() == [0, 0, 1, 2]
    for backend in ("aiter", "per_row"):
        indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=DEVICE)
        get_indexer_topk(backend)(logits, seq_lens, next_n, indices, top_k, num_columns)
        check_topk(logits, indices, row_ends)


@pytest.mark.parametrize("num_columns", [16_384, 133_120, 1_048_576])
@pytest.mark.parametrize("next_n", [1, 2])
@requires_rocm
@torch.inference_mode()
def test_aiter_dsa_decode_topk_matches_per_row(num_columns: int, next_n: int) -> None:
    """DeepSeek-V3.2 / GLM-5 decode (uncompressed, k=2048): AITER and the
    in-tree kernel select the exact top-k in the leading slots of each row."""
    from vllm._aiter_ops import rocm_aiter_ops

    top_k = 2048
    if not rocm_aiter_ops.is_indexer_top_k_enabled():
        pytest.skip("AITER top-k is unavailable")
    lengths = torch.tensor(
        [0, 1, 2047, 2048, 2049, 5000, num_columns - 3, num_columns],
        dtype=torch.int32,
        device=DEVICE,
    )
    offsets = torch.arange(next_n, dtype=torch.int32, device=DEVICE)
    row_ends = (lengths[:, None] - next_n + 1 + offsets).clamp_min(0)
    seq_lens = row_ends if next_n > 1 else row_ends.flatten()
    rows = row_ends.numel()
    set_random_seed(0)
    logits = distinct_logits(rows, num_columns)
    columns = torch.arange(num_columns, device=DEVICE)
    logits.masked_fill_(columns[None, :] >= row_ends.reshape(-1, 1), 1e20)

    native = torch.full((rows, top_k), -777, dtype=torch.int32, device=DEVICE)
    per_row_decode(logits, seq_lens, native, top_k, next_n)
    aiter = torch.full_like(native, -777)
    rocm_aiter_ops.indexer_top_k_decode(logits, 1, seq_lens.reshape(-1), aiter, top_k)
    for indices in (native, aiter):
        check_topk(logits, indices, row_ends)


def test_aiter_c4a_decode_topk_kernel_selection(monkeypatch) -> None:
    from vllm._aiter_ops import rocm_aiter_ops

    enable_aiter_topk(monkeypatch)
    is_supported = rocm_aiter_ops.is_indexer_top_k_supported

    eligible_cases = [
        dict(compress_ratio=4, num_rows=1, max_valid_seq_len=65_536),
        dict(compress_ratio=4, num_rows=257, max_valid_seq_len=250_000),
        dict(compress_ratio=2, num_rows=384, max_valid_seq_len=1_048_576),
    ]
    for kwargs in eligible_cases:
        assert is_supported(is_prefill=False, **kwargs) is True

    ineligible_cases = [
        dict(compress_ratio=4, num_rows=1, max_valid_seq_len=65_537),
        dict(compress_ratio=4, num_rows=256, max_valid_seq_len=125_000),
    ]
    for kwargs in ineligible_cases:
        assert is_supported(is_prefill=False, **kwargs) is False

    enable_aiter_topk(monkeypatch, enabled=False)
    assert is_supported(**eligible_cases[0], is_prefill=False) is False


def test_aiter_dsa_decode_topk_kernel_selection(monkeypatch) -> None:
    """The uncompressed DSA indexer takes AITER for k=2048 on every decode
    shape, including FULL-graph decode bounded only by max_model_len."""
    from vllm._aiter_ops import rocm_aiter_ops

    enable_aiter_topk(monkeypatch)
    is_supported = rocm_aiter_ops.is_indexer_top_k_supported

    for kwargs in (
        dict(is_prefill=False, num_rows=1, max_valid_seq_len=1_048_576),
        dict(is_prefill=False, num_rows=256, max_valid_seq_len=4096),
    ):
        assert is_supported(compress_ratio=1, topk_tokens=2048, **kwargs) is True
        assert is_supported(compress_ratio=1, topk_tokens=1024, **kwargs) is False
        assert is_supported(compress_ratio=1, **kwargs) is False

    enable_aiter_topk(monkeypatch, enabled=False)
    assert (
        is_supported(
            is_prefill=False,
            compress_ratio=1,
            num_rows=1,
            max_valid_seq_len=1_048_576,
            topk_tokens=2048,
        )
        is False
    )


def test_dsv4_native_top_k_window() -> None:
    """The in-tree decode kernel's measured advantage band over AITER, which
    applies to the DSV4 indexer only."""
    from vllm._aiter_ops import rocm_aiter_ops

    prefers_native = rocm_aiter_ops.dsv4_indexer_prefers_native_top_k
    assert prefers_native(num_rows=128, num_columns=524_288, topk_tokens=512) is True
    assert prefers_native(num_rows=385, num_columns=524_288, topk_tokens=512) is False
    assert prefers_native(num_rows=128, num_columns=1_048_577, topk_tokens=512) is False
    assert prefers_native(num_rows=128, num_columns=524_288, topk_tokens=1024) is False


@requires_cuda
def test_sparse_indexer_topk_aiter_requires_rocm() -> None:
    """Explicitly requesting the ROCm-only backend must fail fast elsewhere."""
    logits = torch.randn(8, 4096, device=DEVICE)
    with pytest.raises(RuntimeError, match="ROCm"):
        make_topk("aiter").resolve_backend(logits, 2048, 8)


@requires_rocm
def test_sparse_indexer_topk_backend_resolution_aiter(monkeypatch) -> None:
    """Aiter is opt-in only: "auto" never selects it (the shape-dependent perf
    gate lives at the call site), and an explicit request validates hard
    capability only."""

    def decode_kernel(*args, **kwargs) -> None:
        pass

    patch_aiter_topk(monkeypatch, decode_kernel)
    logits = torch.randn(8, 4096, device=DEVICE)

    assert make_topk("auto").resolve_backend(logits, 2048, 8) == "per_row"
    assert make_topk("aiter").resolve_backend(logits, 2048, 8) == "aiter"
    # Shapes the call-site perf gate would decline still resolve.
    assert make_topk("aiter").resolve_backend(logits, 512, 8) == "aiter"
    with pytest.raises(RuntimeError, match="topk_tokens must be in"):
        make_topk("aiter").resolve_backend(logits, 3000, 8)

    enable_aiter_topk(monkeypatch, enabled=False)
    with pytest.raises(RuntimeError, match="no tuned indexer top-k kernels"):
        make_topk("aiter").resolve_backend(logits, 2048, 8)

    patch_aiter_topk(monkeypatch, decode_kernel, enabled=False)
    assert make_topk("auto").resolve_backend(logits, 2048, 8) == "per_row"
    with pytest.raises(RuntimeError, match="VLLM_ROCM_USE_AITER"):
        make_topk("aiter").resolve_backend(logits, 2048, 8)


@requires_rocm
@torch.inference_mode()
def test_sparse_indexer_topk_aiter_applies_no_shape_gate(monkeypatch) -> None:
    """Shape-dependent perf gating belongs to the call site: once "aiter" is
    requested the dispatcher must run it for every supported k, including the
    k=512 band the in-tree kernel wins on DSV4's compressed-KV logits."""
    from vllm.model_executor.layers import indexer_topk

    calls: list[tuple] = []
    patch_aiter_topk(monkeypatch, lambda *args: calls.append(args))
    monkeypatch.setattr(
        indexer_topk.ops,
        "top_k_per_row_decode",
        lambda *args: pytest.fail("an explicit aiter request must not fall back"),
    )
    logits = torch.randn(8, 4096, device=DEVICE)
    seq_lens = torch.full((8, 1), 4096, dtype=torch.int32, device=DEVICE)
    indices = torch.empty((8, 512), dtype=torch.int32, device=DEVICE)
    op = make_topk("aiter")
    op(logits, seq_lens, 1, indices, 512, 4096)
    # Past the compressed-context boundary the call-site gate declines on.
    op(logits, seq_lens, 1, indices, 512, 4 * (64 * 1024 + 1))
    assert len(calls) == 2


@requires_rocm
@torch.inference_mode()
def test_sparse_indexer_topk_auto_stays_on_per_row(monkeypatch) -> None:
    """The "auto" chain must not reach AITER on its own: the decode top-k runs
    through top_k_per_row_decode unless the caller asks for the aiter backend."""
    from vllm.model_executor.layers import indexer_topk

    def decode_kernel(*args, **kwargs) -> None:
        pytest.fail("AITER is opt-in only")

    patch_aiter_topk(monkeypatch, decode_kernel)
    calls: list[tuple] = []
    monkeypatch.setattr(
        indexer_topk.ops, "top_k_per_row_decode", lambda *args: calls.append(args)
    )
    logits = torch.randn(8, 4096, device=DEVICE)
    seq_lens = torch.full((8, 1), 4096, dtype=torch.int32, device=DEVICE)
    indices = torch.empty((8, 1024), dtype=torch.int32, device=DEVICE)
    make_topk("auto")(logits, seq_lens, 1, indices, 1024, 4096)
    assert len(calls) == 1


@requires_rocm
@torch.inference_mode()
def test_sparse_indexer_topk_aiter_expands_mtp_row_ends(monkeypatch) -> None:
    """AITER's decode kernel only implements the next_n == 1 form, so spec
    decode batches must be handed the equivalent expanded per-row ends."""
    from vllm.model_executor.layers import indexer_topk

    recorded: dict = {}

    def decode_kernel(logits, next_n, seq_lens, *args, **kwargs) -> None:
        recorded["next_n"] = next_n
        recorded["row_ends"] = seq_lens.tolist()

    patch_aiter_topk(monkeypatch, decode_kernel)
    monkeypatch.setattr(
        indexer_topk.ops,
        "top_k_per_row_decode",
        lambda *args: pytest.fail("an explicit aiter request must not fall back"),
    )
    logits = torch.randn(4, 4096, device=DEVICE)
    seq_lens = torch.tensor([[100], [200]], dtype=torch.int32, device=DEVICE)
    indices = torch.empty((4, 1024), dtype=torch.int32, device=DEVICE)
    make_topk("aiter")(logits, seq_lens, 2, indices, 1024, 800)
    assert recorded["next_n"] == 1
    assert recorded["row_ends"] == [99, 100, 199, 200]
