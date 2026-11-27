# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the sparse-indexer (DSA) top-k tests.

`check_topk` is the one correctness check: the selected scores must equal
`torch.topk`'s bit for bit (NaN ranks above +inf, ties may pick any index)
and each row holds `min(length, k)` distinct in-range indices, then -1.
`KINDS` are the adversarial score distributions; `make_logits` writes NaN
past every row's length so a kernel that reads it fails.
"""

import functools
import importlib.util
import os

import pytest
import torch
from vllm.platforms import current_platform

DEVICE = "cuda"
TOP_KS = (512, 1024, 2048)
# k values beyond TOP_KS for the kernels that take any k (DeepSelect: <= 4096).
ANY_K = (*TOP_KS, 3000)
RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
INT32_MIN = -(2**31)
F32, BF16 = torch.float32, torch.bfloat16
DTYPE_NAMES = {F32: "fp32", BF16: "bf16"}


# ---------------------------------------------------------------------------
# Platform
# ---------------------------------------------------------------------------


IS_CUDA = current_platform.is_cuda()


IS_ROCM = current_platform.is_rocm()


IS_CUDA_ALIKE = current_platform.is_cuda_alike()


IS_SM100 = IS_CUDA and current_platform.is_device_capability_family(100)


def max_smem_per_block() -> int:
    props = torch.cuda.get_device_properties(0)
    # ROCm reports no opt-in limit; the default per-block limit is the maximum.
    return getattr(
        props, "shared_memory_per_block_optin", props.shared_memory_per_block
    )


HAS_COOPERATIVE = (
    IS_CUDA
    and current_platform.has_device_capability(90)
    and not current_platform.is_device_capability_family(110)
    and not current_platform.is_device_capability_family(120)
)


def has_flashinfer_topk() -> bool:
    # find_spec only: importing flashinfer initializes CUDA at import time.
    return IS_CUDA and importlib.util.find_spec("flashinfer") is not None


def has_flashinfer_gvr2() -> bool:
    """FlashInfer with the gvr_2 top_k_varlen backend (SM100 family only).
    Probes the package tree so collection does not import flashinfer."""
    if not (IS_SM100 and has_flashinfer_topk()):
        return False
    spec = importlib.util.find_spec("flashinfer")
    assert spec is not None and spec.submodule_search_locations
    return any(
        os.path.exists(os.path.join(p, "topk_varlen", "kernels", "gvr2_topk_host.py"))
        for p in spec.submodule_search_locations
    )


def _on_gfx950() -> bool:
    if not IS_ROCM:
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


requires_cuda = pytest.mark.skipif(not IS_CUDA, reason="requires CUDA")


# top_k_per_row and persistent_topk also build for ROCm.
requires_cuda_alike = pytest.mark.skipif(
    not IS_CUDA_ALIKE, reason="requires CUDA or ROCm"
)


# BF16 persistent_topk refuses devices below 128 KiB per block (gfx942: 64 KiB).
requires_bf16_topk_smem = pytest.mark.skipif(
    IS_CUDA_ALIKE and max_smem_per_block() < 128 * 1024,
    reason="bfloat16 persistent_topk requires 128 KiB shared memory",
)


requires_rocm = pytest.mark.skipif(not IS_ROCM, reason="requires ROCm")


requires_cooperative = pytest.mark.skipif(
    not HAS_COOPERATIVE, reason="cooperative_topk requires SM90+ (not SM11x/SM12x)"
)


# DeepSelect, gvr_2 and decode_topk are built for sm_100a/sm_103a only.
requires_sm100 = pytest.mark.skipif(not IS_SM100, reason="requires SM100a/SM103a")


requires_flashinfer = pytest.mark.skipif(
    not has_flashinfer_topk(), reason="requires flashinfer top-k"
)


requires_flashinfer_gvr2 = pytest.mark.skipif(
    not has_flashinfer_gvr2(), reason="requires flashinfer gvr_2 top-k on SM100"
)


requires_gfx950 = pytest.mark.skipif(
    not _on_gfx950(), reason="exercises the gfx950 launch configuration"
)


# ---------------------------------------------------------------------------
# Reference check
# ---------------------------------------------------------------------------


def _keys(scores: torch.Tensor) -> torch.Tensor:
    """int32 keys in torch.topk's order: NaN above +inf, -0.0 equal to 0.0."""
    x = scores.float()
    x = torch.where(x == 0, 0.0, x)
    bits = x.view(torch.int32)
    keys = torch.where(bits < 0, -(bits & 0x7FFFFFFF), bits)
    return keys.masked_fill_(x.isnan(), 0x7FC00000)


def check_topk(
    logits: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    starts: torch.Tensor | None = None,
) -> None:
    """Assert `indices` is an exact top-k of each row of `logits`.

    Row `r` holds `n = min(length, k)` distinct indices into
    `logits[r, start:start + length]` (relative to `start`) followed by -1,
    and its selected scores equal the `n` largest of the row. Lengths are
    clamped to [0, width - start].
    """
    rows, k = indices.shape
    width = logits.shape[1]
    starts = (
        torch.zeros(rows, dtype=torch.long, device=logits.device)
        if starts is None
        else starts.reshape(-1).long()
    )
    ends = lengths.reshape(-1).long().clamp(min=0)
    ends = torch.minimum(ends, width - starts)
    span = max(int(ends.max()), 1)
    cols = torch.arange(span, device=logits.device)
    keys = _keys(logits.gather(1, (starts[:, None] + cols).clamp(max=width - 1)))
    keys.masked_fill_(cols >= ends[:, None], INT32_MIN)

    n = ends.clamp(max=k)
    leading = torch.arange(k, device=logits.device) < n[:, None]
    in_range = (indices >= 0) & (indices < ends[:, None])
    bad = torch.where(leading, ~in_range, indices != -1).any(1)
    ordered = indices.sort(1).values
    bad |= ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any(1)
    if bad.any():
        r = int(bad.nonzero()[0])
        raise AssertionError(
            f"row {r} (length {int(ends[r])}, k {k}): expected {int(n[r])} "
            f"distinct in-range indices then -1, got {indices[r].tolist()[:16]}..."
        )

    want = keys.topk(min(k, span), dim=1).values
    got = keys.gather(1, indices.long().clamp(0, span - 1))
    got = got.masked_fill_(~leading, INT32_MIN).sort(1, descending=True).values
    wrong = (got[:, : want.shape[1]] != want).any(1)
    if wrong.any():
        r = int(wrong.nonzero()[0])
        miss = int((got[r, : want.shape[1]] != want[r]).sum())
        raise AssertionError(
            f"row {r} (length {int(ends[r])}, k {k}): {miss} selected scores "
            f"differ from torch.topk ({int(wrong.sum())} rows wrong)"
        )


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _shuffle(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    perm = torch.rand(x.shape, generator=gen, device=x.device).argsort(1)
    return x.gather(1, perm)


def planted(x: torch.Tensor, gen: torch.Generator, *blocks) -> torch.Tensor:
    """`x` with `blocks` (values per row) written left to right, then each
    row shuffled."""
    col = 0
    for block in blocks:
        n = max(0, min(block.shape[-1], x.shape[1] - col))
        x[:, col : col + n] = block[..., :n]
        col += n
    return _shuffle(x, gen)


@functools.cache
def _fp16_halfway() -> tuple[float, torch.Tensor]:
    """The top FP32 key of the FP16 coarse bin holding -1.0 (an FP16 halfway
    point that rounds into the bin) and the next keys, which round into the
    bin above."""

    def coarse_bin(x: torch.Tensor) -> torch.Tensor:
        bits = x.half().view(torch.int16).to(torch.int32) & 0xFFFF
        key = torch.where(bits >= 0x8000, ~bits & 0xFFFF, bits | 0x8000)
        return key >> 6

    one = torch.tensor([-1.0])
    bits = int(one.view(torch.int32)) - torch.arange(0, 1 << 14, dtype=torch.int32)
    values = bits.view(torch.float32)
    top = int((coarse_bin(values) == coarse_bin(one)).nonzero().max())
    near = values[top + 1 : top + 2048]
    assert bool((coarse_bin(near) == coarse_bin(one) + 1).all())
    return float(values[top]), near.to(DEVICE)


def _ramp(n: int, base: float = 1.0) -> torch.Tensor:
    """Distinct increasing scores within one coarse (FP16) bin."""
    return base + torch.arange(n, device=DEVICE, dtype=F32) * (0.01 / max(n, 1))


def _kind_nan(r, w, k, dtype, g, n):
    # Fewer NaNs than k, mixed with +inf and the largest finite values.
    values = [float("nan")] * (k // 4) + [float("inf")] * (k // 8)
    values += [torch.finfo(dtype).max] * (k // 8) + [1e30] * (k // 8)
    return planted(randn(r, w, g), g, torch.tensor(values, device=DEVICE))


def _kind_fp16_halfway(r, w, k, dtype, g, n):
    halfway, near = _fp16_halfway()
    picks = near[torch.randint(0, near.numel(), (r, 50), generator=g, device=DEVICE)]
    return planted(
        torch.full((r, w), -100.0, device=DEVICE),
        g,
        5 + torch.rand(r, k - 60, generator=g, device=DEVICE),
        picks,
        torch.full((3000,), halfway, device=DEVICE),
    )


def _kind_signed_zeros(r, w, k, dtype, g, n):
    # Zero and subnormal coarse bins take the full radix rescan.
    tiny = torch.arange(1, 1501, device=DEVICE, dtype=F32) * 1e-40
    return planted(
        torch.full((r, w), -100.0, device=DEVICE),
        g,
        5 + torch.rand(r, k - 10, generator=g, device=DEVICE),
        torch.zeros(1500, device=DEVICE),
        torch.full((1000,), -0.0, device=DEVICE),
        tiny,
        -tiny,
    )


def _kind_sampled_peaks(r, w, k, dtype, g, n):
    # The largest scores sit where a strided sample looks: the sample keeps
    # fewer than k survivors and sampling kernels must fall back.
    x = torch.zeros(r, w, device=DEVICE)
    s = torch.arange(4096, device=DEVICE)
    x[:, (s // 32) * (w // 128) + s % 32] = 1 + s.float() / 4096
    return x


# Sampled-cutoff kernels (split-row top-k) estimate the threshold from
# SAMPLES scores evenly spaced over each row's length: position i of a row of
# length n is ((2 i + 1) n) // (2 SAMPLES).
SAMPLES = 2048


def sample_positions(n: torch.Tensor, samples: int = SAMPLES) -> torch.Tensor:
    """[rows, samples] evenly spaced columns of rows of lengths `n` (rows
    shorter than `samples` repeat columns; empty rows sample column 0)."""
    i = torch.arange(samples, device=DEVICE, dtype=torch.int64)
    n = n.reshape(-1, 1).long().clamp(min=1)
    return ((2 * i + 1) * n) // (2 * samples)


def _kind_sample_trap(r, w, k, dtype, g, n):
    # Every other sample position holds a top score: the sample promises
    # SAMPLES / 2 survivors above the largest gaussian, which is fewer than k
    # for k = 2048, so a sampled cutoff keeps too few and must fall back.
    x = randn(r, w, g)
    pos = sample_positions(n)[:, ::2]
    return x.scatter_(1, pos, 10 + rand(r, pos.shape[1], g))


def _kind_sample_anti(r, w, k, dtype, g, n):
    # Every sample position holds the row's lowest score: a sampled cutoff
    # lands at the bottom and the whole row survives (candidate overflow).
    x = 5 + randn(r, w, g)
    return x.scatter_(1, sample_positions(n), -50.0)


def _kind_block(r, w, k, dtype, g, n):
    # 4k top scores packed contiguously mid-row: one CTA's slice (or split)
    # holds every survivor and overflows its local candidate list.
    x = randn(r, w, g)
    cols = torch.arange(w, device=DEVICE)
    start = (n.reshape(-1, 1).long() // 2).clamp(max=max(w - 4 * k, 0))
    inside = (cols >= start) & (cols < start + 4 * k)
    return x + 10 * inside


def _kind_recent(r, w, k, dtype, g, n):
    # Real decode rows: most top scores are among the latest tokens (a ramp
    # over the last 8192), plus attention-sink tokens and scattered needles.
    x = randn(r, w, g)
    cols = torch.arange(w, device=DEVICE)
    age = n.reshape(-1, 1).long() - 1 - cols
    recent = (age >= 0) & (age < 8192)
    x += torch.where(recent, 3.5 * (1 - age / 8192), 0.0)
    x[:, :4] += 6
    needles = torch.randint(0, w, (r, 64), generator=g, device=DEVICE)
    return x.scatter_add_(1, needles, torch.full((r, 64), 4.0, device=DEVICE))


def _kind_spike(r, w, k, dtype, g, n):
    # k - 1 large scores, the rest one tied value: the threshold is the tie
    # and any one of the row's other columns completes the top-k.
    x = torch.zeros(r, w, device=DEVICE)
    cols = rand(r, w, g).topk(k - 1, dim=1).indices
    return x.scatter_(1, cols, 10.0)


def rand(r, n, g):
    return torch.rand(r, n, generator=g, device=DEVICE)


def randn(r, n, g):
    return torch.randn(r, n, generator=g, device=DEVICE)


# kind -> fn(rows, width, k, dtype, generator, lengths) -> [rows, width] FP32.
KINDS = {
    "gaussian": lambda r, w, k, d, g, n: randn(r, w, g),
    # 1024 distinct scores sharing their top 22 bits.
    "lsb10": lambda r, w, k, d, g, n: (
        torch.randint(0, 1024, (r, w), generator=g, device=DEVICE, dtype=torch.int32)
        | 0x3F900000
    ).view(F32),
    "ties": lambda r, w, k, d, g, n: torch.randint(
        0, 6, (r, w), generator=g, device=DEVICE
    ).float(),
    "constant": lambda r, w, k, d, g, n: torch.ones(r, w, device=DEVICE),
    # Every score in one coarse bin: candidate overflow.
    "ramp": lambda r, w, k, d, g, n: _ramp(w).expand(r, w),
    "narrow": lambda r, w, k, d, g, n: 1 + rand(r, w, g) * 0.01,
    "narrow_neg": lambda r, w, k, d, g, n: -1.01 + rand(r, w, g) * 0.01,
    # The largest scores at the row's end (the last CTA's partition).
    "rising": lambda r, w, k, d, g, n: (
        randn(r, w, g) * 0.1 + torch.linspace(0, 8, w, device=DEVICE)
    ),
    "signed_zeros": _kind_signed_zeros,
    "infs": lambda r, w, k, d, g, n: planted(
        randn(r, w, g),
        g,
        torch.full((3000,), float("inf"), device=DEVICE),
        torch.full((3000,), float("-inf"), device=DEVICE),
    ),
    # Fewer finite scores than k: the threshold falls in the -inf bin.
    "few_finite": lambda r, w, k, d, g, n: planted(
        torch.full((r, w), float("-inf"), device=DEVICE), g, randn(r, k - 10, g)
    ),
    "nan": _kind_nan,
    # "nan" with the sign bit set on every NaN (make_logits sets the bits
    # after the dtype cast, which drops a NaN's sign): still above +inf.
    "nan_neg": _kind_nan,
    # More NaNs than k overflow the top coarse bin.
    "nan_flood": lambda r, w, k, d, g, n: planted(
        randn(r, w, g), g, torch.full((4 * k,), float("nan"), device=DEVICE)
    ),
    "fp16_halfway": _kind_fp16_halfway,
    # A final radix window larger than the per-CTA staging buffer.
    "window": lambda r, w, k, d, g, n: planted(
        torch.full((r, w), -100.0, device=DEVICE),
        g,
        5 + rand(r, k - 10, g),
        1 + torch.randint(0, 8, (100000,), generator=g, device=DEVICE) * 2.0**-23,
    ),
    "sampled_peaks": _kind_sampled_peaks,
    "sample_trap": _kind_sample_trap,
    "sample_anti": _kind_sample_anti,
    "block": _kind_block,
    "recent": _kind_recent,
    "spike": _kind_spike,
    # Candidates both above and inside the threshold bin.
    "above_threshold": lambda r, w, k, d, g, n: planted(
        torch.full((r, w), -2.0, device=DEVICE), g, _ramp(k // 4, 2.0), _ramp(4096)
    ),
    # Exactly persistent_topk's per-bin candidate capacity, and one more.
    "capacity": lambda r, w, k, d, g, n: planted(
        torch.full((r, w), -2.0, device=DEVICE), g, _ramp(max(k, 1024))
    ),
    "capacity_plus_one": lambda r, w, k, d, g, n: planted(
        torch.full((r, w), -2.0, device=DEVICE), g, _ramp(max(k, 1024) + 1)
    ),
}

NAN_KINDS = ("nan", "nan_neg", "nan_flood")

# Kinds for the long-row (512K-1M) shapes, where the full KINDS matrix would
# take minutes: gaussian plus the ones that defeat sampled cutoffs, overflow a
# CTA's candidate list or model real decode rows.
LONG_KINDS = ("gaussian", "recent", "sample_trap", "sample_anti", "block", "spike")


def ragged_lengths(rows: int, width: int, k: int) -> torch.Tensor:
    """Row 0 spans the width; the others cycle through path and alignment
    boundaries (cluster sizes, register path, sampled_topk's short rows, k,
    empty rows)."""
    edges = [width, width - 1, width // 2 + 3, 65537, 65536, 65535, 32769, 32768]
    edges += [16385, 16384, 4097, k + 1, k, k - 1, 1, 0]
    pattern = list(dict.fromkeys(n for n in edges if 0 <= n <= width))
    lengths = [pattern[r % len(pattern)] for r in range(rows)]
    return torch.tensor(lengths, dtype=torch.int32, device=DEVICE)


def long_lengths(rows: int, width: int) -> torch.Tensor:
    """Every row long: row 0 spans the width, the others end up to 64K
    columns earlier (a decode batch of long-context requests)."""
    r = torch.arange(rows, dtype=torch.int64, device=DEVICE)
    return (width - (r * 7919) % 65536).to(torch.int32)


def mixed_lengths(
    long: list[int], num_short: int, k: int, short_max: int = 65536, seed: int = 0
) -> torch.Tensor:
    """`long` row lengths spread evenly among `num_short` short rows of
    lengths in [k, short_max] (a ragged decode batch where a few long
    requests set the cost)."""
    gen = torch.Generator().manual_seed(seed)
    short = torch.randint(k, short_max + 1, (num_short,), generator=gen).tolist()
    rows = len(long) + num_short
    slots = {round((i + 0.5) * rows / len(long) - 0.5) for i in range(len(long))}
    assert len(slots) == len(long)
    it_long, it_short = iter(long), iter(short)
    lengths = [next(it_long) if r in slots else next(it_short) for r in range(rows)]
    return torch.tensor(lengths, dtype=torch.int32, device=DEVICE)


def make_logits(
    kind: str,
    rows: int,
    width: int,
    k: int,
    dtype: torch.dtype,
    lengths: torch.Tensor,
    seed: int = 0,
) -> torch.Tensor:
    """[rows, width] scores of `kind`, NaN past each row's length. Rows are
    1024-byte aligned (as DeepGEMM's logits are) and the storage covers every
    row's full stride."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    stride = (width + 511) // 512 * 512
    storage = torch.empty(rows, stride, dtype=dtype, device=DEVICE)
    logits = storage[:, :width]
    lengths = lengths.reshape(-1).clamp(0, width)
    logits.copy_(KINDS[kind](rows, width, k, dtype, gen, lengths))
    if kind == "nan_neg":
        bits = logits.view(torch.int16 if dtype.itemsize == 2 else torch.int32)
        bits.masked_fill_(logits.isnan(), -1)
    cols = torch.arange(stride, device=DEVICE)
    storage.masked_fill_(cols >= lengths.reshape(-1, 1), float("nan"))
    return logits


@functools.lru_cache(maxsize=1)
def _cached_logits(kind, rows, width, k, dtype, lengths, seed) -> torch.Tensor:
    return make_logits(
        kind, rows, width, k, dtype, torch.tensor(lengths, device=DEVICE), seed
    )


def shared_logits(
    kind: str,
    rows: int,
    width: int,
    k: int,
    dtype: torch.dtype,
    lengths: torch.Tensor,
    seed: int = 0,
) -> torch.Tensor:
    """`make_logits`, reused by consecutive callers with the same arguments
    (the long-row tests share one input across backends and row counts).
    Callers must not write to it."""
    return _cached_logits(
        kind, rows, width, k, dtype, tuple(lengths.reshape(-1).tolist()), seed
    )


# ---------------------------------------------------------------------------
# SparseIndexerTopk and AITER
# ---------------------------------------------------------------------------


def make_topk(backend: str):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk

    cfg = VllmConfig(kernel_config={"sparse_indexer_topk_backend": backend})
    with set_current_vllm_config(cfg):
        return SparseIndexerTopk()


def enable_aiter_topk(monkeypatch, enabled: bool = True) -> None:
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(
        rocm_aiter_ops, "is_indexer_top_k_enabled", lambda: enabled, raising=True
    )


def patch_aiter_topk(monkeypatch, decode_kernel, enabled: bool = True) -> None:
    from vllm._aiter_ops import rocm_aiter_ops

    enable_aiter_topk(monkeypatch, enabled=enabled)
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: enabled, raising=True)
    monkeypatch.setattr(rocm_aiter_ops, "indexer_top_k_decode", decode_kernel)


def distinct_logits(rows: int, num_columns: int) -> torch.Tensor:
    # Distinct values per row: the selected set is unique.
    return torch.rand(rows, num_columns, device=DEVICE).argsort(dim=1).float()
