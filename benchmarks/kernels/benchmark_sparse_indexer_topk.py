# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the sparse-indexer (DSA) top-k for decode and prefill.

  decode   one row per query token, per-row lengths, inside CUDA graphs.
           Backends: the `SparseIndexerTopk` names (`auto` and
           `persistent` are dispatchers: `auto` picks by row count,
           `persistent` runs the radix kernel, FilteredTopK or sampled_topk
           by row count and `max_seq_len`) and `decode_topk`, the DSv4.1
           specialized-decode CuTe kernel (BF16, SM100). Opt-in: `torch`,
           `aiter`. Drivers can register more in `EXTRA_DECODE_BACKENDS`.
           The default rows sit on both sides of every launch-shape switch
           (cooperative_topk's cluster sizes at 4/8/33 rows and its 64-row
           limit, persistent_topk's 32/64-row paths, cluster-wave edges).
  prefill  a chunk of query rows (hundreds to thousands, eager) from the
           dense logits of one `fp8_mqa_logits` call. Backends:
           `per_row` (top_k_per_row_prefill, the default), `deep_select`
           (BF16 logits with sparse_indexer_topk_backend=deep_select),
           `flashinfer_gvr2` (`SparseIndexerTopk.forward_prefill` takes it
           for chunks of at least GVR2_PREFILL_MIN_ROWS rows; timed at every
           row count here to place that gate) and `aiter` (ROCm with
           VLLM_ROCM_USE_AITER=1).
  prefill-indexer
           the whole prefill indexer stage on captured GLM inputs
           (--real-captures: q, k, key scales, head weights, row ranges):
           `dense+<top-k>` is vLLM's dense path (per window of the 512 MiB
           logits budget, `fp8_fp4_mqa_logits` in each --dtypes logits
           dtype, then the top-k; all windows in one CUDA graph) and
           `fused` is `fused_mqa_topk_prefill` (#60618, SM100), which
           never materializes the logits. Each line says which path
           production takes for that chunk (`fused_mqa_topk_chunk`). A
           result is exact (#60618's definition) when its layout is valid and,
           on 65 evenly spaced rows, the selected keys' FP32 scores match the
           FP32 top-k scores (rtol 2e-4, atol 1e-4); BF16 logits are not
           exact at the boundary. It ends with #60618's table:
           medians over captures per key count, fused's speedup over each
           dense path, and exact captures.

Each backend is timed the way production runs it, separately for each indexer
logits dtype: `fp32` logits are passed as is; `bf16` logits go directly to
a backend that reads them, and otherwise the timed call includes production's
`as_fp32_logits` upcast (marked `+up`). A backend that cannot take an
input is left out of that line; ratios are relative to the first backend
listed that ran.

Methodology: one input per configuration and seed, shared by all backends, in
whole-row storage aligned for DeepSelect, NaN past every row's span;
CUDA-graph timing (`do_bench_cudagraph`, median) over `--rounds` rounds
with the backend order rotated. Each line reports the median over --seeds
inputs (seeds of the synthetic scores, or draws of captured rows), the
min-max over every round and, with more than one seed, the worst seed
(`worst`). `--check` verifies every backend against `torch.topk`
(selected values, any tie order) before timing it.

L2 state (decode and prefill):
  warm (default)  the same input on every call, as an isolated microbenchmark
                  sees it; it understates in-model calls.
  --cold-l2       every call reads another copy of the input (enough copies to
                  cover 2.5x the L2 cache, all captured in the graph), so no
                  call finds its input in L2; nothing is flushed inside the
                  timed region.
  --after-scoring (decode, SM100 with DeepGEMM) each call first runs the real
                  paged MQA logits kernel (`fp8_fp4_paged_mqa_logits`,
                  --heads x 128, random FP8 queries and KV cache, block size
                  64), as the indexer does, and the top-k reads its output;
                  the reported time is (scoring + top-k) - (scoring alone),
                  and the scores are what the scoring kernel produces
                  (--distributions and --layouts set only the row lengths).

Co-tenants: before and after every line the benchmark lists the compute
processes on its GPU (NVML); if another one is there, the line is rerun up to
--cotenant-retries times and otherwise marked `CONTAMINATED` (CSV column
`cotenant`).

Scores (--distributions):
  gaussian     N(0, 1).
  narrow       1 + 0.01 * U(0, 1): most scores share one coarse histogram bin.
  narrow_neg   `-narrow`.
  zeros        k - 10 scores in [5, 6), the rest 0.0 (a single-key threshold).
  sample_trap  gaussian, with a top score on every other one of 2048 evenly
               spaced positions of each row: a sampled cutoff keeps too few.
  sample_anti  gaussian + 5, with each row's lowest score on those 2048
               positions: a sampled cutoff keeps the whole row.
  block        gaussian, with 4k top scores contiguous mid-row (one CTA's
               slice holds every survivor).
  recent       gaussian, the top scores mostly among the last 8192 tokens,
               plus 4 sink tokens and 64 needles (real decode rows look so).
  real_<src>   GLM-5.2 indexer logits (src: mrcr2needle, mrcr8needle, random):
               the last query rows of a captured prefill at exactly the
               requested width, with their real per-row lengths (which replace
               --layouts). Computed on first use from
               `<src>_keys<width>_layer<L>.pt` captures in --real-captures
               (read-only) and cached in --real-cache; widths without a capture
               are skipped. A capture is a dict of one prefill chunk's
               indexer inputs: `q` [rows, heads, dim] FP8, `k` [keys, dim]
               FP8, `ks` [keys] key scales, `w` [rows, heads] head weights,
               and `starts`, `ends` [rows] (each row's key range); the
               prefill-indexer subcommand reads the same files.
  cap_<label>  (decode) real decode rows from --decode-captures
               `<label>=<dir>`: `*.pt` files, each a dict of the decode
               top-k call's inputs: `logits` [rows, width] (FP32 or BF16),
               `row_ends` (int, one per row), `k` (int) and optionally
               `stride0` (the logits row stride). A line of R rows at width
               W stacks R captured rows whose length rounds to W (log2), a
               different draw per seed, at the captured k (--top-k is
               ignored) and in the captured storage (columns past each row
               as captured). Replaces --layouts.

Row layouts (--layouts):
  full    every row spans the width, or its first --valid fraction.
  ragged  (decode) lengths uniform in [k, width].
  mixed   (decode) --long-rows rows span the width, spread evenly among short
          rows of lengths uniform in [k, --short-max].
  causal  (prefill) the chunk is the last `rows` tokens of a `width`-token
          context: row r sees width - rows + r + 1 keys, from column 0
          (DeepGEMM's compressed logits).
  offset  (prefill) causal lengths, but row r starts at column
          r * width / (2 * rows) (absolute-column logits, as XPU and AITER
          produce them); deep_select and flashinfer_gvr2 skip it.

Examples:
    python benchmarks/kernels/benchmark_sparse_indexer_topk.py decode \\
        --rows 1 64 --widths 32768 262144 --top-k 2048 --check
    # A few 1M requests among short ones, cold L2, worst of 5 inputs:
    python benchmarks/kernels/benchmark_sparse_indexer_topk.py decode \\
        --layouts mixed --long-rows 2 --rows 24 48 --widths 1048576 \\
        --distributions gaussian recent sample_trap --cold-l2 --seeds 5
    # Real GLM-5.3 decode rows at 1M, 1-65 rows:
    python benchmarks/kernels/benchmark_sparse_indexer_topk.py decode \\
        --decode-captures glm53=<dir> --distributions cap_glm53 \\
        --widths 1048576 --rows 1 4 34 65 --seeds 5 --check
    python benchmarks/kernels/benchmark_sparse_indexer_topk.py prefill \\
        --rows 1024 4096 --widths 32768 131072 --layouts causal offset
    # PR #60618's kernel table (B300, 16,384-row chunks, top-2048):
    python benchmarks/kernels/benchmark_sparse_indexer_topk.py prefill-indexer \\
        --real-captures <dir> --dtypes bf16

"""

import csv
import glob
import itertools
import math
import os
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch
from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk
from vllm.model_executor.layers.sparse_attn_indexer import as_fp32_logits
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    is_workspace_manager_initialized,
)

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
REAL_SOURCES = ["mrcr2needle", "mrcr8needle", "random"]
SYNTHETIC = ["gaussian", "narrow", "narrow_neg", "zeros"]
SYNTHETIC += ["sample_trap", "sample_anti", "block", "recent"]
DISTRIBUTIONS = SYNTHETIC + [f"real_{s}" for s in REAL_SOURCES]
# Positions a sampled cutoff looks at (split-row top-k samples 2048).
SAMPLES = 2048

# A prepared call: the timed function and its path ("" or "+up"), or None if
# the backend cannot take the input. Decode and prefill backends take the
# logits as the argument (``Call``); prefill-indexer calls take none.
Call = tuple[Callable[[torch.Tensor], None], str] | None
Prepared = tuple[Callable[[], None], str] | None

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def make_scores(dist: str, lens: torch.Tensor, width: int, k: int, gen):
    """FP32 scores [rows, width] of a synthetic distribution; the
    position-dependent ones are placed relative to each row's length."""
    shape = (lens.numel(), width)
    if dist == "zeros":
        x = torch.zeros(shape, device="cuda")
        cols = torch.rand(shape, generator=gen, device="cuda").topk(k - 10).indices
        top = 5.0 + torch.rand((shape[0], k - 10), generator=gen, device="cuda")
        return x.scatter_(1, cols, top)
    if dist in ("narrow", "narrow_neg"):
        x = 1.0 + 0.01 * torch.rand(shape, generator=gen, device="cuda")
        return -x if dist == "narrow_neg" else x
    x = torch.randn(shape, generator=gen, device="cuda")
    n = lens.reshape(-1, 1).long().clamp(min=1)
    pos = ((2 * torch.arange(SAMPLES, device="cuda") + 1) * n) // (2 * SAMPLES)
    cols = torch.arange(width, device="cuda")
    if dist == "sample_trap":
        top = 10 + torch.rand(pos[:, ::2].shape, generator=gen, device="cuda")
        x.scatter_(1, pos[:, ::2], top)
    elif dist == "sample_anti":
        x.add_(5).scatter_(1, pos, -50.0)
    elif dist == "block":
        start = (n // 2).clamp(max=max(width - 4 * k, 0))
        x += 10 * ((cols >= start) & (cols < start + 4 * k))
    elif dist == "recent":
        age = n - 1 - cols
        x += torch.where((age >= 0) & (age < 8192), 3.5 * (1 - age / 8192), 0.0)
        x[:, :4] += 6
        needles = torch.randint(0, width, (shape[0], 64), generator=gen, device="cuda")
        x.scatter_add_(1, needles, torch.full(needles.shape, 4.0, device="cuda"))
    elif dist != "gaussian":
        raise ValueError(f"unknown distribution: {dist}")
    return x


def real_logits(source: str, width: int, rows: int, args):
    """FP32 logits [rows, width] and lengths of the last `rows` query rows of
    a captured prefill, or None. The indexer's dense logits are
    sum_h w[r, h] * relu(q[r, h] . k[j] * ks[j]), as DeepGEMM computes them."""
    n = max(256, rows)
    path = os.path.join(
        args.real_cache, f"{source}_w{width}_l{args.real_layer}_r{n}_fp32.pt"
    )
    if not os.path.exists(path):
        name = f"{source}_keys{width}_layer{args.real_layer}.pt"
        found = sorted(
            glob.glob(
                os.path.join(
                    args.real_captures or "",
                    f"{source}*_keys{width}_layer{args.real_layer}.pt",
                )
            )
        )
        exact = [f for f in found if os.path.basename(f) == name]
        if not args.real_captures or not (exact or found):
            return None
        x = torch.load((exact or found)[0], map_location="cuda", weights_only=True)
        if x["k"].shape[0] != width or x["q"].shape[0] < n:
            return None
        q, w = x["q"][-n:].float(), x["w"][-n:].float()
        ks = x["ks"].float().reshape(-1)
        logits = torch.empty((n, width), device="cuda")
        step = max(1, (1 << 28) // (4 * n * q.shape[1]))  # ~256 MB intermediate
        for j in range(0, width, step):
            kf = x["k"][j : j + step].float() * ks[j : j + step, None]
            s = torch.einsum("rhd,cd->rhc", q, kf).relu_()
            logits[:, j : j + step] = torch.einsum("rhc,rh->rc", s, w)
        lens = (x["ends"][-n:] - x["starts"][-n:]).int()
        os.makedirs(args.real_cache, exist_ok=True)
        torch.save({"logits": logits.cpu(), "lens": lens.cpu()}, path)
    data = torch.load(path, map_location="cuda", weights_only=True)
    return data["logits"][-rows:], data["lens"][-rows:].int()


@dataclass
class CapturedRow:
    logits: torch.Tensor  # [width] on the CPU, as captured
    end: int
    k: int
    stride: int


def length_bucket(n: int) -> int:
    """The power of two nearest `n` in log2 (1040001 -> 1048576)."""
    return 1 << round(math.log2(max(n, 1)))


_CAPTURES: dict[str, dict[int, list[CapturedRow]]] = {}


def decode_captures(label: str, args) -> dict[int, list[CapturedRow]]:
    """Rows of every capture in --decode-captures `label=<dir>`, by length
    bucket. Files are read only."""
    if label not in _CAPTURES:
        dirs = dict(s.split("=", 1) for s in args.decode_captures or [])
        pools: dict[int, list[CapturedRow]] = {}
        for path in sorted(glob.glob(os.path.join(dirs[label], "*.pt"))):
            d = torch.load(path, map_location="cpu", weights_only=True)
            ends = d["row_ends"].reshape(-1).tolist()
            stride = int(d.get("stride0", d["logits"].shape[1]))
            for row, end in zip(d["logits"], ends):
                pools.setdefault(length_bucket(end), []).append(
                    CapturedRow(row.clone(), int(end), int(d["k"]), stride)
                )
        _CAPTURES[label] = pools
    return _CAPTURES[label]


def captured_inputs(label: str, rows: int, width: int, seed: int, args):
    """`rows` captured rows of length bucket `width` (a different draw per
    seed) in their captured storage, their lengths and k; or None."""
    pool = decode_captures(label, args).get(width)
    if not pool:
        return None
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(pool), generator=gen).tolist()
    picks = [pool[order[i % len(pool)]] for i in range(rows)]
    cols = picks[0].logits.numel()
    stride = max(p.stride for p in picks)
    storage = torch.full(
        (rows, stride), math.nan, dtype=picks[0].logits.dtype, device="cuda"
    )
    for i, p in enumerate(picks):
        storage[i, :cols] = p.logits.to(device="cuda")
    lens = torch.tensor([p.end for p in picks], dtype=torch.int32, device="cuda")
    return storage[:, :cols], lens, picks[0].k


def row_spans(layout: str, rows: int, width: int, k: int, gen, args):
    """int32 (starts, lengths) of each row; see the module docstring."""
    r = torch.arange(rows, dtype=torch.int32, device="cuda")
    starts = torch.zeros_like(r)
    if layout == "full":
        return starts, torch.full_like(r, max(1, round(width * args.valid)))
    if layout in ("ragged", "mixed"):
        lo = min(k, width)
        hi = width if layout == "ragged" else max(lo, min(args.short_max, width))
        lens = torch.randint(lo, hi + 1, (rows,), generator=gen, device="cuda")
        if layout == "mixed":
            n = min(args.long_rows, rows)
            slots = [round((i + 0.5) * rows / n - 0.5) for i in range(n)]
            lens[slots] = width
        return starts, lens.int()
    lens = (width - rows + 1 + r).clamp(min=0)
    if layout == "offset":
        starts = r * width // (2 * rows)
        lens = torch.minimum(lens, width - starts)
    return starts, lens


def inputs(dist: str, layout: str, rows: int, width: int, k: int, seed: int, args):
    """(FP32 scores [rows, width] with NaN outside every row's span, int32
    starts, lengths, k), or None if unavailable. Captured decode rows keep
    their dtype and captured storage."""
    if dist.startswith("cap_"):
        got = captured_inputs(dist.removeprefix("cap_"), rows, width, seed, args)
        if got is None:
            return None
        x, lens, cap_k = got
        return x, torch.zeros_like(lens), lens, cap_k
    if dist.startswith("real_"):
        got = real_logits(dist.removeprefix("real_"), width, rows, args)
        if got is None:
            return None
        x, lens = got
        return x, torch.zeros_like(lens), lens, k
    gen = torch.Generator(device="cuda").manual_seed(seed)
    starts, lens = row_spans(layout, rows, width, k, gen, args)
    x = make_scores(dist, lens, width, k, gen)
    cols = torch.arange(width, device="cuda")[None]
    span = (cols >= starts[:, None]) & (cols < (starts + lens)[:, None])
    return x.masked_fill(~span, math.nan), starts, lens, k


def aligned_rows(x: torch.Tensor, width: int, stride: int = 0) -> torch.Tensor:
    """Copy into whole-row storage (some kernels read full strides) whose row
    stride DeepSelect accepts and is at least `stride`; columns past x's
    width are NaN."""
    align = 16
    try:
        import vllm._deepselect_C  # noqa: F401

        align = max(align, torch.ops.deep_select.get_alignment_requirement()[0])
    except ImportError:
        pass
    elems = align // x.element_size()
    stride = (max(width, stride) + elems - 1) // elems * elems
    out = torch.full((x.shape[0], stride), math.nan, dtype=x.dtype, device=x.device)
    out[:, : x.shape[1]] = x
    return out[:, :width]


def check(
    scores: torch.Tensor,
    starts: torch.Tensor,
    lens: torch.Tensor,
    out: torch.Tensor,
    k: int,
) -> bool:
    """Count, range, uniqueness, -1 fill and selected values vs torch.topk.
    Row r spans scores[r, starts[r]:starts[r] + lens[r]] and its indices are
    relative to starts[r]."""
    cols = torch.arange(max(int(lens.max()), 1), device=scores.device)
    idx = (starts.long()[:, None] + cols).clamp(max=scores.shape[1] - 1)
    scores = scores.float().gather(1, idx)
    n = lens.long().clamp(max=k)
    valid = out >= 0
    if ((out < 0) & (out != -1)).any() or not torch.equal(valid.sum(1), n):
        return False
    if ((out >= lens[:, None]) & valid).any():
        return False
    s = torch.where(valid, out, -1).sort(1).values
    if ((s[:, 1:] == s[:, :-1]) & (s[:, 1:] >= 0)).any():
        return False
    visible = torch.where(cols[None] < lens[:, None], scores, -math.inf)
    if visible.isnan().any():  # NaN in range: order undefined, structure only
        return True
    ref = visible.topk(min(k, visible.shape[1]), dim=1).values
    ref = torch.nn.functional.pad(ref, (0, k - ref.shape[1]), value=-math.inf)
    ref = ref.masked_fill(
        torch.arange(k, device=ref.device)[None] >= n[:, None], -math.inf
    )
    got = scores.gather(1, out.clamp(min=0).long()).masked_fill(~valid, -math.inf)
    return torch.equal(got.sort(1, descending=True).values, ref)


# ---------------------------------------------------------------------------
# Decode backends: prepare(example, starts, lens, k, out) -> Call
# ---------------------------------------------------------------------------

DECODE_BACKENDS = [
    "auto",
    "cooperative",
    "persistent",
    "deep_select",
    "flashinfer",
    "flashinfer_gvr2",
    "per_row",
    "decode_topk",
]
EXTRA_DECODE_BACKENDS: dict[str, Callable[[], Callable[..., Call]]] = {}


def decode_module_backend(module) -> Callable[..., Call]:
    """Adapt a SparseIndexerTopk-like module (resolve_backend/forward, optional
    takes_bf16_logits) to the decode path: BF16 it cannot read is upcast."""

    def prepare(example, starts, lens, k, out) -> Call:
        rows = example.shape[0]
        takes = getattr(module, "takes_bf16_logits", None)
        direct = example.dtype == torch.float32 or (
            takes is not None and takes(example, k, rows)
        )
        try:
            module.resolve_backend(
                example if direct else as_fp32_logits(example, lens), k, rows
            )
        except (RuntimeError, ValueError):
            return None
        max_seq_len = int(lens.max())

        def run(x: torch.Tensor) -> None:
            y = x if direct else as_fp32_logits(x, lens)
            module.forward(y, lens, 1, out, k, max_seq_len)

        return run, "" if direct else "+up"

    return prepare


def decode_topk_backend() -> Callable[..., Call]:
    """DSv4.1 specialized-decode top-k (token mode): BF16 only, SM100."""
    try:
        from vllm.models.deepseek_v41.nvidia.specialized_decode.attention import (
            decode_topk_cute as dk,
        )
    except ImportError:
        return lambda *a: None

    def prepare(example, starts, lens, k, out) -> Call:
        if example.dtype != torch.bfloat16 or not dk.supports_decode_topk(
            example, example.shape[0], k, block_mode=False
        ):
            return None
        return lambda x: dk.decode_topk(x, lens, out, block_mode=False), ""

    return prepare


def make_decode_backend(name: str) -> Callable[..., Call]:
    if name in EXTRA_DECODE_BACKENDS:
        return EXTRA_DECODE_BACKENDS[name]()
    if name == "decode_topk":
        return decode_topk_backend()
    return decode_module_backend(SparseIndexerTopk(name))


# ---------------------------------------------------------------------------
# Prefill backends: prepare(example, starts, lens, k, out) -> Call
# ---------------------------------------------------------------------------

PREFILL_BACKENDS = ["per_row", "deep_select", "flashinfer_gvr2", "aiter"]


def prefill_fp32_backend(launch: Callable, column_zero: bool = False):
    """An FP32-only kernel `launch(logits, starts, ends, out, k)`: BF16
    logits are upcast inside the timed call. `column_zero` kernels read
    every row from column 0."""

    def prepare(example, starts, lens, k, out) -> Call:
        if column_zero and bool(starts.any()):
            return None
        ends = starts + lens
        up = example.dtype != torch.float32

        def run(x: torch.Tensor) -> None:
            launch(as_fp32_logits(x, ends) if up else x, starts, ends, out, k)

        return run, "+up" if up else ""

    return prepare


def per_row_prefill(logits, starts, ends, out, k) -> None:
    torch.ops._C.top_k_per_row_prefill(
        logits, starts, ends, out, out.shape[0], *logits.stride(), k
    )


def gvr2_prefill(logits, starts, ends, out, k) -> None:
    from flashinfer.topk_varlen.kernels.gvr2_topk_host import run_varlen

    run_varlen(
        logits,
        None,
        ends,
        out,
        next_n=1,
        compress_ratio=1,
        max_seq_len=logits.shape[1],
        top_k=k,
    )


def aiter_prefill(logits, starts, ends, out, k) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _launch_aiter_top_k_per_row_prefill,
    )

    _launch_aiter_top_k_per_row_prefill(logits, starts, ends, out, k)


def deep_select_prefill_backend() -> Callable[..., Call]:
    from vllm.model_executor.layers.indexer_topk import (
        deep_select_topk,
        is_deep_select_supported,
    )

    def prepare(example, starts, lens, k, out) -> Call:
        if bool(starts.any()) or not is_deep_select_supported(example, k):
            return None
        return lambda x: deep_select_topk(x, k, end=lens, output_idx=out), ""

    return prepare


def make_prefill_backend(name: str) -> Callable[..., Call]:
    sm100 = current_platform.is_cuda() and (
        current_platform.is_device_capability_family(100)
    )
    if name == "per_row":
        return prefill_fp32_backend(per_row_prefill)
    if name == "deep_select" and sm100:
        return deep_select_prefill_backend()
    if name == "flashinfer_gvr2" and sm100:
        return prefill_fp32_backend(gvr2_prefill, column_zero=True)
    if name == "aiter" and current_platform.is_rocm():
        from vllm._aiter_ops import rocm_aiter_ops

        if rocm_aiter_ops.is_indexer_top_k_enabled():
            return prefill_fp32_backend(aiter_prefill)
    return lambda *a: None


# Rows on both sides of each decode launch-shape switch (see the docstring).
DECODE_ROWS = [1, 4, 5, 8, 9, 15, 16, 22, 23, 32, 33, 34, 48, 49, 50, 64, 65]
DECODE_ROWS += [96, 128, 256]

MODES = {
    # mode: (backend factory, default backends, layouts, default rows, widths)
    "decode": (
        make_decode_backend,
        DECODE_BACKENDS,
        ["full", "ragged", "mixed"],
        DECODE_ROWS,
        [4096, 16384, 32768, 65536, 131072, 262144, 524288, 1048576],
    ),
    "prefill": (
        make_prefill_backend,
        PREFILL_BACKENDS,
        ["causal", "full", "offset"],
        [256, 1024, 2048, 4096, 8192],
        [8192, 16384, 32768, 65536, 131072],
    ),
}


# ---------------------------------------------------------------------------
# Input feeds: what the timed calls read (warm, cold L2, after scoring)
# ---------------------------------------------------------------------------


@dataclass
class Feed:
    """`example` is the input that prepare and --check see. `copies`
    rotate across the calls captured in a graph (cold L2); `score` (after
    scoring) produces the logits inside each call."""

    example: torch.Tensor
    copies: list[torch.Tensor]
    score: Callable[[], torch.Tensor] | None = None

    def bind(self, call: Callable[[torch.Tensor], None]) -> Callable[[], None]:
        if self.score is not None:
            score = self.score
            return lambda: call(score())
        if len(self.copies) == 1:
            x = self.copies[0]
            return lambda: call(x)
        it = itertools.cycle(self.copies)
        return lambda: call(next(it))


def storage_bytes(x: torch.Tensor) -> int:
    return x.stride(0) * x.shape[0] * x.element_size()


def cold_copies(x: torch.Tensor, budget: int) -> list[torch.Tensor]:
    """`x` and copies of it (same strides) covering 2.5x the L2 cache, at
    most `budget` bytes in all (inputs that large are cold already)."""
    l2 = torch.cuda.get_device_properties(x.device).L2_cache_size
    size = storage_bytes(x)
    n = max(2, math.ceil(2.5 * l2 / size))
    n = max(1, min(n, budget // size))
    copies = [x]
    for _ in range(n - 1):
        c = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
        copies.append(c.copy_(x))
    return copies


class PagedScoring:
    """The decode indexer's scoring kernel (DeepGEMM varlen paged MQA logits,
    SM100): row r scores `lens[r]` keys of its own request into
    [rows, width] logits of `dtype`. Random FP8 queries and KV cache; each
    row's KV blocks are distinct up to `kv_budget` bytes, then reused."""

    BLOCK = 64
    DIM = 128

    def __init__(self, lens, width, dtype, heads, kv_budget, seed):
        from vllm.utils.deep_gemm import (
            fp8_fp4_paged_mqa_logits,
            get_num_sms,
            get_paged_mqa_logits_metadata,
            mqa_logits_mx_inputs,
            mqa_logits_use_mx_scales,
        )

        if not mqa_logits_use_mx_scales():
            raise RuntimeError("--after-scoring needs SM100 DeepGEMM (MX scales)")
        gen = torch.Generator(device="cuda").manual_seed(seed)
        rows = lens.numel()
        blocks = math.ceil(width / self.BLOCK)
        bytes_per_block = self.BLOCK * (self.DIM + 4)
        pool = max(blocks, min(rows * blocks, kv_budget // bytes_per_block))
        fp8 = current_platform.fp8_dtype()
        kv = torch.randn(pool, self.BLOCK * self.DIM, generator=gen, device="cuda").to(
            fp8
        )
        # Four equal UE8M0 bytes per token: scale 1.0 (exponent 127).
        scales = torch.full((pool, self.BLOCK * 4), 127, dtype=torch.uint8)
        self.kv = torch.cat([kv.view(torch.uint8), scales.cuda()], 1).view(
            pool, self.BLOCK, 1, self.DIM + 4
        )
        r = torch.arange(rows, device="cuda")[:, None] * blocks
        self.tables = ((r + torch.arange(blocks, device="cuda")) % pool).int()
        q = torch.randn(rows, 1, heads, self.DIM, generator=gen, device="cuda")
        q = q.to(fp8)
        weights = torch.randn(rows, heads, generator=gen, device="cuda")
        q_scale, self.weights = mqa_logits_mx_inputs(q, None, weights, dtype)
        self.q = (q, q_scale)
        self.lens = lens.reshape(-1, 1).int()
        self.indices = torch.arange(rows, dtype=torch.int32, device="cuda")
        self.schedule = get_paged_mqa_logits_metadata(
            self.lens, self.BLOCK, get_num_sms(), indices=self.indices
        )
        self.width, self.dtype = width, dtype
        self._run = fp8_fp4_paged_mqa_logits

    def __call__(self) -> torch.Tensor:
        return self._run(
            self.q,
            self.kv,
            self.weights,
            self.lens,
            self.tables,
            self.schedule,
            max_model_len=self.width,
            logits_dtype=self.dtype,
            indices=self.indices,
        )


# ---------------------------------------------------------------------------
# Co-tenant guard
# ---------------------------------------------------------------------------


class CotenantGuard:
    """Other compute processes on this process's GPU (NVML), or None when
    NVML is unavailable (the guard is then off)."""

    def __init__(self) -> None:
        self.handle = None
        try:
            from vllm.third_party import pynvml

            pynvml.nvmlInit()
            uuid = str(torch.cuda.get_device_properties(0).uuid)
            uuid = uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid.encode())
        except Exception as e:  # noqa: BLE001
            print(f"# co-tenant guard off: {e}")

    def others(self) -> set[int]:
        if self.handle is None:
            return set()
        procs = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        return {p.pid for p in procs} - {os.getpid()}


# ---------------------------------------------------------------------------
# Prefill indexer: MQA scoring + top-k (dense) vs fused_mqa_topk_prefill
# ---------------------------------------------------------------------------

INDEXER_BACKENDS = ["dense+per_row", "dense+deep_select", "fused"]
# Opt-in: "dense+flashinfer_gvr2".


@dataclass
class Chunk:
    """The last `rows` query rows of a captured prefill chunk."""

    name: str
    q: torch.Tensor  # [rows, H, D] fp8
    k: torch.Tensor  # [keys, D] fp8
    k_scales: torch.Tensor  # [keys] fp32 (fused)
    k_dense: torch.Tensor  # [keys] as fp8_fp4_mqa_logits takes them
    w: torch.Tensor  # [rows, H] fp32
    starts: torch.Tensor  # [rows] int32
    ends: torch.Tensor  # [rows] int32
    max_seqlen_k: int  # longest row (host value: the calls are graph-captured)

    @property
    def dense_rows(self) -> int:
        """Rows per dense call under the logits budget."""
        from vllm.v1.attention.backends.mla.indexer import dense_logits_rows

        return dense_logits_rows(self.max_seqlen_k)

    def windows(self) -> list[slice]:
        rows, step = self.q.shape[0], self.dense_rows
        return [slice(lo, min(lo + step, rows)) for lo in range(0, rows, step)]

    def logits(self, sl: slice, dtype: torch.dtype) -> torch.Tensor:
        """Dense MQA logits of rows `sl`, relative to each row's start."""
        from vllm.utils.deep_gemm import fp8_fp4_mqa_logits, mqa_logits_mx_inputs

        q_scale, w = mqa_logits_mx_inputs(self.q[sl], None, self.w[sl], dtype)
        return fp8_fp4_mqa_logits(
            (self.q[sl], q_scale),
            (self.k, self.k_dense),
            w,
            self.starts[sl],
            self.ends[sl],
            self.max_seqlen_k,
            logits_dtype=dtype,
        )


def capture_chunks(args):
    """Chunks of every capture in --real-captures with at least --min-keys
    keys, for each --rows (the last rows see the most keys)."""
    from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
        fused_mqa_topk_available,
        fused_mqa_topk_key_scales,
    )
    from vllm.utils.deep_gemm import mqa_logits_use_mx_scales

    def keys_of(path: str) -> int:  # <source>_keys<N>[_layer<L>].pt
        return int(re.search(r"_keys(\d+)", os.path.basename(path)).group(1))

    paths = glob.glob(os.path.join(args.real_captures or "", "*_keys*.pt"))
    for path in sorted(paths, key=lambda p: (keys_of(p), p)):
        if keys_of(path) < args.min_keys:
            continue
        x = torch.load(path, map_location="cuda", weights_only=True)
        # Power-of-two key scales as the SM100 indexer K cache stores them
        # (4 equal UE8M0 bytes); the fused op takes them back as FP32.
        s = x["ks"].float().reshape(-1)
        exp = torch.where(s > 0, torch.log2(s).round().long() + 127, 0)
        ue8m0 = (exp * 0x01010101).to(torch.int32)
        k_dense = ue8m0 if mqa_logits_use_mx_scales() else s
        if fused_mqa_topk_available():
            s = fused_mqa_topk_key_scales(ue8m0.view(torch.uint8).view(-1, 4))
        for rows in args.rows:
            if rows > x["q"].shape[0]:
                continue
            yield Chunk(
                os.path.basename(path).removesuffix(".pt"),
                x["q"][-rows:].contiguous(),
                x["k"],
                s,
                k_dense,
                x["w"][-rows:].float().contiguous(),
                x["starts"][-rows:].int().contiguous(),
                x["ends"][-rows:].int().contiguous(),
                int((x["ends"][-rows:] - x["starts"][-rows:]).max()),
            )
        del x
        torch.cuda.empty_cache()


def dense_backend(name: str, c: Chunk, dtype: torch.dtype, k: int, out) -> Prepared:
    """The vLLM dense prefill path: per window, MQA logits in `dtype`, then
    the top-k (FP32-only top-ks get production's upcast, marked "+up")."""
    from vllm.model_executor.layers.indexer_topk import (
        deep_select_topk,
        is_deep_select_supported,
    )

    topk = name.removeprefix("dense+")
    windows = c.windows()
    probe = c.logits(windows[0], dtype)
    if topk == "deep_select":
        # Production ranks only BF16 logits with DeepSelect.
        if dtype != torch.bfloat16 or not is_deep_select_supported(probe, k):
            return None
    elif topk not in ("per_row", "flashinfer_gvr2"):
        raise ValueError(f"unknown dense top-k: {topk}")
    up = topk != "deep_select" and dtype != torch.float32

    def run() -> None:
        for sl in windows:
            lens = c.ends[sl] - c.starts[sl]
            logits = c.logits(sl, dtype)
            if up:
                logits = as_fp32_logits(logits, lens)
            if topk == "deep_select":
                deep_select_topk(logits, k, end=lens, output_idx=out[sl])
            elif topk == "per_row":
                per_row_prefill(logits, torch.zeros_like(lens), lens, out[sl], k)
            else:
                gvr2_prefill(logits, None, lens, out[sl], k)

    return run, "+up" if up else ""


def fused_backend(c: Chunk, k: int, out) -> Prepared:
    from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
        FUSED_MQA_TOPK_TOPKS,
        fused_mqa_topk_available,
        fused_mqa_topk_prefill,
    )

    if not fused_mqa_topk_available() or k not in FUSED_MQA_TOPK_TOPKS:
        return None
    args = (c.q, c.k, c.k_scales, c.w, c.starts, c.ends, out)
    return lambda: fused_mqa_topk_prefill(*args), ""


def reference_scores(c: Chunk, ids: torch.Tensor) -> torch.Tensor:
    """FP32 scores of rows `ids`: sum_h w * relu(q . k) * s, -inf outside
    each row's key range (absolute key columns)."""
    q, w = c.q[ids], c.w[ids]
    kf = c.k.float()
    out = torch.zeros(q.shape[0], kf.shape[0], device=q.device)
    for h in range(q.shape[1]):
        out += w[:, h, None] * torch.relu(q[:, h].float() @ kf.T)
    out *= c.k_scales[None]
    cols = torch.arange(kf.shape[0], device=q.device)[None]
    outside = (cols < c.starts[ids, None]) | (cols >= c.ends[ids, None])
    return out.masked_fill(outside, -math.inf)


def exact(c: Chunk, outs: list[torch.Tensor | None], k: int) -> list[bool | None]:
    """#60618's "exact top-k": every row holds min(length, k) distinct
    in-range indices, then -1, and on 65 evenly spaced rows the selected keys'
    FP32 scores equal the FP32 top-k scores (rtol 2e-4, atol 1e-4)."""
    rows = c.q.shape[0]
    ids = torch.unique(
        torch.cat(
            [
                torch.arange(0, rows, max(1, rows // 64), device="cuda"),
                torch.tensor([rows - 1], device="cuda"),
            ]
        )
    )
    scores = reference_scores(c, ids)
    lens = (c.ends - c.starts).long()
    n = lens.clamp(max=k)
    want = scores.topk(min(k, scores.shape[1]), dim=1).values
    slots = torch.arange(want.shape[1], device="cuda")[None]
    want = want.masked_fill(slots >= n[ids, None], -math.inf)
    results: list[bool | None] = []
    for out in outs:
        if out is None:
            results.append(None)
            continue
        valid = out >= 0
        ok = bool((valid.sum(1) == n).all()) and bool((out[~valid] == -1).all())
        ok &= not bool(((out >= lens[:, None]) & valid).any())
        srt = torch.where(valid, out, -1).sort(1).values
        ok &= not bool(((srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] >= 0)).any())
        key = (out[ids].long() + c.starts[ids, None]).clamp(min=0)
        got = scores.gather(1, key).masked_fill(~valid[ids], -math.inf)
        got = got.sort(1, descending=True).values[:, : want.shape[1]]
        ok &= torch.allclose(got, want, rtol=2e-4, atol=1e-4)
        results.append(ok)
    return results


def bench_indexer(args) -> list[dict]:
    from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
        fused_mqa_topk_chunk,
    )

    dense = [b for b in args.backends if b.startswith("dense+")]
    names = [f"{b}[{dt}]" for dt in args.dtypes for b in dense]
    names += [b for b in args.backends if b == "fused"]
    table = []
    for c in capture_chunks(args):
        rows, keys, max_len = c.q.shape[0], c.k.shape[0], c.max_seqlen_k
        for k in args.top_k:
            outs = [
                torch.full((rows, k), -7, dtype=torch.int32, device="cuda")
                for _ in names
            ]
            prepared = []
            for name, out in zip(names, outs):
                if name == "fused":
                    prepared.append(fused_backend(c, k, out))
                else:
                    b, dt = name[:-1].split("[")
                    prepared.append(dense_backend(b, c, DTYPES[dt], k, out))
            res = time_prepared(prepared, args.rounds, args.rep)
            ok = exact(c, [o if p else None for o, p in zip(outs, prepared)], k)
            for r, good in zip(res, ok):
                if r is not None:
                    r.ok = good
            fused = fused_mqa_topk_chunk(max_len, keys, k, rows, c.dense_rows)
            label = (
                f"{c.name:34s} {rows:5d}x{keys:<7d} k={k:<5d} "
                f"{'fused' if fused else 'dense'}-routed"
            )
            rows_csv = report(
                label,
                names,
                res,
                dict(
                    mode=args.mode,
                    capture=c.name,
                    rows=rows,
                    keys=keys,
                    max_seqlen_k=max_len,
                    k=k,
                    dense_calls=len(c.windows()),
                    production=("fused" if fused else "dense"),
                ),
            )
            table += rows_csv
    summarize_indexer(table, names)
    return table


def summarize_indexer(table: list[dict], names: list[str]) -> None:
    """PR #60618-style table: per (keys, rows, k), the median over captures
    of each path's time, fused's speedup over each dense path (median and
    range across captures) and how many captures each path got exact."""
    groups: dict[tuple, dict[str, dict[str, dict]]] = {}
    for row in table:
        key = (int(row["keys"]), int(row["rows"]), int(row["k"]))
        groups.setdefault(key, {}).setdefault(row["capture"], {})[row["backend"]] = row
    ran = {row["backend"] for row in table}
    names = [n for n in names if n in ran]
    dense = [n for n in names if n != "fused"]
    head = ["Keys", "Rows", "k", "Captures", "Calls"] + [f"{n} ms" for n in names]
    head += [f"fused vs {n}" for n in dense if "fused" in names]
    head += [f"exact {n}" for n in names]
    print("\n| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    for (keys, rows, k), caps in sorted(groups.items()):
        cells = [f"{keys}", f"{rows}", f"{k}", f"{len(caps)}"]
        cells.append(str(next(iter(next(iter(caps.values())).values()))["dense_calls"]))
        for n in names:
            ms = [float(c[n]["us"]) / 1e3 for c in caps.values() if n in c]
            cells.append(f"{statistics.median(ms):.3f}" if ms else "-")
        for n in dense if "fused" in names else []:
            sp = [
                float(c[n]["us"]) / float(c["fused"]["us"])
                for c in caps.values()
                if n in c and "fused" in c
            ]
            cells.append(
                f"{statistics.median(sp):.2f}x ({min(sp):.2f}-{max(sp):.2f})"
                if sp
                else "-"
            )
        for n in names:
            got = [c[n]["ok"] for c in caps.values() if n in c]
            cells.append(f"{sum(int(g) for g in got)}/{len(got)}" if got else "-")
        print("| " + " | ".join(cells) + " |")


# ---------------------------------------------------------------------------
# Timing and reporting
# ---------------------------------------------------------------------------


@dataclass
class Result:
    us: float
    lo: float
    hi: float
    path: str
    ok: bool | None
    worst: float | None = None
    seeds: int = 1


def bench_config(
    make_backend, names, feed: Feed, starts, lens, k, args
) -> tuple[list[Result | None], float | None]:
    """Results per backend, and the scoring kernel's time alone (after
    scoring), which the results exclude."""
    x = feed.example
    rows = x.shape[0]
    outs = [torch.full((rows, k), -7, dtype=torch.int32, device="cuda") for _ in names]
    calls = [make_backend(n)(x, starts, lens, k, o) for n, o in zip(names, outs)]
    prepared: list[Prepared] = [
        (feed.bind(c[0]), c[1]) if c is not None else None for c in calls
    ]
    if feed.score is not None:
        prepared.append((lambda: feed.score(), "score"))
    verify = (lambda i: check(x, starts, lens, outs[i], k)) if args.check else None
    res = time_prepared(prepared, args.rounds, args.rep, verify)
    if feed.score is None:
        return res, None
    score = res.pop()
    if score is None:
        raise RuntimeError("the scoring kernel failed")
    for r in res:
        if r is not None:
            r.us, r.lo, r.hi = r.us - score.us, r.lo - score.us, r.hi - score.us
    return res, score.us


def time_prepared(
    prepared: list[Prepared], rounds: int, rep: int, verify=None
) -> list[Result | None]:
    """Time each prepared call (None: skipped) with `do_bench_cudagraph`
    (`rep` ms, median) over `rounds` rounds with the order rotated;
    `verify(i)` checks call i's output after its first run."""
    prepared = list(prepared)
    # FlashInfer gvr_2-style kernels keep a scratch buffer per CUDA stream and
    # can only create it outside graph capture. do_bench_cudagraph captures on
    # a different stream than it warms up on, so run every backend once on the
    # capture stream first. This is also the output verify() checks.
    if torch.cuda.graph.default_capture_stream is None:
        torch.cuda.graph.default_capture_stream = torch.cuda.Stream()
    capture = torch.cuda.graph.default_capture_stream
    capture.wait_stream(torch.cuda.current_stream())
    for i, p in enumerate(prepared):
        if p is None:
            continue
        try:
            with torch.cuda.stream(capture):
                p[0]()
            capture.synchronize()
        except (RuntimeError, AssertionError, ValueError):
            prepared[i] = None
    # do_bench_cudagraph runs on its own stream; inputs must be ready first.
    torch.accelerator.synchronize()

    live = [i for i, p in enumerate(prepared) if p is not None]
    ok = {i: verify(i) if verify and prepared[i][1] != "score" else None for i in live}
    samples: dict[int, list[float]] = {i: [] for i in live}
    for r in range(rounds):
        for j in range(len(live)):
            i = live[(r + j) % len(live)]
            ms = triton.testing.do_bench_cudagraph(
                prepared[i][0], rep=rep, return_mode="median"
            )
            samples[i].append(ms * 1e3)
    return [
        Result(
            statistics.median(samples[i]),
            min(samples[i]),
            max(samples[i]),
            prepared[i][1],
            ok[i],
        )
        if i in samples
        else None
        for i in range(len(prepared))
    ]


def merge_seeds(per_seed: list[list[Result | None]]) -> list[Result | None]:
    """One result per backend over seeds: the median and the worst of the
    per-seed medians, min-max over every round, exact only if every seed
    was."""
    merged: list[Result | None] = []
    for results in zip(*per_seed):
        got = [r for r in results if r is not None]
        if not got:
            merged.append(None)
            continue
        oks = [r.ok for r in got if r.ok is not None]
        merged.append(
            Result(
                statistics.median(r.us for r in got),
                min(r.lo for r in got),
                max(r.hi for r in got),
                got[0].path,
                all(oks) if oks else None,
                max(r.us for r in got),
                len(got),
            )
        )
    return merged


def report(label: str, names, res: list[Result | None], row: dict) -> list[dict]:
    """Print one line (ratios vs the first backend that ran) and return its
    CSV rows."""
    first = next((r for r in res if r), None)
    cells = [
        f"{n}{r.path} {r.us:.1f} ({r.lo:.1f}-{r.hi:.1f})"
        f"{f' worst {r.worst:.1f}' if r.seeds > 1 else ''}"
        f"{'' if r is first else f' {r.us / first.us:.2f}x'}"
        f"{' WRONG' if r.ok is False else ''}"
        for n, r in zip(names, res)
        if r
    ]
    flag = " CONTAMINATED" if row.get("cotenant") else ""
    print(f"{label}{flag} | " + " | ".join(cells), flush=True)
    return [
        dict(
            row,
            backend=n,
            path=r.path,
            us=f"{r.us:.3f}",
            us_worst=f"{r.worst:.3f}" if r.worst is not None else "",
            seeds=r.seeds,
            min=f"{r.lo:.3f}",
            max=f"{r.hi:.3f}",
            ok="" if r.ok is None else int(r.ok),
        )
        for n, r in zip(names, res)
        if r
    ]


def parse_args():
    p = FlexibleArgumentParser(description=__doc__)
    modes = p.add_subparsers(dest="mode", required=True)
    for mode, (_, backends, layouts, rows, widths) in MODES.items():
        m = modes.add_parser(mode, help=f"{mode} top-k")
        m.add_argument("--backends", nargs="+", default=backends)
        m.add_argument("--layouts", nargs="+", default=layouts[:1], choices=layouts)
        m.add_argument(
            "--valid",
            type=float,
            default=1.0,
            help="full layout: fraction of each row's columns before its end",
        )
        m.add_argument("--rows", type=int, nargs="+", default=rows)
        m.add_argument("--widths", type=int, nargs="+", default=widths)
        m.add_argument("--top-k", type=int, nargs="+", default=[512, 1024, 2048])
        m.add_argument(
            "--dtypes", nargs="+", default=list(DTYPES), choices=list(DTYPES)
        )
        m.add_argument(
            "--distributions",
            nargs="+",
            default=["gaussian"],
            help=f"{', '.join(DISTRIBUTIONS)} or cap_<label> (decode)",
        )
        m.add_argument("--rounds", type=int, default=5)
        m.add_argument("--rep", type=int, default=20, help="do_bench_cudagraph ms")
        m.add_argument("--seed", type=int, default=0)
        m.add_argument(
            "--seeds",
            type=int,
            default=1,
            help="inputs per line (synthetic seeds or captured-row draws); "
            "lines report the median and the worst",
        )
        m.add_argument("--check", action="store_true", help="verify vs torch.topk")
        m.add_argument(
            "--cold-l2",
            action="store_true",
            help="rotate input copies covering 2.5x the L2 cache",
        )
        m.add_argument(
            "--cold-l2-budget",
            type=float,
            default=8.0,
            help="GiB of input copies at most",
        )
        m.add_argument(
            "--cotenant-retries",
            type=int,
            default=2,
            help="reruns of a line that had another process on the GPU",
        )
        m.add_argument(
            "--real-captures", default=os.environ.get("VLLM_TOPK_REAL_CAPTURES")
        )
        m.add_argument(
            "--real-cache", default=os.path.expanduser("~/.cache/vllm/topk_real")
        )
        m.add_argument("--real-layer", type=int, default=9)
        m.add_argument("--csv", help="write one row per (configuration, backend)")
        if mode == "decode":
            m.add_argument(
                "--logits-width",
                type=int,
                default=0,
                help="rows this wide (>= width; NaN padding), like decode "
                "logits sized to max_model_len",
            )
            m.add_argument("--long-rows", type=int, default=1, help="mixed layout")
            m.add_argument(
                "--short-max",
                type=int,
                default=65536,
                help="mixed layout: longest short row",
            )
            m.add_argument(
                "--decode-captures",
                nargs="+",
                metavar="LABEL=DIR",
                help="real decode rows for the cap_<label> distributions",
            )
            m.add_argument(
                "--after-scoring",
                action="store_true",
                help="run the paged MQA logits kernel before each top-k",
            )
            m.add_argument("--heads", type=int, default=64, help="after scoring")
            m.add_argument(
                "--kv-budget",
                type=float,
                default=16.0,
                help="after scoring: GiB of distinct KV cache at most",
            )
    m = modes.add_parser(
        "prefill-indexer", help="MQA scoring + top-k vs fused, on captures"
    )
    m.add_argument("--backends", nargs="+", default=INDEXER_BACKENDS)
    m.add_argument(
        "--dtypes",
        nargs="+",
        default=list(DTYPES),
        choices=list(DTYPES),
        help="dense logits dtypes (indexer_logits_dtype)",
    )
    m.add_argument("--rows", type=int, nargs="+", default=[16384])
    m.add_argument("--min-keys", type=int, default=32768)
    m.add_argument("--top-k", type=int, nargs="+", default=[2048])
    m.add_argument("--rounds", type=int, default=1)
    m.add_argument("--rep", type=int, default=200, help="do_bench_cudagraph ms")
    m.add_argument("--real-captures", default=os.environ.get("VLLM_TOPK_REAL_CAPTURES"))
    m.add_argument("--csv", help="write one row per (configuration, backend)")
    args = p.parse_args()
    if args.mode != "prefill-indexer":
        labels = [
            s.split("=", 1)[0] for s in getattr(args, "decode_captures", None) or []
        ]
        for dist in args.distributions:
            label = dist.removeprefix("cap_")
            if dist not in DISTRIBUTIONS and not (
                dist.startswith("cap_") and label in labels
            ):
                p.error(
                    f"unknown distribution {dist!r} (cap_* needs --decode-captures)"
                )
    return args


def make_feed(x: torch.Tensor, lens: torch.Tensor, seed: int, args) -> Feed:
    if getattr(args, "after_scoring", False):
        scoring = PagedScoring(
            lens, x.shape[1], x.dtype, args.heads, int(args.kv_budget * 2**30), seed
        )
        example = scoring()
        return Feed(example, [example], scoring)
    if args.cold_l2:
        return Feed(x, cold_copies(x, int(args.cold_l2_budget * 2**30)))
    return Feed(x, [x])


def bench_line(make_backend, args, dist, layout, rows, width, k, dt):
    """(label, results merged over seeds, CSV fields) of one line, or None."""
    per_seed, scores_us, max_len, line_k = [], [], 0, k
    for s in range(args.seeds):
        got = inputs(dist, layout, rows, width, k, args.seed + s, args)
        if got is None:
            return None
        x, starts, lens, line_k = got
        storage = max(getattr(args, "logits_width", 0), width, x.shape[1])
        scores = aligned_rows(x.to(DTYPES[dt]), storage, x.stride(0))
        feed = make_feed(scores, lens, args.seed + s, args)
        res, score_us = bench_config(
            make_backend, args.backends, feed, starts, lens, line_k, args
        )
        per_seed.append(res)
        max_len = max(max_len, int(lens.max()))
        if score_us is not None:
            scores_us.append(score_us)
        del feed, scores, x
    l2 = (
        "after_scoring"
        if getattr(args, "after_scoring", False)
        else ("cold" if args.cold_l2 else "warm")
    )
    label = f"{dist:16s} {layout:6s} {dt} {rows:5d}x{width:<8d} k={line_k:<5d}"
    if scores_us:
        label += f" score={statistics.median(scores_us):.1f}"
    fields = dict(
        mode=args.mode,
        dist=dist,
        layout=layout,
        dtype=dt,
        rows=rows,
        width=width,
        k=line_k,
        max_seq_len=max_len,
        l2=l2,
        score_us=f"{statistics.median(scores_us):.3f}" if scores_us else "",
    )
    return label, merge_seeds(per_seed), fields


def main() -> None:
    args = parse_args()
    if not is_workspace_manager_initialized():
        init_workspace_manager(torch.device("cuda"))
    mode = args.mode
    l2 = (
        "after scoring"
        if getattr(args, "after_scoring", False)
        else ("cold L2" if getattr(args, "cold_l2", False) else "warm L2")
    )
    print(
        f"# {mode} | {current_platform.get_device_name()} | torch "
        f"{torch.__version__} | {l2} | median us (min-max) over "
        f"{args.rounds} rounds x {getattr(args, 'seeds', 1)} seeds | "
        "ratio vs first listed"
    )
    if mode == "prefill-indexer":
        write_csv(args.csv, bench_indexer(args))
        return
    make_backend = MODES[mode][0]
    guard = CotenantGuard()
    table = []
    done: set[tuple] = set()
    for dist in args.distributions:
        real = dist.startswith(("real_", "cap_"))
        layouts = [dist.split("_")[0]] if real else args.layouts
        for layout, width, rows, k, dt in itertools.product(
            layouts, args.widths, args.rows, args.top_k, args.dtypes
        ):
            if dist.startswith("cap_"):
                # Captured rows carry their own k: one line per dtype.
                if (dist, width, rows, dt) in done:
                    continue
                done.add((dist, width, rows, dt))
            for attempt in range(args.cotenant_retries + 1):
                before = guard.others()
                line = bench_line(make_backend, args, dist, layout, rows, width, k, dt)
                others = before | guard.others()
                if line is None or not others or attempt == args.cotenant_retries:
                    break
                print(f"# co-tenant pids {sorted(others)}: rerun {attempt + 1}")
            if line is None:
                continue
            label, res, fields = line
            fields["cotenant"] = " ".join(map(str, sorted(others)))
            table += report(label, args.backends, res, fields)
    write_csv(args.csv, table)


def write_csv(path: str | None, table: list[dict]) -> None:
    if path and table:
        fields = list(dict.fromkeys(f for row in table for f in row))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(table)


if __name__ == "__main__":
    main()
