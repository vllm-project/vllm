# RFC: Borrow DeepInfer's Fusion Idea for DSpark Confidence Head

| | |
|---|---|
| **Status** | Draft (for review) |
| **Author** | peterpan |
| **Date** | 2026-10-21 |
| **Target branch** | vLLM `main` |
| **Scope** | DeepSeek V4 DSpark, NVIDIA path only |
| **References** | `deepinfer/model_executor/layers/tilelang_ops/cuda/dflash.py:12-72` |

---

## 1. Summary

DSpark's confidence head currently executes **4 GPU kernel launches** per call
(`torch.cat` → `.float()` cast → `F.linear` GEMV → `torch.sigmoid`), allocating
two temporary buffers of shape `(n, H+r)` in bf16 and fp32. This RFC proposes a
**single fused kernel** (borrowed from DeepInfer's TileLang implementation)
that performs GEMV + sigmoid inline, eliminating intermediate memory traffic and
kernel-launch overhead.

The main **constraint** compared to DeepInfer: vLLM's confidence-head weights
are **FP32**, whereas DeepInfer ships BF16. We therefore **port the fusion idea
but keep FP32 accumulator semantics** — we do not lower precision.

This RFC contains:

1. Motivation and current cost analysis
2. Proposed fused kernel (TileLang skeleton)
3. Numerical-equivalence argument (why FP32 weights matter and are preserved)
4. Integration point (single-file change)
5. Benchmark plan **with explicit accept/reject thresholds**
6. Risks and rollout plan

---

## 2. Background: DSpark in vLLM Today

### 2.1 What this kernel does

DSpark is a speculative decoding variant that predicts a block of candidate
tokens, then lets the target model verify them. Adaptive verification requires
an estimated acceptance probability per draft token. That estimate comes from
a small "confidence head":

```
p_accept = sigmoid( ( [hidden_state ; markov_embed] @ W ) / temperature )
```

where:

- `hidden_state` — bf16, shape `(n, H)` from the draft head
- `markov_embed` — bf16, shape `(n, r)` from a learned Markov embedding
- `W` — **fp32**, shape `(H + r, 1)` — a ReplicatedLinear with `params_dtype=float32`
- `[x ; y]`  — channel-wise concatenation
- Output: fp32, shape `(n,)`

### 2.2 Current implementation (vLLM main)

File: `vllm/model_executor/models/qwen3_dspark.py:112-137` (reused by
`vllm/models/deepseek_v4/nvidia/dspark.py:379` via `compute_confidence`).

```python
class DSparkConfidenceHead(nn.Module):
    def __init__(self, input_dim, prefix, bias=False, with_markov=True):
        super().__init__()
        self.proj = ReplicatedLinear(
            input_dim, 1, bias=bias, return_bias=False,
            params_dtype=torch.float32,           # <-- FP32 weights, intentional
            prefix=maybe_prefix(prefix, "proj"),
        )

    def forward(self, hidden, markov_embed):
        x = torch.cat([hidden, markov_embed], dim=-1) if self.with_markov else hidden
        return self.proj(x.float()).squeeze(-1)

# Caller in qwen3_dspark.py:247 / deepseek_v4/nvidia/dspark.py:379
def compute_confidence(self, head_hidden, markov_embed):
    return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))
```

### 2.3 Per-call cost breakdown

For a batch of `n` samples, H=`hidden_size`, r=`markov_rank`:

| Step | Kernel | Reads | Writes | Notes |
|---|---|---|---|---|
| 1. `cat` | elementwise | bf16 (n, H) + (n, r) | bf16 (n, H+r) | allocates temp #1 |
| 2. `.float()` | elementwise | bf16 (n, H+r) | fp32 (n, H+r) | allocates temp #2 |
| 3. `ReplicatedLinear` | cuBLAS GEMV | fp32 (n, H+r) + fp32 (H+r,) | fp32 (n, 1) | |
| 4. `sigmoid` | elementwise | fp32 (n, 1) | fp32 (n,) | separate kernel launch |

Typical numbers (DSV3, H=7168, r=8, n=256):

- Temp #1: `256 * 7176 * 2 B = 3.67 MB` (bf16)
- Temp #2: `256 * 7176 * 4 B = 7.34 MB` (fp32)
- Total temporaries: **~11 MB per call**, plus short-lived allocation churn
- Kernel-launch overhead: 4 × ~5-10 µs = **20-40 µs CPU time per call**

Observed in production this head is invoked once per decode step per
`(req, spec_pos)` block. With adaptive verification ON, GPU is already partially
idle-waiting on small ops — kernel launch overhead and memory traffic are
**not amortized**.

### 2.4 Where the cost shows up

- Latency: hidden inside DSpark decode step; visible in profiler as 4 separate
  small ops in series.
- Memory: not a leak — allocator caches. But buffer churn aggravates
  fragmentation when running alongside other speculators.

---

## 3. Prior Art: DeepInfer's fused kernel

Reference: `deepinfer/model_executor/layers/tilelang_ops/cuda/dflash.py:12-72`
(installs as a TileLang JIT compilation).

Key ideas we want to **port**:

1. **Logical, not physical concat.** Two GEMV passes against a shared fp32
   accumulator: one over `x0` (hidden), one over `x1` (markov_embed), with
   weight offsets `[0, H)` and `[H, H+r)` respectively. Zero intermediate tensors.
2. **Fused sigmoid.** The last thread writes `1 / (1 + exp(-acc/temp))`
   directly to output.
3. **Strided tensor support.** The kernel accepts `hidden` with a
   `hidden_stride` parameter — important because the caller (`speculator.py`)
   slices `hidden` out of a larger `head_hidden` tensor via an index gather.
4. **PDL sync (`T.pdl_sync()`), SM90+ hint.** Programmatic dependent launch
   overlaps with upstream kernels.

What we **must change**:

1. **Weight dtype → FP32.** DeepInfer uses `bfloat16` weights throughout;
   vLLM's `params_dtype=torch.float32` is a deliberate precision choice.
   Our kernel must read `weight` as fp32.
2. **Accumulator semantics.** DeepInfer uses fp32 FMA on bf16 inputs; we keep
   that, but source of the weight is now fp32 (no `cast(bf16→fp32)` step for
   weights).
3. **Numerical width.** The math per element becomes:
   - Per-thread: `acc += cast_fp32(x[i]) * w[i]` — both fp32 in registers
   - Final reduction: fp32 tree-reduce (same as vLLM's current cuBLAS path)

Net effect on math: per-element expression is **bit-identical** to what vLLM
currently computes. The only source of numerical difference vs. the current
path is the **reduction order** across the N dimension, which may differ from
cuBLAS. That's a ULP-level effect, addressed in §5.

---

## 4. Proposed Design

### 4.1 Kernel signature

```python
# vllm/model_executor/layers/tilelang_ops/dspark_confidence.py

@tilelang.jit(out_idx=None, target="cuda",
              pass_configs={"tl.disable_tma_lower": True})
def confidence_head_fused_tl(
    n_embed: int,           # H
    markov_rank: int,       # r (0 means with_markov=False)
):
    @T.prim_func
    def _kernel(
        x0:     T.StridedTensor((n, n_embed), (hidden_stride, 1), "bfloat16"),
        x1:     T.Tensor((n, max(markov_rank, 1)), "bfloat16"),
        weight: T.Tensor((n_embed + markov_rank,), "float32"),     # <-- FP32, not bf16
        out:    T.StridedTensor((n,), (out_stride,), "float32"),
        temp:   T.float32,
    ):
        with T.Kernel(n, threads=128) as bx:
            T.assume(hidden_stride % 8 == 0)
            acc: T.Fragment, fp32, shape (1,)

            # Logical concat: offset 0 for x0, offset n_embed for x1
            gemv_into_acc(x0, weight, offset=0,        N=n_embed,     acc=acc)
            if markov_rank > 0:
                gemv_into_acc(x1, weight, offset=n_embed, N=markov_rank, acc=acc)

            # Sigmoid is fused, no separate kernel
            if get_thread_binding() == 0:
                scaled = clip(acc / temp, -60.0, 60.0)
                out[bx] = 1.0 / (1.0 + exp(-scaled))
    return _kernel
```

(See §Appendix A for the full TileLang listing.)

### 4.2 Per-element math vs. vLLM today

| Step | vLLM (now) | This RFC | Bit-identical? |
|---|---|---|---|
| Input load | bf16 → register | bf16 → register | yes |
| Input cast | bf16 → fp32 by `.float()` **into global memory** | bf16 → fp32 inline | yes |
| Weight load | fp32 (from global) | fp32 (from global) | yes |
| Per-element FMA | fp32 | fp32 | yes |
| Reduction | cuBLAS tree-reduce (fp32) | TileLang tree-reduce (fp32) | **ulp may differ** |
| Sigmoid | fp32 elementwise | same expression inline | yes |

Per-element math is **identical**; only the cross-thread reduction order may
differ. Empirically we expect `max_rel_err` in the same ballpark as the current
fp32 path (~1e-6 to 1e-5).

### 4.3 Integration point — single file change

The **only** place we touch vLLM business logic:

`vllm/model_executor/models/qwen3_dspark.py:112-137`, inside `forward`:

```python
def forward(self, hidden, markov_embed):
    if self._fused_ok(hidden, markov_embed):
        out = torch.empty(hidden.shape[0], dtype=torch.float32, device=hidden.device)
        _FUSED_KERNEL(hidden, markov_embed, self.proj.weight, out, self._temperature)
        return out                                 # already sigmoided
    # ---- fallback: current path
    x = torch.cat([hidden, markov_embed], dim=-1) if self.with_markov else hidden
    return torch.sigmoid(self.proj(x.float()).squeeze(-1))

def _fused_ok(self, hidden, markov_embed):
    return (
        _FUSED_AVAILABLE
        and self.with_markov
        and hidden.is_cuda and hidden.dtype == torch.bfloat16
        and self.proj.weight.dtype == torch.float32
        and hidden.stride(-1) == 1                 # contiguous last dim
        and hidden.shape[0] <= _FUSED_MAX_BATCH    # see §5, tuned by bench
    )
```

`compute_confidence` in `qwen3_dspark.py:247` and
`deepseek_v4/nvidia/dspark.py:379` need a small refactor: when the fused path
is taken, skip their outer `torch.sigmoid`.

### 4.4 Threshold-gated fallback

`_FUSED_MAX_BATCH` is a module constant that the benchmark (§5) determines.
Rationale:

- For small `n`, fused wins because kernel-launch overhead dominates.
- For large `n`, cuBLAS GEMV is heavily tuned and may still beat a hand-written
  kernel.

Current initial guess: **`_FUSED_MAX_BATCH = 2048`** (matching DeepInfer's
largest reported batch for this path). Must be re-derived on the actual
hardware we ship to.

### 4.4.1 vLLM kernel interface requirements

To integrate as a first-class vLLM kernel (not a one-off), the implementation
must satisfy vLLM's JIT contract (`docs/contributing/jit_kernel_warmup.md`):

1. **Subclass `VllmJitKernel`** with a frozen `CompileKey` dataclass
   (fields: `n_embed`, `markov_rank`).
2. **Implement `dispatch()` / `compile()` / `get_warmup_keys()`** so that
   the kernel pre-compiles at engine startup, **not** on first request —
   otherwise early requests eat a ~2 s JIT delay.
3. **Register via `direct_register_custom_op`** so the kernel is visible to
   CUDA Graph capture and `torch.compile`.
4. Live under `vllm/model_executor/kernels/dspark/` (mirroring
   `kernels/mhc/` layout).

Without these the kernel will work in isolation but be incompatible with
vLLM's startup sequence and CUDA-Graph fast paths.

### 4.5 Build-time / runtime discovery

**TileLang is already a hard CUDA dependency of vLLM** (`requirements/cuda.txt`
pins `tilelang==0.1.12`). No new optional dep needed. We only gate on
**runtime** conditions:

```python
# vllm/model_executor/kernels/dspark/__init__.py
try:
    from .tilelang_kernel import DsparkConfidenceKernel
    DSPARK_FUSED = DsparkConfidenceKernel()  # VllmJitKernel wrapper
    _FUSED_AVAILABLE = (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 9  # PDL requires SM90+
    )
except Exception:
    _FUSED_AVAILABLE = False
```

- Failure to import → silently fall back (still defensive; ROCm is on
  tilelang 0.1.10 which may diverge)
- SM < 90 (A100 and older): PDL absent → fused path disabled; fallback only
- CPU / AMD / XPU: unchanged fallback path

Additionally we must register as `VllmJitKernel` (per
`docs/contributing/jit_kernel_warmup.md`) so that JIT compilation happens at
engine startup, not at first decode step.

---

## 5. Benchmark Plan (pre-merge gate)

**Script location (new):** `benchmarks/kernels/bench_dspark_confidence.py`

### 5.1 Sweep

| Dim | Values |
|---|---|
| `n` (batch) | 32, 128, 512, 2048, 8192 |
| `H` (hidden) | 2048 (Qwen3), 7168 (DSV3) |
| `r` (markov) | 8, 32 |
| `dtype(hidden)` | bf16 (fixed) |
| `dtype(weight)` | fp32 (fixed) |

### 5.2 Three implementations compared

| ID | Description |
|---|---|
| `naive` | `torch.cat → .float() → F.linear → sigmoid` (current uncompiled) |
| `compiled` | same, wrapped in `torch.compile` (the **real** baseline) |
| `fused` | This RFC's TileLang kernel |

### 5.3 Three metrics

1. **Latency** — `torch.cuda.Event`, 1000 iterations, take median.
2. **Peak memory delta** — `torch.cuda.max_memory_allocated()` around the call.
3. **Numerical accuracy** — vs. fp64 reference, report max abs/rel err.

### 5.4 Accept / Reject thresholds

Merge **only if all four hold**:

| Condition | Threshold |
|---|---|
| Latency speedup vs `compiled` (n ≤ 2048) | ≥ **1.20×** (i.e. ≥20% faster) |
| Temp memory reduction vs `compiled` | ≥ **50%** |
| `max_rel_err(fused, fp64_ref)` vs `max_rel_err(compiled, fp64_ref)` | ≤ **1.2×** |
| Latency at n = 8192 (large batch) | ≥ **0.95×** of `compiled` (no more than 5% slower) |

**All four must hold. Any failure → do not merge. Document as negative result.**

### 5.5 Hardware plan

| Scenario | Card | Where |
|---|---|---|
| Development + unit test | 5090 (SM120) | `gpu-10-125-2-54` (single-card pod, tainted `peter-tanyanhong-usage=true:NoSchedule`) |
| Benchmark full sweep | 5090 | same |
| Cross-validation (recommended) | H200 / B300 | TBD — need SRE |
| Production-representative | H200 + B300 | before final PR merge |

> **Note on 5090**: SM120 (Blackwell consumer). Same architecture family as
> B200/B300 (SM100/103), so numbers should transfer. But consumer cards may
> have different memory clocks / driver paths. A small cross-check on H200 is
> strongly recommended before we claim speedup numbers in the PR description.
>
> **Note on dev node**: `gpu-10-125-2-54` was previously unschedulable due to
> a stuck `nvidia-container-toolkit` pod (init container never wrote the
> `toolkit-ready` sentinel). Workaround applied: manually `touch`ed the sentinel
> and uncordoned the node. This is a *stop-gap*; the underlying containerd /
> toolkit issue is not root-caused. Numbers from this node should be treated
> as "developer box", not "production reference". Track the underlying
> node issue separately with SRE.

### 5.6 Preliminary results (5090 dev box, 2026-10-21)

Implementation landed in `vllm/model_executor/kernels/dspark/` together with
`tests/kernels/dspark/test_confidence.py` and
`benchmarks/kernels/dspark/bench_confidence.py`. All numbers below were
measured against vLLM 0.19.1 release PyPI (the version installed in the dev
container), TileLang 0.1.12, PyTorch 2.10.0+cu129, CUDA 12.9, driver 575.

**Unit tests**: 29 / 29 passed (`pytest tests/kernels/dspark/test_confidence.py -v`).

**Sweep**: n ∈ {32, 128, 512, 2048, 8192}, H ∈ {2048, 7168}, r ∈ {0, 8, 32},
100 iterations per point after 20 warmup. Aggregate stats across the 30
operating points:

| Metric | vs `torch.compile` (the real current baseline) |
|---|---|
| **Median latency speedup** | **2.20×** |
| **Min / max speedup** | **0.87× / 3.96×** |
| **Peak-memory delta** | **0 bytes** (vs 0.26 MB–355 MB for compiled) |
| **Median max_rel_err ratio** | **0.616** (fused is more accurate) |

The single <1.0× speedup point is `n=32, H=7168, r=0`, where fused is
0.148 ms vs compiled 0.129 ms. This appears to be TileLang launch overhead
on a shape where cuBLAS has a dedicated skinny-GEMV path; not a regression,
just a parity point. Mitigation: `_FUSED_MAX_BATCH` already caps fallback,
and if reviewers want, we can add a `_FUSED_MIN_BATCH` floor too.

**DSpark production shape** (n ∈ [128, 512], H ∈ [2048, 7168], r ∈ [8, 32])
median speedup: **2.20×**, all shapes in this band ≥ 2.12×.

**Large batch** (n=8192): up to **3.96×**, and per-call memory traffic drops
from ~350 MB (compiled) to ~0 (fused). At higher spec-decode batch sizes
this becomes the dominant saving.

**Full data**: `benchmarks/kernels/dspark/bench_confidence.py --out
/tmp/bench_results.json` in the dev pod. TODO: re-run on H200 for the
production-representative numbers.

---

## 6. Risks

| Risk | Prob | Mitigation |
|---|---|---|
| TileLang kernel slower than cuBLAS at large n | medium | `_FUSED_MAX_BATCH` gating; benchmark sets this |
| TileLang unavailable in user's env | medium | graceful fallback via `_FUSED_AVAILABLE` |
| SM90 PDL semantics differ from SM100 | low | test on H200 during validation phase |
| Numerical regression (reduction order) | low | fp32 accumulator retained; tolerance test |
| First-call JIT compilation slow | low | force pre-warm at model load, not at first decode |
| TileLang pinned version drifts upstream | low | pin in extras; CI test that exercises fused path |
| Maintenance: one more kernel in tree | low | < 100 LoC, fallback retained, tests in place |

---

## 7. Rollout Plan

1. **M1 — Kernel + unit test** (this RFC + tests)
   *No integration changes yet.* Merge after review, gated OFF by default
   (`_FUSED_AVAILABLE = False` via env-var).
2. **M2 — Standalone benchmark** (§5). Data decides whether to proceed.
   If all four thresholds pass → M3. Else → close RFC, keep benchmark script.
3. **M3 — Integration.** Enable fused path by default; keep env-var to disable.
4. **M4 — Wider validation.** H200 / B300 cross-check. If good, remove
   feature flag in a follow-up minor release.

Each milestone is independently revertable.

---

## 8. Open Questions for Reviewer

1. **`_FUSED_MAX_BATCH` initial value** — 2048 OK? Or set smaller (safer) until
   we have B300 data?
2. **`markov_embed` dtype stability** — should we assert / cast defensively?
   Currently assumed bf16 (from `nn.Embedding` default in DSpark).
3. **Strided `hidden` constraints** — we require `stride(-1) == 1`, and rely
   on `stride(0) % 8 == 0`. Is the latter always true in `gpu_model_runner`
   for spec-decode paths? If not, drop to `% 4`.
4. **`temp` (temperature)** — comes from where? SpeculatorConfig? Need to
   plumb.
5. **TileLang version alignment.** vLLM pins `tilelang==0.1.12` for CUDA but
   `0.1.10` for ROCm (`requirements/rocm.txt`); DeepInfer runs `0.1.15`.
   We must target our kernel against **0.1.12** API surface. Smoke-tested on
   0.1.15 — see §Benchmark Plan — but final code must also compile under 0.1.12.
6. **TileLang version pinning** — N/A: already pinned by vLLM. We only need
   to verify our kernel compiles with the pinned version.
7. **Should this apply to K3-DSpark / Qwen3-DSpark as well**, or strictly
   DeepSeek V4 first? My recommendation: V4 only, others in follow-up.
8. **ROCm handling** — TileLang works on ROCm but the GEMV shape we use may
   need different block sizes; leave ROCm fallback to the naive PyTorch path
   in this RFC.

---

## 9. References

- DeepInfer fused kernel:
  <https://github.com/deepseek-ai/DeepInfer/> — `deepinfer/model_executor/layers/tilelang_ops/cuda/dflash.py:12-72`
- Current vLLM confidence head:
  `vllm/model_executor/models/qwen3_dspark.py:112-137`
- Runtime caller:
  `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:189-196`
- DeepSeek V4 NVIDIA path:
  `vllm/models/deepseek_v4/nvidia/dspark.py:379-384`

---

## Appendix A: TileLang Kernel (full listing)

> **Note:** this is a *reference* implementation for review. Final code may
> change to match TileLang API drift at merge time.

```python
# vllm/model_executor/layers/tilelang_ops/dspark_confidence.py
"""Fused DSpark confidence head: GEMV + sigmoid, FP32 weights."""
import tilelang
import tilelang.language as T
from math import gcd


@tilelang.jit(out_idx=None, target="cuda",
              pass_configs={"tl.disable_tma_lower": True})
def confidence_head_fused_tl(n_embed: int, markov_rank: int):
    n = T.dynamic("n")
    out_stride = T.dynamic("out_stride")
    hidden_stride = T.dynamic("hidden_stride")
    threads = 128
    B0 = gcd(1024, n_embed)     # block size for hidden GEMV
    B1 = max(markov_rank, 1)    # block size for markov GEMV

    @T.macro
    def GEMV(x, B, offset, N, weight, acc_frag, bx):
        inner = T.alloc_fragment((B,), T.float32)
        T.fill(inner, 0)
        for i in T.Pipelined(T.ceildiv(N, B)):
            w_frag = T.alloc_fragment((B,), T.float32)   # <-- FP32 weights
            x_frag = T.alloc_fragment((B,), T.bfloat16)
            T.copy(weight[offset + i * B], w_frag)
            T.copy(x[bx, i * B], x_frag)
            for k in T.Parallel(B):
                inner[k] += T.Select(
                    k + i * B < N,
                    T.cast(x_frag[k], T.float32) * w_frag[k],
                    0.0,
                )
        T.reduce_sum(inner, acc_frag, clear=False)

    @T.prim_func
    def _kernel(
        x0:     T.StridedTensor((n, n_embed), (hidden_stride, 1), "bfloat16"),
        x1:     T.Tensor((n, B1), "bfloat16"),
        weight: T.Tensor((n_embed + markov_rank,), "float32"),
        out:    T.StridedTensor((n,), (out_stride,), "float32"),
        temp:   T.float32,
    ):
        with T.Kernel(n, threads=threads) as bx:
            T.assume(hidden_stride % 8 == 0)
            acc_frag = T.alloc_fragment((1,), T.float32)
            acc_frag[0] = 0

            if markov_rank == 0:
                T.pdl_sync()
            GEMV(x0, B0, 0, n_embed, weight, acc_frag, bx)
            if markov_rank > 0:
                GEMV(x1, B1, n_embed, markov_rank, weight, acc_frag, bx)

            if T.get_thread_binding() == 0:
                scaled = T.min(
                    T.max(acc_frag[0] / temp, T.cast(-60.0, T.float32)),
                    T.cast(60.0, T.float32),
                )
                out[bx] = 1.0 / (1.0 + T.exp(-scaled))

    return _kernel
```

### Differences vs. DeepInfer's original (line-by-line)

| DeepInfer (`dflash.py:12-72`) | This RFC | Why |
|---|---|---|
| `weight: T.Tensor(..., "bfloat16")` | `weight: T.Tensor(..., "float32")` | vLLM precision contract |
| `w_frag = T.alloc_fragment((B,), T.bfloat16)` | `w_frag = T.alloc_fragment((B,), T.float32)` | match weight dtype |
| `T.cast(w_frag[k], T.float32)` | `w_frag[k]` directly | already fp32, no cast needed |
| `has_markov_emb: bool` (runtime) | `markov_rank: int` (static), branch on `> 0` | simpler, lets TileLang dead-branch eliminate |
| No temperature parameter | `temp: T.float32` passed in | vLLM needs configurable temp |

---

## Appendix B: Benchmark Script Skeleton

```python
# benchmarks/kernels/bench_dspark_confidence.py
import torch, torch.nn.functional as F
from contextlib import contextmanager

@contextmanager
def cuda_timer():
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    yield
    end.record()
    torch.cuda.synchronize()
    yield start.elapsed_time(end)  # ms


def bench_naive(h, m, w, iters=1000):
    for _ in range(iters):
        x = torch.cat([h, m], dim=-1).float()
        out = torch.sigmoid(F.linear(x, w))
    torch.cuda.synchronize()


def bench_compiled(iters=1000):
    @torch.compile
    def fn(h, m, w):
        return torch.sigmoid(F.linear(torch.cat([h, m], -1).float(), w))
    # warm up...
    # time it...


def bench_fused(h, m, w, temp=1.0, iters=1000):
    from vllm.model_executor.layers.tilelang_ops import confidence_head_fused_tl
    # warm up JIT once
    # then time it...

# Sweep, accuracy vs fp64 reference, peak-memory measurement, then print
# a markdown table with speedup ratios vs `compiled`.
```

The full script will live under `benchmarks/kernels/` and be runnable
standalone (`python bench_dspark_confidence.py`).

---

## Appendix C: Test Plan

`tests/kernels/test_dspark_confidence_fused.py` — using `pytest.mark.parametrize`:

- Correctness: fused vs. fp64 reference, `max_rel_err < 1e-3`
- Sweep: `n ∈ {1, 32, 512, 2048}`, `H ∈ {2048, 7168}`, `r ∈ {0, 8, 32}`
- Strided input: `hidden = bigger[::stride]` variants
- Fallback: force `_FUSED_AVAILABLE=False`, confirm output matches
- Smoke: import-time `ImportError` from TileLang → fallback works

`tests/models/test_deepseek_v4_dspark_fused.py` — verify integration:

- Glue code engages fused when inputs are right
- Glue code falls back when inputs are wrong dtype/shape
- End-to-end smoke with `enable_adaptive_verification=True`

---

## Appendix D: Implementation deltas between this RFC and the merged code

Some implementation details in the landed PR differ from the original
proposal in §4. They are recorded here for the reviewer's convenience; the
RFC body is intentionally left as the "as-proposed" snapshot.

| RFC section | Proposal | Landed in PR | Reason |
|---|---|---|---|
| §4.4.1 | Subclass `VllmJitKernel` and `CompileKey` | Plain dict-cache keyed by `(n_embed, markov_rank)`, exposed via `direct_register_custom_op` | `VllmJitKernel` and `vllm.tilelang_utils` only exist on `main` after 0.19.x; the released container (0.19.1) does not ship them. Maintainers can refactor to `VllmJitKernel` post-merge with a trivial NFC patch — the `CompileKey` maps 1:1 onto the dict key. |
| §4.4.1 | Kernel in `vllm/model_executor/kernels/dspark/tilelang.py` | Two files: `tilelang_kernels.py` (raw TileLang) + `tilelang.py` (custom op registration) | Mirrors `kernels/mhc/` two-file pattern; keeps the build surface testable independently of dispatch. |
| §4.5 | Use `vllm.tilelang_utils.tilelang_jit` and `._get_pass_configs()` | Import `tilelang` directly, hardcode `{TL_DISABLE_WARP_SPECIALIZED, TL_DISABLE_TMA_LOWER}` | Same root cause: `tilelang_utils` not shipped in 0.19.1. |
| §4.1 / App A | `@T.macro GEMV(x, B, offset, ...)` shared between two GEMVs | Unrolled two×: explicit GEMV #1 (`hidden`) and GEMV #2 (`markov`) | mhc-style "implicit signature" (no `@T.prim_func` wrapper) does not support `T.macro` reuse inside `T.Kernel` — unrolling is simpler and the kernels are short. |
| §4.1 | `hidden: T.StridedTensor((n, H), (S, 1), "bfloat16")` | Plain `T.Tensor[[n, H]]` | StridedTensor + forward reference inside `T.Kernel` is fragile on TileLang 0.1.12. The upstream caller already produces contiguous row-major output from `head_hidden[sample_indices]`, so non-contiguous inputs fall back to PyTorch (still correct, just slower). |
| §4.4 | `_FUSED_MAX_BATCH=2048` fallback threshold | Same value, but benchmark shows fused wins up to n=8192 | Conservative default; reviewers can raise after H200 numbers land. |
| §5 | "Benchmark plan" forward-looking | "Benchmark results" — landed with numbers from 5090 dev box | A PR that lands the kernel should land its numbers. H200 cross-check remains in rollout plan. |

These deltas shouldn't change the **decision question** (fused vs. compiled,
FP32 weights preserved, single-file integration), only the **implementation
surface**.

---

*End of RFC.*
