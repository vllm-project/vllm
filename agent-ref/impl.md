
Below is a **developer‑facing reference** to live next to your PRD. Think of it as the “how‑to implement” notebook: concrete file touchpoints, scaffolding code, math you’ll rely on (with full equations), knobs to expose, and profiling hooks. It’s written so you can **prototype on macOS (MLX / CPU)** and then port to **vLLM (CUDA/ROCm/XPU)** with minimal re‑thinking.

> Legend:
>
> * **EPS** = EigenPage Summaries (page‑group sketches + decode‑time gating)
> * **Group** = a contiguous set of PagedAttention **blocks** treated as a unit for summarization (e.g., 8 blocks per group)
> * **K**/**V** = key/value tensors in the KV cache
> * **d** = per‑head dimensionality; **m** = sketch dimension (JL)
> * **JL** = Johnson–Lindenstrauss random projection; **FD** = Frequent‑Directions sketch
> * **Strict mode** = safety mode that guarantees no quality regressions (see thresholds)

---

## 0) Where to hook in vLLM (map of the land)

**PagedAttention & KV cache**

* vLLM partitions each sequence’s KV cache into **fixed‑size blocks** (often called “pages”) and fetches those blocks during attention via a block table, enabling near‑optimal memory utilization and sharing. This paging abstraction is your seam for **group‑level gating** before the attention kernel runs. ([vLLM Blog][1])

**Relevant modules/APIs (v1 engine)**

* `vllm/v1/core/kv_cache_manager.py` – allocates/manages KV blocks, exposes block/group views used by the scheduler and attention path. (Doc page shows public API surfacing page/block counts and prefix sharing helpers.) Your gating step should emit a **filtered list of block IDs** per (layer, head) to hand to the attention launcher. ([VLLM Documentation][2])
* `vllm/core/block_manager.*` & `vllm/core/block/*` (older paths still referenced in API docs) – block allocators, utilities, and prefix caching logic; useful to understand grouping/alignment of blocks. Your summaries should align with these **block boundaries**. ([VLLM Documentation][3])
* **Engine args** / **OpenAI server** – you’ll surface EPS flags here (parity for offline/online use). ([VLLM Documentation][4])

**CPU builds for correctness**

* vLLM supports a **CPU backend** (AVX‑512 recommended); build with `VLLM_TARGET_DEVICE=cpu` to run unit/integration tests for wiring without a GPU. ([vLLM][5])

> Note: vLLM v1 is the only engine now (v0 removed); expect code paths under `vllm/v1/*` for scheduler/KV. ([GitHub][6])

---

## 1) Data structures you will add

```python
# eps/types.py
from dataclasses import dataclass
import torch

@dataclass
class EpsConfig:
    method: str        # "jl" or "1pc"
    m: int             # JL sketch dim (e.g., 8)
    alpha: float       # safety factor (>= 1.0)
    last_n: int        # always visit last N groups
    top_pages: int|None  # optional cap per head
    group_blocks: int  # g: blocks per group
    head_scope: str    # "retrieval" or "all"
    seed: int          # for Phi init

@dataclass
class EpsState:
    # Shapes are [L, H, G, m, m] for JL; or [L, H, G, ...] for 1PC.
    G: torch.Tensor         # JL: Gram sketches (fp32)
    frob2: torch.Tensor     # optional ||K_p||_F^2 per group (fp32)
    max_row: torch.Tensor   # optional max ||k_i|| per group (fp32)
    U1_q8: torch.Tensor|None  # 1PC: packed int8 eigenvector per group
    lmbda1: torch.Tensor|None # 1PC: top eigenvalue per group (fp32)
    resid2: torch.Tensor|None # 1PC: residual energy per group (fp32)
    Phi: torch.Tensor       # [H, d, m], per-head projection (Rademacher ±1)
```

Allocation rules (on device that owns K):

* **JL (primary)**: allocate `G` as `[L,H,G,m,m]` fp32; initialize to zeros.
* **1PC (optional, retrieval heads only)**: allocate `U1_q8` `[L,H,G,d]` int8 plus per‑group scales, `lmbda1`, `resid2`.

> **Groups**: compute `G = ceil(context_len / (group_blocks * block_size_tokens))` per layer. Keep an indexer to map a token’s logical block → **group id**. Align with the block manager’s block size; expose `--eps-group-blocks`. ([vLLM Blog][1])

---

## 2) Math you will implement (full equations)

### 2.1 Safe bound using energy in a page‑group

Let page‑group (p) contain keys (k_i\in\mathbb{R}^d). Define the *page covariance*
[
C_p ;=;\sum_{i\in p} k_i k_i^\top ;\in; \mathbb{R}^{d\times d}.
]

For a query (q), the **energy** of that page is
[
E_p(q) ;=; \sum_{i\in p} (q^\top k_i)^2 ;=; q^\top C_p, q.
]
By Cauchy–Schwarz on vectors (a_i = q^\top k_i),
[
\max_{i\in p} |q^\top k_i|
;\le; \big(\sum_i (q^\top k_i)^2\big)^{1/2}
;=; \sqrt{q^\top C_p q}.
]
Thus a **conservative upper bound** on *any* single‑token dot product in page (p) is
[
B_p(q) ;\equiv; \sqrt{,q^\top C_p, q, }.
]
We will safely approximate (B_p) via sketches (JL or 1PC) and compare it to a **running threshold** (T) (the current M‑th best seen so far), skipping page (p) if (B_p < T/\alpha) with (\alpha\ge 1). (Strict mode uses larger (\alpha) and guardrails below.)

### 2.2 JL sketch (primary method)

Pick a per‑head random Rademacher projection (\Phi\in{\pm 1}^{d\times m}). Maintain
[
G_p ;=; \sum_{i\in p} (\Phi^\top k_i)(\Phi^\top k_i)^\top ;\in; \mathbb{R}^{m\times m}.
]
For query (q), compute (z=\Phi^\top q) and approximate
[
\boxed{ \quad \hat{B}_p(q) ;=; \sqrt{,z^\top G_p z,} ;\approx; \sqrt{,q^\top C_p q,}. \quad }
]
JL guarantees (norm/inner‑product preservation) imply you can make the relative error small with modest (m), with probability (1-\delta) across a finite set of queries. Use (\alpha \ge 1) to budget worst‑case error. ([Wikipedia][7])

**Cost per token** for JL update: (O(dm + m^2)) FLOPs (e.g., with (d=128,m=8): (1088) FLOPs). See budgets in §7.

### 2.3 1‑PC (tighten where needed)

Let (C_p = U\Lambda U^\top). Keep top eigenpair ((u_1,\lambda_1)) and residual energy (r^2 = \operatorname{tr}(C_p) - \lambda_1). Then
[
q^\top C_p q ;=; \lambda_1 (u_1^\top q)^2 ;+; \sum_{i\ge 2}\lambda_i (u_i^\top q)^2
;\le; \lambda_1 (u_1^\top q)^2 ;+; r^2 |q|_2^2,
]
so
[
\boxed{\quad B^{(1)}_p(q) ;\le; \sqrt{\lambda_1},|u_1^\top q| ;+; \sqrt{r^2},|q|_2.\quad}
]
Store (u_1) as **int8 + per‑vector scale** for space efficiency; keep (\lambda_1,r^2) as fp32.

### 2.4 FD (adjacent option)

If you prefer a deterministic streaming sketch, FD maintains (B\in\mathbb{R}^{\ell\times d}) such that (|A^\top A - B^\top B|_2) is bounded; you can compute a JL‑like bound in the sketch space or recover PCs. Complexity (O(d\ell)) per row with strong error bounds. (Good safety alternative if JL makes you uneasy.) ([Chbrown][8])

### 2.5 Thresholding and safety

Let (M) be the number of page‑groups you allow the kernel to visit (per head). Visit **last‑N recent** groups unconditionally, then iterate groups **newest→oldest**, maintaining a heap of size (M) with current best contributions (from actually visited pages). Skip group (p) if
[
\hat{B}_p(q) ;<; T/\alpha,
]
where (T) is the running M‑th best and (\alpha\ge 1) (e.g., 1.05–1.10).
**Strict mode** = larger (\alpha) + always‑visit last‑N + optional scalar fallback bound using (|K_p|_F) to further reduce false skips.

---

## 3) End‑to‑end control flow (vLLM)

1. **K write path (prefill/early decode)**: when a token’s **K** lands in its block, compute the sketch update *without re‑reading K*.

   * JL: `y = Phi.T @ k` (GEMV), then `G[group] += y y^T`.
   * Track `frob2[group] += k.norm()**2` and `max_row[group] = max(max_row[group], k.norm())` (optional fallback).
2. **Decode pre‑pass (per step)**: for each (layer, head), compute `z = Phi.T @ q`, then for groups in **newest→oldest** order, compute `B_p`; build a **visit list** using (T), (\alpha), `last_n`, and `top_pages`.
3. **PagedAttention call**: pass the filtered **group/block indices** so the kernel only fetches KV from **visited** groups. (This mirrors the existing block‑table indirection.) ([vLLM Blog][1])

---

## 4) Implementation scaffolding (drop‑in code you can edit)

> These snippets are **ours** (not copied from vLLM) and are designed to fit into the v1 codebase idioms. File names are suggestions; adapt to current tree.

### 4.1 Summary updates on K write

```python
# eps/summarizer.py
import torch

@torch.no_grad()
def jl_update_G(
    G: torch.Tensor,           # [L,H,G,m,m], fp32
    Phi_h: torch.Tensor,       # [d,m] for this head
    k: torch.Tensor,           # [d]
    l: int, h: int, g: int):   # layer, head, group ids
    # y = Phi^T k  -> [m]
    y = torch.matmul(Phi_h.T, k)                       # GEMV
    G[l, h, g].add_(torch.outer(y, y))                 # rank-1 update

def update_scalars(frob2, max_row, k, l, h, g):
    v = torch.dot(k, k)
    frob2[l,h,g] += v
    max_row[l,h,g] = torch.maximum(max_row[l,h,g], torch.sqrt(v))
```

**Where to call**: immediately after the engine writes K into a block and before it discards local registers/L2. You’ll insert a small call at the point where **block id** and **(layer, head)** are known and map that block id to a **group id** via your `group_blocks` aggregation.

### 4.2 Pre‑pass selector (build visit lists)

```python
# eps/selector.py
import torch
from .types import EpsConfig, EpsState

def build_visit_list_jl(
    cfg: EpsConfig,
    st: EpsState,
    q_head: torch.Tensor,       # [d] for a given (l,h) at this step
    groups_by_recency: list[int],
    l: int, h: int) -> list[int]:
    z = torch.matmul(st.Phi[h].T, q_head)              # [m]
    visit = set(groups_by_recency[:cfg.last_n])        # always visit last-N
    T = float("-inf")
    heap = []   # store best contributions; optionally size-capped to cfg.top_pages

    # Score and select newest->oldest
    for g in groups_by_recency:
        if g in visit:                                # unconditional
            score = torch.sqrt(torch.matmul(z, torch.matmul(st.G[l,h,g], z)))
            T = max(T, score.item())
            continue
        # quadratic form (cheap): z^T G z
        score = torch.sqrt(torch.matmul(z, torch.matmul(st.G[l,h,g], z)))
        if score >= T / cfg.alpha:
            visit.add(g)
            T = max(T, score.item())
            # push to heap if you want a top_pages cap

    # Optional cap: keep only the top M groups by 'score'
    if cfg.top_pages is not None and len(visit) > cfg.top_pages:
        # re-score visited groups and keep top M
        scored = []
        for g in visit:
            v = torch.sqrt(torch.matmul(z, torch.matmul(st.G[l,h,g], z))).item()
            scored.append((v, g))
        visit = set([g for _, g in sorted(scored, reverse=True)[:cfg.top_pages]])

    # Return in recency order to help coalescing
    return [g for g in groups_by_recency if g in visit]
```

### 4.3 Plumb indices to PagedAttention

You will **filter** the per‑sequence **block indices** (or a mask) before launching attention:

```python
# eps/integration.py
def filter_kv_indices_by_groups(kv_indices, group_of_block, visit_groups):
    """kv_indices: e.g., [num_blocks] tensor of block ids for one (l,h,sequence).
       group_of_block: callable/block->group lookup
       visit_groups: set/list of group ids to keep."""
    keep_mask = torch.tensor([group_of_block(b) in visit_groups for b in kv_indices],
                             device=kv_indices.device, dtype=torch.bool)
    return kv_indices[keep_mask]
```

**Where to call**: just before building the final `kv_indices` (or corresponding mask tensor) consumed by the attention kernel. This is typically inside the **attention launch** path after the scheduler finalizes batch composition for this step. (Exact function names can move between releases; search in `vllm/v1/core/*` for where the attention op receives the per‑sequence block lists.) ([VLLM Documentation][9])

### 4.4 Flags, env, and telemetry

Expose flags in your engine/server arg plumbing:

```python
# eps/flags.py
def add_eps_flags(parser):
    parser.add_argument("--eps-method", choices=["off", "jl", "1pc"], default="off")
    parser.add_argument("--eps-dim", type=int, default=8)
    parser.add_argument("--eps-alpha", type=float, default=1.1)
    parser.add_argument("--eps-last-n", type=int, default=8)
    parser.add_argument("--eps-top-pages", type=int, default=None)
    parser.add_argument("--eps-group-blocks", type=int, default=8)
    parser.add_argument("--eps-heads", choices=["retrieval", "all"], default="retrieval")
```

Add **NVTX** ranges so Nsight Systems shows an “EPS pre‑pass” slice per step:

```python
import torch
def with_nvtx_range(name):
    class _Ctx:
        def __enter__(self): torch.cuda.nvtx.range_push(name)
        def __exit__(self, exc_type, exc_val, exc_tb): torch.cuda.nvtx.range_pop()
    return _Ctx()

with with_nvtx_range("eps_prepass"):
    # compute z, B_p, visit masks
    ...
```

PyTorch exposes these NVTX helpers; Nsight Systems will display them in the timeline. ([PyTorch Documentation][10])

**Environment knobs** (FYI for docs): vLLM has environment variables for v1 engine features; consult the env‑vars docs if you need to gate behavior without adding CLI args. ([VLLM Documentation][11])

---

## 5) Grouping logic (blocks → groups)

PagedAttention stores KV in **fixed‑size blocks** (the docs and blog describe the paging abstraction). You’ll group **g** consecutive blocks into a **group** (a.k.a. *super‑page*) to keep metadata small and alignment clean:

```python
# eps/grouping.py
def compute_group_id(logical_block_id: int, group_blocks: int) -> int:
    return logical_block_id // group_blocks

def build_groups_by_recency(num_blocks: int, group_blocks: int) -> list[int]:
    num_groups = (num_blocks + group_blocks - 1) // group_blocks
    # newest group has the highest id if blocks append at the end
    return list(reversed(range(num_groups)))
```

This keeps your summary tensors bounded and reduces overhead on devices where the **native block size is small**. (PagedAttention’s central idea and block abstraction are explained in the official blog/design docs.) ([vLLM Blog][1])

---

## 6) Head scoping (retrieval vs. all)

Apply EPS **only on “retrieval heads”** initially (lower risk, lower memory). You can adopt DuoAttention’s head characterization: heads that lose accuracy when restricted to recent tokens (vs. streaming heads that don’t). Keep a configurable list per layer/head and limit summaries/pre‑pass to those heads. ([arXiv][12])

---

## 7) Budgets & formulas (so you don’t regress)

Let (L) layers, (H) heads, head dim (d), JL dim (m), context length (L_c), block size (B) tokens, group size (g) blocks ⇒ group span (S=gB) tokens ⇒ groups per layer (G=\lceil L_c/S\rceil).

**Memory (JL):** (L\cdot H\cdot G\cdot m^2\cdot 4) bytes (fp32 Gram matrix).
Example: (L=H=32, L_c=65{,}536, B=16, g=8, S=128, G=512, m=8 \Rightarrow )
( 32\cdot 32\cdot 512\cdot 8^2\cdot 4 \approx 128) MiB. (Add a few MiB for scalars.)

**Compute per token (JL update):** (O(dm + m^2)). With (d=128, m=8): (1088) FLOPs per (layer,head,token) summarized. This is negligible relative to attention FLOPs and **amortized** over large contexts.

**Pre‑pass:** one GEMV (z=\Phi^\top q) per head + (G) **quadratic forms** (z^\top G_p z) (cheap (m\times m) matvecs). Keep (m) small (8–16).

---

## 8) Strict‑mode guardrails (concrete rules)

* **Always visit** the last `--eps-last-n` groups for each head.
* **Safety factor** `--eps-alpha`: default 1.1; strict mode 1.2–1.3 (empirical).
* **Scalar fallback bound** (optional): if `sqrt(frob2[p]) * ||q|| < T/alpha`, skip without touching `G_p`.
* **Top‑pages cap** (`--eps-top-pages`): never visit more than **M** groups per head; apply after selection.

---

## 9) Unit & integration tests (ready‑to‑run snippets)

### 9.1 JL quadratic‑form preservation

```python
def test_jl_quadratic_form(seed=0, d=128, m=8, n=4096, reps=5):
    torch.manual_seed(seed)
    Phi = torch.empty(d, m).bernoulli_().mul_(2).sub_(1)  # ±1
    for _ in range(reps):
        K = torch.randn(n, d) / (d ** 0.5)
        C = K.t() @ K
        G = (K @ Phi).t() @ (K @ Phi)
        for _ in range(32):
            q = torch.randn(d)
            lhs = torch.sqrt(q @ C @ q)
            z  = Phi.t() @ q
            rhs = torch.sqrt(z @ G @ z)
            rel_err = (rhs - lhs).abs() / (lhs + 1e-9)
            assert rel_err.item() < 0.2  # loose unit test; tune in sweeps
```

Use this to calibrate `m` and a safe `alpha`. JL background: see lemma references and performance on quadratic forms. ([Wikipedia][7])

### 9.2 Selector oracle (false‑skip budget)

```python
def exact_page_energy(q, Ks_in_page):   # oracle
    # sqrt(sum_i (q·k_i)^2)
    return torch.sqrt(sum((q @ k)**2 for k in Ks_in_page) + 1e-9)

def test_selector_no_false_skips():
    # build synthetic pages with known "needle" in a distant page
    # ensure strict mode never skips the needle page
    ...
```

### 9.3 Integration (CPU path)

* Build vLLM for CPU, run a decode step where you log `pages_total`, `pages_visited`, `bytes_kv_read_est`, and confirm **filtered indices** go into the attention path without errors. ([vLLM][5])

---

## 10) Profiling & telemetry (NVTX, Nsight)

Add NVTX ranges (Python side: `torch.cuda.nvtx.range_push/pop`) around: (a) EPS pre‑pass, (b) attention launch, (c) kernel. Capture with Nsight Systems (`nsys profile ...`). Keep ranges sparse and descriptive. ([PyTorch Documentation][10])

---

## 11) Adjacent math/concepts you may reach for

* **Frequent‑Directions** (deterministic sketch): streaming (B) with (|A^\top A - B^\top B|_2) control; can recover PCs or use (B) as the sketch domain for quadratic forms. ([Chbrown][8])
* **Matrix determinant lemma** & **D‑optimal design** (future KV‑eviction variant):
  (\det(G + u u^\top) = \det(G)\big(1 + u^\top G^{-1} u\big)). Maintain a **Cholesky** (G=LL^\top) to compute marginal gains and do rank‑1 updates/downdates. (This is for a separate KV eviction policy; not needed for EPS v1.)
* **Spectral residual** as acceptance signal (speculative decoding): build a **Krylov/Lanczos** approximation of the attention kernel to estimate unmodeled mass; outside v1 scope, but useful later.

---

## 12) Interactions with the rest of the stack

* **PagedAttention** is exact attention implemented with block indirection; EPS **only filters which blocks you visit at a step**. Keep it aligned with the block table and masks. ([vLLM Blog][1])
* **KV eviction / offload** (NACL, KeDiff/KeyDiff, InfiniGen, LMCache): EPS composes—eviction changes *what’s stored*, EPS changes *what’s read* per step. If you enable CPU offload (LMCache), expect sizeable latency — use only for functional testing. ([LMCache][13])
* **Head specialization (DuoAttention):** in v1, scope EPS to **retrieval heads** first. ([arXiv][12])

---

## 13) Mac prototype (MLX) pointers (optional but handy)

* Use **MLX** on macOS to implement JL updates and pre‑pass quickly; its Metal kernels make iteration fast while your vLLM CPU path proves wiring. Port the exact same flags, shapes, and metrics to avoid drift. ([VLLM Documentation][14])

---

## 14) Reference equations & inequalities (quick copy/paste)

* **Energy bound**:
  (\max_i |q^\top k_i| \le |K^\top q|_2 = \sqrt{q^\top C q}).

* **JL preservation** (norms/inner products): for (k \ge c\varepsilon^{-2}\log N), with high prob.,
  ((1-\varepsilon)|x|_2^2 \le |\Phi^\top x|_2^2 \le (1+\varepsilon)|x|_2^2). Apply to (x=K^\top q). ([Wikipedia][7])

* **1‑PC bound with residual**:
  (q^\top C q \le \lambda_1 (u_1^\top q)^2 + r^2 |q|_2^2), (r^2=\operatorname{tr}(C)-\lambda_1).

* **FD guarantee** (spectral/Frobenius error): see Liberty (KDD’13), Ghashami et al. (SIAM). ([Chbrown][8])

---

## 15) “Gotchas” (and fixes)

* **Where to get group ids**: always derive from **logical** block order (recency). Don’t rely on physical block ids; those move. (PagedAttention decouples logical vs. physical—by design.) ([vLLM Blog][1])
* **Batch divergence**: if every sequence decides on a different set, coalescing drops. Keep **last‑N** common, limit `top_pages`, and sort visit lists **newest→oldest**.
* **Speculative decoding**: ensure **draft** and **target** apply the **same pruned set**, or disable EPS for draft.
* **Small block sizes**: on devices where block size is very small, **group** multiple blocks (`--eps-group-blocks`) or memory overhead balloons.
* **NVTX noise**: too many tiny ranges make Nsight unreadable. One “eps_prepass” range per step is plenty. ([NVIDIA Developer][15])

---

## 16) Step‑by‑step integration checklist

1. **Add flags** (`--eps-*`) to engine args and OpenAI server. ([VLLM Documentation][16])
2. **Create `eps/` package** with `types.py`, `summarizer.py`, `selector.py`, `grouping.py`, `integration.py`.
3. **Init** `EpsConfig/EpsState` at model load; build `Phi` (per head) with fixed seed; allocate `G` & stats.
4. **Wire K‑write hook**: after each K lands in a block, call `jl_update_G(...)`.
5. **Decode pre‑pass**: per (layer, head), compute `z`, `B_p`, `visit_groups`.
6. **Filter indices**: convert `visit_groups` → block mask or filtered `kv_indices`.
7. **Telemetry**: counters for `pages_total`, `pages_visited`, `bytes_kv_read_est`, `eps_prepass_ms`; add **NVTX** range. ([PyTorch Documentation][10])
8. **CPU integration test**: vLLM CPU build, single step, validate correctness of indices and that attention runs. ([vLLM][5])
9. **Cloud GPU validation**: run long‑context grids; collect **tokens/s**, **p50/p95**, **pages visited ratio**, and **LongBench/RULER** deltas in **strict/default** modes. (Use official server entrypoint.) ([VLLM Documentation][16])

---

## 17) Appendix: formulas for budgets & calibration

* **Memory (JL)**: (M_{\text{JL}} = 4,L H G m^2) bytes.
* **Per‑token update cost**: (C_{\text{update}} = d m + m^2) FLOPs.
* **Pre‑pass cost (per head)**: one (d\times m) GEMV + (G) quadratic forms (two (m)-length dot products each).
* **Choosing (m,\alpha)**: start (m=8), (\alpha=1.1) (default). Increase (m) to reduce **loose bounds**; increase (\alpha) to limit **false skips**. Use synthetic stress tests to set strict‑mode (\alpha) (e.g., 1.2–1.3). JL guidance on (\varepsilon \sim \sqrt{\tfrac{\log N}{m}}). ([Wikipedia][7])

---

## 18) Pointers to upstream docs (for maintainers)

* **PagedAttention (design/blog):** conceptual overview & block abstraction. ([vLLM Blog][1])
* **KV managers / block utils (API docs):** where block allocation & views live. ([VLLM Documentation][9])
* **Engine/server args:** where to put flags and how users consume them. ([VLLM Documentation][4])
* **CPU backend build instructions:** for correctness on non‑GPU machines. ([vLLM][5])
* **NVTX & Nsight Systems:** how to add ranges and collect traces. ([PyTorch Documentation][10])
* **DuoAttention (retrieval heads):** rationale for head scoping. ([arXiv][12])
* **JL & FD references:** math background for your bounds/sketches. ([Wikipedia][7])

---

### Final note

This reference is **implementation‑first**: the code skeletons are meant to drop into a feature branch and evolve into real diffs. Where exact function names differ across vLLM point releases, lean on the **module‑level APIs** linked above and keep EPS **block‑aligned** with clean **pre‑pass → filtered indices → attention** control flow. That keeps the PR small, testable on CPU, and straightforward to validate with Nsight traces and long‑context grids.

[1]: https://blog.vllm.ai/2023/06/20/vllm.html "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention | vLLM Blog"
[2]: https://docs.vllm.ai/en/v0.9.2/api/vllm/v1/core/kv_cache_manager.html?utm_source=chatgpt.com "vllm.v1.core.kv_cache_manager"
[3]: https://docs.vllm.ai/en/v0.10.1/api/vllm/core/block_manager.html?utm_source=chatgpt.com "vllm.core.block_manager"
[4]: https://docs.vllm.ai/en/latest/configuration/engine_args.html?utm_source=chatgpt.com "Engine Arguments - vLLM"
[5]: https://nm-vllm.readthedocs.io/en/latest/getting_started/cpu-installation.html?utm_source=chatgpt.com "Installation with CPU - vLLM - Read the Docs"
[6]: https://github.com/vllm-project/vllm/releases?utm_source=chatgpt.com "Releases · vllm-project/vllm"
[7]: https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma?utm_source=chatgpt.com "Johnson–Lindenstrauss lemma"
[8]: https://chbrown.github.io/kdd-2013-usb/kdd/p581.pdf?utm_source=chatgpt.com "Simple and Deterministic Matrix Sketching - chbrown@github"
[9]: https://docs.vllm.ai/en/v0.10.2/api/vllm/v1/core/index.html?utm_source=chatgpt.com "core - vLLM"
[10]: https://docs.pytorch.org/docs/stable/generated/torch.cuda.nvtx.range_push.html?utm_source=chatgpt.com "torch.cuda.nvtx.range_push"
[11]: https://docs.vllm.ai/en/stable/configuration/env_vars.html?utm_source=chatgpt.com "Environment Variables - vLLM"
[12]: https://arxiv.org/abs/2410.10819?utm_source=chatgpt.com "DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads"
[13]: https://docs.lmcache.ai/getting_started/quickstart/offload_kv_cache.html?utm_source=chatgpt.com "Example: Offload KV cache to CPU"
[14]: https://docs.vllm.ai/en/stable/getting_started/installation/cpu.html?utm_source=chatgpt.com "CPU - vLLM"
[15]: https://developer.nvidia.com/nsight-systems/get-started?utm_source=chatgpt.com "Getting Started with Nsight Systems"
[16]: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html?utm_source=chatgpt.com "OpenAI-Compatible Server - vLLM"
