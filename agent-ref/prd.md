
Below is a complete **Product Requirements Document (PRD)** to implement **EigenPage Summaries (EPS)**—a page‑level decode‑time gating mechanism for **PagedAttention**—with everything you’ll need to build, test, and evaluate it end‑to‑end.

---

# PRD: EigenPage Summaries (EPS) for PagedAttention

**Owner:** You
**Stakeholders:** vLLM maintainers, infra/serving engineers, long‑context users
**Status:** Draft → Implementation behind flag → Cloud validation → Upstream PRs
**Last updated:** Nov 5, 2025

---

## 1) Problem & Opportunity

**Problem.** In long‑context serving, decode is often **memory‑bandwidth bound** because attention kernels repeatedly read many KV cache pages that contribute little to the current token’s logits. PagedAttention fixes *fragmentation* by storing KV in fixed‑size blocks (“pages”), but it still scans many blocks per decode step. Skipping low‑contribution pages reduces HBM reads and tail latency. ([arXiv][1])

**Opportunity.** Attach a tiny **summary per (layer, head, page‑group)** that yields a fast **upper bound** on a page’s best‑possible contribution to the current query (q). If the bound is below a dynamic threshold, **don’t read** that page. Orthogonal to eviction/compression (which change what you *store*), EPS changes what you *read* per step. Related work shows KV management and head specialization are impactful, but there’s space for decode‑time gating at the page level. ([arXiv][2])

---

## 2) Goals & Non‑Goals

### Goals (ship and measure)

1. **Throughput & latency:** Reduce **KV bytes read** and improve **p50/p95** decode latency at long contexts (≥32k) at *fixed quality*.
2. **Safety:** Provide **conservative bounds** + **strict mode** to avoid false skips on “needle” cases.
3. **Adoptability:** Minimal changes to PagedAttention call‑site; clean **flags**, **telemetry**, and **revert path**.
4. **Compatibility:** Works with vLLM **CUDA**/**ROCm**/**XPU**; CPU path for correctness. No assumption of new model training. ([VLLM Docs][3])

### Non‑Goals (for v1)

* Not a replacement for attention; not a global compressor.
* No new kernel for attention itself (only pre‑pass and summaries).
* No Metal/MPS backend for vLLM (prototype on Mac in **MLX**; validate on cloud GPU). ([VLLM Docs][4])

---

## 3) User Stories

* **Serving Eng:** “I enable `--eps` on a 64k context service and see **~10–20% fewer KV reads** and **p95↓** with no measurable quality drop.”
* **Research Eng:** “I can **tune EPS** (JL dim, alpha, always‑visit last‑N pages) and **log** skip ratios/bytes read; strict mode guarantees no regressions on RULER/LongBench.” ([arXiv][5])
* **Maintainer:** “Change is behind a flag, fully instrumented (NVTX markers, counters), with a documented rollback.” ([NVIDIA Docs][6])

---

## 4) Scope & Interfaces

### 4.1 New CLI Flags

* `--eps-method [off|jl|1pc]` (default: `off`)
* `--eps-dim <m>` (JL projection size, default: `8`)
* `--eps-alpha <α>` (safety factor, default: `1.1`)
* `--eps-last-n <N>` (always visit last N page‑groups, default: `8`)
* `--eps-top-pages <M>` (cap on pages visited per head; optional)
* `--eps-heads [retrieval|all]` (default: `retrieval`)
* `--eps-group-blocks <g>` (how many KV **blocks** per **page‑group**; default: `8`)

> **Note.** vLLM’s KV cache is organized in **blocks** (a.k.a. “pages”) of tokens (CUDA commonly 16 tokens; other devices vary). We build EPS summaries over **page‑groups** of `g` contiguous blocks to keep metadata small while aligning with PagedAttention. ([VLLM Docs][7])

### 4.2 Telemetry (counters & NVTX)

* Per step: `pages_total`, `pages_visited`, `pages_skipped`, `bytes_kv_read_est`, `eps_prepass_ms` (NVTX range), `decode_ms`.
* Per run: distributions + head‑wise visit ratios. (NVTX gives timeline slices for pre‑pass). ([NVIDIA GitHub][8])

---

## 5) Design Overview

### 5.1 Page‑Group Summaries (S_p)

We maintain a compact summary (S_p) for each **(layer, head, page‑group p)** as we write **K** into the KV cache (no extra K reads):

**Option A — EPS‑JL (primary)**

* Fix a random sign projection (\Phi\in\mathbb{R}^{d\times m}) per head.
* For each key (k\in\mathbb{R}^d) written to group (p): **project** (y=\Phi^\top k\in\mathbb{R}^m) and **accumulate** (G_p \leftarrow G_p + y y^\top).
* At decode, for a query (q): compute (z=\Phi^\top q) once per head; upper‑bound proxy
  [
  B_p \approx \sqrt{, z^\top G_p, z,},
  ]
  which (by JL) approximates (q^\top C_p q) where (C_p=\sum_{i\in p} k_i k_i^\top). Choose (m) and a safety factor (\alpha) to keep false‑skip probability below budget. ([Computer Science][9])

**Option B — EPS‑1PC (tighten selectively)**

* When a page‑group **seals**, compute top‑1 eigenpair ((u_1,\lambda_1)) of (C_p) (or from a small sketch) and residual energy (r^2=\operatorname{tr}(C_p)-\lambda_1).
* Bound:
  [
  B_p \le \sqrt{\lambda_1},|u_1^\top q| + \sqrt{r^2},|q|.
  ]
* Store (u_1) **int8‑quantized** + two fp32 scalars. (Use it only on **retrieval heads** to cap memory.)

**Why these?** JL & FD sketches are **streaming** and **deterministic** with strong error bounds for covariance approximation; top‑PC is tighter when the group is low‑rank. ([Chbrown][10])

### 5.2 Skip Decision (per step, per head)

Iterate groups **newest → oldest**. Keep a heap (size (M)) of best observed contributions; let (T) be the current (M)-th best. **Skip** group (p) if
[
B_p < T / \alpha,
]
with **strict mode** guardrails:

* Always visit the last `--eps-last-n` groups.
* Optional scalar fallback bound using (|K_p|_F) (cheap to track) to avoid underestimation.

### 5.3 Where it plugs into vLLM

1. **On K write** (prefill or decode): update the summary tensor for the owning **page‑group**; never re‑read K.
2. **Decode pre‑pass**: compute per‑head (z=\Phi^\top q) + group scores (B_p); produce a **filtered list** of group IDs to visit.
3. **PagedAttention call**: provide the list so the kernel only reads those blocks. (This mirrors how mask/indices tensors tell kernels which blocks to access.) ([Jonathan Chang][11])

---

## 6) Data Layout & Budgets

Let:

* head dim (d) (e.g., 128 in LLaMA‑like 7B),
* block size (B) tokens (CUDA commonly 16),
* group size (g) blocks → **superpage** length (S=gB) tokens,
* context length (L_c) tokens, heads (H), layers (L), JL dim (m).

**# groups per layer** (= \lceil L_c / S \rceil). **Total tuples** (= L \times H \times \lceil L_c / S \rceil).

**Example (sanity):** (L_c=65{,}536), (B=16), (g=8\Rightarrow S=128), (H=L=32), (m=8).

* Groups/layer (= 65{,}536 / 128 = 512).
* Tuples (= 32 \times 32 \times 512 = 1{,}024 \times 512 = 524{,}288).
* EPS‑JL memory: store (G_p\in\mathbb{R}^{m\times m}) = (64) fp32 = **256 B** → **524,288 × 256 B = 134,217,728 B = 128 MiB**.
* Per‑token compute (all heads/layers): (d m + m^2 = 128×8 + 8^2 = 1088) FLOPs per tuple → **1,114,112 FLOPs** per new token across 32×32 = 73.0 GFLOPs for a 64k prefill (1,114,112 × 65,536 = 73,014,444,032).
  These fit vLLM memory/computation envelopes when amortized; **grouping** is essential because raw block size on CUDA is small (often 16 tokens). ([LMCache Blog][12])

---

## 7) Dependencies & Environments

* **vLLM backends:** CUDA 12.x wheels (docs mention CUDA 12.8), **ROCm ≥6.3**, **Intel XPU**; **Mac/Metal GPU is not supported** (do CPU builds only on macOS). ([VLLM Docs][4])
* **Mac prototyping:** Use **MLX** (Metal‑native array framework) to prototype the EPS pre‑pass and correctness tests quickly. ([ML Explore][13])
* **Profiling:** Nsight Systems + **NVTX** ranges for pre‑pass & attention stages. ([NVIDIA GitHub][8])

---

## 8) Detailed Algorithm & Math (minimal theory you’ll need)

### 8.1 Johnson–Lindenstrauss (JL) for quadratic forms

Random projection (\Phi\in{\pm 1}^{d\times m}) preserves norms/inner products with high probability; hence (z^\top G_p z \approx q^\top C_p q) when (z=\Phi^\top q), (G_p=\sum (\Phi^\top k)(\Phi^\top k)^\top). Choose (m) to bound relative error (\varepsilon) (e.g., (m=\mathcal{O}(\varepsilon^{-2}\log n)) for (n) vectors). Use (\alpha \ge 1) as a safety multiplier. ([Computer Science][9])

### 8.2 Frequent Directions (FD) (optional)

FD maintains a streaming sketch (B\in\mathbb{R}^{\ell\times d}) s.t. (|A^\top A - B^\top B|_2) is bounded; you can derive (G_p) in the sketch domain or recover PCs. Deterministic, (O(d\ell)) per row. Good replacement if you want stronger guarantees than pure JL. ([arXiv][14])

### 8.3 Top‑k PCs bound

(C_p = U\Lambda U^\top). For retained PCs (U_k,\Lambda_k) and residual (r^2=\operatorname{tr}(C_p)-\sum_{i\le k}\lambda_i),
[
q^\top C_p q \le | \Lambda_k^{1/2} U_k^\top q|_2^2 + r^2|q|_2^2.
]
Use (k=1) selectively (retrieval heads only) to tighten bounds without much memory.

### 8.4 Log‑det / D‑optimal design (future variant)

If you later implement a **KV eviction** policy, the matrix determinant lemma gives the marginal log‑det gain: (\log\det(G+!uu^\top)=\log\det G+\log(1+u^\top G^{-1}u)). Keep/dismiss tokens by marginal gains; update **Cholesky** (G=LL^\top) incrementally (rank‑1 update/downdate). ([Wikipedia][15])

---

## 9) Functional Requirements

1. **Summaries:** Maintain (S_p) for every (layer, head, page‑group).
2. **No additional K reads:** Update summaries **when K is written** (prefill/decode), via a fused or adjacent operation.
3. **Decode pre‑pass:** Before attention, compute (B_p) for candidate groups and build a **visit list** (head‑local).
4. **Gating:** PagedAttention **only visits** pages in the list; others are skipped for that step.
5. **Strict mode safety:** Always visit last‑N groups; `alpha≥1` safety margin; optional scalar fallback bound.
6. **Head scoping:** Allow `--eps-heads=retrieval` (only summarize **retrieval heads**, informed by DuoAttention/analysis). ([arXiv][16])
7. **Instrumentation:** Expose counters + NVTX ranges; produce bytes‑read estimates. ([NVIDIA GitHub][8])

---

## 10) Non‑Functional Requirements

* **Quality:** Δaccuracy ≤ **0.3% abs** on LongBench/RULER under default mode; **0.0%** in strict mode within statistical error. ([arXiv][5])
* **Overhead:** EPS pre‑pass adds **≤5–8%** wall‑time at 32–64k contexts; net speedup must be positive.
* **Memory:** EPS summaries ≤ **150 MiB** for (32 layers × 32 heads × 64k context, `g=8`, `m=8`).
* **Determinism:** Given seeds for (\Phi), the skip set is reproducible.

---

## 11) Architecture & Code Touchpoints

* **Where:** vLLM engine path that (a) writes K into per‑block buffers, (b) assembles **kv_indices** (or equivalent) for PagedAttention, and (c) launches the attention kernel.
* **Add:**

  * `eps/` module with: projection builder, group summarizer, pre‑pass selector, metrics logger.
  * Storage tensors for summaries: `[layer][head][group][m][m]` (fp32) or packed int8 vectors for 1PC.
  * Flags plumbed through engine args and OpenAI‑compatible server (for parity). ([VLLM Docs][17])

**Control flow (decode step):**

1. Build (z=\Phi^\top q) (batched GEMV per head).
2. For groups in **newest→oldest** order: compute (B_p), update heap, apply `alpha`/`last_n` rules.
3. Emit a compact, sorted list of group IDs to visit; **no inner‑kernel branching**.
4. Call PagedAttention with filtered `kv_indices`. (Shapes similar to *flex* block masks.) ([Jonathan Chang][11])

---

## 12) Interactions & Compatibility

* **PagedAttention block size.** CUDA devices often default to **16 tokens**/block; EPS operates on **page‑groups** of `g` blocks to keep metadata small. Some devices/configs allow different block sizes; keep `--eps-group-blocks` configurable. ([LMCache Blog][12])
* **KV eviction/compression.** EPS composes with **NACL**, **KeDiff/KeyDiff**, **InfiniGen**, and **KV‑Compress**: eviction reduces the set you store; EPS reduces what you read **per step**. Run ablations with each when possible. ([arXiv][2])
* **Retrieval heads / DuoAttention.** Prefer applying EPS on **retrieval heads** only in v1 (DuoAttention finds a small set of heads responsible for long‑range retrieval). This limits risk and cost. ([arXiv][16])
* **Speculative decoding.** If target and draft see different pages, acceptance may suffer. In v1, either (a) **disable EPS for draft**, or (b) apply the **same pruned set** to both.
* **Mac development.** vLLM has **no Metal GPU support**; develop logic/tests on macOS **CPU** or **MLX**, then validate performance on a CUDA/ROCm/XPU cloud box. ([VLLM Docs][3])

---

## 13) Evaluation Plan

### 13.1 Datasets & Tasks

* **LongBench** (ACL’24 & v2, 2024/2025), **RULER** (2024) incl. NIAH variants; optionally HELM Long‑Context tasks. ([arXiv][5])

### 13.2 Hardware & Engines

* NVIDIA L4/A10G/A100/H100 (CUDA 12.x), and one AMD MI300 if available (ROCm ≥ 6.3). Record driver/tool versions. ([VLLM Docs][4])

### 13.3 Metrics

* **Speed:** tokens/s; **latency:** p50/p95; **HBM proxy:** KV bytes read; **EPS overhead:** pre‑pass ms.
* **Quality:** task accuracy/F1/EM per benchmark.
* **Behavior:** pages visited ratio vs. context length; false‑skip rate vs. strict/default.

### 13.4 Grids & Ablations

* Context {8k, 32k, 64k, 128k}; Batch {1, 4, 8}.
* `m ∈ {4, 8, 16}`, `α ∈ {1.0, 1.05, 1.1}`, `last_n ∈ {4, 8}`; `heads={retrieval,all}`; `group_blocks g∈{4,8,16}`.
* With/without eviction methods (NACL/KeDiff/InfiniGen/KV‑Compress) when feasible. ([arXiv][2])

### 13.5 Success & Kill Criteria

* **Ship** if KV bytes read **↓ ≥10–20%** at ≥32k with **p95↓ ≥7–15%** and **Δquality ≤0.3% abs** (strict mode: ~0).
* **Kill** or iterate if KV bytes read **↓ <5%** at 32–64k or quality regressions persist even in strict mode (then publish a negative result).
  (These thresholds align with your hiring‑signal playbook: measurable deltas + artifacts.) 

---

## 14) Test Plan

### 14.1 Unit Tests (math & logic)

* **JL preservation tests:** random (q,K) sets; verify (z^\top G z) tracks (q^\top C q) within tolerance (sweeps over seeds, (m)). ([Computer Science][9])
* **Selector oracle:** Construct synthetic sequences where ground‑truth top‑M pages are known; assert EPS chooses supersets in **strict mode** and matches within tolerance in default.
* **Edge cases:** All‑recent tokens; all‑distant needles; high‑rank vs low‑rank groups; empty groups.

### 14.2 Integration Tests (CPU path)

* vLLM‑CPU build: run the decode pre‑pass, ensure filtered `kv_indices` are passed, ensure no kernel errors (even if slow). ([VLLM Docs][18])

### 14.3 End‑to‑End (GPU)

* Dockerized vLLM (CUDA 12.x); run long‑context grids, capture NVTX‑annotated Nsight traces, dump CSVs for pages visited & bytes read. ([VLLM Docs][4])

---

## 15) Risks & Mitigations

| Risk                              | Impact                           | Mitigation                                                                                                           |
| --------------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Loose bounds ⇒ little pruning** | Speedup vanishes                 | Increase `m`; raise `α`; use 1PC only on retrieval heads; keep last‑N guard. ([arXiv][16])                           |
| **False skips ⇒ quality drop**    | Hallucinations on “needle” tasks | Strict mode + last‑N; scalar fallback bound with (|K_p|_F).                                                          |
| **Batch divergence**              | Coalescing degradation           | Build visit masks first; **sorted newest→oldest**; cap per‑step variance by visiting last‑N for all sequences.       |
| **Memory bloat**                  | Too many groups (small blocks)   | Summarize **page‑groups** (`g≥8`), not raw blocks; only summarize retrieval heads in v1. ([LMCache Blog][12])        |
| **Spec‑dec mismatch**             | Lower acceptance rate            | Share pruned sets between draft/target or disable EPS on draft.                                                      |
| **Mac dev friction**              | Can’t validate perf locally      | Prototype in **MLX**, then run **short cloud sessions** for plots; CPU path for unit/integration. ([ML Explore][13]) |

---

## 16) Rollout & Operational Plan

1. **PR‑0 (Instrumentation):** Add NVTX ranges + counters (pages visited/skipped, bytes read est., pre‑pass ms). Off by default. ([NVIDIA GitHub][8])
2. **PR‑1 (EPS‑JL behind flag):** Add summaries, pre‑pass, head scoping, strict mode. Default OFF.
3. **PR‑2 (Docs + Benchkit):** How‑to, flags, plots on standard grids; include Nsight screenshots.
4. **(Optional) PR‑3 (EPS‑1PC selective):** Retrieval heads only, quantized vectors.
5. **Cloud validation:** Publish CSV/plots & trace packs with exact commands (per your playbook templates). 

---

## 17) Development Plan (Mac‑first, GPU for validation)

* **Mac (MLX) prototype:** Implement JL summaries, selector, oracle tests; export the *same metrics* you’ll later log in vLLM. ([ML Explore][13])
* **vLLM‑CPU:** Build locally (Apple Silicon/X86) to wire flags, storage, and the pre‑pass; run unit/integration tests. ([VLLM Docs][18])
* **Cloud GPU runs:** Use official vLLM CUDA images; collect tokens/s, p50/p95, pages‑visited ratios, LongBench/RULER deltas. (Nsight+NVTX for traces.) ([VLLM Docs][4])

---

## 18) Acceptance Criteria (for v1 merge upstream)

* **Functionality:** EPS flags work; summaries are updated with no extra K reads; pre‑pass builds valid visit lists; PagedAttention respects them; strict mode produces identical quality to baseline on test suite.
* **Performance:** At ≥32k contexts, report **≥10% fewer KV bytes read** and **p95↓** on at least one 7B‑class model **without** quality loss (strict) and with **≤0.3%** drop (default).
* **Observability:** NVTX ranges/counters documented; CSVs reproducible with single script; sample traces included.
* **Docs:** README section with safety/rollback; ablation plots and known limitations.

---

## 19) Implementation Sketch (pseudocode)

```python
# Offline (once): per head random ±1 projection (seeded)
Phi[layer, head]  # shape [d, m]

# On K write (prefill/decode), for token t in head h, layer l:
grp = page_group_of_token(t)        # each group = g blocks
y   = Phi[l,h].T @ k_t              # [m], O(dm)
G[l,h,grp] += outer(y, y)           # [m,m], O(m^2)
# Optional: track Frobenius norm & max row-norm scalars per group

# Decode pre-pass (per step), for head h, layer l:
z = Phi[l,h].T @ q                  # [m]
T = -inf; heap = new top-M structure
visit = set(last_N_recent_groups)

for grp in groups_newest_to_oldest:
    B = sqrt( z.T @ G[l,h,grp] @ z )    # cheap quadratic form
    if B >= T / alpha:
        visit.add(grp)
        T = update_heap_and_get_Mth_best(B)
# Export visit list; pass to attention kernel via kv_indices/mask
```

---

## 20) Things to Watch Out For

* **Block size variability.** CUDA often uses block size 16 and caps supported sizes by build config; always summarize over **groups** of blocks. ([GitHub][19])
* **Indices plumbing.** Ensure filtered `kv_indices` match kernel expectations (masks for flex attention or paged indices). ([Jonathan Chang][11])
* **Head classification.** If you implement `--eps-heads=retrieval`, reuse DuoAttention’s identification approach or a simple heuristic (e.g., heads with high long‑range attention). ([arXiv][16])
* **Tooling versions.** vLLM GPU wheels are built for **CUDA 12.x**; use matching drivers or Docker images. ([VLLM Docs][4])
* **Profiling hygiene.** Keep NVTX ranges balanced and few; otherwise Nsight timelines get unreadable. ([NVIDIA Docs][20])

---

## 21) Related Work (for positioning)

* **PagedAttention & vLLM:** foundational paging of KV cache; improves throughput by reducing fragmentation. ([arXiv][1])
* **Head specialization:** **DuoAttention** (retrieval vs streaming heads) and analyses of retrieval heads motivate head‑scoped policies. ([arXiv][16])
* **KV eviction/compression:** **NACL**, **KeDiff/KeyDiff**, **InfiniGen**, **KV‑Compress**—EPS is complementary (decode gating vs storage policy). ([arXiv][2])

---

## 22) Documentation & Examples to Ship with PR

* **README (EPS):** Concept, flags, strict mode, defaults, when to use/not use.
* **Ablation notebook:** sweeps over `m`, `alpha`, `last_n`, `group_blocks`; plots: pages‑visited ratio vs quality.
* **Repro scripts:** bench grids for {8k, 32k, 64k} × {1,4,8}.
* **Trace pack:** Nsight `.nsys-rep` with NVTX ranges showing reduced HBM reads/attention time slices.
* **Rollback instructions:** set `--eps-method off`.

---

## 23) Career Alignment (why this maps to hiring signals)

This project generates **mergeable PRs**, a **benchmark harness** with reproducible plots, and **trace packs**—exact signals that recruiters and maintainers trust. Keep weekly **floors** (PRs, experiments, traces, posts) and publish results—even negative ones—per the playbook.

---

## Appendix A — Math & References (quick lookup)

* **PagedAttention & vLLM:** arXiv + blog + design docs. ([arXiv][1])
* **Hardware support (install docs):** CUDA/ROCm/XPU; Apple silicon CPU is experimental. ([VLLM Docs][3])
* **JL lemma:** Stanford notes; modern overview. ([Computer Science][9])
* **Frequent Directions:** Liberty (KDD’13), Ghashami et al. (SODA/SIAM). ([Chbrown][10])
* **DPP/log‑det intuition:** Kulesza & Taskar monograph. ([Alex Kulesza][21])
* **Cholesky rank‑1 update/downdate & matrix determinant lemma:** references for stable incremental updates. ([Wikipedia][22])
* **Retrieval heads & DuoAttention:** arXiv papers and ICLR/OpenReview page. ([arXiv][23])
* **Benchmarks:** **LongBench** (ACL’24 & v2), **RULER** (NVIDIA). ([ACL Anthology][24])
* **NVTX/Nsight:** official docs. ([NVIDIA GitHub][8])
* **Mac MLX:** docs & GitHub. ([ML Explore][13])

---

## Appendix B — Minimal Data Structures (shapes)

* `Phi`: `[L, H, d, m]` (or per‑head shared across layers).
* `G`: `[L, H, num_groups, m, m]` (row‑major; fp32).
* `stats`: optional scalars `[L, H, num_groups]` for (|K_p|_F^2), `max_row_norm`.
* `visit_mask`: bitset or compact list per (batch, layer, head) for groups to visit.

---

## Appendix C — Mac Prototype Checklist (MLX)

1. Implement JL projection & Gram updates with **MLX**; ensure kernels run on Metal. ([ML Explore][13])
2. Synthetic oracle (exact page scores) vs EPS page list; compute false‑skip rate under sweeps.
3. Emit the same JSONL metrics and plots you will in vLLM.
4. Port selector + flags into vLLM‑CPU, then validate on cloud GPU.

---

If you want, I can also draft a **flag spec & header stubs** (names, types) and a **BenchKit** run script that reproduces the ablations above with CSV outputs and NVTX‑annotated traces.

[1]: https://arxiv.org/abs/2309.06180?utm_source=chatgpt.com "Efficient Memory Management for Large Language Model Serving with PagedAttention"
[2]: https://arxiv.org/abs/2408.03675?utm_source=chatgpt.com "NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time"
[3]: https://docs.vllm.ai/en/stable/getting_started/installation/index.html?utm_source=chatgpt.com "Installation - vLLM"
[4]: https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html?utm_source=chatgpt.com "GPU - vLLM"
[5]: https://arxiv.org/abs/2308.14508?utm_source=chatgpt.com "LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding"
[6]: https://docs.nvidia.com/nsight-visual-studio-edition/2020.1/nvtx/index.html?utm_source=chatgpt.com "NVIDIA Tools Extension Library (NVTX)"
[7]: https://docs.vllm.ai/en/latest/design/paged_attention.html?utm_source=chatgpt.com "Paged Attention - vLLM"
[8]: https://nvidia.github.io/NVTX/?utm_source=chatgpt.com "NVTX - NVIDIA Tools Extension Library"
[9]: https://cs.stanford.edu/people/mmahoney/cs369m/Lectures/lecture1.pdf?utm_source=chatgpt.com "Johnson-Lindenstrauss Lemma"
[10]: https://chbrown.github.io/kdd-2013-usb/kdd/p581.pdf?utm_source=chatgpt.com "Simple and Deterministic Matrix Sketching - chbrown@github"
[11]: https://jonathanc.net/blog/vllm-flex-attention-from-scratch?utm_source=chatgpt.com "vLLM from scratch with FlexAttention – Jonathan Chang's Blog"
[12]: https://blog.lmcache.ai/en/2025/04/29/bringing-state-of-the-art-pd-speed-to-vllm-v1-with-lmcache/?utm_source=chatgpt.com "Bringing State-Of-The-Art PD Speed to vLLM v1 with ..."
[13]: https://ml-explore.github.io/mlx/?utm_source=chatgpt.com "MLX 0.29.4 documentation"
[14]: https://arxiv.org/abs/1501.01711?utm_source=chatgpt.com "Frequent Directions : Simple and Deterministic Matrix ..."
[15]: https://en.wikipedia.org/wiki/Matrix_determinant_lemma?utm_source=chatgpt.com "Matrix determinant lemma"
[16]: https://arxiv.org/abs/2410.10819?utm_source=chatgpt.com "DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads"
[17]: https://docs.vllm.ai/en/v0.7.2/serving/openai_compatible_server.html?utm_source=chatgpt.com "OpenAI-Compatible Server - vLLM"
[18]: https://docs.vllm.ai/en/stable/getting_started/installation/cpu.html?utm_source=chatgpt.com "CPU - vLLM"
[19]: https://github.com/vllm-project/vllm/issues/14319?utm_source=chatgpt.com "[Doc]: Why is max block_size on CUDA 32? · Issue #14319"
[20]: https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nsight-systems-nvtx-trace.htm?utm_source=chatgpt.com "Tracing NVTX Events with NVIDIA Nsight Systems"
[21]: https://www.alexkulesza.com/pubs/dpps_fnt12.pdf?utm_source=chatgpt.com "Determinantal Point Processes for Machine Learning Contents"
[22]: https://en.wikipedia.org/wiki/Cholesky_decomposition?utm_source=chatgpt.com "Cholesky decomposition"
[23]: https://arxiv.org/abs/2404.15574?utm_source=chatgpt.com "Retrieval Head Mechanistically Explains Long-Context Factuality"
[24]: https://aclanthology.org/2024.acl-long.172/?utm_source=chatgpt.com "LongBench: A Bilingual, Multitask Benchmark for Long ..."
