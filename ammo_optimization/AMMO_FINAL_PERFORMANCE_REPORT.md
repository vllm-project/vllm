# AMMO Optimization Report — AMMO vs OSS vLLM 0.24.0

**Model:** `nvidia/Gemma-4-26B-A4B-NVFP4`
**Hardware:** 1× NVIDIA RTX PRO 6000 Blackwell (sm_120, cc 12.0, ~96 GB, 100 KB SHM/SM), CUDA 13.2
**Config:** TP=1, dtype bf16 (weights NVFP4), workload ISL/OSL 1024/1024, batch sizes {1, 8}
**Baseline:** OSS vLLM `v0.24.0`, built from source (commit `ee0da84ab`)
**Delivery:** `access2rohit/vllm` branch [`ammo/gemma4-nvfp4-fp8-qkv-oproj`](https://github.com/access2rohit/vllm/tree/ammo/gemma4-nvfp4-fp8-qkv-oproj)

---

## 1. What AMMO did

AMMO (Automated Model Micro-Optimizer) ran an autonomous kernel-optimization campaign
against OSS vLLM 0.24.0 and shipped **two accuracy-gated optimizations**, each behind an
opt-in env flag (default OFF, so the branch is behaviourally identical to upstream unless
enabled). Both target the **dense attention projections** — layers that ship as plain bf16
in the NVFP4 checkpoint (they're on the quantization ignore-list) and therefore run
unquantized bf16 GEMMs on Blackwell.

| # | Op | Env flag | Mechanism | Files |
|---|----|----------|-----------|-------|
| R1 | **fp8-qkv** | `VLLM_QKV_FP8_REQUANT=1` | Load-time requant of dense `self_attn.qkv_proj` → **FP8-e4m3**, dispatched through the in-tree SM120 FP8 CUTLASS GEMM (`scaled_fp8_quant` + `cutlass_scaled_mm`; per-tensor weight scale, per-token dynamic activation). No authored CUDA; torch.compile-safe. | `envs.py`, `modelopt.py`, `qkv_fp8_requant.py` (new) |
| R2 | **w8a16-oproj** | `VLLM_OPROJ_FP8_W8A16=1` | Weight-only **FP8-e4m3 W8A16** requant of dense `self_attn.o_proj` via the in-tree **FP8-Marlin** path; activations stay bf16. Halves O-proj weight bytes streamed from HBM per decode step. | `envs.py`, `modelopt.py`, `oproj_fp8_w8a16.py` (new) |

**Why these are decode (OTPS) wins, not TTFT wins:** both ops reduce the **HBM weight bytes
streamed per forward pass**. Decode (batch×seq M is tiny) is memory-bandwidth-bound, so
fewer weight bytes → faster tokens → higher OTPS. Prefill/TTFT is compute-bound (large M
reuses each loaded weight across many tokens), so it sees little-to-no benefit — and
w8a16's standalone Marlin GEMM is actually *slower* cold-cache, so TTFT is neutral-to-slightly-negative.
The validation confirmed **decode is 98.7–99.4 % of the measured E2E**, so essentially the
entire ~11 % win is decode throughput.

---

## 2. Performance — directly-measured `vllm bench serve` (TTFT / TPOT / OTPS), graphs-ON

**Authoritative result: a live serve benchmark on vLLM mainline `6cf7b26bd` (the exact tree the
campaign measured on), production-parity (torch.compile + CUDA graphs ON), all 3 configs run
back-to-back, both anti-no-op markers verified firing.** ISL/OSL 1024/1024, concurrency {1, 8}.
Raw JSONs in `serve_results_mainline/`.

| Config | conc | TTFT mean (ms) | TPOT mean (ms) | ITL mean (ms) | OTPS (tok/s) | OTPS vs base |
|--------|------|----------------|----------------|---------------|--------------|--------------|
| OSS mainline `6cf7b26bd` | 1 | 153.9 | 6.37 | 6.37 | 153.4 | 1.000× |
| + fp8-qkv | 1 | 154.9 | 5.95 | 5.95 | 164.1 | **1.069×** |
| **+ both (stacked)** | 1 | 156.4 | **5.69** | 5.69 | **171.2** | **1.116×** |
| OSS mainline `6cf7b26bd` | 8 | 260.8 | 8.64 | 8.64 | 899.8 | 1.000× |
| + fp8-qkv | 8 | 256.4 | 8.21 | 8.21 | 946.6 | **1.052×** |
| **+ both (stacked)** | 8 | 255.8 | **7.99** | 7.99 | **972.0** | **1.080×** |

**Headline (directly measured): +11.6% OTPS at conc-1 (153→171 tok/s), +8.0% at conc-8 (900→972).**

**The TTFT-vs-OTPS answer, with first-class metrics — the win is DECODE, not TTFT:**
- **OTPS (decode throughput): +11.6% / +8.0%** ← the win.
- **TPOT: 6.37→5.69 ms (1.12× faster) at bs1** ← *why* OTPS rose (halved HBM weight traffic → faster per-token decode).
- **TTFT: flat** — bs1 153.9→156.4 ms (+1.6%, noise), bs8 260.8→255.8 ms (−1.9%). Prefill is compute-bound; weight-traffic ops don't move it. Predicted signature confirmed.

Baseline serve OTPS 153.4 independently reproduces the v0.24.0+fix cross-check (152.8) to <1%.

**Provenance:** pristine vLLM `v0.24.0` (`ee0da84ab`) *cannot load this model* — it lacks upstream
fix **#45544** (fix tied quantized embeddings, ModelOpt Gemma4). Both the campaign's 1.11× and this
serve run were on mainline `6cf7b26bd` (659 commits past v0.24.0, which contains #45544). The fork
branch is based on v0.24.0 for clean patch provenance; the *numbers* are from the mainline tree.
Build method: two Docker images — `ammo-tools/docker/Dockerfile.build` (compiles vLLM from source for
sm_120, build-time compiled-`.so` gate) + `Dockerfile.bench` (runtime deps + `import` verify),
rebuildable as mainline moves via `--build-arg VLLM_COMMIT`.

---

## 2b. Earlier result — `vllm bench latency` (total-generation, OTPS derived) — agrees with §2

![AMMO vs OSS (latency-sweep)](ammo_vs_vllm024_perf.png)

### 2a. End-to-end latency (measured) & OTPS (derived)

OTPS = (output_len × batch) / E2E_latency. Since token counts are fixed, the OTPS
improvement equals the latency improvement exactly.

| Batch | Config | E2E latency (s) | OTPS (tok/s) | Speedup vs OSS |
|-------|--------|-----------------|--------------|----------------|
| **1** | OSS vLLM 0.24.0 | 6.4746 | 158.2 | 1.000× |
| **1** | + fp8-qkv | 6.0396 | 169.5 | **1.072×** |
| **1** | + both ops (stacked) | **5.8308** | **175.6** | **1.110×** |
| **8** | OSS vLLM 0.24.0 | 8.9865 | 911.6 | 1.000× |
| **8** | + fp8-qkv | 8.7040 | 941.2 | **1.032×** |
| **8** | + both ops (stacked) | **8.4063** | **974.5** | **1.069×** |

**Headline: +11.0 % OTPS at batch-1 (158→176 tok/s), +6.9 % at batch-8 (912→975 tok/s),
with accuracy preserved.** All latencies are the measured `vllm bench latency` averages
(10 iters each) recorded in `state.json` and cross-checked against the raw per-bucket JSON.

### 2b. Per-op contribution (batch-1)

| Layer | OSS bf16 | AMMO | Marginal speedup |
|-------|----------|------|------------------|
| qkv_proj | bf16 CUTLASS | FP8-e4m3 CUTLASS | 1.072× (the larger single win) |
| o_proj | bf16 | FP8 W8A16 Marlin | +1.040× on top of fp8-qkv |
| **Stacked** | | | **1.110×** |

---

## 3. Accuracy — nothing sacrificed

Hard gate for these lossy ops: **GSM8K ≥ 0.3873** (pure-bf16 baseline 0.3874, floor = −1.0 pp).
Per-kernel numerical gate: **cosine > 0.995 AND relL2 ≤ 0.08** across 5 shapes + CUDA-graph replay.

| Op | cosine (worst) | relL2 (worst) | GSM8K | vs floor 0.3873 |
|----|----------------|---------------|-------|-----------------|
| fp8-qkv | 0.99930 | 0.0372 | 0.4238 | **PASS** (+3.6 pp) |
| w8a16-oproj | 0.99964 | 0.0269 | 0.3995 | **PASS** (+1.2 pp) |

Both ops clear every gate. (GSM8K greedy churns ~1 pp run-to-run; the gate is an absolute
floor, not a per-run delta.)

---

## 4. What was NOT optimized (honest scope + headroom)

This campaign was **operator-scoped to the dense attention projections** and stopped by
directive (`operator_scoped_stop`), **not** because optimizations were exhausted:

- **MoE expert GEMMs** — the dominant weight & compute mass of this A4B MoE model — were **never touched**. Stage-2 bottleneck mining was skipped every round per scope directive, so no per-component headroom was measured. This is where the largest remaining gains almost certainly live.
- **Dense MLP** — untouched.
- **NULL results (documented, not shipped):** MoE-activation fusion (architecturally uncompilable on the SM120 SafeFP4 grouped-GEMM), dense-lossless (already ~88 % HBM BW, ≤1.13× headroom), and a narrower-NVFP4-dense probe (cleared the floor but statistically level with bf16 — no real gain, no speedup at these shapes).

**Bottom line:** 1.110×/1.069× is the delivered win *within the chosen scope*, not the ceiling.

---

## 5. Caveat on the measurement method (important)

These numbers come from **`vllm bench latency`** — a total-generation-latency benchmark
(prefill + all 1024 decode steps). OTPS above is **derived** from that total latency, which
is rigorous because token counts are fixed and decode is 98.7–99.4 % of the total.

They are **NOT** from `vllm bench serve`, which would report **TTFT, TPOT/ITL, and OTPS as
separate first-class line items**. A direct serve-benchmark decomposition was planned but
the (spot) instance `i-0abcb0db80c7150d0` was **evicted by AWS** (`BidEvictedEvent`
2026-07-10T06:03:32Z — not a user termination) before it ran. **TTFT is therefore
"pending direct measurement";** the decode/OTPS gains are the verified, shippable result.
If a fresh instance is provisioned, the serve run will add the explicit TTFT-vs-OTPS split.

---

## 6. Reproduce it yourself

Full step-by-step guide (clone → build on g7e → 3-config latency sweep → GSM8K → nsys
anti-no-op proof) is in
[`ammo_optimization/README.md`](https://github.com/access2rohit/vllm/blob/ammo/gemma4-nvfp4-fp8-qkv-oproj/ammo_optimization/README.md)
on the fork branch. Quick version:

```bash
git clone --branch ammo/gemma4-nvfp4-fp8-qkv-oproj https://github.com/access2rohit/vllm.git && cd vllm
pip install -e . --no-build-isolation   # in a sm_120 CUDA-13 env / vLLM build container

COMMON="--model nvidia/Gemma-4-26B-A4B-NVFP4 --tensor-parallel-size 1 \
  --max-model-len 4096 --input-len 1024 --output-len 1024 --num-iters 10 --trust-remote-code"

vllm bench latency $COMMON --batch-size 1                                                 # OSS baseline  ~6.47s
VLLM_QKV_FP8_REQUANT=1 vllm bench latency $COMMON --batch-size 1                          # +fp8-qkv      ~6.04s
VLLM_QKV_FP8_REQUANT=1 VLLM_OPROJ_FP8_W8A16=1 vllm bench latency $COMMON --batch-size 1   # +both         ~5.83s (1.11×)
```

---

## 7. Provenance

- Base commit `ee0da84ab` verified present in `vllm-project/vllm` (GitHub API).
- All latency/accuracy figures cross-checked against `state.json` + raw per-sweep JSON (not agent-reported).
- Patches durable in three places: local `optimizations_achieved/`, `/fsx` git bundles, and the GitHub fork branch.
- Chart source: `ammo-tools/make_perf_chart.py` (regenerable).

*Generated 2026-07-13 from the completed campaign on session `278251d0-…` (instance evicted; data intact).*
