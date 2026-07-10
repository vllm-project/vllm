# AMMO Campaign — Stage-7 Final Report
## nvidia/Gemma-4-26B-A4B-NVFP4 on RTX PRO 6000 Blackwell (sm_120)

- **Instance**: i-0abcb0db80c7150d0, g7e.2xlarge, 8 vCPU, us-west-2c, 1x RTX PRO 6000 Blackwell (sm_120, cc=12.0, ~96GB, 100KB SHM/SM), CUDA 13.2. SPOT.
- **vLLM**: built from source, tag v0.24.0 (docker_commit ee0da84ab). TP=1, dtype bf16 (weights NVFP4), workload il1024/ol1024, batch sizes {1,8}.
- **Session**: 278251d0-7dfa-4f7d-88ad-0a1b1be266e1. Champions/workers = opus-4-8.
- **Status**: campaign_complete / operator_scoped_stop. No Round 3.
- **Provenance note**: this REPORT.md was authored by the operator harness from raw-JSON-verified numbers after the in-container report-writer sub-agent exited without persisting its output; all figures are cross-checked against state.json + per-sweep JSON.

## Baseline (pure-bf16 dense; frozen Round-1 anchor)
- E2E latency: **bs1 6.4746 s / bs8 8.9865 s** (reproduced within noise of the original 6.4792/9.0800).
- GSM8K golden accuracy: 0.3973 (524/1319); re-measured 0.3874 (511/1319) — greedy run-to-run churn ~1pp. Floor for lossy ops = 0.3873.
- golden_refs md5 5ea568f8… (verified intact / never re-anchored across the campaign).

## SHIPPED optimizations (both merged to session mainline; cumulative stacked)

### R1 — fp8-qkv  (commit b779d6ddc)  [LOSSY, SHIPPED]
Load-time requant of the otherwise-bf16 dense attention QKV projection (`*.self_attn.qkv_proj`) to FP8-e4m3, dispatched through the in-tree SM120 FP8 CUTLASS GEMM (scaled_fp8_quant + cutlass_scaled_mm; per-tensor weight scale, per-token dynamic activation). No authored CUDA; torch.compile-safe by construction. Files: vllm/envs.py, modelopt.py, new qkv_fp8_requant.py.
- Gate 5.1a (per-kernel): cosine 0.99930, relL2 0.0372 (all 5 shapes M=1..1024 + adversarial + CUDA-graph replay) — PASS.
- Gate 5.1b (GSM8K@1319): 0.4238 (559/1319) ≥ floor 0.3873 — PASS.
- Anti-no-op: "VLLM_QKV_FP8_REQUANT active" marker fired (opt_supervisor.log).
- Marginal E2E: **bs1 1.072x / bs8 1.032x** (measured; bs8 below the 1.050x prior label — recorded as measured).

### R2 — w8a16-oproj  (commit 5f9768170)  [LOSSY, SHIPPED]
Weight-only FP8-e4m3 W8A16 requant of the dense attention output projection (`*.self_attn.o_proj`, shape [out=2816, in=4096]) via the in-tree FP8-Marlin path (prepare_fp8_layer_for_marlin + apply_fp8_marlin_linear); activations stay bf16. Chosen over NVFP4-W4A16 on o_proj because NVFP4 gives relL2 ~9.5% (fails ≤8% gate) whereas FP8-W8A16 gives ~2.7%. Files: vllm/envs.py, modelopt.py, new oproj_fp8_w8a16.py.
- Gate 5.1a (per-kernel): cosine 0.99964, relL2 0.0269 (worst-case, all shapes + adversarial + CUDA-graph) — PASS. Independent validator used its own reference (adversarial separation).
- Gate 5.1b (GSM8K@1319): 0.3995 (527/1319) ≥ floor 0.3873; correctly anchored to pure-bf16 (NOT re-anchored to fp8-qkv-on) — PASS.
- Gate 5.2 (isolated micro-bench): NULLed honestly (overhead-bound at tiny M) — E2E is the ship number.
- Anti-no-op: "VLLM_OPROJ_FP8_W8A16 active" marker fired (opt_supervisor.log; clean baseline-vs-opt differential) + nsys Gate-5.3a dispatch proof. (The require-fastpath per-bucket grep miss was a known info_once false-negative, resolved on authoritative evidence; guard not weakened.)
- Marginal E2E (on top of fp8-qkv baseline): **bs1 1.040x / bs8 1.027x** (>> 0.25% min-improvement bar).

### CUMULATIVE (fp8-qkv + w8a16-oproj) vs pure-bf16 baseline
- **bs1 1.110x  (6.4746 → 5.8308 s)**
- **bs8 1.069x  (8.9865 → 8.4063 s)**
- state.json cumulative_speedup_vs_round1 = 1.11042.

## NULL / diagnostic results (documented, NOT shipped)
- **moe-act-fusion (R1, lossless)** — architectural-impossibility null: the SM120 SafeFP4 grouped-GEMM has max CTA_N=256 but gated-GELU needs gate/up halves (inter_size 704) co-resident in one epilogue tile → a pure-epilogue fusion is uncompilable on the production kernel (vLLM/TRT-LLM source encodes this via static_assert). Honest null; 0-byte patch.
- **dense-lossless (R1)** — null: cache-busted re-measurement showed the dense bf16 slice already runs at ~88% of achievable HBM BW → only ~1.13x lossless headroom; not worth a same-dtype op.
- **narrower-NVFP4-dense probe (owed from prior run) — DIAGNOSTIC-ONLY null**: NVFP4-quantized the least-sensitive dense layers (by weight-error relL2 proxy), W4A16, GSM8K per variant vs pure-bf16 (both shipped flags OFF).
  - Variant A (8 layers): GSM8K 0.417 (550/1319), Δ +2.96pp, McNemar p=0.091 (NOT significant).
  - Variant B (16 layers): GSM8K 0.4124 (544/1319), Δ +2.50pp, McNemar p=0.152 (NOT significant).
  - Both clear the 0.3873 floor — i.e. a narrow least-sensitive NVFP4 scope did NOT reproduce the prior 25-layer v-slice collapse (−15.3pp → 0.2441). But the deltas are within GSM8K noise (no real gain), the weak relL2 proxy barely differentiates layers (~0.5% spread; all layers ~9.4% relL2), and the break-point between 16 and ~25 layers was NOT mapped (2 of 3 GSM8K budget used). Verdict: not shippable as a win; a genuinely narrow scope preserves accuracy within noise but yields no measured speedup benefit here (NVFP4 dense is overhead-floor-limited at these shapes). Disposition: diagnostic, no merge, no PR.

## Off-box durability
All shipped patches (git-authoritative), validation docs, state.json, probe results, and the R2 ship audit were collected off the (spot) box to /fsx and to the workspace optimizations_achieved/ directory. Both ship commits are git-bundled to /fsx checkpoints.
