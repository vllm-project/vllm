# w8a16-oproj — Validation Results

**Decision: PASS** (marginal E2E win on top of shipped fp8-qkv; both batch sizes clear the 0.25% min bar and are statistically significant at p<0.01).

## Implementation summary and scope

Weight-only **W8A16 FP8-e4m3** requant of the dense attention output projection
`self_attn.o_proj` (GEMM [out=2816, in=4096]), dispatched through the in-tree
**FP8-Marlin** path. The bf16 checkpoint weight (ignore-listed → normally
`UnquantizedLinearMethod`) is requantized once at load time with per-output-channel
scales, repacked via `prepare_fp8_layer_for_marlin(size_k_first=True,
input_dtype=None)`, and applied via `apply_fp8_marlin_linear` against **unquantized
bf16 activations**. Halves the O-proj weight bytes streamed from HBM per decode step.

Files modified:
- `vllm/model_executor/layers/quantization/oproj_fp8_w8a16.py` (new — `OprojFp8W8A16LinearMethod`)
- `vllm/model_executor/layers/quantization/modelopt.py` (sibling branch in `is_layer_excluded`, gated on `VLLM_OPROJ_FP8_W8A16` + `prefix.endswith("self_attn.o_proj")`)
- `vllm/envs.py` (`VLLM_OPROJ_FP8_W8A16: bool = False`, default "0")

Scope adherence: full scope, no descope. Fires only on ignore-listed bf16 o_proj
layers. No bias (dead branch passed through for generality).

**Device:** RTX PRO 6000 Blackwell, sm_120 cap(12,0), 188 SMs, 100KB shmem/SM,
99KB per-block opt-in (NOT Hopper 228KB), ~1598 GB/s peak BW.

## Gate 5.1a — kernel correctness (validator, independent tests)

Floors: cosine > 0.995 AND relL2 ≤ 0.08.

| M | cosine | relL2 | verdict |
|---|--------|-------|---------|
| 1 | 0.999651 | 0.026434 | PASS |
| 8 | 0.999638 | 0.026891 | PASS |
| 16 | 0.999644 | 0.026668 | PASS |
| 64 | 0.999643 | 0.026714 | PASS |
| 1024 | 0.999647 | 0.026582 | PASS |
| CUDA-graph replay | 0.999646 | 0.026624 | PASS (eager_match True) |

**5.1a PASS.** Cross-check: champion smoke test (own tree, worst cos 0.9996 / relL2
0.027) agrees with validator to 4 decimals — no discrepancy.

## Gate 5.2 — kernel speedup (validator): NULL (honest)

Cold-cache (production-representative), FP8-Marlin slower than bf16 at every M
(speedup 0.50–0.71×; warm-cache 0.75–1.50×). Recorded as measured — **no kernel-level
win claimed.** The E2E benefit below comes from halved HBM weight traffic in real
decode, not from a faster standalone GEMM.

## Gate 5.3a — kernel dispatch proof: PASS

`nsys stats --report cuda_gpu_kern_sum rounds/2/profiling/nsys/opt_bs1.nsys-rep`:
**30 instances** of `void marlin::Marlin<…,(int)256,…>(…)` = 30 decoder layers × 1
o_proj each. The Marlin W8A16 GEMM provably fired — not a bf16 no-op.

## Gate 5.1b — GSM8K accuracy: PASS

Golden refs = PURE-BF16. `correctness_verdict.json`: opt **40.0% (527/1319)** vs
baseline 38.7% (511/1319), delta **+1.21pp**, threshold 37.74% (baseline − 1.0pp).
Absolute floor 0.3873 also cleared. Verdict PASS.

## Gate 5.3 — E2E marginal latency (STACKED: baseline = fp8-qkv on; opt = fp8-qkv + w8a16 on)

Same-tree / same-cache run. Raw per-bucket vLLM latency JSONs; 10 iters each.
Verdict thresholds: min 0.25%, noise 0.5%, catastrophic −5.0%.

| BS | baseline avg (s) | opt avg (s) | improvement | Welch t (p<0.01?) | verdict |
|----|------------------|-------------|-------------|-------------------|---------|
| 1 | 6.063683 | 5.830780 | **+3.84%** | t=20.9 ✓ | PASS |
| 8 | 8.634578 | 8.406295 | **+2.64%** | t=3.72 ✓ | PASS |

Both gaps are statistically significant at p<0.01 (bs1 gap = 7σ of per-sample spread;
bs8 SE 0.061s, gap 0.228s). Decode is 98.7–99.4% of E2E; decode-only improvement
tracks E2E (bs1 +3.81%, bs8 +2.75%), consistent with the halved-HBM-weight-traffic
mechanism. No multi-launch aggregate (single-launch run), so no automated NOISE flag;
manual Welch test rules out noise at both BS.

## Fastpath marker — infrastructure false-negative (documented, not a no-op)

Invocation 2 exited 1: `Fast-path evidence FAILED for opt at bs1.
Missing=['VLLM_OPROJ_FP8_W8A16 active']`. **Root cause is structural log-routing, not
a missing kernel:**
- The marker fires via `logger.info_once` at **model-load** (`opt_supervisor.log:61`,
  `oproj_fp8_w8a16.py:116`, EngineCore pid=98896), routed to stderr → supervisor log.
- `--require-fastpath` (sweep line 4654) reads only the **per-bucket bench-stdout log**
  (`opt_log_p = logs_dir/f"{opt_label}_{tag}.log"`, line 3996), which contains just
  `Warming up…` / `Avg latency:` / percentile lines — no `logger` output.
- Baseline has identical routing (QKV marker only in `baseline_supervisor.log:59`) and
  passed solely because enforcement is OPT-ONLY.

`logger.info_once` cannot be made to appear in per-bucket stdout without violating the
once-per-process contract or adding a `print` on the timed path. Dispatch is instead
proven authoritatively by Gate 5.3a (30 Marlin instances) + the supervisor-log marker
+ the GSM8K accuracy shift. Escalated to team-lead for contract adjudication.

## Repro

```bash
# opt (fp8-qkv + w8a16 stacked)
VLLM_QKV_FP8_REQUANT=1 VLLM_OPROJ_FP8_W8A16=1 vllm bench latency \
  --model nvidia/Gemma-4-26B-A4B-NVFP4 --tensor-parallel-size 1 \
  --max-model-len 4096 --input-len 1024 --output-len 1024 \
  --batch-size {1,8} --num-iters 10 --trust-remote-code
# baseline: drop VLLM_OPROJ_FP8_W8A16=1
# dispatch proof:
nsys stats --report cuda_gpu_kern_sum rounds/2/profiling/nsys/opt_bs1.nsys-rep | grep -i marlin
```

commit_sha: 7ed4ad8dc8a8229213a26d73f80e0769018d0697

## Projection Accuracy

- Op ID: `w8a16-oproj`
- Projected E2E improvement: n/a

| BS | Projected | Realized | |Δ| (pp) | Ratio | Flag |
|----|-----------|----------|---------|-------|------|
| 1 | n/a | +3.84% | 0.00 | — | ok |
| 8 | n/a | +2.64% | 0.00 | — | ok |

All batch sizes within projection tolerance.
