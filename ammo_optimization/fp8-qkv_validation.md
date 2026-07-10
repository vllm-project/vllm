# Track Validation Results — fp8-qkv (Round 1)

**Classification:** lossy (per-tensor FP8-e4m3 weight requant)
**Verdict:** PASS (both batch sizes)
**Branch / commit:** `tracks/fp8-qkv-r1` @ `e2e0ec2e4`
**Category:** `weight_layout_transform`
**Provenance:** Operator-directed re-realization. Patch `/tmp/fp8-qkv.diff.patch`
(byte-identical to a prior host-shipped patch) re-applied and re-validated fresh
on this tree (`v0.23.1rc0-962-g6cf7b26bd`). Stage 2 mining + Stage 3 debate were
skipped by explicit operator decision (scope pre-decided).

## Mechanism

Load-time requant of the otherwise-bf16 dense attention QKV projection
(`self_attn.qkv_proj`) to FP8-e4m3, gated behind `VLLM_QKV_FP8_REQUANT=1`.
The projection weights are on the model's NVFP4 quant-config `ignore` list, so
they normally receive `UnquantizedLinearMethod` (bf16). When the flag is set,
`QkvFp8RequantLinearMethod` instead:

- **`process_weights_after_loading`** (off the timed path): per-tensor FP8-e4m3
  weight requant, `scale = w.abs().amax() / 448.0`, stored transposed
  (`[K, N]`, `stride(0)==1`) as the in-tree `Fp8LinearMethod` does.
- **`apply`** (in-graph): per-token dynamic activation quant via
  `ops.scaled_fp8_quant(use_per_token_if_dynamic=True)`, then
  `ops.cutlass_scaled_mm` on the SM120 FP8 CUTLASS GEMM.

No CUDA kernel authored — reuses two in-tree custom ops, so the torch.compile
contract holds by construction (functional op reuse, no in-place mutation, no
`data_ptr()` on a traced tensor).

**Scope:** all 30 decoder-layer `self_attn.qkv_proj` (in=2816, out=8192;
q=4096 / k=2048 / v=2048, head_dim=256). Confirmed all 30 layers'
`self_attn` are in the NVFP4 `exclude_modules` list → bf16 → the requant fires
on every layer (not a silent no-op).

## Gate 5.1a — Kernel numerical correctness (independent validator)

Independent `ammo-impl-validator` built a fresh fp32-upcast bf16 reference
(`y_ref = x @ W.T`), distinct from the FP8 candidate path. Bar: cosine > 0.995
AND relL2 ≤ 0.08 (lossy track).

| M (tokens) | cosine | relL2 | pass |
|-----------:|-------:|------:|:----:|
| 1    | 0.99931 | 0.03719 | ✅ |
| 8    | 0.99930 | 0.03745 | ✅ |
| 16   | 0.99930 | 0.03748 | ✅ |
| 64   | 0.99930 | 0.03746 | ✅ |
| 1024 | 0.99930 | 0.03753 | ✅ |

max_abs_err = 0.15625; NaN/INF check PASS; CUDA-graph capture+replay PASS
(cosine 0.99932, relL2 0.03689). **Overall: PASS.**

Corroborates the prior host-side artifact (cosine 0.99927–0.99935, relL2
0.036–0.038).

## Gate 5.1b — GSM8K accuracy vs golden refs (n=1319)

| | accuracy | correct |
|---|---:|---:|
| baseline (measured) | 0.3874 | 511/1319 |
| opt | **0.4238** | 559/1319 |

- Operator hard floor: **0.3873** → PASS.
- Authoritative sweep threshold (baseline 0.3874 − 1.0pp tolerance): 0.3774 → PASS.
- accuracy_delta = +0.0364; churn_rate = 0.394 (symmetric gain/loss — expected
  non-deterministic MoE/NVFP4 variance, not a real accuracy gain).

**Verdict: PASS.** (verdict file:
`rounds/1/sweeps/opt/fp8-qkv/json/correctness_verdict.json`)

## Gate 5.3 — E2E latency (production parity: CUDA graphs + torch.compile)

Opt-dir baseline JSONs are byte-identical (md5-confirmed by audit) to the
frozen Stage-1 anchor — same baseline, no drift/env leak.

| BS | baseline avg (s) | opt avg (s) | speedup | per-BS verdict |
|---:|-----------------:|------------:|--------:|:--------------:|
| 1 | 6.47461 | 6.03962 | **1.0720x** | PASS |
| 8 | 8.98654 | 8.70403 | **1.0325x** | PASS |

Per-BS: noise tolerance 0.5% (PASS ≥ 1.005x), catastrophic 5% (REGRESSED < 0.95x).
Both bs1 and bs8 are PASS.

## Fast-path evidence (no-op guard)

Sweep ran with `--require-fastpath`. Marker
`VLLM_QKV_FP8_REQUANT active: dense QKV projection requantized bf16->fp8_e4m3`
fired **once** at engine load (`opt_supervisor.log:59`, `qkv_fp8_requant.py:98`)
and **0 times** in all baseline logs. The requant genuinely executed.

## Production parity

`enforce_eager=False`, `VLLM_COMPILE` (level 3), `CUDAGraphMode.FULL_AND_PIECEWISE`.
No `--enforce-eager` / `TORCH_COMPILE_DISABLE` in opt logs.

## Amdahl consistency

QKV bf16→fp8 saves 0.692 GB/decode-step ≈ 6.5% of decode HBM traffic
(≈10.7 GB/step @ 1.7 TB/s, 6.28 ms TPOT) → ceiling ≈ 1.069x. Measured bs1
1.072x sits **at** the ceiling → no contamination. bs8's lower 1.032x is
expected: the QKV GEMM shifts compute-bound at higher batch, less
bandwidth-limited.

## Audit

T_AUDIT_S45 verified all six dimensions against primary artifacts
(`rounds/1/audits/stage_45.md`).
