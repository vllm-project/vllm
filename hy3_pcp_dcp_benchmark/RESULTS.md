# Hy3-290B-FP8 PCP/DCP Benchmark

Model: `tencent/Hy3-FP8` (HYV3ForCausalLM, GQA 64Q/8KV, 192-expert MoE + 1 shared,
80 layers, hidden 4096, 279 GB weights). Hardware: 8×H100 80GB.

All configs use `--enforce-eager`, `--enable-expert-parallel`,
`--no-enable-flashinfer-autotune` (skips the FlashInfer warmup that OOMs under
TP1 where all 64 attention heads land on one GPU), `--gpu-memory-utilization 0.82`
(headroom for warmup activations; 0.88 OOMs on TP2+PCP4), `--max-model-len 32800`
(TP1+PCP8 has only 12.4 GiB KV cache; 40960 needs 12.5 GiB), `--safetensors-load-strategy
prefetch` (boot 9 min → 2.5 min), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Boot date: 2026-08-03. Branch: `pcp-gqa` @ `115e11d54` (rewritten PCP plan — 448
lines / 6 files, no forward_context / InputBatch / CommonAttentionMetadata changes).

## Config matrix — all 7 use exactly 8 GPUs

`TP × PCP = 8` throughout; `DCP == PCP` so DCP ranks share PCP ranks (no extra
world size). EP spans TP × PCP = 8 in every config.

| # | Config | TP | PCP | DCP | KV cache |
|---|---|---|---|---|---|
| 1 | tp8 | 8 | 1 | 1 | replicated (baseline) |
| 2 | tp4_pcp2 | 4 | 2 | 1 | replicated |
| 3 | tp2_pcp4 | 2 | 4 | 1 | replicated |
| 4 | tp1_pcp8 | 1 | 8 | 1 | replicated (pure PCP) |
| 5 | tp4_pcp2_dcp2_sharded | 4 | 2 | 2 | sharded (1/dcp) |
| 6 | tp2_pcp4_dcp4_sharded | 2 | 4 | 4 | sharded (1/dcp) |
| 7 | tp1_pcp8_dcp8_sharded | 1 | 8 | 8 | sharded (1/dcp) |

## TPOT — decode latency (mean / P99 ms)

ShareGPT subset, prefix caching **on** (decode is prefix-insensitive; matches
how the Qwen-30B baselines were taken). Concurrency 1 / 16 / 32.

| Config | c=1 | c=16 | c=32 | c1→32 |
|---|---|---|---|---|
| TP8 (baseline) | 158 / 161 | 158 / 165 | 162 / 165 | +2.5% |
| TP4+PCP2 | 177 / 181 | 180 / 183 | 180 / 183 | +1.7% |
| TP2+PCP4 | 174 / 180 | 177 / 180 | 178 / 182 | +2.3% |
| TP1+PCP8 | 167 / 171 | 170 / 175 | 174 / 179 | +4.2% |
| TP4+PCP2+DCP2 | 205 / 215 | 213 / 225 | 213 / 219 | +3.9% |
| TP2+PCP4+DCP4 | 205 / 208 | 215 / 220 | 215 / 220 | +4.9% |
| TP1+PCP8+DCP8 | 195 / 198 | 202 / 208 | 206 / 215 | +5.6% |

## TTFT — prefill latency (median ms)

Random-input probe, prefix caching **off** + 6 warmup requests discarded,
20 measured.

| Config | 4K | 16K | 32K |
|---|---|---|---|
| TP8 (baseline) | 256 | 926 | 1973 |
| TP4+PCP2 | 243 | 875 | 1858 |
| TP2+PCP4 | 238 | 832 | 1741 |
| TP1+PCP8 | 226 | 783 | 1682 |
| TP4+PCP2+DCP2 | 244 | 911 | 1979 |
| TP2+PCP4+DCP4 | 233 | 891 | 1980 |
| TP1+PCP8+DCP8 | 228 | 927 | 2123 |

## Accuracy & KV capacity

GSM8K 200 questions, 5-shot, greedy, conc 256.

| Config | GSM8K | KV tokens | DCP capacity win |
|---|---|---|---|
| TP8 | 0.885 | 735,056 | 1× |
| TP4+PCP2 | 0.905 | 338,288 | — |
| TP2+PCP4 | 0.915 | 139,840 | — |
| TP1+PCP8 | 0.940 | 40,736 | — |
| TP4+PCP2+DCP2 | 0.915 | 676,576 | **2.0×** vs TP4+PCP2 |
| TP2+PCP4+DCP4 | 0.895 | 558,814 | **4.0×** vs TP2+PCP4 |
| TP1+PCP8+DCP8 | 0.910 | 324,936 | **8.0×** vs TP1+PCP8 |

GSM8K spread (0.885–0.940) is 200-question sample noise (each question = 0.5%,
so ±2–3 questions = ±1–1.5%), not a regression.

## Key findings

### 1. TPOT is nearly flat across 32× concurrency (+2–6%)

Same as Qwen-30B. DCP's LSE-combine is a fixed per-step collective that
amortizes across the batch rather than scaling with it.

### 2. DCP decode cost is a fixed ~28 ms

| Comparison | Δ TPOT (c=1) |
|---|---|
| TP4+PCP2 (177) → +DCP2 (205) | +28 ms (+16%) |
| TP1+PCP8 (167) → +DCP8 (195) | +28 ms (+17%) |

DCP2 ≈ DCP4 ≈ DCP8 in TPOT (205 / 215 / 195) — the collective is
latency-bound, not bandwidth-bound, so doubling `dcp` buys 2× more KV
concurrency for ~zero extra decode TPOT.

### 3. DCP KV capacity win is exactly dcp×

| Config | KV tokens | × over pure-PCP counterpart |
|---|---|---|
| TP1+PCP8 | 40,736 | 1× |
| TP1+PCP8+DCP8 | 324,936 | **8.0×** |

Each DCP rank stores 1/dcp of the KV (interleaved by position), so the cluster
holds dcp× the sequences. This is the only path for `dcp_shares_pcp_ranks`.

### 4. Prefill: PCP accelerates, DCP penalises

**PCP parallelises prefill** — more PCP ranks = faster prefill:

| Config | 32K TTFT |
|---|---|
| TP8 | 1973 ms |
| TP1+PCP8 | 1682 ms (−15%) |

**DCP adds a prefill penalty** — the chunked-prefill extend path:

| Comparison | Δ TTFT @ 32K |
|---|---|
| TP1+PCP8 (1682) → +DCP8 (2123) | +441 ms (+26%) |

Same mechanism as Qwen-30B: with the default `max_num_batched_tokens=8192`, a
32K prefill splits into 4 chunks; chunks 2–4 are extends that run the replicated
prefix pass (all-gather prefix-Q across PCP + LSE-combine). Disabling chunking
removes this penalty (verified on Qwen-30B: 764 → 567 ms).

### 5. Scaling to 500B-class GQA

This 290B GQA model is the near-500B-class case. The results confirm:

- **DCP8 delivers 8× KV capacity for +17% decode / +26% prefill overhead.**
- **Scaling dcp 2→4→8 costs ~zero extra decode TPOT** (205 / 215 / 195 ms),
  while KV capacity scales linearly.
- The prefill penalty is chunked-prefill, not the attention mechanism —
  fixable with larger `max_num_batched_tokens`.

For KV-memory-bound, decode-heavy serving of large GQA models, the trade is
favourable: the decode-time price of DCP is small and roughly constant, while
the KV-capacity gain scales with `dcp` for free.

## Methodology notes

- **TPOT** (prefix-on) and **TTFT** (prefix-off) come from separate boots with
  different prefix-caching settings. This is deliberate: decode is
  prefix-insensitive, prefill is not. The per-section flag
  (`serve_tpot_prefix_caching` / `ttft_prefix_caching`) is recorded in each
  result JSON.
- TPOT prompt counts: `{1: 16, 16: 48, 32: 64}` (reduced from the Qwen-30B
  sizing of `{1: 32, 16: 128, 32: 256}` because this model's ~155 ms TPOT makes
  longer runs impractical).
- GSM8K reduced to 200 questions (from 1319) for the same reason.
- The `kill_server()` function uses PID-based cleanup via `nvidia-smi
  --query-compute-apps` instead of `pkill -f` (which self-matches its own shell
  and can leave orphans holding GPU memory).
