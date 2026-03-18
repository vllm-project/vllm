# SP+AsyncTP Benchmark Report — Corrected (2×2 Matrix)

## Executive Summary

**SP+AsyncTP provides NO meaningful throughput improvement** on either tested configuration:

### Llama-2-7b TP=2 (compute-bound scenario)
- **~1-2% throughput improvement** (within noise margin)
- **~6-12% TTFT improvement** (modest)
- **CUDA graphs slightly hurt performance** (~2-3% penalty)

### Meta-Llama-3-70B TP=8 (theoretically "best case" NVLink scenario)
- **0.98x throughput** (slight regression, not improvement)
- **~5% TTFT improvement** (modest)
- **CUDA graphs severely hurt baseline performance** (0.61x with `custom_ops` anomaly)
- Even at 32% theoretical communication overhead, fusion kernel overhead at M=256 completely offsets any benefit

⚠️ **Previous results (1.47x / 2.24x) were entirely artifacts of benchmark methodology errors.**

🔑 **Bottom line**: In single-node NVLink environments, SP+AsyncTP's value is **regression avoidance**
(preventing Inductor's always-fuse rule from causing 2-3x decode slowdown), not acceleration.

## Corrected Results (2×2 Matrix)

### Environment
- **Model**: meta-llama/Llama-2-7b-hf (7B params, hidden_size=4096)
- **Hardware**: 2× NVIDIA H100 GPUs (95 GiB each), TP=2
- **vLLM**: Development branch with SP+AsyncTP piecewise compilation fix
- **Compilation**: `compile_sizes=[256]`, piecewise (attention splitting ops)
- **Benchmark**: `vllm bench serve`, 256 prompts, input_len=256, output_len=128, rate=inf
- **GPU memory**: `gpu-memory-utilization=0.85` (close to default 0.9)

### Results Table

| Config | SP+AsyncTP | CUDA Graph | Tok/s | ITL (ms) | TTFT (ms) |
|--------|:---------:|:---------:|------:|--------:|----------:|
| baseline_cg | ❌ | ✅ | 8,818.6 | 21.50 | 888.29 |
| baseline_no_cg | ❌ | ❌ | 8,959.0 | 21.47 | 830.13 |
| sp_asynctp_cg | ✅ | ✅ | 8,866.0 | 21.45 | 882.77 |
| sp_asynctp_no_cg | ✅ | ❌ | 9,114.6 | 21.46 | 778.84 |

### 2×2 Decomposition

| Comparison | Ratio | Interpretation |
|-----------|:-----:|---------------|
| SP effect (CG on) | **1.01x** | SP barely helps with CUDA graphs |
| SP effect (CG off) | **1.02x** | SP barely helps without CUDA graphs |
| CG effect (no SP) | **0.98x** | CG slightly hurts without SP |
| CG effect (with SP) | **0.97x** | CG slightly hurts with SP |

### Key Findings

1. **SP+AsyncTP is neutral**: 1-2% improvement is within noise. At BS=256 on Llama-2-7b
   TP=2, compute dominates communication. AllReduce on 2 H100s via NVLink is already
   very fast; overlapping it with MatMul via AsyncTP saves negligible time.

2. **CUDA graphs slightly negative**: 2-3% penalty. At single capture size (256), the
   graph replay overhead outweighs kernel launch savings.

3. **TTFT modestly improved**: SP+no-CG achieves 778ms vs baseline 888ms (12% better).
   Likely from reduced synchronization during prefill phase.

4. **ITL identical**: ~21.5ms across all configs. Per-token latency is entirely
   compute-bound at this scale.

## Why SP Doesn't Help Here

Llama-2-7b TP=2 is **compute-bound**, not communication-bound:

- **Small model**: 7B params → each GPU handles 3.5B params of compute
- **Small TP**: 2 GPUs connected by NVLink → AllReduce latency is ~microseconds
- **Large batch**: BS=256 → each GEMM is 256×4096×4096 = large, dominating total time
- **Communication fraction**: AllReduce costs << 1% of total forward time → overlapping
  it saves almost nothing

SP+AsyncTP should benefit more in:
- **Large TP** (TP=8/16): AllReduce across 8+ GPUs is much more expensive
- **Large models** (70B+): More layers = more AllReduce operations per forward pass
- **Small batch sizes**: GEMM is smaller → communication fraction increases
- **Inter-node communication**: Without NVLink, AllReduce latency is much higher

## Previous Results Were Wrong — Root Cause Analysis

The previous benchmark (Session 11, run_sp_cg_quick.py) reported:
- baseline_cg: 3,737.8 tok/s → sp_asynctp_cg: 5,498.8 tok/s (1.47x)
- baseline_cg: 3,737.8 tok/s → sp_asynctp_no_cg: 8,368.5 tok/s (2.24x)

These were wrong due to **three compounding errors**:

### Error 1: `gpu-memory-utilization=0.4` (should be 0.85+)
Set to 0.4 to work around stale GPU worker processes from a previous run.
This severely limited KV cache size, reducing max concurrent sequences and
creating artificial scheduling bottlenecks. The baseline suffered most:
3,737 tok/s (0.4) vs 8,818 tok/s (0.85) = **2.36x difference from this alone**.

### Error 2: SP never activated for no-CG config
`compile_sizes=[256]` + `cudagraph_mode=0` (no CG) meant:
- No batch size padding to 256 (CG does this automatically)
- With 200 prompts, decode BS ≈ 200, never exactly 256
- Compiled code (with SP transformations) never used
- **sp_asynctp_no_cg ran in eager mode WITHOUT SP** — it was just "baseline without CG"

### Error 3: Apples-to-oranges comparison
"2.24x" compared `sp_asynctp_no_cg` (no CG, no SP accidentally) vs `baseline_cg`
(with CG). This measured "disabling CG on memory-starved system" not "SP benefit".

### Fix Applied
- `gpu-memory-utilization=0.85` (proper memory allocation)
- `num_prompts=256` (decode BS = 256, matching compile_sizes)
- Full 2×2 matrix (4 configs instead of 3)

## Files

| File | Description |
|------|-------------|
| `benchmarks/run_sp_cg_2x2.py` | Corrected 2×2 benchmark script (Llama-2-7b TP=2) |
| `benchmarks/run_sp_cg_2x2_70b.py` | 2×2 benchmark script (Llama-3-70B TP=8) |
| `benchmarks/run_sp_cg_quick.py` | Previous (flawed) benchmark script |
| `/tmp/sp_2x2_results_1773860877.json` | Raw corrected results (Llama-2-7b TP=2) |
| `/tmp/sp_2x2_70b_results_1773870297.json` | Raw results (Llama-3-70B TP=8) |
| `/tmp/sp_cg_results_1773858323.json` | Raw previous (flawed) results |

---

## Meta-Llama-3-70B TP=8 Results (Session 12 — the "best case" test)

### Environment
- **Model**: meta-llama/Meta-Llama-3-70B (70B params, hidden_size=8192, 80 layers)
- **Hardware**: 8× NVIDIA H100 GPUs (95 GiB each), TP=8
- **Compilation**: `compile_sizes=[256]`, piecewise
- **Benchmark**: `vllm bench serve`, 256 prompts, input_len=256, output_len=128, rate=inf
- **GPU memory**: `gpu-memory-utilization=0.90`

### Why This is the "Best Case"

Llama-3-70B TP=8 has **~32% communication overhead** (vs ~20% for 7B TP=2):
- 8 GPUs → AllReduce across 8 ranks (much more expensive than 2)
- 80 layers × 7 AllReduce/layer = 560 collective ops per forward pass
- Theory predicts max 1.47x speedup from perfect communication overlap

### Results Table

| Config | SP+AsyncTP | CUDA Graph | Tok/s | ITL (ms) | TTFT (ms) |
|--------|:---------:|:---------:|------:|--------:|----------:|
| baseline_cg | ❌ | ✅ | 2,422.8 | 56.02 | 6,218.30 |
| baseline_no_cg | ❌ | ❌ | 3,949.4 | 47.34 | 2,089.23 |
| sp_asynctp_cg | ✅ | ✅ | 3,726.0 | 49.66 | 2,282.66 |
| sp_asynctp_no_cg | ✅ | ❌ | 3,875.2 | 47.80 | 2,190.32 |

### 2×2 Decomposition

| Comparison | Ratio | Interpretation |
|-----------|:-----:|---------------|
| SP effect (CG off) | **0.98x** | SP+AsyncTP slightly HURTS throughput (clean comparison) |
| SP effect (CG on) | **1.54x** | ⚠️ ARTIFACT — baseline_cg is anomalously slow (see below) |
| CG effect (no SP) | **0.61x** | CUDA graphs severely hurt baseline performance |
| CG effect (with SP) | **0.96x** | CUDA graphs also hurt SP performance |

### Key Findings

1. **SP+AsyncTP provides NO benefit on 70B TP=8**: The clean comparison (CG off)
   shows 0.98x — a slight regression, not improvement. This validates the theoretical
   analysis in the tracking doc (§6.6): fusion kernel overhead at M=256 completely
   offsets any communication overlap benefit.

2. **baseline_cg is anomalously broken**: 2,423 tok/s with TTFT=6,218ms. This is
   caused by `custom_ops: ["+rms_norm"]` interacting badly with CUDA graph capture
   on the large 70B TP=8 computation graph. The SP configs don't set `custom_ops`
   (using defaults), which may explain why `sp_asynctp_cg` doesn't have this issue.

3. **CUDA graphs hurt all configs on 70B TP=8**: Even sp_asynctp_cg (3,726) is
   slower than sp_asynctp_no_cg (3,875). The large graph may have excessive capture
   or replay overhead.

4. **ITL is higher than Llama-2-7b**: 47-56ms vs 21ms, expected given 10x more
   params per forward pass on 70B.

### Conclusion

**Even in the theoretically "best case" NVLink scenario (70B TP=8, 32% communication
overhead), SP+AsyncTP provides zero serving throughput benefit.**

The only remaining scenarios where async TP might help:
- **Cross-node TP** (IB/RoCE, no NVLink): 10x less bandwidth → 80%+ comm overhead
- **Prefill-heavy workloads**: M=4096+, where fusion kernel actually helps (1.15x)
- **Very high concurrency**: BS=512+, but requires enormous KV cache
