# Plan: Tune FlashInfer AllReduce Fusion Limits for Hopper TP=4/8

## Task Description

The FlashInfer fused allreduce+RMSNorm kernel in vLLM has extremely conservative size thresholds
for SM90 (Hopper) at TP=4 and TP=8. These thresholds determine up to what tensor size the fused
kernel is used instead of the separate allreduce + RMSNorm path. On 8x H200 GPUs with NVSwitch
interconnect, the fused kernel likely performs well at much larger sizes than the current limits
allow. This plan benchmarks systematically and proposes new, evidence-based thresholds.

**Current SM90 limits (conservative):**
```python
FI_ALLREDUCE_FUSION_MAX_SIZE_MB = {
    90: {2: 64, 4: 2, 8: 0.5},   # TP=4: 2MB, TP=8: 0.5MB
}
_FI_ALLREDUCE_ONE_SHOT_MAX_SIZES_MB = {
    90: {2: 32, 4: 2, 8: 0.5},
}
```

**Blackwell SM100 limits (reference — what we're targeting):**
```python
FI_ALLREDUCE_FUSION_MAX_SIZE_MB = {
    100: {2: 64, 4: 32, 8: 1},   # TP=4: 32MB, TP=8: 1MB
}
_FI_ALLREDUCE_ONE_SHOT_MAX_SIZES_MB = {
    100: {2: 32, 4: 4, 8: 1},
}
```

**Gap:** TP=4 is 16x lower on Hopper than Blackwell. TP=8 is 2x lower. H200 has NVSwitch with
900 GB/s bisection bandwidth — far above the threshold where 2MB and 0.5MB limits are justified.

## Objective

When complete:
1. Kernel-level benchmarks prove the crossover point where fused > unfused for TP=4 and TP=8 on H200
2. New SM90 thresholds are set based on measured data (not guesses)
3. End-to-end latency benchmarks with Qwen3-VL-2B (TP=4) and Qwen3-VL-8B (TP=8) confirm real model speedup
4. The OneShot thresholds are also tuned based on oneshot-vs-twoshot crossover data
5. All benchmark artifacts are archived with timestamps in a dedicated results directory
6. A clean PR branch exists in `../h200-flash-infer-allreduce-fusion` with the code change + benchmark evidence

## Problem Statement

The `FI_ALLREDUCE_FUSION_MAX_SIZE_MB` dict in `allreduce_rms_fusion.py:54-65` controls whether
vLLM uses FlashInfer's fused allreduce+RMSNorm kernel or falls back to separate operations.
When a tensor exceeds the threshold, the fusion pass marks itself as not applicable for that
compile range (via `is_applicable_for_range`), and every forward pass for those batch sizes
runs the slower unfused path.

**How the threshold propagates:**
1. `PassConfig.flashinfer_max_size(world_size)` reads the dict (line 154-169 of `compilation.py`)
2. `AllReduceFusionPass.__init__` converts MB to `max_token_num = max_size / (hidden_dim * element_size)`
3. `VllmConfig` inserts a compile range split point at `max_token_num`
4. For compile ranges where `end > max_token_num`, the fusion pass is skipped entirely

**Impact calculation for Qwen3-VL-8B at TP=8:**
- hidden_size = 4096, element_size = 2 (bfloat16)
- Current threshold: 0.5MB = 524,288 bytes
- max_token_num = 524,288 / (4096 * 2) = 64 tokens
- This means the fusion is ONLY active for batches of <=64 tokens — any decode batch >64 falls back

**Impact calculation for Qwen3-VL-2B at TP=4:**
- hidden_size = 2048, element_size = 2 (bfloat16)
- Current threshold: 2MB = 2,097,152 bytes
- max_token_num = 2,097,152 / (2048 * 2) = 512 tokens
- Better, but still misses large batch decode

## Solution Approach

### Phase 1: Kernel-Level Microbenchmarks
Run `benchmark_fused_collective.py` at TP=4 and TP=8 across a sweep of token counts (16 to 16384)
for both model hidden dims (2048 and 4096). This measures the fused vs unfused crossover point
directly, with both oneshot and twoshot modes.

### Phase 2: Determine New Thresholds
Analyze the crossover data. The new threshold = the largest tensor size where fused is still faster
than unfused, with a 5% safety margin.

### Phase 3: Apply Code Change
Modify `allreduce_rms_fusion.py` to update the SM90 dict entries for TP=4 and TP=8.
Also update `_FI_ALLREDUCE_ONE_SHOT_MAX_SIZES_MB` based on oneshot-vs-twoshot crossover.

### Phase 4: End-to-End Validation
Run `vllm bench latency` and `vllm bench throughput` with the real Qwen3-VL models at TP=4
and TP=8, comparing baseline (old thresholds) vs tuned (new thresholds) using the
`fi_allreduce_fusion_max_size_mb` config override.

### Phase 5: Correctness Verification
Run the existing allreduce fusion unit tests to ensure no regressions.

## Relevant Files

### Core Files to Modify
- `vllm/compilation/passes/fusion/allreduce_rms_fusion.py` — Lines 54-81: The `FI_ALLREDUCE_FUSION_MAX_SIZE_MB` and `_FI_ALLREDUCE_ONE_SHOT_MAX_SIZES_MB` dicts. **This is the ONLY production code change.**

### Config/Infrastructure (read-only reference)
- `vllm/config/compilation.py` — Lines 97-182: `PassConfig` class, `flashinfer_max_size()` method, `default_fi_allreduce_fusion_max_size_mb()`. Understand how thresholds propagate.
- `vllm/config/vllm.py` — Lines 105-121: `enable_allreduce_rms_fusion()` guard. Lines 1363-1378: compile range split point insertion.
- `vllm/compilation/passes/pass_manager.py` — Lines 122-123: Where `AllReduceFusionPass` is instantiated in the pass pipeline.

### Benchmarking Tools
- `benchmarks/kernels/benchmark_fused_collective.py` — The kernel microbenchmark (1114 lines). Uses CUDA graphs, compares fused vs unfused, supports oneshot/twoshot, outputs markdown.
- `vllm/benchmarks/latency.py` — End-to-end single-batch latency benchmark via `vllm bench latency`.
- `vllm/benchmarks/throughput.py` — End-to-end throughput benchmark via `vllm bench throughput`.

### Test Files
- `tests/compile/passes/distributed/test_fusion_all_reduce.py` — Unit test for pattern matching (TP=2, 326 lines).
- `tests/compile/fusions_e2e/test_tp2_ar_rms.py` — E2E fusion tests with llama3, qwen3 models.
- `tests/compile/fusions_e2e/common.py` — `Matches` namedtuple, log pattern regexes.
- `tests/compile/fusions_e2e/models.py` — Model definitions with expected match counts.

### Model Configs
- `/fsx/models/Qwen3-VL-2B-Instruct/config.json` — hidden_size=2048, 28 layers
- `/fsx/models/Qwen3-VL-8B-Instruct/config.json` — hidden_size=4096, 36 layers

### New Files to Create
- `../h200-flash-infer-allreduce-fusion/results/` — All benchmark results
- `../h200-flash-infer-allreduce-fusion/scripts/` — Custom benchmark runner scripts

## Model Architecture Reference

### Qwen3-VL-2B-Instruct (for TP=4 benchmarks)
| Parameter | Value |
|-----------|-------|
| hidden_size | 2048 |
| intermediate_size | 6144 |
| num_hidden_layers | 28 |
| head_dim | 128 |

**Allreduce tensor size per layer:** hidden_size * batch_size * element_size (bfloat16=2B)
- At 512 tokens: 2048 * 512 * 2 = 2 MB (current limit for TP=4)
- At 2048 tokens: 2048 * 2048 * 2 = 8 MB
- At 8192 tokens: 2048 * 8192 * 2 = 32 MB

### Qwen3-VL-8B-Instruct (for TP=8 benchmarks)
| Parameter | Value |
|-----------|-------|
| hidden_size | 4096 |
| intermediate_size | 12288 |
| num_hidden_layers | 36 |
| head_dim | 128 |

**Allreduce tensor size per layer:** hidden_size * batch_size * element_size
- At 64 tokens: 4096 * 64 * 2 = 0.5 MB (current limit for TP=8)
- At 256 tokens: 4096 * 256 * 2 = 2 MB
- At 1024 tokens: 4096 * 1024 * 2 = 8 MB

## Implementation Phases

### Phase 1: Foundation (Steps 1-5)
Create worktree, set up results directory structure, verify GPU topology, run sanity checks on
existing benchmark tooling.

### Phase 2: Kernel Microbenchmarks (Steps 6-10)
Run `benchmark_fused_collective.py` at TP=4 and TP=8 with fine-grained token sweeps covering
both model hidden dims. Capture oneshot vs twoshot crossover data.

### Phase 3: Analysis and Threshold Determination (Steps 11-13)
Analyze benchmark data to find the exact crossover points. Determine new thresholds with safety
margins. Document the reasoning.

### Phase 4: Code Change and Unit Tests (Steps 14-17)
Apply the code change. Run existing unit tests and e2e fusion tests.

### Phase 5: End-to-End Model Benchmarks (Steps 18-22)
Run real model benchmarks comparing baseline vs tuned thresholds. Capture latency/throughput
improvement evidence.

### Phase 6: Final Validation and PR Prep (Steps 23-25)
Aggregate all results, create summary report, prepare clean commit.

## Step by Step Tasks

IMPORTANT: Execute every step in order, top to bottom. **STOP and investigate if any validation fails before proceeding to the next step.**

---

### 1. Create Git Worktree and Branch

- Create the worktree from the current `main` branch:
  ```bash
  cd /workspace/vllm
  git fetch origin main
  git worktree add ../h200-flash-infer-allreduce-fusion -b h200-flashinfer-allreduce-fusion-tuning origin/main
  ```
- Change to the worktree directory for all subsequent work

**VALIDATION (all must pass):**
- [ ] `cd ../h200-flash-infer-allreduce-fusion && git branch --show-current` outputs `h200-flashinfer-allreduce-fusion-tuning`
- [ ] `git log --oneline -1 origin/main` matches `git log --oneline -1 HEAD` (worktree is at main HEAD)
- [ ] `ls vllm/compilation/passes/fusion/allreduce_rms_fusion.py` exists
- [ ] `python -c "import vllm; print(vllm.__version__)"` runs without error from the worktree

---

### 2. Create Results Directory Structure

- Create a timestamped results directory:
  ```
  ../h200-flash-infer-allreduce-fusion/results/
  ├── kernel_benchmarks/
  │   ├── tp4/
  │   └── tp8/
  ├── e2e_benchmarks/
  │   ├── baseline/
  │   └── tuned/
  ├── analysis/
  ├── test_results/
  └── summary/
  ```
- Create a `scripts/` directory for custom benchmark runners
- Create a `README.md` in the results dir documenting the experiment

**VALIDATION (all must pass):**
- [ ] `ls -la results/kernel_benchmarks/tp4 results/kernel_benchmarks/tp8 results/e2e_benchmarks/baseline results/e2e_benchmarks/tuned results/analysis results/test_results results/summary scripts/` — all directories exist
- [ ] `cat results/README.md` contains the experiment description

---

### 3. Verify GPU Topology and NVSwitch

- Run `nvidia-smi topo -m` to confirm NVSwitch interconnect
- Run `nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv` to verify 8x H200 SM9.0
- Save output to `results/gpu_topology.txt`

**VALIDATION (all must pass):**
- [ ] Output shows exactly 8 GPUs, all "NVIDIA H200"
- [ ] All GPUs show compute capability 9.0
- [ ] NVSwitch connections visible (NV12 or NV18 links between all GPU pairs, not PIX/PHB)
- [ ] `results/gpu_topology.txt` file exists and is non-empty
- [ ] All GPUs show ~143 GB memory (143771 MiB)

---

### 4. Verify FlashInfer Installation and Comm Module

- Check FlashInfer is importable and has the required comm functions:
  ```python
  python -c "
  import flashinfer.comm as fc
  assert hasattr(fc, 'allreduce_fusion'), 'Missing allreduce_fusion'
  assert hasattr(fc, 'create_allreduce_fusion_workspace'), 'Missing workspace creator'
  assert hasattr(fc, 'AllReduceFusionPattern'), 'Missing pattern enum'
  print('FlashInfer version:', getattr(fc, '__version__', 'unknown'))
  print('AllReduceFusionPattern members:', [x for x in dir(fc.AllReduceFusionPattern) if not x.startswith('_')])
  print('OK: All required FlashInfer comm functions present')
  "
  ```
- Save output to `results/flashinfer_check.txt`

**VALIDATION (all must pass):**
- [ ] Script prints "OK: All required FlashInfer comm functions present"
- [ ] `allreduce_fusion` and `create_allreduce_fusion_workspace` both exist
- [ ] `AllReduceFusionPattern` has `kARResidualRMSNorm` member
- [ ] No import errors

---

### 5. Verify Baseline Code State and Run Existing Fused Collective Benchmark (Sanity — TP=2)

- Run the existing benchmark at TP=2 (known-working config) to verify the benchmark tooling works:
  ```bash
  torchrun --nproc_per_node=2 benchmarks/kernels/benchmark_fused_collective.py \
    --num-tokens 128 512 1024 \
    --hidden-dim 4096 \
    --quant-modes none \
    --warmup 5 --trials 20 \
    --output-file results/kernel_benchmarks/sanity_tp2.md
  ```
- This is a SANITY check only — we expect fused to be faster at these sizes for TP=2

**VALIDATION (all must pass):**
- [ ] Command exits with code 0 (no crashes)
- [ ] `results/kernel_benchmarks/sanity_tp2.md` exists and contains markdown tables
- [ ] For ALL token counts (128, 512, 1024): FlashInfer fused oneshot time < standard allreduce time
- [ ] All timing values are finite (no `inf` values indicating failures)
- [ ] Timing values are in a reasonable range (0.001ms - 10ms per op, not 0 or absurdly large)

---

### 6. Run TP=4 Kernel Benchmark — Hidden Dim 2048 (Qwen3-VL-2B)

- Run comprehensive benchmark at TP=4 with the 2B model's hidden dim:
  ```bash
  torchrun --nproc_per_node=4 benchmarks/kernels/benchmark_fused_collective.py \
    --num-tokens 16 32 64 128 256 512 1024 2048 4096 8192 \
    --hidden-dim 2048 \
    --quant-modes none,fp8 \
    --warmup 10 --trials 40 \
    --output-file results/kernel_benchmarks/tp4/hidden2048_full.md
  ```
- The wide token range (16-8192) maps to tensor sizes from 0.0625MB to 32MB — this covers
  far beyond the current 2MB limit and approaches the Blackwell 32MB limit

**VALIDATION (all must pass):**
- [ ] Command exits with code 0
- [ ] `results/kernel_benchmarks/tp4/hidden2048_full.md` exists and has results for ALL 10 token counts
- [ ] Every row has timing data for both "standard" and "flashinfer_fused" variants
- [ ] Both `none` and `fp8` quant modes have results
- [ ] At small sizes (16, 32, 64 tokens = 0.0625-0.25MB): fused should be faster — confirm this
- [ ] Results include both oneshot and twoshot variants
- [ ] No `inf` values in any result

---

### 7. Run TP=4 Kernel Benchmark — Hidden Dim 4096 (Qwen3-VL-8B cross-reference)

- Run at TP=4 with the 8B model's hidden dim for completeness:
  ```bash
  torchrun --nproc_per_node=4 benchmarks/kernels/benchmark_fused_collective.py \
    --num-tokens 16 32 64 128 256 512 1024 2048 4096 \
    --hidden-dim 4096 \
    --quant-modes none,fp8 \
    --warmup 10 --trials 40 \
    --output-file results/kernel_benchmarks/tp4/hidden4096_full.md
  ```

**VALIDATION (all must pass):**
- [ ] Command exits with code 0
- [ ] `results/kernel_benchmarks/tp4/hidden4096_full.md` exists and has results for ALL 9 token counts
- [ ] Every row has timing data for both "standard" and "flashinfer_fused" variants
- [ ] Both `none` and `fp8` quant modes have results
- [ ] At small sizes (16, 32 tokens): fused should be faster — confirm this
- [ ] The crossover point (where fused becomes slower) can be identified or confirmed absent (fused always faster)
- [ ] No `inf` values

---

### 8. Run TP=8 Kernel Benchmark — Hidden Dim 4096 (Qwen3-VL-8B)

- Run at TP=8 with the 8B model's hidden dim (primary TP=8 workload):
  ```bash
  torchrun --nproc_per_node=8 benchmarks/kernels/benchmark_fused_collective.py \
    --num-tokens 16 32 64 128 256 512 1024 2048 4096 8192 \
    --hidden-dim 4096 \
    --quant-modes none,fp8 \
    --warmup 10 --trials 40 \
    --output-file results/kernel_benchmarks/tp8/hidden4096_full.md
  ```
- Token range 16-8192 maps to 0.125MB-64MB for hidden_dim=4096

**VALIDATION (all must pass):**
- [ ] Command exits with code 0
- [ ] `results/kernel_benchmarks/tp8/hidden4096_full.md` exists and has results for ALL 10 token counts
- [ ] ALL timing values are finite (no `inf`)
- [ ] At 64 tokens (0.5MB = current limit): fused should still be faster — confirm
- [ ] At 128 tokens (1MB = Blackwell TP=8 limit): check if fused is still faster
- [ ] At 256+ tokens: identify where crossover occurs (or if fused is always faster up to 8192)
- [ ] Both oneshot and twoshot results present
- [ ] Both `none` and `fp8` results present

---

### 9. Run TP=8 Kernel Benchmark — Hidden Dim 2048 (Qwen3-VL-2B cross-reference)

- Cross-reference with smaller hidden dim at TP=8:
  ```bash
  torchrun --nproc_per_node=8 benchmarks/kernels/benchmark_fused_collective.py \
    --num-tokens 16 32 64 128 256 512 1024 2048 4096 8192 \
    --hidden-dim 2048 \
    --quant-modes none,fp8 \
    --warmup 10 --trials 40 \
    --output-file results/kernel_benchmarks/tp8/hidden2048_full.md
  ```

**VALIDATION (all must pass):**
- [ ] Command exits with code 0
- [ ] `results/kernel_benchmarks/tp8/hidden2048_full.md` exists and has results for ALL 10 token counts
- [ ] ALL timing values are finite
- [ ] Crossover analysis: identify the token count where fused becomes slower than unfused
- [ ] Results are consistent with the hidden4096 data (crossover at similar tensor MB size, not token count)
- [ ] Both oneshot and twoshot results present

---

### 10. Run Targeted Re-benchmarks Around Crossover Points

- Based on Steps 6-9, identify the approximate crossover points for each (TP, hidden_dim) combo
- Re-run with finer granularity around each crossover point. For example, if TP=4/hidden=2048
  crosses over between 4096-8192 tokens, run:
  ```bash
  torchrun --nproc_per_node=4 benchmarks/kernels/benchmark_fused_collective.py \
    --num-tokens 4096 5120 6144 7168 8192 \
    --hidden-dim 2048 \
    --quant-modes none,fp8 \
    --warmup 10 --trials 40 \
    --output-file results/kernel_benchmarks/tp4/hidden2048_crossover_detail.md
  ```
- Repeat for each configuration where the crossover was in the measured range
- If fused was faster at ALL sizes tested (no crossover found), document this and note
  the maximum tested size as the safe threshold

**VALIDATION (all must pass):**
- [ ] For each (TP, hidden_dim) combo, there is EITHER:
  - A crossover detail file with fine-grained results, OR
  - A documented note that fused was faster at ALL tested sizes
- [ ] Crossover points are identified to within 1024-token precision (or finer)
- [ ] Results files saved in appropriate `results/kernel_benchmarks/tpN/` directories
- [ ] The crossover point in MB is consistent across different hidden_dim values for the same TP
  (i.e., the threshold is tensor-size-dependent, not token-count-dependent)

---

### 11. Create Crossover Analysis Script

- Write a Python script `scripts/analyze_crossover.py` that:
  1. Reads all markdown benchmark result files from `results/kernel_benchmarks/`
  2. Parses the tables to extract (num_tokens, hidden_dim, fused_time, unfused_time, speedup)
  3. For each (TP, quant_mode, oneshot/twoshot) combo, finds the crossover token count
  4. Converts crossover token counts to MB thresholds
  5. Applies a 10% safety margin (reduce threshold by 10% below crossover)
  6. Outputs a summary JSON and a human-readable markdown report
  7. Outputs the recommended new values for `FI_ALLREDUCE_FUSION_MAX_SIZE_MB` and
     `_FI_ALLREDUCE_ONE_SHOT_MAX_SIZES_MB`

**VALIDATION (all must pass):**
- [ ] Script runs without error: `python scripts/analyze_crossover.py`
- [ ] Outputs `results/analysis/crossover_analysis.json` with structured data
- [ ] Outputs `results/analysis/crossover_analysis.md` with human-readable summary
- [ ] The recommended thresholds are:
  - Greater than the current SM90 values (2MB for TP=4, 0.5MB for TP=8) — must improve
  - Less than or equal to Blackwell values (32MB for TP=4, 1MB for TP=8) — sanity upper bound (can exceed if data supports it)
  - Supported by at least 2 independent data points (different hidden dims showing consistent crossover)
- [ ] The OneShot threshold recommendations are <= the main fusion threshold (OneShot is a subset)
- [ ] The safety margin is documented and justified

---

### 12. Peer-Review the Recommendations

- Read `results/analysis/crossover_analysis.md` carefully
- Verify the recommended values against the raw benchmark data:
  - For each recommended threshold T, confirm that the raw data shows fused faster at T
  - For each recommended threshold T, confirm that the raw data shows fused slower (or within margin) at 1.2*T
- Document any anomalies (e.g., non-monotonic speedup curves, outlier measurements)

**VALIDATION (all must pass):**
- [ ] For EVERY recommended threshold: there exists raw data showing fused is faster at that size
- [ ] For EVERY recommended threshold: the 10% safety margin is actually applied
- [ ] No recommended threshold is lower than the current value (we only raise limits, never lower)
- [ ] The recommendations are consistent: TP=4 threshold > TP=8 threshold (larger world size = more communication overhead)
- [ ] If any anomalies were found, they are documented in `results/analysis/anomalies.md`

---

### 13. Document Threshold Decision

- Write `results/analysis/threshold_decision.md` summarizing:
  - The old values and why they were conservative
  - The new values and the evidence supporting them
  - The safety margin applied
  - Which models and configurations were tested
  - Caveats (e.g., "tested on H200 with NVSwitch; older Hopper systems without NVSwitch may differ")

**VALIDATION (all must pass):**
- [ ] `results/analysis/threshold_decision.md` exists and is >500 bytes
- [ ] Contains a clear before/after comparison table
- [ ] References specific benchmark result files for every claim
- [ ] Caveats section is present and addresses NVSwitch dependency

---

### 14. Apply the Code Change to `allreduce_rms_fusion.py`

- Modify `FI_ALLREDUCE_FUSION_MAX_SIZE_MB` dict in `allreduce_rms_fusion.py:54-65`:
  - Update `90: {4: NEW_TP4_VALUE, 8: NEW_TP8_VALUE}` (keep TP=2 at 64)
- Modify `_FI_ALLREDUCE_ONE_SHOT_MAX_SIZES_MB` dict in `allreduce_rms_fusion.py:70-81`:
  - Update `90: {4: NEW_ONESHOT_TP4, 8: NEW_ONESHOT_TP8}` (keep TP=2 at 32)
- **DO NOT** change the SM100 (Blackwell) values
- Add a brief comment explaining the H200 benchmarking origin

**VALIDATION (all must pass):**
- [ ] `git diff vllm/compilation/passes/fusion/allreduce_rms_fusion.py` shows ONLY changes to lines 54-81
- [ ] SM100 dict entries are UNCHANGED (verify with grep)
- [ ] TP=2 values for SM90 are UNCHANGED (still 64MB and 32MB)
- [ ] New TP=4 value is > 2 (old) and matches the analysis recommendation
- [ ] New TP=8 value is > 0.5 (old) and matches the analysis recommendation
- [ ] OneShot TP=4 value <= main fusion TP=4 value
- [ ] OneShot TP=8 value <= main fusion TP=8 value
- [ ] `python -c "from vllm.compilation.passes.fusion.allreduce_rms_fusion import FI_ALLREDUCE_FUSION_MAX_SIZE_MB; print(FI_ALLREDUCE_FUSION_MAX_SIZE_MB)"` outputs the new values correctly
- [ ] `python -c "from vllm.compilation.passes.fusion.allreduce_rms_fusion import _FI_ALLREDUCE_ONE_SHOT_MAX_SIZES_MB; print(_FI_ALLREDUCE_ONE_SHOT_MAX_SIZES_MB)"` outputs the new values
- [ ] File still compiles: `python -m py_compile vllm/compilation/passes/fusion/allreduce_rms_fusion.py`

---

### 15. Update PassConfig Docstring

- Update the docstring in `vllm/config/compilation.py:131-148` where `fi_allreduce_fusion_max_size_mb`
  is documented with the old default values. The inline dict comment should reflect the new SM90 values.

**VALIDATION (all must pass):**
- [ ] The docstring in `compilation.py` now shows the updated SM90 dict values
- [ ] The docstring for SM100 is UNCHANGED
- [ ] `python -m py_compile vllm/config/compilation.py` succeeds
- [ ] `git diff vllm/config/compilation.py` shows ONLY docstring changes (no logic changes)

---

### 16. Run Unit Tests — Allreduce Fusion Pattern Matching

- Run the allreduce fusion pattern matching test:
  ```bash
  torchrun --nproc_per_node=2 -m pytest tests/compile/passes/distributed/test_fusion_all_reduce.py -v --timeout=120 2>&1 | tee results/test_results/unit_test_pattern_matching.log
  ```
- This test verifies that the AllReduceFusionPass correctly matches and replaces patterns.
  It does NOT depend on the threshold values (it tests pattern matching at small sizes).

**VALIDATION (all must pass):**
- [ ] All tests pass (exit code 0)
- [ ] `results/test_results/unit_test_pattern_matching.log` shows "passed" for all test cases
- [ ] No warnings about "AllReduce fusion pass is disabled" in the output
- [ ] Pattern match count is correct (typically 4 per test model)

---

### 17. Run E2E Fusion Tests

- Run the end-to-end fusion tests at TP=2:
  ```bash
  torchrun --nproc_per_node=2 -m pytest tests/compile/fusions_e2e/test_tp2_ar_rms.py -v -x --timeout=300 2>&1 | tee results/test_results/e2e_fusion_test.log
  ```
- These tests verify that the fusion pass correctly produces numerically equivalent results
  when fused vs unfused. They use dummy weights and small models.

**VALIDATION (all must pass):**
- [ ] All tests pass (exit code 0)
- [ ] `results/test_results/e2e_fusion_test.log` shows "passed" for all test cases
- [ ] Log contains "Replaced N patterns" messages confirming fusion was applied
- [ ] No CUDA errors or OOM errors

---

### 18. Baseline E2E Latency Benchmark — Qwen3-VL-2B at TP=4

- Run latency benchmark with the OLD threshold (override to exactly 2MB to simulate baseline):
  ```bash
  vllm bench latency \
    --model /fsx/models/Qwen3-VL-2B-Instruct \
    --tensor-parallel-size 4 \
    --input-len 128 --output-len 32 \
    --batch-size 1 4 16 64 256 1024 \
    --num-iters-warmup 10 --num-iters 30 \
    --trust-remote-code \
    --compilation-config '{"pass_config": {"fi_allreduce_fusion_max_size_mb": 2.0, "fuse_allreduce_rms": true}}' \
    --output-json results/e2e_benchmarks/baseline/qwen3vl_2b_tp4_latency.json \
    2>&1 | tee results/e2e_benchmarks/baseline/qwen3vl_2b_tp4_latency.log
  ```
  NOTE: `vllm bench latency` may not support multiple batch sizes in one invocation. If so,
  run separately for each batch size: 1, 16, 64, 256, 1024. Save each to a separate JSON:
  `qwen3vl_2b_tp4_latency_bs{N}.json`

**VALIDATION (all must pass):**
- [ ] Command(s) complete without error
- [ ] JSON result files exist for each batch size tested
- [ ] Latency values are in a plausible range (not 0, not absurdly large)
- [ ] Log files show the model loaded successfully on 4 GPUs
- [ ] Log shows "Flashinfer max size: 2 MB" (confirming the 2MB override was applied)
- [ ] For batch_size=1024: check if fusion was skipped (it should be, at 2MB threshold: 2MB/(2048*2)=512 tokens, so bs=1024 > 512)

---

### 19. Tuned E2E Latency Benchmark — Qwen3-VL-2B at TP=4

- Run the same benchmark with the NEW threshold:
  ```bash
  vllm bench latency \
    --model /fsx/models/Qwen3-VL-2B-Instruct \
    --tensor-parallel-size 4 \
    --input-len 128 --output-len 32 \
    --batch-size 1 4 16 64 256 1024 \
    --num-iters-warmup 10 --num-iters 30 \
    --trust-remote-code \
    --compilation-config '{"pass_config": {"fi_allreduce_fusion_max_size_mb": NEW_VALUE, "fuse_allreduce_rms": true}}' \
    --output-json results/e2e_benchmarks/tuned/qwen3vl_2b_tp4_latency.json \
    2>&1 | tee results/e2e_benchmarks/tuned/qwen3vl_2b_tp4_latency.log
  ```
  Use the same batch sizes as Step 18 for direct comparison.

**VALIDATION (all must pass):**
- [ ] Command(s) complete without error
- [ ] JSON result files exist for each batch size tested
- [ ] Log shows the NEW threshold value in "Flashinfer max size" message
- [ ] For small batch sizes (1, 16): latency should be approximately equal to baseline (fusion was already active)
- [ ] For large batch sizes (256, 1024): latency should be EQUAL OR BETTER than baseline (fusion now active where it wasn't before)
- [ ] NO batch size shows significant latency REGRESSION (>5% slower than baseline)
- [ ] If any regression is detected: STOP and investigate before proceeding

---

### 20. Baseline E2E Latency Benchmark — Qwen3-VL-8B at TP=8

- Run latency benchmark with the OLD threshold (override to 0.5MB):
  ```bash
  vllm bench latency \
    --model /fsx/models/Qwen3-VL-8B-Instruct \
    --tensor-parallel-size 8 \
    --input-len 128 --output-len 32 \
    --batch-size 1 4 16 64 128 256 \
    --num-iters-warmup 10 --num-iters 30 \
    --trust-remote-code \
    --compilation-config '{"pass_config": {"fi_allreduce_fusion_max_size_mb": 0.5, "fuse_allreduce_rms": true}}' \
    --output-json results/e2e_benchmarks/baseline/qwen3vl_8b_tp8_latency.json \
    2>&1 | tee results/e2e_benchmarks/baseline/qwen3vl_8b_tp8_latency.log
  ```

**VALIDATION (all must pass):**
- [ ] Command(s) complete without error
- [ ] JSON result files exist for each batch size tested
- [ ] Log shows "Flashinfer max size: 0 MB" or "Flashinfer max size: 0.5 MB" (confirming 0.5MB override)
- [ ] At 0.5MB: max_token_num = 524288/(4096*2) = 64 tokens. So bs=64 should be the cutoff.
- [ ] For batch_size=1, 4, 16: fusion should be active
- [ ] For batch_size=128, 256: fusion should NOT be active (confirming the conservative limit)
- [ ] Latency values are plausible

---

### 21. Tuned E2E Latency Benchmark — Qwen3-VL-8B at TP=8

- Run with the NEW threshold:
  ```bash
  vllm bench latency \
    --model /fsx/models/Qwen3-VL-8B-Instruct \
    --tensor-parallel-size 8 \
    --input-len 128 --output-len 32 \
    --batch-size 1 4 16 64 128 256 \
    --num-iters-warmup 10 --num-iters 30 \
    --trust-remote-code \
    --compilation-config '{"pass_config": {"fi_allreduce_fusion_max_size_mb": NEW_VALUE, "fuse_allreduce_rms": true}}' \
    --output-json results/e2e_benchmarks/tuned/qwen3vl_8b_tp8_latency.json \
    2>&1 | tee results/e2e_benchmarks/tuned/qwen3vl_8b_tp8_latency.log
  ```

**VALIDATION (all must pass):**
- [ ] Command(s) complete without error
- [ ] Log shows the NEW threshold in "Flashinfer max size" message
- [ ] For small batch sizes (1, 4, 16): latency approximately equal to baseline
- [ ] For medium batch sizes (64, 128): latency EQUAL OR BETTER than baseline
- [ ] For large batch sizes (256): latency EQUAL OR BETTER than baseline
- [ ] NO batch size shows >5% latency regression
- [ ] If any regression: STOP and reduce the threshold until no regression

---

### 22. E2E Throughput Benchmark Comparison

- Run throughput benchmarks for both models at both configs to measure real-world impact:
  ```bash
  # Baseline TP=4
  vllm bench throughput \
    --model /fsx/models/Qwen3-VL-2B-Instruct \
    --tensor-parallel-size 4 \
    --dataset-name random --input-len 512 --output-len 128 \
    --num-prompts 200 --trust-remote-code \
    --compilation-config '{"pass_config": {"fi_allreduce_fusion_max_size_mb": 2.0, "fuse_allreduce_rms": true}}' \
    --output-json results/e2e_benchmarks/baseline/qwen3vl_2b_tp4_throughput.json \
    2>&1 | tee results/e2e_benchmarks/baseline/qwen3vl_2b_tp4_throughput.log

  # Tuned TP=4
  vllm bench throughput \
    --model /fsx/models/Qwen3-VL-2B-Instruct \
    --tensor-parallel-size 4 \
    --dataset-name random --input-len 512 --output-len 128 \
    --num-prompts 200 --trust-remote-code \
    --compilation-config '{"pass_config": {"fi_allreduce_fusion_max_size_mb": NEW_VALUE, "fuse_allreduce_rms": true}}' \
    --output-json results/e2e_benchmarks/tuned/qwen3vl_2b_tp4_throughput.json \
    2>&1 | tee results/e2e_benchmarks/tuned/qwen3vl_2b_tp4_throughput.log
  ```
  Repeat for TP=8 with the 8B model.

**VALIDATION (all must pass):**
- [ ] All 4 benchmark runs (2 models x 2 configs) complete without error
- [ ] JSON result files exist for all runs
- [ ] For TP=4: tuned throughput >= baseline throughput (tokens/s)
- [ ] For TP=8: tuned throughput >= baseline throughput (tokens/s)
- [ ] No throughput regression >2% in any configuration
- [ ] Throughput improvement is documented as a percentage

---

### 23. Create Comparison Report

- Write `scripts/create_comparison_report.py` that:
  1. Reads all baseline and tuned JSON results
  2. Computes latency improvement per batch size
  3. Computes throughput improvement
  4. Generates a comprehensive markdown report with tables and analysis
- Run the script and save to `results/summary/comparison_report.md`

**VALIDATION (all must pass):**
- [ ] Script runs without error
- [ ] `results/summary/comparison_report.md` exists and is >1000 bytes
- [ ] Report contains:
  - [ ] Kernel-level crossover analysis table (from Step 11)
  - [ ] E2E latency comparison table for BOTH models
  - [ ] E2E throughput comparison for BOTH models
  - [ ] Clear statement of the old vs new threshold values
  - [ ] Percentage improvement for each configuration
  - [ ] A "Regressions" section (should state "None detected" or list any)
- [ ] ALL improvement percentages are >= 0% (no regressions documented without explanation)

---

### 24. Create Clean Commit

- Stage ONLY the production code changes:
  ```bash
  git add vllm/compilation/passes/fusion/allreduce_rms_fusion.py
  git add vllm/config/compilation.py
  ```
- Do NOT stage benchmark scripts, results, or analysis files (those stay local for review)
- Create the commit with a descriptive message referencing the benchmark evidence

**VALIDATION (all must pass):**
- [ ] `git diff --staged --stat` shows EXACTLY 2 files changed
- [ ] `git diff --staged vllm/compilation/passes/fusion/allreduce_rms_fusion.py` shows only threshold dict changes + comments
- [ ] `git diff --staged vllm/config/compilation.py` shows only docstring changes
- [ ] No test files, benchmark scripts, or result files are staged
- [ ] `git status` shows the results/ and scripts/ dirs as untracked (expected)
- [ ] The commit message includes:
  - [ ] The old and new threshold values
  - [ ] A mention of H200/NVSwitch benchmarking
  - [ ] Reference to the benchmark methodology

---

### 25. Final Validation — Complete Check

- Verify the entire results directory is intact and complete:
  ```
  results/
  ├── gpu_topology.txt                        # Step 3
  ├── flashinfer_check.txt                    # Step 4
  ├── kernel_benchmarks/
  │   ├── sanity_tp2.md                       # Step 5
  │   ├── tp4/
  │   │   ├── hidden2048_full.md              # Step 6
  │   │   ├── hidden4096_full.md              # Step 7
  │   │   └── hidden2048_crossover_detail.md  # Step 10 (if applicable)
  │   └── tp8/
  │       ├── hidden4096_full.md              # Step 8
  │       ├── hidden2048_full.md              # Step 9
  │       └── hidden4096_crossover_detail.md  # Step 10 (if applicable)
  ├── analysis/
  │   ├── crossover_analysis.json             # Step 11
  │   ├── crossover_analysis.md               # Step 11
  │   └── threshold_decision.md               # Step 13
  ├── e2e_benchmarks/
  │   ├── baseline/
  │   │   ├── qwen3vl_2b_tp4_latency*.json   # Step 18
  │   │   ├── qwen3vl_8b_tp8_latency*.json   # Step 20
  │   │   ├── qwen3vl_2b_tp4_throughput.json  # Step 22
  │   │   └── qwen3vl_8b_tp8_throughput.json  # Step 22
  │   └── tuned/
  │       ├── qwen3vl_2b_tp4_latency*.json   # Step 19
  │       ├── qwen3vl_8b_tp8_latency*.json   # Step 21
  │       ├── qwen3vl_2b_tp4_throughput.json  # Step 22
  │       └── qwen3vl_8b_tp8_throughput.json  # Step 22
  ├── test_results/
  │   ├── unit_test_pattern_matching.log      # Step 16
  │   └── e2e_fusion_test.log                 # Step 17
  └── summary/
      └── comparison_report.md                # Step 23
  ```

**VALIDATION (all must pass):**
- [ ] Every file listed above exists and is non-empty
- [ ] `git log --oneline -1` shows the commit from Step 24
- [ ] `git diff HEAD~1..HEAD --stat` confirms only 2 production files changed
- [ ] Re-run the import check: `python -c "from vllm.compilation.passes.fusion.allreduce_rms_fusion import FI_ALLREDUCE_FUSION_MAX_SIZE_MB as d; assert d[90][4] > 2; assert d[90][8] > 0.5; print('New thresholds verified:', d[90])"` — passes
- [ ] Re-run the OneShot check: `python -c "from vllm.compilation.passes.fusion.allreduce_rms_fusion import _FI_ALLREDUCE_ONE_SHOT_MAX_SIZES_MB as d; assert d[90][4] >= 2; assert d[90][8] >= 0.5; print('New oneshot thresholds verified:', d[90])"` — passes
- [ ] No uncommitted changes to production code: `git diff vllm/` is empty
- [ ] Branch is ready for manual push and PR creation

---

## Testing Strategy

### Correctness Testing (Steps 16-17)
1. **Unit test: Pattern matching** — Confirms the fusion pass still correctly identifies and replaces
   allreduce+RMSNorm patterns in the computation graph. This test is threshold-independent.
2. **E2E fusion test** — Confirms that the fused kernel produces numerically equivalent output to
   the unfused path. Tests with llama3 and qwen3 models at TP=2.

### Performance Testing (Steps 6-10, 18-22)
1. **Kernel microbenchmarks** — Direct comparison of fused vs unfused at every relevant tensor size.
   Uses CUDA graphs for accurate measurement. Tests both oneshot and twoshot modes.
2. **E2E latency benchmarks** — Measures real model inference latency across batch sizes.
   Direct A/B comparison of baseline vs tuned thresholds.
3. **E2E throughput benchmarks** — Measures sustained throughput under load.

### Regression Prevention
- Every E2E step validates NO regression >5% at any batch size
- If regression detected: STOP, reduce threshold, re-benchmark
- The safety margin (10% below crossover) provides additional buffer

## Acceptance Criteria

1. **Kernel evidence**: For BOTH TP=4 and TP=8, benchmark data shows the fused kernel is faster than
   unfused at tensor sizes above the current SM90 limits
2. **No regression**: No E2E latency regression >5% at ANY batch size for ANY model
3. **Measurable improvement**: At least one (model, batch_size) configuration shows >2% latency improvement
4. **Code change minimal**: Only `allreduce_rms_fusion.py` (threshold dicts) and `compilation.py` (docstring) are modified
5. **Tests pass**: All existing unit tests and E2E fusion tests pass
6. **Results traceable**: Every claim in the comparison report links back to a specific result file
7. **OneShot tuned**: OneShot thresholds are also updated with evidence
8. **Clean commit**: Single commit with descriptive message, no unrelated changes

## Validation Commands

Execute these commands from the worktree root to validate completion:

```bash
# 1. Verify the code compiles
python -m py_compile vllm/compilation/passes/fusion/allreduce_rms_fusion.py
python -m py_compile vllm/config/compilation.py

# 2. Verify new thresholds are higher than old
python -c "
from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
    FI_ALLREDUCE_FUSION_MAX_SIZE_MB as main_dict,
    _FI_ALLREDUCE_ONE_SHOT_MAX_SIZES_MB as oneshot_dict,
)
# SM90 TP=4 increased from 2
assert main_dict[90][4] > 2, f'TP=4 main threshold not increased: {main_dict[90][4]}'
# SM90 TP=8 increased from 0.5
assert main_dict[90][8] > 0.5, f'TP=8 main threshold not increased: {main_dict[90][8]}'
# SM90 TP=2 unchanged
assert main_dict[90][2] == 64, f'TP=2 main threshold changed: {main_dict[90][2]}'
# SM100 unchanged
assert main_dict[100] == {2: 64, 4: 32, 8: 1}, f'SM100 changed: {main_dict[100]}'
# OneShot <= main
assert oneshot_dict[90][4] <= main_dict[90][4], 'OneShot TP=4 > main'
assert oneshot_dict[90][8] <= main_dict[90][8], 'OneShot TP=8 > main'
print('ALL THRESHOLD VALIDATION PASSED')
print(f'SM90 main: {main_dict[90]}')
print(f'SM90 oneshot: {oneshot_dict[90]}')
"

# 3. Verify git state
git diff --stat HEAD~1..HEAD  # Should show exactly 2 files

# 4. Verify results exist
ls results/summary/comparison_report.md
ls results/analysis/threshold_decision.md
ls results/kernel_benchmarks/tp4/hidden2048_full.md
ls results/kernel_benchmarks/tp8/hidden4096_full.md

# 5. Run pattern matching test
torchrun --nproc_per_node=2 -m pytest tests/compile/passes/distributed/test_fusion_all_reduce.py -v --timeout=120
```

## Notes

- **NVSwitch dependency**: The H200 machines in this setup have NVSwitch providing high bisection
  bandwidth. Hopper systems WITHOUT NVSwitch (e.g., PCIe-only H100) may not benefit from raising
  these thresholds as aggressively. The PR description should note this.

- **One-size override limitation**: The `fi_allreduce_fusion_max_size_mb` config field overrides
  the threshold for ALL world sizes (it's a single float, not per-world-size). This means users
  can't fine-tune per-TP. Our change to the default dict is the only way to set per-TP defaults.

- **FlashInfer OneShot restriction**: OneShot mode has a hard limit of 64MB/world_size built into
  FlashInfer itself. For TP=8, that's 8MB max. For TP=4, that's 16MB max. Our OneShot thresholds
  must respect this.

- **Compile range split**: When the threshold is raised, the compile range split point moves to a
  higher token count. This means more batch sizes use the fused kernel. If the fused kernel is
  faster at those sizes (which our benchmarks confirm), this is pure upside.

- **No new dependencies**: This change modifies only two dict literals and a docstring. No new
  imports, no new files, no new dependencies.

- **Benchmark runtime estimate**: Steps 6-10 (kernel benchmarks) take ~5-10 min each. Steps 18-22
  (E2E benchmarks) take ~10-20 min each due to model loading and warmup. Total estimated wall
  time: 3-5 hours including analysis and report writing.
