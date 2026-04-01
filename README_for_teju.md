# vLLM KV Cache Tiering — PSC Bridges-2 Validation
**For:** Teju  
**By:** Rishi Nagaraj  
**Date:** April 1, 2026  
**vLLM Version:** `0.16.0rc2.dev369+g003800536.d20260330.cu124`

---

## Overview

This documents the **complete end-to-end validation** of vLLM's KV Cache CPU offloading and tiering features on PSC Bridges-2 (`v100-32` GPUs, `compute_70`). This includes every patch made to get the system working, all test results, and the current benchmarking status.

---

## Cluster Environment

| Setting | Value |
|---|---|
| Cluster | PSC Bridges-2 (`bridges2.psc.edu`) |
| GPU | Tesla V100-SXM2-32GB (`compute_70`) |
| CUDA | `12.4.0` (via `module load cuda/12.4.0`) |
| GCC | `10.2.0` (via `module load gcc/10.2.0`) |
| Python | 3.12.12 (CPython) |
| vLLM | `0.16.0rc2.dev369+g003800536.d20260329.cu124` |
| torch | `2.5.1+cu124` → later auto-upgraded to `2.10.0` by vLLM deps |

---

## Patches Applied to the Codebase

### 1. `csrc/torch_bindings.cpp` — Hopper Kernel Guard

**Problem:** The `dsv3_fused_a_gemm` kernel (used in DeepSeek-V3 MoE) was unconditionally registered in `torch_bindings.cpp`. On V100 (`compute_70`), this caused:
```
ImportError: undefined symbol: dsv3_fused_a_gemm_impl
```

**Fix:** Commented out the registration block at lines 243–245:
```cpp
// Removed: TORCH_LIBRARY_FRAGMENT(vllm, m) {
//   m.def("dsv3_fused_a_gemm(...)", ...);
// }
```

This is safe — V100 doesn't support FP8 MoE, so this kernel is unused.

---

### 2. `scripts/psc_submit.sh` — Environment and Quota Management

Three critical environment variables were injected to avoid PSC home-directory quota crashes (`Errno 122`):

```bash
export HF_TOKEN="..."                   # HuggingFace auth for gated models
export HF_HOME="/jet/home/.../workspace/vllm/hf_cache"   # Move model downloads out of ~/.cache
export TRITON_CACHE_DIR="/jet/home/.../workspace/vllm/triton_cache"  # Move torch.compile PTX cache
export XDG_CACHE_HOME="/jet/home/.../workspace/vllm/xdg_cache"      # Move general cache
```

Without these, PyTorch's Triton compiler caches compiled GPU kernels to `~/.triton/cache` and crashes with Disk Quota Exceeded mid-run.

---

### 3. `scripts/gcp_setup_and_test.sh` — KVTransferConfig API Migration

The upstream vLLM `KVTransferConfig` Pydantic schema changed significantly:

| Old (broken) | New (fixed) |
|---|---|
| `kv_offloading_spec: { cpu: { eviction_policy: 'lru' } }` | `kv_connector_extra_config: { eviction_policy: 'lru', cpu_bytes_to_use: 500e6 }` |
| `kv_role` not required | `kv_role: 'kv_both'` now **mandatory** (Pydantic validator) |

All four offline inference phases (5–8) were patched to use the new flat schema with `kv_role: 'kv_both'`.

---

### 4. `kv_cache_tiering/benchmarks/benchmark.py` — Batched Inference

**Problem:** The original benchmark submitted prompts one-at-a-time in a sequential loop. This means the GPU KV cache is always empty at the start of each request — no concurrent pressure, no evictions ever fired.

**Fix:** Changed to submit all prompts in a **single batched `llm.generate(all_prompts, ...)`** call:
```python
# OLD: sequential, never fills cache
for prompt in prompts:
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)

# NEW: batched, all requests compete for KV blocks simultaneously
outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
```

This forces vLLM's scheduler to run all requests concurrently, saturating the KV pool and triggering real evictions to CPU.

---

## Test Results

### Phases 1–4: Unit Tests, Import Smoke Test, Score Estimator GPU Test

All passed. Run: **March 29, 2026, 23:21 UTC**.

Key confirmation:
```
Phase 3: Import Smoke Test — PASSED
Phase 4: Score Estimator GPU Test — PASSED
  req-1: 2 blocks, scores=[64.096, 64.234]
  req-2: 2 blocks, scores=[64.163, 63.749]
```

The attention score computation Triton kernels are functional on V100.

---

### Phases 5–8: Offline Inference (LRU / Attention / Hybrid / Prefetching)

All passed. The model (`facebook/opt-125m`) generated consistent output across all four eviction strategies:

```
Prompt: "The key to effective machine learning is..."
Output: " the visualization of data. It means you can imagine..."

✅ Phase 5 (LRU Eviction):        PASSED
✅ Phase 6 (Attention Eviction):   PASSED  
✅ Phase 7 (Hybrid Eviction):      PASSED
✅ Phase 8 (Prefetching):          PASSED
```

Note: No evictions happened during these tests — the model is too small relative to the 15+ GiB available KV memory.

---

## Benchmark Results

### Run 1 — opt-125m, 300 prompts, gpu_mem_util=0.20 (sequential)
> File: `benchmark_results/results_20260401_013454.json`

| Policy | Tok/sec | Avg Latency | P95 Latency | Evictions |
|---|---|---|---|---|
| LRU | **541.3** | 385.9 ms | 500.0 ms | 0 |
| Attention | 537.1 | 388.8 ms | 503.6 ms | 0 |
| Hybrid | 536.5 | 389.3 ms | 504.2 ms | 0 |

**Root cause of 0 evictions:** GPU had 5.86 GiB KV space; opt-125m is tiny (~0.24 GiB model), 300×256 tokens still nowhere near filling that.

---

### Run 2 — Llama-3.2-1B, 200 prompts, gpu_mem_util=0.30 (still sequential)
> File: `benchmark_results/results_20260401_123047.json`

| Policy | Tok/sec | Avg Latency | Evictions |
|---|---|---|---|
| LRU | 195.0 | 2342 ms | 0 |
| Attention | 194.1 | 2353 ms | 0 |
| Hybrid | 194.1 | 2354 ms | 0 |

**Root cause of 0 evictions:** Still sequential — each request finds an empty cache and completes before the next one starts.

---

### Run 3 — Batched Inference (PENDING)
> Fix applied: `benchmark.py` now uses a single `llm.generate(all_prompts)` call

This is the run that should produce actual evictions. With Llama-3.2-1B-Instruct at 30% GPU utilization and 200 concurrent requests, the KV pool should overflow.

---

## What This Proves / Doesn't Prove

### ✅ Proven
- The entire vLLM KV offloading stack compiles, imports, and runs correctly on V100
- Attention score computation (Triton kernels) works at inference time
- LRU, Attention-aware, and Hybrid eviction policies initialize correctly
- The CPU prefetcher initializes and completes without errors
- API schema is correctly configured end-to-end on a live GPU

### ⏳ In Progress
- **Eviction quality under pressure** — Batched benchmark pending (Run 3)
- **Prefetch hit rate** — Needs concurrent long-sequence workload
- **Throughput comparison by policy** — Pending Run 3 results

---

## Files Changed

| File | Change |
|---|---|
| `csrc/torch_bindings.cpp` | Commented out `dsv3_fused_a_gemm` registration |
| `scripts/psc_submit.sh` | Added `HF_TOKEN`, `HF_HOME`, `TRITON_CACHE_DIR`, `XDG_CACHE_HOME` |
| `scripts/gcp_setup_and_test.sh` | Migrated `kv_offloading_spec` → `kv_connector_extra_config`, added `kv_role` |
| `scripts/psc_benchmark.sh` | New SLURM script for stress benchmark |
| `kv_cache_tiering/benchmarks/benchmark.py` | Changed to batched `llm.generate()` to force concurrent KV pressure |

---

## How to Reproduce

```bash
# On PSC Bridges-2:
cd ~/workspace/vllm

# Run the validation suite
sbatch scripts/psc_submit.sh
tail -f vllm_test.log

# Run the benchmark (after psc_submit completes)
sbatch scripts/psc_benchmark.sh
tail -f vllm_bench.log

# Pull results locally
rsync -avz rnagaraj@bridges2.psc.edu:~/workspace/vllm/benchmark_results/ ./benchmark_results/
rsync -avz rnagaraj@bridges2.psc.edu:~/workspace/vllm/test_results/ ./test_results/
```
