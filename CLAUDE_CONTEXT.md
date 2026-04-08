# KV Cache Tiering Project — Claude Code Context

## Who you're helping
**Akshara NS** — working on a CMU 11868 (Large Language Model Systems) course project.
Running all experiments on **PSC Bridges-2** (V100-32GB GPUs, `compute_70`). Nothing runs locally.

---

## Project Summary

We extend vLLM's PagedAttention with **CPU-tier KV cache offloading**, using attention-aware eviction
and predictive prefetching. The core hypothesis: evicting low-attention blocks to CPU (instead of LRU)
keeps important blocks on GPU longer, reducing decode-time stalls.

**Key result so far:** Under ShareGPT stress test (12% GPU VRAM, 200 multi-turn convos, Llama-3.2-1B):
- Attention policy: +9.2% throughput vs LRU, 1401ms avg latency
- Hybrid policy: +8.6% throughput vs LRU, 1408ms avg latency

---

## Repo Structure

| Remote | URL | Purpose |
|--------|-----|---------|
| `origin` | `https://github.com/akshara-ns/vllm-kvtier` | Akshara's fork (your working repo) |
| `team` | `https://github.com/rishirajFS/vllm-kv-tier` | Rishi's fork (teammate, main implementer) |
| `upstream` | `https://github.com/vllm-project/vllm` | Main vLLM repo |

Akshara's fork is based on upstream/main with Rishi's custom commits cherry-picked on top.

### Key custom files (from team/main, cherry-picked into origin)
```
kv_cache_tiering/
  benchmarks/
    benchmark.py          # Main benchmark harness — batched llm.generate()
    parse_longbench.py    # Throughput analysis for LongBench results
  
scripts/
  psc_submit.sh           # SLURM job: validation suite (Phases 1–8)
  psc_benchmark.sh        # SLURM job: stress benchmark
  slurm_longbench_7b.sh   # SLURM job: LongBench 7B evaluation
  gcp_setup_and_test.sh   # KVTransferConfig API setup

vllm/
  kv_offload/
    cpu.py                # CPU KV pool manager
    attention_manager.py  # Attention-weighted eviction (257 LOC)
    score_estimator.py    # L2 norm importance proxy (139 LOC)
  
csrc/
  torch_bindings.cpp      # Patched: dsv3_fused_a_gemm guard removed for V100

benchmark_results/        # JSON output files from completed runs
test_results/             # Unit test output
```

---

## What's been implemented (Rishi's work — DO NOT duplicate)

1. **Attention-weighted eviction policy** — uses hidden-state L2 norms as importance proxy,
   sorts blocks by cumulative score, evicts lowest first. EMA decay factor 0.95.
2. **Hybrid eviction policy** — α·attention + β·recency + γ·frequency
3. **LRU baseline** — standard vLLM default
4. **Score pipeline** — GPU worker → CPU scheduler via KVConnectorOutput
5. **53 unit tests** — all passing on V100
6. **V100 compatibility patches** — `dsv3_fused_a_gemm` guard, KVTransferConfig API migration
7. **PSC environment setup** — HF_HOME, TRITON_CACHE_DIR, XDG_CACHE_HOME to avoid quota crashes
8. **Batched benchmark harness** — fixed sequential→batched to force concurrent KV pressure

### Completed benchmark results
| Run | Model | Workload | Key finding |
|-----|-------|----------|-------------|
| Run 1 | opt-125m | 300 prompts, gpu=0.20 | 0 evictions — model too small |
| Run 2 | Llama-3.2-1B | 200 prompts, sequential | 0 evictions — sequential, no pressure |
| Run 3 (ShareGPT) | Llama-3.2-1B | 200 multi-turn, gpu=0.12 | ✅ Attention +9.2%, Hybrid +8.6% vs LRU |
| Midterm | Qwen2.5-7B | 10 batched, gpu=0.60 | Attention +7.3% throughput, -11.3% P95 latency |

---

## Akshara's contribution area

Akshara owns: **benchmark harness extensions, new workloads, metrics, analysis, visualization, report writing**
(This is "Member 3" from the proposal work split.)

### What is NOT yet done (Akshara's work to do)

1. **MS-MARCO RAG workload** — clustered spatial locality, target: shows +8.27% (per teammate's plan)
2. **HumanEval-Repo code workload** — local hotspot pattern, target: shows +2.83%
3. **Synthetic control workload** — uniform random access, should show ±0.5% (validates methodology)
4. **Fix eviction counter metrics** — `total_evictions: 0` is a bug in V1 API polling; needs fix
5. **ROUGE-L / BERTScore quality metrics** — prove policies are lossless in practice
6. **Memory pressure sensitivity sweep** — vary `gpu_mem_util` (0.20→0.60→0.12) on ShareGPT
7. **Visualization / plotting** — latency CDFs, throughput vs pressure curves, eviction rate over time
8. **Final report writing** — analysis, limitations, conclusion sections

---

## PSC Bridges-2 Environment

```bash
# Cluster: bridges2.psc.edu
# GPU: Tesla V100-SXM2-32GB (compute_70)
# CUDA: 12.4.0
# Python: 3.12.12
# vLLM: 0.16.0rc2.dev369+g003800536.d20260330.cu124

# Required env vars (always set these, or jobs crash with Errno 122 quota errors):
export HF_TOKEN="..."
export HF_HOME="/jet/home/.../workspace/vllm/hf_cache"
export TRITON_CACHE_DIR="/jet/home/.../workspace/vllm/triton_cache"
export XDG_CACHE_HOME="/jet/home/.../workspace/vllm/xdg_cache"

# Submit jobs:
sbatch scripts/psc_submit.sh       # validation suite
sbatch scripts/psc_benchmark.sh   # benchmark

# Pull results locally:
rsync -avz rnagaraj@bridges2.psc.edu:~/workspace/vllm/benchmark_results/ ./benchmark_results/
```

---

## KVTransferConfig (current working schema)

```python
# CORRECT — use this schema (old kv_offloading_spec is broken)
kv_connector_extra_config={
    "eviction_policy": "attention",  # or "lru", "hybrid"
    "cpu_bytes_to_use": 500e6
}
kv_role="kv_both"  # mandatory — Pydantic validator enforces this
```

---

## Known issues / gotchas

- **`total_evictions: 0` in JSON** — benchmark harness polls old V0 API counters; V1 API counters
  are not wired up correctly. The 9% throughput variance proves evictions happened. Needs a fix.
- **V100 has no FlashAttention2** — score estimator uses L2 norms of hidden states instead of
  raw attention weights (FlashAttention doesn't expose the weight matrix)
- **SSL broken on Bridges-2** — use `HF_ENDPOINT=https://huggingface.co` and disable SSL verify
  for dataset downloads
- **Token overflow** — truncate prompts to `max_model_len` budget before submitting to vLLM

---

## Research context (for report writing)

**Key papers:**
- PagedAttention / vLLM (Kwon et al., SOSP 2023) — block-based KV, our foundation
- H2O (Zhang et al., NeurIPS 2023) — attention-based eviction, drops tokens; we offload instead
- ScissorHands (Liu et al., NeurIPS 2023) — persistence of importance hypothesis, lossy
- FlexGen (Sheng et al., ICML 2023) — 3-tier offloading, offline placement; we do online
- Attention Sinks (Xiao et al., ICLR 2024) — initial tokens get high scores naturally (our system handles this)

**Our differentiator:** lossless (offload vs drop) + online attention-aware + production latency target

---

## Branching convention

```bash
# Always work on a branch, never commit directly to main
git checkout -b akshara/ms-marco-workload
git checkout -b akshara/eviction-metrics-fix
git checkout -b akshara/visualization

# Sync with Rishi's latest
git fetch team
git cherry-pick <hash>   # pick specific commits, don't merge blindly
```
