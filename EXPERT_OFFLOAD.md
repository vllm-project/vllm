# W4A16 Cold-Expert CPU Offload (RDNA3 / gfx1100)

Run a MoE model whose expert weights exceed VRAM by keeping the full expert set
in **CPU pinned RAM** and streaming a fixed-size **hot subset onto the GPU** each
forward. Targets W4A16 (compressed-tensors, `pack-quantized` int4) MoE on AMD
RDNA3 using the native `moe_gptq_gemm_rdna3` kernel. Proven end-to-end on the
full **Qwen3.5-122B-A10B** (256 experts / top-8 / 48 layers) on 2×24 GB.

Depends only on the upstream RDNA3 W4A16 MoE kernel (`moe_gptq_gemm_rdna3`) and
the `CompressedTensorsWNA16RDNA3MoEMethod` path — no other RDNA3-stack changes.

---

## How to use

Set one environment variable — the number of experts kept resident **per layer,
per rank**:

```bash
VLLM_MOE_EXPERT_CACHE_SIZE=64      # 0 (default) = disabled, normal all-resident path
```

That is the only knob. When `> 0`, the RDNA3 W4A16 MoE method automatically:
loads experts to pinned CPU, builds a `C`-slot GPU cache, and streams experts in
per step. When `0` / unset, behaviour is unchanged (all experts on GPU).

### Full serving example (122B on 2×24 GB)

```python
import os
os.environ["VLLM_MOE_EXPERT_CACHE_SIZE"] = "64"

# IMPORTANT (tp>1): precompile the runtime kernel in the MAIN process before
# LLM() spawns workers, or the two workers race to JIT-compile it -> ninja-lock
# deadlock. See "Gotchas".
import importlib.util, sys
_spec = importlib.util.spec_from_file_location(
    "expert_gather",
    "<vllm>/model_executor/layers/fused_moe/expert_gather.py")
_eg = importlib.util.module_from_spec(_spec); sys.modules["_egpre"] = _eg
_spec.loader.exec_module(_eg); _eg._mod()          # compiles / loads the .so

from vllm import LLM
llm = LLM(
    model="/models/Qwen3.5-122B-A10B-4bit",
    tensor_parallel_size=2,
    kv_cache_dtype="int8_per_token_head",   # frees VRAM for the cache (+ enforce_eager, see below)
    language_model_only=True,               # multimodal model -> skip vision tower
    max_model_len=2048,
    gpu_memory_utilization=0.85,
    trust_remote_code=True,
    # enforce_eager=True,                   # REQUIRED if kv_cache_dtype=int8 in this container build
)
```

### Parameter cheat-sheet

| Parameter | Where | Meaning / recommended |
|---|---|---|
| `VLLM_MOE_EXPERT_CACHE_SIZE` | env | **The knob.** Experts resident per layer/rank. `0`=off. `64`≈8 GB/rank cache on the 122B (fits 2×24 GB with KV). `128`≈16 GB/rank (does **not** fit alongside model+KV on 24 GB). |
| `tensor_parallel_size` | LLM | `2` recommended. TP is transparent to the cache (routing is TP-invariant; each rank caches its own shard, no cross-rank coordination). tp=1 works but is slower here (2× compute > the AR it avoids). |
| `kv_cache_dtype` | LLM | `int8_per_token_head` recommended for hybrid/large models — halves KV so the cache fits. In this container build int8 KV **requires `enforce_eager=True`** (int8+cudagraph = page-fault; fix lives on `rdna3_full_stack` but not in the installed `.so`). |
| `enforce_eager` | LLM | Cudagraph **is** supported by the offload (the gather is capturable — see below). Only forced to `True` when using int8 KV in this build. |
| `language_model_only` | LLM | `True` for multimodal checkpoints (Qwen3.5-*) to skip the vision tower. |
| `gpu_memory_utilization` | LLM | ~0.85. The GPU cache is allocated out-of-band during weight processing; leave headroom. |

**Sizing rule of thumb** (per rank, per layer, C experts): cache VRAM ≈
`C × per_expert_bytes × num_layers`. For the 122B (per-expert ≈ 175 MB/layer at
the 4 gathered planes for the given TP shard): `C=64 → ~8.4 GB/rank`,
`C=32 → ~4.2 GB/rank`. CPU pinned master holds **all** E experts ≈ 32 GB/rank
(64 GB total for the 122B) — this is the model itself and is unavoidable.

---

## How it is implemented

Three pieces. The cache is generic over W4A16 (agnostic to the int4 layout — it
only slices dim-0 of the fused expert tensors) and is consumed by the RDNA3
kernel unchanged.

### 1. Runtime-compiled HIP gather kernels — `fused_moe/expert_gather.py`

Two kernels, both **HIP-graph capturable** (fixed launch shapes, no host sync),
compiled at import via `torch.utils.cpp_extension.load_inline` (**no `_rocm_C`
rebuild**):

- **`ec_plan`** (1 workgroup): dedup the step's needed experts; on a *hit* freshen
  the slot's LRU clock; on a *miss* pick a clock-LRU victim (current-step experts
  are auto-protected because hits/just-loaded slots are freshened to the max
  clock this step, so the argmin never selects them); update the persistent
  `slot_of_expert[E]` / `expert_of_slot[C]` maps; emit an `(expert→slot)` copy
  plan padded to a fixed length.
- **`ec_copy`** (1 block per plan entry): for each valid entry, **zero-copy** the
  expert's row block from device-mapped pinned host memory into its GPU slot
  (vectorized `int4`); padded entries are skipped by a per-thread branch.

The skip-on-hit **reuse** happens *inside* the kernel (SIMT branch), not via CUDA
graph control flow — which is why it captures on HIP even though HIP has no
conditional graph nodes. Measured on gfx1100: in-kernel zero-copy from pinned
host = **~26 GB/s (== DMA)**, all-hit overhead ≈ 0.007 ms/layer.

`ExpertOffloadCache` (same file) ties it together: pinned master + GPU slots per
plane + the state tensors; `ensure(topk_ids)` runs `plan` + one `copy` per plane
and returns `topk_ids` remapped into slot space `[0, C)`.

### 2. Wiring — `compressed_tensors_moe/compressed_tensors_moe_wna16_rdna3.py`

- **`create_weights`**: when the env is set, allocates the big expert params on
  **CPU pinned** (via a `with torch.device("cpu")` context) so the full E-expert
  set never has to fit in VRAM at load.
- **`_process_weights_offload`**: vLLM's `device_loading_context` moves each
  layer's params to the GPU before this runs, so it shuffles the packed weights
  **in-place on the GPU** (fast), moves each prepared plane to a CPU pinned
  master, and **frees the GPU param** (`p.data = empty`) — freeing is essential
  or the 48 layers pile up on the GPU. Zero-points are identical across experts,
  so they are a single static `[C, …]` GPU tensor (not gathered). 4 planes are
  gathered: `w13/w2_weight_packed`, `w13/w2_weight_scale`.
- **`_rdna3_fused_moe`**: on the offload path — `ec.ensure(topk_ids)` → remap →
  `moe_align_block_size(..., global_num_experts=C)` → the existing
  `moe_gptq_gemm_rdna3` GEMM over the `[C,…]` cache tensors. For batches whose
  working set could exceed `C` it **chunks the tokens** so each chunk needs ≤ C
  experts.

---

## Status (2026-07-03)

- **Works E2E**: full 122B, TP2, coherent output. **Cudagraph proven** (capture +
  500 replays of `ensure()` correct; full-model capture also succeeds — needs
  `C≤48` to fit graph pools alongside model+KV on 24 GB).
- **Speed**: ~13.6 tok/s decode @ C=64 cudagraph. Bound by compute + copy, near
  the model/box ceiling (the TP all-reduce is NOT the bottleneck — profiler
  "all_reduce 55%" is wait time; isolated AR is 56 µs).
- **Quality preserved** (offload is numerically transparent — the gather is
  correct and deterministic).

## Known limitations

- **Prefill thrashing (biggest gap):** offload is decode-optimized (bs=1, small
  working set). Long-prompt / high-concurrency **prefill** has a working set ≫ C
  → the token-chunk loop reloads the cache repeatedly (an 8192-token forward can
  copy terabytes → RPC timeout). Mitigate with small `max_num_batched_tokens`
  (e.g. 256) + low `max_num_seqs`. **v2 fix:** process prefill *expert-major*
  (load each unique expert once, GEMM all its tokens; reuse the `moe_align`
  grouping).
- **C sizing on 24 GB:** `C=128` (16 GB cache) does not fit with model + KV;
  `C=64` (8 GB) is the practical operating point.
- **RAM:** the pinned master = full model experts (~64 GB for the 122B). Pinned
  memory is non-swappable; this is the price of cudagraph (an mmap/pageable
  master would need copies outside capture).

## Gotchas (all hit during bring-up)

- **tp>1 kernel JIT deadlock:** both workers race to `load_inline`-compile the
  same extension → ninja lock deadlock. **Precompile in the main process** before
  `LLM()` (see example). Also delete stale locks:
  `find /root/.cache/torch_extensions -name lock -delete`.
- **`device_loading_context` GPU pile-up:** do **not** keep a reference to the
  GPU param as the master — free it and hold a CPU pinned copy (fixed in
  `_process_weights_offload`).
- **int8 KV + cudagraph = page fault** in the installed build → use
  `enforce_eager=True` with int8 KV (or fp16 KV + cudagraph if it fits).
- Kill stray `VLLM::` workers by that name, not the script name (spawn detaches
  them); `-9`-killed workers leak pinned RAM — reclaim with
  `echo 3 > /proc/sys/vm/drop_caches`.
- `datasets` 5.x needs `"openai/openai_humaneval"`, not bare `"openai_humaneval"`.
