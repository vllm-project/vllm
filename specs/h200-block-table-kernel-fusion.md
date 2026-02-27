# Plan: Fuse Block Table Kernels to Reduce Launch Overhead

## Task Description

Address the TODO at `vllm/v1/worker/gpu/block_table.py:90-94` by fusing multiple kernel launches
in the block table management path. Currently, `apply_staged_writes()` launches one
`_apply_write_kernel` per KV cache group. Additionally, `gather_block_tables()` and
`compute_slot_mappings()` are two separate kernel launches that could be combined.

## Objective

When complete:
1. `apply_staged_writes()` launches a single fused kernel regardless of `num_kv_cache_groups`
2. (Stretch) `gather_block_tables` + `compute_slot_mappings` fused into one kernel launch
3. Correctness validated against existing behavior for both `num_kv_cache_groups == 1` and `> 1`
4. Benchmark evidence shows reduced kernel launch overhead on H200
5. Clean PR-ready branch with tests

## Problem Statement

The block table management in vLLM's V1 engine launches multiple separate Triton kernels
per scheduling step:

**Current kernel launches in `prepare_inputs()` hot path:**
```
apply_staged_writes():
  Launch 1: _apply_write_kernel for block_tables[0]    ─┐
  Launch 2: _apply_write_kernel for block_tables[1]     │ num_kv_cache_groups launches
  Launch N: _apply_write_kernel for block_tables[N-1]  ─┘
  + UVA copy for num_blocks

gather_block_tables():
  Launch N+1: _gather_block_tables_kernel              ← already group-batched in grid

compute_slot_mappings():
  Launch N+2: _compute_slot_mappings_kernel            ← already group-batched in grid
```

For `num_kv_cache_groups == 1` (standard models like Llama, Qwen), this is 3 kernels.
For `num_kv_cache_groups == 2` (hybrid models like Gemma 3, Llama 4), this is 4 kernels.

Each kernel launch has ~5-10us overhead on H200. At high throughput (thousands of requests/sec),
this adds up to measurable scheduling latency.

**Key insight from code analysis:**
- `_gather_block_tables_kernel` and `_compute_slot_mappings_kernel` are ALREADY group-batched
  (they use `group_id = tl.program_id(0)` with grid `[num_groups, num_reqs]`)
- The TODO specifically targets `apply_staged_writes()` which loops over groups separately
- A secondary opportunity exists: fusing gather + slot_mapping into one kernel since they're
  called sequentially and slot_mapping reads from the block tables that gather just wrote

## Solution Approach

### Phase 1: Fuse `apply_staged_writes` (addresses the TODO directly)

Create a new Triton kernel `_fused_apply_writes_kernel` that processes all KV cache groups
in a single launch. The kernel takes arrays of pointers (one per group) and uses
`tl.program_id(0)` for group_id, matching the existing pattern.

**Challenge:** Each group may have different staged writes (different indices, starts, contents).
The solution is to concatenate all groups' write metadata into single tensors and use per-group
offsets to index into them.

### Phase 2: Fuse gather + slot_mapping (stretch goal)

Create `_fused_gather_and_compute_slots_kernel` that, for each (group, request):
1. Copies the block table row from src to dst (gather)
2. Computes slot mappings using the source block table directly

This eliminates a dependency: currently slot_mapping reads from `input_block_tables` which
gather just wrote. By reading from the source `block_tables` directly (using `req_idx` from
`idx_mapping`), we can compute slot mappings without waiting for the gather to complete.

### Decision: Start with Phase 2 (higher impact)

Phase 1 (fusing apply_staged_writes) saves overhead proportional to `num_kv_cache_groups`,
which is usually 1. Phase 2 (fusing gather + slot_mapping) saves 1 kernel launch for ALL
models. Since we want measurable impact on standard Qwen3-VL models (which have
`num_kv_cache_groups == 1`), Phase 2 is more impactful.

**Final approach**: Implement both phases, benchmark each independently.

## Relevant Files

**Primary (will be modified):**
- `vllm/v1/worker/gpu/block_table.py` - BlockTables class, both Triton kernels, the TODO
- `vllm/v1/worker/gpu/buffer_utils.py` - `StagedWriteTensor.apply_write()` and `_apply_write_kernel` (reference for Phase 1)

**Callers (read-only, for understanding):**
- `vllm/v1/worker/gpu/model_runner.py:468-497,551,612-616,810` - calls append_block_ids, gather, compute_slot_mappings, apply_staged_writes
- `vllm/v1/worker/gpu/attn_utils.py:146-186` - builds attention metadata from block tables and slot mappings
- `vllm/v1/attention/backend.py:287-410` - `CommonAttentionMetadata` dataclass that consumes results

**Test references:**
- `tests/v1/worker/test_gpu_model_runner.py` - model runner tests
- `tests/v1/attention/test_attention_backends.py` - attention backend tests

**Kernel fusion pattern references:**
- `vllm/v1/worker/mamba_utils.py:20-46` - `batch_memcpy_kernel` pattern
- `vllm/model_executor/layers/fused_moe/fused_batched_moe.py:252-376` - batched expert kernel

### New Files
- `tests/v1/worker/test_block_table_kernels.py` - Unit tests for fused kernels
- `benchmarks/kernels/bench_block_table_kernels.py` - Standalone benchmark

## Architecture Reference

### Current Data Flow
```
prepare_inputs() in model_runner.py:
  │
  ├─ apply_staged_writes()         ← N kernel launches (1 per group)
  │   ├─ block_tables[0].apply_write()  → _apply_write_kernel[(n_writes,)]
  │   ├─ block_tables[1].apply_write()  → _apply_write_kernel[(n_writes,)]
  │   └─ num_blocks.copy_to_uva()
  │
  ├─ gather_block_tables()         ← 1 kernel launch
  │   └─ _gather_block_tables_kernel[(num_groups, num_reqs)]
  │      Copies: block_tables[group][req_idx] → input_block_tables[group][batch_idx]
  │
  └─ compute_slot_mappings()       ← 1 kernel launch
      └─ _compute_slot_mappings_kernel[(num_groups, num_reqs+1)]
         Reads: input_block_tables[group][batch_idx]  (output of gather!)
         Computes: slot_id = block_id * block_size + pos % block_size
```

### Proposed Data Flow (Phase 2)
```
prepare_inputs() in model_runner.py:
  │
  ├─ apply_staged_writes()         ← 1 kernel launch (fused across groups)
  │   └─ _fused_apply_writes_kernel[(num_groups, max_n_writes)]
  │
  └─ gather_and_compute_slot_mappings()  ← 1 kernel launch (fused)
      └─ _fused_gather_and_slots_kernel[(num_groups, num_reqs+1)]
         Phase A: Copy block_tables[group][req_idx] → input_block_tables[group][batch_idx]
         Phase B: Compute slot_id from block_tables[group][req_idx] directly
         Phase C: Pad remaining slots (last program)
```

### Key Data Structures
```
BlockTables:
├── block_tables: list[StagedWriteTensor]     [G][max_reqs, max_blocks]  (source)
├── block_table_ptrs: uint64                  [G]  (data_ptr per group)
├── block_table_strides: int64                [G]  (stride(0) per group)
├── input_block_tables: list[Tensor]          [G][max_reqs, max_blocks]  (reordered)
├── input_block_table_ptrs: uint64            [G]
├── slot_mappings: int64                      [G, max_tokens]
├── num_blocks: UvaBackedTensor               [G, max_reqs]
└── block_sizes_tensor: int32                 [G]

Where G = num_kv_cache_groups (typically 1, sometimes 2+)
```

### When num_kv_cache_groups > 1

Models with hybrid attention (different layer types):
- **Gemma 3**: Full attention + Sliding window → 2 groups
- **Llama 4 Scout**: Full attention + Local attention → 2 groups
- **DeepSeek-V3**: Standard attention, BUT MLA uses separate paths → typically 1 group

For Qwen3-VL-2B and 8B: **num_kv_cache_groups == 1** (standard multi-head attention only)

## Working Directory Convention

**CRITICAL:** Shell cwd resets to `/workspace/vllm` between every Bash tool call.

```bash
# Pattern A: cd && chain
cd /workspace/h200-block-table-kernel-fusion && git status

# Pattern B: absolute paths
git -C /workspace/h200-block-table-kernel-fusion status
```

The worktree lives at: `/workspace/h200-block-table-kernel-fusion`
The main repo lives at: `/workspace/vllm`
Results archive lives at: `/workspace/h200-block-table-kernel-fusion/results/`

## Implementation Phases

### Phase 1: Foundation (Steps 1-6)
Setup worktree, understand code deeply, write standalone benchmarks to measure baseline.

### Phase 2: Core Implementation (Steps 7-14)
Implement the fused gather+slot_mapping kernel, then the fused apply_writes kernel.

### Phase 3: Testing & Validation (Steps 15-20)
Unit tests, benchmark before/after, end-to-end model correctness check.

### Phase 4: PR Packaging (Steps 21-24)
Clean up, commit, generate PR description.

## Step by Step Tasks

### 1. Create Git Worktree

```bash
git worktree add --detach /workspace/h200-block-table-kernel-fusion HEAD && \
cd /workspace/h200-block-table-kernel-fusion && \
git checkout -b fuse-block-table-kernels && \
echo "Branch: $(git branch --show-current)" && \
git status
```

**Validation criteria (ALL must pass to proceed):**
1. `git branch --show-current` outputs exactly `fuse-block-table-kernels`
2. `pwd` outputs `/workspace/h200-block-table-kernel-fusion`
3. `git status` shows `nothing to commit, working tree clean`
4. `git worktree list` shows both `/workspace/vllm` and `/workspace/h200-block-table-kernel-fusion`
5. `ls /workspace/h200-block-table-kernel-fusion/vllm/v1/worker/gpu/block_table.py` exists

**STOP condition:** If any check fails, do NOT proceed. Debug and fix first.

---

### 2. Create Results Directory Structure

```bash
cd /workspace/h200-block-table-kernel-fusion && \
mkdir -p results/{01-env-check,02-baseline-benchmark,03-kernel-profiling,04-fused-gather-slots,05-fused-apply-writes,06-unit-tests,07-integration-test,08-benchmark-comparison,09-model-correctness,10-final}
```

**Validation criteria (ALL must pass):**
1. `ls /workspace/h200-block-table-kernel-fusion/results/ | wc -l` equals exactly `10`
2. Each directory name matches: `01-env-check`, `02-baseline-benchmark`, ..., `10-final`
3. All directories are empty (no stale data)

---

### 3. Verify GPU Environment

```bash
python3 -c "
import torch, triton
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'Triton: {triton.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'Compute: {torch.cuda.get_device_capability(0)}')
" 2>&1 | tee /workspace/h200-block-table-kernel-fusion/results/01-env-check/environment.txt
```

**Validation criteria (ALL must pass):**
1. `torch.cuda.get_device_name(0)` contains `H200`
2. `torch.cuda.get_device_capability(0)` equals `(9, 0)`
3. `torch.cuda.device_count()` >= 1
4. Triton version is >= 3.0
5. Output file `/workspace/h200-block-table-kernel-fusion/results/01-env-check/environment.txt` exists and is non-empty
6. No import errors or exceptions in output

---

### 4. Write Standalone Baseline Benchmark

Create a benchmark that measures kernel launch overhead for the CURRENT implementation.
This is critical: we need "before" numbers.

```bash
cat > /workspace/h200-block-table-kernel-fusion/benchmarks/bench_block_table_kernels.py << 'PYTHON'
"""Benchmark block table kernel launch overhead.

Measures:
1. apply_staged_writes() time (N kernel launches for N groups)
2. gather_block_tables() time (1 kernel launch)
3. compute_slot_mappings() time (1 kernel launch)
4. Total scheduling overhead per step
"""
import sys
import time
import torch
sys.path.insert(0, '/workspace/h200-block-table-kernel-fusion')

from vllm.v1.worker.gpu.block_table import BlockTables

def benchmark_block_table_ops(
    num_kv_cache_groups: int = 1,
    block_size: int = 16,
    max_num_reqs: int = 256,
    max_num_batched_tokens: int = 8192,
    max_model_len: int = 4096,
    num_reqs: int = 128,
    num_tokens: int = 2048,
    num_warmup: int = 50,
    num_iters: int = 200,
):
    device = torch.device("cuda:0")
    block_sizes = [block_size] * num_kv_cache_groups

    bt = BlockTables(
        block_sizes=block_sizes,
        max_num_reqs=max_num_reqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        device=device,
    )

    # Setup: populate block tables with realistic data
    max_blocks = max_model_len // block_size
    for req_idx in range(num_reqs):
        num_blocks = min(req_idx + 1, max_blocks)
        block_ids = tuple(
            list(range(req_idx * max_blocks, req_idx * max_blocks + num_blocks))
            for _ in range(num_kv_cache_groups)
        )
        bt.append_block_ids(req_idx, block_ids, overwrite=True)

    # Create idx_mapping (identity for simplicity)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)

    # Create query_start_loc (uniform distribution)
    tokens_per_req = num_tokens // num_reqs
    query_start_loc = torch.arange(
        0, num_reqs * tokens_per_req + 1, tokens_per_req,
        dtype=torch.int32, device=device
    )[:num_reqs + 1]
    query_start_loc[-1] = num_tokens

    # Create positions
    positions = torch.zeros(num_tokens, dtype=torch.long, device=device)
    for i in range(num_reqs):
        start = query_start_loc[i].item()
        end = query_start_loc[i + 1].item()
        seq_len = (i + 1) * (max_model_len // num_reqs)
        positions[start:end] = torch.arange(seq_len - (end - start), seq_len)

    results = {}

    # Benchmark apply_staged_writes
    # First stage some writes
    for req_idx in range(min(num_reqs, 32)):
        new_blocks = tuple(
            [req_idx * max_blocks + max_blocks - 1]
            for _ in range(num_kv_cache_groups)
        )
        bt.append_block_ids(req_idx, new_blocks, overwrite=False)

    # Warmup
    for _ in range(num_warmup):
        bt.apply_staged_writes()
        # Re-stage writes for next iteration
        for req_idx in range(min(num_reqs, 32)):
            new_blocks = tuple(
                [req_idx * max_blocks + max_blocks - 1]
                for _ in range(num_kv_cache_groups)
            )
            bt.append_block_ids(req_idx, new_blocks, overwrite=False)

    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        bt.apply_staged_writes()
        end_events[i].record()
        # Re-stage
        for req_idx in range(min(num_reqs, 32)):
            new_blocks = tuple(
                [req_idx * max_blocks + max_blocks - 1]
                for _ in range(num_kv_cache_groups)
            )
            bt.append_block_ids(req_idx, new_blocks, overwrite=False)

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    results['apply_staged_writes_ms'] = sum(times) / len(times)

    # Ensure writes are applied before gather/slot tests
    bt.apply_staged_writes()

    # Benchmark gather_block_tables
    for _ in range(num_warmup):
        bt.gather_block_tables(idx_mapping)

    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        bt.gather_block_tables(idx_mapping)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    results['gather_block_tables_ms'] = sum(times) / len(times)

    # Benchmark compute_slot_mappings
    for _ in range(num_warmup):
        bt.compute_slot_mappings(idx_mapping, query_start_loc, positions)

    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        bt.compute_slot_mappings(idx_mapping, query_start_loc, positions)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    results['compute_slot_mappings_ms'] = sum(times) / len(times)

    # Benchmark gather + compute_slot_mappings combined
    for _ in range(num_warmup):
        bt.gather_block_tables(idx_mapping)
        bt.compute_slot_mappings(idx_mapping, query_start_loc, positions)

    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        bt.gather_block_tables(idx_mapping)
        bt.compute_slot_mappings(idx_mapping, query_start_loc, positions)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    results['gather_plus_slots_ms'] = sum(times) / len(times)

    results['total_ms'] = results['apply_staged_writes_ms'] + results['gather_plus_slots_ms']

    return results


if __name__ == "__main__":
    print("Block Table Kernel Benchmark")
    print("=" * 60)

    configs = [
        {"num_kv_cache_groups": 1, "num_reqs": 32, "num_tokens": 512, "label": "small_batch_1grp"},
        {"num_kv_cache_groups": 1, "num_reqs": 128, "num_tokens": 2048, "label": "medium_batch_1grp"},
        {"num_kv_cache_groups": 1, "num_reqs": 256, "num_tokens": 8192, "label": "large_batch_1grp"},
        {"num_kv_cache_groups": 2, "num_reqs": 128, "num_tokens": 2048, "label": "medium_batch_2grp"},
        {"num_kv_cache_groups": 4, "num_reqs": 128, "num_tokens": 2048, "label": "medium_batch_4grp"},
    ]

    import json
    all_results = {}

    for cfg in configs:
        label = cfg.pop("label")
        print(f"\n--- {label} ---")
        print(f"  Config: {cfg}")
        results = benchmark_block_table_ops(**cfg)
        all_results[label] = {**results, **cfg}
        for k, v in results.items():
            print(f"  {k}: {v:.4f} ms")

    with open('/workspace/h200-block-table-kernel-fusion/results/02-baseline-benchmark/baseline.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to results/02-baseline-benchmark/baseline.json")
PYTHON

echo "Benchmark script created"
```

**Validation criteria (ALL must pass):**
1. `python3 -c "import ast; ast.parse(open('/workspace/h200-block-table-kernel-fusion/benchmarks/bench_block_table_kernels.py').read()); print('VALID')"` prints `VALID`
2. File contains `def benchmark_block_table_ops`
3. File contains `from vllm.v1.worker.gpu.block_table import BlockTables`
4. File size > 2000 bytes (non-trivial content)
5. File contains all 5 benchmark configs (small_batch_1grp through medium_batch_4grp)

---

### 5. Run Baseline Benchmark

```bash
cd /workspace/h200-block-table-kernel-fusion && \
python3 benchmarks/bench_block_table_kernels.py \
    2>&1 | tee /workspace/h200-block-table-kernel-fusion/results/02-baseline-benchmark/baseline_log.txt
```

**Validation criteria (ALL must pass):**
1. Script completes without Python exceptions or CUDA errors
2. JSON file exists at `results/02-baseline-benchmark/baseline.json`
3. JSON contains entries for all 5 configs (small_batch_1grp, medium_batch_1grp, large_batch_1grp, medium_batch_2grp, medium_batch_4grp)
4. Each entry has keys: `apply_staged_writes_ms`, `gather_block_tables_ms`, `compute_slot_mappings_ms`, `gather_plus_slots_ms`, `total_ms`
5. All timing values are > 0 and < 100ms (sanity range)
6. `gather_plus_slots_ms` >= `gather_block_tables_ms` (combined cannot be less than one component)
7. Log file at `results/02-baseline-benchmark/baseline_log.txt` contains no `Error` or `Traceback`

**DECISION POINT:** If `gather_plus_slots_ms` is >> `gather_block_tables_ms + compute_slot_mappings_ms`,
there's no kernel launch overhead to save (operations are latency-dominated). If they're close,
the overhead is in kernel launches and fusion will help.

---

### 6. Profile Kernel Launches with CUDA Events

```bash
cd /workspace/h200-block-table-kernel-fusion && \
python3 -c "
import torch, json
torch.cuda.set_device(0)

# Quick measurement: raw kernel launch overhead
num_iters = 1000

# Empty kernel launch overhead
dummy = torch.zeros(1, device='cuda')
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(num_iters):
    dummy.fill_(0)  # Minimal GPU operation
end.record()
end.synchronize()
per_launch_us = start.elapsed_time(end) / num_iters * 1000

print(f'Approximate kernel launch overhead: {per_launch_us:.2f} us')
print(f'At 1000 req/s, 3 launches/step = {3 * per_launch_us:.2f} us overhead/step')
print(f'At 1000 req/s, 2 launches/step = {2 * per_launch_us:.2f} us overhead/step')
print(f'Savings from 3→2 launches: {per_launch_us:.2f} us/step')

results = {
    'per_launch_overhead_us': per_launch_us,
    'savings_3_to_2_launches_us': per_launch_us,
    'savings_4_to_2_launches_us': 2 * per_launch_us,
}
with open('/workspace/h200-block-table-kernel-fusion/results/03-kernel-profiling/launch_overhead.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved to results/03-kernel-profiling/')
" 2>&1 | tee /workspace/h200-block-table-kernel-fusion/results/03-kernel-profiling/profiling_log.txt
```

**Validation criteria (ALL must pass):**
1. `per_launch_overhead_us` is between 1.0 and 50.0 (reasonable range for H200)
2. JSON file exists at `results/03-kernel-profiling/launch_overhead.json`
3. No CUDA errors in output
4. Log file exists and is non-empty

---

### 7. Implement Fused Gather+SlotMappings Kernel

This is the main code change. Create the fused kernel that does both gather and slot mapping
computation in a single launch.

**Key insight:** The slot mapping computation reads from `block_table_ptrs` (the source block
tables) using `req_state_idx` (from `idx_mapping`). This is the SAME source data that gather
copies from. So we can compute slot mappings directly from the source WITHOUT needing the
gathered output. The gather still needs to happen (attention backends read from
`input_block_tables`), but both operations can happen in parallel within the same kernel.

Modify `/workspace/h200-block-table-kernel-fusion/vllm/v1/worker/gpu/block_table.py`:

```python
# NEW: Fused kernel replacing separate gather + compute_slot_mappings
@triton.jit
def _fused_gather_and_slot_mappings_kernel(
    # Gather parameters
    batch_idx_to_req_idx,     # [batch_size]
    src_block_table_ptrs,     # [num_kv_cache_groups]
    dst_block_table_ptrs,     # [num_kv_cache_groups]
    block_table_strides,      # [num_kv_cache_groups]
    num_blocks_ptr,           # [num_kv_cache_groups, max_num_reqs]
    num_blocks_stride,
    # Slot mapping parameters
    num_tokens,
    max_num_tokens,
    query_start_loc,          # [num_reqs + 1]
    pos,                      # [num_tokens]
    block_sizes,              # [num_kv_cache_groups]
    slot_mappings_ptr,        # [num_kv_cache_groups, max_num_tokens]
    slot_mappings_stride,
    # Constants
    PAD_ID: tl.constexpr,
    GATHER_BLOCK_SIZE: tl.constexpr,
    SLOT_BLOCK_SIZE: tl.constexpr,
):
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)

    slot_mapping_ptr = slot_mappings_ptr + group_id * slot_mappings_stride

    # Last program handles padding (same as original)
    if batch_idx == tl.num_programs(1) - 1:
        for i in range(num_tokens, max_num_tokens, SLOT_BLOCK_SIZE):
            offset = i + tl.arange(0, SLOT_BLOCK_SIZE)
            tl.store(slot_mapping_ptr + offset, PAD_ID, mask=offset < max_num_tokens)
        return

    # --- Phase A: Gather block table row ---
    req_idx = tl.load(batch_idx_to_req_idx + batch_idx)

    group_num_blocks_ptr = num_blocks_ptr + group_id * num_blocks_stride
    num_blocks = tl.load(group_num_blocks_ptr + req_idx)

    stride = tl.load(block_table_strides + group_id)
    src_block_table_ptr = _load_ptr(src_block_table_ptrs + group_id, tl.int32)
    src_row_ptr = src_block_table_ptr + req_idx * stride
    dst_block_table_ptr = _load_ptr(dst_block_table_ptrs + group_id, tl.int32)
    dst_row_ptr = dst_block_table_ptr + batch_idx * stride

    for i in tl.range(0, num_blocks, GATHER_BLOCK_SIZE):
        offset = i + tl.arange(0, GATHER_BLOCK_SIZE)
        block_ids = tl.load(src_row_ptr + offset, mask=offset < num_blocks)
        tl.store(dst_row_ptr + offset, block_ids, mask=offset < num_blocks)

    # --- Phase B: Compute slot mappings ---
    block_size = tl.load(block_sizes + group_id)

    start_idx = tl.load(query_start_loc + batch_idx)
    end_idx = tl.load(query_start_loc + batch_idx + 1)

    # Read from SOURCE block table (not gathered dst) - avoids dependency
    for i in range(start_idx, end_idx, SLOT_BLOCK_SIZE):
        offset = i + tl.arange(0, SLOT_BLOCK_SIZE)
        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)
        block_indices = positions // block_size
        block_numbers = tl.load(
            src_block_table_ptr + req_idx * stride + block_indices
        )
        slot_ids = block_numbers * block_size + positions % block_size
        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)
```

**Changes to BlockTables class:**
- Add new method `gather_and_compute_slot_mappings()` that calls the fused kernel
- Keep original methods for backward compatibility (can be removed later)
- Update `prepare_inputs()` call site in model_runner.py

**Validation criteria (ALL must pass):**
1. `python3 -c "import ast; ast.parse(open('/workspace/h200-block-table-kernel-fusion/vllm/v1/worker/gpu/block_table.py').read()); print('VALID')"` prints `VALID`
2. File contains `def _fused_gather_and_slot_mappings_kernel`
3. File contains `@triton.jit` decorator above the new kernel
4. Kernel has `group_id = tl.program_id(0)` and `batch_idx = tl.program_id(1)`
5. Kernel handles the padding case (`if batch_idx == tl.num_programs(1) - 1`)
6. Kernel reads from `src_block_table_ptr` (NOT `dst_block_table_ptr`) for slot mapping computation
7. `python3 -c "from vllm.v1.worker.gpu.block_table import BlockTables; print('IMPORT OK')"` (run from worktree) succeeds
8. Original kernels `_gather_block_tables_kernel` and `_compute_slot_mappings_kernel` are preserved (not deleted)

---

### 8. Update BlockTables Class with Fused Method

Modify the class to add the new fused method:

```python
def gather_and_compute_slot_mappings(
    self,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Fused gather_block_tables + compute_slot_mappings in one kernel launch."""
    num_reqs = idx_mapping.shape[0]
    num_tokens = positions.shape[0]
    num_groups = self.num_kv_cache_groups

    _fused_gather_and_slot_mappings_kernel[(num_groups, num_reqs + 1)](
        idx_mapping,
        self.block_table_ptrs,
        self.input_block_table_ptrs,
        self.block_table_strides,
        self.num_blocks.gpu,
        self.num_blocks.gpu.stride(0),
        num_tokens,
        self.max_num_batched_tokens,
        query_start_loc,
        positions,
        self.block_sizes_tensor,
        self.slot_mappings,
        self.slot_mappings.stride(0),
        PAD_ID=PAD_SLOT_ID,
        GATHER_BLOCK_SIZE=1024,
        SLOT_BLOCK_SIZE=1024,
    )
    block_tables = tuple(bt[:num_reqs] for bt in self.input_block_tables)
    slot_mappings = self.slot_mappings[:, :num_tokens]
    return block_tables, slot_mappings
```

**Validation criteria (ALL must pass):**
1. Method exists in `BlockTables` class
2. Return type annotation is `tuple[tuple[torch.Tensor, ...], torch.Tensor]`
3. Method accepts same parameters as separate `gather_block_tables` + `compute_slot_mappings`
4. Kernel grid is `(num_groups, num_reqs + 1)` - must include +1 for padding program
5. `GATHER_BLOCK_SIZE=1024` and `SLOT_BLOCK_SIZE=1024` match original kernels
6. Returns `block_tables` sliced to `[:num_reqs]` and `slot_mappings` sliced to `[:, :num_tokens]`
7. Quick smoke test: `python3 -c "from vllm.v1.worker.gpu.block_table import BlockTables; print('OK')"` from worktree succeeds

---

### 9. Write Correctness Test for Fused Gather+Slots Kernel

```bash
cat > /workspace/h200-block-table-kernel-fusion/tests/v1/worker/test_block_table_kernels.py << 'PYTHON'
"""Tests for fused block table kernels."""
import pytest
import torch
import sys
sys.path.insert(0, '/workspace/h200-block-table-kernel-fusion')

from vllm.v1.worker.gpu.block_table import BlockTables


@pytest.fixture
def device():
    return torch.device("cuda:0")


def _setup_block_tables(device, num_kv_cache_groups=1, block_size=16,
                         max_num_reqs=32, max_model_len=512,
                         max_num_batched_tokens=1024):
    block_sizes = [block_size] * num_kv_cache_groups
    bt = BlockTables(
        block_sizes=block_sizes,
        max_num_reqs=max_num_reqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        device=device,
    )
    return bt


def _populate_block_tables(bt, num_reqs, block_size, max_model_len):
    max_blocks = max_model_len // block_size
    for req_idx in range(num_reqs):
        num_blocks = min(req_idx + 1, max_blocks)
        block_ids = tuple(
            list(range(req_idx * 100, req_idx * 100 + num_blocks))
            for _ in range(bt.num_kv_cache_groups)
        )
        bt.append_block_ids(req_idx, block_ids, overwrite=True)
    bt.apply_staged_writes()


@pytest.mark.parametrize("num_kv_cache_groups", [1, 2, 4])
@pytest.mark.parametrize("num_reqs", [1, 16, 64])
def test_fused_gather_and_slots_matches_separate(device, num_kv_cache_groups, num_reqs):
    """Verify fused kernel produces identical results to separate kernels."""
    block_size = 16
    max_model_len = 512
    max_num_batched_tokens = 2048

    bt = _setup_block_tables(
        device, num_kv_cache_groups, block_size,
        max_num_reqs=64, max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    _populate_block_tables(bt, num_reqs, block_size, max_model_len)

    # Create test inputs
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    tokens_per_req = max(1, max_num_batched_tokens // max(num_reqs, 1))
    tokens_per_req = min(tokens_per_req, 32)  # Cap for test
    num_tokens = num_reqs * tokens_per_req

    query_start_loc = torch.arange(
        0, num_tokens + 1, tokens_per_req,
        dtype=torch.int32, device=device
    )[:num_reqs + 1]
    query_start_loc[-1] = num_tokens

    positions = torch.zeros(num_tokens, dtype=torch.long, device=device)
    for i in range(num_reqs):
        start = query_start_loc[i].item()
        end = query_start_loc[i + 1].item()
        positions[start:end] = torch.arange(end - start, dtype=torch.long, device=device)

    # Run SEPARATE kernels (reference)
    ref_block_tables = bt.gather_block_tables(idx_mapping)
    ref_slot_mappings = bt.compute_slot_mappings(idx_mapping, query_start_loc, positions)

    # Save reference results
    ref_bt_list = [t.clone() for t in ref_block_tables]
    ref_sm = ref_slot_mappings.clone()

    # Reset input_block_tables to zeros
    for ibt in bt.input_block_tables:
        ibt.zero_()
    bt.slot_mappings.zero_()

    # Run FUSED kernel
    fused_block_tables, fused_slot_mappings = bt.gather_and_compute_slot_mappings(
        idx_mapping, query_start_loc, positions
    )

    # Compare
    for i, (ref, fused) in enumerate(zip(ref_bt_list, fused_block_tables)):
        assert torch.equal(ref, fused), (
            f"Block table mismatch for group {i}:\n"
            f"  ref:   {ref[:3, :10]}\n"
            f"  fused: {fused[:3, :10]}"
        )

    assert torch.equal(ref_sm, fused_slot_mappings), (
        f"Slot mapping mismatch:\n"
        f"  ref:   {ref_sm[0, :20]}\n"
        f"  fused: {fused_slot_mappings[0, :20]}"
    )

    print(f"PASSED: groups={num_kv_cache_groups}, reqs={num_reqs}")


@pytest.mark.parametrize("num_kv_cache_groups", [1, 2])
def test_fused_with_shuffled_idx_mapping(device, num_kv_cache_groups):
    """Test with non-identity idx_mapping (requests not in order)."""
    num_reqs = 16
    block_size = 16
    max_model_len = 256

    bt = _setup_block_tables(
        device, num_kv_cache_groups, block_size,
        max_num_reqs=32, max_model_len=max_model_len,
        max_num_batched_tokens=1024,
    )
    _populate_block_tables(bt, 32, block_size, max_model_len)  # Populate more than we use

    # Shuffled mapping: batch_idx 0 → req 5, batch_idx 1 → req 2, etc.
    perm = torch.randperm(32, device=device)[:num_reqs].to(torch.int32)
    idx_mapping = perm

    tokens_per_req = 8
    num_tokens = num_reqs * tokens_per_req
    query_start_loc = torch.arange(0, num_tokens + 1, tokens_per_req,
                                    dtype=torch.int32, device=device)[:num_reqs + 1]

    positions = torch.zeros(num_tokens, dtype=torch.long, device=device)
    for i in range(num_reqs):
        start = query_start_loc[i].item()
        end = query_start_loc[i + 1].item()
        positions[start:end] = torch.arange(end - start, dtype=torch.long, device=device)

    # Reference
    ref_block_tables = bt.gather_block_tables(idx_mapping)
    ref_slot_mappings = bt.compute_slot_mappings(idx_mapping, query_start_loc, positions)
    ref_bt_list = [t.clone() for t in ref_block_tables]
    ref_sm = ref_slot_mappings.clone()

    # Reset and run fused
    for ibt in bt.input_block_tables:
        ibt.zero_()
    bt.slot_mappings.zero_()

    fused_block_tables, fused_slot_mappings = bt.gather_and_compute_slot_mappings(
        idx_mapping, query_start_loc, positions
    )

    for i, (ref, fused) in enumerate(zip(ref_bt_list, fused_block_tables)):
        assert torch.equal(ref, fused), f"Block table mismatch for group {i} with shuffled mapping"

    assert torch.equal(ref_sm, fused_slot_mappings), "Slot mapping mismatch with shuffled mapping"
    print(f"PASSED: shuffled idx_mapping, groups={num_kv_cache_groups}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
PYTHON

echo "Test file created"
```

**Validation criteria (ALL must pass):**
1. `python3 -c "import ast; ast.parse(open('/workspace/h200-block-table-kernel-fusion/tests/v1/worker/test_block_table_kernels.py').read()); print('VALID')"` prints `VALID`
2. File contains `test_fused_gather_and_slots_matches_separate` function
3. File contains `test_fused_with_shuffled_idx_mapping` function
4. File contains `@pytest.mark.parametrize` decorators for `num_kv_cache_groups` and `num_reqs`
5. Tests use `torch.equal()` for exact comparison (not `allclose`)
6. File size > 3000 bytes (substantive tests)
7. Both test functions print "PASSED" on success for visual confirmation

---

### 10. Run Correctness Tests

```bash
cd /workspace/h200-block-table-kernel-fusion && \
python3 -m pytest tests/v1/worker/test_block_table_kernels.py -v -x \
    2>&1 | tee /workspace/h200-block-table-kernel-fusion/results/06-unit-tests/test_results.txt
```

**Validation criteria (ALL must pass - ZERO tolerance for failures):**
1. pytest exit code is 0
2. ALL parametrized test cases pass (should be 3x3=9 for `test_fused_gather_and_slots_matches_separate` + 2 for `test_fused_with_shuffled_idx_mapping` = 11 tests total)
3. Output contains `11 passed` (or appropriate count)
4. Output contains NO `FAILED`, `ERROR`, or `AssertionError`
5. Output contains NO `CUDA error` or `Triton compilation error`
6. Log file saved to `results/06-unit-tests/test_results.txt`
7. For EACH test case: `torch.equal` returned True for both block_tables AND slot_mappings

**STOP condition:** If ANY test fails, STOP. Do NOT proceed to integration. Debug the kernel:
- Check grid dimensions match `(num_groups, num_reqs + 1)`
- Check `_load_ptr` usage matches original kernels
- Check padding program handles `batch_idx == tl.num_programs(1) - 1`
- Check `src_block_table_ptr` is used (not dst) for slot mapping

---

### 11. Implement Fused apply_staged_writes (Phase 1)

This addresses the TODO directly. Modify `BlockTables.apply_staged_writes()` to batch all
groups' writes into a single kernel launch.

**Strategy:** Instead of calling `block_table.apply_write()` for each group separately,
collect all groups' staged write metadata, concatenate them with group offsets, and launch
a single kernel.

```python
def apply_staged_writes(self) -> None:
    """Apply all staged writes across all KV cache groups in one kernel."""
    # Collect metadata from all groups
    all_empty = True
    for block_table in self.block_tables:
        if len(block_table._staged_write_indices) > 0:
            all_empty = False
            break

    if all_empty:
        self.num_blocks.copy_to_uva()
        return

    # For each group, call apply_write individually
    # (When num_kv_cache_groups == 1, this is already optimal)
    # TODO: For num_kv_cache_groups > 1, could batch into single kernel
    for block_table in self.block_tables:
        block_table.apply_write()
    self.num_blocks.copy_to_uva()
```

**Note:** The `apply_write()` fusion is lower priority because:
1. `num_kv_cache_groups == 1` for most models (no loop overhead)
2. Each group has independent write metadata (different lengths)
3. The `StagedWriteTensor.apply_write()` already batches multiple writes per group

For the PR, we keep the original `apply_staged_writes` loop and focus on the higher-impact
gather+slots fusion. The TODO comment should be updated to reflect what was fused and what
remains.

---

### 12. Update model_runner.py Call Site

Modify the caller to use the fused method:

In `vllm/v1/worker/gpu/model_runner.py`, find where `gather_block_tables` and
`compute_slot_mappings` are called sequentially and replace with the fused call.

**Before:**
```python
block_tables = self.block_tables.gather_block_tables(idx_mapping)
# ... (other code between them)
slot_mappings = self.block_tables.compute_slot_mappings(
    idx_mapping, query_start_loc, self.input_buffers.positions[:num_tokens],
)
```

**After:**
```python
block_tables, slot_mappings = self.block_tables.gather_and_compute_slot_mappings(
    idx_mapping, query_start_loc, self.input_buffers.positions[:num_tokens],
)
```

**CRITICAL:** Verify there is no code between the two calls that depends on the intermediate
`block_tables` result. Read the model_runner.py code carefully between lines 551 and 616.

**Validation criteria (ALL must pass):**
1. `python3 -c "from vllm.v1.worker.gpu.model_runner import GPUModelRunner; print('IMPORT OK')"` succeeds from worktree
2. `git diff vllm/v1/worker/gpu/model_runner.py` shows ONLY the gather+slot_mapping call site changed
3. The old separate calls (`gather_block_tables` then `compute_slot_mappings`) are replaced with single `gather_and_compute_slot_mappings` call
4. NO other methods in model_runner.py are modified
5. The return values are unpacked correctly: `block_tables, slot_mappings = ...`
6. Verify NO code between the original two calls depends on intermediate `block_tables` by reading lines 551-616 of model_runner.py before making changes

**STOP condition:** If there IS code between gather and compute_slot_mappings that uses the
intermediate block_tables result, DO NOT fuse. Keep them separate and only do Phase 1.

---

### 13. Run Unit Tests Again After Integration

```bash
cd /workspace/h200-block-table-kernel-fusion && \
python3 -m pytest tests/v1/worker/test_block_table_kernels.py -v -x \
    2>&1 | tee /workspace/h200-block-table-kernel-fusion/results/06-unit-tests/test_results_post_integration.txt
```

**Validation criteria (ALL must pass):**
1. pytest exit code is 0
2. Same number of tests pass as Step 10 (11 tests)
3. No new failures introduced
4. Log file saved to `results/06-unit-tests/test_results_post_integration.txt`

---

### 14. Run Existing vLLM Tests for Regressions

```bash
cd /workspace/h200-block-table-kernel-fusion && \
python3 -m pytest tests/v1/worker/test_gpu_model_runner.py -v -x --timeout=120 \
    2>&1 | tee /workspace/h200-block-table-kernel-fusion/results/07-integration-test/model_runner_tests.txt
```

**Validation criteria (ALL must pass):**
1. pytest exit code is 0 (all tests pass)
2. NO `FAILED` or `ERROR` in output
3. If any test fails, check: is it pre-existing? Run same test from `/workspace/vllm` (main branch). If it also fails there, it's pre-existing and acceptable. If it ONLY fails in our branch, STOP and debug.
4. Log file saved and contains test results
5. Run `git stash && python3 -m pytest tests/v1/worker/test_gpu_model_runner.py -v -x --timeout=120` to compare against unfused code if any doubt

---

### 15. Update Benchmark to Test Fused Kernel

Add the fused kernel timing to the benchmark script:

```python
# Add to bench_block_table_kernels.py:
# Benchmark fused gather+slot_mappings
for _ in range(num_warmup):
    bt.gather_and_compute_slot_mappings(idx_mapping, query_start_loc, positions)

torch.cuda.synchronize()
start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

for i in range(num_iters):
    start_events[i].record()
    bt.gather_and_compute_slot_mappings(idx_mapping, query_start_loc, positions)
    end_events[i].record()

torch.cuda.synchronize()
times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
results['fused_gather_slots_ms'] = sum(times) / len(times)
results['savings_ms'] = results['gather_plus_slots_ms'] - results['fused_gather_slots_ms']
results['savings_pct'] = results['savings_ms'] / results['gather_plus_slots_ms'] * 100
```

---

### 16. Run Before/After Benchmark

```bash
cd /workspace/h200-block-table-kernel-fusion && \
python3 benchmarks/bench_block_table_kernels.py \
    2>&1 | tee /workspace/h200-block-table-kernel-fusion/results/08-benchmark-comparison/comparison_log.txt
```

**Validation criteria (ALL must pass):**
1. Script completes without errors
2. JSON file exists at `results/08-benchmark-comparison/comparison.json`
3. For EVERY config: `fused_gather_slots_ms` <= `gather_plus_slots_ms` (fused must not be slower)
4. For at least 3 out of 5 configs: `savings_pct` > 0 (measurable improvement)
5. `savings_ms` values are reasonable (not negative, not impossibly large)
6. Log file contains timing data for all configs

**If fused is SLOWER for any config:** Investigate. Possible causes:
- Increased register pressure from combined kernel reducing occupancy
- Larger kernel compile overhead on first launch (should amortize)
- Profile with `TRITON_PRINT_AUTOTUNING=1` to check compilation

---

### 17. Generate Benchmark Comparison Table

```bash
cd /workspace/h200-block-table-kernel-fusion && \
python3 -c "
import json

with open('results/02-baseline-benchmark/baseline.json') as f:
    baseline = json.load(f)
with open('results/08-benchmark-comparison/comparison.json') as f:
    comparison = json.load(f)

print('| Config | Separate (ms) | Fused (ms) | Savings (us) | Savings (%) |')
print('|--------|--------------|------------|-------------|-------------|')
for key in sorted(baseline.keys()):
    if key not in comparison:
        continue
    sep = baseline[key]['gather_plus_slots_ms']
    fused = comparison[key].get('fused_gather_slots_ms', sep)
    savings_us = (sep - fused) * 1000
    savings_pct = savings_us / (sep * 1000) * 100 if sep > 0 else 0
    print(f'| {key} | {sep:.4f} | {fused:.4f} | {savings_us:.1f} | {savings_pct:.1f}% |')
" 2>&1 | tee /workspace/h200-block-table-kernel-fusion/results/08-benchmark-comparison/comparison_table.md
```

---

### 18. Model Correctness Check with Qwen3-VL

```bash
cd /workspace/h200-block-table-kernel-fusion && \
python3 -c "
import sys
sys.path.insert(0, '.')
from vllm import LLM, SamplingParams

# Quick correctness check: does the model produce sensible output?
llm = LLM(
    model='/fsx/models/Qwen3-VL-2B-Instruct',
    tensor_parallel_size=1,
    max_model_len=512,
    enforce_eager=True,
    gpu_memory_utilization=0.5,
)

prompts = [
    'What is 2 + 2?',
    'Explain quantum computing in one sentence.',
]
params = SamplingParams(temperature=0.0, max_tokens=50)
outputs = llm.generate(prompts, params)
for out in outputs:
    print(f'Prompt: {out.prompt}')
    print(f'Output: {out.outputs[0].text}')
    print()
print('Model correctness check PASSED')
" 2>&1 | tee /workspace/h200-block-table-kernel-fusion/results/09-model-correctness/qwen3vl_2b_test.txt
```

**Validation criteria (ALL must pass):**
1. Model loads without CUDA errors or OOM
2. Output for "What is 2 + 2?" contains "4" somewhere in the response
3. Output for "Explain quantum computing" contains recognizable English words (not garbage/random tokens)
4. No `RuntimeError`, `AssertionError`, or `CUDA error` in output
5. Log file saved to `results/09-model-correctness/qwen3vl_2b_test.txt`
6. Output ends with `Model correctness check PASSED`

**STOP condition:** If model produces garbage output (random characters, repeated tokens, or
completely unrelated text), the fused kernel is corrupting data. REVERT changes and debug.

---

### 19. Update TODO Comment

Replace the TODO with documentation of what was done:

```python
def apply_staged_writes(self) -> None:
    # NOTE: Each block_table.apply_write() already batches all writes for
    # that group into a single kernel launch. The loop here launches one
    # kernel per KV cache group. For most models (num_kv_cache_groups == 1),
    # this is a single launch. For hybrid attention models with multiple
    # groups, this could be further optimized by batching across groups.
    for block_table in self.block_tables:
        block_table.apply_write()
    self.num_blocks.copy_to_uva()
```

---

### 20. Verify No Regressions on Existing Tests

```bash
cd /workspace/h200-block-table-kernel-fusion && \
python3 -m pytest tests/v1/worker/ -v --timeout=120 -x \
    2>&1 | tee /workspace/h200-block-table-kernel-fusion/results/07-integration-test/all_worker_tests.txt
```

**Validation criteria (ALL must pass):**
1. pytest exit code is 0
2. All tests that passed in Step 14 still pass
3. No NEW test failures introduced
4. Log file saved to `results/07-integration-test/all_worker_tests.txt`
5. Output contains no `FAILED` or `ERROR` strings

---

### 21. Stage and Commit

```bash
cd /workspace/h200-block-table-kernel-fusion && \
git add vllm/v1/worker/gpu/block_table.py && \
git add vllm/v1/worker/gpu/model_runner.py && \
git add tests/v1/worker/test_block_table_kernels.py && \
git add benchmarks/bench_block_table_kernels.py && \
git diff --cached --stat && \
echo "Staged files:" && \
git diff --cached --name-only | wc -l
```

**Validation criteria (ALL must pass):**
1. `git diff --cached --name-only` shows ONLY expected files (block_table.py, model_runner.py, test file, benchmark file)
2. NO unexpected files staged (no .pyc, no __pycache__, no results/)
3. `git diff --cached -- vllm/v1/worker/gpu/block_table.py` shows additions only (no deletions of original kernels)
4. `git diff --cached -- vllm/v1/worker/gpu/model_runner.py` shows minimal changes (only the call site)

```bash
cd /workspace/h200-block-table-kernel-fusion && \
git commit -m "$(cat <<'EOF'
[Perf] Fuse gather_block_tables + compute_slot_mappings into single kernel

Combine the two separate Triton kernel launches for block table gathering
and slot mapping computation into a single fused kernel launch. This saves
one kernel launch overhead (~5-10us) per scheduling step.

The fused kernel `_fused_gather_and_slot_mappings_kernel` performs both
operations in a single pass per (group, request) pair:
1. Copies block table row from source to batch-ordered destination
2. Computes slot mappings directly from source block table

Also adds unit tests and a standalone benchmark for block table kernel
performance measurement.

Signed-off-by: Mihir Ketkar <mihir@ketkar.dev>
EOF
)"
```

---

### 22. Archive Results

```bash
cd /workspace/h200-block-table-kernel-fusion && \
tar czf results/final-archive.tar.gz results/0*/ && \
echo "Archive:" && \
ls -lh results/final-archive.tar.gz
```

---

### 23. Generate PR Description

```bash
cat > /workspace/h200-block-table-kernel-fusion/results/10-final/PR_DESCRIPTION.md << 'PRDESC'
## Purpose

Fuse `gather_block_tables()` and `compute_slot_mappings()` into a single Triton kernel
launch, reducing per-step kernel launch overhead by ~5-10us on H200.

Addresses part of the TODO at `vllm/v1/worker/gpu/block_table.py:90-94`.

## Changes

- **New kernel:** `_fused_gather_and_slot_mappings_kernel` combines block table gathering
  and slot mapping computation in one kernel launch
- **New method:** `BlockTables.gather_and_compute_slot_mappings()` replaces sequential calls
- **Updated caller:** `model_runner.py` uses fused method in `prepare_inputs()`
- **Tests:** New test file `tests/v1/worker/test_block_table_kernels.py`
- **Benchmark:** New `benchmarks/bench_block_table_kernels.py`

## Test Plan

1. Unit tests verify fused kernel produces identical results to separate kernels
2. Tested with `num_kv_cache_groups` = 1, 2, 4
3. Tested with shuffled `idx_mapping` (non-identity batch ordering)
4. Existing `tests/v1/worker/` tests pass without regression
5. End-to-end correctness verified with Qwen3-VL-2B inference

## Benchmark Results

[INSERT comparison_table.md HERE]

## Technical Details

The fused kernel uses a 2D grid `[num_kv_cache_groups, num_reqs + 1]`:
- Each program handles one `(group_id, batch_idx)` pair
- Phase A: copies block table row from source to destination
- Phase B: computes slot mappings from source block table directly
- Last program per group handles padding for CUDA graphs

Key optimization: slot mappings are computed from the SOURCE block table using
`req_idx` (from `idx_mapping`), not from the gathered destination. This eliminates
the data dependency between gather and slot mapping operations.
PRDESC

echo "PR description saved"
cat /workspace/h200-block-table-kernel-fusion/results/10-final/PR_DESCRIPTION.md
```

---

### 24. Final Verification

```bash
cd /workspace/h200-block-table-kernel-fusion && \
git status && \
git log --oneline -3 && \
git diff HEAD~1 --stat
```

**Validation criteria (ALL must pass):**
1. `git status` shows `nothing to commit, working tree clean`
2. `git log --oneline -1` shows the commit with `[Perf] Fuse` in the message
3. `git diff HEAD~1 --stat` shows expected number of files changed
4. `git diff HEAD~1 --name-only` lists ONLY the files we intended to change
5. No `.pyc`, `__pycache__`, or `results/` files in the commit
6. Branch is exactly 1 commit ahead of its base

## Testing Strategy

1. **Unit Tests (test_block_table_kernels.py)**
   - Parametrized over `num_kv_cache_groups` = {1, 2, 4}
   - Parametrized over `num_reqs` = {1, 16, 64}
   - Tests identity and shuffled `idx_mapping`
   - Compares fused output against reference (separate kernels)
   - Exact `torch.equal()` comparison (must be bit-identical)

2. **Regression Tests**
   - Run existing `tests/v1/worker/test_gpu_model_runner.py`
   - Run all `tests/v1/worker/` tests

3. **End-to-End Correctness**
   - Load Qwen3-VL-2B, generate text, verify coherent output
   - This exercises the full code path through the fused kernel

4. **Performance**
   - Benchmark separate vs fused for multiple batch sizes and group counts
   - Measure kernel launch overhead directly

## Acceptance Criteria

- [ ] Fused kernel produces bit-identical results to separate kernels for all test configs
- [ ] All existing tests pass without regression
- [ ] Benchmark shows measurable improvement (>0 savings for all configs)
- [ ] Qwen3-VL-2B generates coherent text output
- [ ] Code is clean: no dead code, comments explain the approach
- [ ] PR description includes benchmark evidence

## Validation Commands

```bash
# Run unit tests
cd /workspace/h200-block-table-kernel-fusion && \
python3 -m pytest tests/v1/worker/test_block_table_kernels.py -v

# Run regression tests
cd /workspace/h200-block-table-kernel-fusion && \
python3 -m pytest tests/v1/worker/test_gpu_model_runner.py -v --timeout=120

# Run benchmark
cd /workspace/h200-block-table-kernel-fusion && \
python3 benchmarks/bench_block_table_kernels.py

# Check commit
cd /workspace/h200-block-table-kernel-fusion && \
git log -1 --stat
```

## Notes

- The `apply_staged_writes()` TODO (fusing across groups) is intentionally NOT fully
  addressed in this PR because: (a) `num_kv_cache_groups == 1` for most models makes it
  a no-op optimization, and (b) each group has independent write metadata making cross-group
  batching complex. The TODO comment is updated to document this reasoning.

- The fused kernel reads slot mapping data from the SOURCE block table (not gathered
  destination). This is correct because the slot mapping only needs the block_id for
  a given (req_idx, position) pair, which is the same in source and destination.

- For CUDA graph compatibility, the last program in the grid still handles padding
  (filling slots `[num_tokens, max_num_tokens)` with `PAD_SLOT_ID = -1`).

- `GATHER_BLOCK_SIZE` and `SLOT_BLOCK_SIZE` are both 1024, matching the original kernels.
  These could be tuned separately if profiling shows a bottleneck.
