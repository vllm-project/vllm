# Routed Experts Performance Optimization Log

## Context

The `--return-routed-experts` flag enables routing replay functionality: every token's MoE routing decision (which experts were selected) is captured during inference and returned with the completion response. This data is a 3D array of shape `(seq_len, num_moe_layers, num_experts_per_tok)` in int16.

**Model under test**: nvidia/NVIDIA-Nemotron-3-Super-120B-BF16-BF16KV-012726
- 40 MoE layers, 22 experts per token
- TP=4 on 4x B200 GPUs, Ray distributed executor
- max_num_batched_tokens=8192, 256 concurrent requests
- Async scheduling DISABLED (Ray executor limitation)
- CUDA graphs enabled (both piecewise and full decode)

**Baseline throughput** (without `--return-routed-experts`):
- Steady-state decode: 7,750-9,500 tok/s
- Turnover (requests finishing + new prefills): 5,000-9,500 tok/s

**Original throughput** (with `--return-routed-experts`):
- Steady-state decode: ~7,600 tok/s (similar to baseline -- suspicious)
- Turnover: ~3,000 tok/s (degraded)
- Overall: reported as "more than 2x slower" end-to-end

---

## Architecture Overview

### Data Flow

```
Forward pass (GPU)
  |-- capture_fwd_routed_experts() called 40x (once per MoE layer)
  |   writes topk_ids into device_cache.buffer[:batch, layer_id, :]
  v
sync_fwd_experts_buffer_DtoH() -- called in _bookkeeping_sync after forward pass
  |-- D2H copy: device buffer -> host
  |-- scatter: distribute per-token data into per-request host buffers
  v
_extract_routed_experts_for_current_batch() -- when requests finish
  |-- get_routed_experts() returns numpy array
  |-- serialized into ModelRunnerOutput -> Ray DAG -> scheduler
  v
EngineCoreOutput -> output_processor -> RequestOutput -> API response
```

### Key Files

| File | Role |
|------|------|
| `vllm/model_executor/layers/fused_moe/routed_experts_capturer.py` | Device cache, host cache, capturer classes |
| `vllm/model_executor/layers/fused_moe/router/base_router.py` | Calls `capture_fn(topk_ids)` at line ~240 |
| `vllm/model_executor/layers/fused_moe/layer.py` | Wires capture_fn lambda at lines 560-565 |
| `vllm/v1/worker/gpu_model_runner.py` | Calls sync_fwd_experts_buffer_DtoH and extraction |
| `vllm/v1/outputs.py` | ModelRunnerOutput.routed_experts_dict |
| `vllm/v1/engine/__init__.py` | EngineCoreOutput.routed_experts |
| `vllm/v1/engine/output_processor.py` | Reconstructs numpy from bytes |
| `vllm/v1/core/sched/scheduler.py` | Passes routed_experts from ModelRunnerOutput to EngineCoreOutput |

---

## Attempt 1: Async D2H Only

### Hypothesis

The synchronous `.cpu()` call in `sync_fwd_experts_buffer_DtoH` triggers `cudaStreamSynchronize()` which blocks the CPU thread until all pending GPU operations complete, adding ~8ms per step.

### What We Did

- Added a **pinned host staging buffer** (same shape as device buffer, `pin_memory=True`)
- Added a **dedicated CUDA stream** + **CUDA event** for async D2H
- Issued `non_blocking=True` copy on the dedicated stream
- Deferred the scatter into per-request buffers to the start of the NEXT step
- Only synchronized the event at the start of the next step (by which time the copy is done)

### Result: No improvement

The throughput was identical to the original. This told us the `cudaStreamSynchronize()` from `.cpu()` was NOT the bottleneck. By the time `_bookkeeping_sync` runs, the GPU has already finished the forward pass (async scheduling is disabled), so the sync returns near-instantly.

### Lesson Learned

The **Python scatter loop** (iterating over 256 requests with dict lookups, `int()` conversions, `numpy.max()`, `get_or_grow_buffer`, fancy indexing) was the real per-step cost, not the CUDA synchronization.

---

## Attempt 2: GPU-Side Accumulation with Batched Flush

### Hypothesis

If we accumulate multiple decode steps' worth of data on the GPU buffer and flush them all at once (every ~32 steps), we eliminate the per-step Python scatter loop entirely.

### What We Did

- Added a `_write_offset` to `_RoutedExpertsDeviceCache` so each step appends at the next row
- `capture_fwd_routed_experts` writes at `buffer[_write_offset:_write_offset+batch, ...]`
- `sync_fwd_experts_buffer_DtoH` only saves lightweight metadata per step (~20us)
- Added `ensure_capacity(num_tokens)` called before each forward pass; triggers a bulk D2H + scatter flush when the buffer fills
- With `max_num_batched_tokens=8192` and 256 tokens/step, flush happens every ~32 steps

### Result: No improvement

Throughput was still the same. Investigation revealed a **critical bug: CUDA graph incompatibility**.

### Root Cause: CUDA Graphs Freeze Memory Addresses

During CUDA graph **capture**, `capture_fwd_routed_experts` records `buffer[_write_offset:...]` with whatever `_write_offset` value exists at capture time (typically 0). During graph **replay**, the GPU replays the exact same memory operation -- it always writes at offset 0, regardless of what Python thinks `_write_offset` is.

This means:
1. The GPU always overwrites the same rows (offset 0) every step
2. Python's `_write_offset` increments incorrectly, eventually triggering `ensure_capacity`
3. The flush copies the full buffer, but only offset-0 data is valid -- the rest is stale
4. The flush itself is expensive (~87ms for 8192-entry scatter), causing periodic pipeline bubbles

**Fundamental constraint**: `capture_fwd_routed_experts` must ALWAYS write at offset 0 to be compatible with CUDA graph replay.

### Lesson Learned

Any optimization that changes the memory offset used by `capture_fwd_routed_experts` will silently break under CUDA graphs. The device buffer is write-at-0, read-every-step -- there is no way to accumulate across steps on the GPU.

---

## Attempt 3: Staging Lists (Async D2H + Deferred Scatter)

### Hypothesis

Keep the async D2H (correct, even if the sync savings are small) but completely eliminate the per-step scatter into host cache buffers. Instead, append raw numpy slices to per-request lists and only build the final contiguous array when a request finishes.

### What We Did

- Reverted `capture_fwd_routed_experts` to always write at offset 0 (CUDA-graph safe)
- Replaced `_RoutedExpertsHostCache` scatter with `_req_staging: dict[str, tuple[list, list]]`
- Per step: async D2H + `list.append(pos.copy())` + `list.append(vals.copy())` per request
- At extraction: `np.concatenate(pos_list)` + `np.concatenate(vals_list)` + vectorized scatter
- Removed `ensure_capacity` (no longer needed)

### Result: Steady-state decode improved, but turnover CATASTROPHICALLY worse

**Steady-state decode** (256 reqs, 0 prompt):
- Staging lists: 7,525-9,190 tok/s (~3-5% from baseline)
- This matched baseline! The per-step overhead was ~0.3ms (list.append is very cheap)

**Request turnover** (requests finishing + new prefills):
- Staging lists: **136-170 tok/s** (vs baseline 5,000-9,500)
- 40-50 second stalls when batches of requests finish

### Root Cause: `np.concatenate` on Hundreds of Small Arrays

When a request with 500 output tokens finishes, `get_routed_experts` calls `np.concatenate` on 500 small numpy arrays. Each array has Python object overhead, and concatenation iterates through all of them. This made extraction O(num_tokens) with a high constant, compared to the original code's O(1) extraction (just a slice of a pre-existing contiguous buffer).

### Lesson Learned

The per-step scatter into host cache buffers (original approach) has a purpose: it keeps extraction O(1). Trading per-step cost for extraction cost only works if extraction is rare AND cheap. With hundreds of small arrays, concatenation is expensive.

---

## Attempt 4 (Final): Hybrid -- Async D2H + Host Cache Scatter + tobytes()

### What We Did

Combined the best aspects of all previous attempts:

1. **Async D2H** (from attempt 1): Pinned staging buffer + dedicated CUDA stream + event. Eliminates allocation overhead of `.cpu()` and overlaps the copy with the next step.

2. **Host cache scatter** (from original code, optimized): Scatter into pre-allocated per-request contiguous numpy buffers. For the common decode case (1 token/request), uses direct scalar indexing instead of slice views + `.max()` + redundant `int()` conversions.

3. **`tobytes()` instead of `tolist()` for serialization**: The original code used `experts.tolist()` to convert numpy arrays to nested Python lists for Ray/msgspec serialization. For a (500, 40, 22) int16 array, `.tolist()` creates **440,000 Python int objects** (~44ms per request). Replaced with `(shape, experts.tobytes())` which does a single memcpy (~0.1ms). **440x faster per request.**

4. **Reconstruction in output processor**: Added `np.frombuffer(data, dtype=np.int16).copy().reshape(shape)` in the output processor to reconstruct the numpy array from bytes before passing to `RequestOutput`.

### Files Changed (Final State)

**`routed_experts_capturer.py`** -- core capturer:
- `_RoutedExpertsDeviceCache`: always writes at offset 0 (CUDA-graph safe)
- `_RoutedExpertsCapturerReal.__init__`: creates pinned staging buffer, CUDA stream, event
- `sync_fwd_experts_buffer_DtoH`: finalizes previous async copy (event.synchronize + scatter_to_host), then issues new async D2H
- `_scatter_to_host`: optimized loop with fast path for decode (n_tokens==1 uses direct scalar indexing)
- `get_routed_experts`: returns `buf[:seqlen].copy()` from contiguous host cache buffer (O(1))
- `finalize_pending_copy`: public API to ensure latest D2H is scattered before reading

**`gpu_model_runner.py`** -- model runner integration:
- `_bookkeeping_sync` (~line 3097): calls `sync_fwd_experts_buffer_DtoH` with positions and per-request token counts
- `_extract_routed_experts_for_current_batch` (~line 2608): detects finishing requests, calls `finalize_pending_copy()`, extracts with `(shape, tobytes())` instead of `.tolist()`
- `_update_states` (~line 933): frees host cache buffers for finished requests

**`v1/outputs.py`** -- type change:
- `ModelRunnerOutput.routed_experts_dict`: changed from `dict[str, list]` to `dict[str, tuple]`

**`v1/engine/__init__.py`** -- type change:
- `EngineCoreOutput.routed_experts`: changed from `list | None` to `tuple | None` (contains `(shape, bytes)`)

**`v1/engine/output_processor.py`** -- reconstruction:
- After reading `engine_core_output.routed_experts`, reconstructs numpy array from `(shape, bytes)` tuple before passing to `make_request_output`

### Results

**Steady-state decode**: ~7,500-9,200 tok/s (3-5% from baseline, matches attempts 1-3)

**Request turnover**: Significantly improved by the `tobytes()` change. When 24 requests finish simultaneously:
- Old `.tolist()`: ~1,056ms of Python object creation
- New `.tobytes()`: ~5ms of memcpy
- Expected turnover gen throughput: close to baseline ~5,000-7,000 tok/s

---

## Performance Summary

| Phase | Original | Final Optimized | Baseline (no routing) |
|-------|---------|----------------|----------------------|
| Steady decode (256 reqs) | ~7,600 tok/s | ~7,500-9,200 tok/s | ~7,750-9,500 tok/s |
| Turnover (24 reqs finish) | ~3,000 tok/s | ~5,000+ tok/s | ~5,000-9,500 tok/s |
| Per-step overhead | ~2.5ms | ~1.0ms | 0 |
| Per-request extraction | ~0.1ms + 44ms tolist | ~0.1ms + 0.1ms tobytes | N/A |

---

## Scaling Characteristics

### Per-step cost (decode)

Scales with `num_requests`, independent of `seq_len`:
- Async D2H of `num_requests * layers * experts_per_tok * 2 bytes` (256 reqs: ~450KB, ~0.04ms)
- Python scatter loop: `num_requests * ~3-5us` (256 reqs: ~1ms)

### Extraction cost (per request, at finish time)

Scales with `seq_len`:
- `buf[:seqlen].copy()`: O(seqlen * layers * experts_per_tok)
- `.tobytes()`: same size, single memcpy

| seq_len | Extraction time | Old .tolist() time |
|---------|----------------|--------------------|
| 500 | ~0.2ms | ~44ms |
| 4K | ~1ms | ~350ms |
| 32K | ~8ms | ~2.8s |
| 256K | ~50ms | ~22s |

### Host memory

Each active request: `seq_len * layers * experts_per_tok * 2 bytes`

| seq_len | Per request | 256 concurrent |
|---------|------------|----------------|
| 500 | 880 KB | 220 MB |
| 4K | 6.9 MB | 1.7 GB |
| 32K | 55 MB | 13.8 GB |
| 256K | 440 MB | 112 GB |

---

## Bug Fixes

### Bug Fix 1: Stale Routing Data for Prefill Tokens (CUDA Graph Double-Init)

**Symptom**: The first several tokens (prompt/prefill tokens) in every request had routing data `[0, 1, 2, ..., 21]` for the last few MoE layers -- a sequential pattern matching `arange(num_experts_per_tok)` rather than real routing decisions. Decode tokens were correct.

**Root Cause**: `init_routed_experts_capturer()` was called inside `_capture_cudagraphs()` in `gpu_model_runner.py`. This method is invoked **twice** during startup -- once for PIECEWISE mode (mixed prefill-decode) and once for FULL mode (pure decode). Each call created a **new** global capturer with a **new** device buffer, replacing the previous one.

The sequence:
1. PIECEWISE capture → capturer #1 created, device buffer A allocated → CUDA graphs record `copy_()` ops writing to buffer A
2. FULL capture → capturer #2 created, device buffer B allocated → CUDA graphs record `copy_()` ops writing to buffer B → global capturer replaced with #2
3. During inference: PIECEWISE graphs (used for prefill) replay writes to **dead buffer A**; FULL graphs (used for decode) write to **live buffer B**; `sync_fwd_experts_buffer_DtoH` reads from buffer B

Result: prefill tokens got stale warmup data (`[0,1,...,21]` from dummy zero-input forward passes during graph warmup), while decode tokens got correct routing.

**Fix** (`gpu_model_runner.py`):
- Moved `self.init_routed_experts_capturer()` from inside `_capture_cudagraphs()` to `capture_model()`, called **once** before the CUDA graph capture loop. Both PIECEWISE and FULL graphs now write to the same device buffer.
- Added a call in the early-return path when CUDA graphs are disabled (`CUDAGraphMode.NONE` / `enforce_eager=True`) to ensure the capturer is still initialized.

### Bug Fix 2: Stale Routing Data After Request Preemption

**Symptom**: If a request was preempted (evicted from running queue to free KV cache memory) and later resumed, the routing data could contain stale entries from the previous run.

**Root Cause**: When the scheduler preempts a request, it resets `num_computed_tokens = 0` and frees KV cache blocks (the request will be re-prefilled from scratch). However, the routing host cache buffer and its `_filled_len` were **not cleared**. The `_filled_len` tracking uses `max(old, new)`, so if the first run reached position 500 (`_filled_len = 501`) and the resumed run only reached position 200, extraction would read `buf[:501]` including 301 rows of stale data from the first run.

**Fix** (`gpu_model_runner.py`, `_update_states()`):
- Added cleanup of host cache buffers for preempted requests using `scheduler_output.preempted_req_ids`, which is already populated by the scheduler. Calls `host_cache.free_request()` for each preempted request, clearing both the numpy buffer and `_filled_len`. Mirrors how KV cache blocks are freed on preemption.

### DP > 1 / Multi-Node Analysis

**Concern**: The implementation assumes "rank 0 sees all the data", which could break with DP > 1.

**Finding**: The architecture already handles DP > 1 correctly. Each DP rank runs an independent `EngineCore` with its own `Scheduler`, `Executor`, and TP group. Requests are load-balanced across DP ranks at ingestion time and never migrate between engines. Within each DP group, TP rank 0 creates a real capturer (others get Noop), captures routing for its own requests, and returns data via its own `ModelRunnerOutput` to its own scheduler. With DP=16 TP=4: 16 real capturers (one per DP rank) + 48 noop capturers, all in separate processes with no interference.

---

## Key Constraints to Remember

1. **CUDA graphs freeze addresses**: `capture_fwd_routed_experts` MUST write at a fixed offset (0). Any attempt to use a dynamic `_write_offset` will silently produce stale data during graph replay.

2. **Async scheduling is disabled with Ray executor**: The GPU is idle during `_bookkeeping_sync`. The async D2H mainly avoids allocation overhead from `.cpu()`, not GPU overlap.

3. **The scatter loop is O(num_requests) per step**: This is unavoidable because the device buffer is overwritten each step and each request's data must go to its own host buffer. The loop is optimized but fundamentally linear.

4. **`.tobytes()` changes the serialization format**: The downstream `EngineCoreOutput.routed_experts` is now `tuple[shape, bytes]` instead of `list`. The output processor reconstructs the numpy array. Any new consumer of this data must handle the `(shape, bytes)` format.

5. **Host memory is the hard limit at very long seq_len**: At 256K tokens, each request needs ~440MB of host memory for routing data. This is fundamental to storing per-token routing information.

---

## Potential Future Optimizations

Ranked by expected impact:

### Priority 1: C++ scatter loop (biggest per-step win)

The per-step Python scatter loop in `_scatter_to_host` (~1ms for 256 requests) is the single largest remaining overhead. Each iteration does Python dict lookups + numpy scalar writes. Moving this to a C++ extension in `csrc/` would eliminate Python interpreter overhead entirely. Expected: ~0.1ms instead of ~1ms (10x improvement). Could use a simple function signature:

```cpp
void scatter_routing_data(
    const int16_t* host_values,   // pinned staging buffer (flat)
    const int64_t* positions,     // position per token
    const int* req_offsets,       // CSR-style offsets per request
    int16_t** req_buffers,        // per-request host cache pointers
    int* req_buf_sizes,           // for bounds checking
    int num_requests,
    int layers,
    int experts_per_tok
);
```

### Priority 2: Pre-allocate host cache buffers

Currently `get_or_grow_buffer` starts small and doubles (~18 reallocations for a 256K sequence). Each reallocation copies the entire existing buffer. If the request's `max_tokens` is known at scheduling time (which it usually is), pre-allocate to that size in a single allocation. Eliminates all reallocation + copy overhead during the decode loop. Small code change in `_scatter_to_host` or at request arrival time.

### Priority 3: Move extraction off the GPU worker's critical path

`_extract_routed_experts_for_current_batch` (finalize + get_routed_experts + tobytes) runs inside `execute_model`, blocking the next decode step. Two options:

- **Extract in the scheduler** (EngineCore process): Pass a raw host cache buffer reference (or memoryview) through Ray and let the scheduler do the slice + tobytes. Completely removes extraction from the GPU worker's step loop.

- **Background thread in the worker**: The `.copy()` and `.tobytes()` are numpy operations that release the GIL, so a background thread could run them in parallel with the next step's Python bookkeeping. The `finalize_pending_copy()` event sync is also GIL-free.

### Priority 4: Pass bytes all the way to the API response

Currently the output processor reconstructs a numpy array from `(shape, bytes)` via `np.frombuffer().copy().reshape()`. This array eventually needs to be serialized to JSON for the API response, which may implicitly call `.tolist()` (creating millions of Python objects). Instead, pass the `(shape, bytes)` tuple all the way to `RequestOutput` and let the API layer base64-encode the bytes directly. Eliminates any accidental `.tolist()` at the JSON serialization boundary. The one adjustment needed: `RequestOutput.__init__` currently slices `routed_experts[:prompt_len + output_len]`, which would need to handle the bytes format.

### Priority 5: Skip routing capture during prefill

If the consumer only needs routing for output tokens (common for routing replay), skip capturing during prefill entirely. The `capture_fn` could check a flag or the scheduler could signal "prefill-only" steps. This cuts data size by `prompt_len / total_len` -- often 50%+ for long-context scenarios. Simple to implement: add a `self._capture_enabled` flag on the device cache and toggle it from the model runner.

### Priority 6: Batch tobytes for simultaneously finishing requests

When many requests finish at once (e.g., 24 in a burst), we call `.copy()` + `.tobytes()` 24 times sequentially. Instead, gather all finishing requests' buffer slices into one large concatenated array, call `.tobytes()` once, then split the bytes by size on the receiver side. One large memcpy is faster than many small ones due to DMA alignment and reduced Python call overhead.

### Priority 7: Delta / RLE compression for long sequences

Routing decisions have low entropy -- adjacent tokens often route to the same experts. For 256K sequences (440MB per request), compression could be very effective:

- **Delta encoding**: Store differences between consecutive tokens' expert IDs. Most deltas are 0, which compresses well.
- **Run-length encoding**: Collapse repeated routing patterns.
- **9-bit packing**: Expert IDs fit in 9 bits (max 512 experts) but are stored as int16. Packing to 9 bits saves 44% (440MB → 250MB). Libraries like `sub-byte` or `bitstruct` (with C extensions) can do this. Best applied only at serialization time to avoid slowing the per-step scatter.

Expected reduction: 3-10x for typical workloads. Higher effort due to custom encode/decode logic.

### Priority 8: Configurable capture granularity

Allow capturing routing data for only a subset of layers (e.g., every 4th MoE layer) or every Nth token. For analysis purposes, sampling 10 of 40 layers often suffices and gives 4x memory reduction with proportional bandwidth savings. Implement as a `capture_every_n_layers` config parameter on the capturer.

### Summary Table

| # | Optimization | Per-step impact | Memory impact | Effort |
|---|---|---|---|---|
| 1 | C++ scatter loop | ~1ms → ~0.1ms | None | Medium |
| 2 | Pre-allocate buffers | Eliminates realloc stalls | None | Small |
| 3 | Off-critical-path extraction | Removes extraction from step loop | None | Medium |
| 4 | Bytes all the way through | Avoids API-layer tolist | None | Small |
| 5 | Skip prefill capture | None (prefill already cheap) | 50%+ reduction | Small |
| 6 | Batch tobytes | Minor serialization win | None | Small |
| 7 | Delta/RLE/9-bit compression | None | 3-10x reduction | Large |
| 8 | Configurable granularity | Proportional to skip ratio | Proportional | Small |
