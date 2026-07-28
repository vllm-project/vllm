# PR-C — Async ensure, PILOT, O_DIRECT / disk PIPE

You are implementing **PR-C** of the Colibri-parity hierarchical plan.
**Depends on PR-A and PR-B.**

## Goal

Hide staging latency the way Colibri does with **PIPE** + **PILOT**: overlap
I/O with compute, prefetch the next layer, and make `O_DIRECT` actually work.

## Context

- Ensure path: `manager.py` (`ensure_and_remap`, `_ensure_layer`,
  `prefetch_experts`)
- Slots DMA: `device_slots.py` (copy stream + events)
- PILOT: `pilot.py` + `hooks.py` (`maybe_pilot_prefetch`)
- Disk: `disk_store.py` (O_DIRECT probe often falls back to buffered I/O)
- MoE hook: `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`

## Requirements

### 1. Split schedule vs wait (async ensure)

- Add an API like `schedule_ensure` / `wait_ensure` (names can match style):
  - Start H2D on the hierarchical copy stream without blocking the caller.
  - Block only immediately before the MoE expert GEMM needs the weights.
- Keep **batch-union** and same-batch protect semantics.
- Attribute wait time to `h2d_stall_ns` (PR-A metrics).

### 2. Disk PIPE

- Fix `O_DIRECT` reads: aligned buffers (512/4096), padded length, then copy
  into the pinned RAM frame. If DIRECT fails, increment a
  `disk_direct_fallback` counter — do not silently pretend DIRECT is on.
- Priority queue: demand reads > PILOT prefetch.
- Coalesce to one pread per expert row when ExpertStore layout allows
  (`format.py`).

### 3. Working PILOT

- At wrap/`post_init`, register MoE **gate** modules via
  `PilotPrefetcher.register_gate` for every MoE layer.
- `--tier-pilot`: prefetch next layer experts into RAM (and optionally device
  slots that are free / not protected by the current batch).
- `--tier-pilot-real`: optional stronger lookahead (document cost).
- Log / metric: `pilot_predict_hits` vs `pilot_predict_misses` when the next
  layer’s real topk is known.

### 4. Tests

- Unit: schedule/wait ordering with fake streams if feasible.
- Unit: O_DIRECT alignment helper.
- Unit: gate registration + prefetch calls `prefetch_experts` for `layer+1`.

## Non-goals

- Dual-SSD, NUMA, speculation, atlas, CUDA/XPU graphs.

## Test plan

```bash
.venv/bin/python -m pytest tests/model_executor/offloader/test_hierarchical_offload.py -v
# Hal: Ornith or Mixtral with disk store if available
.venv/bin/python benchmarks/hierarchical_tier_bakeoff.py \
  --model /tank/nas/models/Ornith-1.0-35B-MXFP4 \
  --tier-num-slots 32 --tier-ram-gb 8 --tier-pilot \
  --warm 8 --max-tokens 64 \
  --output /tmp/hier_pr_c_pilot.json
```

Acceptance: vs PR-B same config without pilot/async, warm decode shows lower
`h2d_stall_ns` and/or higher device hit rate; report numbers in PR.

## PR description must include

- Duplicate check, tests/results, AI-assist statement, stall/hit metrics.

## Done when

- Async ensure + PILOT registration + real DIRECT path landed and measured.
