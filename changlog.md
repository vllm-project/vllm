# Changelog

## 2026-04-28: Case 1 Passive MoE CPU Offload Fix

Case 1 is selected for MoE models when `--moe-cpu-offload` is set.

### Fixed

- Corrected compact expert staging for passive CPU offload.
- Case 1 now builds a fresh per-layer/per-wave `global_expert_id -> gpu_slot_id`
  map after routing.
- Active experts are copied from CPU memory into compact GPU slots.
- Routed `topk_ids` are remapped from global expert ids to compact GPU slot ids
  before fused expert compute.
- Fused expert compute runs with the staged GPU slot count and no persistent
  expert map.
- GPU expert copies are retired after each wave, and the temporary map is
  cleared.

### Impact

- Fixes degenerate completions caused by mixing compact staged slot ids with the
  original global expert id space.
- Keeps passive offload as a fresh per-layer/per-wave transfer and compute path,
  without persistent active, working, missing, or hot expert lists.
- Preserves the main Case 1 goal: run large MoE models on smaller GPUs by
  keeping expert weights CPU-backed and staging only routed experts on GPU.

### Performance Note

Case 1 trades throughput for memory capacity. Inference can be slower than
full-GPU expert residency because active expert weights are copied from CPU
memory to GPU memory during execution.

### Validation

- Focused MoE offload tests pass:

```bash
dev/moe/.venv/bin/python -m pytest tests/kernels/moe/test_moe_cpu_offload.py -q
```

- Harness validation has covered one `gemma-4-26B-A4B-it` endpoint on one 40GB
  GPU and two `gemma-4-26B-A4B-it` endpoints sharing one 40GB GPU.

## 2026-04-28: Case 2 GPU Active Expert Prefetch

Case 2 is selected for MoE models when `--moe-gpu-prefetch <num>` is set.

### Added

- Adds public flag `--moe-gpu-prefetch <num>`.
- Keeps full MoE weights CPU-backed.
- Keeps a bounded active expert cache in GPU memory.
- Maintains active, working, and missing expert lists.
- Uses a pager thread to load missing experts and retire cold experts.
- Computes an effective prefetch size from the requested value and model active
  expert count.

### Impact

- Provides a separate GPU expert residency mode for repeated routed experts.
- Logs Case 2 selection, requested/effective prefetch size, and rate-limited
  pager state.

## Flag Precedence

- Dense models ignore MoE offload flags and use normal vLLM.
- MoE models with `--moe-gpu-prefetch <num>` use GPU active expert prefetch.
- MoE models with `--moe-cpu-offload` use passive CPU offload.
- If both flags are set, `--moe-gpu-prefetch <num>` takes precedence.
