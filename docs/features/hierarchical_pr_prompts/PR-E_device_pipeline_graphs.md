# PR-E — Device pipeline hygiene + experimental graphs

You are implementing **PR-E** of the Colibri-parity hierarchical plan.
**Depends on PR-C** (async H2D). Dual-SSD (PR-D) is optional.

## Goal

Keep the **activation pipeline on-device** (Colibri `CUDA_PIPE` spirit) and make
slot buffers **address-stable** so experimental graphs can be attempted without
breaking remaps.

## Context

- MoE apply: `moe_runner.py`, XPU experts under
  `vllm/model_executor/layers/fused_moe/experts/xpu_moe.py`
- Slots: `device_slots.py` — fixed `[slots, …]` buffers, row copies in-place
- Config: `tier_allow_cuda_graphs` (currently forces eager by default)

## Requirements

### 1. Activation residency audit

- Trace hierarchical + XPU MoE forward for accidental host round-trips or
  global synchronizes on the critical path.
- Ensure `wait` is only on **weight** copy events, not on hidden states.
- Add a debug metric or log-once if a sync is unavoidable.

### 2. Slot pointer stability

- Assert (test) that after many `ensure_from_host_rows` calls,
  `param.data_ptr()` for slot-backed parameters remains unchanged.
- Document that graphs may capture MoE only when this holds and remapping is
  outside the graph or slots cover the captured unique set.

### 3. Experimental graphs path

- Keep default: hierarchical ⇒ `enforce_eager` unless
  `--tier-allow-cuda-graphs`.
- When allowed, document supported mode (e.g. graph attention / non-expert;
  MoE eager with overlapped H2D) rather than promising full-graph MoE.
- Add a guarded smoke test or “skip if no graphs” test.

### 4. Docs

- Update `docs/features/hierarchical_expert_offload.md` and
  `docs/xpu/HIERARCHICAL_EXPERT_OFFLOAD.md` with pipeline + graph caveats.

## Non-goals

- Full Colibri Metal backend.
- Spec decode or atlas.
- Changing quant kernels.

## Test plan

```bash
.venv/bin/python -m pytest tests/model_executor/offloader/test_hierarchical_offload.py -v
# Hal smoke with tier_allow_cuda_graphs only if XPU graphs are known working
```

Acceptance: pointer-stability test green; no new host activation copies on the
hot path in a short profile note in the PR.

## PR description must include

- Duplicate check, tests/results, AI-assist statement, graph support matrix.

## Done when

- Audit fixes + stability test + docs; graphs remain opt-in and honest about
  limits.
