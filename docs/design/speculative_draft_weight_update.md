# Runtime Draft Weight Update for Speculative Decoding

## Introduction

This document proposes a small set of runtime interfaces for updating
speculative draft model weights in vLLM without relying on internal attribute
access or process-entry monkey patches.

Today, vLLM supports several speculative decoding methods, including draft
models, EAGLE, Eagle3, and DFlash. However, the public runtime weight update
surfaces only target the verifier model. External integrations that need to
refresh draft model weights at runtime must currently reach into internal
implementation details such as:

- whether the active model runner stores the proposer on `drafter` or
  `speculator`,
- whether the proposer exposes the underlying module through `get_model()` or
  `model`,
- and how child processes can execute custom startup logic before engine or
  worker initialization.

These workarounds are fragile across internal refactors and make it harder to
build reliable training or control loops around speculative decoding.

This proposal adds two worker-facing APIs for draft weight access.

## Goals

- Add a stable runtime API for resolving the active speculative draft model.
- Add a stable runtime API for loading already-remapped draft model weights.
- Preserve support for fused parameter loaders such as
  `param.weight_loader(param, tensor, shard_id)`.
- Ensure DFlash-specific post-load buffer rebuilding happens automatically when
  needed.
## Non-goals

- This proposal does not define how external integrations transport draft
  weights between processes or machines.
- This proposal does not define parameter name translation from external
  training layouts to vLLM inference layouts.
- This proposal does not change the public `speculative_config` schema.
- This proposal does not add a new target-model weight update path.
- This proposal does not extend `EAGLEConfig` with additional alias fields.
  Existing DFlash config translation is sufficient when callers provide the
  expected input fields.
- This proposal does not define a new process-start plugin or hook mechanism.

## Background

vLLM currently exposes the following target-model weight update lifecycle:

- `reload_weights()`
- `start_weight_update()`
- `update_weights()`
- `finish_weight_update()`

These flows are scoped to `self.model_runner.model`, that is, the verifier
model. There is no public equivalent for speculative draft models.

At the same time, draft model ownership is intentionally an internal detail:

- one model runner path uses `drafter`,
- another model runner path uses `speculator`,
- and different proposers may expose the underlying module through different
  accessors.

This is acceptable for internal implementation, but it is not a stable contract
 for runtime integrations.

## Problem Statement

The missing public interfaces create three concrete problems.

### 1. Draft model discovery is unstable

External integrations need to locate the draft model in order to update it.
Today this requires knowledge of internal runner and proposer fields that may
change as speculative decoding implementations evolve.

### 2. Draft model weight loading is incomplete

Some speculative models contain fused parameters that must be loaded through
`weight_loader`, optionally with a `shard_id`. A generic `param.data.copy_`
implementation is not sufficient for all cases.

In addition, DFlash maintains fused KV buffers derived from layer weights.
Loading parameters directly without rebuilding those buffers can leave the draft
inference path reading stale state.

## Proposal

The proposal adds two pieces:

1. `Worker.get_draft_model()`
2. `Worker.update_speculative_model_weights(weights)`

### Proposal 1: `Worker.get_draft_model()`

Add a worker-facing accessor that returns the active speculative draft model, or
`None` when speculative decoding is disabled or the active method does not use a
module-backed draft model.

Proposed signature:

```python
def get_draft_model(self) -> nn.Module | None:
    ...
```

This API is placed on `Worker`, not directly on one model-runner
implementation. The reason is that vLLM already hides the V1/V2 runner choice
behind the worker. A worker-level method can bridge runner differences without
forcing external callers to understand which runner is active.

Expected behavior:

- Return the resolved draft module when a module-backed speculative proposer is
  active.
- Return `None` when no such module exists.
- Keep internal runner/proposer field names as a private concern of vLLM.

### Proposal 2: `Worker.update_speculative_model_weights(weights)`

Add a worker-facing API that loads already-remapped draft weights into the
resolved speculative model.

Proposed signature:

```python
def update_speculative_model_weights(
    self,
    weights: list[tuple[str, Tensor, str | int | None]],
) -> dict[str, Any]:
    ...
```

Each element is:

- `name`: target parameter name in vLLM draft-model layout,
- `tensor`: the weight shard or full tensor to load,
- `shard_id`: optional fused-parameter shard identifier.

Expected behavior:

1. Resolve the draft model through `get_draft_model()`.
2. Return `{"loaded_params": 0, "has_draft_model": False}` when no draft model
   is present.
3. Iterate over `draft_model.named_parameters()`.
4. For matching parameters:
   - call `param.weight_loader(param, tensor, shard_id)` when available,
   - fall back to `param.data.copy_(tensor)` otherwise.
5. Rebuild DFlash fused KV buffers by calling
   `draft_model.model._build_fused_kv_buffers()` when that method exists.
6. Return a small summary such as
   `{"loaded_params": N, "has_draft_model": True}`.

This API intentionally assumes the input names have already been translated into
vLLM draft-model parameter names. Name remapping and routing policy remain
outside vLLM core.

### Why post-load DFlash rebuild belongs in this API

DFlash may derive fused inference buffers from layer parameters during model
loading. A direct runtime parameter update bypasses the normal `load_weights`
path, so it must trigger equivalent rebuild logic before inference uses the new
weights.

Putting that rebuild inside the vLLM API makes correctness part of the contract
instead of an external convention.

## Compatibility and Migration

This proposal is additive.

- Existing speculative decoding usage remains unchanged.
- Existing target-model weight update APIs remain unchanged.
- External integrations can migrate incrementally from internal monkey patches
  to the new worker-facing APIs.

## Summary

The core idea is small:

- provide a stable way to find the draft model,
- provide a stable way to update its weights correctly.

These additions reduce dependence on internal implementation details while
keeping transport, remapping, and orchestration policy outside the vLLM core.
