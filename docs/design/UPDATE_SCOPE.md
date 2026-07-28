# Weight Update Scope

## Status

This document defines the implementation plan for explicit weight-update
scopes. It complements `TRANSACTION.md`; it does not move the existing arena
verification points or claim to provide rollback.

## Goals

An update declares exactly which model state it is allowed to change. The
declaration is also the completion contract:

- omitted state outside the scope is preserved;
- missing state inside the scope fails completion;
- state received outside an explicit scope is rejected;
- checkpoint-format partial updates never split a post-load processing unit;
- a LoRA update replaces or removes one complete adapter while preserving the
  base model.

Transport chunks are not scopes. In particular, splitting a full checkpoint or
adapter into IPC/NCCL buckets does not make each bucket a partial update.

## Scope kinds

### Full checkpoint weights

The default and backward-compatible scope. Every source/target/fragment event
observed during the initial real checkpoint load is required.

### Partial checkpoint weights

The caller declares exact checkpoint source names. Each worker resolves those
names against its rank-local initial-load manifest. A selected source expands
to every local target fragment it drove during the baseline load.

Checkpoint restoration and `process_weights_after_loading` operate on whole
modules. Therefore a partial checkpoint scope is accepted only when, for every
module, it selects either no baseline events or all baseline events. Selecting
only Q from a packed QKV layer, one quantization scale, or some experts from a
post-load processing unit is rejected before layerwise mutation starts.

Only selected modules are restored to meta, wrapped, replayed, and processed.

### Kernel-format weights

The caller declares exact target parameter names. These updates use in-place
`copy_` and do not restore checkpoint layout or rerun post-load processing.
Parameter-level subsets are therefore allowed, subject to exact name, shape,
and dtype checks.

### LoRA adapter

A LoRA scope identifies one adapter and an operation (`replace` or `remove`).
Replacement means a complete adapter replacement, not an incremental patch of
the old adapter. The base model is outside the scope.

Two payload forms are supported by the completion model:

- path/artifact, as used by SkyRL;
- an exact tensor manifest accumulated across transport buckets, as used by
  VERL before constructing a tensor-backed LoRA request.

The adapter is staged and validated before it is installed. Validation covers
the declared tensor manifest, PEFT configuration identity, supported target
modules, rank/shape compatibility, and A/B pairing. Expert-parallel workers may
materialize different local subsets, but must agree on the global declaration.

## Data model

`UpdateScope` is a serializable tagged structure:

```text
kind=base_checkpoint, mode=full
kind=base_checkpoint, mode=partial, source_names=[...]
kind=base_kernel, target_names=[...]
kind=lora_adapter, operation=replace|remove,
    adapter_id=..., adapter_name=..., tensor_names=[...],
    config_digest=..., artifact_digest=...
```

The scope is normalized at the API boundary. Duplicate names, invalid field
combinations, empty explicit subsets, and inconsistent declarations fail
before transfer or model mutation.

Checkpoint event identity is structured as source, target, and fragment.
Formatting it into a string is for diagnostics only; scope resolution must not
parse the legacy display string.

## Lifecycle

```text
declare scope
  -> normalize and validate declaration
  -> resolve rank-local effective manifest
  -> validate checkpoint layer closure
  -> initialize only affected state
  -> receive chunks and reject out-of-scope names
  -> replay/process affected layers or stage a complete adapter
  -> reconcile expected == received for explicit scopes
  -> produce a rank-local report
  -> caller collects all reports and decides commit/abort
```

The existing arena verification remains in its current location. This work
does not reorder verification relative to copy-back.

## LoRA integration

SkyRL path:

```text
LoRAAdapterScope(artifact declaration)
  -> load a staged LoRAModel from the shared path
  -> validate artifact/config/local tensors
  -> install with the existing load-in-place adapter path
```

VERL tensor path:

```text
LoRAAdapterScope(exact tensor_names)
  -> receive all buckets
  -> require received_names == tensor_names
  -> construct and validate the staged LoRAModel
  -> replace the adapter once
```

If a dummy base model has not received its first complete real base update,
adapter-only replacement is rejected.

## Compatibility

- Omitting a scope retains full checkpoint behavior.
- Existing `expected_names` is accepted during migration, but when an explicit
  scope also supplies source/tensor names the declarations must agree.
- Exact completion is enabled for explicit scopes. Legacy full reload keeps
  its current compatibility behavior until all loaders emit structured
  receipts.

## Implementation sequence

1. Add scope types, normalization, structured event identities, and
   scope-aware reports.
2. Resolve checkpoint source subsets, enforce whole-layer closure, initialize
   only affected layers, and reconcile the effective manifest.
3. Add exact kernel-target completion.
4. Add LoRA artifact/tensor manifests and staged adapter replacement.
5. Return reports from workers and enforce the all-rank decision above the
   worker; invalidate prefix, multimodal, and encoder caches after commit.

## Required tests

- legacy full reload remains unchanged;
- one or multiple complete checkpoint layers can be selected;
- a split QKV, quantized layer, or expert processing unit is rejected before
  initialization;
- unknown, missing, duplicate, and unexpected names fail;
- PP no-op and TP/EP rank-local resolution are valid;
- kernel target subsets require exact names, shapes, and dtypes;
- path-backed and tensor-backed LoRA replacement succeed;
- incomplete A/B pairs, manifest/config mismatch, and adapter update before a
  real dummy-base sync fail without removing the old adapter;
- multi-bucket LoRA installs exactly once after all tensors arrive;
- cache invalidation and all-rank report enforcement occur before resume.
