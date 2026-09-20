# Attention Sink Weight Update

## Summary

Some model `load_weights()` implementations load checkpoint-backed attention sink
parameters by copying directly into the parameter:

```python
param.data.copy_(narrow_weight)
params_dict[name][: n].copy_(narrow_weight)
```

These copies bypass the parameter's `weight_loader`, which is the only entry point
that the online weight reload path can observe. Cold (initial) load works, but
during RL / online weight update / layerwise reload the new sink value never
reaches the tensor storage the attention kernel reads, so the runtime keeps
serving the previous sink values.

This document specifies the defect, why the direct copy is invisible to the
reload framework, and a uniform fix: every checkpoint-backed, online-updatable
parameter must be loaded through the parameter's *current* `weight_loader`.

## Motivation

Layerwise reload (`vllm/model_executor/model_loader/reload/layerwise.py`)
supports updating model weights in place, which is what RL and online-serving
weight updates rely on. It can only replay work it saw pass through
`param.weight_loader`.

For each layer, `initialize_layerwise_reload`:

1. saves the live kernel tensors that attention kernels and CUDA graphs hold
   (`info.kernel_tensors`),
2. restores the layer's parameters and buffers onto the `meta` device
   (`restore_layer_on_meta`), and
3. replaces each parameter's `weight_loader` with `online_process_loader`, which
   records every load into `info.loaded_weights` and accumulates
   `info.load_numel`.

`finalize_layerwise_reload` then materializes each layer, replays the recorded
loads through the *original* loader, runs
`process_weights_after_loading`, and finally copies the processed result back
into the saved kernel tensor storage.

```text
Checkpoint
   |
   v
model-specific preprocessing        <- optional TP slicing / representation change
   |
   v
param.weight_loader                 <- the single contract boundary
   |
   +-- cold load
   |
   +-- online_process_loader (during live update)
            |
            v
      buffered in info.loaded_weights
            |
            v
      original loader replay
            |
            v
      kernel tensor storage
```

A direct `copy_` on a parameter removes that parameter from the pipeline. The
bug is therefore not that a copy happens, but that the copy is not reachable
through the loader contract.

## Problem

### The failure mode, measured

`params_dict` is built from `self.named_parameters()` at the start of
`load_weights()`, which runs *inside* the reload window. The entries are
therefore the `meta` parameters that `restore_layer_on_meta` installed, not the
live kernel tensors.

A bare `copy_` therefore writes into the parameter instance that the reload
materialized. It contributes nothing to `info.load_numel`, so the framework has
no record that the sink was loaded. When the layer is finalized the saved kernel
tensor is put back with `_place_kernel_tensors`; the sink bytes are only
transferred to it by `_copy_and_restore_kernel_tensors`, which runs on the
`_layerwise_process` path. Whether the update survives depends on incidental
details of when the sink load happens and whether the layer reaches
`load_numel >= load_numel_total`, which in turn depends on the order and
provenance of the other weights.

The observable result, measured through a real `initialize_layerwise_reload` /
`finalize_layerwise_reload` cycle with a padded sink, is not a clean stale value
but unwritten storage:

```text
OLD     got=[0.0, 0.0, 4235.352, 0.0]  want=[-0.0, -1.0, -2.0, -3.0]
        tail_is_neginf=False  buffered=['proj']  data_ptr_stable=True
helper  got=[-0.0, -1.0, -2.0, -3.0]   want=[-0.0, -1.0, -2.0, -3.0]
        tail_is_neginf=True   buffered=[]       data_ptr_stable=True
```

Two consequences follow, and both are worse than "keeps the old value":

1. The sink holds whatever the freshly allocated kernel tensor contained. A
   reload that reports success can silently substitute garbage for the sink.
2. The `-inf` padding tail is not re-established, because the load that would
   have written it was never seen. Padded heads can pick up arbitrary values
   instead of remaining disabled.

`data_ptr` is stable in both cases, so a pointer-identity check alone does not
detect the bug; the values must be asserted.

### Why the padding makes this non-trivial

DeepSeek V4 / V4.1 pad the sink parameter to the platform's padded head count
and initialize the padding to `-inf` so padded heads have no sink effect:

```python
self.attn_sink = nn.Parameter(
    torch.full((self.padded_heads,), -float("inf"), dtype=torch.float32),
    requires_grad=False,
)
```

The checkpoint slice for a rank has only the local heads, so the loader cannot
simply receive the sliced checkpoint tensor: `param.shape` is `padded_heads`
while the slice is `n_local_heads`, and the padding semantics would be lost. The
padding must be re-established on every load, not only at construction.

### Confirmed affected sites

20 direct-copy sites in 19 files:

| Model family | Files |
| --- | --- |
| DeepSeek V4 target | `deepseek_v4/{cpu,xpu,amd,nvidia}/model.py` |
| DeepSeek V4 MTP | `deepseek_v4/{xpu,amd,nvidia}/mtp.py` |
| DeepSeek V4 DSpark | `deepseek_v4/{xpu,amd,nvidia}/dspark.py` |
| DeepSeek V4.1 target | `deepseek_v41/{amd,nvidia}/model.py` |
| DeepSeek V4.1 DSpark | `deepseek_v41/{amd,nvidia}/dspark.py` |
| HY-V4 target / MTP | `hy_v4/nvidia/{model,mtp}.py` |
| MiMo-V2 target | `model_executor/models/mimo_v2.py` |
| GPT-OSS | `model_executor/models/gpt_oss.py` (3 quant paths) |

Not affected, and deliberately left alone: Granite and GraniteMoE (TP slicing
already lives in a `sharded_weight_loader` attached to the parameter), Qwen3-DFlash
and Laguna (they slice `loaded_weight` during preprocessing and then yield it to
`AutoWeightsLoader`, so `param.weight_loader` is still the entry point), and
MiMo-V2 MTP (already calls `param.weight_loader`).

Sites whose names merely contain `sink` but are not checkpoint parameters, such
as Sinkhorn configuration (`hc_sinkhorn_iters`, `hc_sinkhorn_eps`), runtime
backend arguments, and config flags, are out of scope.

## Proposed change

### Contract

> Every checkpoint-backed, online-updatable parameter must be loaded through the
> parameter's current `weight_loader`. Model `load_weights()` implementations
> must not write to such a parameter with a bare `copy_`.

Model-specific preprocessing 鈥?TP slicing, reshaping, dtype conversion,
reconstructing a padded representation 鈥?stays in `load_weights`. The loader
receives a tensor already in the parameter's runtime shape and therefore must
not slice again.

Reading the loader dynamically is required, because during a live update
`param.weight_loader` *is* `online_process_loader`:

```python
weight_loader = getattr(param, "weight_loader", default_weight_loader)
weight_loader(param, runtime_weight)
```

Caching the constructor-time loader would reintroduce the same bypass.

### DeepSeek: a shared padded sink loader

The padding reconstruction is identical across the 14 DeepSeek files, so it
belongs in one shared helper rather than 14 copies. The helper rebuilds the
padded runtime representation, keeping the padding tail at `-inf`, and then
routes the full-shape tensor through the current loader:

```python
def load_padded_attn_sink(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    head_start: int,
    head_end: int,
) -> None:
    """Load a TP-sliced checkpoint sink into a padded runtime sink parameter.

    Args:
        param: Padded runtime sink parameter of shape ``(padded_heads,)``.
        loaded_weight: Global checkpoint sink tensor.
        head_start: First global head index owned by this rank.
        head_end: One past the last global head index owned by this rank.
    """
    local_weight = loaded_weight[head_start:head_end]

    runtime_weight = torch.full(
        param.shape,
        -float("inf"),
        dtype=param.dtype,
        device=local_weight.device,
    )

    if local_weight.numel():
        runtime_weight[: local_weight.shape[0]].copy_(local_weight)

    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, runtime_weight)
```

`runtime_weight` is always materialized so it has the same number of elements as
`param`. This matters for two reasons:

- the online path wraps the loader in `get_numel_loaded`, which counts
  `torch.ops.aten.copy_` and asserts the source and destination element counts
  agree, so a narrow/short source would be rejected;
- the replay in `_layerwise_process` loads into a freshly materialized
  parameter, whose padding must be re-established by the load itself.

An expanded (broadcast) source is not used to save memory: it would satisfy the
element count but could let a loader write real values into the padding tail
instead of `-inf`.

Filling with `-inf` (rather than leaving `torch.empty` contents) is intentional.
During replay the destination is `torch.empty_strided`, so every byte of a newly
materialized parameter must be written by its loader; leaving the tail unwritten
would expose uninitialized memory. It also makes an under-filled tail fail loudly
rather than silently inheriting stale sink values.

The helper is defined once and imported by the V4 and V4.1 modules. V4.1 has no
classic MTP path, so no V4.1 MTP change is proposed.

### HY-V4

The HY-V4 sink parameter is already the local-head shape, so it needs no padding
reconstruction. TP slicing stays in `load_weights` and the sliced tensor is
passed to the current loader:

```python
if "learnable_sink_param" in name:
    param = params_dict[name]
    local_weight = loaded_weight[head_rank_start:head_rank_end]
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, local_weight)
    loaded_params.add(name)
    continue
```

The parameter must not additionally be given a `sharded_weight_loader`, which
would slice a second time.

### MiMo-V2

`mimo_v2.py` is brought in line with the already-correct `mimo_v2_mtp.py`:
narrow the checkpoint tensor by rank, then call the current loader. MTP is not
modified.

### GPT-OSS

All three load paths (`_load_weights_mxfp4`, `_load_weights_quark`,
`_load_weights_other`) have their own copy of the sink branch and must all be
fixed; fixing only the default quantization path would leave MXFP4 and Quark
checkpoints broken. The file already has a `_get_weight_loader` helper, which is
reused rather than duplicated.

While fixing these, `_load_weights_mxfp4` is also missing the
`loaded_params.add(name)` that the other two paths perform, so its return value
under-reports the sink.

## Consequences

### Correctness

- Live updates to attention sinks become reachable by the reload framework: the
  load is recorded, replayed, processed, and copied into the kernel tensor.
- The emitted `loaded_params` set becomes accurate for sink parameters, so
  callers that validate "all parameters loaded" see the truth.
- Padded heads remain `-inf` after every reload, so padded heads never
  accidentally acquire a sink.

### Compatibility

Cold-load behavior is intended to be unchanged: the same checkpoint slice reaches
the same parameter with the same value, and the padding tail keeps the same
`-inf` value it had at construction. The sink parameter's dtype, shape, and
device do not change.

### Cost

One extra full-shape temporary per sink parameter per load. Sink parameters are
`(padded_heads,)`, so this is a few kilobytes 鈥?negligible next to the
projections loaded alongside them. Its lifetime is a single load call.

### Risk

The main risk is a double TP slice: if a fixed site keeps a `sharded_weight_loader`
on the parameter while also slicing in `load_weights`, the shard is taken twice
and is silently wrong on ranks other than 0. The fix therefore requires the
slicing owner to be unique, and tests must cover `tp_rank != 0`.

## Alternatives

**Special-case sinks in `layerwise.py`.** Rejected. The framework is not wrong
about the contract; the model loaders violate it. Detecting sinks by name in the
framework would spread model knowledge into shared machinery and leave every
other direct-copy parameter broken.

**Route the sliced checkpoint tensor straight to the loader.** Rejected for
DeepSeek: shape mismatch against the padded parameter, loss of padding semantics,
and a rejected element-count assertion in `get_numel_loaded`.

**Give the sink parameter a `sharded_weight_loader`.** Rejected: that solves cold
load and layerwise reload for the unpadded case, but it cannot reconstruct the
`-inf` padding. It is the right answer for Granite, where the parameter is
unpadded.

**Keep the direct copy and have the framework diff every tensor.** Rejected:
defeats the purpose of the loader contract, is expensive, and still cannot tell
a caller-supplied iterator from checkpoint state.

## Testing

Testing must exercise a real layerwise reload rather than asserting only that a
mock loader was called, because the defect is precisely a load that the framework
never sees.

Required coverage:

- **Unit, CPU, no GPU needed.** A minimal module holding a padded sink parameter
  goes through `record_metadata_for_reloading` 鈫?`initialize_layerwise_reload` 鈫?
  `load_weights` 鈫?`finalize_layerwise_reload`. Assert that the sink takes the new
  checkpoint value, that `sink[:n_local]` equals the expected local slice, that
  `sink[n_local:]` is all `-inf`, and that the kernel tensor's `data_ptr` is
  unchanged across the reload.
- **Valid padding per rank.** Repeat with `tp_rank != 0` so a double slice or an
  off-by-one in `head_start`/`head_end` cannot hide behind rank 0.
- **TP=1 and TP=2**, ideally TP=4.
- **End-to-end oracle.** Cold-load checkpoint B and run a forward pass; then
  cold-load A, live-update to B, and run forward again; assert the two outputs
  agree. This is the only check that spans `load_weights` 鈫?
  `online_process_loader` 鈫?replay 鈫?`process_weights_after_loading` 鈫?kernel
  tensor 鈫?attention backend.
- **Regression guard.** Existing cold-load tests for the touched models must keep
  passing, and Granite / Qwen3-DFlash / Laguna / MiMo-V2 MTP must be confirmed
  untouched.

## Implementation plan

Each step is a separate pull request. Tasks 2鈥? are stacked on task 1 because
they import the shared helper; tasks 6鈥? are independent of task 1.

| Task | Scope | Depends on | PR |
| --- | --- | --- | --- |
| 1 | Shared DeepSeek padded sink helper + tests | RFC | #57797 |
| 2 | DeepSeek V4 target: `cpu`, `xpu`, `amd`, `nvidia` `model.py` | 1 | #57798 |
| 3 | DeepSeek V4 MTP: `xpu`, `amd`, `nvidia` `mtp.py` | 1 | #57799 |
| 4 | DeepSeek V4 DSpark: `xpu`, `amd`, `nvidia` `dspark.py` | 1 | #57800 |
| 5 | DeepSeek V4.1 target and DSpark | 1 | #57801 |
| 6 | HY-V4 target and MTP | RFC | #57803 |
| 7 | MiMo-V2 target, aligned with MTP | RFC | #57804 |
| 8 | GPT-OSS all three quantization load paths | RFC | #57805 |
| 9 | Audit: no checkpoint-backed sink load bypasses `weight_loader` | 2鈥? | #57806 |

Task 9 is an executable guard rather than a manual re-run of the search: it
rejects the three direct-write forms in every module that creates or loads a
sink, and fails when a sink parameter appears in a module that is not listed. It
was verified to catch all 20 original sites with line numbers and to pass on the
fixed tree.

## Status

Implementation is complete on the branches linked above, and the framework-level
behaviour is verified on H20 (CUDA 13.0, torch 2.13.0).

Verified:

- `ruff check` / `ruff format` against the repository config: clean.
- The full suite passes on H20: `43 passed`.
  - `tests/model_executor/model_loader/test_attn_sink_reload.py` drives the real
    layerwise lifecycle. It pins the defect
    (`test_direct_copy_loses_the_update_and_corrupts_padding`) and the fix
    (reaches kernel tensor storage, `data_ptr` stable, padding tail `-inf`,
    repeated reloads, load buffered by `online_process_loader`, meta-parameter
    load through the wrapped loader).
  - `tests/model_executor/model_loader/test_attention_sink_load_audit.py`
    catches the pre-fix revisions of all eight affected files with line numbers.
  - `tests/models/deepseek_v4/test_attn_sink_loader.py` covers the helper's
    padding and loader-routing semantics for every rank.
- The measurement above confirms this is not a cosmetic refactor: the pre-fix
  path leaves the sink unwritten and the padding tail unset.

Still outstanding:

- The end-to-end oracle described under Testing (cold-load B vs cold-load A plus
  live update to B, comparing forward outputs) needs a real DeepSeek checkpoint
  and the matching attention backend. Tracked as follow-up.
- The CPU and XPU backends cannot be exercised on the H20 cluster; those
  variants share the same helper and are covered structurally by the audit.

## Open questions

1. Should the padding constant be a parameter of the helper? Fixed `-inf` is what
   both V4 and V4.1 use today.
2. Should `layerwise.py` gain a cheap assertion that no parameter received a
   direct write during reload, to catch this class of bug generally? That is a
   follow-up; this RFC deliberately does not change the framework.
3. HY-V4 stores its sink as local heads; if that representation ever becomes
   padded it should adopt the same helper as DeepSeek.
