# Runtime-format weight reload

## Goal

Update FP8/FP4 serving weights without restoring checkpoint tensors and without
calling `process_weights_after_loading` (PWAL).

## Protocol contract

Set:

```python
WeightTransferConfig(backend="ipc", weight_format="runtime")
```

Every transmitted tensor must already match the receiving rank's live runtime
schema:

- runtime Parameter/Buffer name;
- TP/EP-local shape;
- dtype and packed representation;
- backend-specific FP8/FP4 weight and scale layout.

Registered Parameters/Buffers use their normal fully-qualified names. A plain
tensor attribute may also be addressed by its fully-qualified attribute name.
Reload-arena tensors use:

```text
@reload_arena/<module-name>:<slot-name>
```

This is required for FP4 backends whose final gscales/input scales live in the
arena or as runtime tensor aliases rather than checkpoint Parameters.

Consequently, checkpoint FP8/FP4 tensors cannot be sent directly when the
selected backend requires shuffle, transpose, scale swizzle, or repacking. That
conversion must happen before transfer, normally by exporting tensors from a
model initialized with the same backend and parallel configuration.

## Receiver lifecycle

```text
start_weight_update
    -> open RuntimeReloadSession

update_weights(chunk)
    -> resolve live Parameter/Buffer
    -> validate the next tensor
    -> target.copy_(received_tensor)

finish_weight_update
    -> close the session
```

The receiver does not allocate checkpoint or runtime staging tensors. NCCL
broadcasts directly into the live Parameter/Buffer; IPC copies directly from
the sender's mapped allocation. It does not call `model.load_weights`, PWAL,
quantization, repacking, or kernel rebuild. Existing Parameter/Buffer objects
and storage addresses remain unchanged.

Packed NCCL/IPC is rejected in runtime mode because packing necessarily adds a
staging buffer. Transfers must use the unpacked per-tensor protocol.

## Failure semantics

Because updates are streamed directly and no rollback buffer exists, a failure
cannot restore tensors already written earlier in the same or a previous
chunk. The worker must be treated as tainted and removed from serving until a
complete runtime-format update or restart succeeds.

## Parallelism

Runtime tensors are rank-local. For TP/EP deployments, the sender must deliver
the correct shard and expert layout to each worker. Broadcasting one full
checkpoint tensor to every rank is not a valid runtime-format update.

The current NCCL dense backend has one broadcast source and is therefore
restricted to TP=PP=1 in runtime mode. IPC can select a per-GPU handle and is
the supported transport for rank-local model-parallel runtime tensors.

## Verification

Unit tests cover FP8 tensors, packed FP4 tensors, scales, ordinary tensor
attributes, reload-arena slots, direct NCCL destinations, IPC mapped sources,
packed-transfer rejection, and PWAL traps.

An H200 end-to-end probe exported the backend-native state of a cold-loaded FP8
model and directly applied 422 runtime tensors to another model. Results:

- PWAL was replaced with a function that raises on every call; reload passed;
- GPU allocator delta during reload was zero;
- no Parameter object or storage address changed;
- inference changed from the source checkpoint's result to the target result;
- the same result passed after full and piecewise CUDA Graph capture.

Native NVFP4 execution cannot be tested on H200. Its packed weights and
registered/plain/arena scale destinations are covered by the runtime schema
tests; hardware execution requires a Blackwell-class GPU.
