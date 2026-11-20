# Staging Weights

`stage_weights` lets a backend prepare the next weight version on the inference
nodes while the engine keeps serving, so the pause only covers the apply. None of
the in-tree engines (NCCL, IPC, sparse NCCL, sharded RDT, NCCL M2N) stage: their
transfers already fit inside the pause.

A round with staging looks like this:

```text
publish (trainer side)             engines serving
client.stage_weights(info)         engines serving
client.pause_generation(mode)
start → update_weights → finish    apply only
client.resume_generation()
```

`stage_weights` on `AsyncLLM`, `LLM`, the `/stage_weights` route and
`VLLMWeightSyncClient` returns once every worker has finished staging, or raises
the first worker's staging error.

## When to Use It

- **Inference-side work on slow sources.** Anything that prepares the next
  version on the inference nodes and lands outside the GPU: fetching a
  checkpoint or delta from disk or object storage to local disk or NVMe,
  decoding or decompressing it, or filling pinned host memory.
- **Not trainer-side work.** Publishing, encoding deltas and uploading stay in
  `send_weights`, before the `stage_weights` call.
- **Not GPU-to-GPU transfers** (NCCL, IPC, NIXL/RDMA from trainer GPUs). They
  fit inside the pause, and staging them needs a second full copy of the weights
  in GPU memory plus an extra copy at apply time. Pull straight into live
  storage between `start_weight_update` and `finish_weight_update` instead.

## Implementing It

Set `supports_staging = True` on your `WeightTransferEngine` and override
`stage_weights(update_info)`, which takes the same typed update info as
`receive_weights`. Write plain blocking code: the worker runs it on a background
thread and `receive_weights` can assume the staged data is ready.

```python
class MyEngine(WeightTransferEngine[MyInitInfo, MyUpdateInfo]):
    supports_staging = True

    def stage_weights(self, update_info: MyUpdateInfo) -> None:
        # Fetch and decode into host memory reserved at init.
        self._staged = fetch_and_decode(update_info.version, into=self._host_buffer)

    def receive_weights(self, update_info: MyUpdateInfo) -> None:
        # Copy what stage_weights prepared into the live weights.
        ...
```

## Ordering

- Call `stage_weights` only outside a `start_weight_update` /
  `finish_weight_update` window; inside one it raises.
- `start_weight_update` waits for staging in progress and re-raises its error.
  An error is reported once, by `stage_weights` or by `start_weight_update`.
- Calling it again overwrites the previous staged version: the new call runs
  after the one in flight.
- Engine shutdown during staging relies on timeouts, as for `update_weights`.

## Rules While the Engine Serves

- **Don't touch live state.** Don't write model parameters, buffers or
  attributes, start a reload, or change anything the forward pass reads.
- **Don't build or capture a load plan.** `plan_loads` runs between
  `start_reload` and `finish_reload`, on the live model, so the first update and
  any update after a layout change are unstaged. vLLM enforces this:
  `start_reload` and `plan_loads` raise on the staging thread.
- **No collectives.** Don't use the model's process groups or start NCCL
  collectives. Use point-to-point transports, file I/O or copies.
- **Keep Python light.** The thread shares the GIL with the worker loop. Do
  heavy work (I/O, decompression, hashing, copies) in native code that releases
  the GIL.
- **Or move Python-heavy work to a subprocess.** `stage_weights` may start a
  subprocess and wait on it; the waiting thread releases the GIL. Hand results
  back through files, shared memory or pinned host buffers. This doesn't work
  when the staged data must be a CUDA tensor in the worker.
- **Avoid GPU memory.** The KV cache is sized at startup to fill the GPU. Stage
  into pinned host memory, allocated once and reused.
- **Keep the GPU idle.** If staging must touch the GPU, use copies only, on a
  side stream, and no device-wide syncs.
- **Bound CPU and host RAM.** Cap thread pools and memory: the worker's
  CPU-side scheduling runs at the same time.
- **Bound bandwidth.** Staging traffic shares disk, NIC and PCIe with serving
  traffic (EP all-to-all, KV transfer).
