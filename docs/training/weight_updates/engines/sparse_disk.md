# Sparse Disk

The sparse disk backend moves sparse weight patches through storage instead of a
trainer-to-inference NCCL group. While the engine serves, the trainer publishes
each version's patches as one file. Every inference worker keeps its own part of
the checkpoint in host memory, laid out as its load plan writes it. In the
update, the worker applies the new files to that copy and copies the whole part
into the model, so every update is a full direct reload through the load plan,
with post-load processing rerun.

Use it when the trainer and inference cannot share an NCCL group, or when many
replicas pull the same update. Inside one NVLink domain, a dense NCCL update is
usually faster. [ModelExpress](modelexpress.md) also installs deltas from object
storage, through its own service; this backend needs only a shared directory.

## Usage

The inference side only selects the backend; `VLLM_SERVER_DEV_MODE=1` exposes
the HTTP weight-transfer endpoints:

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve my-model \
    --weight-transfer-config '{"backend": "sparse_disk"}'
```

The trainer passes patches in checkpoint coordinates, as for
[sparse NCCL](nccl.md#sparse-nccl), with strictly increasing indices per tensor:

```python
from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.sparse_disk_engine import (
    SparseDiskTrainerInitInfo,
)

engine = WeightTransferTrainerFactory.trainer_init(
    SparseDiskTrainerInitInfo(
        patch_dir="/shared/run-42/patches",   # every inference host reads this path
        rank=0,
    ),
    client=HTTPVLLMWeightSyncClient("http://localhost:8000"),
)
engine.send_weights()     # once, before serving: builds the host copies

engine.publish(patches)   # while the engine serves
engine.send_weights()     # between /pause and /resume: apply the version and reload
```

`publish` writes the next version (1, 2, ...) with `save_sparse_patches`;
`send_weights` runs start, update and finish for the last published version. An
orchestrator that publishes files itself can call `save_sparse_patches(patches,
patch_dir, version)` and drive the client with `{"version": version}`.

## Storage and the host copy

- **Patch files.** `patch_file(patch_dir, version)`, named
  `patch-<version, 12 digits>.safetensors`, holds the changes from
  `version - 1`; write it with `save_sparse_patches`. Versions are consecutive
  and each is written once, in place and without a rename, as object-store
  mounts need. A file is read only in the update, so place it on the inference
  host (or in its page cache) before the pause.
- **Base checkpoint.** Version `base_version` is the checkpoint the servers
  started from, and the first file holds the changes from exactly that
  checkpoint; nothing checks this, and nothing checks the weights after an
  update either. A trainer that restarts needs a new `patch_dir`, and servers
  restarted from the checkpoint it resumes from.
- **Host copy.** The first update builds the load plan, as dense NCCL with
  `use_load_plan` does, and copies every part of the startup checkpoint the
  worker loads into host memory, pinned where available: about the checkpoint
  size divided by the tensor-parallel size per worker, for every server. That
  update also reads the checkpoint, so send it before serving (the
  `send_weights()` above, at `base_version`). Later updates apply the new files
  and copy the worker's part into the model. Applying maps the patch on the
  worker's device, in chunks sized to its free memory. Pinned host memory cannot
  be swapped or reclaimed; a tensor that is not sharded, or that the load plan
  leaves to `load_weights`, is held whole by every worker.
- **Names.** Patches name checkpoint tensors, as the default loader yields
  them. A patch of a tensor the model does not load, such as a tied alias, is
  checked and ignored.

## Versions and Recovery

- A server that starts later, or restarts, serves its startup checkpoint until
  it is initialized and updated: send it `init_weight_transfer_engine` with
  `patch_dir` and its checkpoint's `base_version`, then update it to
  `{"version": <current>}`. That update builds the copy and replays every file
  after `base_version`, so its time grows with the number of versions: save a
  full checkpoint every so often and start new servers from it. A server keeps
  the `base_version` of its first init.
- Sending the last update again reloads the copy, for example after a failed
  update or a level 2 sleep.
- Every worker checks the whole file before writing any of it, so a file with a
  bad patch is refused by every worker, with an error naming the file, and
  leaves every copy at the last version. Only such a file may be corrected
  under the same version: delete it and write it again with
  `save_sparse_patches`; after any other failed update, keep the file. Either
  way the model stays mid-update, so keep generation paused and send an update
  that succeeds. On tensor-parallel servers, restart the server first: the next
  request currently gets the failed update's error back. An apply that fails
  partway discards the copy, which the next update rebuilds.
- Versions only move forward. To go back, restart the servers from a full
  checkpoint and use a new `patch_dir`.

## Limitations

- Needs the default loader reading the checkpoint on the host (not
  `fastsafetensors` or `instanttensor`), no secondary weight sources, and a load
  plan: direct reload, without EPLB or online quantization. Other
  configurations are refused at `init_weight_transfer_engine`; a model whose
  loads cannot be planned fails at the first update.
- Draft model weights are not updated.
- The host copy takes host memory for every server, pinned where available, and
  is rebuilt after a restart.
