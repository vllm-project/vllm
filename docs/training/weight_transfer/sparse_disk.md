# Sparse Disk

The sparse disk backend moves sparse weight patches through storage instead of a
trainer-to-inference NCCL group. While the engine serves, the trainer publishes
each version's patches as one file. In the update, every inference host applies
the new files to its own writable copy of the checkpoint, and every worker
reloads the whole model from that copy, as a disk reload would.

Use it when the trainer and inference cannot share an NCCL group, or when many
replicas pull the same update. Because each update is a full reload, it works
for every configuration that reloads from checkpoint files, including FP8
checkpoints and weights that are re-laid out after loading. The pause lasts as
long as reading and applying the new files and that reload, however small the
patch.

## Usage

The inference side only selects the backend:

```python
llm = LLM(model=model_path, weight_transfer_config=WeightTransferConfig(backend="sparse_disk"))
```

The trainer passes patches in checkpoint coordinates, as for
[sparse NCCL](nccl.md#sparse-nccl):

```python
from vllm.distributed.weight_transfer import WeightTransferTrainerFactory
from vllm.distributed.weight_transfer.sparse_disk_engine import (
    SparseDiskTrainerInitInfo,
)

engine = WeightTransferTrainerFactory.trainer_init(
    SparseDiskTrainerInitInfo(
        patch_dir="/shared/run-42/patches",   # every inference host reads this path
        copy_dir="/local_nvme/vllm-copy",     # host-local, on every inference host
        rank=0,
    ),
    client=client,
)

engine.publish(patches)   # while the engine serves
pause_generation()
engine.send_weights()     # apply the new version and reload
resume_generation()
```

`publish` writes the next version (1, 2, ...) with `save_sparse_patches`;
`send_weights` runs start, update and finish for the last published version. An
orchestrator that publishes files itself can call `save_sparse_patches(patches,
patch_dir, version)` and drive the client with `{"version": version}`.

## Storage

- **Patch files.** `patch_dir/patch-<version>.safetensors` holds the changes from
  `version - 1`: flat indices and replacement values per checkpoint tensor.
  Versions are consecutive and each is written once, in place and without a
  rename, as object-store mounts need. A host that builds its copy replays every
  file after its servers' `base_version`, so keep the files after the oldest
  checkpoint a server may still start from.
- **Base checkpoint.** Version `base_version` is the checkpoint the servers
  started from, and the first file holds the changes from exactly that
  checkpoint; nothing checks this. A trainer that restarts needs a new
  `patch_dir`, and servers restarted from the checkpoint it resumes from.
- **Host copy.** `init_weight_transfer_engine` copies the startup checkpoint's
  safetensors files into `copy_dir`, once per host. Every rank and server on the
  host that follows the same `patch_dir` shares one copy. One worker at a time
  writes it, and never while a reload on the host still reads it: an update
  that would have to is refused, as is one to a version older than the copy's.
  So servers that share a copy must be updated together, to the same version;
  servers that update on their own schedule need their own `copy_dir`.
  `copy_dir` must be on a local filesystem (not NFS or Lustre). It takes the
  checkpoint's size on disk, and the warm copy stays in the page cache, which
  container memory limits count; a tmpfs `copy_dir` avoids disk writes when
  memory allows. At about 1% density nearly every page of the copy is rewritten,
  so write-back is about one model size per version.

## Versions and Recovery

- A server that starts later, or restarts, serves its startup checkpoint until
  it is initialized and updated: send it `init_weight_transfer_engine` with
  `patch_dir`, `copy_dir` and its checkpoint's `base_version`, then update it to
  `{"version": <current>}`. It reuses the host's copy, or rebuilds it from its
  startup checkpoint and replays the files after `base_version`.
- An apply that fails or is interrupted leaves the copy marked unfinished; no
  update to an older version loads it, and the next update redoes that version.
- A file with a bad patch is refused before the copy is touched, on every host,
  and no later version can be applied past it. Delete the file, set the trainer
  engine's `version` back by one and publish the corrected patches.
- A bad or missing file is found in `update_weights`, after
  `start_weight_update`, so the weight update stays incomplete, as after any
  failed update: keep generation paused until an update succeeds.
- After a host reboot the copy is rebuilt, because writes still in the page cache
  may have been lost.
- Versions only move forward. To go back, restart the servers from a full
  checkpoint and use a new `patch_dir`.

## Limitations

- Needs the default loader and a safetensors checkpoint.
- Draft model weights are not updated.
- `init_weight_transfer_engine` copies the whole checkpoint on a host without a
  copy, and the server generates nothing until it returns: initialize before
  sending traffic, and give `HTTPVLLMWeightSyncClient` a `timeout` longer than
  that copy (default 300 s).
