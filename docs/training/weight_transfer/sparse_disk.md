# Sparse Disk Engine

The `sparse_disk` weight transfer engine applies sparse, flat-index weight
patches that the trainer writes to a file on storage shared with the inference
workers. It carries the same patches as [sparse NCCL](nccl.md#sparse-nccl) and
applies them the same way; only the transport differs.

## When to Use Sparse Disk

- Training and inference are disaggregated and share a filesystem (a network
  volume, or a local disk when they are colocated) but not an NCCL group.
- The trainer has no GPU, or should not hold a GPU communicator open.
- Each update changes a small fraction of the weights, so a patch is much
  smaller than a checkpoint.

## How It Works

1. The trainer writes the round's patches to one safetensors file under
   `patch_dir`. The file appears under its final name only once it is complete.
2. The trainer calls `update_weights` with the file's absolute path.
3. Every inference worker reads the file onto its device and applies it through
   the model's native `load_weights()`, which maps checkpoint coordinates to
   rank-local tensor-parallel and packed runtime parameters.
4. The trainer deletes the file once the call returns.

Patches are applied in place, so the layerwise reload lifecycle is not used and
`start_weight_update` and `finish_weight_update` are no-ops.

## Usage

Select the backend on the inference side:

```bash
vllm serve my-model \
    --weight-transfer-config '{"backend": "sparse_disk"}'
```

On the trainer side, build the engine once and send each round's patches:

```python
from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.sparse_disk_engine import (
    SparseDiskTrainerInitInfo,
)
from vllm.distributed.weight_transfer.sparse_nccl_engine import SparseWeightPatch

engine = WeightTransferTrainerFactory.trainer_init(
    init_info=SparseDiskTrainerInitInfo(patch_dir="/shared/patches", rank=0),
    client=HTTPVLLMWeightSyncClient("http://localhost:8000"),
)

engine.send_weights(
    [
        SparseWeightPatch(
            name="model.layers.0.mlp.down_proj.weight",
            indices=flat_indices,           # int32 or int64, 1-D
            values=new_values,              # same length as indices
            full_shape=tuple(param.shape),
        )
    ]
)
```

Each `send_weights(patches)` call owns a complete `start` / `update` / `finish`
lifecycle. As with sparse NCCL, `send_weight_chunk(patches)` sends one chunk
inside a lifecycle the caller owns.

`patch_dir` must already exist and be visible at the same absolute path to the
trainer and to every inference worker.

## Patch File Format

A patch file is a safetensors file whose metadata has `format` set to
`vllm-sparse-patch-v1` and `patches` set to a JSON list of
`{"name": ..., "shape": [...]}` entries. Entry `i` has two tensors, `{i}.indices`
and `{i}.values`. Names, shapes, and indices are in checkpoint coordinates.

A process that does not use the trainer engine can write files with
`save_sparse_patches` and pass their paths to `update_weights` itself.

## Limitations

- The limits of the sparse loader apply: floating-point, unquantized weights
  whose final load step is a same-shaped copy.
- Every worker reads the whole file, so with tensor parallelism it is read once
  per rank.
- An update needs free GPU memory on every rank: the patch, plus the loader's
  full checkpoint-shaped tensor for the largest weight. Leave headroom with
  `--gpu-memory-utilization`.
- The caller owns diffing and restart or reseed after a partial failure. A
  failed update does not roll back patches that were already applied.
- Workers open whatever path `update_weights` names. Expose the weight transfer
  endpoints only to trusted callers.
