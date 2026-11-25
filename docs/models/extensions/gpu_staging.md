# Loading Safetensors Weights Through the GPU (GPU Staging)

GPU staging is a mode of the default safetensors loader that reads checkpoint
tensors straight to the GPU, so that the weight loaders slice, pad and copy
them on the GPU instead of on the CPU. It speeds up loading with tensor
parallelism, where every rank reads whole tensors and keeps only its shard.

## When it is used

It is on by default on CUDA when all of these hold:

- `--tensor-parallel-size` is greater than 1 and expert parallelism is off
  (it measured no faster at TP 1 or with `--enable-expert-parallel`);
- the load format is `safetensors` (or `auto` resolving to it), with the
  `lazy` or `prefetch` `--safetensors-load-strategy` or none;
- `enable_multithread_load` is not set.

Otherwise weights load as before.

## Turning it off or on

```bash
# Off:
vllm serve <model> --tensor-parallel-size 8 \
    --model-loader-extra-config '{"gpu_staging": false}'
```

`{"gpu_staging": true}` asks for it explicitly: it then also applies on other
platforms that expose CUDA devices, and raises an error at startup when the
load strategy or `enable_multithread_load` rules it out.

## How it works

Each rank reads adjacent tensors in merged ranges of up to 64 MiB with
`preadv` (8 threads) into pinned host buffers, copies each range to the GPU on
a side stream, and hands each tensor to the model's weight loader as a GPU
tensor with the checkpoint's dtype and shape. Tensors the model reports as
unused (`is_unused_checkpoint_weight`) are not read. Tensors over 512 MiB and
tensors of other dtypes are loaded as before.

Memory cost: 3 pinned host buffers and 3 GPU buffers, each the size of the
largest range read (at most 512 MiB), held until the load ends.
