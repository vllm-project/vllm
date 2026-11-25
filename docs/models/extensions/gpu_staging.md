# Loading Safetensors Weights Through the GPU (GPU Staging)

GPU staging is an opt-in mode of the default safetensors loader that reads
checkpoint tensors straight to the GPU, so that the weight loaders slice, pad
and copy them on the GPU instead of on the CPU. It speeds up loading with
tensor parallelism, where every rank reads whole tensors and keeps only its
shard. It is off by default.

## Enabling

```bash
vllm serve <model> --tensor-parallel-size 8 \
    --model-loader-extra-config '{"gpu_staging": true}'
```

It works with the default `safetensors` load format and the `lazy` or
`prefetch` `--safetensors-load-strategy` (or none), and can't be combined with
`enable_multithread_load`. It needs CUDA.

With `--tensor-parallel-size 1` or `--enable-expert-parallel`, the option is
ignored (with one log line): it measured no faster there.

## How it works

Each rank reads adjacent tensors in merged ranges of up to 64 MiB with
`preadv` (8 threads) into pinned host buffers, copies each range to the GPU on
a side stream, and hands each tensor to the model's weight loader as a GPU
tensor with the checkpoint's dtype and shape. Tensors the model reports as
unused (`is_unused_checkpoint_weight`) are not read. Tensors over 512 MiB and
tensors of other dtypes are loaded as before.

Memory cost: 3 pinned host buffers and 3 GPU buffers, each the size of the
largest range read (at most 512 MiB), held until the load ends.
