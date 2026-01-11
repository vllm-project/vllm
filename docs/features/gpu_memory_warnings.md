# GPU Memory Warnings

vLLM includes an optional GPU memory monitoring system that warns users when GPU memory usage exceeds a configurable threshold. This can be helpful for preventing Out-Of-Memory (OOM) crashes by providing early warnings.

## Enabling Warnings

To enable GPU memory warnings, use the `--enable-gpu-memory-warning` flag:

```bash
vllm serve facebook/opt-125m --enable-gpu-memory-warning
```

## Configuration

You can configure the warning threshold using `--gpu-memory-warning-threshold` (default: 0.9, i.e., 90%):

```bash
vllm serve facebook/opt-125m \
    --enable-gpu-memory-warning \
    --gpu-memory-warning-threshold 0.85
```

## How It Works

When enabled, vLLM periodically checks the GPU memory usage (reserved memory vs total memory). If the usage ratio exceeds the threshold, a warning log is emitted.

Example warning:

```text
WARNING 01-06 21:00:00 gpu_memory_monitor.py:134] GPU 0 memory usage high: 92.5% (reserved: 3.65GB / 3.95GB, allocated: 3.50GB). Consider reducing --max-num-seqs, --max-model-len, or using a smaller model to avoid OOM.
```

To prevent log spam, warnings are rate-limited (default: once every 60 seconds).

## When to Use

This feature is particularly useful when:

- Running on GPUs with limited VRAM.
- Experimenting with new models or configurations.
- Debugging OOM issues.

It has zero overhead when disabled (default).
