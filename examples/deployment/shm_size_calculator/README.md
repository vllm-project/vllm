# vLLM /dev/shm Size Calculator

A tool to estimate the recommended shared memory (`/dev/shm`) size for vLLM deployments using tensor parallel inference.

## Overview

When running vLLM with tensor parallelism (multiple GPUs sharing a single model), processes on the same machine communicate via shared memory. Docker and Kubernetes containers typically ship with a very small `/dev/shm` (often 64 MiB), which is insufficient for large language models.

This calculator estimates the minimum shared memory needed based on:
- Model size (parameters in billions)
- Weight precision (bf16, fp16, fp8, int8, int4)
- Tensor parallel degree (number of GPUs)
- Maximum concurrent requests
- Maximum sequence length

## Methodology

The calculation accounts for four IPC (inter-process communication) components:

### 1. NCCL Communication Buffers (Dominant)
- NCCL (NVIDIA Collective Communications Library) uses ring buffers for GPU-to-GPU data exchange during tensor parallel all-reduce operations.
- **Formula:** `model_size_bytes × 0.01 × num_gpu_pairs`
- Where `num_gpu_pairs = GPUs × (GPUs - 1) / 2`
- `model_size_bytes = params × 1e9 × bytes_per_param` (varies by dtype)
- This is the largest component because NCCL needs buffers proportional to model weight size.

### Weight Precision (Dtype) Impact
NCCL buffers scale linearly with the number of bytes per parameter. Lower-precision formats reduce shared memory proportionally:

| Dtype | Bytes per param | NCCL reduction vs BF16 |
|---|---|---|
| `bf16` / `fp16` | 2 | baseline |
| `fp8` / `int8` | 1 | 50% less |
| `int4` | 0.5 | 75% less |

Other components (PyTorch descriptors, KV cache metadata, MMAP buffers) are unaffected by dtype — they are negligible (< 0.05 GiB total).

### 2. PyTorch Tensor Descriptors
- Each GPU worker shares tensor metadata (pointers, shapes, data types) via shared memory.
- **Formula:** `num_layers × 1000 × 256_bytes × num_gpus`
- Approximately 1000 descriptors per transformer layer per worker.

### 3. KV Cache Metadata (Block Tables)
- vLLM uses PagedAttention with block tables. Each block table entry occupies shared memory.
- **Formula:** `max_requests × blocks_per_request × 8_bytes × num_gpus`
- Block size is 16 tokens; blocks per request = `ceil(max_sequence_length / 16)`.

### 4. MMAP Batch Exchange Buffers
- Token batches are exchanged between the scheduler and GPU workers via memory-mapped files.
- **Formula:** `max_requests × max_sequence_length × 2_bytes × num_gpus`
- 2 bytes per token (uint16 for token IDs).

### Safety Margin
A 30% safety margin is added to the total to account for runtime overhead and edge cases.

### What Is NOT Included
- **Model weights** — These reside entirely in GPU VRAM, not in `/dev/shm`.
- **KV cache values** — The actual KV cache tensors are in VRAM; only the metadata (block tables) is in shared memory.

## Quick Start

```bash
# Basic usage: 30B model, 4 GPUs, 20 concurrent requests, 8192 max sequence
python examples/deployment/shm_size_calculator.py \
    --model-params 30 \
    --tensor-parallel-size 4 \
    --max-concurrent-requests 20 \
    --max-seq-length 8192

# Output as JSON (for automation)
python examples/deployment/shm_size_calculator.py \
    --model-params 70 \
    --tensor-parallel-size 8 \
    --json

# FP8-quantized model (NCCL buffers halved)
python examples/deployment/shm_size_calculator.py \
    --model-params 70 \
    --tensor-parallel-size 8 \
    --dtype fp8
```

## CLI Reference

### Required Arguments
| Argument | Description |
|---|---|
| `--model-params N` | Model size in billions of parameters (e.g., `30`, `70`) |

### Optional Arguments
| Argument | Short | Description | Default |
|---|---|---|---|
| `--tensor-parallel-size N` | `--tp N` | Number of GPUs for tensor parallelism | `4` |
| `--max-concurrent-requests N` | `--requests N` | Maximum concurrent requests | `20` |
| `--max-seq-length N` | `--seq N` | Maximum sequence length | `8192` |
| `--dtype N` | — | Weight precision (bf16, fp16, fp8, int8, int4) | `bf16` |
| `--hidden-size N` | — | Model hidden size (auto-estimated if omitted) | auto |
| `--num-layers N` | — | Number of transformer layers (auto-estimated if omitted) | auto |
| `--num-attention-heads N` | — | Number of attention heads (auto-estimated if omitted) | auto |
| `--model-profile NAME` | — | Use a built-in model profile (see `--model-profiles`) | — |
| `--model-profiles` | — | List all available model profiles | — |
| `--json` | — | Output as JSON instead of formatted table | — |

### Built-in Model Profiles

| Profile | Hidden Size | Layers | Heads |
|---|---|---|---|
| `mistral-7b` | 4096 | 32 | 32 |
| `mistral-8x7b` | 4096 | 58 | 32 |
| `mistral-8x22b` | 6144 | 72 | 48 |
| `mistral-large` | 12288 | 72 | 96 |
| `mistral-medium` | 5120 | 40 | 32 |
| `llama-3.1-8b` | 4096 | 32 | 32 |
| `llama-3.1-70b` | 8192 | 80 | 64 |
| `llama-3.1-405b` | 16384 | 126 | 128 |
| `deepseek-v3` | 7168 | 61 | 128 |
| `qwen-2.5-72b` | 8192 | 80 | 64 |
| `gemma-2-27b` | 4608 | 46 | 32 |
| `phi-4` | 6144 | 44 | 48 |

### Examples

```bash
# Use a built-in profile
python examples/deployment/shm_size_calculator.py \
    --model-profile mistral-medium \
    --tensor-parallel-size 4

# Large model with custom architecture params
python examples/deployment/shm_size_calculator.py \
    --model-params 405 \
    --tensor-parallel-size 8 \
    --hidden-size 16384 \
    --num-layers 126 \
    --max-concurrent-requests 16 \
    --max-seq-length 32768

# List all available model profiles
python examples/deployment/shm_size_calculator.py --model-profiles
```

## When Is Shared Memory Needed?

Use `/dev/shm` when:
- **Tensor parallelism > 1** — Multiple GPUs share one model instance
- **Multi-node deployments** — Processes across nodes need IPC coordination
- **GPUDirect RDMA** — Direct GPU-to-GPU communication over InfiniBand/RoCE

Skip `/dev/shm` configuration when:
- **Single GPU** — No inter-GPU IPC required
- **CPU-only inference** — No NCCL communication involved

## Kubernetes Configuration

In a Helm chart or Kubernetes manifest, configure `/dev/shm` as a memory-backed `emptyDir`:

```yaml
volumes:
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: 64Gi

volumeMounts:
  - name: shm
    mountPath: /dev/shm
```

Or use the vLLM Helm chart's `shm` configuration:

```bash
helm install vllm . \
    --set shm.enabled=true \
    --set shm.size="64Gi"
```

## Limitations

- This is an **estimate** based on published PyTorch and NCCL internals. Actual usage may vary.
- The dtype selection affects NCCL buffer size linearly (see Dtype Impact table above).
- Custom or experimental models may have different IPC patterns.
- Production deployments should monitor actual `/dev/shm` usage via `df -h /dev/shm` and adjust accordingly.

