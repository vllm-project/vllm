# CPU KV Cache Offloading

## Overview

This enhanced vLLM fork includes optimizations for CPU KV cache offloading that enable **dramatically larger context windows** by utilizing system RAM. This allows you to exceed GPU memory limitations and achieve context sizes 30%+ larger than the base configuration.

## Key Features

- **Extended Context Windows**: Achieve 52k+ tokens on a 24GB GPU (RTX 4090)
- **Smart Memory Validation**: Automatic accounting for both GPU and CPU offload capacity
- **Optimized Memory Transfers**: Intelligent pinned/unpinned memory allocation to prevent GPU OOM
- **Production Ready**: Includes web management interface and systemd service integration
- **Minimal Performance Impact**: ~15% throughput reduction with proper configuration

## Performance Results

**Test Configuration:**
- GPU: NVIDIA RTX 4090 (24GB)
- RAM: 128GB DDR4
- Model: Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit (~15GB)
- CPU Offload: 50,000 blocks (~78GB RAM)

**Achieved:**
- Max Context: 52,000 tokens (vs 39,344 baseline = **+32% increase**)
- Throughput: ~45 tokens/second
- GPU Memory: 22.37 GB
- CPU Memory: ~78 GB for KV cache

## Implementation Details

### Patch #1: Memory Validation Enhancement

**File**: `vllm/v1/core/kv_cache_utils.py` (lines 698-712)

Added CPU offload capacity recognition during KV cache memory validation:

```python
# PATCH: Account for CPU offload memory if configured
if (vllm_config.kv_transfer_config is not None and
    vllm_config.kv_transfer_config.kv_connector == "OffloadingConnector"):
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    if "num_cpu_blocks" in extra_config:
        num_cpu_blocks = extra_config["num_cpu_blocks"]
        # Calculate CPU offload capacity
        sample_spec = next(iter(kv_cache_spec.values()))
        cpu_offload_bytes = num_cpu_blocks * sample_spec.page_size_bytes
        available_memory += cpu_offload_bytes
        logger.info(
            f"CPU offload enabled: Adding {cpu_offload_bytes / GiB_bytes:.2f} GiB "
            f"({num_cpu_blocks} blocks) to available KV cache memory. "
            f"Total available: {available_memory / GiB_bytes:.2f} GiB"
        )
```

### Patch #2: Smart Pinned Memory Allocation

**File**: `vllm/v1/kv_offload/worker/cpu_gpu.py` (lines 95-114)

Prevents GPU OOM when allocating large CPU tensors:

```python
# PATCH: Disable pinned memory for large allocations (>10GB)
tensor_size_bytes = 1
for dim in cpu_shape:
    tensor_size_bytes *= dim
tensor_size_bytes *= gpu_tensor.element_size()
tensor_size_gb = tensor_size_bytes / (1024**3)

use_pin_memory = pin_memory and tensor_size_gb < 10.0
if not use_pin_memory and pin_memory:
    logger.info(
        "Disabling pinned memory for large CPU tensor (%.2f GB) "
        "to avoid GPU memory pressure", tensor_size_gb)

self.cpu_tensors.append(
    torch.zeros(cpu_shape,
                dtype=gpu_tensor.dtype,
                device="cpu",
                pin_memory=use_pin_memory))
```

## Quick Start

### Basic Usage (Python API)

```python
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Configure CPU offloading
kv_transfer_config = KVTransferConfig(
    kv_connector="OffloadingConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "num_cpu_blocks": 50000,  # ~1.5GB per layer
        "block_size": 16,
    },
)

# Initialize LLM with CPU offload
llm = LLM(
    model="path/to/your/model",
    dtype="auto",
    max_model_len=52000,  # Larger context!
    gpu_memory_utilization=0.88,
    enforce_eager=True,
    max_num_seqs=16,
    tensor_parallel_size=1,
    kv_transfer_config=kv_transfer_config,
    enable_prefix_caching=True,
)

# Generate with large context
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
outputs = llm.generate(["Your prompt here"], sampling_params)
print(outputs[0].outputs[0].text)
```

### Production Deployment (OpenAI API Server)

Use the included production launcher: `tools/vllm_cpu_offload_server.py`

```python
#!/usr/bin/env python3
import sys
import runpy
from vllm.config import KVTransferConfig

# Configure CPU offloading
kv_transfer_config = KVTransferConfig(
    kv_connector="OffloadingConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "num_cpu_blocks": 50000,
        "block_size": 16,
    },
)

# Set up API server arguments
sys.argv = [
    "vllm.entrypoints.openai.api_server",
    "--model", "/path/to/your/model",
    "--dtype", "auto",
    "--max-model-len", "52000",
    "--gpu-memory-utilization", "0.88",
    "--enforce-eager",
    "--max-num-seqs", "16",
    "--tensor-parallel-size", "1",
    "--enable-prefix-caching",
    "--host", "0.0.0.0",
    "--port", "8000",
]

# Inject KVTransferConfig via monkey-patching
import vllm.engine.arg_utils
original_create_engine_config = vllm.engine.arg_utils.EngineArgs.create_engine_config

def patched_create_engine_config(self, *args, **kwargs):
    if self.kv_transfer_config is None:
        self.kv_transfer_config = kv_transfer_config
    return original_create_engine_config(self, *args, **kwargs)

vllm.engine.arg_utils.EngineArgs.create_engine_config = patched_create_engine_config

# Run the API server
if __name__ == "__main__":
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
```

### Systemd Service Integration

Create `/etc/systemd/system/vllm.service`:

```ini
[Unit]
Description=vLLM Large Language Model Server with CPU Offloading (52k context)
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser

Environment="PATH=/home/youruser/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

ExecStart=/home/youruser/miniconda3/bin/python3 /path/to/vllm_cpu_offload_server.py

Restart=on-failure
RestartSec=10

LimitNOFILE=65535
StandardOutput=append:/var/log/vllm.log
StandardError=append:/var/log/vllm.err

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable vllm.service
sudo systemctl start vllm.service
```

## Configuration Guidelines

### Calculating `num_cpu_blocks`

The total CPU memory used is:
```
CPU Memory = num_cpu_blocks × num_layers × block_size_bytes
```

For a typical 30B parameter model with 48 layers:
- Each block is ~32KB
- 50,000 blocks = 50,000 × 48 × 32KB ≈ **78GB RAM**

**Recommendations:**
- Start with 50,000 blocks for 128GB RAM systems
- Adjust based on available RAM: `num_cpu_blocks = (available_ram_gb × 1024³) / (num_layers × 32000)`
- Leave at least 20-30GB for OS and other processes

### GPU Memory Utilization

- Use `gpu_memory_utilization=0.85-0.90` for optimal performance
- Lower values (0.80) if you experience GPU OOM
- Higher values (0.92) may work on newer GPUs with better memory management

### Context Window Sizing

Maximum context depends on:
1. GPU memory available for model weights
2. CPU offload capacity (num_cpu_blocks)
3. Number of concurrent sequences (max_num_seqs)

Formula (approximate):
```
max_tokens ≈ (gpu_kv_cache + cpu_offload_bytes) / (kv_cache_per_token × max_num_seqs)
```

## Web Management Interface

This fork includes a FastAPI-based web management interface at `tools/vllm_manager.py`.

**Features:**
- Real-time monitoring (GPU, CPU, Memory, Disk)
- Service control (Start/Stop/Restart)
- Model switching
- Context window configuration
- Auto-detection of running services

**Start the manager:**
```bash
cd /path/to/vllm-fork
python tools/vllm_manager.py
```

Access at `http://localhost:7999/`

## Troubleshooting

### Error: "CUDA out of memory" during startup

**Cause:** Pinned memory allocation for large CPU tensors consumes GPU memory.

**Solution:** This is fixed by Patch #2. Verify you're using the patched version.

### Error: "ValueError: KV cache is larger than available memory"

**Cause:** Memory validation doesn't recognize CPU offload capacity.

**Solution:** This is fixed by Patch #1. Verify you're using the patched version.

### Low Performance / Slow Generation

**Possible causes:**
1. **Too many concurrent sequences**: Reduce `max_num_seqs` to 8-16
2. **Excessive CPU-GPU transfers**: Ensure `num_cpu_blocks` is not unnecessarily large
3. **CPU bottleneck**: Monitor CPU usage; upgrade to faster RAM if needed

**Expected performance:**
- ~45 tokens/sec with 52k context on RTX 4090
- ~15% slower than baseline without CPU offload

### Out of RAM

**Cause:** `num_cpu_blocks` too high for available system memory.

**Solution:** Reduce `num_cpu_blocks`:
```python
# For 64GB RAM systems
"num_cpu_blocks": 25000,  # ~38GB

# For 128GB RAM systems
"num_cpu_blocks": 50000,  # ~78GB

# For 256GB RAM systems
"num_cpu_blocks": 100000,  # ~156GB
```

## Examples

See `examples/cpu_offload_example.py` for a complete working example.

## Performance Benchmarks

| Configuration | Max Context | Throughput | GPU Memory | CPU Memory |
|--------------|-------------|------------|------------|------------|
| Baseline (no offload) | 39,344 | 53 tok/s | 23.5 GB | - |
| CPU Offload (50k blocks) | 52,000 | 45 tok/s | 22.4 GB | ~78 GB |
| Improvement | **+32%** | -15% | -1.1 GB | +78 GB |

**Hardware:** RTX 4090 (24GB), 128GB DDR4, Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit

## Technical Details

### Memory Layout

```
┌─────────────────────────────────────┐
│         GPU Memory (24GB)           │
├─────────────────────────────────────┤
│  Model Weights: ~15GB               │
│  GPU KV Cache: ~3GB                 │
│  Activation Memory: ~4GB            │
│  Reserved: ~2GB                     │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│       CPU Memory (128GB)            │
├─────────────────────────────────────┤
│  OS + Services: ~20GB               │
│  CPU KV Cache Offload: ~78GB        │
│  Available: ~30GB                   │
└─────────────────────────────────────┘
```

### Transfer Mechanism

1. **Prefill Phase**: KV cache blocks allocated in GPU memory
2. **Eviction**: Least-recently-used blocks transferred to CPU RAM
3. **Retrieval**: Needed blocks transferred back to GPU on demand
4. **Optimization**: Unpinned memory for large transfers (>10GB) to save GPU resources

## Contributing

Found an issue or have an improvement? Please open an issue or PR at:
https://github.com/datagram1/vllm

## License

Same as upstream vLLM: Apache 2.0

## Acknowledgments

- Original vLLM project: https://github.com/vllm-project/vllm
- CPU offload implementation based on vLLM PR #27770
- Optimizations developed for production deployment on RTX 4090 + 128GB RAM systems
