# Running vLLM on Intel BMG70 (Battlemage) GPUs

This guide describes how to set up and validate vLLM on Intel BMG70
(Battlemage) discrete GPUs using the Intel XPU backend.

## Prerequisites

### Hardware

- One or more Intel BMG70 (Battlemage) discrete GPUs
- Host system with PCIe 4.0/5.0 slots
- Sufficient system RAM (recommended: 64 GB+ for large models)

### Software Stack (CRI)

The following Intel CRI software stack components are required:

| Component | Tested Version | Notes |
|-----------|---------------|-------|
| GPU Driver (i915/xe) | Kernel 6.x+ | Intel GPU driver with BMG support |
| Intel Compute Runtime | 25.48.36300.8 | User-mode driver (UMD) |
| Intel Graphics Compiler (IGC) | 2.24.8 | |
| Level-Zero Loader | 1.26.0 | |
| Intel oneAPI Base Toolkit | 2025.3 | Including oneMKL, oneDNN |
| Intel DPC++ Compiler | 2025.3 | |
| oneCCL | 2021.15.7 | **Must use this version for BMG support** |
| PyTorch | 2.10.0+xpu | From `https://download.pytorch.org/whl/xpu` |

## Installation

### Option 1: Docker (Recommended)

The simplest way to run vLLM on BMG70 is using the provided XPU Docker image:

```bash
# Build the XPU Docker image
docker build -t vllm-xpu -f docker/Dockerfile.xpu .

# Run with GPU access
docker run \
    --device /dev/dri:/dev/dri \
    --net=host \
    --ipc=host \
    --privileged \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -e HF_TOKEN=$HF_TOKEN \
    vllm-xpu \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enforce-eager \
    --block-size 64 \
    --dtype bfloat16
```

### Option 2: Bare-Metal Installation

1. **Install Intel GPU drivers and compute runtime** following
   [Intel's installation guide](https://dgpu-docs.intel.com/).

2. **Install oneAPI components:**

    ```bash
    # Add Intel package repository
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
      | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] \
      https://apt.repos.intel.com/oneapi all main" \
      | sudo tee /etc/apt/sources.list.d/oneAPI.list

    sudo apt update && sudo apt install -y intel-oneapi-compiler-dpcpp-cpp-2025.3
    ```

3. **Install oneCCL with BMG support:**

    ```bash
    wget https://github.com/uxlfoundation/oneCCL/releases/download/2021.15.7/intel-oneccl-2021.15.7.8_offline.sh
    bash intel-oneccl-2021.15.7.8_offline.sh -a --silent --eula accept
    source /opt/intel/oneapi/setvars.sh --force
    source /opt/intel/oneapi/ccl/2021.15/env/vars.sh --force
    ```

4. **Install vLLM with XPU support:**

    ```bash
    pip install -r requirements/xpu.txt
    VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation .
    ```

## Verifying the Installation

Check that XPU devices are visible:

```bash
# List Level-Zero GPU devices
sycl-ls
# Expected output includes: [level_zero:gpu]

# If xpu-smi is installed
xpu-smi discovery
```

Verify vLLM can detect XPU:

```bash
python -c "import torch; print(torch.xpu.device_count(), 'XPU devices found')"
python -c "import vllm; from vllm import LLM"
```

## Running Models

### Single-GPU Inference (TP=1)

```bash
# Set required environment variables
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_TARGET_DEVICE=xpu

# Start the server
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enforce-eager \
    --block-size 64 \
    --dtype bfloat16

# Test with curl
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}]
    }'
```

### Multi-GPU Inference (TP>1)

For tensor parallelism across multiple BMG70 cards, configure oneCCL:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_TARGET_DEVICE=xpu

# Source oneCCL environment
source /opt/intel/oneapi/setvars.sh --force
source /opt/intel/oneapi/ccl/2021.15/env/vars.sh --force

vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enforce-eager \
    --block-size 64 \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --distributed-executor-backend mp
```

## XPU-Specific Flags

| Flag | Value | Reason |
|------|-------|--------|
| `--enforce-eager` | (set) | XPU does not support CUDA graphs |
| `--block-size` | `64` | Optimal block size for Intel GPU memory |
| `--dtype` | `bfloat16` | Recommended precision for BMG70 |
| `--distributed-executor-backend` | `mp` | Use multiprocessing for multi-GPU |

## Environment Variables

| Variable | Value | Purpose |
|----------|-------|--------|
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | Required for XPU multiprocessing |
| `VLLM_TARGET_DEVICE` | `xpu` | Target Intel XPU backend |
| `ZE_AFFINITY_MASK` | e.g. `0,1` | Pin to specific GPU devices |

## Known Limitations

- **No CUDA graph support**: Always use `--enforce-eager`.
- **oneCCL multi-GPU hangs**: Some multi-GPU configurations may
  experience hangs during collective operations with `shm_broadcast`.
  If this occurs, try setting `CCL_ATL_TRANSPORT=ofi`.
- **Memory constraints**: BMG70 has ~24 GB VRAM per card. Large models
  require tensor parallelism across multiple cards.

## Running Benchmarks

XPU-specific benchmark configurations are available:

```bash
cd benchmarks

# Latency benchmark
vllm bench latency \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enforce-eager --block-size 64 --dtype bfloat16 \
    --load-format dummy --num-iters 15

# Throughput benchmark
vllm bench throughput \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enforce-eager --block-size 64 --dtype bfloat16 \
    --load-format dummy --num-prompts 200 --backend vllm
```

Pre-configured benchmark suites for CI are in:

- `.buildkite/performance-benchmarks/tests/latency-tests-xpu.json`
- `.buildkite/performance-benchmarks/tests/throughput-tests-xpu.json`
- `.buildkite/performance-benchmarks/tests/serving-tests-xpu.json`
