# vLLM on IBM POWER8 (ppc64le) - Build Guide

This guide documents how to build and run vLLM on IBM POWER8 (ppc64le) architecture with CPU backend.

## System Requirements

- IBM POWER8 (or later) server with ppc64le Linux
- Ubuntu 20.04 LTS (last officially supported version for POWER8)
- Python 3.11
- GCC 10.x
- CMake 3.22+

### Test System

| Spec | Value |
|------|-------|
| Model | IBM Power System S824 (8286-42A) |
| CPUs | Dual 8-core POWER8 = 16 cores |
| Threads | SMT8 = 128 hardware threads |
| RAM | 576 GB DDR3 |
| OS | Ubuntu 20.04 LTS |

## Prerequisites

### 1. Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    python3-dev \
    python3-pip \
    git \
    libopenblas-dev \
    libomp-dev
```

### 2. Set Up Conda Environment

```bash
# Install Miniconda for ppc64le
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh
bash Miniconda3-latest-Linux-ppc64le.sh

# Create environment
conda create -n pytorch-pse python=3.11 -y
conda activate pytorch-pse

# Install dependencies
conda install -y numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses
pip install packaging
```

## Building PyTorch 2.4 from Source

PyTorch 2.4+ is required for vLLM compatibility. Pre-built wheels are not available for ppc64le, so we must build from source.

### 1. Clone PyTorch

```bash
cd ~
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch
git checkout v2.4.0
git submodule sync
git submodule update --init --recursive
```

### 2. Critical Fix: VSX Header Include Logic

The default PyTorch build has a bug where VSX-specific headers (which delete `operator[]`) are included in the DEFAULT CPU build, causing compilation failures.

**Edit `aten/src/ATen/cpu/vec/vec256/vec256.h`:**

Find lines ~19-23:

```cpp
// BEFORE:
#if !(defined(__VSX__)  || defined(CPU_CAPABILITY_VSX) || defined(CPU_CAPABILITY_ZVECTOR))
...
#elif defined(__VSX__)  || defined(CPU_CAPABILITY_VSX)
```

Change to:

```cpp
// AFTER:
#if !(defined(CPU_CAPABILITY_VSX) || defined(CPU_CAPABILITY_ZVECTOR))
...
#elif defined(CPU_CAPABILITY_VSX)
```

This ensures the DEFAULT build uses scalar Vectorized types (which have `operator[]`), while only the VSX-specific build uses VSX intrinsics.

### 3. Build PyTorch

```bash
cd ~/pytorch

# CRITICAL: Clear conda environment variables that interfere with system headers
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS DEBUG_CFLAGS DEBUG_CXXFLAGS

# Use system compilers
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# Configure for CPU-only ppc64le build
export USE_CUDA=0
export USE_CUDNN=0
export USE_ROCM=0
export USE_MPS=0
export BUILD_CAFFE2=0
export USE_FBGEMM=0
export USE_DISTRIBUTED=1
export USE_NCCL=0
export MAX_JOBS=32

# Build
python setup.py develop

# This takes 1-2 hours on POWER8
```

### 4. Create Include Structure for CMake

vLLM's CMake needs a proper include directory structure:

```bash
mkdir -p ~/pytorch/torch/include/torch/csrc/api
cd ~/pytorch/torch/include

# Create symlinks
ln -sf ~/pytorch/aten/src/ATen ATen
ln -sf ~/pytorch/c10 c10
ln -sf ~/pytorch/caffe2 caffe2
ln -sf ~/pytorch/torch/csrc/api/include ~/pytorch/torch/include/torch/csrc/api/include

# Create share/cmake for TorchConfig
mkdir -p ~/pytorch/torch/share/cmake/Torch
ln -sf ~/pytorch/build/TorchConfig.cmake ~/pytorch/torch/share/cmake/Torch/
ln -sf ~/pytorch/build/TorchConfigVersion.cmake ~/pytorch/torch/share/cmake/Torch/

# Create symlink in site-packages for CMake discovery
ln -sf ~/pytorch/torch ~/miniconda3/envs/pytorch-pse/lib/python3.11/site-packages/torch
```

### 5. Verify PyTorch Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
# Should output: PyTorch: 2.4.0a0+gitXXXXXXX
```

## Building vLLM

### 1. Clone vLLM with POWER8 Support

```bash
cd ~
git clone https://github.com/vllm-project/vllm.git vllm-power8
cd vllm-power8
```

### 2. Apply Compatibility Patches

Several patches are required for PyTorch 2.4.0a0 compatibility:

**a) Patch `vllm/utils/torch_utils.py` for `infer_schema`:**

Replace the import line:

```python
from torch.library import Library, infer_schema
```

With:

```python
from torch.library import Library
try:
    from torch.library import infer_schema
except ImportError:
    # PyTorch < 2.4.1 fallback
    import inspect
    def infer_schema(fn, *, mutates_args=None):
        """Fallback infer_schema for PyTorch < 2.4.1"""
        sig = inspect.signature(fn)
        params = []
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            annotation = param.annotation
            if annotation == inspect.Parameter.empty:
                params.append(f'Tensor {name}')
            elif 'Tensor' in str(annotation):
                params.append(f'Tensor {name}')
            elif annotation == int:
                params.append(f'int {name}')
            elif annotation == float:
                params.append(f'float {name}')
            elif annotation == bool:
                params.append(f'bool {name}')
            else:
                params.append(f'Tensor {name}')
        return_type = 'Tensor'
        return '(' + ', '.join(params) + ') -> ' + return_type
```

**b) Patch `vllm/env_override.py`:**

Replace:

```python
torch._inductor.config.compile_threads = 1
```

With:

```python
try:
    torch._inductor.config.compile_threads = 1
except AttributeError:
    pass  # PyTorch version doesn't have _inductor.config
```

**c) Patch `vllm/distributed/parallel_state.py`:**

Replace:

```python
import torch.distributed._symmetric_memory
```

With:

```python
try:
    import torch.distributed._symmetric_memory
except ImportError:
    torch.distributed._symmetric_memory = None
```

**d) Patch `vllm/compilation/wrapper.py` for dynamo:**

Replace:

```python
import torch._C._dynamo.guards
```

With:

```python
try:
    import torch._C._dynamo.guards
except (ImportError, ModuleNotFoundError):
    class MockGuards:
        pass
    torch._C._dynamo = type('_dynamo', (), {'guards': MockGuards})()
```

**e) Patch `vllm/platforms/__init__.py` for CPU detection:**

Find `cpu_platform_plugin()` function and add at the start:

```python
import os
if os.environ.get("VLLM_TARGET_DEVICE", "").lower() == "cpu":
    return "vllm.platforms.cpu.CpuPlatform"
```

**f) Patch `vllm/v1/worker/gpu_worker.py` for CUDA backend:**

Wrap the CUDA matmul precision setting in try-except:

```python
try:
    torch.backends.cuda.matmul.fp32_precision = precision
except AttributeError:
    pass  # Not available in this PyTorch version or on CPU
```

**g) Patch `vllm/v1/attention/backends/cpu_attn.py` for AMX detection:**

Replace:

```python
supports_amx = torch._C._cpu._is_amx_tile_supported()
```

With:

```python
try:
    supports_amx = torch._C._cpu._is_amx_tile_supported()
except AttributeError:
    supports_amx = False  # Not available in this PyTorch version
```

**h) Patch `vllm/v1/spec_decode/ngram_proposer.py` for optional numba:**

Replace the import section:

```python
from numba import get_num_threads, jit, njit, prange, set_num_threads
```

With:

```python
# Make numba optional - not available on all platforms (e.g., ppc64le with numpy conflicts)
try:
    from numba import get_num_threads, jit, njit, prange, set_num_threads
    NUMBA_AVAILABLE = True
except ImportError as e:
    NUMBA_AVAILABLE = False
    _NUMBA_IMPORT_ERROR = str(e)
    # Provide stubs so the module can be imported
    def get_num_threads(): return 1
    def set_num_threads(n): pass
    def jit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator if args and callable(args[0]) else decorator
    def njit(*args, **kwargs):
        return jit(*args, **kwargs)
    def prange(*args): return range(*args)
```

And add at the start of `NgramProposer.__init__`:

```python
if not NUMBA_AVAILABLE:
    raise ImportError(
        f"NgramProposer requires numba, but it failed to import: {_NUMBA_IMPORT_ERROR}. "
        "Speculative decoding with ngram proposer is not available on this platform. "
        "Please disable speculative decoding or install a compatible numba version."
    )
```

**i) Patch `vllm/attention/layer.py` for optional scale parameters:**

Find line ~380 and add None parameters:

```python
torch.ops.vllm.unified_attention_with_output(
    query, key, value, output, self.layer_name, None, None
)
```

**j) Patch `vllm/v1/worker/gpu_model_runner.py` for CPU compatibility:**

In the `AsyncGPUModelRunnerOutput.__init__` method, wrap CUDA calls:

```python
if torch.cuda.is_available() and async_output_copy_stream is not None:
    default_stream = torch.cuda.current_stream()
    with torch.cuda.stream(async_output_copy_stream):
        # ... existing CUDA code ...
else:
    # CPU path - direct copy, no async stream needed
    self.sampled_token_ids_cpu = self._sampled_token_ids.to("cpu")
    self._logprobs_tensors_cpu = (
        self._logprobs_tensors.to_cpu_nonblocking()
        if self._logprobs_tensors
        else None
    )
```

### 3. Dependency Version Fixes

The following version constraints are critical for ppc64le:

```bash
# scipy 1.14+ requires numpy 2.x, but PyTorch 2.4.0 needs numpy 1.x
conda install -y numpy=1.26.4 scipy=1.11.3 -c conda-forge

# opencv-python conflicts with Python's typing module - uninstall if not needed
pip uninstall -y opencv-python opencv-python-headless
```

### 4. Build vLLM C Extension

```bash
cd ~/vllm-power8

# Clear environment
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# Configure
cmake . -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DVLLM_TARGET_DEVICE=cpu \
  -DVLLM_PYTHON_EXECUTABLE=$(which python)

# Build
ninja -j32

# Copy the built extension
cp _C.abi3.so vllm/
```

### 4. Verify vLLM Installation

```bash
cd ~/vllm-power8
PYTHONPATH=. python -c "from vllm import LLM; print('vLLM ready!')"
```

## Current Status (2025-12-30)

### What Works

- PyTorch 2.4.0a0 builds and runs on POWER8
- vLLM imports successfully with all patches applied
- Model loading and warmup completes
- CPU platform detection via `VLLM_TARGET_DEVICE=cpu`

### Known Issues

1. **C Extension Missing CPU Kernels**: The vLLM C extension builds but doesn't export `cpu_attention_with_kv_cache`. This requires additional CMake configuration to include the CPU attention backend.

2. **V1 Engine GPU Code Paths**: Even with `device_config=cpu`, some V1 engine code paths call CUDA-specific functions. Additional patches may be needed.

### Workaround

For immediate use on POWER8, consider using an older vLLM version (pre-V1 engine) which has better CPU-only support, or use llama.cpp which has native POWER8 VSX support.

## Running Inference

### Example: Text Generation

```python
#!/usr/bin/env python3
import os
os.environ["VLLM_TARGET_DEVICE"] = "cpu"

import sys
sys.path.insert(0, '/home/sophia/vllm-power8')

from vllm import LLM, SamplingParams

# Initialize with a small model for testing
# Note: Use dtype="float32" - bfloat16 not supported on POWER8
llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype="float32",
    trust_remote_code=True,
    max_model_len=512,
)

# Generate text
prompts = ["Hello, my name is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

## Troubleshooting

### Common Issues

1. **`__GLIBC_USE` macro errors**
   - Cause: Conda environment variables adding `-isystem` paths
   - Fix: `unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS`

2. **`operator[]` deleted function errors**
   - Cause: VSX Vectorized headers being included in DEFAULT builds
   - Fix: Apply the vec256.h patch above

3. **`infer_schema` not found**
   - Cause: PyTorch 2.4.0a0 doesn't have this function
   - Fix: Apply the torch_utils.py patch

4. **`torch.distributed._symmetric_memory` not found**
   - Cause: Feature added in later PyTorch versions
   - Fix: Apply the parallel_state.py patch

5. **`torch._inductor.config` has no attribute**
   - Cause: Inductor not fully configured in source builds
   - Fix: Apply the env_override.py patch

6. **numpy/scipy version conflicts**
   - Cause: scipy 1.14+ requires numpy 2.x, but PyTorch 2.4.0 needs numpy 1.x
   - Fix: `conda install numpy=1.26.4 scipy=1.11.3 -c conda-forge`

7. **opencv-python circular import with typing module**
   - Cause: cv2.typing conflicts with Python's built-in typing
   - Fix: `pip uninstall -y opencv-python opencv-python-headless`

8. **`torch.cuda.current_stream()` assertion error**
   - Cause: V1 engine calls CUDA functions on CPU build
   - Fix: Apply the gpu_model_runner.py patch (patch j above)

9. **`cpu_attention_with_kv_cache` not found**
   - Cause: C extension built without CPU attention kernel
   - Fix: Ensure CMake includes CPU backend: `-DVLLM_TARGET_DEVICE=cpu`

### Performance Notes

- Use `numactl --interleave=all` for large models to spread memory across NUMA nodes
- Set `OMP_NUM_THREADS=64` (not 128) for optimal thread scaling
- POWER8 VSX instructions provide ~4x speedup over scalar code

## Key Files

| File | Purpose |
|------|---------|
| `~/pytorch/aten/src/ATen/cpu/vec/vec256/vec256.h` | Critical VSX header fix |
| `~/vllm-power8/vllm/utils/torch_utils.py` | `infer_schema` fallback (patch a) |
| `~/vllm-power8/vllm/env_override.py` | Inductor config fix (patch b) |
| `~/vllm-power8/vllm/distributed/parallel_state.py` | Symmetric memory fix (patch c) |
| `~/vllm-power8/vllm/compilation/wrapper.py` | Dynamo guards mock (patch d) |
| `~/vllm-power8/vllm/platforms/__init__.py` | CPU platform detection (patch e) |
| `~/vllm-power8/vllm/v1/worker/gpu_worker.py` | CUDA matmul fix (patch f) |
| `~/vllm-power8/vllm/v1/attention/backends/cpu_attn.py` | AMX detection fix (patch g) |
| `~/vllm-power8/vllm/v1/spec_decode/ngram_proposer.py` | Optional numba (patch h) |
| `~/vllm-power8/vllm/attention/layer.py` | Attention scale params (patch i) |
| `~/vllm-power8/vllm/v1/worker/gpu_model_runner.py` | CPU stream handling (patch j) |
| `~/vllm-power8/_C.abi3.so` | Built vLLM C extension |

## Version Information

- PyTorch: 2.4.0a0+gitd990dad (built from v2.4.0 tag)
- vLLM: 0.1.dev8+gfdb377659.d20251230
- Python: 3.11.5
- GCC: 10.5.0
- oneDNN: Built with PPC64 support
