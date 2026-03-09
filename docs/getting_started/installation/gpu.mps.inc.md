# --8<-- [start:installation]

vLLM has experimental support for GPU-accelerated inference on Apple Silicon using the MPS (Metal Performance Shaders) backend. This enables running LLM inference on the unified GPU in M1/M2/M3/M4 Macs.

!!! warning "Experimental"
    MPS support is under active development. Some features available on CUDA (PagedAttention, tensor parallelism, continuous batching for high-throughput serving) are not yet implemented. MPS is best suited for single-user local inference.

# --8<-- [end:installation]
# --8<-- [start:requirements]

- Hardware: Apple Silicon Mac (M1, M2, M3, or M4 series)
- OS: macOS 15 (Sequoia) or later
- Memory: 16 GB unified memory minimum, 24+ GB recommended
- Python: 3.10 -- 3.13
- PyTorch: 2.9+ with MPS support

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

There is no extra information on creating a new Python environment for this device.

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

Currently, there are no pre-built MPS wheels. You must build from source.

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

Clone the MPS support branch and install:

```bash
git clone https://github.com/robtaylor/vllm.git
cd vllm
git checkout mps-platform-support
pip install -e ".[dev]"
```

Verify MPS platform detection:

```bash
python -c "
import torch
print('MPS available:', torch.backends.mps.is_available())
from vllm.platforms import current_platform
print('Platform:', current_platform.device_type)
"
```

### Installing Metal quantization kernels (optional)

For accelerated INT4 (AWQ/GPTQ) and GGUF inference, build and install the Metal dequantization kernels. These require [Nix](https://determinate.systems/nix-installer/) to build.

```bash
# INT4 dequantization (AWQ + GPTQ)
cd kernels-community/dequant-int4
nix build
cp -r result/torch210-metal-aarch64-darwin/ \
  $(python -c "import site; print(site.getsitepackages()[0])")/dequant_int4/

# GGUF dequantization (Q4_0, Q8_0, Q4_K, and more)
cd ../dequant-gguf
nix build
cp -r result/torch210-metal-aarch64-darwin/ \
  $(python -c "import site; print(site.getsitepackages()[0])")/dequant_gguf/
```

Without these kernels, quantized models will still work but use a slower PyTorch fallback path.

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:pre-built-images]

Docker is not applicable for MPS. macOS does not support GPU passthrough to containers.

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

Docker is not applicable for MPS. macOS does not support GPU passthrough to containers.

# --8<-- [end:build-image-from-source]
# --8<-- [start:supported-features]

### Running inference

MPS requires spawn multiprocessing. Set the environment variable before running:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

Example with a small model:

```bash
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='distilgpt2', dtype='float16', max_model_len=128)
output = llm.generate(['Hello, world!'], SamplingParams(max_tokens=32))
print(output[0].outputs[0].text)
"
```

Example with a quantized model (requires Metal kernels above):

```bash
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-1.5B-Instruct-AWQ', dtype='float16',
          max_model_len=512, quantization='awq')
print(llm.generate(['Explain quantum computing.'],
                    SamplingParams(max_tokens=64))[0].outputs[0].text)
"
```

### Performance

Typical throughput on Apple Silicon (varies by chip and memory):

| Model | Quantization | Throughput |
|-------|-------------|------------|
| GGUF small model | Q8_0 | ~62 tok/s |
| GGUF small model | Q4_0 | ~45 tok/s |
| Qwen2.5-1.5B | INT4 AWQ | ~17 tok/s |
| Qwen2.5-1.5B | INT4 GPTQ | ~16 tok/s |

### Memory guidelines

MPS uses unified memory shared between CPU and GPU. When the KV cache exceeds approximately 40% of system RAM, Metal's memory manager can thrash, causing 50-100x slowdowns.

The default KV cache allocation is set conservatively to 25% of system RAM. On a 24 GB system this allows roughly 9 GB for KV cache. Adjust with `gpu_memory_utilization` if needed.

### Known limitations

- No PagedAttention on Metal (uses PyTorch SDPA)
- No tensor parallelism (single GPU only)
- No continuous batching optimizations
- GGUF Q4_K_M models may be slow if the model uses Q6_K layers (numpy fallback)
- `fork()` crashes on MPS -- `VLLM_WORKER_MULTIPROC_METHOD=spawn` is required

### Troubleshooting

**Slow inference (50-100x slower than expected)**:
KV cache memory thrashing. Try a smaller model or set `gpu_memory_utilization=0.2`.

**SIGSEGV during startup**:
Set `VLLM_WORKER_MULTIPROC_METHOD=spawn`.

**"No module named 'vllm.platforms.mps'"**:
Ensure you are on the `mps-platform-support` branch.

# --8<-- [end:supported-features]
