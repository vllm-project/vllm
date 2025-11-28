# vLLM Metal Backend for Apple Silicon

This directory contains the native Metal implementation of paged attention for vLLM on Apple Silicon (MPS).

## Overview

The Metal backend provides GPU-accelerated attention operations on Apple Silicon using native Metal Performance Shaders kernels. This implementation enables proper GPU utilization on macOS, replacing the previous CPU-based fallback (TorchSDPA).

### Key Features

- **Paged Attention**: Native Metal kernels for paged K/V cache management
- **V1 and V2 Kernels**: Support for both simple (V1) and partitioned (V2) attention
- **Grouped Query Attention**: Full support for GQA and Multi-Query Attention (MQA)
- **Memory Efficiency**: Uses vLLM's paged memory layout for efficient cache management
- **Multiple Data Types**: Supports float32 and float16 (half precision)

## Architecture

### File Structure

```
csrc/metal/
├── metal_common.h              # Shared data structures and constants
├── paged_attention_v1.metal    # V1 attention kernel (seq_len <= 8192)
├── paged_attention_v2.metal    # V2 attention kernel with partitioning
├── cache_ops.metal             # Cache operations (reshape, copy, swap)
├── metal_context.h/mm          # Metal device and pipeline management
├── metal_kernels.h/cpp         # C++ kernel launchers
└── metal_bindings.cpp          # PyTorch operator bindings
```

### Python Integration

```
vllm/v1/attention/backends/
└── metal_attn.py               # MetalAttentionBackend implementation

vllm/platforms/
└── mps.py                      # MPS platform configuration (updated)

vllm/attention/backends/
└── registry.py                 # Backend registry (updated)
```

## Building

### Prerequisites

- macOS with Apple Silicon (M1, M2, M3, M4, etc.)
- Xcode Command Line Tools
- Python 3.10+
- PyTorch with MPS support

### Build Instructions

1. **Set the target device to Metal:**

```bash
export VLLM_TARGET_DEVICE=metal
```

2. **Build vLLM with Metal support:**

```bash
cd /path/to/vllm
pip install -e .
```

Or using CMake directly:

```bash
mkdir build && cd build
cmake -G Ninja \
    -DVLLM_TARGET_DEVICE=metal \
    -DVLLM_PYTHON_EXECUTABLE=$(which python3) \
    -DCMAKE_INSTALL_PREFIX=.. \
    ..
cmake --build . --target install
```

### Manual Metal Kernel Compilation

If you need to compile the Metal kernels manually:

```bash
# Compile Metal sources to AIR (Apple Intermediate Representation)
xcrun -sdk macosx metal -std=metal3.0 -O3 -ffast-math \
    csrc/metal/paged_attention_v1.metal \
    csrc/metal/paged_attention_v2.metal \
    csrc/metal/cache_ops.metal \
    -o metal_kernels.air

# Create metallib (Metal library)
xcrun -sdk macosx metallib metal_kernels.air -o vllm_metal_kernels.metallib
```

## Usage

Once built, the Metal backend will be automatically selected when running vLLM on Apple Silicon:

```python
from vllm import LLM, SamplingParams

# Initialize model (will automatically use Metal backend on MPS)
llm = LLM(model="meta-llama/Llama-2-7b-hf", device="mps")

# Generate
prompts = ["Hello, my name is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### Verifying Metal Backend

Check the logs to confirm Metal backend is being used:

```
INFO: Using Metal native backend for MPS with paged attention.
INFO: Metal device initialized: Apple M3 Max
```

## Implementation Details

### Paged Attention Kernels

#### V1 Kernel (paged_attention_v1.metal)

- **Use case**: Sequences up to 8192 tokens
- **Strategy**: Single-pass kernel processing entire sequence
- **Grid dimensions**: `(num_heads, num_seqs, 1)`
- **Threadgroup size**: 128 threads

**Algorithm**:
1. Load query vector into registers
2. For each token in sequence:
   - Map to physical cache block using block table
   - Compute Q·K attention score
3. Compute softmax over attention scores
4. Compute weighted sum of values (attention output)

#### V2 Kernel (paged_attention_v2.metal)

- **Use case**: Long sequences (> 8192 tokens)
- **Strategy**: Partitioned reduction with 512-token chunks
- **Grid dimensions**: `(num_heads, num_seqs, max_num_partitions)`
- **Phases**:
  1. **Partition kernel**: Compute attention for each 512-token partition
  2. **Reduce kernel**: Combine partition results with proper weighting

### Memory Layout

#### K/V Cache Layout

**Key Cache**: `[num_blocks, num_kv_heads, head_size/x, block_size, x]`
- Vectorized layout with `x=16` for coalesced memory access

**Value Cache**: `[num_blocks, num_kv_heads, head_size, block_size]`
- Standard layout

#### Block Tables

**Shape**: `[num_seqs, max_num_blocks_per_seq]`
- Maps logical sequence blocks to physical cache blocks
- Enables efficient paging and memory reuse

### Supported Configurations

| Parameter | Values |
|-----------|--------|
| Head sizes | 64, 80, 96, 112, 128, 256 |
| Block size | 16 (default for MPS) |
| Data types | float32, float16 |
| Max sequence length | Unlimited (with V2 kernel) |

## Performance

### Expected Improvements

Compared to the previous TorchSDPA fallback:

- **GPU Utilization**: 0% → 70-90%
- **Throughput**: 3-5x improvement on decode
- **Latency**: 2-4x reduction per token

### Benchmarking

Run the included benchmark:

```bash
python benchmarks/benchmark_paged_attention_metal.py
```

## Troubleshooting

### Metal Extension Not Found

If you see:
```
Metal C extension not available, falling back to Torch SDPA
```

**Solution**: Rebuild vLLM with `VLLM_TARGET_DEVICE=metal`

### Kernel Not Found Errors

If Metal context fails to find a kernel:

1. Verify metallib was compiled and installed:
   ```bash
   ls vllm/vllm_metal_kernels.metallib
   ```

2. Check available kernels:
   ```python
   from vllm._metal_C import *
   # Should import without errors
   ```

### Memory Issues

Apple Silicon uses unified memory. If you encounter OOM:

1. Reduce `max_num_seqs` or `max_num_batched_tokens`
2. Use smaller model or quantization
3. Monitor memory:
   ```bash
   sudo powermetrics --samplers smc | grep -i "GPU Power"
   ```

## Contributing

### Adding New Kernels

To add a new Metal kernel:

1. **Define kernel in .metal file**:
   ```metal
   kernel void my_kernel(constant Args & args [[buffer(0)]], ...) {
       // Implementation
   }
   ```

2. **Add launcher in metal_kernels.cpp**:
   ```cpp
   void my_kernel_launcher(args...) {
       // Setup and dispatch
   }
   ```

3. **Add Python binding in metal_bindings.cpp**:
   ```cpp
   m.def("my_kernel", &vllm::metal::my_kernel_launcher, ...);
   ```

4. **Update CMakeLists.txt** to include new source files

### Testing

Run tests:
```bash
pytest tests/kernels/test_metal_attention.py
```

## References

- **llama.cpp Metal Implementation**: Reference for Metal kernel patterns
- **vLLM CUDA Kernels**: Original paged attention algorithm
- **Metal Programming Guide**: https://developer.apple.com/metal/
- **Metal Shading Language Specification**: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf

## License

This implementation follows vLLM's Apache 2.0 license.

## Acknowledgments

- llama.cpp team for Metal kernel reference implementation
- vLLM team for the paged attention architecture
- Apple Metal team for excellent documentation
