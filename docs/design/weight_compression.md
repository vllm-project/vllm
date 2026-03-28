# Model Weight Compression with GPU-Side Decompression

## Context

Enable inference on models larger than GPU VRAM by:
1. Pre-compressing weights offline (smaller disk footprint, less I/O)
2. Streaming compressed weights to GPU during inference (smaller PCIe transfers)
3. Decompressing directly on GPU via CUDA kernels (no CPU round-trip in hot path)

Target hardware: NVIDIA CUDA GPUs. Primary use: already-quantized models (GPTQ INT4,
AWQ INT4, INT8, FP8) where lossless compression adds a further 20–35% size reduction
on top of quantization.

---

## GPU Decompression Library: nvCOMP

NVIDIA's official GPU compression library:
- **Repo**: https://github.com/NVIDIA/nvcomp
- **Supported algorithms**: LZ4, GDeflate, Zstd, Deflate, ANS, Bitcomp, Cascaded
- **Integration path**: Link `libnvcomp` in CMakeLists; wrap in `csrc/weight_decompress.cu`
- **Build flag**: `-DVLLM_NVCOMP_PATH=/path/to/nvcomp/install`

**Fallback**: If nvCOMP is not installed, automatically fall back to CPU zlib
decompression + async CUDA stream transfer (nearly same latency via overlap).

---

## Algorithm Selection by Tensor Dtype

| Dtype | Example | GPU Algorithm | CPU Fallback | Typical Reduction |
|-------|---------|---------------|--------------|-------------------|
| INT4 packed | GPTQ `qweight` (int32) | LZ4 / GDeflate (nvCOMP) | zlib | 15–30% |
| INT8 | AWQ/SmoothQuant | LZ4 / GDeflate (nvCOMP) | zlib | 20–35% |
| FP8 | `float8_e4m3fn` | LZ4 / GDeflate (nvCOMP) | zlib | 20–30% |
| BF16/FP16 | scales, norms | GDeflate + ZipNN pre-pass | zlib | 30–50% |

**Note**: ZipNN performs a float-specific byte-shuffle before compression (separates
exponent/mantissa bytes), improving compression ratio by ~17% on float tensors. Applied
offline only (CPU); GPU decompression uses standard algorithms.

---

## Data Format: Per-Tensor Compressed Index

For GPU decompression to work without parsing safetensors on GPU, we use a custom
per-tensor format. This avoids the "decompress shard → parse header on GPU" problem.

```
model_compressed/
  config.json                  # copied verbatim
  tokenizer.json               # copied verbatim
  tokenizer_config.json        # copied verbatim
  compression_index.json       # tensor registry (see below)
  weights_00001.cbin           # concatenated compressed tensor bytes (shard 1)
  weights_00002.cbin           # shard 2, ...
```

**`compression_index.json`**:
```json
{
  "algorithm": "deflate",
  "zipnn_preshuffle": true,
  "total_original_bytes": 9663676416,
  "total_compressed_bytes": 6924288000,
  "tensors": {
    "model.layers.0.self_attn.q_proj.qweight": {
      "shard": "weights_00001.cbin",
      "byte_offset": 0,
      "compressed_size": 131072,
      "original_size": 524288,
      "shape": [512, 2048],
      "dtype": "int32",
      "compression_ratio": 0.25
    },
    "model.layers.0.self_attn.q_proj.scales": {
      "shard": "weights_00001.cbin",
      "byte_offset": 131072,
      "compressed_size": 12288,
      "original_size": 32768,
      "shape": [512, 64],
      "dtype": "float16",
      "compression_ratio": 0.375
    }
  }
}
```

**Grouping**: Tensors from the same transformer layer go into the same shard file,
in parameter order. This enables sequential reads for whole-layer streaming.

---

## Files

| File | Description |
|------|-------------|
| `tools/compress_weights.py` | Offline compression CLI |
| `vllm/model_executor/model_loader/compressed_loader.py` | Loader: index parsing, CPU decompression, JIT mode |
| `csrc/weight_decompress.cu` | CUDA kernel: nvCOMP LZ4/GDeflate GPU decompression |
| `csrc/torch_bindings.cpp` | Registers `decompress_tensor` / `compress_tensor` custom ops |
| `CMakeLists.txt` | Adds `weight_decompress.cu` to `_C`; finds nvCOMP |
| `vllm/model_executor/model_loader/__init__.py` | Registers `"compressed"` load format |
| `benchmarks/benchmark_compression.py` | Benchmark suite |

---

## Architecture: Two Modes

### Mode 1: Load-All (default)

Decompress at startup. Zero runtime overhead. Smaller disk footprint only.

```
Disk (.cbin) → CPU decompress → CPU tensor → H2D → GPU param (normal loading)
```

### Mode 2: JIT Streaming (`enable_jit_decompress=true`)

Keep compressed bytes in CPU RAM. Transfer compressed bytes over PCIe (smaller).
GPU decompresses directly into model parameter tensors.

```
Disk (.cbin) → CPU pinned RAM (compressed)
                     │
                     │ H2D (compressed — smaller PCIe transfer)
                     ▼
               GPU scratch buffer (compressed bytes)
                     │
                     │ nvCOMP LZ4/GDeflate kernel
                     ▼
               GPU model parameter tensors → forward pass → free
```

Used together with `--cpu-offload-gb` to control how many layers stay in CPU RAM.

### JIT Forward Wrapper

The `CompressedModelLoader` wraps each CPU-offloaded module's `forward()`:

1. Transfer compressed parameter bytes to GPU (`non_blocking=True`)
2. Synchronize stream
3. Call `torch.ops._C.decompress_tensor(compressed_gpu, shape, dtype)` per parameter
4. Call `functional_call(module, decompressed_weights, args, kwargs)`
5. Free GPU decompressed tensors (compressed bytes stay on CPU for next call)

**Async prefetch** (`prefetch_layers=1`): Begin H2D transfer of layer N+1 on a secondary
CUDA stream while layer N's forward pass runs. Doubles effective PCIe bandwidth utilization.

---

## CUDA Kernel: `csrc/weight_decompress.cu`

Exposes two Python-callable ops:

```cpp
// Decompress a compressed tensor on GPU
torch::Tensor decompress_tensor(
    torch::Tensor compressed_bytes,  // uint8 on GPU
    std::vector<int64_t> shape,
    std::string dtype,
    int64_t original_size,
    std::string algorithm = "lz4"    // "lz4" or "gdeflate"
);

// Compress a raw tensor on GPU (for offline compression or JIT setup)
torch::Tensor compress_tensor(
    torch::Tensor raw_gpu,
    std::string algorithm,           // "lz4" or "gdeflate"
    int64_t level = 0
);
```

Registered under `torch.ops._C` (the main vLLM CUDA extension).

**Fallback without nvCOMP**: `decompress_tensor` falls back to CPU zlib for
`deflate`-compressed models. `lz4` and `gdeflate` always require nvCOMP.

---

## Building with nvCOMP

```bash
# Download nvCOMP from https://github.com/NVIDIA/nvcomp/releases
# Then configure vLLM:
cmake .. -DVLLM_NVCOMP_PATH=/path/to/nvcomp/install

# Or add to CMAKE_PREFIX_PATH:
cmake .. -DCMAKE_PREFIX_PATH=/path/to/nvcomp/install
```

Without `-DVLLM_NVCOMP_PATH`, vLLM compiles normally with CPU zlib fallback only.
GDeflate and GPU LZ4 decompression/compression are disabled.

---

## Compressed tensor format (LZ4 + GDeflate)

Both algorithms use the same chunked layout:

```
[4 bytes: uint32_t  n_chunks]
[n_chunks × 4 bytes: uint32_t comp_size[i]]
[chunk_0 compressed bytes]
[chunk_1 compressed bytes]
...
```

`CHUNK_SIZE = 65536` bytes (uncompressed, except last chunk). This constant is
shared between `csrc/weight_decompress.cu`, `tools/compress_weights.py`, and
`vllm/model_executor/model_loader/compressed_loader.py`.
