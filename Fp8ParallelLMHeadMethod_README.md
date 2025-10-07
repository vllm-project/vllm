# Fp8ParallelLMHeadMethod Implementation

## Overview
This document describes the implementation of `Fp8ParallelLMHeadMethod`, a custom FP8 quantization method for `ParallelLMHead` that uses a customized TMA persistent GEMM kernel with **CUDA graph support** for improved performance.

## Location
- **File**: `/home/ubuntu/pr/vllm/vllm/model_executor/layers/quantization/fp8.py`
- **Class**: `Fp8ParallelLMHeadMethod(LinearMethodBase)`
- **Lines**: ~1289-1520

## Key Features

### 1. **Per-Channel FP8 Quantization**
- Supports FP8E4M3FN weight format
- Per-channel weight scaling (one scale per vocabulary token/output channel)
- BF16 activation (no activation quantization)
- BF16 output for logits

### 2. **CUDA Graph Capture and Replay**
- **Automatic CUDA graph capture** for common batch sizes
- **Graph replay** for reduced kernel launch overhead
- **Batch size padding** to align with graph capture sizes
- **Persistent buffer management** for graph inputs/outputs
- **Fallback to non-graph execution** for large batches

### 3. **TMA Persistent GEMM Kernel Integration**
- Uses customized TMA persistent GEMM kernel via `fp8_tma_linear` wrapper
- Optimized for CUDA graph capture and high-performance inference
- Graceful fallback to standard vLLM FP8 kernel if unavailable

### 4. **EAGLE Model Compatibility**
- Designed to work with EAGLE3 speculative decoding
- Compatible with quantized LM head weights loaded from separate checkpoint
- Integrates with vLLM's quantization framework

## Implementation Details

### Class Structure
```python
class Fp8ParallelLMHeadMethod(LinearMethodBase):
    def __init__(self, quant_config: Fp8Config, layer: torch.nn.Module)
    def create_weights(...)
    def process_weights_after_loading(...)
    def apply(...)  # CUDA graph enabled
```

### Method Responsibilities

#### `__init__(quant_config, layer)`
- Initializes the method with FP8 config
- Stores vllm_config for CUDA graph batch sizes
- Imports TMA persistent GEMM kernel dynamically
- Sets output dtype to BF16
- Handles fallback if TMA kernel is unavailable

#### `create_weights(...)`
- Creates FP8 weight parameter (vocab_size, hidden_size)
- Creates per-channel weight scale parameter (vocab_size,) in BF16
- Requires FP8-serialized checkpoint (raises error otherwise)
- No input activation scale (activation quantization disabled)

#### `process_weights_after_loading(layer)`
- Validates weight dtype (FP8E4M3FN)
- Converts weight_scale to BF16 if needed
- Handles scale shape normalization (squeeze if 2D)
- Validates per-channel scale dimensions
- Logs processed weight information

#### `apply(layer, x, bias=None)` - **CUDA Graph Enabled**

**CUDA Graph Initialization**:
```python
# Initialize CUDA graph buffers on first call
if not hasattr(layer, '_tma_graphs'):
    layer._tma_graphs = {}          # Captured CUDA graphs
    layer._tma_inputs = {}          # Input buffers for each graph
    layer._tma_outputs = {}         # Output buffers for each graph
    layer._tma_weights = {}         # Weight tensors for each graph
    layer._tma_weight_scales = {}   # Scale tensors for each graph
    layer.cudagraph_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
```

**Execution Flow**:
1. **Ensure BF16 input**: Convert if needed
2. **Initialize CUDA graph buffers**: On first call
3. **Call `fp8_tma_linear` wrapper**:
   - Automatically pads batch size to nearest graph size
   - Captures graph on first use for each batch size
   - Replays graph on subsequent calls
   - Falls back to non-graph for large batches
4. **Fallback handling**:
   - If `fp8_tma_linear` unavailable: Use TMA kernel directly
   - If TMA kernel fails: Use standard vLLM FP8 kernel

**CUDA Graph Capture Process** (handled by `fp8_tma_linear`):
```python
# Example for batch_size=5, will use graph_key=8
M_cap = 8  # Padded batch size
if graph_key not in layer._tma_graphs:
    # Create buffers
    layer._tma_inputs[graph_key] = torch.zeros((M_cap, K), dtype=bf16, device=device)
    layer._tma_outputs[graph_key] = torch.zeros((M_cap, N), dtype=bf16, device=device)
    
    # Capture graph
    layer._tma_graphs[graph_key] = torch.cuda.CUDAGraph()
    with torch.cuda.graph(layer._tma_graphs[graph_key]):
        layer._tma_outputs[graph_key] = matmul_tma_persistent(
            layer._tma_inputs[graph_key],
            layer._tma_weights[graph_key],
            layer._tma_weight_scales[graph_key],
            ...
        )

# Copy input and replay
layer._tma_inputs[graph_key][:num_tokens] = x
layer._tma_graphs[graph_key].replay()
output = layer._tma_outputs[graph_key][:num_tokens]
```

## Integration with Fp8Config

The method is registered in `Fp8Config.get_quant_method()`:

```python
def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Optional["QuantizeMethodBase"]:
    # ... other layer types ...
    elif isinstance(layer, ParallelLMHead):
        return Fp8ParallelLMHeadMethod(self, layer)
    return None
```

## Usage Example

From the EAGLE proposer (`/home/ubuntu/pr/vllm/vllm/v1/spec_decode/eagle.py`):

```python
# Create FP8 quantization config for per-channel quantization
fp8_config = Fp8Config(
    is_checkpoint_fp8_serialized=True,
    activation_scheme=None  # No activation quantization
)

# Create ParallelLMHead with FP8 quantization (CUDA graph enabled)
self.model.lm_head = ParallelLMHead(
    model_config.vocab_size,
    model_config.hidden_size,
    bias=False,
    quant_config=fp8_config,
    prefix="lm_head"
)

# Load FP8 weights from checkpoint
lm_head_state_dict = torch.load(lm_head_path, map_location="cpu")
# - lm_head.weight: (vocab_size, hidden_size) FP8E4M3FN
# - lm_head.weight_scale: (vocab_size,) or (vocab_size, 1) BF16

# Weight loading is handled by vLLM's weight loading mechanism
```

## CUDA Graph Benefits

### Performance Improvements
1. **Reduced kernel launch overhead**: Graph replay is ~10-100x faster than launching kernels
2. **Better GPU utilization**: Reduced CPU-GPU synchronization
3. **Predictable latency**: Consistent execution time for same batch size
4. **Memory efficiency**: Persistent buffers avoid repeated allocations

### Graph Capture Batch Sizes
Default sizes (configurable via vllm_config):
```python
[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
```

### Memory Overhead
For each captured batch size `M`:
- Input buffer: `M × hidden_size × 2 bytes` (BF16)
- Output buffer: `M × vocab_size × 2 bytes` (BF16)
- Example (hidden=4096, vocab=128k, M=512):
  - Input: 512 × 4096 × 2 = 4 MB
  - Output: 512 × 128k × 2 = 128 MB
  - Total per graph: ~132 MB

## TMA Persistent GEMM Kernel Requirements

The kernel (`fp8_tma_linear` wrapper) expects:
- **Input (x)**: (M, K) BF16 tensor
- **Weight**: (N, K) FP8E4M3FN tensor (already transposed)
- **Weight Scale**: (N,) BF16 tensor (per-channel scales)
- **Returns**: (M, N) BF16 output

Key features:
- **CUDA graph capture support** via persistent buffers
- **TensorDescriptor caching** for performance
- **Automatic batch size padding** to graph sizes
- **Configurable block sizes** and warp specialization
- **Per-channel scaling** applied in kernel epilogue

## Benefits

1. **Performance**: 
   - CUDA graph capture reduces launch overhead by 10-100x
   - TMA persistent kernel optimized for LM head computation
2. **Memory**: FP8 weights reduce memory footprint by ~2x
3. **Accuracy**: Per-channel scaling maintains better accuracy than per-tensor
4. **Compatibility**: Seamless integration with vLLM's quantization framework
5. **Fallback**: Graceful degradation to standard FP8 kernel if needed
6. **Scalability**: Supports various batch sizes with automatic padding

## Limitations

1. Requires FP8-serialized checkpoint (no on-the-fly quantization)
2. Only supports per-channel weight quantization (no block quantization)
3. No activation quantization (BF16 activations only)
4. CUDA graph capture adds ~100-200 MB memory overhead
5. First call per batch size slower (graph capture overhead)
6. TMA kernel requires specific CUDA compute capability (SM80+)

## Testing Recommendations

1. **Correctness**:
   - Verify weight loading from checkpoint
   - Validate output accuracy vs. BF16 baseline
   - Test with different batch sizes
   
2. **Performance**:
   - Benchmark performance vs. standard FP8 kernel
   - Measure CUDA graph capture overhead (first call)
   - Verify graph replay speedup (subsequent calls)
   
3. **CUDA Graph**:
   - Test all cudagraph_batch_sizes
   - Verify padding behavior (batch_size → graph_key)
   - Test fallback for large batches
   
4. **Robustness**:
   - Verify fallback behavior when TMA kernel unavailable
   - Test memory usage with multiple graphs
   - Validate concurrent execution (multiple streams)

## Performance Expectations

### CUDA Graph Speedup
- **First call** (capture): ~100-500 ms overhead
- **Subsequent calls** (replay): 10-100x faster than non-graph
- **Typical speedup**: 5-20 ms → 0.1-2 ms per forward pass

### Memory Usage
- **Per graph**: ~100-200 MB (depends on vocab_size)
- **Total**: ~1-2 GB for all default batch sizes
- **Trade-off**: Memory for speed (worthwhile for inference)

## Related Files

- **Main implementation**: `/home/ubuntu/pr/vllm/vllm/model_executor/layers/quantization/fp8.py`
- **TMA kernel**: `/home/ubuntu/pr/vllm/vllm/model_executor/layers/quantization/kernels/scaled_mm/tma_persistent_gemm.py`
- **fp8_tma_linear wrapper**: `/home/ubuntu/gpt_oss_opt/cudagraph/Aversion_sep_24_scale/fp8.py` (lines 73-238)
- **EAGLE proposer**: `/home/ubuntu/pr/vllm/vllm/v1/spec_decode/eagle.py`
- **ParallelLMHead**: `/home/ubuntu/pr/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py`

## Debugging Tips

1. **Enable logging**: Set `VLLM_LOGGING_LEVEL=DEBUG` to see graph capture info
2. **Check graph keys**: Verify batch size → graph_key mapping
3. **Memory profiling**: Use `torch.cuda.memory_summary()` to track allocations
4. **Disable graphs**: Compare with non-graph execution for debugging
5. **Kernel errors**: Check for shape mismatches in TMA kernel invocation
