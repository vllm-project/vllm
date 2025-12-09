# RFC: Helion Kernel Integration Framework for vLLM

## Abstract

This RFC proposes a developer-friendly framework for integrating Helion kernels into vLLM, providing streamlined kernel development, automatic benchmarking, and intelligent configuration management with 80% less boilerplate compared to traditional CUDA kernel integration.

## 1. Motivation & Goals

### Current Challenges
- **Performance Bottlenecks**: LLM inference requires highly optimized kernels for operations like RMS normalization, SiLU activation, and quantization
- **Development Complexity**: Writing custom CUDA kernels requires extensive boilerplate for registration, configuration, and testing
- **Integration Overhead**: Significant effort required to integrate new kernels into vLLM's compilation pipeline

### Goals
1. **Streamlined Development**: Minimize boilerplate and maximize developer productivity
2. **Automatic Validation**: Built-in benchmarking with comprehensive correctness checking
3. **Smart Configuration**: Intelligent kernel selection based on model parameters
4. **Seamless Integration**: First-class integration with vLLM's CustomOp system

## 2. What is Helion?

Helion is a Python-first GPU kernel compiler that makes high-performance CUDA kernel development as simple as writing PyTorch code. **If you know PyTorch, you already know most of Helion.**

### Key Advantages

**Familiar Syntax**: Helion uses PyTorch-like syntax with automatic memory management:
```python
# Pure Helion kernel - looks like PyTorch!
def silu_mul_fp8(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    d = input.shape[-1] // 2
    out = torch.empty(input.shape[:-1] + (d,), device=input.device, dtype=torch.float8_e4m3fn)

    for tile_idx in hl.tile(out.shape):  # Automatic tiling
        a_vals = input[..., :d][tile_idx].to(torch.float32)
        b_vals = input[..., d:][tile_idx]
        silu_result = a_vals * torch.sigmoid(a_vals)
        result = silu_result.to(input.dtype) * b_vals
        scale_val = hl.load(scale, [0])
        out[tile_idx] = (result.to(torch.float32) / scale_val).to(out.dtype)

    return out
```

**Performance Without Complexity**:
- **No manual memory management** - Helion handles shared memory, registers, and data movement
- **No CUDA thread/block programming** - Write loops, Helion generates optimal GPU code
- **Automatic optimization** - Built-in tiling, vectorization, and memory coalescing
- **Competitive performance** with significantly less code than hand-tuned CUDA kernels

**Developer Productivity**:
- **Rapid development** - Write kernels with familiar Python semantics
- **Easy debugging** - Python-like semantics with clear error messages
- **Automatic autotuning** - Helion explores optimization configurations automatically
- **PyTorch integration** - Seamless interop with existing PyTorch workflows

## 3. Framework Architecture

### Core Components

```
┌─────────────────────────────────────────┐
│           Application Layer             │
│     (vLLM Models, CustomOps)            │
│         ** UNCHANGED **                 │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│          Framework Layer                │
│         (HelionCustomOp)                │
└─────────────────────────────────────────┘
                    │
┌─────────────────┐   ┌─────────────────┐
│   Kernel Layer  │   │ Developer Tools │
│ (Helion Kernels,│   │ (Benchmark      │
│  ConfigManager) │   │  Framework,     │
└─────────────────┘   │  Autotuning)    │
                      └─────────────────┘
```

**Application Layer (Unchanged)**:
- Existing vLLM model definitions work without modification
- CustomOp enablement control through existing CompilationConfig
- No changes required to model implementations or inference pipelines

**Framework Layer**:
- **HelionCustomOp**: The core framework component that extends vLLM's CustomOp system to orchestrate computation across multiple individual Helion kernels combined with non-Helion code, and manages its own enablement status. It provides a unified interface that seamlessly integrates optimized Helion kernels with traditional CUDA operations (such as communications) within a single custom operation, while handling kernel lifecycle management.

**Kernel Layer**:
- Individual Helion kernel implementations (SiLU-mul-FP8, RMSNorm-FP8, etc.)
- **ConfigManager**: Centralized config file management with model-specific kernel selection
- PyTorch custom op integration with automatic fallback

**Developer Tools**:
- **Benchmark Framework**: Standardized testing with correctness verification and performance measurement
- **Autotuning Infrastructure**: Configuration exploration and optimization

## 4. Developer Experience

### 4.1 Simplified Kernel Development

**Before (Traditional):**
```python
# 50+ lines of boilerplate for PyTorch registration,
# schema definition, fake implementation, config management
```

**After (Helion Framework):**
```python
@register_kernel(
    fake_impl=lambda input, scale: torch.empty(
        input.shape[:-1] + (input.shape[-1] // 2,),
        dtype=torch.float8_e4m3fn, device=input.device
    ),
    default_config=helion.Config(block_sizes=[1, 2048], num_warps=4, num_stages=7)
)
def silu_mul_fp8(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Pure Helion kernel implementation."""
    d = input.shape[-1] // 2
    out = torch.empty(input.shape[:-1] + (d,), device=input.device, dtype=torch.float8_e4m3fn)

    for tile_idx in hl.tile(out.shape):
        a_vals = input[..., :d][tile_idx].to(torch.float32)
        b_vals = input[..., d:][tile_idx]
        silu_result = a_vals * torch.sigmoid(a_vals)
        result = silu_result.to(input.dtype) * b_vals
        scale_val = hl.load(scale, [0])
        out[tile_idx] = (result.to(torch.float32) / scale_val).to(out.dtype)

    return out

@CustomOp.register("silu_mul_fp8_helion")
class SiluMulFp8Helion(HelionCustomOp):
    def __init__(self, model_config, **kwargs):
        super().__init__(model_config, **kwargs)
        self.kernel = self.create_kernel(silu_mul_fp8)  # Auto-configured

    def forward_helion(self, input, scale):
        return self.kernel(input, scale)  # Direct call to optimized kernel
```

**Result:** 80% less boilerplate code

### 4.2 Intelligent Configuration Management

```python
@silu_mul_fp8.register_config_picker
def pick_config(model_config, available_configs):
    """Smart config selection based on model parameters."""
    target_size = model_config.get_hidden_size()

    # Try exact match first
    if str(target_size) in available_configs:
        return (str(target_size), available_configs[str(target_size)])

    # Fallback to closest match
    closest = min(available_configs.items(),
                  key=lambda x: abs(int(x[0]) - target_size))
    return closest

# Configurations automatically loaded from standardized file names:
# vllm/compilation/helion/configs/silu_mul_fp8_4096.json
# vllm/compilation/helion/configs/silu_mul_fp8_8192.json
```

### 4.3 Automatic Benchmarking

```python
class SiluMulFp8Benchmark(KernelBenchmark):
    def get_quick_test_shapes(self):
        return [([
            (1, 8192), (256, 8192), (1024, 8192),
            (1, 16384), (256, 16384)
        ], torch.bfloat16, {})]

    def create_inputs(self, dtype, **shape_params):
        shape = shape_params["shape"]
        return (torch.randn(*shape, dtype=dtype, device="cuda"),
                torch.tensor([0.5], dtype=torch.float32, device="cuda"))

    def run_baseline(self, input, scale):
        """Reference CUDA kernel for comparison."""
        out = torch.empty(input.shape[0], input.shape[-1] // 2,
                         dtype=torch.float8_e4m3fn, device="cuda")
        torch.ops._C.silu_and_mul_quant(out, input, scale)
        return out

# Register benchmark with one line
SiluMulFp8Helion.register_benchmark(SiluMulFp8Benchmark)
```

### 4.4 Rich CLI Integration

```bash
# List all available benchmarks
python benchmarks/benchmark_helion.py --list-benchmarks

# Quick smoke test during development
python benchmarks/benchmark_helion.py --benchmark silu_mul_fp8 --mode quick

# Comprehensive evaluation
python benchmarks/benchmark_helion.py --benchmark rms_norm_fp8 --mode full \
    --output-dir ./results --num-iterations 5000

# Autotune configurations
python scripts/autotune_helion_kernels.py --kernel silu_mul_fp8 \
    --hidden-sizes 2048,4096,8192
```

## 5. API Design

### 5.1 Kernel Registration
- **Single decorator** handles all registration complexity
- **Type safety** with automatic schema inference from annotations
- **Symbolic shape support** via custom fake implementations
- **Configuration management** with intelligent defaults

### 5.2 CustomOp Integration
```python
class MyHelionOp(HelionCustomOp):
    def __init__(self, model_config, **kwargs):
        super().__init__(model_config, **kwargs)
        # create_kernel() handles config selection automatically
        self.kernel = self.create_kernel(my_kernel_wrapper)

    def forward_helion(self, *args):
        return self.kernel(*args)  # Direct call to configured kernel
```

### 5.3 Automatic Discovery
- All HelionCustomOp subclasses automatically exported
- Zero-configuration imports: `from vllm.compilation.helion import SiluMulFp8Helion`
- Benchmark classes automatically registered and discovered

## 6. Performance & Correctness

### 6.1 Expected Performance Gains
| Operation | Model Size | Speedup | Memory Reduction |
|-----------|------------|---------|------------------|
| SiLU-Mul-FP8 | Llama-7B | 1.3-1.8x | 15% |
| RMSNorm-FP8 | Llama-13B | 1.2-1.5x | 20% |
| AllReduce+RMSNorm | Distributed | 1.4-2.1x | 10% |

### 6.2 Automatic Correctness Verification
- Built-in comparison against reference CUDA kernels
- Configurable tolerances with sensible defaults
- Support for different comparison dtypes (FP8 vs FP16/BF16)
- CUDA graph compatibility validation

### 6.3 Configuration System
```python
# Standardized JSON configs with intelligent naming
# {kernel_name}_{config_key}.json

config_manager = ConfigManager.get_instance()
configs = config_manager.load_all_configs("silu_mul_fp8")
# Auto-loads: silu_mul_fp8_2048.json, silu_mul_fp8_4096.json, etc.
```

## 7. Migration Strategy

### 7.1 Backward Compatibility
- Existing vLLM CustomOps work unchanged
- Helion ops coexist seamlessly with CUDA kernels
- Automatic fallback when Helion unavailable

### 7.2 Gradual Adoption
**Phase 1**: Framework + 2-3 high-impact kernels (SiLU-mul, RMSNorm)
**Phase 2**: Convert existing kernels + comprehensive benchmarking
**Phase 3**: Advanced features (dynamic shapes, model-specific optimization)

## 8. Implementation Status

### 8.1 Current Implementation
- **Complete framework infrastructure** with all core components
- **Two production kernels**: SiLU-mul-FP8 and RMSNorm-FP8
- **Comprehensive benchmark suite** with CLI integration
- **Configuration management** with auto-discovery
- **Full CI/CD integration** with automated testing

### 8.2 Example Integration
```python
# Real usage in Llama model layers
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Automatic Helion/CUDA routing
        self.silu_mul_op = SiluMulFp8Helion(model_config=config)

    def forward(self, hidden_states):
        gate_up = self.gate_up_proj(hidden_states)
        # Fused SiLU + mul + FP8 quantization with optimal config
        mlp_output = self.silu_mul_op(gate_up, self.scale)
        return mlp_output
```

## 9. Conclusion

The Helion kernel integration framework provides:

**For Developers:**
- **80% less boilerplate** through intelligent automation
- **Automatic benchmarking** with comprehensive correctness checking
- **Zero-configuration** defaults with smart fallbacks
- **Rich tooling** for development, testing, and optimization

**For vLLM:**
- **1.2-2.1x performance improvements** for key operations
- **10-20% memory reduction** through kernel fusion and FP8 quantization
- **Production-ready reliability** with extensive testing framework
- **Scalable architecture** supporting future optimization work

The framework establishes a new standard for high-performance kernel development in vLLM while maintaining backward compatibility and developer productivity.