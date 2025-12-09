# RFC: Helion Kernel Integration Framework for vLLM

## Motivation & Goals
Helion is PyTorch's latest innovation in authoring custom kernels, featuring simple and familiar syntax, good developer experience and superior performance.

This RFC proposes a developer-friendly framework for integrating Helion kernels into vLLM, making custom op in vLLM more efficient, enjoyable to write and performant in production.

## What is Helion?

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
- **Reuse PyTorch ops and torch.compile** - Authors can leverage PyTorch ops to express complex semantics
- **Easy debugging** - Python-like semantics with clear error messages
- **Automatic autotuning** - Helion explores optimization configurations automatically
- **PyTorch integration** - Seamless compose with many existing PyTorch workflows and components

### How Helion Works

A Helion kernel describes computation only - developers focus on the algorithm without worrying about GPU scheduling details. Helion has a smart autotuning process that can figure out the best configuration for a given kernel and input shapes. The configuration describes the compute schedule (block sizes, memory access patterns, etc.). Given a kernel and configuration, Helion generates high-performance GPU kernels automatically.

```
┌─────────────────────┐    ┌──────────────────┐
│   Helion Kernel     │    │    Autotuning    │
│  (Computation Only) │    │  (Input Shapes)  │
└─────────────────────┘    └──────────────────┘
           │                         │
           └─────────┬───────────────┘
                     │
           ┌─────────▼─────────┐
           │   Configuration   │
           │ (Compute Schedule)│
           └─────────┬─────────┘
                     │
           ┌─────────▼─────────┐
           │ High-Performance  │
           │   GPU Kernel      │
           └───────────────────┘
```

## Framework Architecture

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

## Authoring Experience

To create a CustomOp that leverages Helion kernels, one need to implement a Helion kernel function decorated by @register_kernel
and a subclass of HelionCustomOp that uses the Helion kernel via `create_kernel`. For example:

**Kernel Registration Decorator:**
```python
@register_kernel
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
```

**HelionCustomOp Integration:**
```python
@CustomOp.register("silu_mul_fp8_helion")
class SiluMulFp8Helion(HelionCustomOp):
    def __init__(self, model_config, **kwargs):
        super().__init__(model_config, **kwargs)
        self.silu_mul_fp8 = self.create_kernel(silu_mul_fp8)  # Auto-configured

    def forward_helion(self, input, scale):
        return self.silu_mul_fp8(input, scale)  # Direct call to optimized kernel
```


In around 20 lines of code, this example implements an efficient Helion kernel that is usable by vLLM.
It handles PyTorch registration, schema inference, configuration management, and **automatic fake kernel generation** from type annotations automatically.

## helion.Config Management

Helion kernels rely on helion.Config to decide the most efficient compute schedule. Each helion.Config
describes the best tiling/parallelism pattern to map Helion computation into actual performant GPU kernels.
helion.Configs are usually discovered through autotuning process, where helion performs a search within
the parameter space of computation schedules. However, Helion's builtin autotuning mechanism can only target
one set of input shapes at a time and generate one config while vLLM targets hundreds of models and numerous
GPU types, it would be cumbersome for a kernel author to manage many configs and perform dispatching at runtime.

I propose to tackle config management with two simple hooks on helion kernels. Developers can register their own
autotune inputs generator and helion config picker to simplify autotuning while retaining high flexibility.


See following examples for autotuning hook implementation for silu_mul_fp8:

```python
@silu_mul_fp8.register_autotune_inputs_generator
def generate_silu_mul_fp8_autotune_inputs() -> dict[str, tuple]:
    """Generate autotune inputs for common hidden_size values."""
    inputs = {}
    hidden_sizes = [2048, 4096, 5120, 8192]
    batch_size = 256  # Representative batch size for autotuning

    for hidden_size in hidden_sizes:
        input_tensor = torch.randn(
            batch_size, 2 * hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")

        config_name = str(hidden_size)
        inputs[config_name] = (input_tensor, scale)

    return inputs

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
```

- **`register_autotune_inputs_generator`**: Generates representative input tensors for autotuning based on model configuration. Used during development/deployment to find the best kernel configurations through performance testing. It returns a dictionary of {config name: inputs used to autotune for the config}. Our autotune tool can
then work based on this dictionary to generate many configs, each optimized for one set of input shapes.

- **`register_config_picker`**: Selects the appropriate configuration at runtime based on model parameters (e.g., hidden size). Used during inference to choose the optimal kernel configuration. It is not shown in the prototype example, but we would allow picking a config based on GPU type as well (like H100 vs B100).

**Autotune Tool Integration:**
```bash
# Generate optimal configurations for specific model sizes
python scripts/autotune_helion_kernels.py --kernel silu_mul_fp8 \
    --hidden-sizes 2048,4096,8192

# This creates optimized config files:
# vllm/compilation/helion/configs/silu_mul_fp8_2048.json
# vllm/compilation/helion/configs/silu_mul_fp8_4096.json
# vllm/compilation/helion/configs/silu_mul_fp8_8192.json
```

The autotune tool uses the registered input generator to create test cases, explores different Helion configurations, and automatically saves the best-performing configurations as JSON files that the config picker can later select from.


## Automatic Benchmarking

To understand whether an authored Helion kernel/custom op fares well against other implementations,
one just needs to implement a subclass of KernelBenchmark, providing implementation for
getting benchmark inputs and a baseline to compare against. 

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

With the subclass implemented, developer can run a command to get a performance report. e.g.:

```
python benchmarks/benchmark_helion.py --benchmark silu_mul_fp8_helion --hidden-size=2048

......logs......

============================================================
Summary Statistics
============================================================
Total configurations tested: 242

Speedup:
  Average: 1.42x
  Median:  1.20x
  Min:     0.60x
  Max:     2.85x

Latency (ms):
  Baseline - Avg: 0.0134, Min: 0.0030, Max: 0.2139
  Helion   - Avg: 0.0079, Min: 0.0048, Max: 0.0948
============================================================
```

## What Happens at Runtime

When a vLLM server starts, the framework automatically discovers available configuration files (e.g., `silu_mul_fp8_4096.json`) and uses the registered `config_picker` function to select optimal configurations based on model parameters like hidden size. During vLLM's compilation setup, fusion passes register `HelionCustomOp` instances with the selected configurations and use PyTorch's pattern matcher to identify operation sequences that can be replaced with Helion kernels. When vLLM runs forward passes during its warmup phase, Helion compiles the Python kernel functions to optimized GPU code (approximately 100ms per kernel), with each kernel compiled only once per model configuration and cached for all subsequent inference requests. This approach ensures that even the small compilation overhead is absorbed during server startup, with no additional latency during actual inference serving.

### Overhead
As started earlier, each Helion kernel is compiled once per model, each compilation takes around 100-200ms based on my experience so far. A model that uses 10 Helion kernels would need an additional 1-2 seconds during its cold compilation. Compiled helion kernels can be easily cached like other torch.compile artifacts, so future runs will not incur the compilation cost again.


## Performance evaluations

WIP


## Open Questions for Review

### Default Configuration Support
Should the `@register_kernel` decorator support a `default_config` parameter for specifying fallback Helion configurations when model-specific configs are unavailable?

**Current Implementation**: Kernels rely entirely on external JSON configuration files with model-specific naming (e.g., `silu_mul_fp8_4096.json`)

**Potential Benefits of Default Config**:
- Simplified initial development and testing
- Graceful fallback when specific model configurations don't exist
- Reduced dependency on external configuration files

**Questions**:
- Would this add complexity that conflicts with the configuration management system?
- Should default configs be embedded in code or still externalized?
- How would default configs interact with the autotuning infrastructure?

**Reviewer Input Requested**: Do you see value in supporting default configurations, or should we maintain the current approach of requiring external configuration files for all deployment scenarios?