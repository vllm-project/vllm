# RFC: Helion Kernel Integration Framework for vLLM

(with significant inputs from @zou3519, @ProExpertProg, @mgoin, @xiaohongchen1991)

## Motivation & Goals
Helion is PyTorch's latest innovation in authoring custom kernels, featuring simple and familiar syntax, good developer experience, and superior performance.

This RFC proposes a developer-friendly framework for integrating Helion kernels into vLLM, making custom ops in vLLM more efficient, enjoyable to write, and performant in production.

The proposed integration is [prototyped here](https://github.com/vllm-project/vllm/pull/29051). It is still being actively developed for path finding, so there might be missing features, bugs, lint errors or even minor discrepancies between prototype implementation and what is described in this RFC.

## High-Level Overview

We aim to implement the following workflows for Helion authoring and deployment.

### Development Workflow
1. **Write Helion Kernels**: Implement GPU kernels using familiar PyTorch syntax with `@register_kernel`
2. **Create CustomOps**: Extend `HelionCustomOp` to integrate kernels into vLLM's compilation system
3. **Autotune Configurations**: Generate optimized configs for target models and hardware via autotuning hooks
4. **Benchmark Performance**: Validate kernel performance against baselines using the integrated benchmark framework

### Deployment Flow
1. **Server Startup**: vLLM discovers pre-generated configuration files and selects optimal configs based on model parameters
2. **Compilation Setup**: Fusion passes register CustomOps and identify operation sequences for replacement
3. **Warmup Phase**: Helion compiles kernel functions to optimized GPU code (~100ms per kernel, cached for reuse)
4. **Inference**: Optimized kernels execute with zero additional latency, seamlessly integrated into vLLM's execution pipeline

The framework requires **no changes to existing vLLM models** and integrates through the established CustomOp system, providing automatic fallback when Helion kernels are unavailable.

Please see the following sections for details of the workflows described above.

### Prototype Status

The [prototype PR](https://github.com/vllm-project/vllm/pull/29051) demonstrates the core framework capabilities but does not yet implement all features described in this RFC. The following features are **not yet implemented** in the prototype:

- **Batch size-based dispatching**: The runtime dispatcher that selects optimal kernel configs based on actual batch size (described in "Runtime Batch Size Dispatching")
- **GPU platform-based dispatching**: Config selection based on GPU type (e.g., H100 vs B200) mentioned in `config_picker`

These features are designed and specified in this RFC for future implementation. The prototype focuses on validating the core integration pattern and demonstrating performance potential.

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
- **Automatic optimization** - Built-in tiling, vectorization, and memory coalescing. Helion kernels can be understood by Inductor and fuse with other operators automatically.
- **Competitive performance** with significantly less code than hand-tuned CUDA kernels
- **Flexible deployment**: Easy to configure/compile/dispatch for different input shapes and platforms, to reach optimal performance in real-world deployment.

**Developer Productivity**:
- **Rapid development** - Write kernels with familiar Python semantics
- **Reuse PyTorch ops and torch.compile** - Authors can leverage PyTorch ops to express complex semantics
- **Easy debugging** - Python-like semantics with clear error messages
- **Automatic autotuning** - Helion explores optimization configurations automatically
- **PyTorch integration** - Seamlessly composes with many existing PyTorch workflows and components

### How Helion Works

A Helion kernel describes computation only—developers focus on the algorithm without worrying about GPU scheduling details. Helion has a smart autotuning process that can figure out the best configuration for a given kernel and input shapes. The configuration describes the compute schedule (block sizes, memory access patterns, etc.). Given a kernel and configuration, Helion generates high-performance GPU kernels automatically.

```
┌─────────────────────┐    ┌──────────────────┐
│   Helion Kernel     │    │    Autotuning    │
│  (Computation Only) │    │  (Input Shapes)  │
└─────────────────────┘    └──────────────────┘
    |      │                         │
    |      └─────────┬───────────────┘
    |                │
    |      ┌─────────▼─────────┐
    |      │   Configuration   │
    |      │ (Compute Schedule)│
    |      └─────────┬─────────┘
    └────────────────┘
            |
            |
  ┌─────────▼────────┐
  │ High-Performance │
  │   GPU Kernel     │
  └──────────────────┘
```

## Helion+vLLM Integration Architecture

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
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   Kernel Layer  │   │ Developer Tools │   │  Saved Configs  │
│ (Helion Kernels,│   │ (Benchmark      │   │ (JSON files     │
│  ConfigManager) │   │  Framework,     │   │  from Auto-     │
└─────────────────┘   │  Autotuning)    │   │  tuning)        │
                      └─────────────────┘   └─────────────────┘
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

**Developer Tools**:
- **Benchmark Framework**: Standardized testing with correctness verification and performance measurement
- **Autotuning Infrastructure**: Configuration exploration and optimization

**Saved Configs**:
- **Config Sets**: Optimized configurations organized into directories by model parameters (e.g., `silu_mul_fp8/4096/`), with each directory containing all batch-size-specific configs. This allows time-consuming autotuning at development time rather than on the deployment critical path.
- **Model and Batch-Size Optimization**: Each config set targets specific model parameter values and GPU architectures; batch size variants within the set cover different batch size ranges
- **Runtime Dispatching**: At startup, the `config_picker` selects a config set based on model parameters. The framework then handles batch size dispatching automatically at CUDA graph capture time.

## Authoring Experience

To create a CustomOp that leverages Helion kernels, one needs to implement a Helion kernel function decorated by @register_kernel and a subclass of HelionCustomOp that uses the Helion kernel via `create_kernel`:

**Writing Helion kernel, decorate with @vllm.helion.register_kernel:**
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
        # create_kernel returns a dispatcher that handles batch-size selection
        self.silu_mul_fp8 = self.create_kernel(silu_mul_fp8)

    def forward_helion(self, input, scale):
        # Dispatcher automatically selects optimal config based on input batch size
        return self.silu_mul_fp8(input, scale)
```

**Note**: `create_kernel` returns a `HelionKernelDispatcher` that wraps multiple compiled kernels optimized for different batch sizes. The dispatch overhead is minimal (single binary search) and amortized across the kernel execution time.

**Summary**: In around 20 lines of code, this example implements an efficient Helion kernel that is usable by vLLM. It handles PyTorch registration, schema inference, configuration management, and **automatic fake kernel generation** from type annotations automatically.


## helion.Config Management & Autotuning Infrastructure

To ensure Helion+vLLM can effectively manage many Helion kernel configs to cover different input shape/types, batch sizes, and GPU types, we propose to tackle config management with two simple hooks on Helion kernels. Developers can register their own autotune inputs generator and Helion config picker to simplify autotuning while retaining high flexibility.

The framework abstracts batch size handling away from kernel authors. Configs are organized into **config sets** (e.g., by hidden_size), where each config set contains all batch-size-specific variants. Kernel authors only select which config set to use based on model parameters—the framework handles batch size dispatching automatically.

See the following examples for autotuning hook implementation for silu_mul_fp8. The following implementation allows it to autotune for 4 most common hidden sizes among popular LLMs across multiple batch sizes, and pick the most appropriate config set at runtime based on model parameters.

```python
@silu_mul_fp8.register_autotune_inputs_generator
def generate_silu_mul_fp8_autotune_inputs(batch_sizes: list[int]) -> dict[str, dict[int, tuple]]:
    """
    Generate autotune inputs for common hidden_size values across batch sizes.

    Returns a nested dictionary: {config_set_name: {batch_size: inputs}}
    The framework handles batch size iteration internally.
    """
    config_sets = {}
    hidden_sizes = [2048, 4096, 5120, 8192]

    for hidden_size in hidden_sizes:
        config_set_name = str(hidden_size)
        config_sets[config_set_name] = {}

        for batch_size in batch_sizes:
            input_tensor = torch.randn(
                batch_size, 2 * hidden_size, dtype=torch.bfloat16, device="cuda"
            )
            scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
            config_sets[config_set_name][batch_size] = (input_tensor, scale)

    return config_sets

@silu_mul_fp8.register_config_picker
def pick_config(
    model_config,
    available_config_sets: dict[str, ConfigSet]
) -> ConfigSet:
    """
    Smart config selection based on model parameters.

    Each ConfigSet is an opaque object containing all batch-size-specific
    configs for a given model parameter combination. The framework handles
    batch size dispatching automatically—kernel authors just pick the
    config set that matches their model parameters.
    """
    target_size = model_config.get_hidden_size()

    # available_config_sets keys are like "2048", "4096", "5120", "8192"
    # Each ConfigSet internally contains configs for all batch sizes

    if str(target_size) in available_config_sets:
        return available_config_sets[str(target_size)]

    # Fallback to closest hidden_size match
    closest_key = min(available_config_sets.keys(),
                      key=lambda x: abs(int(x) - target_size))
    return available_config_sets[closest_key]
```

Specifically:

- **`register_autotune_inputs_generator`**: Generates representative input tensors for autotuning. Accepts a `batch_sizes` parameter (provided by the autotuning tool) and returns a nested dictionary `{config_set_name: {batch_size: inputs}}`. The framework iterates over batch sizes internally, so kernel authors only define the config set structure (e.g., by hidden_size).

- **`register_config_picker`**: Selects the appropriate **config set** at runtime based on model parameters. Returns a single `ConfigSet` object that encapsulates all batch-size variants—kernel authors never deal with batch size selection directly. The framework handles batch size dispatching automatically at graph capture time. It is not shown in the prototype example, but we would also allow picking config sets based on GPU type (like H100 vs B100).

**Autotune Tool Integration:**

To leverage the registered config hooks and generate configs for the kernel, one can run this command:

```bash
# Generate helion.Configs with specified batch size ranges
# Emphasis on small batch sizes which are common in LLM inference (decode phase)
> python scripts/autotune_helion_kernels.py --kernel silu_mul_fp8 \
    --batch-sizes 1,2,4,8,32,128,256,1024

# This creates config sets organized by model parameters (hidden_size),
# with each config set containing all batch size variants:
# vllm/compilation/helion/configs/silu_mul_fp8/
# ├── 2048/                    # Config set for hidden_size=2048
# │   ├── bs1.json
# │   ├── bs2.json
# │   ├── bs4.json
# │   ├── bs8.json
# │   ├── bs32.json
# │   ├── bs128.json
# │   ├── bs256.json
# │   └── bs1024.json
# ├── 4096/                    # Config set for hidden_size=4096
# │   ├── bs1.json
# │   ├── bs2.json
# │   └── ...
# ├── 5120/
# │   └── ...
# └── 8192/
#     └── ...
```

The autotune tool uses the registered input generator to create test cases, explores different Helion configurations, and saves configs organized by config set. Each config set directory is treated as an atomic unit—the `config_picker` selects a directory (e.g., `4096/`), and the framework loads all batch size variants within it.

Note: autotuning is a time consuming process, its time cost depends on size of kernel, size of tensors and number of input shapes to autotune for. Each set of input can take from minutes to over an hour to complete. Helion provides a "quick" autotuning mode that can cut the time significantly during development iteration. Before final deployment, one can use full autotuning mode to get the optimal config and save it.


## Benchmarking Infrastructure [Optional]

To compare performance of Helion kernels, one just needs to implement a subclass of KernelBenchmark, providing implementations for getting benchmark inputs and a baseline to compare against. 

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

With the subclass implemented, a developer can run a command to get a performance report and representative GPU trace for further analysis. For example:

```bash
> python benchmarks/benchmark_helion.py --benchmark silu_mul_fp8_helion --hidden-size=2048

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

Trace files saved to: /tmp/helion_benchmark_silu_mul_fp8_helion_20251209_222801
```

## Fusion Pass Integration Example

The framework integrates seamlessly with vLLM's existing fusion pass system. Here's how the `ActivationQuantFusionPass` detects and replaces operations with Helion kernels:

**Pattern Matching and Replacement Logic:**
```python
class SiluMulFp8StaticQuantPattern(ActivationQuantPattern):
    def register(self, pm_pass: PatternMatcherPass, vllm_config: VllmConfig = None):
        # Create Helion custom op instance if available and model_config is provided
        if (
            HELION_IMPORT_AVAILABLE
            and vllm_config is not None
            and vllm_config.model_config is not None
        ):
            try:
                self.helion_custom_op = SiluMulFp8Helion(
                    model_config=vllm_config.model_config  # Auto-configures optimal config
                )
            except Exception:
                self.helion_custom_op = None

        def pattern(input: torch.Tensor, scale: torch.Tensor):
            # Match: SiluAndMul -> FP8Quantization sequence
            result_silu_mul = self.silu_and_mul_matcher(input)
            result_quant = self.quant_matcher(result_silu_mul, scale)
            return result_quant[0]

        def replacement(input: torch.Tensor, scale: torch.Tensor):
            # Check if Helion custom op is available and enabled
            if self.helion_custom_op is not None and self.helion_custom_op.enabled():
                return self.helion_custom_op.forward_helion(input, scale)
            else:
                d = input.shape[-1] // 2
                output_shape = input.shape[:-1] + (d,)
                result = torch.empty(output_shape, device=input.device, dtype=self.quant_dtype)
                return torch.ops._C.silu_and_mul_quant(result, input, scale)[1]

        # Register pattern replacement with PyTorch's pattern matcher
        register_replacement(pattern, replacement, inputs, fwd_only, pm_pass)
```

`helion_custom_op.enabled()` reuses all existing vLLM CustomOp enablement infrastructure to decide whether a HelionCustomOp should be used.


## What Happens at Runtime

When a vLLM server starts, the framework automatically discovers available config sets (e.g., `silu_mul_fp8/4096/`) and uses the registered `config_picker` function to select a **ConfigSet** based on static model parameters like hidden size. The selected ConfigSet contains all batch-size-specific configs for that model parameter combination. During vLLM's compilation setup, fusion passes register `HelionCustomOp` instances with the selected ConfigSet and use PyTorch's pattern matcher to identify operation sequences that can be replaced with Helion kernels. During warmup, Helion compiles **multiple kernel variants** from the ConfigSet (one per batch-size threshold, approximately 100ms each). At inference time with cudagraph enabled, there is **zero dispatch overhead**—the appropriate kernel variant is selected during graph capture, and the captured graph executes directly.

### Runtime Batch Size Dispatching

The framework performs two-level config selection:

1. **Startup-time selection**: Based on static model parameters (hidden_size, GPU type), the `config_picker` returns a `ConfigSet`. Kernel authors only deal with this level—selecting the right config set for their model.

2. **Graph capture-time dispatching**: The framework internally extracts batch-size-specific configs from the ConfigSet and dispatches to the optimal kernel variant during cudagraph capture. Once captured, inference executes the graph directly with zero dispatch overhead.

**Internal Dispatch Logic** (handled by the framework, not exposed to kernel authors):
```python
class HelionKernelDispatcher:
    def __init__(self, config_set: ConfigSet):
        """Initialize from a ConfigSet containing all batch size variants."""
        # Framework extracts batch size configs from the opaque ConfigSet
        self.thresholds = config_set.get_batch_size_thresholds()
        self.kernels = [config_set.get_compiled_kernel(bs) for bs in self.thresholds]

    def __call__(self, *args, **kwargs):
        # Infer batch_size from first tensor argument
        batch_size = args[0].shape[0]

        # Binary search for largest threshold <= batch_size
        idx = bisect.bisect_right(self.thresholds, batch_size) - 1
        idx = max(0, idx)  # Fallback to smallest if batch_size < min threshold

        return self.kernels[idx](*args, **kwargs)
```

**Example**: For a model with hidden_size=4096, the `config_picker` returns the `ConfigSet` from `silu_mul_fp8/4096/`. The framework then handles batch size dispatching internally (using configs autotuned at batch sizes [1, 2, 4, 8, 32, 128, 256, 1024]):
- Batch size 1 → uses config from `bs1.json`
- Batch size 3 → uses config from `bs2.json`
- Batch size 6 → uses config from `bs4.json`
- Batch size 16 → uses config from `bs8.json`
- Batch size 64 → uses config from `bs32.json`
- Batch size 200 → uses config from `bs128.json`
- Batch size 512 → uses config from `bs256.json`
- Batch size 2048 → uses config from `bs1024.json`

### Overhead
As stated earlier, each Helion kernel variant is compiled once per (model_config, batch_size_threshold) combination; each compilation takes around 100-200ms based on my experience so far. A model using 10 Helion kernels with 8 batch-size thresholds (e.g., [1, 2, 4, 8, 32, 128, 256, 1024]) would compile 80 kernel variants, requiring an additional 8-16 seconds during cold compilation. Since vLLM captures cudagraph for all batch sizes during warmup, all kernel variants are compiled eagerly. Compiled Helion kernels can be easily cached like other torch.compile artifacts, so future runs will not incur the compilation cost again.


## Performance Evaluations

To demonstrate the competitive performance potential of Helion, we implemented three different types of ops and measured their performance with representative input shapes.

Operators:
- silu_mul_fp8: element-wise + quantization
- rmsnorm_fp8: element-wise + reduction + quantization
- allreduce_add_rmsnorm: collective + element-wise + reduction

Platforms for evaluation:
- 2 x H200 (only one node used for silu_mul_fp8, rmsnorm_fp8)
- 2 x B200 (only one node used for silu_mul_fp8, rmsnorm_fp8)
- All with cudagraph enabled
- Warmup for 50 iterations, then benchmark 5000 iterations back to back

Autotuning parameters:
- Full Helion autotuning mode
- Autotuned for 4 different representative `hidden_size`: 2048, 4096, 5120, 8192
- To save time, batch size 256 is used to autotune all configs, but they are tested on all batch sizes. This should give us the lower bound of speedups since in real production environment, batch-size specific configs will be used

Autotuning/Benchmarking script: scripts/comprehensive_helion_benchmark.py

### NVIDIA H200 Results

| Kernel | Autotuned for Hidden Size | Helion Avg (ms) | Baseline Avg (ms) | GeoMean Speedup | Median Speedup | Min-Max Speedup |
|--------|---------------------------|-----------------|-------------------|-------------|----------------|-----------------|
| **silu_mul_fp8** | 2048 | 0.0129 | 0.0163 | **1.27x** | 1.24x | 0.88x - 2.48x |
| | 4096 | 0.0130 | 0.0164 | **0.99x** | 1.14x | 0.19x - 3.33x |
| | 5120 | 0.0122 | 0.0163 | **1.28x** | 1.23x | 0.85x - 2.42x |
| | 8192 | 0.0181 | 0.0165 | **1.04x** | 0.98x | 0.66x - 2.08x |
| **rms_norm_fp8** | 2048 | 0.0050 | 0.0057 | **1.08x** | 1.00x | 0.66x - 2.12x |
| | 4096 | 0.0050 | 0.0059 | **1.11x** | 1.00x | 0.66x - 2.69x |
| | 5120 | 0.0050 | 0.0057 | **1.08x** | 1.00x | 0.66x - 2.14x |
| | 8192 | 0.0054 | 0.0056 | **1.03x** | 1.00x | 0.41x - 1.94x |
| **allreduce_add_rmsnorm** | 2048 | 0.1365 | 0.1283 | **0.92x** | 0.86x | 0.81x - 2.29x |
| | 4096 | 0.1297 | 0.1256 | **0.94x** | 0.88x | 0.83x - 2.46x |
| | 5120 | 0.1338 | 0.1276 | **0.93x** | 0.87x | 0.82x - 2.40x |
| | 8192 | 0.1350 | 0.1274 | **0.92x** | 0.86x | 0.81x - 2.42x |

### NVIDIA B200 Results

| Kernel | Autotuned for Hidden Size | Helion Avg (ms) | Baseline Avg (ms) | GeoMean Speedup | Median Speedup | Min-Max Speedup |
|--------|---------------------------|-----------------|-------------------|-------------|----------------|-----------------|
| **silu_mul_fp8** | 2048 | 0.0069 | 0.0160 | **2.12x** | 2.00x | 1.26x - 3.73x |
| | 4096 | 0.0064 | 0.0160 | **2.21x** | 2.01x | 1.26x - 3.70x |
| | 5120 | 0.0065 | 0.0160 | **2.21x** | 2.00x | 1.26x - 3.78x |
| | 8192 | 0.0063 | 0.0160 | **2.14x** | 2.00x | 1.02x - 3.73x |
| **rms_norm_fp8** | 2048 | 0.0045 | 0.0054 | **1.20x** | 1.26x | 0.68x - 1.72x |
| | 4096 | 0.0041 | 0.0053 | **1.26x** | 1.27x | 1.00x - 1.90x |
| | 5120 | 0.0038 | 0.0052 | **1.32x** | 1.24x | 0.98x - 2.00x |
| | 8192 | 0.0043 | 0.0052 | **1.27x** | 1.26x | 0.44x - 1.90x |
| **allreduce_add_rmsnorm** | 2048 | 0.0986 | 0.1276 | **1.05x** | 0.73x | 0.59x - 6.41x |
| | 4096 | 0.0987 | 0.1275 | **1.05x** | 0.73x | 0.55x - 6.50x |
| | 5120 | 0.0979 | 0.1266 | **1.04x** | 0.73x | 0.57x - 6.70x |
| | 8192 | - | - | - | - | - |

*Note: allreduce_add_rmsnorm hidden size 8192 failed during B200 benchmarking, so no results yet*

### Performance Summary

**Overall Geometric Mean Speedups (across all configurations):**

| Platform | **silu_mul_fp8** | **rms_norm_fp8** | **allreduce_add_rmsnorm** | **Overall** |
|----------|------------------|------------------|--------------------------|-------------|
| **H200** | **1.14x** | **1.07x** | **0.93x** (slowdown) | **1.04x** |
| **B200** | **2.17x** | **1.26x** | **1.05x** | **1.49x** |
| **B200 Advantage** | +90.5% | +17.4% | +12.8% | **+42.7%** |
