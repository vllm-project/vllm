# vLLM IR: Functional Intermediate Representation

## Motivation

vLLM IR is a **functional intermediate representation (IR)** that fills the gap between
low-level `torch` ops and vLLM layers like `RMSNorm` and quantization operators,
By separating operator **semantics** from the **implementation** and **dispatching**,
vLLM IR simplifies both compilation and kernel registration & dispatching simultaneously.
It operates as a **dialect** in the torch FX representation, allowing full interoperability
with “regular” torch ops & custom torch ops/kernels, as well as a piecewise migration from
the previous `CustomOp` approach.

Key design principles:

- **Eager-compile consistency**: identical behavior (barring minor numerics) in eager and compiled modes
- **Simple, transparent, yet powerful kernel selection**: good visibility and control allowing easy debugging
- **Convention over configuration**: near-zero boilerplate required to register ops and implementations
- **Extensibility**: ops and implementations can be registered anywhere, in-tree or out-of-tree
- **Interoperability**: fully compatible with “regular” torch ops & custom torch ops/kernels,
reducing developer friction and allowing piecewise migration

The clean semantics/implementation separation enables a unified and extensible dispatching mechanism,
allowing multiple kernels per-platform and powerful kernel selection. The separation also facilitates
cleaner testing and benchmarking, removing much of the boilerplate standard for legacy approaches.

By delaying kernel selection until late in the compilation process, the compiler can operate on
a higher-level representation, which has the following main benefits:

- Pattern matching in fusion/transformation passes only requires a single, simple pattern per op
- OOT compiler backends can lower from the higher-level representation (in-progress)
- The compiler can autotune over available implementations (future feature)

## Quick Overview

### Declaring an IR Operation

IR operations are declared using the `@register_op` decorator with a native PyTorch implementation that defines the op's semantics:

```python
# vllm/ir/ops/layernorm.py
from torch import Tensor
from vllm.ir import register_op

@register_op(has_reduction=True)
def rms_norm(x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None) -> Tensor:
    """Weighted root-mean-square layer normalization"""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x_var = x if variance_size is None else x[..., :variance_size]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    x = x.to(orig_dtype)
    if weight is not None:
        x = x * weight
    return x
```

The native implementation serves three purposes:

1. **Semantic definition**: Specifies the exact semantics of the operation, including shapes and strides
2. **Default implementation**: Used when no other (better) implementation is available
3. **Reference for testing**: Other implementations must match these semantics

### Registering Implementations

Kernel implementations are registered using the `register_impl` decorator on the IR op object:

```python
# vllm/kernels/vllm_c.py
from vllm import ir

rms_norm_no_var = lambda x, weight, epsilon, variance_size=None: variance_size is None

@ir.ops.rms_norm.register_impl("vllm_c", supports_args=rms_norm_no_var, supported=current_platform.is_cuda_alike())
def rms_norm(x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None) -> Tensor:
    output = torch.empty_like(x)
    torch.ops._C.rms_norm(output, x, weight, epsilon)
    return output
```

Implementations can specify:

- `supported`: Static boolean indicating if this implementation is available
- `supports_args`: Function checking if the implementation supports specific arguments
- `batch_invariant`: Whether this implementation produces batch-invariant results
- `inplace`: Whether this implementation reuses input memory for outputs

### Using IR Operations in Models

IR operations are imported and called directly in model code:

```python
# vllm/model_executor/layers/layernorm.py
from vllm import ir

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: Tensor, residual: Tensor | None = None):
        if residual is None:
            return ir.ops.rms_norm(x, self.weight, self.variance_epsilon)

        # Use maybe_inplace overload to allow implementation to reuse input memory for outputs
        # (using x or residual after this call is undefined behavior)
        return ir.ops.fused_add_rms_norm.maybe_inplace(
            x, residual, self.weight, self.variance_epsilon
        )
```

### Configuring Kernel Selection

Kernel selection is controlled via priority lists in the configuration.
Priority lists specify the order in which implementations are considered,
with the first supported implementation being selected.
This includes the static support check (`supported=...`),
the dynamic arg support check (`supports_args=...`),
and batch-invariance filtering if `VLLM_BATCH_INVARIANT=1` is set.

#### Command Line Configuration

Use `--ir-op-priority.<op_name>=<provider1>,<provider2>,...`:

```bash
# CUDA: Use vllm_c implementation for rms_norm
vllm serve meta-llama/Llama-3.2-1B \
  --ir-op-priority.rms_norm=vllm_c

# ROCm: Try aiter first, fall back to vllm_c, then native
vllm serve meta-llama/Llama-3.2-1B \
  --ir-op-priority.rms_norm=aiter,vllm_c,native

# Configure multiple operations
vllm serve meta-llama/Llama-3.2-1B \
  --ir-op-priority.rms_norm=vllm_c \
  --ir-op-priority.fused_add_rms_norm=vllm_c
```

#### Python Configuration

```python
from vllm import LLM
from vllm.config import VllmConfig, KernelConfig

llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    vllm_config=VllmConfig(
        kernel_config=KernelConfig(
            ir_op_priority={
                "rms_norm": ["vllm_c", "native"],
                "fused_add_rms_norm": ["vllm_c", "native"],
            }
        )
    )
)
```

#### Platform Defaults

Each platform provides default priority lists that are automatically applied:

```python
# CUDA platform defaults (when compiling with Inductor)
{
  "rms_norm": ["native"],  # Native torch is default
  "fused_add_rms_norm": ["native"],
}

# CUDA platform defaults (eager or Dynamo-only)
{
  "rms_norm": ["vllm_c", "native"],
  "fused_add_rms_norm": ["vllm_c", "native"],
}

# ROCm platform defaults (future - currently same as CUDA)
{
    "rms_norm": ["aiter", "vllm_c", "native"],
    "fused_add_rms_norm": ["aiter", "vllm_c", "native"],
}
```

User-specified priorities are prepended to platform defaults,
so you only need to specify the out-of-order implementations,
other implementations are appended automatically.

## Compilation Pipeline

vLLM IR heavily customizes the `torch.compile`-based compilation process to allow custom compile
passes to operate on high-level IR while still producing efficient low-level code at the end.
The compilation pipeline consists of several stages:

### 1. Dynamo Tracing

When `torch.compile` traces the model's forward pass, vLLM IR operations appear as custom operations
in the `vllm_ir` torch library. These operations are opaque to Dynamo, meaning they appear directly
in the FX graph without decomposition:

```python
# Python code (epsilon=1e-5)
x1 = ir.ops.rms_norm(x, weight, epsilon)
x2, residual_out = ir.ops.fused_add_rms_norm,maybe_inplace(x1, residual, weight, epsilon)

# FX graph after Dynamo tracing
x1 = torch.ops.vllm_ir.rms_norm.default(x, weight, 1e-5); x = None
out = torch.ops.vllm_ir.fused_add_rms_norm.maybe_inplace(x1, residual, weight, 1e-5); x1 = residual = None
x2 = out[0]
residual_out = out[1]
```

### 2. AOTAutograd and Functionalization

AOTAutograd functionalizes the graph, converting any mutating operations to functional equivalents.
For vLLM IR operations with `maybe_inplace` overloads, we perform this manually before AOTAutograd,
converting them to the functional `default` overload using the pre-grad custom pass hook.

```python
# After functionalization
x1 = torch.ops.vllm_ir.rms_norm.default(x, weight, 1e-5); x = None
out = torch.ops.vllm_ir.fused_add_rms_norm.default(x1, residual, weight, 1e-5); x1 = residual = None
x2 = out[0]
residual_out = out[1]
```

The pass also tracks which inputs were "donated" (passed to `maybe_inplace`),
storing this information in vLLM's  `PassContext` for later use in clone elimination.

### 3. IR Fusion and Transformation Passes

After functionalization, custom vLLM passes operate on the functional FX graph containing high-level IR operations.
These passes can perform fusion, distribute operations for sequence parallelism, and other transformations:

```python
# Example: Sequence Parallelism (see SequenceParallelismPass)
# Before SP pass

all_reduce = torch.ops.vllm.all_reduce(x, "tp:0")
rms_norm = torch.ops.vllm_ir.rms_norm(all_reduce, weight, 1e-5)

# after SP pass
reduce_scatter = torch.ops.vllm.reduce_scatter(x, "tp:0")
rms_norm = torch.ops.vllm_ir.rms_norm(all_reduce, weight, 1e-5)
all_gather = torch.ops.vllm.all_gather(x, "tp:0")
```

Fusion passes benefit from the high-level representation: they don't need to match against low-level PyTorch operations,
handle different kernel implementations separately, or deal with functionalization of custom kernels.

### 4. IR Lowering

The lowering pass (`VllmIRLoweringPass`) replaces each vLLM IR operation with its selected implementation.
The implementation is chosen based on the priority list and support predicates,
using the **fake tensors** in the graph's metadata in place of op arguments:

```python
# Implementation selection, same in eager dispatch and compile lowering
def dispatch(*args) -> IrOpImpl:
  for provider in priority_list:  # e.g., ["vllm_c", "native"]
    impl = ir_op.impls[provider]
    if not impl.supported:
      continue
    if impl.supports_args and not impl.supports_args(*args):
      continue
    return impl

# make_fx uses torch.fx.symbolic_trace
impl_graph = make_fx(selected_impl.impl_fn)
# Replace IR op node with impl_graph's nodes
match.replace_by_example(selected_impl.impl_fn, node.args)
```

For example, lowering `rms_norm` with the `vllm_c` implementation:

```python
# Before lowering (IR op)
rms_norm = torch.ops.vllm_ir.rms_norm.default(x, weight, 1e-5)

# After lowering (vllm_c implementation traced)
# Note: Lowering does not currently functionalize, this will likely change in the future.
empty =  torch.ops.aten.empty.memory_format(x.shape, ...)
rms_norm = torch.ops._C.rms_norm(empty, x, weight, 1e-5)
```

When lowering an implementation that mutates inputs (`inplace=True`),
the lowering pass inserts clones to preserve functional semantics:

```python
# vllm_c implementation for fused_add_rms_norm mutates its first two arguments
# Lowered with clones for safety
clone_default = torch.ops.aten.clone.default(x)
clone_default_1 = torch.ops.aten.clone.default(residual)
fused_add_rms_norm = torch.ops._C.fused_add_rms_norm.default(clone_default, clone_default_1, weight, 1e-5)
```

### 5. Clone Cleanup

After lowering, the clone elimination pass (`UnsafeCloneEliminationPass`) removes unnecessary clones introduced during lowering.
This pass is essential for achieving zero-copy behavior when using in-place kernels with `maybe_inplace`.
The pass removes a clone if:

- the cloned input is created in the graph and not used again in the graph
- the cloned input is a graph parameter, marked as donated

```python
# After cleanup (donated inputs, no subsequent uses)
fused_add_rms_norm = torch.ops._C.fused_add_rms_norm.default(x, residual, weight, 1e-5)
```

The combination of inplace functionalization (tracking donated inputs) and clone cleanup enables the compiler to safely
use in-place kernels without adding redundant copies or increasing the memory usage.

### 6. Inductor Optimization and Codegen

After IR lowering and cleanup, the graph contains only standard PyTorch operations and platform-specific custom ops.
Inductor then performs its standard codegen:

- **Inductor lowering and pointwise fusion**: Fusing element-wise operations, reductions, etc.
- **Memory planning**: Determining buffer allocation and reuse
- **Kernel generation**: Generating Triton or C++ code for fused operations
- **Autotuning**: Selecting the best kernel configurations

### Pipeline Summary

```text
Model Forward Pass
    ↓
[Dynamo Tracing] → FX Graph with vllm_ir.* ops
    ↓
[Pre-grad: Inplace Functionalization] → maybe_inplace → default, track donated inputs
    ↓
[AOTAutograd] → Functionalization
    ↓
[Post-grad: IR Fusion Passes] → Fuse high-level IR ops (e.g., rms_norm + quant)
    ↓
[Post-grad: IR Lowering] → vllm_ir.* ops → impl ops (with clones if needed)
    ↓
[Post-grad: Clone Cleanup] → Remove unnecessary clones using donated input info
    ↓
[Inductor] → Pattern matching, fusion, memory planning, codegen
    ↓
Compiled Code
```

## Core vLLM IR Concepts

### Operation Declaration

Operations are declared with the `@register_op` decorator, which creates an `IrOp` object:

```python
@register_op(
    name=None,           # Operation name (defaults to function name)
    has_reduction=False, # Whether op performs reduction (affects batch invariance)
    activations=None,    # List of activation parameters (defaults to params starting with 'x')
    allow_inplace=False, # Whether to create a maybe_inplace overload
)
def op_name(...):
    ...
```

**Parameters:**

- `has_reduction`: Set to `True` for operations like normalization that reduce across dimensions. This affects batch invariance requirements.
- `activations`: List of parameter names considered "activations" (typically consumed by `maybe_inplace`). Defaults to parameters starting with `x`.
- `allow_inplace`: Creates a `maybe_inplace` overload for memory-efficient execution (see below).

### The `maybe_inplace` Overload

The `maybe_inplace` overload is a critical feature for memory efficiency in LLM inference.
It signals that the caller doesn't need to preserve the activation inputs after the operation,
allowing in-place implementations to reuse input memory for outputs.

#### Semantics and Usage

```python
# Standard usage: inputs are preserved
out, res_out = ir.ops.fused_add_rms_norm(x, residual, weight, epsilon)
# x and residual are unchanged, out and res_out are new tensors

# maybe_inplace: inputs may be modified
out, res_out = ir.ops.fused_add_rms_norm.maybe_inplace(x, residual, weight, epsilon)
# x and residual may be modified (undefined behavior to use them after this)
# out and res_out may alias x and residual
```

Using an activation input after passing it to `maybe_inplace` is **undefined behavior**:

```python
# WRONG: Using x after donating it
out, res_out = ir.ops.fused_add_rms_norm.maybe_inplace(x, residual, weight, epsilon)
result = out + x  # ERROR: x was donated!
```

If you need to preserve an input, either use the default overload or clone manually:

```python
# Option 1: Use default overload
out, res_out = ir.ops.fused_add_rms_norm(x, residual, weight, epsilon)
result = out + x  # OK: x is preserved

# Option 2: Clone before maybe_inplace
out, res_out = ir.ops.fused_add_rms_norm.maybe_inplace(x.clone(), residual, weight, epsilon)
result = out + x  # OK: x is preserved, clone was donated
```

#### Compilation Behavior

During compilation, the inplace functionalization pass validates that donated inputs are
not used again and converts `maybe_inplace` to the functional `default` overload:

```python
# Inplace functionalization pass (pre-grad)
for node in graph.nodes:
    if node.target == torch.ops.vllm_ir.fused_add_rms_norm.maybe_inplace:
        # Check that activation inputs aren't used after this node
        for activation_arg in activation_inputs:
            for user in activation_arg.users:
                if user appears after node:
                    raise ValueError(f"Input {activation_arg} donated but used again")

        # Convert to default overload
        node.target = torch.ops.vllm_ir.fused_add_rms_norm.default

        # Track donated graph inputs for later clone elimination
        for i, arg in enumerate(node.args):
            if arg.op == "placeholder" and i in activation_indices:
                pass_context.donated_input_ids.add(node_to_idx[arg])
```

The donated input information is then used by the clone cleanup pass to eliminate
unnecessary copies when in-place kernels are lowered.

#### Eager Mode Behavior

In eager mode (without `torch.compile`), `maybe_inplace` enables **maximally memory-efficient**
execution by allowing the IR operation to dispatch directly to in-place implementations:

```python
# Eager dispatch logic for maybe_inplace
impl: IrOpImpl = ir_op.dispatch(*args)
return impl.impl_fn(*args)

# Eager dispatch logic for default:
impl: IrOpImpl = ir_op.dispatch(*args)
if impl.inplace:
  args = [
    arg.clone() if i in ir_op.activations else arg
    for i, arg in enumerate(args)
  ]
return impl.impl_fn(*args)
```

The combination of `maybe_inplace` in model code and in-place kernel implementations provides optimal memory efficiency
in both eager and compiled modes, with identical semantics in both cases.

#### Memory Savings Example

Consider a transformer layer with residual connections:

```python
# Without maybe_inplace (2 allocations per layer)
hidden_states = self.attention(input)
normed, residual = ir.ops.fused_add_rms_norm(hidden_states, input, weight, eps)
# Memory: input (preserved), hidden_states (preserved), normed (new), residual (new)

# With maybe_inplace (0 allocations per layer when using in-place kernel)
hidden_states = self.attention(input)
normed, residual = ir.ops.fused_add_rms_norm.maybe_inplace(hidden_states, input, weight, eps)
# Memory: normed (reuses hidden_states), residual (reuses input)
```

### Implementation Registration

Implementations are registered using the `register_impl` method:

```python
@ir.ops.op_name.register_impl(
    provider="provider_name",  # Unique identifier (e.g., "vllm_c", "aiter", "triton")
    supported=True,            # Static availability check
    supports_args=None,        # Dynamic argument support check
    batch_invariant=False,     # Whether implementation is batch-invariant
)
def impl_fn(...):
    ...
```

**Provider naming conventions:**

- `native`: Reserved for the native torch implementation (declared with `@register_op`)
- `vllm_c`: C++/CUDA kernels via `torch.ops._C`
- `aiter`: AMD AITER library
- `triton_*`: Triton kernels (e.g., `triton_batch_invariant`)
- Platform/library names for other implementations

**Support checking:**

- `supported`: Static boolean, checked once at import time (e.g., `HAS_TRITON`, `is_cuda_alike()`)
- `supports_args`: Function `(*args, **kwargs) -> bool` checking argument compatibility
    - Called with **fake tensors** during compilation for zero-cost checking
    - Called with **real tensors** during eager mode dispatch
    - Should NOT check batch sizes or add guards based on values

Example support predicate:

```python
def aiter_rms_norm_supports(x, weight, epsilon, variance_size=None):
    # Check dtype (OK: doesn't depend on batch size)
    if x.dtype not in [torch.float16, torch.bfloat16]:
        return False
    # Check optional parameter (OK: static check)
    if variance_size is not None:
        return False
    return True

@ir.ops.rms_norm.register_impl("aiter", supports_args=aiter_rms_norm_supports)
def rms_norm(...):
    ...
```

### Batch Invariance

Batch-invariant implementations produce identical per-example outputs regardless of batch size.
This is critical for RLHF and other training scenarios where numerical reproducibility is required.

```python
@ir.ops.rms_norm.register_impl("triton_batch_invariant", batch_invariant=True)
def rms_norm(...):
    # Implementation uses careful reduction algorithms to ensure
    # output is identical irrespective of the batch size
    ...
```

Batch-invariant kernels are automatically selected when `VLLM_BATCH_INVARIANT=1` is set.

### Eager Mode vs Compile Mode

vLLM IR operations behave identically in eager and compile modes:

**Eager mode:**

- Direct dispatch to implementation based on priority list
- Support checked with real tensor arguments
- Minimal overhead (can be optimized further if needed)

**Compile mode:**

- IR ops appear in FX graph as `torch.ops.vllm_ir.*` custom ops
- Lowering selects implementation using fake tensors
- Full integration with Inductor optimizations

This consistency enables:

- Prototyping in eager mode with confidence
- Debugging by disabling compilation
- Gradual migration from eager to compiled execution

## Other Topics

### Out-of-Tree Implementations

External platforms can register implementations without modifying vLLM:

```python
# In external package
from vllm import ir

@ir.ops.rms_norm.register_impl("my_platform", supported=is_my_platform())
def rms_norm(x, weight, epsilon, variance_size=None):
    return my_platform.rms_norm(x, weight, epsilon)
```

Then configure priority to use your implementation:

```python
class MyPlatform(Platform):
  def get_default_ir_op_priority(self):
    return IrOpPriorityConfig(rms_norm=['my_platform', 'native'])

# Users can still override priority in the same way
llm = LLM(ir_op_priority=IrOpPriorityConfig(rms_norm=['custom_oot_kernel']))
```

### Debugging and Observability

!!! note
    Please let us know how observability can be improved for your use-case!

Enable debug logging to see kernel selection:

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve ...
```

This logs:

- Which implementations are selected for each operation
- Why implementations were rejected (unsupported, args not supported)
- Compilation cache hits/misses
- IR lowering statistics

Check selected implementations in compiled graphs:

```python
# After compilation, inspect the lowering pass
lowering_pass = backend.lowering_pass
print(lowering_pass.selected_impls)
# Output: {'rms_norm': {'node_123': 'vllm_c', 'node_456': 'vllm_c'}}
```

## Migration from CustomOp

vLLM IR is designed to coexist with and gradually replace `CustomOp`:

1. **Op declaration**: Convert `CustomOp` class `PluggableLayer` and move `forward_native` to `@register_op` function
2. **Implementation registration**: Use `@ir.ops.op_name.register_impl` instead of overriding methods
3. **Layer usage**: Replace `self.op(...)` with `ir.ops.op_name(...)`
4. **Configuration**: Migrate `--compilation-config.custom-ops` to `--ir-op-priority`

The migration can be done incrementally, one operation at a time.

## See Also

- [torch.compile Integration](torch_compile.md) - General compilation infrastructure
- [Custom Operations](custom_op.md) - Legacy custom op system
- [Kernel Development Guide](../../contributing/kernel_development.md) - Writing and registering kernels
