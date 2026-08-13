# JIT Kernel Warmup

vLLM uses JIT-generated kernels from Triton, CuTeDSL, TileLang, and other backends. This contract makes their required specializations available during startup, before the first request, by warming the kernel's **compile-key space** without dummy runtime launches or real tensor allocation.

Use it when adding a warmable JIT kernel or migrating an existing warmup path.

## In This Guide

- [1. Quickstart](#1-quickstart): for contributors adding or migrating a warmable kernel.
- [2. Search-Space Reference](#2-search-space-reference): for contributors defining non-trivial compile-key spaces.
- [3. Maintainer Reference](#3-maintainer-reference): for backend integration, cache behavior, registry lifecycle, and tracer changes.

## 1. Quickstart

Each warmable kernel defines its compile-key mapping and compile-only entry point beside its normal runtime implementation. The startup registry then warms only the wrappers selected by the current engine configuration.

### Define the Kernel Wrapper

Here, a **kernel wrapper** (or just **wrapper**) is an instance of a concrete `VllmJitKernel` subclass.

Expose one wrapper near the kernel's normal runtime entry point. Prefer this shape:

```python
class MyKernel(VllmJitKernel["MyKernel.CompileKey"]):

    @dataclass(frozen=True)
    class CompileKey:
        ...

    @staticmethod
    def kernel(...):
        ...

    def dispatch(self, ...) -> CompileKey:
        return self.CompileKey(...)

    def get_warmup_keys(self, ...) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(...)

    def compile(self, compile_key: CompileKey) -> None:
        ...

    def __call__(self, ...):
        return self.kernel(...)


MY_KERNEL = MyKernel()
```

`CompileKey`, `dispatch(...)`, and `get_warmup_keys(...)` are backend-agnostic. Backend-specific behavior belongs in `kernel(...)`, `compile(...)`, and `__call__(...)`.

The module-level singleton should be used by warmup and by the runtime call path. This keeps dispatch behavior shared instead of duplicated.

### Choose Compile-Key Fields

`CompileKey` must be frozen and hashable. Include only fields on which the backend specializes, such as tile sizes, head dimensions, dtypes, pointer alignment classes, or backend selectors; exclude runtime-only values. When unsure, inspect the backend cache key, specialization arguments, or verbose JIT-monitor output.

### Generate Warmup Keys

Use `_trace_dispatch(self.dispatch)` to describe representative inputs. The tracer maps them through the same specialization logic and deduplicates equal keys:

```python
def get_warmup_keys(self, vllm_config: VllmConfig) -> list[CompileKey]:
    max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    return self._trace_dispatch(self.dispatch)(
        num_tokens=WarmupIntRange(1, max_tokens + 1),
    )
```

Use independent ranges or alternatives for cartesian products, `zip_inputs(...)` for coupled rows, and `_when` for validity constraints. The complete syntax is documented in [Search-Space Reference](#2-search-space-reference).

### Compile Without Launching

`compile(compile_key)` means "make this specialization available". Depending on the backend, that may compile from source, call a compile-only API, load an already-built artifact, or compile on cache miss.

`compile(...)` should not launch a real inference workload or allocate large real tensors. Each DSL should expose fake tensor/spec descriptors suitable for compilation only.

### Register the Selected Wrapper

Register the wrapper where the runtime implementation is selected:

```python
MY_KERNEL.register_warmup()
```

Registration records metadata only. It does not compile or launch the kernel. Repeated registrations from equivalent layers are allowed and deduplicated later.

### Review Checklist

- Warm actual compile keys rather than representative non-key inputs.
- Keep specialization mapping in `dispatch(...)` instead of duplicating it in warmup code.
- Use fake tensors or backend compile-only descriptors; never perform a dummy runtime launch.
- Keep registration metadata-only so model construction remains cheap and side-effect free.
- Compile under `kernel_warmup()` so feature gates, logging, ordering, and exception handling remain centralized.
- Keep runtime execution and startup compilation separate and easy to review.
- Use the module-level wrapper singleton from both warmup and runtime paths.

## 2. Search-Space Reference

### How Tracing Works

`_trace_dispatch(...)` expands the inputs declared by `get_warmup_keys(...)`. Each concrete combination becomes a `dispatch_values` mapping from input names to selected values. `_when` may reject that mapping; otherwise the tracer evaluates `dispatch(...)` to construct one `CompileKey`. Equal keys are deduplicated after all combinations are evaluated.

One call to `dispatch(...)` returns one key, but many input points may map to the same key. Prefer this traced mapping over manually reconstructing keys in warmup code; `dispatch(...)` should express the same specialization logic used by the runtime path.

### Define Input Spaces

Use ranges and alternatives for independent axes, `zip_inputs(...)` for coupled rows, and `_when` for validity constraints.

#### Integer Ranges

Use `WarmupIntRange` for integer ranges:

```python
return self._trace_dispatch(self.dispatch)(
    num_prefills=WarmupIntRange(1, max_prefills + 1),
)
```

`WarmupIntRange(start, stop, step)` follows Python `range(...)` semantics: `start` is inclusive, `stop` is exclusive, and `step` defaults to 1.

For non-linear integer sequences, use `advance` to provide the action that computes each next value:

```python
return self._trace_dispatch(self.dispatch)(
    num_tokens=WarmupIntRange(
        1,
        max_tokens + 1,
        advance=lambda value: next_power_of_2(value) + 1,
    ),
)
```

This is useful for traversing specialization boundaries without enumerating every integer. `advance` cannot be combined with a non-default `step`, and it must return a value greater than its input so expansion always makes forward progress.

#### Independent Alternatives

Use tuples or lists for independent alternatives. Multiple expanded inputs form a cartesian product:

```python
return self._trace_dispatch(self.dispatch)(
    query_slice_start=WarmupIntRange(0, 2),
    query_slice_stop=(1, 2 * max_tokens - 1, 2 * max_tokens),
    COMPRESS_RATIO=list(compress_ratios),
)
```

#### Coupled Inputs

Use `zip_inputs(...)` when values must vary together row-by-row:

```python
WARMUP_INPUTS = zip_inputs(
    dict(compress_ratio=1, topk=0, topk_width=512),
    dict(compress_ratio=4, topk=512, topk_width=512),
)


return self._trace_dispatch(self.dispatch)(
    WARMUP_INPUTS,
    WINDOW_SIZE=window_size,
)
```

Multiple `zip_inputs(...)` groups may be passed as positional arguments. The tracer forms the cartesian product across groups while preserving row-wise coupling inside each group.

Every row in a `zip_inputs(...)` group must use the same string keys. A `zip_inputs(...)` group cannot specify a field that is also specified as a keyword input to `_trace_dispatch(...)`.

#### Conditional Filtering

Use `_when=...` to filter generated input points before they are passed to `dispatch(...)`. This is useful when independent ranges contain invalid combinations, but the validity rule belongs with the kernel warmup definition.

```python
def _is_valid_warmup_input(
    self,
    *,
    query_len: int,
    num_reqs: int,
    max_num_batched_tokens: int,
) -> bool:
    return query_len + num_reqs - 1 <= max_num_batched_tokens


return self._trace_dispatch(self.dispatch)(
    query_len=WarmupIntRange(1, max_tokens + 1),
    num_reqs=WarmupIntRange(1, max_reqs + 1),
    max_num_batched_tokens=max_tokens,
    _when=self._is_valid_warmup_input,
)
```

`_when` accepts a function, bound method, or lambda and supports the same AST subset as `dispatch(...)`, including local assignments in function predicates.

The predicate is evaluated on the expanded warmup inputs. If it returns `False`, that input point is skipped and no `CompileKey` is produced for it.

### Write Dispatch Rules

#### Local Assignments

The traced body may contain local assignments, optionally annotated, followed by one `return self.CompileKey(...)` call. Local assignments let a kernel name intermediate specialization choices once and reuse them across fields:

```python
def dispatch(
    self,
    *,
    num_tokens: int,
    vectorized: bool,
) -> CompileKey:
    block_size = next_power_of_2(num_tokens)
    return self.CompileKey(
        BLOCK_SIZE=block_size,
        VECTOR_WIDTH=4 if vectorized and block_size >= 4 else 1,
    )
```

#### Supported Expressions

The evaluator supports these expressions inside local assignments and `CompileKey(...)` fields:

| Feature | What It Allows |
| --- | --- |
| Names | Read dispatch inputs, local assignments, defaults, and module globals. |
| Constants | Use literals such as integers, strings, booleans, and `None`. |
| Attributes | Read structured values such as `cfg.block_size` or `mla_dims.v_head_dim`. |
| Subscriptions | Read sequence positions or mapping values such as `config[0]` and `config["block_size"]`. |
| Tuple/list literals | Build shapes, strides, and other small structured fields. |
| Conditional expressions | Select a field with `x if condition else y`. |
| Boolean expressions | Combine predicates with `and`, `or`, and `not`. |
| Comparisons | Use `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in`, `is`, and `is not`. |
| Arithmetic | Use `+`, `-`, `*`, `//`, `%`, and `**`. |
| Unary minus | Build negative sentinel values or signed descriptors. |
| Helper calls | Call helpers with positional and explicit keyword arguments. |

Python builtins such as `min(...)`, `max(...)`, and `len(...)` are resolved unless the name is overridden locally or globally.

#### Helper Calls

Helpers are useful for small specialization rules:

```python
def dispatch(self, *, num_tokens: int, block_size: int) -> CompileKey:
    return self.CompileKey(
        PADDED_TOKENS=round_up(num_tokens, multiple=block_size),
    )
```

`_trace_dispatch(...)` does not inspect helper bodies. It evaluates the call arguments and invokes the helper as ordinary Python, so control flow inside that helper is outside the AST interpreter's scope. Keep helpers deterministic and side-effect free.

#### Direct Keyword Forwarding

For many direct pass-through fields, the dispatch `**kwargs` parameter may be unpacked into `CompileKey(...)`:

```python
def dispatch(
    self,
    *,
    num_tokens: int,
    **compile_key_fields: int,
) -> CompileKey:
    return self.CompileKey(
        **compile_key_fields,
        block_size=next_power_of_2(num_tokens),
    )
```

Unmatched dispatch arguments become compile-key fields and warmup inputs. Keep transformed inputs named and explicit. The unpacking must use the dispatch method's own `**kwargs` parameter directly and exactly once; arbitrary mappings, repeated unpacking, and helper-call `**kwargs` are rejected. The fully explicit form remains supported and is often clearer for non-trivial mappings.

#### Unsupported Syntax

Conditional expressions are supported, but statement-level `if` blocks directly inside `dispatch(...)` or `_when` are not. Loops, comprehensions, lambda expressions, mutation, slices, dict/set literals, tuple-unpacking assignments, multiple returns, star-argument calls, and backend imports are also unsupported in traced bodies. Put environment and model gating in `get_warmup_keys(...)` or the outer warmup entry point.

### Result Handling

#### Input Discovery

The tracer expands only inputs that affect the returned `CompileKey`:

```python
return self._trace_dispatch(self.dispatch)(
    num_tokens=WarmupIntRange(1, max_tokens + 1),
    unused_input=WarmupIntRange(0, 100),
)
```

Because `unused_input` is not referenced by `dispatch(...)`, it is ignored instead of adding an axis to the search space. Default dispatch arguments are honored when the corresponding warmup input is omitted.

#### Compile-Key Deduplication

`_trace_dispatch(...)` deduplicates the resulting keys while preserving order. This is important when many runtime-like inputs map to the same static bucket.

For example, this warmup range expands every token count, but the compile key only depends on the power-of-two bucket:

```python
def dispatch(
    self,
    *,
    num_tokens: int,
) -> CompileKey:
    return self.CompileKey(
        BLOCK_SIZE=next_power_of_2(num_tokens),
    )


def get_warmup_keys(self, vllm_config: VllmConfig) -> list[CompileKey]:
    max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    return self._trace_dispatch(self.dispatch)(
        num_tokens=WarmupIntRange(1, max_tokens + 1),
    )
```

For `max_tokens == 8`, the expanded inputs are `1, 2, 3, 4, 5, 6, 7, 8`, but the returned keys are:

```python
[
    CompileKey(BLOCK_SIZE=1),
    CompileKey(BLOCK_SIZE=2),
    CompileKey(BLOCK_SIZE=4),
    CompileKey(BLOCK_SIZE=8),
]
```

Deduplication happens after `dispatch(...)` is evaluated, so the warmup system removes duplicate compile keys, not duplicate input values. `CompileKey` must be hashable for this to work; using `@dataclass(frozen=True)` is the standard pattern.

## 3. Maintainer Reference

### Shared Wrapper Methods

`VllmJitKernel` provides the common lifecycle:

- `warmup(*args, **kwargs)` calls `get_warmup_keys(...)` and then `compile(compile_key)` for each key.
- `_trace_dispatch(dispatch)` expands warmup inputs, evaluates the traced dispatch logic, and returns deduplicated keys.
- `_get_or_compile(compile_key)` returns an executor cached by the wrapper, compiling through the monitored path on a miss.

### Backend Integration

#### Triton

Compile through Triton's compile-only warmup API with fake pointer descriptors and the static values from `CompileKey`. Runtime calls the native JIT entry point normally; Triton's cache handles hits, while `jit_monitor` reports unexpected runtime compilation.

#### CuTeDSL

Compile with fake tensors or symbolic shape descriptors and store the returned JIT Executor in the wrapper cache. Runtime derives the same key and calls `_get_or_compile(...)`; monitor mode determines whether a miss is rejected, warned and compiled, or silently compiled.

#### TileLang and Prebuilt Backends

TileLang uses its compile-only path and native runtime cache. A backend with prebuilt artifacts may implement `compile(compile_key)` as "load or otherwise ensure available" rather than compiling from source.

### Registry Lifecycle

The kernel contract defines which keys a wrapper needs. The per-runner `JitWarmupRegistry` records which wrappers were selected by the current engine configuration.

`JitWarmupRegistry.activate()` scopes registration during model and supporting-infrastructure construction. Registration outside that scope is a no-op. When `enable_jit_warmup` is enabled, `kernel_warmup()` expands the collected registrations: calls without arguments receive `vllm_config`, explicit arguments are forwarded unchanged, and repeated wrapper/key pairs are compiled once.

Register at the narrowest stable selection point. Shared components should register their wrappers directly rather than relying on a global model-name list. Registration must remain cheap: it records immutable metadata and must not compile, launch kernels, or retain large runtime tensors.

Calls without explicit registration arguments receive `vllm_config` when warmup runs. If `get_warmup_keys(...)` instead depends on finalized runtime metadata, pass the smallest immutable values needed to `register_warmup(...)`.

### Extending the Tracer

Keep the accepted AST subset deliberate. Adding an expression form requires evaluator support, input-discovery coverage, focused success and rejection tests, and an actionable error message.

Statement-level `if` is not another expression node: supporting it generally would require a restricted statement interpreter with defined behavior for branch-local assignments, nested branches, early returns, missing-return paths, and input discovery across every path. Prefer conditional expressions or ordinary helper calls until a concrete dispatch rule justifies that complexity.
