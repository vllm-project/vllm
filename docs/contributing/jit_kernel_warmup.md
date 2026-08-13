# JIT Kernel Warmup

vLLM uses JIT-generated kernels from several backends, including Triton, CuTeDSL, TileLang, and backend-specific libraries. These kernels often specialize on static values such as tile sizes, head dimensions, dtypes, pointer alignment, or backend selector choices.

JIT warmup makes those specializations available during engine startup, before the first real request. The shared contract is designed around the kernel's **compile-key space**, not around representative non-key inputs. It also provides a compile-only warmup path, avoiding dummy runtime launches and real tensor allocation.

Use this contract when adding a new warmable JIT kernel or migrating an existing warmup path.

## Design Goals

A warmup implementation should:

- keep warmup logic close to the kernel that owns the specialization rules;
- warm actual compile keys instead of hoping representative runtime inputs map to every needed specialization;
- avoid dummy runtime launches and real tensor allocation;
- run under the standard `kernel_warmup()` path, including logging, ordering, feature gates, and exception handling;
- keep model construction cheap and side-effect free;
- keep runtime execution and startup compilation easy to review separately.

## Kernel Contract

Here, a **kernel wrapper** (or just **wrapper**) is an instance of a concrete `VllmJitKernel` subclass.

Each warmable kernel should expose a wrapper object near the kernel's normal runtime entry point. The backend-agnostic pieces are `CompileKey`, `dispatch(...)`, and `get_warmup_keys(...)`. Backend-specific details belong inside `kernel(...)`, `compile(...)`, and the runtime `__call__(...)` wrapper.

Prefer this shape:

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

The module-level singleton should be used by warmup and by the runtime call path. This keeps dispatch behavior shared instead of duplicated.

### Shared Methods

`VllmJitKernel` provides common mechanics for all warmable kernels:

- `warmup(*args, **kwargs)` calls `get_warmup_keys(...)` and then `compile(compile_key)` for each returned key.
- `_trace_dispatch(dispatch)` expands a warmup input space, evaluates `dispatch(...)` through the AST tracer, and returns deduplicated `CompileKey` objects.
- `compile_key(kwargs)` builds one `CompileKey` from one concrete dispatch input dictionary.
- `_get_or_compile(compile_key)` returns an executor cached by the kernel wrapper. On a miss, it invokes the wrapper's monitored `compile(...)` path and then returns the executor populated by that method.

Runtime miss handling follows the backend's cache model:

- Triton and TileLang call their native JIT entry points normally. Their native cache handles hits, and `jit_monitor` reports unexpected runtime compilation.
- CuTeDSL compile-only warmup stores the returned JIT Executor in the kernel wrapper's cache. Runtime derives the same key and calls `_get_or_compile(...)`; monitor mode determines whether a miss is rejected, warned and compiled, or silently compiled.

### Kernel Activation

The kernel contract defines which compile keys a kernel wrapper needs. The per-runner `JitWarmupRegistry` separately records which wrappers were actually selected by the current engine configuration.

Register a kernel wrapper from the component or backend construction path where its runtime implementation is selected:

```python
MY_KERNEL.register_warmup()
```

`JitWarmupRegistry.activate()` scopes collection during model and supporting infrastructure setup. Registration outside that scope is a no-op. When `enable_jit_warmup` is enabled, `kernel_warmup()` expands collected registrations: calls without arguments receive `vllm_config`, explicit arguments are forwarded unchanged, and repeated wrapper/key pairs are compiled once. Registration itself only records metadata; it never compiles or launches.

Register at the narrowest stable selection point. Shared components should register their own wrappers rather than relying on a global model-name list. Repeated registration from equivalent layers is allowed and is deduplicated by the registry.

If a kernel wrapper uses a nonstandard `get_warmup_keys(...)` signature, pass those arguments explicitly:

```python
MY_KERNEL.register_warmup(
    shapes=((hidden_size, num_experts),),
    m_values=range(1, 17),
)
```

### Compile Key

`CompileKey` is a frozen dataclass that identifies one compiled specialization. It must be hashable so warmup can deduplicate keys.

Include fields that the backend actually specializes on. Avoid fields that are only runtime values. When unsure, inspect the backend JIT cache key, specialization arguments, or the JIT monitor in verbose mode (`--jit-monitor-verbose`) to find uncovered compile keys:

```text
Triton kernel JIT compilation during inference: _compute_slot_mapping_kernel (
constexprs={BLOCK_SIZE=1024, CP_KV_CACHE_INTERLEAVE_SIZE=1, PAD_ID=-1, TOTAL_CP_RANK=0, TOTAL_CP_WORLD_SIZE=1};
...)
```

### Compile Method

`compile(compile_key)` means "make this specialization available". Depending on the backend, that may compile from source, call a compile-only API, load an already-built artifact, or compile on cache miss.

`compile(...)` should not launch a real inference workload or allocate large real tensors. Each DSL should expose fake tensor/spec descriptors suitable for compilation only.

## AST-Traced Dispatch

The warmup system uses Python AST to trace `dispatch(...)`. One call to `dispatch(...)` returns one `CompileKey`, but many input points may map to the same key. It should express the same specialization logic used by the runtime path.

### Dispatch Body

The traced function body may contain:

- local assignments, optionally annotated;
- one `return self.CompileKey(...)` call.

Local assignments let a kernel name intermediate specialization choices once and reuse them across fields:

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

Conditional expressions (`x if condition else y`) are supported, but statement-level `if` blocks are not supported directly inside traced `dispatch(...)` or `_when` bodies. The tracer expects a straight-line sequence of local assignments followed by one return expression. Small, pure helpers called by traced expressions execute as normal Python with concrete values and may use ordinary control flow, including `if` blocks. Do not put loops, mutation, side effects, or backend imports directly inside traced functions. Put environment and model gating in `get_warmup_keys(...)` or the outer warmup entry point.

### Expression Features

The AST evaluator supports the following expression features inside local assignments and `CompileKey(...)` fields:

| Feature | What It Allows |
| --- | --- |
| Names | Read dispatch inputs, local assignments, defaults, and module globals. |
| Constants | Use literals such as integers, strings, booleans, and `None`. |
| Attributes | Read structured config values such as `cfg.block_size` or `mla_dims.v_head_dim`. |
| Subscriptions | Read tuple/list positions or mapping values such as `config[0]` and `config["block_size"]`. |
| Tuple/list literals | Build structured compile-key fields such as shapes, strides, and small descriptors. |
| Conditional expressions | Select fields with `x if condition else y` without statement-level branching. |
| Boolean expressions | Combine predicates with `and`, `or`, and `not`. |
| Comparisons | Use `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in`, `is`, and `is not`. |
| Arithmetic | Use `+`, `-`, `*`, `//`, `%`, and `**` for bucket and tile calculations. |
| Unary minus | Build negative sentinel values or signed descriptors. |
| Helper calls | Call small helper functions with positional arguments and explicit keyword arguments. |

Helper calls are useful for small, pure specialization helpers:

```python
def dispatch(self, *, num_tokens: int, block_size: int) -> CompileKey:
    return self.CompileKey(
        PADDED_TOKENS=round_up(num_tokens, multiple=block_size),
    )
```

The tracer supports Python builtins such as `min(...)`, `max(...)`, and `len(...)`, unless that name is overridden locally or globally.

For dispatch methods with many direct pass-through fields, the dispatch `**kwargs` parameter may be unpacked directly into `CompileKey(...)`:

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

All unmatched dispatch arguments are then compile-key fields and warmup inputs. Named parameters remain explicit when dispatch transforms them. The unpacking must use the dispatch method's `**kwargs` parameter directly and may appear only once. The fully explicit form remains supported and is clearer for dispatch methods with non-trivial field mappings. Unpacking another mapping or expression, or unpacking the parameter more than once, is rejected. Helper calls also cannot use `**kwargs`.

Unsupported constructs directly inside traced function bodies currently include loops, statement-level `if`, comprehensions, lambda expressions, mutation, slices, dict/set literals, and star-argument calls. If a dispatch rule needs ordinary control flow, move that logic into a small, pure helper function and call it from a supported expression.

### Input Discovery

The tracer only expands inputs that affect the returned `CompileKey`.

```python
return self._trace_dispatch(self.dispatch)(
    num_tokens=WarmupIntRange(1, max_tokens + 1),
    unused_input=WarmupIntRange(0, 100),
)
```

Here, `unused_input` is an arbitrary example; the name has no special meaning. Because it is not referenced by `dispatch(...)`, it is ignored. This lets dispatch accept runtime context that does not affect compilation without adding unnecessary axes to the warmup search space.

Default dispatch arguments are honored. If a field depends on a parameter with a default and `get_warmup_keys(...)` does not pass that parameter, the default is used when building the key.

## Warmup Input Expansion

`get_warmup_keys(...)` returns the representative compile keys needed for a given vLLM configuration. Prefer deriving keys through `_trace_dispatch(...)` instead of manually reconstructing the compile key. `_trace_dispatch(...)` expands only arguments used by `dispatch(...)`; unused warmup inputs are ignored.

### Integer Ranges

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

### Independent Alternatives

Use tuples or lists for independent alternatives. Multiple expanded inputs form a cartesian product:

```python
return self._trace_dispatch(self.dispatch)(
    query_slice_start=WarmupIntRange(0, 2),
    query_slice_stop=(1, 2 * max_tokens - 1, 2 * max_tokens),
    COMPRESS_RATIO=list(compress_ratios),
)
```

### Coupled Inputs

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

### Conditional Filtering

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

### Key Deduplication

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
