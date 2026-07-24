# JIT Kernel Warmup

vLLM uses JIT-generated kernels from several backends, including Triton,
CuTeDSL, TileLang, and backend-specific libraries. These kernels often
specialize on static values such as tile sizes, head dimensions, dtypes,
pointer alignment, or backend selector choices.

JIT warmup makes those specializations available during engine startup, before
the first real request. The shared contract is designed around the kernel's
**compile-key space**, not around representative non-key inputs. It also
provides a compile-only warmup path, avoiding dummy runtime launches and real
tensor allocation.

Use this contract when adding a new warmable JIT kernel or migrating an
existing warmup path.

## Design Goals

A warmup implementation should:

- keep warmup logic close to the kernel that owns the specialization rules;
- warm actual compile keys instead of hoping representative runtime inputs map
  to every needed specialization;
- avoid dummy runtime launches and real tensor allocation;
- run under the standard `kernel_warmup()` path, including logging, ordering,
  feature gates, and exception handling;
- keep model construction cheap and side-effect free;
- keep runtime execution and startup compilation easy to review separately.

## Kernel Contract

Each warmable kernel should expose a wrapper object near the kernel's normal
runtime entry point. The backend-agnostic pieces are `CompileKey`,
`dispatch(...)`, and `get_warmup_keys(...)`. Backend-specific details belong
inside `kernel(...)`, `compile(...)`, and the runtime `__call__(...)` wrapper.

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

The module-level singleton should be used by warmup and by the runtime call
path. This keeps dispatch behavior shared instead of duplicated.

### Shared Methods

`VllmJitKernel` provides common mechanics for all warmable kernels:

- `warmup(*args, **kwargs)` calls `get_warmup_keys(...)` and then
  `compile(compile_key)` for each returned key.
- `_trace_dispatch(dispatch)` expands a warmup input space, evaluates
  `dispatch(...)` through the AST tracer, and returns deduplicated
  `CompileKey` objects.
- `compile_key(kwargs)` builds one `CompileKey` from one concrete dispatch input
  dictionary.

### Compile Key

`CompileKey` is a frozen dataclass that identifies one compiled specialization.
It must be hashable so warmup can deduplicate keys.

Include fields that the backend actually specializes on. Avoid fields that are
only runtime values. When unsure, inspect the backend JIT cache key,
specialization arguments, or the JIT monitor in verbose mode
(`--jit-monitor-verbose`) to find uncovered compile keys:

```text
Triton kernel JIT compilation during inference: _compute_slot_mapping_kernel (
constexprs={BLOCK_SIZE=1024, CP_KV_CACHE_INTERLEAVE_SIZE=1, PAD_ID=-1, TOTAL_CP_RANK=0, TOTAL_CP_WORLD_SIZE=1};
...)
```

### Compile Method

`compile(compile_key)` means "make this specialization available". Depending on
the backend, that may compile from source, call a compile-only API, load an
already-built artifact, or compile on cache miss.

`compile(...)` should not launch a real inference workload or allocate large
real tensors. Each DSL should expose fake tensor/spec descriptors suitable for
compilation only.

## AST-Traced Dispatch

The warmup system uses Python AST to trace `dispatch(...)`. One call to
`dispatch(...)` returns one `CompileKey`, but many input points may map to the
same key. It should express the same specialization logic used by the runtime
path.

### Dispatch Body

The traced function body may contain:

- local assignments, optionally annotated;
- one `return self.CompileKey(...)` call.

Local assignments let a kernel name intermediate specialization choices once
and reuse them across fields:

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

Do not put loops, statement-level `if` blocks, mutation, side effects, or
backend imports inside `dispatch(...)`. Put environment and model gating in
`get_warmup_keys(...)` or the outer warmup entry point.

### Expression Features

The AST evaluator supports the following expression features inside local
assignments and `CompileKey(...)` fields:

| Feature | What It Allows |
| --- | --- |
| Names | Read dispatch inputs, local assignments, defaults, and module globals. |
| Constants | Use literals such as integers, strings, booleans, and `None`. |
| Attributes | Read structured config values such as `cfg.block_size` or `mla_dims.v_head_dim`. |
| Tuple/list literals | Build structured compile-key fields such as shapes, strides, and small descriptors. |
| Conditional expressions | Select fields with `x if condition else y` without statement-level branching. |
| Boolean expressions | Combine predicates with `and`, `or`, and `not`. |
| Comparisons | Use `==`, `!=`, `<`, `<=`, `>`, and `>=`. |
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

Helper calls cannot use `**kwargs`, and `CompileKey(...)` cannot be constructed
from `**kwargs`. This keeps traced fields explicit.

Unsupported constructs currently include loops, statement-level `if`,
comprehensions, lambda expressions, mutation, subscripting, dict/set literals,
`in`/`not in` comparisons, and star-argument calls. If a dispatch rule needs
those features, move that logic into a small helper function and call it from a
supported expression.

### Input Discovery

The tracer only expands inputs that affect the returned `CompileKey`.

```python
return self._trace_dispatch(self.dispatch)(
    num_tokens=WarmupIntRange(1, max_tokens + 1),
    debug_probe=WarmupIntRange(0, 100),
)
```

If `debug_probe` is not referenced by `dispatch(...)`, it is ignored. This
allows callers to pass broad context without accidentally multiplying the warmup
space.

Default dispatch arguments are honored. If a field depends on a parameter with a
default and `get_warmup_keys(...)` does not pass that parameter, the default is
used when building the key.

## Warmup Input Expansion

`get_warmup_keys(...)` returns the representative compile keys needed for a
given vLLM configuration. Prefer deriving keys through `_trace_dispatch(...)`
instead of manually reconstructing the compile key. `_trace_dispatch(...)`
expands only arguments used by `dispatch(...)`; unused warmup inputs are
ignored.

### Integer Ranges

Use `WarmupIntRange` for integer ranges:

```python
return self._trace_dispatch(self.dispatch)(
    num_prefills=WarmupIntRange(1, max_prefills + 1),
)
```

`WarmupIntRange(start, stop, step)` follows Python `range(...)` semantics:
`start` is inclusive, `stop` is exclusive, and `step` defaults to 1.

### Independent Alternatives

Use tuples or lists for independent alternatives. Multiple expanded inputs form
a cartesian product:

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

Multiple `zip_inputs(...)` groups may be passed as positional arguments. The
tracer forms the cartesian product across groups while preserving row-wise
coupling inside each group.

Every row in a `zip_inputs(...)` group must use the same string keys. A
`zip_inputs(...)` group cannot specify a field that is also specified as a
keyword input to `_trace_dispatch(...)`.

### Conditional Filtering

Use `_when=...` to filter generated input points before they are passed to
`dispatch(...)`. This is useful when independent ranges contain invalid
combinations, but the validity rule belongs with the kernel warmup definition.

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

The predicate is evaluated on the expanded warmup inputs. If it returns
`False`, that input point is skipped and no `CompileKey` is produced for it.

### Key Deduplication

`_trace_dispatch(...)` deduplicates the resulting keys while preserving order.
This is important when many runtime-like inputs map to the same static bucket.

For example, this warmup range expands every token count, but the compile key
only depends on the power-of-two bucket:

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

For `max_tokens == 8`, the expanded inputs are `1, 2, 3, 4, 5, 6, 7, 8`,
but the returned keys are:

```python
[
    CompileKey(BLOCK_SIZE=1),
    CompileKey(BLOCK_SIZE=2),
    CompileKey(BLOCK_SIZE=4),
    CompileKey(BLOCK_SIZE=8),
]
```

Deduplication happens after `dispatch(...)` is evaluated, so the warmup system
removes duplicate compile keys, not duplicate input values. `CompileKey` must be
hashable for this to work; using `@dataclass(frozen=True)` is the standard
pattern.
