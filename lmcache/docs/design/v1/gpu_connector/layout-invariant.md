# GPU KV Cache Layout — Single Source of Truth

## Invariant

> **`normalize_kv_and_discover_format` is the only place that parses
> KV-cache layout.** It also returns the canonical (post-permute) form
> of the kv_caches alongside the format, so callers never need a
> separate normalization step. Every other module queries KV-cache
> information via helpers in `lmcache/v1/gpu_connector/utils.py` that
> accept an `EngineKVFormat` argument.

"Layout parsing" means: list-nesting depth, tensor-dimension ordering,
HND vs NHD, MLA vs MHA, per-layer vs cross-layer. All of that is
encoded in `EngineKVFormat`; downstream code must never re-derive it
from raw shapes.

## Canonical type

```python
DiscoverableKVCache = Union[torch.Tensor, list["DiscoverableKVCache"]]
```

Every KV-cache value in LMCache is one of these shapes:

- a single `torch.Tensor` (vLLM cross-layer, TRT-LLM),
- a flat `list[torch.Tensor]` (vLLM per-layer, SGLang MLA),
- a nested `list[list[torch.Tensor]]` (SGLang MHA's `[K_list, V_list]`).

Engine adapters that hand us other containers (vLLM's `dict[str, Tensor]`)
are responsible for unwrapping to this form before calling any helper.

## Package structure

The layout logic lives in `lmcache/v1/gpu_connector/kv_format/`. The
public `utils.py` helpers below are a thin **facade** that delegates
into it, so the single-source-of-truth surface is unchanged for callers.

```
kv_format/
├── types.py       # DiscoverableKVCache, LayoutHints (foundational types)
├── contiguity.py  # attempt_permute_to_contiguous_view (zero-copy view recovery)
├── detection.py   # detect_format() orchestration
├── specs/         # geometry layer
│   ├── base.py    #   KVFormatSpec ABC + shape_desc/concrete_shape rendering
│   ├── registry.py #  auto-discovers the spec files; get_spec/get_spec_class
│   └── <engine_kv_format>.py  #   one file per format
└── detectors/     # per-engine detection layer
    ├── base.py    #   EngineDetector ABC + measure_structure()
    ├── registry.py #  auto-discovers the detector files; get_detector
    └── <engine>.py            #   one file per engine (a single discover())
```

- **One spec class per format — pure geometry.** Each `EngineKVFormat`
  maps to exactly one `KVFormatSpec` subclass that knows how to index a
  value of that format. A spec describes *only* layout geometry; both the
  class and its file are named after the format member (e.g.
  `NB_NL_TWO_BS_NH_HS_Spec` in `nb_nl_two_bs_nh_hs.py`) — a geometry
  encoding, never an engine. `get_spec(kv, fmt)` returns an instance for
  geometry; `get_spec_class(fmt)` returns the class for static facts
  (`is_mla`, `is_hnd`, `is_cross_layer`, `attention_backends`).
- **Backend labels are diagnostic, colocated on the spec.** A single
  `EngineKVFormat` can be produced by several (engine, attention-backend)
  combinations, so each spec lists them in `attention_backends` (a tuple),
  first entry = canonical representative. `get_attention_backend(fmt)` (the
  `utils` facade) returns that first entry for logging. These labels are
  diagnostic only — they never drive geometry decisions, and they replace
  the old hand-maintained `EngineKVFormat → label` dict.
- **The enum is the single identity.** Each spec declares its
  `engine_kv_format` in its class body; the registry is derived from it.
  No separate string id, no engine attribute. The C++ `EngineKVFormat`
  enum is the one authority for which formats exist. Its **member name is
  the layout legend** — a `_`-joined token sequence where `X` marks a
  list-nesting boundary (`TWO_X_NL_X_NBBS_NH_HS` → `2 x NL x [PBS, NH, HS]`).
  The symbolic `shape_desc(fmt)` and numeric `concrete_shape(fmt, size)` are
  both *rendered* from that name (`specs/base.py`), so they cannot drift from
  the enum or from each other.
- **One file per spec; auto-discovered in one place.** Each spec lives in its
  own `specs/<engine_kv_format>.py` (named after the format it implements).
  `specs/registry.py` imports every file in the folder and indexes each spec by
  the `engine_kv_format` it declares, into the `SPECS` table that `get_spec` /
  `get_spec_class` look up. Adding a format = just drop a new file; the
  discovery is one readable loop in `registry.py` (no `__init_subclass__`, no
  registration scattered across files). There is no inheritance taxonomy:
  "family" base classes are avoided (formats vary on ≥5 orthogonal axes —
  engine, per-/cross-layer, MLA/MHA, NHD/HND, fused/separate PBS — which a
  single inheritance spine cannot model without orphans). Add structure only
  when a concrete need appears.
- **Detection is the one engine-aware layer.** `detect_format` does the
  engine-agnostic contiguous-view recovery, then dispatches to the
  `EngineDetector` for the `EngineType`. Each `detectors/<engine>.py` is one
  `discover(kv, hints)` that reshapes the engine's raw layout into canonical
  form *and* identifies the format in one step (returning `(format, kv)`);
  `detectors/registry.py` auto-discovers them the same way `specs/` does. The
  spec layer never sees `EngineType`. Adding an engine = just drop a new
  detector file.

## Adding a new format

1. Add the enum value in `csrc/kv_transfer_types.h` (the single
   backend-agnostic definition shared by every accelerator backend), then
   register it in each backend's pybind module — `csrc/pybind.cpp` (CUDA)
   and `csrc/sycl/pybind_sycl.cpp` (SYCL/XPU).
2. Add a branch in the engine's `detectors/<engine>.py` `discover()`. It keys
   off `(list_depth, tensor_ndim)` from `measure_list_depth_until_tensor`,
   returning `(format, kv)`; any reshape-via-hints (e.g. TRT-LLM's 4-D `view`'d
   to 6-D) happens in the same method before the shape checks.
3. Add a `KVFormatSpec` subclass as a new `specs/<engine_kv_format>.py` file
   (named after the format, declaring its `engine_kv_format`). `registry.py`
   discovers it automatically — no other file changes. The ABC makes the
   required accessors explicit (a spec missing one raises `TypeError` on first
   `get_spec`, which the golden test triggers).
4. Add a row to the golden table in
   `tests/v1/gpu_connector/test_kv_format_specs.py` and a detection
   case in `test_kv_format_detection.py`.

No other Python module should need edits. If you're editing
`kv_layer_groups.py`, `gpu_context.py`, or any `KVLayerGroupInfo`
consumer for a new layout — the branching belongs in a spec.

## Helper surface

Every helper below takes `DiscoverableKVCache` and (where layout matters)
an `EngineKVFormat`. Nothing else may index raw shapes.

### Discovery

| Helper | Returns |
|---|---|
| `normalize_kv_and_discover_format(kv_caches, engine, layout_hints)` | `tuple[EngineKVFormat, DiscoverableKVCache]` — the one parser. Returns the canonical (permuted-to-contiguous) kv_caches alongside the detected format; callers must use the returned tensor structure for subsequent operations. |

### Format → engine map

| `EngineKVFormat` | Engine | Layout | Structure |
|---|---|---|---|
| `NB_NL_TWO_BS_NH_HS` | vLLM cross-layer | NHD | bare 6-D tensor `[NB, NL, 2, BS, NH, HS]` |
| `NB_NL_TWO_NH_BS_HS` | TRT-LLM cross-layer | HND | bare 6-D tensor `[NB, NL, 2, NH, BS, HS]` |
| `NL_X_TWO_NB_BS_NH_HS` | vLLM flash-attn | NHD | `NL × [2, NB, BS, NH, HS]` |
| `NL_X_NB_TWO_BS_NH_HS` | vLLM flash-infer | NHD | `NL × [NB, 2, BS, NH, HS]` |
| `NL_X_TWO_NB_NH_BS_HS` | vLLM flash-attn | HND | `NL × [2, NB, NH, BS, HS]` |
| `NL_X_NB_TWO_NH_BS_HS` | vLLM flash-infer | HND | `NL × [NB, 2, NH, BS, HS]` |
| `NL_X_NB_BS_HS` | vLLM MLA | — | `NL × [NB, BS, HS]` |
| `TWO_X_NL_X_NBBS_NH_HS` | SGLang MHA | NHD | `[K_list, V_list]`, each `NL × [PBS, NH, HS]` |
| `TWO_X_NL_X_NB_BS_NH_HS` | SGLang MHA via MP daemon | NHD | `[K_list, V_list]`, each `NL × [NB, BS, NH, HS]` |
| `NL_X_NBBS_ONE_HS` | SGLang MLA | — | `NL × [PBS, 1, HS]` |
| `NL_X_NB_NH_BS_TWO_HS` | vLLM blocks-first fused (CPU) | HND | DEPRECATED (use `NL_X_NB_NH_BS_CS`): `NL × [NB, NH, BS, 2, HS]`, split from raw `[NB, NH, BS, 2·HS]` |
| `NL_X_NB_BS_NH_TWO_HS` | vLLM blocks-first fused | NHD | DEPRECATED (use `NL_X_NB_BS_NH_CS`): `NL × [NB, BS, NH, 2, HS]`, split from raw `[NB, BS, NH, 2·HS]` |
| `NL_X_NB_NH_BS_CS` | vLLM blocks-first fused (unified KV cache) | HND | `NL × [NB, NH, BS, CS]`, raw registration; CS = content size = 2·HS (K/V packed) |
| `NL_X_NB_BS_NH_CS` | vLLM blocks-first fused (unified KV cache) | NHD | `NL × [NB, BS, NH, CS]`, raw registration; CS = content size = 2·HS (K/V packed) |

The two cross-layer formats (`NB_NL_TWO_*`) share a single base
pointer, the kernel walks layers internally via `shape_desc.nl`. Use
`is_cross_layer_format(fmt)` for that dispatch and `is_hnd(fmt)` to
detect head-major within-block layouts.

### Reshape-via-hints (TRT-LLM)

TRT-LLM hands LMCache a 4-D pool tensor
`[NB, NL, 2, num_kv_heads * tokens_per_block * head_dim]` (HND, K and V
interleaved on dim 2). `normalize_kv_and_discover_format` reshapes it
to canonical 6-D form *before* the contiguity check, using
`layout_hints["num_kv_heads" | "tokens_per_block" | "head_dim"]`. The
function also collapses a 1-element list of a 6-D tensor down to the
bare 6-D tensor so detection lands on `list_depth == 0`. Adapters pass
either the 4-D bare tensor or `[4-D]`; the function handles both.

### Scalar accessors

All of these dispatch on `EngineKVFormat`. The ones that can vary per layer
take an optional `layer_idx: int = 0`; passing an explicit index enables
per-layer queries (for heterogeneous groups) without any intermediate
helper.

| Helper | Per-layer? | Notes |
|---|---|---|
| `get_num_layers(kv, fmt)` | no | Total layer count. |
| `get_num_blocks(kv, fmt)` | no | Paged block count (group-level). |
| `get_block_size(kv, fmt)` | no | Tokens per block. |
| `get_page_buffer_size(kv, fmt)` | no | |
| `get_tokens_per_layer(kv, fmt)` | no | |
| `get_elements_per_layer(kv, fmt)` | no | |
| `get_num_heads(kv, fmt, layer_idx=0)` | yes | |
| `get_head_size(kv, fmt, layer_idx=0)` | yes | |
| `get_hidden_dim_size(kv, fmt, layer_idx=0)` | yes | |
| `get_dtype(kv, fmt, layer_idx=0)` | yes | |
| `is_mla(fmt)`, `is_hnd(fmt)` | — | Format predicates. |
| `get_device(kv)` | — | Format-agnostic (descends to any leaf). |

### Pointer and descriptor builders

| Helper | Returns | Notes |
|---|---|---|
| `get_group_data_ptrs(kv, fmt, layer_indices)` | `list[int]` | Pointer array in **kernel-expected order**: `[base]` for cross-layer (`layer_indices` ignored), `[K_0…K_N, V_0…V_N]` for SGLang MHA, per-layer flat elsewhere. Matches the dispatch in `csrc/mp_mem_kernels.cu:161-169`. The pointer-array shape is a property of the format — callers never ask "does this format have per-layer pointers?". |
| `make_page_buffer_shape_desc(kv, fmt, layer_idx, num_layers_in_group, num_blocks, block_size, block_stride_elems)` | `PageBufferShapeDesc` | The kernel-facing shape struct. ``block_stride_elems`` carries the per-block dim-0 element stride; pass the value returned by `resolve_block_stride_and_log_layout` so groups with different physical block sizes (e.g. a compressed DeepSeek V4 indexer group alongside dense layers) share a single GPU pool. |

### Contiguity

| Helper | Returns | Notes |
|---|---|---|
| `attempt_permute_to_contiguous_view(kv)` | `DiscoverableKVCache` | Recursive, metadata-only. No-op if already contiguous; raises `ValueError` for non-permutation-recoverable cases (slicing, `as_strided`). **Never copies.** Walks the full structure and permutes every tensor leaf. Called internally by `normalize_kv_and_discover_format`; remains public only for callers that handle a tensor *outside* the discover flow (`GPUConnectorInterface.initialize_kvcaches_ptr`, `CudaIPCWrapper.__init__`). |

## Forbidden in consumer code

Raw-shape indexing belongs inside the `kv_format` layer (the spec classes);
consumer code must never do any of the following — it queries via the
`utils.py` facade instead:

- `isinstance(kv_cache, (tuple, list))` to distinguish layouts.
- Indexing raw shapes (`tensor.shape[3]`, `len(shape) == 5`) to derive
  dimensions.
- Hand-rolled list-depth probing (`while isinstance(x, list): depth +=
  1; x = x[0]`). There is no public depth helper and there shouldn't
  be one — `normalize_kv_and_discover_format` encapsulates the
  descent, and downstream code only ever needs the resulting
  `EngineKVFormat`.
- Wrapping a tensor with `[tensor]` to adapt to a helper's list-depth
  expectation — the accessors take `layer_idx` directly.
- Hand-rolled pointer assembly (`[t.data_ptr() for t in kv_caches]`) —
  use `get_group_data_ptrs`.
- Hand-rolled device discovery (`kv_caches[0][0].device`) — use
  `get_device`.
- Hand-rolled contiguity fixes (`tensor.contiguous()`, `.clone()`) —
  use `attempt_permute_to_contiguous_view` which refuses to copy.
- "Canonicalize" functions that rewrite `kv_caches` to a uniform shape
  before passing to helpers. The helpers already canonicalize by
  accepting `EngineKVFormat`, and any reshape/normalize step that *is*
  needed lives inside `normalize_kv_and_discover_format` — callers
  receive the canonical form back from that one call.

## Consumers

- **`lmcache/v1/kv_layer_groups.py::KVLayerGroupsManager.__init__`** —
  partitions layers by the 5-tuple `(kv_size, num_heads, head_size,
  block_size, dtype)` using `is_mla`, `get_num_heads`, `get_head_size`,
  `get_block_size`, and `get_dtype` with each layer's index. Including
  `block_size` in the identity lets compressed groups (e.g. a DeepSeek
  V4 indexer with a smaller physical slot count) sit alongside
  non-compressed groups under a single `GPUCacheContext`. Builds a
  `PageBufferShapeDesc` per group via `make_page_buffer_shape_desc`,
  passing the `block_stride_elems` resolved by
  `resolve_block_stride_and_log_layout`. The real constructor is the
  only way in — no test-only shortcuts, no cached topology fields; the
  manager exposes only `kv_layer_groups`, `num_groups`, and
  `get_shape_desc`.
- **`lmcache/v1/platform/cuda/cache_context.py::GPUCacheContext`** —
  constructs the manager directly at init, delegates
  `get_shape_desc(group_idx)` to it, assembles per-group GPU pointer
  tensors via `get_group_data_ptrs`. No parallel `shape_descs_` /
  `hidden_dim_sizes_` state.
- **`lmcache/v1/gpu_connector/gpu_connectors.py::VLLMPagedMemGPUConnectorV3._initialize_kv_cache_pointers`**
  — for the in-process vLLM path, calls
  `normalize_kv_and_discover_format` (which permutes for HND support
  and detects the format in one step) and constructs
  `metadata.kv_layer_groups_manager` lazily on first store/retrieve.
  The adapter (`vllm_v1_adapter.py`) does not participate in format
  discovery — it only stores `self.kv_caches` at register time.

Only `normalize_kv_and_discover_format` consumes `layout_hints`.
`attempt_permute_to_contiguous_view` (called internally) infers the
permutation from strides and needs no hints.

## Implementation note: mypy and the recursive union

Format-dispatched raw indexing on `DiscoverableKVCache`
(`kv_caches.shape[i]`, `kv_caches[0][j]`) is concentrated in the
`kv_format` layer: each `specs/<format>.py` sets
`# mypy: disable-error-code="union-attr,call-overload"` at the file level
(the `detectors/` and `contiguity.py` files use `union-attr` alone, and the
`utils.py` facade keeps the directive for its remaining structural helpers). The
`engine_kv_format` — or, in a spec, the class identity itself — is the
proof the indexing is well-defined, but mypy can't carry that proof
through a recursive Union without per-line casts. The file-level directive
replaces scattered `# type: ignore` comments; all other type checks remain
live.
