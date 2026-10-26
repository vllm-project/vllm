# KV cache axis metadata at connector registration

KV connectors can override `register_kv_caches_with_layouts(kv_caches, layouts)`
to receive producer-declared axis semantics alongside the actual tensors.
The default implementation validates supplied layouts and delegates to the
existing `register_kv_caches(kv_caches)` hook. Existing connectors do not need
to change their method signatures. `MultiConnector` forwards the descriptors
to each child.

## Descriptor contract

`KVCacheTensorLayout.axes` is a tuple of `KVCacheAxis` enum members in **tensor
indexing order**. `KVCacheLayout` continues to
describe the allocator's physical stride permutation. Consumers must use the
actual tensor's shape, strides, dtype, and storage offset for addressing.

Both GPU model runners pass descriptors produced when the allocator creates
logical `[block, head, token/state, content]` views, including views with
padded blocks or layers sharing an allocation. To collect them, pass a fresh
`tensor_layouts` dictionary to `allocate_kv_cache`; the allocator populates it
through `create_kv_cache_views`, alongside the returned tensors. Existing
callers can omit this keyword and keep the tensor-only return value.
HiSparse collects host-view descriptors through the same view creator.
Shared layers inherit their target view's descriptor.

Hybrid groups may have different shapes, dtypes, state granularity and content
encodings while sharing these axis roles. Layer names identify per-layer views;
the actual tensor's strides and storage offset describe cross-layer packing.
Descriptors describe the views registered with the connector, before any
mapping to attention-private kernel-block views. `CONTENT` is opaque: the
layer's `KVCacheSpec` describes its encoding and packing. Axis metadata alone
does not identify quantization scales or split packed K and V.

`STATE` counts positions in the view. The cache spec supplies their token
granularity for compressed or recurrent caches. An explicit `KV` axis has
size two, with K at index zero and V at index one. `SINGLETON` axes have size
one and may occur more than once.

The mapping uses the same layer names as `kv_caches`. A missing entry means
unspecified semantics, not a default layout. Raw storage without semantic
layer views is omitted. A connector that requires explicit semantics must
reject such entries or use its existing spec-aware registration path.

### MLA views without a K/V selector

`KV` specifically selects between separately stored K and V; it is not a count
of cache tensors or latent vectors. MLA stores a shared latent representation
without a separate V allocation, so its descriptor does not need a `KV` axis.
For example, a standard MLA view with shape `[blocks, 1, states, content]`
still declares `(BLOCK, HEAD, STATE, CONTENT)`: the size-one dimension is a
head slot. The spec defines the latent content's encoding.

Custom MLA views must describe their own indexing order. A
`[blocks, states, content]` view declares `(BLOCK, STATE, CONTENT)`; a view
with an extra placeholder dimension `[1, blocks, states, content]` declares
`(SINGLETON, BLOCK, STATE, CONTENT)`. A size-one dimension that has a semantic
role, such as `HEAD`, keeps that role rather than becoming `SINGLETON`.

`validate_shape` accepts these views without a `KV` axis. Declaring a size-one
axis as `KV` is rejected: it would not say whether the entry represents K, V
or a shared latent. Component roles and content encoding require spec-aware
handling rather than relaxing the two-element `KV` contract.

## Custom allocators

An out-of-tree backend or model runner should construct descriptors where it
chooses the registered tensor's layout. It must describe the exact view
passed to the connector, rather than a later backend-private view.

For example, two RBLN-style layouts can have the identical shape
`[2, 2, heads, 1, tokens, head_dim]`. Shape or contiguous strides cannot
distinguish their block and K/V axes. Their declarations are different:

```python
from vllm.v1.kv_cache_layout import KVCacheAxis as Axis
from vllm.v1.kv_cache_layout import KVCacheTensorLayout

tail = (Axis.HEAD, Axis.SINGLETON, Axis.STATE, Axis.HEAD_DIM)
blocks_first = KVCacheTensorLayout((Axis.BLOCK, Axis.KV, *tail))
kv_first = KVCacheTensorLayout((Axis.KV, Axis.BLOCK, *tail))

connector.register_kv_caches_with_layouts(
    {"layer": cache}, {"layer": blocks_first}
)
```

The consumer can call `layout.axis_index(Axis.BLOCK)` to locate the block
dimension and `layout.validate_shape(cache.shape)` to check the rank and
fixed-size axes. Overriding connectors should validate descriptors before
using them and reject inconsistent metadata rather than guess a different
layout. They can map the descriptor to an internal transfer format once
at registration, without importing a cache library's format enum into vLLM.

Any connector-side permutation, squeeze, reshape or component split must
transform the descriptor along with its tensor. After a cache reallocation,
the producer must register the new views with matching descriptors again.

This hook does not itself update external connectors or hardware plugins.
They must opt in to produce and consume the metadata. The descriptor is an
in-process Python API; an external connector owns its transport schema.
Tuple and multi-plane registrations need a matching component description
in a future extension. An axis declaration alone does not describe their
component roles or encoding.
