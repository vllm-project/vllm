# `lmcache.v1.distributed.serde` — Serialization / Deserialization Package

## Scope

This package holds the **generic serde primitives** used by the
distributed storage stack. It does not know anything about L2 adapters
or controllers — it just defines:

- A pair of **sync interfaces** users implement (`Serializer`,
  `Deserializer`) for the actual byte transform.
- An **async interface** (`SerdeProcessor`) with the same
  `submit → eventfd → query` shape as an L2 adapter, so downstream
  consumers can poll it uniformly.
- A default implementation (`AsyncSerdeProcessor`) that turns any pair
  of sync implementations into the async interface by running them in
  a thread pool and signaling an eventfd on completion.
- A factory / registration mechanism so adapters can reference a serde
  by name (`{"type": "fp8", ...}` in JSON config).
- One built-in serde: **fp8 quantization**.

How the async interface is actually plugged into the L2 path lives in
[`docs/design/v1/distributed/l2_adapters/serde_wrapper.md`][wrapper-doc]
— the wrapper is the sole consumer of `SerdeProcessor`'s event fds.

[wrapper-doc]: ../l2_adapters/serde_wrapper.md

## Module Layout

```
lmcache/v1/distributed/serde/
  base.py             # Serializer, Deserializer (sync ABCs)
                      # SerdeProcessor (async ABC)
                      # SerdeConfig, SerdeTaskId
  async_processor.py  # AsyncSerdeProcessor (thread-pool + eventfd wrapper)
  factory.py          # register_serde_factory / create_serde_processor
  fp8.py              # Fp8QuantizationSerializer / Deserializer
  aesgcm.py           # AES-GCM encryption serde (see aesgcm.md)
  key_provider.py     # KeyProvider / HkdfKeyProvider for aesgcm
  multi.py            # MultiSerializer / MultiDeserializer (tuple-shaped
                      # extension; see "Multi-output extension" below)
  utils.py            # serialized_layout_desc, make_temp_key
```

## Two-Layer Interface

```
                 ┌────────────────────────────────┐
 user writes →   │  Serializer / Deserializer     │  sync transform
                 │  (pure: src MemoryObj → dst)   │
                 └──────────────┬─────────────────┘
                                │ wrapped by AsyncSerdeProcessor
                                ▼
                 ┌────────────────────────────────┐
 wrapper uses →  │  SerdeProcessor                │  async:
                 │    submit_serialize(...) → id  │   submit / eventfd /
                 │    query_serialize_result(id)  │   query
                 │    get_serialize_event_fd()    │
                 │    (plus deserialize pair)     │
                 └────────────────────────────────┘
```

- **Sync layer** is where the user cares. Pure Python (or torch) code,
  no threads, no fds. Two abstract methods: `serialize(src, dst, key)`
  and `estimate_serialized_size(layout_desc)`.
- **Async layer** is what the `SerdeL2AdapterWrapper` talks to. It
  owns two eventfds (one for serialize, one for deserialize) that
  must be distinct, and queues completed tasks in a dict the wrapper
  drains.

`AsyncSerdeProcessor` is the default and typically only implementation
of the async layer — most custom serdes only need to provide the sync
classes and register a factory.

## Contracts

### `Serializer.serialize(src, dst, key) -> int`

- `src` is a `MemoryObj` holding KV data (read-locked by the caller).
- `dst` is a `MemoryObj` byte buffer (write-locked by the caller),
  sized ≥ `estimate_serialized_size(layout_of_src)`.
- `key` is the `ObjectKey` for this `src`/`dst` pair, positionally
  aligned with the batch. Serdes that transform bytes per-object
  (e.g. an encryption serde keyed on `cache_salt`) read it;
  transform-agnostic serdes (fp8, turboquant) ignore it.
- Must return the number of bytes actually written to `dst`.
- Must be **deterministic** given the same `src` — the wrapper relies
  on the serialize step being reproducible across retries.
- **Writing raw bytes into `dst`:** go through `dst.byte_array` and cast
  it to the native format first — `memoryview(dst.byte_array).cast("B")`.
  `byte_array` is a ctypes-backed view with format `"<B"`, and CPython
  does not support slice assignment into a non-native format (a bare
  `dst[i:j] = ...` raises `NotImplementedError: memoryview: unsupported
  format <B`). The `aesgcm` serde takes this path; tensor-based serdes
  (fp8, turboquant) instead write through `dst.tensor` and never touch
  `byte_array`.

### `Serializer.estimate_serialized_size(layout_desc) -> int`

- Called once per batch to size the temp buffer **before** any work.
- Must be an **upper bound** on the actual serialized output. Include
  any safety margin inside this method (the fp8 serializer returns
  `1.5 × num_elements` for exactly this reason).
- Must only depend on `layout_desc` (shapes + dtypes). The wrapper
  uses the first object's layout to size temps for the whole batch,
  so a data-dependent estimate would break all-or-nothing allocation.

### `Deserializer.deserialize(src, dst, key) -> None`

- `src` is a byte-buffer MemoryObj filled by L2 load.
- `dst` is a KV-shaped MemoryObj (write-locked), already the correct
  shape and dtype.
- `key` is the `ObjectKey` for this `src`/`dst` pair (see
  `serialize` above for the per-object rationale).
- No return value — the caller observes completion via the async
  layer's event fd.
- Writing raw bytes into `dst` follows the same rule as `serialize`:
  cast via `memoryview(dst.byte_array).cast("B")`.

### `SerdeProcessor` (async)

- `submit_serialize(src_objs, dst_objs, keys) → SerdeTaskId` must be
  non-blocking. `keys` is positionally aligned with `src_objs` /
  `dst_objs` (the wrapper always supplies them). The actual transform
  runs asynchronously.
- `query_serialize_result(task_id) → bool | None` is
  **non-idempotent**: it returns a non-None value exactly once per
  task id. `None` means the task is still in flight.
- `get_serialize_event_fd()` returns an eventfd signaled once per
  completed serialize task; distinct from the deserialize fd.
- The deserialize side has the identical shape.
- `close()` must release both event fds and any worker threads.

## `AsyncSerdeProcessor`

Wraps any `(Serializer, Deserializer)` pair. Internal design:

- One `ThreadPoolExecutor(max_workers=N)` runs both serialize and
  deserialize tasks; they're independent, so the pool is shared.
- One task-id counter under a single lock; two `_completed_*` dicts
  (one per direction) protected by the same lock.
- Two `os.eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)` file descriptors —
  one per direction, signaled after the result is written to the
  completion dict.

**Why a pool, not an asyncio executor:** the CPU-bound fp8 / encryption
transforms release the GIL under torch / native calls, so a real
thread pool is useful. `N=1` is a safe default (one in-flight
transform at a time), and the fp8 factory accepts `max_workers` to
bump it.

## Factory / Registration

```python
from lmcache.v1.distributed.serde import (
    AsyncSerdeProcessor, Deserializer, Serializer,
    register_serde_factory,
)

def _create_my_serde(kwargs: dict[str, object]) -> SerdeProcessor:
    return AsyncSerdeProcessor(MySerializer(...), MyDeserializer(...))

register_serde_factory("mine", _create_my_serde)
```

The factory receives the type-specific kwargs from the JSON config
(everything except `"type"`). The registry is process-global and
rejects duplicate names, matching the pattern already used by
`register_l2_adapter_type`.

The factory is called exactly once per `SerdeL2AdapterWrapper`
construction — each wrapped adapter gets its own `SerdeProcessor`
instance.

## Built-in fp8

`Fp8QuantizationSerializer` / `Fp8QuantizationDeserializer`:

- Cast each element to `torch.float8_e4m3fn` (default) or
  `torch.float8_e5m2`, reinterpret the bytes as `uint8`, and copy into
  the temp buffer.
- Deserialize reinterprets the `uint8` bytes as the chosen fp8 dtype,
  reshapes back to the original KV shape, and casts to the destination
  tensor's dtype.
- `estimate_serialized_size` returns `int(total_elements × 1.5)` — the
  exact fp8 size is `num_elements × 1 byte`, and the 1.5× headroom
  absorbs future format changes or alignment padding.

## Extension Guide

Most custom serdes only need:

1. Two classes implementing `Serializer` / `Deserializer`.
2. A factory function registered at import time.
3. A JSON `serde` sub-dict on the adapter config: `{"type": "mine", ...}`.

Everything else — temp buffer allocation, eventfd plumbing, lock /
lifecycle transitions, all-or-nothing failure handling — is provided
by `AsyncSerdeProcessor` and the wrapper. Users never need to touch
controllers, eventfds, or L1 locks.

## Multi-output extension

The `Serializer` / `Deserializer` classes above operate on one
typed tensor at each endpoint: one tensor in, one byte buffer out
on serialize, and the reverse on deserialize. That shape works
when K and V share a single data type -- the serde just sees them
as one combined tensor. It does not work in two cases:

- **K and V at different data types.** A typed tensor has one
  dtype, so a serde that wants K at one dtype (e.g. fp16/bf16) and
  V at another (e.g. FP8) cannot carry both.
- **One side absent.** For tier-split placements where K or V is
  held outside this serde's data path -- e.g. K kept in L1 (CPU
  pinned host memory) while V flows to L2 (durable storage) -- the
  serialize input has no tensor for the absent slot and the
  deserialize output has no destination for it.

`multi.py` defines the additive contract for this case:

- `MemoryObjGroup = Tuple[Optional[MemoryObj], ...]` is a
  fixed-length tuple of optional MemoryObjs.
- `LayoutDescGroup = Tuple[Optional[MemoryLayoutDesc], ...]` is the
  parallel layout-descriptor tuple used by size estimators.
- `MultiSerializer.serialize(src: MemoryObjGroup, dst: MemoryObj,
  key: ObjectKey)` takes a group whose length equals
  `MultiSerializer.group_size`.
- `MultiDeserializer.deserialize(src: MemoryObj, dst: MemoryObjGroup,
  key: ObjectKey)` produces a group whose length equals
  `MultiDeserializer.group_size`.
- `single_to_multi_serializer(s)` and
  `single_to_multi_deserializer(d)` adapt an existing single-tensor
  pair to the tuple interface as a length-1 group; the adapter is
  layout-equivalent (same on-the-wire bytes as a direct call).

The single-tensor `Serializer` / `Deserializer` ABCs and all their
existing callers — `AsyncSerdeProcessor`, the factory registry, the
L2 adapter wrapper, the built-in `fp8` serde — are unchanged. A
serde implementation that needs multiple tensors at an endpoint
implements `MultiSerializer` / `MultiDeserializer` instead of (or in
addition to) the single-tensor ABCs. The async wiring around the
multi interface — a tuple-aware `AsyncSerdeProcessor` analog and a
tuple-aware `submit_*` shape on the wrapper — is added in a follow-up
once a concrete multi-output serde lands.

### Per-slot semantics

Implementations MUST document, at minimum:

1. The fixed value of `group_size`.
2. The semantic carried by each slot (e.g., `slot 0 = K`,
   `slot 1 = V`).
3. Which slots are required and which may be `None`. A slot that may
   be `None` MUST be tolerated by `estimate_serialized_size` with a
   `None` layout descriptor at the same index.

`None` semantics, mirroring serialize input vs deserialize output:

- **Serialize input.** A `None` slot means the caller is not
  supplying that tensor; e.g., a V-only write where K is held
  outside this serde's data path. The implementation MUST raise
  `ValueError` when a required slot is `None`.
- **Deserialize output.** A `None` slot means the caller does not
  want that tensor materialized; e.g., a V-only read where K is
  sourced elsewhere. The implementation MUST NOT touch the missing
  slot.

### Single-element bridge

A length-1 `MemoryObjGroup` is the trivial bridge to the existing
single-tensor API. `single_to_multi_serializer(inner)` wraps an
existing `Serializer` so callers that work in groups can invoke it
uniformly:

```python
from lmcache.v1.distributed.serde import (
    single_to_multi_serializer,
    single_to_multi_deserializer,
)

multi_s = single_to_multi_serializer(existing_serializer)
multi_d = single_to_multi_deserializer(existing_deserializer)
n = multi_s.serialize((src,), dst_buffer)         # length-1 group
multi_d.deserialize(src_buffer, (dst,))           # length-1 group
```

The wrapper rejects non-unit groups with `ValueError`, rejects a
`None` src slot (single-tensor serializers do not admit absence),
and treats a `None` dst slot as a deliberate skip rather than an
error. On-the-wire bytes are byte-for-byte identical to a direct
call against the underlying single-tensor serde.
