# `DeviceIPCWrapper` — a device-agnostic base for KV-cache IPC

## Motivation

The multiprocess (MP) transport shares a paged KV-cache buffer between the producer process (vLLM / SGLang / TRT-LLM) and the LMCache server by sending an IPC handle for the underlying storage once, at `REGISTER_KV_CACHE` time. Every later `STORE`/`RETRIEVE` carries only paged block IDs, never tensors.

Historically all of these handle wrappers subclassed `CudaIPCWrapper`:

```text
CudaIPCWrapper
├── RawCudaIPCWrapper      (TRT-LLM raw cudaMalloc pool)
└── CpuShmTensorWrapper    (CPU POSIX-SHM, in platform/cpu/shm.py)
```

In fact a CPU shared-memory wrapper and a TRT-LLM raw-pointer wrapper are not CUDA caching-allocator tensors, yet they inherited `_share_cuda_`- based machinery they never used. It also made non-CUDA backends impossible to add cleanly.

## The new hierarchy

A device-agnostic base, `DeviceIPCWrapper`, now owns everything that is not transport-specific. Each concrete wrapper is a direct sibling:

```text
DeviceIPCWrapper                        base: contract + (de)serialize
├── CudaIPCWrapper                      cuda  — torch caching allocator
├── RawCudaIPCWrapper                   cuda — raw cudaMalloc, TRT-LLM
├── CpuShmTensorWrapper                 cpu   — POSIX shared memory
└── MusaIPCWrapper                      musa  — TorchMUSA memory IPC
```

All of them live behind a single msgspec ext code 1 and a single `KVCache = list[DeviceIPCWrapper]` wire type, so new device backends can be added as further siblings without touching the wire format.

## What the base class owns

`DeviceIPCWrapper` (in `platform/base/ipc_wrapper.py`) provides the parts that every
transport shares:

- **Interface fields** — `dtype`, `shape`, `stride`, `storage_offset`, `device_uuid`. Subclasses populate these in `__init__`; the base uses them for equality and the receiving side uses them to rebuild the logical view.
- **Device discovery** — `_get_device_uuid`, `_discover_devices`,`_get_device_index_from_uuid`. These are `@classmethod`s (not static) and route through the `torch_dev` abstraction, so they work across device backends, and a subclass can override `_get_device_uuid` if its backend needs a different identity source.
- **Equality** — `__eq__` uses a `type(self) is type(other)` guard, so two different wrapper subclasses never compare equal even if their fields coincide.
- **Serialization** — `Serialize`/`Deserialize` are `pickle.dumps` / `pickle.loads`. Pickle preserves the concrete subclass identity across the wire, which is what lets a single ext code carry every wrapper type (see below).
- **`to_tensor()`** — abstract; raises `NotImplementedError`. Every subclass overrides it with its transport-specific reconstruction.

## How to dispatch

- `_CUSTOMERIZED_SERIALIZERS` is keyed on `DeviceIPCWrapper` with ext code 1, dispatched by `isinstance` in the encoder hook. Every subclass instance therefore encodes through the same path.
- `Serialize` is `pickle.dumps(obj)` -> the concrete subclass survives on the wire. `Deserialize` reconstructs it and `to_tensor()` dispatches to the correct override.
- `KVCache = list[DeviceIPCWrapper]` is the registered msgspec type, so a single `list[...]` payload can mix any of the wrappers and the server needs zero per-type branching.

## The concrete wrappers

| Wrapper | Device type | Transport | Reconstruction |
|---|---|---|---|
| `CudaIPCWrapper` | `cuda` | `UntypedStorage._share_cuda_()` | `_new_shared_cuda` + `set_()` |
| `RawCudaIPCWrapper` | `cuda` | `cudaIpcGetMemHandle` (raw ptr) | `cudaIpcOpenMemHandle` → CuPy → DLPack |
| `CpuShmTensorWrapper` | `cpu` | POSIX `shm_open` | `mmap` same segment |
| `MusaIPCWrapper` | `musa` | TorchMUSA IPC API | `torch.musa.ipc.export_tensor` + `open_tensor` |

## Platform registration

The factory lookup (`platform.resolve_kv_wrapper_factory`) keys on
`tensor.device.type`, so the integration adapter never has an if/elif
chain.  Concrete wrappers are bound to their device via
`DeviceSpec.ipc_wrapper_cls` — no static `register_kv_wrapper` calls
needed:

- Each concrete subclass carries a ``device_type`` ClassVar (e.g.
  ``"cuda"``) for introspection and exposes a ``wrap`` factory
  classmethod.
- Each accelerator's :class:`DeviceSpec` subclass (e.g.
  :class:`CudaDeviceSpec`) overrides
  :attr:`~DeviceSpec.ipc_wrapper_cls` to return its default
  wrapper class.
- :func:`~lmcache.v1.platform.resolve_kv_wrapper_factory` reads that
  binding off the registered spec and returns the wrapper's ``wrap``
  classmethod so callers can invoke ``factory(tensor)`` uniformly.
- ``RawCudaIPCWrapper`` intentionally stays off the spec so it
  coexists with ``CudaIPCWrapper`` without collision — callers
  (TRT-LLM adapter) instantiate it directly.
- Adding a new accelerator backend only requires shipping a sub-package
  under ``platform/<device>/`` with a ``DeviceSpec`` subclass whose
  ``ipc_wrapper_cls`` returns the wrapper — zero changes to the
  dispatcher.

## Backward compatibility

- The wire envelope remains ext code 1 and `pickle`-over-`Ext`. MUSA sender and receiver processes must use compatible `MusaIPCWrapper` versions because the payload carries a TorchMUSA tensor handle.
- `MusaIPCWrapper` keeps the receiver-side `open_tensor()` owner alive locally and excludes that owner from serialized state.
- `CudaIPCWrapper` / `RawCudaIPCWrapper` keep their names, fields, and behavior; only their base class changed. Existing callers and the TRT-LLM adapter are unaffected.
- The single `isinstance`-based equality check now uses `type(self) is type(other)`, which is stricter and correct.
