# Platform Event IPC Abstraction

## Motivation

The `lmcache_driven` multiprocess handle path uses device events to order
cross-process KV-cache transfers:

1. The worker records an event after producing or reserving KV-cache blocks.
2. The server imports that event and waits before reading or writing shared
   KV-cache memory.
3. The server records a completion event after transfer work is queued.
4. The worker future imports, polls, and waits for that completion event.

CUDA-style backends expose this through `Event(interprocess=True)`,
`Event.ipc_handle()`, and `Event.from_ipc_handle(...)`. MUSA has its own
capability gate and TorchMUSA event module. If generic multiprocess modules
import MUSA code or branch on `device.type == "musa"`, every new accelerator
must add another special case to the same logic layer.

The platform event IPC abstraction keeps that decision in
`lmcache/v1/platform/`. Generic multiprocess code calls one backend-neutral
interface; `DeviceSpec` selects the implementation for the requested device.

## Goals

- Keep the `lmcache_driven` handle path free of backend-specific imports and
  device-type branches.
- Preserve `CUDAMessagingFuture` and `to_cuda_future` compatibility aliases.
- Provide CUDA-style/default and MUSA event backends behind one interface.
- Fail before cross-process memory access when event IPC is unavailable.
- Require each concrete `DeviceSpec` to declare its event IPC capability; do
  not select an unrelated backend through an implicit fallback.

## Non-Goals

- The `STORE` and `RETRIEVE` wire formats do not change; event handles remain
  serialized `bytes` values.
- Tensor memory IPC remains owned by `DeviceIPCWrapper` implementations.
- CacheBlend modules are not migrated in this change; they remain a separate
  CUDA-only integration until a follow-up adopts the same interface.
- Worker adapters continue to construct and initially record their producer
  events. The abstraction covers export from the worker and all server/future
  event operations.

## API

`lmcache/v1/platform/base/event_ipc.py` defines the runtime-checkable protocol:

```python
class EventIPCBackend(Protocol):
    device_type: str

    def check_event_support(self, device: object) -> None: ...
    def create_event(self, device: object) -> object: ...
    def export_event(self, event: object, device: object) -> bytes: ...
    def import_event(self, handle: bytes, device: object) -> object: ...
    def record_event(self, event: object, stream: object) -> None: ...
    def wait_event(self, event: object, stream: object) -> None: ...
    def query_event(self, event: object) -> bool: ...
    def synchronize_event(self, event: object, device: object) -> None: ...
```

The lookup entry point is:

```python
def get_event_ipc_backend(device: object) -> EventIPCBackend: ...
```

`device` may be a `torch.device`-like object, a device-type string, or an
integer device index. The lookup resolves a `DeviceSpec` and returns its
`event_ipc_backend`. A missing `DeviceSpec` is a platform registration error,
and a registered spec without an event backend is an unsupported-capability
error. Both cases raise `RuntimeError` instead of selecting a fallback.

All event, device, and stream values are opaque `object` handles at this layer.
The concrete backend owns their types and ABI details.

## Backend Implementations

```text
lmcache/v1/platform/base/event_ipc.py     # protocol, lookup, default backend
lmcache/v1/platform/base/device_spec.py  # optional DeviceSpec capability
lmcache/v1/platform/cpu/__init__.py      # CPU stub backend registration
lmcache/v1/platform/cuda/__init__.py     # CUDA backend registration
lmcache/v1/platform/musa/event_ipc.py    # MUSA capability gate and adapter
lmcache/v1/platform/musa/ipc_wrapper.py  # MUSA availability and torch.musa loader
```

### Default backend

`DefaultEventIPCBackend` adapts an event module with the CUDA-style API:

```python
event = event_module.Event(interprocess=True)
handle = event.ipc_handle()
remote_event = event_module.Event.from_ipc_handle(device, handle)
event.record(stream)
remote_event.wait(stream)
remote_event.query()
remote_event.synchronize()
```

The base `DeviceSpec.event_ipc_backend` returns `None`. `CpuDeviceSpec`
explicitly binds this adapter to `StubCPUDevice`, while `CudaDeviceSpec` binds
it to `torch.cuda`. Other accelerators must explicitly register a compatible
backend before the multiprocess handle path can use them. This prevents an
unsupported or misspelled device type from silently selecting the active
accelerator's event implementation.

### MUSA backend

`MusaEventIPCBackend` lives under `lmcache/v1/platform/musa/`. It first checks
`is_musa_event_ipc_available()` from `musa/ipc_wrapper.py`, then adapts the
current TorchMUSA event API through `DefaultEventIPCBackend`:

```python
torch_musa = get_torch_musa_module()
event = torch_musa.Event(interprocess=True)
handle = event.ipc_handle()
remote_event = torch_musa.Event.from_ipc_handle(device, handle)
```

The generic layer does not import `platform.musa`; only the MUSA `DeviceSpec`
and backend do. If the opt-in MUSA event API is unavailable,
`check_event_support()` raises a device-named `RuntimeError`. Memory IPC and
server-side block transfer are separate capabilities and are not required to
validate the standalone event backend.

## Registration

Event backends are properties of `DeviceSpec`, alongside device detection and
memory IPC wrapper selection:

```python
class DeviceSpec:
    @property
    def event_ipc_backend(self) -> EventIPCBackend | None:
        return None

class CudaDeviceSpec(DeviceSpec):
    @property
    def event_ipc_backend(self) -> EventIPCBackend:
        return DefaultEventIPCBackend(event_module=torch.cuda, ...)

class MusaDeviceSpec(DeviceSpec):
    @property
    def event_ipc_backend(self) -> EventIPCBackend:
        return MusaEventIPCBackend()
```

This keeps platform capabilities together. Adding another backend requires a
new platform package/spec, not an edit to the multiprocess transfer modules.

## Generic Multiprocess Flow

```text
Worker adapter
  create/record producer event using its existing device API
  get_event_ipc_backend(device)
  export_event(producer_event, device)
  send event handle in STORE/RETRIEVE

Server: lmcache_driven_transfer.py
  get_event_ipc_backend(cache_context.device)
  check_event_support(device)
  import_event(worker_handle, device)
  wait_event(imported_event, cache_context.stream)
  enqueue KV transfer
  create_event(device), record_event(done_event, stream)
  export_event(done_event, device)

Worker: futures.py
  get_event_ipc_backend(device)
  import_event(server_handle, device)
  query_event / wait_event / synchronize_event
```

The following generic modules use only the platform API:

- `lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py`
- `lmcache/v1/multiprocess/futures.py`
- `lmcache/v1/multiprocess/transfer_context/worker_transfer.py`

`DeviceMessagingFuture` is the device-neutral implementation. The old
`CUDAMessagingFuture` name remains an alias, and `to_cuda_future()` remains an
alias for `to_device_future()`.

## Error Contract

`check_event_support(device)` runs before any cross-process memory transfer.
The default backend rejects missing `Event`, missing `interprocess` support, or
missing `Event.from_ipc_handle`. The MUSA backend additionally rejects a
disabled/unavailable MUSA event API, a missing TorchMUSA module, or an event
module without the required interprocess API. For opaque C/pybind event
bindings, capability detection may probe `Event(interprocess=True)` when a
Python signature is unavailable.

Errors include the backend name and missing capability. They must not claim
that CUDA is required when a non-CUDA backend has its own implementation.

## Migration Plan

1. Define `EventIPCBackend`, the default adapter, and lookup in
   `platform/base/event_ipc.py`.
2. Add the optional `DeviceSpec.event_ipc_backend` capability and explicit
   CPU, CUDA, and MUSA registrations.
3. Replace direct event creation/import/wait/record/export/query calls in the
   `lmcache_driven` server path with the platform API.
4. Route `DeviceMessagingFuture` import/query/wait/synchronize through the
   platform API while preserving the CUDA aliases.
5. Route worker-side producer-event export through the platform API.
6. Keep CacheBlend and other out-of-scope CUDA-only integrations unchanged.

## Testing

Tests cover:

- default event creation, export, import, record, wait, query, synchronize;
- CPU backend selection while another accelerator is active;
- strict failures for unregistered devices and unsupported event IPC;
- MUSA capability failure and missing-module failure;
- MUSA delegation to the TorchMUSA event API using a fake module;
- future compatibility aliases and device-aware event behavior;
- structural absence of direct MUSA imports in the generic handle-path modules.

Real MUSA hardware tests remain capability-gated. The platform tests do not
need MUSA hardware because they validate the public backend contract using
injected event modules.

## Review Checklist

- [ ] Generic handle-path modules do not import `platform.musa`.
- [ ] Generic handle-path modules do not branch on a concrete device type.
- [ ] `CUDAMessagingFuture` and `to_cuda_future` remain compatible aliases.
- [ ] Concrete device specs explicitly register their event IPC backends.
- [ ] Missing specs and unsupported event IPC fail without fallback.
- [ ] MUSA event behavior is isolated under `platform/musa`.
- [ ] Unsupported event IPC fails before unsafe memory access.
- [ ] Platform and future tests cover import/export and stream ordering.
