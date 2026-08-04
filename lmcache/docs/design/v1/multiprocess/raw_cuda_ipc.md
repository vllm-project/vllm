# `RawCudaIPCWrapper` — sharing tensors allocated outside PyTorch

## Why a second wrapper

The default `CudaIPCWrapper` in `platform/cuda/ipc_wrapper.py` calls
`tensor.untyped_storage()._share_cuda_()` to publish the storage over
CUDA IPC. That path only works when the storage is owned by PyTorch's
caching allocator. TRT-LLM's KV pool is published via
`at::for_blob(...)` over a raw `cudaMalloc`, which means
`_share_cuda_()` raises and the `vLLM`-style wrapper cannot be used.

`RawCudaIPCWrapper` bypasses PyTorch's IPC layer:

- **Sender** calls `cudaIpcGetMemHandle(data_ptr)` directly via
  `cuda.bindings.runtime` to obtain a portable handle.
- **Receiver** calls `cudaIpcOpenMemHandle(handle, ...)` to map the
  remote pointer, wraps it as a flat `uint8` `cupy.ndarray` via
  `UnownedMemory`, converts to `torch.Tensor` via DLPack, then
  `view(dtype).reshape(shape)` to restore the logical layout.

The `uint8` round-trip is deliberate — `bfloat16` and FP8 dtypes have
no direct CuPy/NumPy equivalent, but the size in bytes is enough.
DLPack carries no dtype semantics for the bytes view; `view(dtype)` on
the torch side restores them.

## Shared base rather than separate type

`RawCudaIPCWrapper` is a **sibling** of `CudaIPCWrapper`: both subclass
the device-agnostic `DeviceIPCWrapper` base (see
[`device_ipc_wrapper_design.md`](device_ipc_wrapper_design.md) for the
full hierarchy). Sharing a single base is load-bearing for the wire
format:

- `KVCache = list[DeviceIPCWrapper]` is the registered msgspec type for
  `REGISTER_KV_CACHE`. msgspec does **not** support unions of custom
  ext-encoded types — adding a parallel class with its own ext code
  would force a wider decoder type and break either round-trip or the
  existing `DeviceIPCWrapper` consumers.
- The customized serializer (`_CUSTOMERIZED_SERIALIZERS`) is keyed on
  `DeviceIPCWrapper` and dispatched by `isinstance`, so every subclass
  instance encodes through the same path with **ext code 1**.
- `Serialize` is `pickle.dumps(obj)`, which preserves the concrete
  subclass identity. On the receiving side `Deserialize` reconstructs
  the concrete subclass and `to_tensor` dispatches to the correct
  override.

The receiving server therefore needs no per-type branching: a
`list[DeviceIPCWrapper]` arriving at
`LMCacheDrivenTransferModule.register_kv_cache` contains any mix of
concrete wrappers, and `to_tensor()` does the right thing.

## Sender-side validation

`RawCudaIPCWrapper.__init__` calls `assert_contiguous` (from
`gpu_connector/utils.py`) instead of permuting. TRT-LLM allocates the
pool contiguously, so the only valid recovery from a non-contiguous
input is "the sender did something wrong" — surface it loudly rather
than silently `.contiguous()`-ing and copying GBs of KV cache.

## Reconstruction lifetime

`UnownedMemory` takes `owner=self`, so the wrapper instance pins the
mapping for the tensor's lifetime. The mapping is dropped when the
wrapper is GC'd; the underlying TRT-LLM allocation outlives the
wrapper. There is no symmetric `cudaIpcCloseMemHandle` call — torch
controls the lifetime through DLPack reference counting plus
CuPy/MemoryPointer's owner field.

## Why no `_share_cuda_` fallback

The wrapper does not try `_share_cuda_()` first. That would couple the
two codepaths, and the failure mode is silent corruption (PyTorch
returns a handle for a different region of memory than what the
caller intended). Keeping `RawCudaIPCWrapper` as its own concrete
sibling of `CudaIPCWrapper` keeps the choice at the call site.
