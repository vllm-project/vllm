# LMCache Rust Raw Block I/O

This crate provides the low-level raw device I/O layer for LMCache via Rust +
PyO3. It is used by both:

- the legacy non-MP `RustRawBlockBackend`
- the MP `raw_block` L2 adapter (`RawBlockL2Adapter`) via `RawBlockCore`

The Rust crate intentionally stays narrow: it owns the raw device handle and
exposes blocking `pwrite_from_buffer` / `pread_into` primitives. Slotting,
checkpointing, recovery, and MP task orchestration all live in Python.

## I/O Engines

`RawBlockDevice` accepts `io_engine`:

- `posix` (default): synchronous Linux `pread` / `pwrite`.
- `io_uring`: direct Rust io_uring syscall path using the existing worker,
  batch, and `wait_iouring` machinery.

`use_iouring=True` remains accepted for backward compatibility. If `io_engine`
is explicitly set, it wins over the legacy flag.

## io_uring_cmd (NVMe Passthrough)

When `io_engine="io_uring"`, you can optionally enable `use_uring_cmd=True` to
use NVMe passthrough via the io_uring command interface for direct device access.

**io_uring_cmd notes:**

- Requires NVMe character device node (`/dev/ngXnY`) instead of the block device
  node (`/dev/nvmeXnY`) for direct NVMe passthrough command.
- Requires `io_engine="io_uring"` to be set.
- Supports `max_data_transfer_size` parameter to split large transfers into
  smaller chunks that fit within device limits.
- Requires `alignment` to be a multiple of the NVMe namespace LBA size. An
  incompatible value is rejected when the device opens.
- When `use_uring_cmd=True`, `use_odirect` is ignored for NVMe namespace
  character devices.
- SQE build failures are returned by `wait_iouring` after the worker releases
  the request's global and per-batch in-flight accounting.

## MP Mode Integration

In MP mode, the stack looks like this:

```text
StoreController / PrefetchController
                |
                v
        RawBlockL2Adapter
                |
                v
           RawBlockCore
                |
                v
         lmcache_rust_raw_block_io
                |
                v
         raw device / file
```

This split lets LMCache reuse the same on-device metadata and recovery model in
both non-MP and MP mode without duplicating the raw-block implementation.

## Zero-Copy Data Path

```text
LMCache LocalCPUBackend (aligned pinned CPU tensor)
                 |
                 |  Python buffer / memoryview (no payload memcpy)
                 v
RustRawBlockBackend (PyO3 boundary)
                 |
                 |  direct pointer path when O_DIRECT constraints are met
                 |  fallback: bounce only for unaligned tail/block
                 v
RawBlockDevice::pwrite_from_buffer / pread_into
                 |
                 v
Block device or file
```

## How To Compare Performance

To compare `local_disk` vs `rust_raw_block` on a real NVMe device:
- Run `local_disk` on an ext4 mount of the device.
- Unmount it.
- Run `rust_raw_block` directly on the raw block device.

Use the benchmark commands in:
- `benchmarks/storage_backend_io/README.md`

No fixed numbers are included here because results are host/device/workload dependent.

## Limitations

- Linux only (`pread` / `pwrite`, O_DIRECT semantics).
- `alignment` and `block_align` must be powers of two, such as 4096.
- O_DIRECT requires aligned offsets and I/O lengths. `batched_write` rejects
  requests whose offset or `total_len` is not a multiple of the configured
  alignment; misaligned write buffers are copied through an aligned bounce
  buffer.

## Build

```bash
cd rust/raw_block
pip install maturin
maturin develop --release
```

## Minimal Usage

```python
from lmcache_rust_raw_block_io import RawBlockDevice

dev = RawBlockDevice("/dev/nvme0n1", True, use_odirect=True, alignment=4096)
dev.pwrite_from_buffer(offset=0, data=b"hello", total_len=4096)

buf = bytearray(4096)
dev.pread_into(offset=0, out=buf, payload_len=5, total_len=4096)
```

io_uring:

```python
dev = RawBlockDevice(
    "/dev/nvme0n1",
    True,
    use_odirect=True,
    alignment=4096,
    io_engine="io_uring",
    iouring_queue_depth=256,
)
```

io_uring with io_uring_cmd (NVMe passthrough):

```python
dev = RawBlockDevice(
    "/dev/ng0n1",  # Note: NVMe character device node
    True,
    use_odirect=False,
    alignment=4096,
    io_engine="io_uring",
    use_uring_cmd=True,
    iouring_queue_depth=256,
    max_data_transfer_size=131072,  # Optional: split large transfers
)
```

## MP Adapter Example

To use the MP adapter from `lmcache server`, pass a `raw_block` L2 adapter
config:

```bash
lmcache server \
  --l1-size-gb 10 \
  --eviction-policy LRU \
  --l1-align-bytes 4096 \
  --l2-adapter '{
    "type": "raw_block",
    "device_path": "/dev/nvme0n1",
    "slot_bytes": 1048576,
    "block_align": 4096,
    "header_bytes": 4096,
    "meta_total_bytes": 268435456,
    "use_odirect": true,
    "io_engine": "io_uring",
    "num_store_workers": 2,
    "num_lookup_workers": 1,
    "num_load_workers": 4
  }'
```

Notes:

- `device_path` should point to an unmounted raw block device or a dedicated
  file used only by LMCache.
- For `use_uring_cmd=true`, `device_path` must use the NVMe character
  device node (e.g., `/dev/ng0n1`) instead of the block device node.
- `block_align` must be a power of two. `slot_bytes`, `header_bytes`, and
  `meta_total_bytes` must be multiples of `block_align`.
- With `use_odirect=true`, LMCache MP L1 alignment must be at least
  `block_align`.
- Restart recovery uses the metadata checkpoint region on the same device.
- Raw-block slot reclamation is driven by the shared/global L2 eviction
  controller or explicit `delete()` calls.
- `raw_block` remains the adapter type for all supported engines.
