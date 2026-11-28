# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read safetensors checkpoint tensors straight to the GPU.

The default loader hands each weight loader a CPU view of the file mapping, so
every narrow, `.contiguous()` and pad happens on the CPU, one tensor at a
time, page-faulting the mapping, before a synchronous pageable copy to the GPU.
`gpu_staged_safetensors_file` instead reads the bytes of the tensors in large,
merged file ranges with `os.preadv` (several threads, GIL released) into a ring
of pinned host buffers, copies each range to the GPU on a separate stream, and
yields each tensor as a GPU tensor with the dtype and shape that `safe_open`
would give. The weight loaders then slice, pad and copy on the GPU.

Staging uses `_NUM_BUFFERS` pinned host and GPU buffers, allocated once per
load and sized to the largest range read (`_CHUNK_BYTES`, or one tensor of up
to `_MAX_TENSOR_BYTES`), plus one weight at a time handed to the loaders.
Tensors larger than `_MAX_TENSOR_BYTES` and those of unknown dtype are yielded
as `safe_open` views, as before.

Enabled with `--model-loader-extra-config '{"gpu_staging": true}'`.
"""

import concurrent.futures
import functools
import json
import os
import queue
import struct
import threading
import time
from collections.abc import Callable, Generator

import torch
from safetensors.torch import safe_open

from vllm.logger import init_logger

logger = init_logger(__name__)

_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "U16": torch.uint16,
    "I32": torch.int32,
    "U32": torch.uint32,
    "I64": torch.int64,
    "U64": torch.uint64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
}
# Adjacent tensors are read with one pread range of up to this many bytes.
_CHUNK_BYTES = 64 << 20
# Larger tensors (e.g. embeddings of large vocabularies) stay safe_open views.
_MAX_TENSOR_BYTES = 512 << 20
_READ_BYTES = 16 << 20  # per pread call; a range is split across threads
_NUM_READ_THREADS = 8
_NUM_BUFFERS = 3  # pinned + GPU staging buffers (ranges in flight)

# Accumulated over a load, logged by `release_gpu_staging`.
_stats = {
    "staged_bytes": 0,
    "staged_tensors": 0,
    "mapped_tensors": 0,
    "read_s": 0.0,
    "wait_s": 0.0,
    "loader_s": 0.0,
}

# (kind, value) per tensor or range of one file: ("map", tensor name) or
# ("range", index into the file's ranges).
_Item = tuple[str, str | int]
# One pread range: [start, end, entries], with entries
# (name, dtype, shape, offset in the range, nbytes).
_Range = list


@functools.cache
def _copy_stream(device: torch.device) -> torch.cuda.Stream:
    # One stream per device for the whole load: the caching allocator reuses
    # blocks only within the stream they were allocated on.
    return torch.cuda.Stream(device)


@functools.cache
def _read_pool() -> concurrent.futures.ThreadPoolExecutor:
    return concurrent.futures.ThreadPoolExecutor(
        _NUM_READ_THREADS, thread_name_prefix="vllm-gpu-staging-read"
    )


def _pread_into(fd: int, buf: memoryview, pos: int) -> None:
    while buf:
        n = os.preadv(fd, [buf], pos)
        if n == 0:
            raise EOFError(f"short read at offset {pos}")
        buf, pos = buf[n:], pos + n


def _pread_range(fd: int, out: torch.Tensor, pos: int) -> None:
    """Read len(out) bytes at file offset pos into the uint8 tensor out."""
    buf = memoryview(out.numpy())
    n = len(buf)
    if n <= _READ_BYTES:
        _pread_into(fd, buf, pos)
        return
    futures = [
        _read_pool().submit(_pread_into, fd, buf[s : s + _READ_BYTES], pos + s)
        for s in range(0, n, _READ_BYTES)
    ]
    for f in futures:
        f.result()


def _plan(path: str, skip: Callable[[str], bool]) -> tuple[list[_Item], list[_Range]]:
    """Return (items, ranges) for one file, in file order, without the
    tensors that `skip` accepts."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    base = 8 + n
    items: list[_Item] = []
    ranges: list[_Range] = []
    cur: _Range | None = None
    for name, info in sorted(header.items(), key=lambda e: e[1]["data_offsets"]):
        if skip(name):
            continue
        start, end = (base + o for o in info["data_offsets"])
        dtype = _DTYPES.get(info["dtype"])
        if dtype is None or not 0 < end - start <= _MAX_TENSOR_BYTES:
            items.append(("map", name))
            cur = None
            continue
        if cur is None or start != cur[1] or end - cur[0] > _CHUNK_BYTES:
            cur = [start, end, []]
            ranges.append(cur)
            items.append(("range", len(ranges) - 1))
        cur[1] = end
        cur[2].append((name, dtype, info["shape"], start - cur[0], end - start))
    return items, ranges


class _Ring:
    """Pinned host and GPU staging buffers of one device, kept for the whole
    load (`release_gpu_staging` frees them), so the extra memory is fixed:
    _NUM_BUFFERS x the largest range read, with no allocator churn."""

    def __init__(self, device: torch.device):
        self.device = device
        self.cap = 0
        self.host: list[torch.Tensor] = []
        self.dev: list[torch.Tensor] = []

    def ensure(self, cap: int) -> None:
        if cap <= self.cap:
            return
        cap = -(-cap // _CHUNK_BYTES) * _CHUNK_BYTES
        torch.accelerator.synchronize(self.device)  # nothing still uses the old ones
        self.host = self.dev = []
        self.host = [
            torch.empty(cap, dtype=torch.uint8, pin_memory=True)
            for _ in range(_NUM_BUFFERS)
        ]
        self.dev = [
            torch.empty(cap, dtype=torch.uint8, device=self.device)
            for _ in range(_NUM_BUFFERS)
        ]
        self.cap = cap


_rings: dict[torch.device, _Ring] = {}


def gpu_staged_safetensors_file(
    path: str, skip: Callable[[str], bool]
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield the tensors of one safetensors file, in file order; see module
    docstring. `skip(name)` drops a tensor before it is read."""
    items, ranges = _plan(path, skip)
    if not ranges:
        with safe_open(path, framework="pt") as st:
            for _, name in items:
                assert isinstance(name, str)
                _stats["mapped_tensors"] += 1
                yield name, st.get_tensor(name)
        return

    device = torch.device("cuda", torch.accelerator.current_device_index())
    ring = _rings.setdefault(device, _Ring(device))
    ring.ensure(max(r[1] - r[0] for r in ranges))
    copy_stream = _copy_stream(device)
    stream = torch.cuda.current_stream(device)
    # Slots the producer may fill, with the event after which the consumer is
    # done reading the slot's GPU buffer (None: never used in this file).
    free: queue.Queue[tuple[int, torch.cuda.Event | None]] = queue.Queue()
    for k in range(_NUM_BUFFERS):
        free.put((k, None))
    ready: queue.Queue[tuple[int, torch.cuda.Event] | BaseException] = queue.Queue()
    stop = threading.Event()
    fd = os.open(path, os.O_RDONLY)

    def get_free() -> tuple[int, torch.cuda.Event | None] | None:
        while not stop.is_set():
            try:
                return free.get(timeout=0.1)
            except queue.Empty:
                pass
        return None

    def produce() -> None:
        try:
            torch.accelerator.set_device_index(device)
            for start, end, _ in ranges:
                slot = get_free()
                if slot is None:
                    return
                k, consumed = slot
                if consumed is not None:
                    # The consumer's copies out of dev[k] are done, and with
                    # them the host-to-device copy that last read host[k].
                    consumed.synchronize()
                host = ring.host[k][: end - start]
                t0 = time.perf_counter()
                _pread_range(fd, host, start)
                _stats["read_s"] += time.perf_counter() - t0
                with torch.cuda.stream(copy_stream):
                    ring.dev[k][: end - start].copy_(host, non_blocking=True)
                    copied = torch.cuda.Event()
                    copied.record(copy_stream)
                ready.put((k, copied))
        except BaseException as e:  # surfaced to the consumer
            ready.put(e)

    producer = threading.Thread(target=produce, daemon=True, name="vllm-gpu-staging")
    producer.start()
    try:
        with safe_open(path, framework="pt") as st:
            for kind, x in items:
                if kind == "map":
                    assert isinstance(x, str)
                    _stats["mapped_tensors"] += 1
                    yield x, st.get_tensor(x)
                    continue
                assert isinstance(x, int)
                t0 = time.perf_counter()
                got = ready.get()
                _stats["wait_s"] += time.perf_counter() - t0
                if isinstance(got, BaseException):
                    raise got
                k, copied = got
                stream.wait_event(copied)
                dev = ring.dev[k]
                for name, dtype, shape, off, nbytes in ranges[x][2]:
                    # A fresh tensor per weight: the staging buffer is reused,
                    # and a loader may keep a reference to what it is given.
                    out = torch.empty(shape, dtype=dtype, device=device)
                    out.view(-1).view(torch.uint8).copy_(dev[off : off + nbytes])
                    _stats["staged_tensors"] += 1
                    _stats["staged_bytes"] += nbytes
                    t1 = time.perf_counter()
                    yield name, out
                    _stats["loader_s"] += time.perf_counter() - t1
                consumed = torch.cuda.Event()
                consumed.record(stream)
                free.put((k, consumed))
    finally:
        stop.set()
        producer.join()
        os.close(fd)
        # The next file's producer must not overwrite buffers still in use.
        copy_stream.synchronize()
        stream.synchronize()


def release_gpu_staging() -> None:
    """Log the load's statistics and free the staging buffers."""
    for ring in _rings.values():
        torch.accelerator.synchronize(ring.device)
    _rings.clear()
    s = _stats
    logger.info(
        "GPU staging: %.2f GiB in %d tensors read to the GPU, %d tensors as "
        "mapped views; pread %.2f s (thread time), waited %.2f s for reads, "
        "%.2f s in weight loaders for staged tensors.",
        s["staged_bytes"] / 2**30,
        s["staged_tensors"],
        s["mapped_tensors"],
        s["read_s"],
        s["wait_s"],
        s["loader_s"],
    )
    for key in s:
        s[key] = type(s[key])()
