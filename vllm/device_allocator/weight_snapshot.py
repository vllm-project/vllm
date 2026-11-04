# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Snapshot of post-processed weights held by the cumem allocator.

The first start saves every live "weights" allocation after
process_weights_after_loading. Later starts build the model with dummy
weights, which repeats the same allocation sequence, and overwrite each
allocation with its saved bytes, skipping checkpoint reads and conversion.
"""

import json
import mmap
import os
import time
from pathlib import Path

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_ALIGN = 4096
_CHUNK = 256 << 20
_MAGIC = "vllm-weight-snapshot-v1"


def snapshot_path(root: str, config_hash: str, rank: int) -> Path:
    return Path(root) / f"{config_hash}-rank{rank}.bin"


def _weight_allocations(allocator) -> list[tuple[int, int]]:
    # pointer_to_data keeps insertion order, i.e. allocation order.
    return [
        (ptr, data.handle[1])
        for ptr, data in allocator.pointer_to_data.items()
        if data.tag == "weights" and not data.is_asleep
    ]


def _round_up(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


def _pinned_buffer() -> torch.Tensor:
    return torch.empty(_CHUNK, dtype=torch.uint8, pin_memory=True)


def save(allocator, path: Path) -> None:
    from vllm.device_allocator.cumem import libcudart

    torch.accelerator.synchronize()
    allocs = _weight_allocations(allocator)
    header = json.dumps(
        {
            "magic": _MAGIC,
            "sizes": [s for _, s in allocs],
            "ptrs": [p for p, _ in allocs],
        }
    ).encode()
    header_len = _round_up(len(header) + 8)
    tmp = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = _pinned_buffer()
    t0 = time.perf_counter()
    with open(tmp, "wb") as f:
        f.write(len(header).to_bytes(8, "little") + header)
        f.write(b"\0" * (header_len - 8 - len(header)))
        for ptr, size in allocs:
            for off in range(0, size, _CHUNK):
                n = min(_CHUNK, size - off)
                libcudart.cudaMemcpy(buf.data_ptr(), ptr + off, n)
                f.write(memoryview(buf.numpy())[:n])
            f.write(b"\0" * (_round_up(size) - size))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    total = sum(s for _, s in allocs)
    logger.info(
        "Weight snapshot saved: %d allocations, %.2f GiB in %.1f s to %s",
        len(allocs),
        total / 2**30,
        time.perf_counter() - t0,
        path,
    )


def restore(allocator, path: Path) -> None:
    """Overwrite the dummy-initialized weight allocations with the snapshot."""
    from vllm.device_allocator.cumem import libcudart

    allocs = _weight_allocations(allocator)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
    try:
        head = mmap.mmap(-1, _ALIGN)
        os.preadv(fd, [head], 0)
        hlen = int.from_bytes(head[:8], "little")
        hbuf = mmap.mmap(-1, _round_up(hlen + 8))
        os.preadv(fd, [hbuf], 0)
        header = json.loads(bytes(hbuf[8 : 8 + hlen]))
        if header.get("magic") != _MAGIC or header["sizes"] != [s for _, s in allocs]:
            raise RuntimeError(
                f"weight snapshot {path} does not match this model's allocation "
                f"sequence ({len(header.get('sizes', []))} saved vs {len(allocs)} "
                "allocated); delete it to rebuild"
            )
        if header["ptrs"] != [p for p, _ in allocs]:
            logger.warning("Weight snapshot restored at different addresses.")
        buf = _pinned_buffer()
        view = memoryview(buf.numpy())
        offset = _round_up(hlen + 8)
        torch.accelerator.synchronize()
        t0 = time.perf_counter()
        for ptr, size in allocs:
            for off in range(0, size, _CHUNK):
                n = min(_CHUNK, size - off)
                os.preadv(fd, [view[: _round_up(n)]], offset + off)
                libcudart.cudaMemcpy(ptr + off, buf.data_ptr(), n)
            offset += _round_up(size)
    finally:
        os.close(fd)
    total = sum(s for _, s in allocs)
    elapsed = time.perf_counter() - t0
    logger.info(
        "Weight snapshot restored: %.2f GiB in %.1f s (%.2f GB/s) from %s",
        total / 2**30,
        elapsed,
        total / elapsed / 1e9,
        path,
    )
