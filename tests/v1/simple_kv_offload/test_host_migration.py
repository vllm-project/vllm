# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm: the kernel migrating pages of the CPU KV cache must not stall the GPU.

`hipHostRegister` leaves a range's pages movable: amdgpu tracks it as a userptr
through an MMU notifier, and when the kernel migrates one of its pages (memory
compaction; here `move_pages(2)` on our own pages) KFD stops every queue of the
process until the whole range is valid again, about 0.25-0.5 s per GiB of 4 KiB
pages. A registered CPU KV cache of 16 GiB stalled the GPU for 4.5 s, one of
128 GiB for 20-30 s. The cache is now never registered; only the small staging
buffer of the copies is GPU-visible.
"""

import ctypes
import os
import threading
import time

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor
from vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

POOL_BYTES = 16 << 30
NUM_PAGES = 1024  # spread over the pool, as a compaction pass does
ROUNDS = 2
BLOCK_BYTES = 1 << 20
GPU_BLOCKS = 8
MAX_STALL_S = 0.5
SYS_MOVE_PAGES = 279  # x86_64
MPOL_MF_MOVE = 2


def _numa_nodes() -> int:
    try:
        return len(
            [d for d in os.listdir("/sys/devices/system/node") if d.startswith("node")]
        )
    except OSError:
        return 0


pytestmark = [
    pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm userptr only"),
    pytest.mark.skipif(os.uname().machine != "x86_64", reason="x86_64 syscall"),
    pytest.mark.skipif(_numa_nodes() < 2, reason="needs two NUMA nodes"),
]


def _transfer(backend, src: list[int], dst: list[int], *, is_store: bool) -> None:
    events: list[tuple[int, torch.Event]] = []
    backend.launch_copy(src, dst, is_store, 0, events)
    deadline = time.monotonic() + 10.0
    while not events and time.monotonic() < deadline:
        time.sleep(0.0005)
    assert events, "background copy was never enqueued"
    events[0][1].synchronize()


def _move_pages(addrs: list[int], nodes: list[int] | None) -> list[int]:
    """`move_pages(2)` of our own pages to `nodes` (None: query); per-page
    status (the node, or -errno)."""
    libc = ctypes.CDLL(None, use_errno=True)
    n = len(addrs)
    pages = (ctypes.c_void_p * n)(*addrs)
    status = (ctypes.c_int * n)()
    to = None if nodes is None else (ctypes.c_int * n)(*nodes)
    rc = libc.syscall(SYS_MOVE_PAGES, 0, n, pages, to, status, MPOL_MF_MOVE)
    if rc < 0:
        pytest.skip(f"move_pages failed: errno {ctypes.get_errno()}")
    return list(status)


def _spread(base: int, nbytes: int, count: int) -> list[int]:
    page = os.sysconf("SC_PAGE_SIZE")
    span = nbytes // page
    step = max(1, span // count)
    return [base + (i * step + 17) % span * page for i in range(min(count, span))]


def _longest_stall_while_migrating(addrs: list[int]) -> tuple[float, int]:
    """Migrate `addrs` to the other NUMA node ROUNDS times while the GPU runs
    small timed steps; the longest step around the migrations and the number
    of pages moved."""
    a = torch.ones(1024, device="cuda")
    steps: list[tuple[float, float]] = []
    stop = threading.Event()

    def gpu_loop():
        while not stop.is_set():
            t = time.monotonic()
            a.add_(1)
            torch.accelerator.synchronize()
            steps.append((t, time.monotonic() - t))

    thread = threading.Thread(target=gpu_loop, daemon=True)
    thread.start()
    time.sleep(1.0)
    longest, moved = 0.0, 0
    try:
        for _ in range(ROUNDS):
            before = _move_pages(addrs, None)
            t0 = time.monotonic()
            _move_pages(addrs, [1 - max(b, 0) for b in before])
            time.sleep(3.0)
            # A step that started after the window has finished: none in it is
            # still pending.
            t1, deadline = time.monotonic(), time.monotonic() + 120
            while not any(t >= t1 for t, _ in steps) and time.monotonic() < deadline:
                time.sleep(0.05)
            after = _move_pages(addrs, None)
            moved += sum(1 for b, x in zip(before, after) if b != x)
            longest = max([longest] + [d for t, d in steps if t + d >= t0 and t < t1])
    finally:
        stop.set()
        thread.join(timeout=120)
    return longest, moved


def test_migrated_cpu_kv_cache_pages_do_not_stall_gpu():
    names = ["layer.0"]
    raw = (torch.arange(GPU_BLOCKS * BLOCK_BYTES, device="cuda") % 101).to(torch.int8)
    config = KVCacheConfig(
        num_blocks=GPU_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(
                size=raw.nbytes,
                layers=names,
                layer_stride=GPU_BLOCKS * BLOCK_BYTES,
                block_stride=BLOCK_BYTES,
            )
        ],
        kv_cache_groups=[],
    )
    worker = SimpleCPUOffloadWorker(None, config, POOL_BYTES)
    try:
        worker.register_kv_caches({"layer.0": raw.view(GPU_BLOCKS, BLOCK_BYTES)})
        backend = worker._backend
        assert isinstance(backend, DmaCopyBackend) and backend._staging is not None
        assert worker.cpu_kv_caches is not None
        pool = next(iter(worker.cpu_kv_caches.values()))
        assert pool.nbytes >= POOL_BYTES // 2
        # Exercise both directions first, so every GPU mapping exists.
        blocks = list(range(GPU_BLOCKS))
        cpu_blocks = [worker.num_cpu_blocks - 1 - b for b in blocks]
        expected = raw.clone()
        _transfer(backend, blocks, cpu_blocks, is_store=True)
        raw.zero_()
        _transfer(backend, cpu_blocks, blocks, is_store=False)
        torch.accelerator.synchronize()
        assert torch.equal(raw, expected)

        stage = backend._staging.halves[0]["layer.0"]
        stage_bytes = 2 * backend._staging.slots * BLOCK_BYTES
        for what, addrs in (
            ("the CPU KV cache", _spread(pool.data_ptr(), pool.nbytes, NUM_PAGES)),
            ("the staging buffer", _spread(stage.data_ptr(), stage_bytes, NUM_PAGES)),
        ):
            longest, moved = _longest_stall_while_migrating(addrs)
            if not moved:
                pytest.skip("move_pages moved no page")
            assert longest < MAX_STALL_S, (
                f"migrating {len(addrs)} pages of {what} stalled the GPU for "
                f"{longest:.2f} s (KFD revalidating GPU-visible host memory)"
            )
    finally:
        worker.shutdown()
