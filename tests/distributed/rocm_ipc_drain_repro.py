# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone HIP IPC export probes (no vLLM imports, single process).

On the MI355 CI agent hipIpcGetMemHandle intermittently returns invalid
argument while the matching allocation succeeds, with tens of GiB free and the
file descriptor count three orders of magnitude below its limit. Two modes:

drain asks whether export capacity is spent by exporting: measure the 6 MiB
ceiling, export and free a batch of 2 MiB buffers, measure again. A healthy
MI355 holds the ceiling through 41216 exports in either fragment mode, so
export traffic alone does not reproduce the failure.

hold asks the other question, whether the limit is on exports that are alive at
the same time. It exports without freeing until a call refuses and reports
which call refused at what count. The agent readings are consistent with such a
limit: a worker whose communicators hold many live exports reads a low ceiling,
a freshly spawned one reads the full limit, and the reading recovers once a
collective releases what it held.
"""

import argparse
import ctypes
import os
import sys
import time

import torch

SIZE_SMALL = 2 << 20
SIZE_LARGE = 6 << 20
CEILING_LIMIT = 256
DRAIN_LIMIT = 768
UNCACHED = 0x3
HANDLE_BYTES = 64

# Sizes and live counts for the hold mode. The small pass reaches a high count
# on 4 GiB so that memory is never the binding constraint, the large pass uses
# the size the failing worker exports and stops well inside 288 GiB of VRAM.
HOLD_PASSES = ((64 << 10, 65536), (SIZE_LARGE, 8192))
HOLD_REPORT_EVERY = 4096


def _lib():
    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipExtMallocWithFlags.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    hip.hipMemGetInfo.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    return hip


def _free_gib(hip) -> float:
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    if hip.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total)) != 0:
        return float("nan")
    return free.value / 2**30


def _probe(hip, size: int, limit: int) -> int:
    """Export up to limit buffers, free every one, return how many went out."""
    ptrs: list[ctypes.c_void_p] = []
    reached = 0
    try:
        while reached < limit:
            ptr = ctypes.c_void_p()
            if hip.hipExtMallocWithFlags(
                ctypes.byref(ptr), ctypes.c_size_t(size), ctypes.c_uint(UNCACHED)
            ):
                hip.hipGetLastError()
                break
            ptrs.append(ptr)
            if hip.hipIpcGetMemHandle(ctypes.create_string_buffer(HANDLE_BYTES), ptr):
                hip.hipGetLastError()
                break
            reached += 1
    finally:
        for ptr in ptrs:
            hip.hipFree(ptr)
        hip.hipGetLastError()
    return reached


def _hold(hip, size: int, limit: int) -> tuple[int, str]:
    """Export up to limit buffers and keep every one alive, then free.

    Returns how many exports were live at once and why it stopped, naming the
    call that refused so an allocation limit is never read as an export limit.
    """
    ptrs: list[ctypes.c_void_p] = []
    handles: list[ctypes.Array] = []
    reached = 0
    stopped = f"no limit below {limit} live"
    try:
        while reached < limit:
            ptr = ctypes.c_void_p()
            rc = hip.hipExtMallocWithFlags(
                ctypes.byref(ptr), ctypes.c_size_t(size), ctypes.c_uint(UNCACHED)
            )
            if rc != 0:
                hip.hipGetLastError()
                stopped = f"malloc refused with {rc}"
                break
            ptrs.append(ptr)
            handle = ctypes.create_string_buffer(HANDLE_BYTES)
            rc = hip.hipIpcGetMemHandle(handle, ptr)
            if rc != 0:
                hip.hipGetLastError()
                stopped = f"export refused with {rc}"
                break
            handles.append(handle)
            reached += 1
            if reached % HOLD_REPORT_EVERY == 0:
                print(
                    f"[HOLD] {size >> 10:6d} KiB: {reached:6d} live,"
                    f" free {_free_gib(hip):.1f} GiB",
                    flush=True,
                )
    finally:
        for ptr in ptrs:
            hip.hipFree(ptr)
        hip.hipGetLastError()
    return reached, stopped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--mode", choices=("drain", "hold"), default="drain")
    args = ap.parse_args()

    if torch.accelerator.device_count() < 1:
        print("[DRAIN] no GPU, skipping", flush=True)
        return 0

    torch.accelerator.set_device_index(0)
    hip = _lib()
    torch.zeros(1, device="cuda")

    print(
        "[DRAIN] fragment_allocator_disabled="
        f"{os.getenv('HSA_DISABLE_FRAGMENT_ALLOCATOR', 'unset')}"
        f" ipc_mode_legacy={os.getenv('HSA_ENABLE_IPC_MODE_LEGACY', 'unset')}",
        flush=True,
    )

    if args.mode == "hold":
        start = time.time()
        for size, limit in HOLD_PASSES:
            reached, stopped = _hold(hip, size, limit)
            print(
                f"[HOLD] SUMMARY {size >> 10} KiB: {reached} live, {stopped},"
                f" free {_free_gib(hip):.1f} GiB after release",
                flush=True,
            )
        print(f"[HOLD] done in {time.time() - start:.1f}s", flush=True)
        return 0

    base = _probe(hip, SIZE_LARGE, CEILING_LIMIT)
    print(f"[DRAIN] baseline 6 MiB: {base}, free {_free_gib(hip):.1f} GiB", flush=True)

    total = base
    start = time.time()
    for rnd in range(args.rounds):
        drained = _probe(hip, SIZE_SMALL, DRAIN_LIMIT)
        after = _probe(hip, SIZE_LARGE, CEILING_LIMIT)
        total += drained + after
        print(
            f"[DRAIN] round {rnd:3d}: 2 MiB {drained:4d}, 6 MiB after {after:4d},"
            f" exports so far {total:6d}, free {_free_gib(hip):.1f} GiB"
            f"{'' if after == base else '  <-- DROPPED'}",
            flush=True,
        )
        if after < base:
            print(
                f"[DRAIN] SUMMARY ceiling fell from {base} to {after} after"
                f" {total} exports, in {time.time() - start:.1f}s",
                flush=True,
            )
            return 0

    print(
        f"[DRAIN] SUMMARY no loss after {total} exports in {time.time() - start:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
