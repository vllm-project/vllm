# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone HIP IPC export drain repro (no vLLM imports, single process).

On the MI355 CI agent an elastic EP worker reads 117 at 6 MiB, runs one
measurement that exports and frees about 355 buffers, and then reads 0. The
capacity never comes back, and the only pass that keeps it is the one with
HSA_DISABLE_FRAGMENT_ALLOCATOR=1.

This asks the same question with nothing else in the process: measure the
6 MiB ceiling, export and free a batch of 2 MiB buffers, measure again. On a
healthy MI355 the ceiling holds through 41216 exports in either fragment mode.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=40)
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
