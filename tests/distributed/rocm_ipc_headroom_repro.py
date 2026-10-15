# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone HIP IPC export headroom repro (no vLLM imports).

Measures how many 6 MiB uncached buffers can be exported with
hipIpcGetMemHandle at four points in the life of a process:

  1. fresh, before any communicator exists
  2. after the first group has run a collective
  3. with a second group built but before its first collective
  4. after that first collective

On an MI355 CI agent the elastic EP test shows the headroom full at the first
point and down to roughly 115 of 256 at the third, recovering at the fourth.
This script asks the same question without vLLM, elastic EP, Ray or a model.
"""

import argparse
import ctypes
import os
import sys
import tempfile
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4
SIZE = 6 << 20
LIMIT = 256
UNCACHED = 0x3
HANDLE_BYTES = 64


def _lib():
    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipExtMallocWithFlags.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    return hip


def _ceiling(hip, size: int, limit: int) -> str:
    """How many exports fit before one is refused. Frees everything after."""
    ptrs: list[ctypes.c_void_p] = []
    reached = 0
    stopped = f"no ceiling below {limit}"
    try:
        while reached < limit:
            ptr = ctypes.c_void_p()
            rc = hip.hipExtMallocWithFlags(
                ctypes.byref(ptr), ctypes.c_size_t(size), ctypes.c_uint(UNCACHED)
            )
            if rc != 0:
                hip.hipGetLastError()
                stopped = f"malloc stopped first rc={rc}"
                break
            ptrs.append(ptr)
            handle = ctypes.create_string_buffer(HANDLE_BYTES)
            rc = hip.hipIpcGetMemHandle(handle, ptr)
            hip.hipGetLastError()
            if rc != 0:
                stopped = f"export refused rc={rc}"
                break
            reached += 1
    finally:
        for ptr in ptrs:
            hip.hipFree(ptr)
        hip.hipGetLastError()
    return f"{reached} held, {stopped}"


def _log(rank, point, msg):
    print(f"[HEADROOM] rank={rank} {point}: {msg}", flush=True)


def _worker(rank: int, rendezvous: str, rounds: int) -> None:
    try:
        torch.accelerator.set_device_index(rank)
        # A port picked by binding and closing a socket is a race, and losing
        # it costs the whole run: rank 0 failed to listen with EADDRINUSE and
        # the other three sat in a 180 second connect timeout. A file needs no
        # such window.
        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            init_method=f"file://{rendezvous}",
            rank=rank,
            world_size=WORLD,
            timeout=timedelta(seconds=180),
        )
        hip = _lib()
        torch.zeros(1, device="cuda")

        _log(rank, "1 fresh", _ceiling(hip, SIZE, LIMIT))

        t = torch.ones(1, device="cuda")
        dist.all_reduce(t)
        torch.accelerator.synchronize()
        _log(rank, "2 after first collective", _ceiling(hip, SIZE, LIMIT))

        # A second communicator over the same ranks, which is what a scale up
        # builds. Nothing is collected on it yet, so this is the point the
        # elastic EP test reaches with roughly 115 left.
        for r in range(rounds):
            extra = dist.new_group(list(range(WORLD)), backend="nccl")
            _log(rank, f"3 round {r} second group built", _ceiling(hip, SIZE, LIMIT))
            t2 = torch.ones(1, device="cuda")
            dist.all_reduce(t2, group=extra)
            torch.accelerator.synchronize()
            _log(rank, f"4 round {r} after its collective", _ceiling(hip, SIZE, LIMIT))
            dist.destroy_process_group(extra)
            _log(
                rank, f"5 round {r} second group destroyed", _ceiling(hip, SIZE, LIMIT)
            )

        dist.destroy_process_group()
    except Exception:
        traceback.print_exc()
        sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    if torch.accelerator.device_count() < WORLD:
        print(f"[HEADROOM] need {WORLD} GPUs, skipping", flush=True)
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        rendezvous = os.path.join(tmp, "rendezvous")
        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(target=_worker, args=(r, rendezvous, args.rounds))
            for r in range(WORLD)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        bad = [p.exitcode for p in procs if p.exitcode]
        print(
            f"[HEADROOM] SUMMARY exit codes {[p.exitcode for p in procs]}", flush=True
        )
        return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
