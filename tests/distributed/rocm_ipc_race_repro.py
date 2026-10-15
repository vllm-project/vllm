# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone HIP IPC export and attach race repro (no vLLM imports).

Everything measured so far exports and never attaches, and none of it
reproduces the MI355 failure: 41216 exports with no loss, and no limit below
64815 exports alive at once. What no probe has driven is the one thing an
elastic EP worker does that these do not, which is export a buffer while a peer
process attaches that same buffer. ROCm/rocm-systems#11892 reports a race
between hsa_amd_ipc_memory_create and hsa_amd_ipc_memory_attach on one buffer
object, and the failure we chase is intermittent, per rank, and recovers, which
is what a race looks like and not what exhaustion looks like.

Four processes, one GPU each, export a batch of buffers and publish the
handles. From then on every rank re-exports its own buffers in a tight loop
while the other three attach and detach those same buffers, with no barrier
inside the loop so the calls genuinely overlap. Every refusal is counted and
attributed to the call that made it.
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
UNCACHED = 0x3
HANDLE_BYTES = 64
LAZY_ENABLE_PEER_ACCESS = 0x1


class _IpcHandle(ctypes.Structure):
    # c_ubyte and not c_char: reading a c_char array field gives back the bytes
    # up to the first NUL, which silently truncates a handle to a few bytes and
    # makes every attach refuse with hipErrorInvalidValue.
    _fields_ = [("reserved", ctypes.c_ubyte * HANDLE_BYTES)]


def _handle_bytes(handle: _IpcHandle) -> bytes:
    return bytes(bytearray(handle.reserved))


def _handle_from(raw: bytes) -> _IpcHandle:
    handle = _IpcHandle()
    ctypes.memmove(ctypes.byref(handle), raw, HANDLE_BYTES)
    return handle


def _lib():
    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipExtMallocWithFlags.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    hip.hipIpcGetMemHandle.argtypes = [
        ctypes.POINTER(_IpcHandle),
        ctypes.c_void_p,
    ]
    hip.hipIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        _IpcHandle,
        ctypes.c_uint,
    ]
    hip.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
    return hip


class _Tally:
    """Counts attempts and refusals, and remembers the first code seen."""

    def __init__(self, name: str):
        self.name = name
        self.tried = 0
        self.failed = 0
        self.first_rc = 0

    def record(self, rc: int) -> None:
        self.tried += 1
        if rc != 0:
            self.failed += 1
            if self.first_rc == 0:
                self.first_rc = rc

    def __str__(self) -> str:
        if not self.failed:
            return f"{self.name} {self.tried} ok"
        pct = 100.0 * self.failed / self.tried
        return (
            f"{self.name} {self.failed}/{self.tried} refused ({pct:.2f}%),"
            f" first rc={self.first_rc}"
        )


def _allocate(hip, count: int) -> list[ctypes.c_void_p]:
    ptrs: list[ctypes.c_void_p] = []
    for _ in range(count):
        ptr = ctypes.c_void_p()
        rc = hip.hipExtMallocWithFlags(
            ctypes.byref(ptr), ctypes.c_size_t(SIZE), ctypes.c_uint(UNCACHED)
        )
        hip.hipGetLastError()
        if rc != 0:
            break
        ptrs.append(ptr)
    return ptrs


def _export_all(hip, ptrs, tally: _Tally) -> list[bytes]:
    """Export every buffer and return the handles that came out."""
    out: list[bytes] = []
    for ptr in ptrs:
        handle = _IpcHandle()
        rc = hip.hipIpcGetMemHandle(ctypes.byref(handle), ptr)
        hip.hipGetLastError()
        tally.record(rc)
        if rc == 0:
            out.append(_handle_bytes(handle))
    return out


def _attach_all(hip, handles: list[bytes], tally: _Tally) -> None:
    """Attach and detach every handle a peer published."""
    for raw in handles:
        mapped = ctypes.c_void_p()
        rc = hip.hipIpcOpenMemHandle(
            ctypes.byref(mapped),
            _handle_from(raw),
            ctypes.c_uint(LAZY_ENABLE_PEER_ACCESS),
        )
        hip.hipGetLastError()
        tally.record(rc)
        if rc == 0:
            hip.hipIpcCloseMemHandle(mapped)
            hip.hipGetLastError()


def _worker(rank: int, rendezvous: str, rounds: int, spins: int, buffers: int) -> None:
    try:
        torch.accelerator.set_device_index(rank)
        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            init_method=f"file://{rendezvous}",
            rank=rank,
            world_size=WORLD,
            timeout=timedelta(seconds=300),
        )
        hip = _lib()
        torch.zeros(1, device="cuda")

        exports = _Tally("export")
        attaches = _Tally("attach")

        for rnd in range(rounds):
            ptrs = _allocate(hip, buffers)
            if len(ptrs) < buffers:
                print(
                    f"[RACE] rank={rank} round {rnd}: only {len(ptrs)} of"
                    f" {buffers} buffers allocated, continuing with those",
                    flush=True,
                )

            published = _export_all(hip, ptrs, exports)
            everyone: list[list[bytes]] = [[] for _ in range(WORLD)]
            dist.all_gather_object(everyone, published)
            # Own handles cannot be attached in the process that exported them.
            peers = [h for r, hs in enumerate(everyone) if r != rank for h in hs]

            # No barrier from here to the end of the loop. Each rank runs at
            # its own speed, so a rank is exporting its buffers while the other
            # three are attaching those same buffers.
            for _ in range(spins):
                _export_all(hip, ptrs, exports)
                _attach_all(hip, peers, attaches)

            for ptr in ptrs:
                hip.hipFree(ptr)
            hip.hipGetLastError()
            print(
                f"[RACE] rank={rank} round {rnd}: {exports}, {attaches}",
                flush=True,
            )
            dist.barrier()

        failures = exports.failed + attaches.failed
        gathered: list[int] = [0] * WORLD
        dist.all_gather_object(gathered, failures)
        if rank == 0:
            total = sum(gathered)
            verdict = "REPRODUCED" if total else "no refusal"
            print(
                f"[RACE] SUMMARY {verdict}, refusals per rank {gathered},"
                f" {total} in total",
                flush=True,
            )
        dist.destroy_process_group()
        sys.exit(1 if failures else 0)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--spins", type=int, default=50)
    ap.add_argument("--buffers", type=int, default=16)
    args = ap.parse_args()

    if torch.accelerator.device_count() < WORLD:
        print(f"[RACE] need {WORLD} GPUs, skipping", flush=True)
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        rendezvous = os.path.join(tmp, "rendezvous")
        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(
                target=_worker,
                args=(r, rendezvous, args.rounds, args.spins, args.buffers),
            )
            for r in range(WORLD)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        codes = [p.exitcode for p in procs]
        print(f"[RACE] SUMMARY exit codes {codes}", flush=True)
        return 1 if any(codes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
