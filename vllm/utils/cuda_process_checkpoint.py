# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Linux CUDA process checkpoint calls run in a separate interpreter."""

import ctypes as C
import time
from typing import TextIO, cast


def cycle(pids, invoke, boundary=lambda op: None):
    if not pids or len(set(pids)) != len(pids):
        raise ValueError("Expected distinct target PIDs")
    for op in ("Lock", "Checkpoint", "Restore", "Unlock"):
        for pid in pids:
            invoke(pid, op)
        boundary(op)


class LockArgs(C.Structure):
    _fields_ = [
        ("timeoutMs", C.c_uint),
        ("reserved0", C.c_uint),
        ("reserved1", C.c_uint64 * 7),
    ]


class DirectDriver:
    def __init__(self, emit):
        self.emit = emit
        self.lib = C.CDLL("libcuda.so.1")
        self.lib.cuInit.argtypes = [C.c_uint]
        rc = self.lib.cuInit(0)
        if rc:
            raise RuntimeError(f"cuInit: {rc}")
        self.lib.cuCheckpointProcessGetState.argtypes = [C.c_int, C.POINTER(C.c_int)]
        self.lib.cuGetErrorName.argtypes = [C.c_int, C.POINTER(C.c_char_p)]
        for op in ("Lock", "Checkpoint", "Restore", "Unlock"):
            getattr(self.lib, "cuCheckpointProcess" + op).argtypes = [
                C.c_int,
                C.c_void_p,
            ]

    def state(self, pid):
        state = C.c_int(-1)
        rc = self.lib.cuCheckpointProcessGetState(pid, C.byref(state))
        return {"rc": rc, "value": state.value}

    def __call__(self, pid, op):
        before = self.state(pid)
        args = LockArgs(10000) if op == "Lock" else None
        start = time.perf_counter()
        rc = getattr(self.lib, "cuCheckpointProcess" + op)(
            pid, C.byref(args) if args is not None else None
        )
        elapsed = time.perf_counter() - start
        name = C.c_char_p()
        self.lib.cuGetErrorName(rc, C.byref(name))
        after = self.state(pid)
        self.emit(
            kind="control",
            pid=pid,
            op=op,
            rc=rc,
            error=name.value.decode() if name.value else None,
            before=before,
            after=after,
            time_s=elapsed,
        )
        if rc or after != {
            "rc": 0,
            "value": {"Lock": 1, "Checkpoint": 2, "Restore": 1, "Unlock": 0}[op],
        }:
            raise RuntimeError(f"{op} pid={pid} rc={rc} state={after}")


class Driver:
    """Persistent helper isolates potentially blocking driver calls."""

    def __init__(self, emit):
        import subprocess
        import sys
        from pathlib import Path

        self.emit = emit
        self.proc = subprocess.Popen(
            [sys.executable, str(Path(__file__)), "--service"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            self._read(30)
        except Exception:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.communicate(timeout=5)
            raise

    def _read(self, timeout):
        import json
        import selectors

        sel = selectors.DefaultSelector()
        try:
            sel.register(cast(TextIO, self.proc.stdout), selectors.EVENT_READ)
            if not sel.select(timeout):
                self.proc.kill()
                raise TimeoutError("CUDA controller deadline exceeded")
            line = cast(TextIO, self.proc.stdout).readline()
            if not line:
                raise RuntimeError("CUDA controller exited unexpectedly")
            return json.loads(line)
        finally:
            sel.close()

    def __call__(self, pid, op):
        import json

        self.emit(kind="control_start", pid=pid, op=op)
        cast(TextIO, self.proc.stdin).write(json.dumps([pid, op]) + "\n")
        cast(TextIO, self.proc.stdin).flush()
        try:
            row = self._read(30)
        except Exception as e:
            self.emit(kind="control_failed", pid=pid, op=op, error=repr(e))
            raise
        self.emit(**row)
        if row.get("rc") != 0 or row.get("after") != {
            "rc": 0,
            "value": {"Lock": 1, "Checkpoint": 2, "Restore": 1, "Unlock": 0}[op],
        }:
            raise RuntimeError(str(row))

    def close(self):
        cast(TextIO, self.proc.stdin).close()
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
            self.proc.wait(timeout=5)


class CudaProcessCheckpoint:
    """Suspend all local worker processes, or resume all before returning."""

    def __init__(self, emit):
        self.emit = emit
        self.driver = Driver(emit)
        self.pids = []
        self.failed = False

    def _run(self, operations):
        if self.failed:
            raise RuntimeError("CUDA process checkpoint failed; restart required")
        try:
            for op in operations:
                for pid in self.pids:
                    self.driver(pid, op)
                self.emit(kind="boundary", stage=op, pids=self.pids)
        except Exception:
            self.failed = True
            raise

    def suspend(self, pids):
        if self.pids:
            raise RuntimeError("CUDA processes are already checkpointed")
        if not pids or len(set(pids)) != len(pids):
            raise ValueError("Expected distinct worker PIDs")
        self.pids = list(pids)
        self._run(("Lock", "Checkpoint"))

    def resume(self):
        if not self.pids:
            return
        self._run(("Restore", "Unlock"))
        self.pids = []

    def close(self):
        self.driver.close()


if __name__ == "__main__":
    import json
    import sys

    def output(**kw):
        print(json.dumps(kw), flush=True)

    driver = DirectDriver(output)
    output(kind="controller_ready")
    for line in sys.stdin:
        try:
            driver(*json.loads(line))
        except Exception:
            break
