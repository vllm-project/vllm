# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Linux CUDA process checkpoint calls run in a separate interpreter."""

import ctypes as C
import time
from collections.abc import Callable, Sequence
from typing import Any, Literal, TextIO, cast

CheckpointOperation = Literal["Lock", "Checkpoint", "Restore", "Unlock"]
_OPERATIONS: tuple[CheckpointOperation, ...] = (
    "Lock",
    "Checkpoint",
    "Restore",
    "Unlock",
)


def cycle(
    pids,
    invoke: Callable[[int, CheckpointOperation], Any],
    boundary: Callable[[CheckpointOperation], None] = lambda op: None,
):
    if not pids or len(set(pids)) != len(pids):
        raise ValueError("Expected distinct target PIDs")
    for op in _OPERATIONS:
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
        for op in _OPERATIONS:
            getattr(self.lib, "cuCheckpointProcess" + op).argtypes = [
                C.c_int,
                C.c_void_p,
            ]

    def state(self, pid):
        state = C.c_int(-1)
        rc = self.lib.cuCheckpointProcessGetState(pid, C.byref(state))
        return {"rc": rc, "value": state.value}

    def invoke(self, pid, op: CheckpointOperation):
        """Return a result without writing to the helper protocol stream."""
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
        return dict(
            kind="control",
            pid=pid,
            op=op,
            rc=rc,
            error=name.value.decode() if name.value else None,
            before=before,
            after=after,
            time_s=elapsed,
        )

    def __call__(self, pid, op: CheckpointOperation):
        row = self.invoke(pid, op)
        self.emit(**row)
        if row["rc"] or row["after"] != {
            "rc": 0,
            "value": {"Lock": 1, "Checkpoint": 2, "Restore": 1, "Unlock": 0}[op],
        }:
            raise RuntimeError(str(row))


def _run_parallel_stage(pids, op: CheckpointOperation, invoke):
    """Run one stage in the helper; never emit partial protocol responses."""
    from concurrent.futures import ThreadPoolExecutor

    def run(pid):
        try:
            return invoke(pid, op)
        except Exception as exc:
            return {"pid": pid, "op": op, "error": repr(exc)}

    # Waiting threads live only in the helper. The parent's stage deadline
    # terminates this process if any driver call blocks.
    with ThreadPoolExecutor(max_workers=len(pids)) as pool:
        return list(pool.map(run, pids))


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

    def __call__(self, pid, op: CheckpointOperation):
        self.run_stage([pid], op)

    def run_stage(self, pids, op: CheckpointOperation):
        import json

        for pid in pids:
            self.emit(kind="control_start", pid=pid, op=op)
        cast(TextIO, self.proc.stdin).write(json.dumps([pids, op]) + "\n")
        cast(TextIO, self.proc.stdin).flush()
        try:
            response = self._read(30)
        except Exception as exc:
            self.emit(kind="control_failed", pids=pids, op=op, error=repr(exc))
            raise
        if not isinstance(response, dict):
            raise RuntimeError(f"Invalid CUDA checkpoint stage response: {response}")
        rows = response.get("results", [])
        if (
            response.get("kind") != "stage"
            or response.get("op") != op
            or not isinstance(rows, list)
            or len(rows) != len(pids)
            or any(
                not isinstance(row, dict)
                or row.get("pid") != pid
                or row.get("op") != op
                for pid, row in zip(pids, rows)
            )
        ):
            raise RuntimeError(f"Invalid CUDA checkpoint stage response: {response}")
        for row in rows:
            self.emit(**row)
        failed = [
            row
            for row in rows
            if row.get("rc") != 0
            or row.get("after")
            != {
                "rc": 0,
                "value": {"Lock": 1, "Checkpoint": 2, "Restore": 1, "Unlock": 0}[op],
            }
        ]
        if failed:
            raise RuntimeError(str(failed))

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.communicate(timeout=1)


class CudaProcessCheckpoint:
    """Suspend all local worker processes, or resume all before returning."""

    def __init__(self, emit):
        self.emit = emit
        self.driver = Driver(emit)
        self.pids = []
        self.failed = False

    def _run(self, operations: Sequence[CheckpointOperation]):
        if self.failed:
            raise RuntimeError("CUDA process checkpoint failed; restart required")
        try:
            for op in operations:
                self.driver.run_stage(self.pids, op)
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
            pids, op = json.loads(line)
            if isinstance(pids, int):
                driver(pids, op)
            else:
                output(
                    kind="stage",
                    op=op,
                    results=_run_parallel_stage(pids, op, driver.invoke),
                )
        except Exception:
            break
