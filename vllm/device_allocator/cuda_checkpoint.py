# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Suspend CUDA process state without destroying captured graphs."""

import os
import subprocess
import sys
import tempfile
from contextlib import suppress


class CudaCheckpoint:
    def __init__(self) -> None:
        self.suspended = False
        self._helper: subprocess.Popen | None = None
        self._run("validate")

    def _run(self, operation: str) -> None:
        if self._helper is not None and self._helper.poll() is None:
            raise RuntimeError("Previous CUDA checkpoint helper has not exited")
        with tempfile.TemporaryFile(mode="w+t") as output:
            self._helper = subprocess.Popen(
                [sys.executable, __file__, operation, str(os.getpid())],
                stdout=output,
                stderr=output,
            )
            try:
                self._helper.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self._helper.kill()
                with suppress(subprocess.TimeoutExpired):
                    self._helper.wait(timeout=5)
                output.seek(0)
                raise RuntimeError(
                    f"CUDA checkpoint {operation} timed out: {output.read(4096)}"
                ) from None
            if self._helper.returncode:
                output.seek(0)
                raise RuntimeError(
                    f"CUDA checkpoint {operation} failed: {output.read()}"
                )

    def suspend(self) -> None:
        # An interrupted helper may have completed the driver operation.
        self.suspended = True
        self._run("suspend")

    def resume(self) -> None:
        if self.suspended:
            self._run("resume")
            self.suspended = False


def _main(operation: str, pid: int) -> None:
    from cuda.bindings import driver

    def check(result):
        error, *values = result
        if error != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(str(error))
        return values[0] if values else None

    check(driver.cuInit(0))
    if check(driver.cuDriverGetVersion()) < 13020:
        raise RuntimeError("CUDA process offload requires CUDA driver API 13.2+")
    state = check(driver.cuCheckpointProcessGetState(pid))
    if operation == "validate":
        return
    states = driver.CUprocessState
    if operation == "resume":
        if state == states.CU_PROCESS_STATE_CHECKPOINTED:
            check(driver.cuCheckpointProcessRestore(pid, None))
            state = states.CU_PROCESS_STATE_LOCKED
        if state == states.CU_PROCESS_STATE_LOCKED:
            check(driver.cuCheckpointProcessUnlock(pid, None))
        elif state != states.CU_PROCESS_STATE_RUNNING:
            raise RuntimeError(f"Cannot resume CUDA process in {state}")
        return
    if operation != "suspend":
        raise ValueError(operation)
    if state == states.CU_PROCESS_STATE_CHECKPOINTED:
        return
    if state == states.CU_PROCESS_STATE_RUNNING:
        args = driver.CUcheckpointLockArgs()
        args.timeoutMs = 30000
        check(driver.cuCheckpointProcessLock(pid, args))
        if (
            check(driver.cuCheckpointProcessGetState(pid))
            != states.CU_PROCESS_STATE_LOCKED
        ):
            raise RuntimeError("CUDA checkpoint lock did not reach LOCKED state")
    elif state != states.CU_PROCESS_STATE_LOCKED:
        raise RuntimeError(f"Cannot suspend CUDA process in {state}")
    check(driver.cuCheckpointProcessCheckpoint(pid, None))


if __name__ == "__main__":
    _main(sys.argv[1], int(sys.argv[2]))
