# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run CPU EPLB planning over a trusted, inherited pipe to a child interpreter."""

import os
import subprocess
import sys
import traceback
import weakref
from multiprocessing import Pipe
from multiprocessing.connection import Connection

import cloudpickle
import torch

from .policy import AbstractEplbPolicy, EplbStats


class EplbPolicyProcessError(RuntimeError):
    """A planner failure that must be reported by every inference rank."""


def _shutdown(connection: Connection, process: subprocess.Popen) -> None:
    connection.close()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


class EplbPolicyProcess:
    """Own one CPU policy process, called serially by the async EPLB thread."""

    def __init__(self, policy: AbstractEplbPolicy, timeout: float = 60):
        self._timeout = timeout
        self._connection, child = Pipe()
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[name] = "1"
        try:
            # Executor workers are daemonic multiprocessing processes.
            self._process = subprocess.Popen(
                [sys.executable, "-m", __name__, str(child.fileno())],
                pass_fds=(child.fileno(),),
                env=env,
            )
        except BaseException:
            self._connection.close()
            raise
        finally:
            child.close()
        self._finalizer = weakref.finalize(
            self, _shutdown, self._connection, self._process
        )
        try:
            self._connection.send_bytes(cloudpickle.dumps(policy))
            self._receive()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        try:
            if not self._connection.poll(self._timeout):
                raise TimeoutError("EPLB policy process timed out")
            result, error = cloudpickle.loads(self._connection.recv_bytes())
            if error is not None:
                raise RuntimeError(f"EPLB policy process failed:\n{error}")
            return result
        except TimeoutError:
            raise
        except (EOFError, OSError) as exc:
            raise RuntimeError("EPLB policy process disconnected") from exc

    def rebalance_experts(
        self,
        stats: EplbStats,
        old_global_expert_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the CPU placement for this immutable planning snapshot."""
        try:
            self._connection.send_bytes(
                cloudpickle.dumps((stats, old_global_expert_indices))
            )
            return self._receive()
        except Exception as exc:
            self.close()
            raise EplbPolicyProcessError(str(exc)) from exc

    def close(self) -> None:
        """Close IPC and reap the child, including an unresponsive policy."""
        self._finalizer()


def _main(fd: int) -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with Connection(fd) as connection:
        try:
            policy = cloudpickle.loads(connection.recv_bytes())
            connection.send_bytes(cloudpickle.dumps((None, None)))
            while True:
                stats, old_map = cloudpickle.loads(connection.recv_bytes())
                result = policy.rebalance_experts(stats, old_map)
                if result.device.type != "cpu":
                    raise ValueError("EPLB policy must return a CPU tensor")
                connection.send_bytes(cloudpickle.dumps((result, None)))
        except (EOFError, BrokenPipeError):
            return
        except Exception:
            connection.send_bytes(cloudpickle.dumps((None, traceback.format_exc())))


if __name__ == "__main__":
    _main(int(sys.argv[1]))
