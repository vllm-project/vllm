# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run CPU EPLB planning over a trusted, inherited socket to a child interpreter."""

import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
import weakref
from contextlib import suppress

import cloudpickle
import torch

from .policy import AbstractEplbPolicy, EplbStats


class EplbPolicyProcessError(RuntimeError):
    """A planner failure that must be reported by every inference rank."""


def _shutdown(
    connection: socket.socket, process: subprocess.Popen, death_writer: int
) -> None:
    os.close(death_writer)
    try:
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    finally:
        with suppress(OSError):
            connection.shutdown(socket.SHUT_RDWR)
        connection.close()


def _send(connection: socket.socket, payload: bytes, deadline: float | None) -> None:
    remaining = None if deadline is None else deadline - time.monotonic()
    if remaining is not None and remaining <= 0:
        raise TimeoutError("EPLB policy process timed out")
    connection.settimeout(remaining)
    connection.sendall(struct.pack("!Q", len(payload)) + payload)


def _receive(connection: socket.socket, deadline: float | None) -> bytes:
    def receive_exactly(size: int) -> bytes:
        result = bytearray()
        while len(result) < size:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError("EPLB policy process timed out")
            connection.settimeout(remaining)
            chunk = connection.recv(min(size - len(result), 1024 * 1024))
            if not chunk:
                raise EOFError("EPLB policy process disconnected")
            result.extend(chunk)
        return bytes(result)

    size = struct.unpack("!Q", receive_exactly(8))[0]
    return receive_exactly(size)


class EplbPolicyProcess:
    """Own one CPU policy process, called serially by the async EPLB thread."""

    def __init__(self, policy: AbstractEplbPolicy, timeout: float = 60):
        try:
            payload = cloudpickle.dumps(policy)
        except Exception as exc:
            raise TypeError(
                f"Async EPLB policy {type(policy).__qualname__} must be serializable. "
                "Initialize private resources lazily or use "
                "__getstate__/__setstate__ or __reduce__ to recreate them in the "
                "policy subprocess."
            ) from exc
        self._timeout = timeout
        self._connection, child = socket.socketpair()
        death_reader, death_writer = os.pipe()
        env = os.environ.copy()
        paths = [p or os.getcwd() for p in sys.path]
        if paths:
            env["PYTHONPATH"] = os.pathsep.join(paths)
        else:
            env.pop("PYTHONPATH", None)
        env["VLLM_TARGET_DEVICE"] = "cpu"
        env["VLLM_PLUGINS"] = ""
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[name] = "1"
        try:
            # Executor workers are daemonic multiprocessing processes.
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-P",
                    "-m",
                    __name__,
                    str(child.fileno()),
                    str(death_reader),
                ],
                pass_fds=(child.fileno(), death_reader),
                env=env,
            )
        except BaseException:
            self._connection.close()
            os.close(death_writer)
            raise
        finally:
            child.close()
            os.close(death_reader)
        self._finalizer = weakref.finalize(
            self, _shutdown, self._connection, self._process, death_writer
        )
        try:
            self._request(payload)
        except BaseException:
            self.close()
            raise

    def _request(self, payload: bytes):
        deadline = time.monotonic() + self._timeout
        try:
            _send(self._connection, payload, deadline)
            result, error = cloudpickle.loads(_receive(self._connection, deadline))
            if error is not None:
                raise RuntimeError(f"EPLB policy process failed:\n{error}")
            return result
        except TimeoutError as exc:
            raise TimeoutError("EPLB policy process timed out") from exc
        except (EOFError, OSError) as exc:
            raise RuntimeError("EPLB policy process disconnected") from exc

    def rebalance_experts(
        self,
        stats: EplbStats,
        old_global_expert_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the CPU placement for this immutable planning snapshot."""
        try:
            return self._request(cloudpickle.dumps((stats, old_global_expert_indices)))
        except Exception as exc:
            self.close()
            raise EplbPolicyProcessError(str(exc)) from exc

    def close(self) -> None:
        """Close IPC and reap the child, including an unresponsive policy."""
        self._finalizer()


def _main(fd: int, death_reader: int) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    def monitor_parent() -> None:
        os.read(death_reader, 1)
        os._exit(0)

    threading.Thread(target=monitor_parent, daemon=True).start()
    torch.set_num_threads(1)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    with socket.socket(fileno=fd) as connection:
        try:
            policy = cloudpickle.loads(_receive(connection, None))
            _send(connection, cloudpickle.dumps((None, None)), None)
            while True:
                stats, old_map = cloudpickle.loads(_receive(connection, None))
                result = policy.rebalance_experts(stats, old_map)
                if result.device.type != "cpu":
                    raise ValueError("EPLB policy must return a CPU tensor")
                _send(connection, cloudpickle.dumps((result, None)), None)
        except (EOFError, ConnectionError):
            return
        except Exception:
            with suppress(OSError):
                _send(
                    connection,
                    cloudpickle.dumps((None, traceback.format_exc())),
                    None,
                )


if __name__ == "__main__":
    _main(int(sys.argv[1]), int(sys.argv[2]))
