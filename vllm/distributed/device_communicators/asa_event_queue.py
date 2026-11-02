# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Local queue control; the explicit stock opt-out needs no native extension.

Payload ring, tensor arena, pickle/OOB, overflow and remote transports remain in
shm_broadcast.py.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager, suppress
from multiprocessing import shared_memory
from typing import Any

import vllm.envs as envs
from vllm.distributed.device_communicators.shm_utils import (
    attach_shared_memory,
    resolve_shm_wait_backend,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class EventQueue:
    ABI = 1
    shm: shared_memory.SharedMemory
    endpoint: Any
    creator: bool
    handle: tuple[str, int, int, int, int, int]
    spin_ns: int

    @classmethod
    def create(cls, readers: int, capacity: int):
        resolve_shm_wait_backend("futex")
        from vllm._asa_shmwait import Endpoint
        from vllm.distributed.device_communicators.shm_broadcast import (
            check_shm_free_space,
        )

        stride = envs.VLLM_ASA_SHM_READER_STRIDE
        if stride not in (16, 128):
            raise ValueError("VLLM_ASA_SHM_READER_STRIDE must be 16 or 128")
        spin_ns = envs.VLLM_ASA_SHM_SPIN_NS
        if not 0 <= spin_ns <= 1_000_000:
            raise ValueError("VLLM_ASA_SHM_SPIN_NS must be 0..1000000")
        control_bytes = 128 + readers * stride
        check_shm_free_space(control_bytes)
        shm = shared_memory.SharedMemory(create=True, size=control_bytes)
        try:
            endpoint = Endpoint(shm.buf, readers, capacity, -1, True, stride)
        except BaseException:
            shm.close()
            shm.unlink()
            raise
        self = cls()
        self.shm, self.endpoint, self.creator = shm, endpoint, True
        self.handle = (shm.name, readers, capacity, stride, spin_ns, cls.ABI)
        self.spin_ns = spin_ns
        return self

    @classmethod
    def attach(cls, handle, rank: int):
        # Follow the writer's resolved backend, never the reader's environment.
        resolve_shm_wait_backend("futex")
        from vllm._asa_shmwait import Endpoint

        name, readers, capacity, stride, spin_ns, abi = handle
        if abi != cls.ABI:
            raise ValueError("event queue ABI mismatch")
        shm = attach_shared_memory(name)
        try:
            endpoint = Endpoint(shm.buf, readers, capacity, rank, False, stride)
        except BaseException:
            shm.close()
            raise
        self = cls()
        self.shm, self.endpoint, self.creator = shm, endpoint, False
        self.handle, self.spin_ns = handle, spin_ns
        return self

    def cancel(self):
        self.endpoint.cancel()

    @contextmanager
    def acquire(
        self, ring, timeout, *, writer: bool, wait_timeout_ms=None, on_timeout=None
    ):
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("queue timeout must be finite and nonnegative")
        if wait_timeout_ms is not None and not callable(wait_timeout_ms):
            raise TypeError("wait_timeout_ms must be callable")
        if on_timeout is not None and not callable(on_timeout):
            raise TypeError("on_timeout must be callable")
        if wait_timeout_ms is None or timeout == 0:
            index = self.endpoint.acquire(timeout=timeout, spin_ns=self.spin_ns)
        else:
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                wait_ms = wait_timeout_ms()
                if wait_ms is not None and (
                    isinstance(wait_ms, bool)
                    or not isinstance(wait_ms, (int, float))
                    or not math.isfinite(wait_ms)
                    or wait_ms <= 0
                ):
                    raise ValueError(
                        "wait policy must return positive milliseconds or None"
                    )
                wait_s = None if wait_ms is None else wait_ms / 1000
                if deadline is not None:
                    remaining = max(0, deadline - time.monotonic())
                    wait_s = remaining if wait_s is None else min(wait_s, remaining)
                try:
                    index = self.endpoint.acquire(timeout=wait_s, spin_ns=self.spin_ns)
                    break
                except TimeoutError:
                    if on_timeout is not None:
                        on_timeout()
                    if deadline is not None and time.monotonic() >= deadline:
                        raise
        commit = False
        try:
            with ring.get_data(index) as buf:
                yield buf
            commit = True
        finally:
            self.endpoint.release(commit if writer else True)

    def __del__(self):
        if hasattr(self, "endpoint"):
            del self.endpoint
        if hasattr(self, "shm"):
            self.shm.close()
            if self.creator:
                with suppress(FileNotFoundError):
                    self.shm.unlink()


def requested():
    mode, reason = resolve_shm_wait_backend(envs.VLLM_ASA_SHM_WAIT)
    if reason is not None:
        logger.info("Using stock MessageQueue wait backend: %s", reason)
    return mode == "futex"
