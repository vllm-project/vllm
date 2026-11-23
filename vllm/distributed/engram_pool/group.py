# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup rendezvous of every GPU that shares a pool's tables."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from functools import cache

import torch

import vllm.envs as envs


class PeerGroup:
    """GPUs of several engines that split each table over their memory. They
    meet only at startup (a file store), making the same exchanges in order."""

    def __init__(self, size: int, path: str) -> None:
        self.world_size = size
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.store = torch.distributed.FileStore(path, size)
        # Peers load weights at their own pace; their engines wait as long.
        self.store.set_timeout(timedelta(seconds=envs.VLLM_ENGINE_READY_TIMEOUT_S))
        # Any assignment works: a rank only decides which rows a GPU holds.
        self.rank_in_group = self.store.add("next_rank", 1) - 1
        if self.rank_in_group >= size:
            raise RuntimeError(
                f"More than {size} GPUs joined the Engram peer group at {path}; "
                "check peer_group_size and use a fresh store path per launch"
            )
        self._step = 0

    def all_gather(self, value: bytes) -> list[bytes]:
        """Every GPU's `value`, in rank order. Collective."""
        self._step += 1
        keys = [f"{self._step}/{rank}" for rank in range(self.world_size)]
        self.store.set(keys[self.rank_in_group], value)
        try:
            self.store.wait(keys)
        except RuntimeError as e:
            missing = [r for r, key in enumerate(keys) if not self.store.check([key])]
            raise TimeoutError(
                f"Engram peer GPUs {missing} did not reach startup step "
                f"{self._step}; every engine of the group must start (see "
                "VLLM_ENGINE_READY_TIMEOUT_S), and none may restart alone"
            ) from e
        return self.store.multi_get(keys)

    @contextmanager
    def startup_stage(self, operation: str) -> Iterator[None]:
        """Report rank-local failures before any peer enters the next collective."""
        error = b""
        try:
            yield
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}".encode()
        failures = [
            f"rank {rank}: {message.decode()}"
            for rank, message in enumerate(self.all_gather(error))
            if message
        ]
        if failures:
            raise RuntimeError(f"{operation} failed: " + "; ".join(failures))

    def barrier(self) -> None:
        self.all_gather(b"")


@cache
def join_peer_group(size: int, path: str) -> PeerGroup:
    """This GPU's membership, shared by all of its process's tables."""
    return PeerGroup(size, path)
