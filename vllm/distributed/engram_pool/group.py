# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup rendezvous of every GPU that shares a pool's tables."""

import os
from datetime import timedelta
from functools import cache

import torch


class PeerGroup:
    """GPUs of several engines that split each table over their memory.

    They meet only at startup, through a file store; lookups then read peers'
    rows directly, so no engine's step ever waits on another's. Every GPU
    builds and loads the same tables in the same order, so its n-th exchange
    is every peer's n-th.
    """

    # Engines load their weights at their own pace before the group can serve.
    TIMEOUT = timedelta(minutes=30)

    def __init__(self, size: int, path: str) -> None:
        self.world_size = size
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.store = torch.distributed.FileStore(path, size)
        self.store.set_timeout(self.TIMEOUT)
        # Any assignment works: a rank only decides which rows a GPU holds.
        self.rank_in_group = self.store.add("next_rank", 1) - 1
        if self.rank_in_group >= size:
            raise RuntimeError(
                f"More than {size} GPUs joined the Engram peer group at {path}; "
                "check peer_group_size and use a fresh store path per launch"
            )
        self._step = 0

    def all_gather(self, value: bytes) -> list[bytes]:
        """Every GPU's value for the next exchange in rank order, once all have
        set theirs."""
        self._step += 1
        key = f"step{self._step}"
        self.store.set(f"{key}/{self.rank_in_group}", value)
        keys = [f"{key}/{rank}" for rank in range(self.world_size)]
        try:
            self.store.wait(keys)
        except RuntimeError as exc:
            missing = [
                rank for rank, name in enumerate(keys) if not self.store.check([name])
            ]
            raise TimeoutError(
                f"Engram peer GPUs {missing} did not reach startup step "
                f"{self._step} within {self.TIMEOUT}; every engine of the group "
                "must start, and none may restart alone"
            ) from exc
        return [self.store.get(name) for name in keys]

    def barrier(self) -> None:
        self.all_gather(b"")


@cache
def join_peer_group(size: int, path: str) -> PeerGroup:
    """This GPU's membership, shared by all of its process's tables."""
    return PeerGroup(size, path)
