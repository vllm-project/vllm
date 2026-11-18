# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared CPU stand-ins for the sharded-RDT tests."""

import gc

import torch

from vllm.distributed.weight_transfer.sharded_rdt_trainer import _RDTProducerServer


def patch_for_cpu(monkeypatch) -> None:
    """Let a producer server start without a GPU or a frozen GC."""
    monkeypatch.setattr(
        torch.accelerator, "current_device_index", lambda *a, **kw: 0, raising=False
    )
    monkeypatch.setattr(gc, "freeze", lambda: None)


def make_server(cls=None, **kw) -> _RDTProducerServer:
    """A real producer whose arena is a CPU tensor (no NIXL)."""
    server = (cls or Server)(**kw)
    server._arena = torch.zeros(server._arena_bytes, dtype=torch.uint8)
    return server


class Server(_RDTProducerServer):
    """A real producer whose frees come from the test (`free`), not from
    Ray dropping the served block."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.outstanding: list[tuple[int, ...]] = []

    def _track_free(self, flat, *key):
        self.outstanding.append(key)

    def free(self, n: int | None = None) -> None:
        """Ray freed the oldest `n` served buckets (all by default)."""
        n = len(self.outstanding) if n is None else n
        for key in self.outstanding[:n]:
            self._freed_bucket(*key)
        del self.outstanding[:n]
