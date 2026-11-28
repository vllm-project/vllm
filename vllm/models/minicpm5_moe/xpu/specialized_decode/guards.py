# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared device health polling, adapted from the reference mono runtime."""

import torch
import torch.distributed as dist


class FailStop:
    CHECK_EVERY = 16

    def __init__(self, health, rank, world, group):
        self.failed = health[0:1]
        self.rank, self.world, self.group = rank, world, group
        self.step_failure = (
            self.failed,
            (
                f"MiniCPM5 XPU mono decode: TP rank {rank} detected a device "
                "or peer failure; discarding this step"
            ),
        )
        self.host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self.copied = None
        self.seen = False
        self.steps = 0

    def after_step(self):
        if torch.xpu.is_current_stream_capturing():
            return
        if self.copied is None or self.copied.query():
            if self.copied is not None:
                self.seen |= bool(self.host[0])
            self.host.copy_(self.failed, non_blocking=True)
            self.copied = torch.xpu.Event()
            self.copied.record()
        self.steps += 1
        if self.steps % self.CHECK_EVERY:
            return
        votes = torch.zeros(self.world, dtype=torch.int32)
        votes[self.rank] = int(self.seen)
        dist.all_reduce(votes, group=self.group)
        if votes.any():
            ranks = [r for r, v in enumerate(votes.tolist()) if v]
            raise RuntimeError(
                f"MiniCPM5 XPU mono decode: TP ranks {ranks} detected a device "
                "or peer failure; stopping the engine"
            )
