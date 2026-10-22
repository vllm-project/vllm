# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Health check of the kernels' error words and the fail-stop that follows."""

from __future__ import annotations

import collections
import faulthandler
import os

import torch

from vllm.logger import init_logger
from vllm.models.glm5next.amd.specialized_decode.mono import abi

logger = init_logger(__name__)

OUTPUT_RANK = 0
EXIT_GRACE_S = 30.0
WORKER_EXIT_CODE = 70

counters: collections.Counter[str] = collections.Counter()


def fail_stop(msg: str, rank: int) -> None:
    """Ranks other than the output rank exit their worker; the output rank raises
    and exits after a grace period with a stack dump, so no worker keeps spinning
    in a collective."""
    counters["incidents"] += 1
    if rank != OUTPUT_RANK:
        logger.critical("%s; terminating this worker", msg)
        os._exit(WORKER_EXIT_CODE)
    logger.error("%s; this worker exits in %.0f s", msg, EXIT_GRACE_S)
    faulthandler.dump_traceback_later(EXIT_GRACE_S, exit=True)
    raise RuntimeError(msg)


class HealthWatch:
    """Error words of every kernel runtime on this rank (one int32 per stage) and,
    on the output rank, every rank's expiry flag.

    ``health()`` is what the runner reads after a step. ``after_step()`` checks the
    previous step's copy without blocking the stream. Words are sticky; only
    ``VLLM_MONO_UNSAFE_WARN=1`` clears them, after logging."""

    def __init__(
        self,
        name: str,
        words: list[torch.Tensor],
        stages: tuple[str, ...],
        rank: int,
        flags: torch.Tensor | None,
    ) -> None:
        for w in words:
            assert w.dtype == torch.int32 and w.numel() == len(stages), w.shape
        self._name, self._stages, self._rank = name, stages, rank
        self._dev = list(words) + ([flags] if flags is not None else [])
        self._has_flags = flags is not None
        pin = torch.cuda.is_available()
        self._host = [
            torch.zeros(t.shape, dtype=t.dtype, pin_memory=pin) for t in self._dev
        ]
        self._event: torch.cuda.Event | None = None
        self._unsafe_warn = abi.unsafe_warn()

    def health(self) -> torch.Tensor:
        return torch.cat([t.to(torch.int32).flatten() for t in self._dev])

    def after_step(self) -> None:
        if self._event is not None and self._event.query():
            self.check([h.tolist() for h in self._host])
        for h, d in zip(self._host, self._dev):
            h.copy_(d, non_blocking=True)
        if self._event is None:
            self._event = torch.cuda.Event()
        self._event.record()

    def check(self, values: list[list[int]]) -> None:
        words = values[:-1] if self._has_flags else values
        flags = values[-1] if self._has_flags else []
        bad = sorted({self._stages[i] for w in words for i, v in enumerate(w) if v})
        peers = [r for r, v in enumerate(flags) if v]
        if not bad and not peers:
            return
        msg = (
            f"{self._name} MonoKernel rank {self._rank}: expired waits"
            f"{' in ' + ', '.join(bad) if bad else ''}"
            f"{'; ranks flagging an expiry: ' + str(peers) if peers else ''}"
            "; tokens of the affected steps are wrong"
        )
        if not self._unsafe_warn:
            fail_stop(msg, self._rank)
        counters["incidents"] += 1
        logger.error("%s [%s=1: continuing]", msg, abi.UNSAFE_WARN_ENV)
        for d in self._dev:
            d.zero_()
