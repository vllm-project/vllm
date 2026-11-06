# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-completed PP progress, separate from posted communication work."""

import json
from collections import deque
from dataclasses import asdict, dataclass
from time import monotonic
from typing import Any

from vllm.v1.worker.pp.boundary import PPBalanceSample, PPBoundary

FEEDBACK_HISTORY = 64
FEEDBACK_KEY = "vllm_dynamic_pp_consumer"


@dataclass(frozen=True)
class PPStageSample:
    batch_id: int
    boundary: int
    version: int
    num_tokens: int
    num_reqs: int
    elapsed_ms: float


@dataclass
class _DeviceStep:
    header: PPBoundary
    num_tokens: int
    num_reqs: int
    start: Any
    end: Any
    started: bool = False


class PPDeviceFeedback:
    """Poll events without synchronizing the device or adding a PP rendezvous.

    Only TP0 participates. PP1 overwrites one small CPU-store snapshot, so an
    idle/stopped producer cannot leave unmatched receives or growing queues.
    All event times are differences on the *same* device; clocks on different
    stages do not need to be synchronized.
    """

    def __init__(
        self, device_module: Any, store: Any, pp_rank: int, timeout_seconds: int
    ):
        self.device_module = device_module
        self.store = store
        self.pp_rank = pp_rank
        self.timeout_seconds = timeout_seconds
        self._steps: deque[_DeviceStep] = deque()
        self._active: _DeviceStep | None = None
        self._samples: deque[PPStageSample] = deque(maxlen=FEEDBACK_HISTORY)
        self._remote_samples: list[PPStageSample] = []
        self._sequence = 0
        self._remote_sequence = -1
        self._last_remote_update: float | None = None
        self._last_paired_batch = -1
        self.num_paired_samples = 0
        self.ready_batch_id = -1
        self.consumed_batch_id = -1
        if pp_rank == 1:
            self._publish()

    def begin(self, header: PPBoundary, num_tokens: int, num_reqs: int) -> None:
        if self._active is not None:
            raise RuntimeError("Dynamic PP timing step was not finished")
        self.poll()
        self._active = _DeviceStep(
            header,
            num_tokens,
            num_reqs,
            self.device_module.Event(enable_timing=True),
            self.device_module.Event(enable_timing=True),
        )

    def forward_start(self, *args: Any) -> None:
        """Model pre-hook: on PP1 this follows receive and input copies."""
        if self._active is not None:
            self._active.start.record()
            self._active.started = True

    def finish(self) -> None:
        """PP0: after output snapshot; PP1: after logits and sampling."""
        step = self._active
        if step is None:
            return
        if not step.started:
            raise RuntimeError("Dynamic PP model start event was not recorded")
        step.end.record()
        self._steps.append(step)
        self._active = None
        self.poll()

    def _publish(self) -> None:
        self.store.set(
            FEEDBACK_KEY,
            json.dumps(
                {
                    "sequence": self._sequence,
                    "consumed_batch_id": self.consumed_batch_id,
                    "samples": [asdict(sample) for sample in self._samples],
                }
            ),
        )

    def poll(self) -> None:
        changed = False
        while self._steps:
            step = self._steps[0]
            if (
                self.pp_rank == 1
                and step.start.query()
                and step.header.batch_id > self.consumed_batch_id
            ):
                self.consumed_batch_id = step.header.batch_id
                changed = True
            if not step.end.query():
                break
            self._steps.popleft()
            if self.pp_rank == 0:
                self.ready_batch_id = step.header.batch_id
            self._samples.append(
                PPStageSample(
                    step.header.batch_id,
                    step.header.boundary,
                    step.header.version,
                    step.num_tokens,
                    step.num_reqs,
                    step.start.elapsed_time(step.end),
                )
            )
            changed = True
        if self.pp_rank == 1:
            if changed:
                self._sequence += 1
                self._publish()
            return
        # check() never waits for the other stage to create the key. get() is
        # safe after it: this single-writer key is overwritten, never deleted.
        if not self.store.check([FEEDBACK_KEY]):
            return
        payload = json.loads(self.store.get(FEEDBACK_KEY))
        if payload["sequence"] > self._remote_sequence:
            self._remote_sequence = payload["sequence"]
            self._last_remote_update = monotonic()
            self.consumed_batch_id = payload["consumed_batch_id"]
            self._remote_samples = [
                PPStageSample(**sample) for sample in payload["samples"]
            ]

    @property
    def queue_depth(self) -> int:
        # A newer consumer credit can race the local producer event query.
        return max(0, self.ready_batch_id - self.consumed_batch_id)

    @property
    def fresh(self) -> bool:
        return (
            self._last_remote_update is not None
            and monotonic() - self._last_remote_update <= self.timeout_seconds
        )

    def take_samples(self) -> list[PPBalanceSample]:
        local = {sample.batch_id: sample for sample in self._samples}
        paired = []
        for remote in self._remote_samples:
            if (
                remote.batch_id <= self._last_paired_batch
                or remote.batch_id not in local
            ):
                continue
            producer = local[remote.batch_id]
            if (
                producer.boundary,
                producer.version,
                producer.num_tokens,
                producer.num_reqs,
            ) != (
                remote.boundary,
                remote.version,
                remote.num_tokens,
                remote.num_reqs,
            ):
                raise ValueError(
                    "Dynamic PP stage feedback does not match its batch header/workload"
                )
            paired.append(
                PPBalanceSample(
                    producer.batch_id,
                    producer.boundary,
                    producer.version,
                    producer.num_tokens,
                    producer.num_reqs,
                    producer.elapsed_ms,
                    remote.elapsed_ms,
                )
            )
            self._last_paired_batch = remote.batch_id
            self.num_paired_samples += 1
        return paired

    def stats(self) -> dict[str, Any]:
        return {
            "ready_queue_depth": self.queue_depth if self.pp_rank == 0 else None,
            "ready_batch_id": self.ready_batch_id if self.pp_rank == 0 else None,
            "consumed_batch_id": self.consumed_batch_id,
            "feedback_fresh": self.fresh if self.pp_rank == 0 else None,
            "feedback_age_seconds": round(monotonic() - self._last_remote_update, 3)
            if self._last_remote_update is not None
            else None,
            "completed_timing_samples": len(self._samples),
            "paired_timing_samples": self.num_paired_samples,
            "pending_device_events": len(self._steps),
        }
