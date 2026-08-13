# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import time
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.core.sched.interface import PrefillAlignmentTelemetry
from vllm.v1.engine import PrefillAlignmentObservation

logger = init_logger(__name__)

_TARGET_STEP_LEAD = 2
_MAX_DELAY_PASSES = 30
_MAX_DELAY_SECONDS = 5.0
_RESEND_INTERVAL_PASSES = 2
_RESEND_INTERVAL_SECONDS = 5.0

ReleasePayload = tuple[int, int, int]


@dataclass(frozen=True)
class PrefillAlignmentRelease:
    wave: int
    release_id: int
    target_step: int
    reason: str

    @property
    def payload(self) -> ReleasePayload:
        return self.wave, self.release_id, self.target_step


class PrefillAlignmentCoordinator:
    """Coordinate nonblocking prefill releases across DP engines."""

    def __init__(self, engine_count: int) -> None:
        self.engine_count = engine_count
        self.current_wave = 0
        self.current_release_id = 0
        self.delayed_passes = 0
        self.delay_started_at: float | None = None
        self.latest_observed_step = 0
        self.pending_release: PrefillAlignmentRelease | None = None
        self.pending_acks: set[int] = set()
        self.pending_release_sent_at = 0.0
        self.last_resend_step = -1
        self.snapshots: dict[int, dict[int, PrefillAlignmentObservation]] = {}
        self.incomplete_started_at: dict[int, float] = {}
        # This is intentionally server-lifetime state. A new request wave must
        # not restore the warmup exception.
        self.skip_first_delay = True

    def reset_wave(self, wave: int) -> None:
        self.current_wave = wave
        self.current_release_id = 0
        self.delayed_passes = 0
        self.delay_started_at = None
        self.latest_observed_step = 0
        self.pending_release = None
        self.pending_acks.clear()
        self.pending_release_sent_at = 0.0
        self.last_resend_step = -1
        self.snapshots.clear()
        self.incomplete_started_at.clear()

    def resize(self, engine_count: int) -> None:
        self.engine_count = engine_count
        self.reset_wave(self.current_wave)

    def update(
        self, engine_index: int, observation: PrefillAlignmentObservation
    ) -> PrefillAlignmentRelease | None:
        if observation.wave < self.current_wave:
            return None
        if observation.wave > self.current_wave:
            self.reset_wave(observation.wave)

        self._acknowledge(engine_index, observation.ack_release_id)
        if self.pending_release is not None:
            return self.retry_due(observation.step)
        if observation.ack_only or observation.release_id != self.current_release_id:
            return None

        self.latest_observed_step = max(self.latest_observed_step, observation.step)
        if self._delay_timed_out():
            return self._release(self.latest_observed_step, "max_delay_time_fail_open")

        step_snapshots = self.snapshots.setdefault(observation.step, {})
        step_snapshots.setdefault(engine_index, observation)
        if len(step_snapshots) < self.engine_count and any(
            item.candidate_deferred for item in step_snapshots.values()
        ):
            self.incomplete_started_at.setdefault(observation.step, time.monotonic())
        else:
            self.incomplete_started_at.pop(observation.step, None)

        if self.incomplete_started_at:
            first_incomplete_step = min(self.incomplete_started_at)
            if (
                self.latest_observed_step - first_incomplete_step + 1
                >= _MAX_DELAY_PASSES
            ):
                return self._release(
                    self.latest_observed_step, "max_delay_passes_fail_open"
                )

        oldest_step = self.latest_observed_step - _MAX_DELAY_PASSES
        self.snapshots = {
            step: values
            for step, values in self.snapshots.items()
            if step > oldest_step
        }
        self.incomplete_started_at = {
            step: started_at
            for step, started_at in self.incomplete_started_at.items()
            if step > oldest_step
        }
        if (
            observation.step not in self.snapshots
            or len(step_snapshots) < self.engine_count
        ):
            return None

        self.snapshots = {
            step: values
            for step, values in self.snapshots.items()
            if step > observation.step
        }
        self.incomplete_started_at = {
            step: started_at
            for step, started_at in self.incomplete_started_at.items()
            if step > observation.step
        }
        ordered = [step_snapshots[i] for i in range(self.engine_count)]
        num_prefillable = sum(item.candidate_deferred for item in ordered)
        if num_prefillable == 0:
            self.delayed_passes = 0
            self.delay_started_at = None
            return None
        if any(item.force_allow for item in ordered):
            return self._release(observation.step, "capacity_force_allow")
        if num_prefillable != self.engine_count:
            return self._delay_or_release(observation.step)

        max_running = max(item.running_batch for item in ordered)
        max_prefill = max(item.max_prefill_batch for item in ordered)
        max_requests = max(item.max_running_requests for item in ordered)
        if max_requests - max_running < max_prefill:
            if self.skip_first_delay:
                self.skip_first_delay = False
                return self._release(observation.step, "first_delay_skip")
            return self._delay_or_release(observation.step)
        return self._release(observation.step, "all_prefillable")

    def _delay_or_release(self, step: int) -> PrefillAlignmentRelease | None:
        now = time.monotonic()
        if self.delay_started_at is None:
            self.delay_started_at = now
        self.delayed_passes += 1
        if self.delayed_passes >= _MAX_DELAY_PASSES:
            return self._release(step, "max_delay_passes_fail_open")
        if now - self.delay_started_at >= _MAX_DELAY_SECONDS:
            return self._release(step, "max_delay_time_fail_open")
        return None

    def _release(self, step: int, reason: str) -> PrefillAlignmentRelease:
        release = PrefillAlignmentRelease(
            wave=self.current_wave,
            release_id=self.current_release_id,
            target_step=step + _TARGET_STEP_LEAD,
            reason=reason,
        )
        self.pending_release = release
        self.pending_acks.clear()
        self.delayed_passes = 0
        self.delay_started_at = None
        self.incomplete_started_at.clear()
        self.pending_release_sent_at = time.monotonic()
        self.last_resend_step = release.target_step
        return release

    def seconds_until_deadline(self) -> float | None:
        if self.pending_release is not None:
            deadline = self.pending_release_sent_at + _RESEND_INTERVAL_SECONDS
        elif (started_at := self._deadline_started_at()) is not None:
            deadline = started_at + _MAX_DELAY_SECONDS
        else:
            return None
        return max(0.0, deadline - time.monotonic())

    def deadline_action_due(self) -> PrefillAlignmentRelease | None:
        if self.pending_release is not None:
            return self.retry_due()
        if self._delay_timed_out():
            return self._release(self.latest_observed_step, "max_delay_time_fail_open")
        return None

    def _delay_timed_out(self) -> bool:
        started_at = self._deadline_started_at()
        return (
            started_at is not None
            and time.monotonic() - started_at >= _MAX_DELAY_SECONDS
        )

    def _deadline_started_at(self) -> float | None:
        started_at = list(self.incomplete_started_at.values())
        if self.delay_started_at is not None:
            started_at.append(self.delay_started_at)
        return min(started_at, default=None)

    def retry_due(self, step: int | None = None) -> PrefillAlignmentRelease | None:
        release = self.pending_release
        if release is None:
            return None
        now = time.monotonic()
        step_due = (
            step is not None and step - self.last_resend_step >= _RESEND_INTERVAL_PASSES
        )
        time_due = now - self.pending_release_sent_at >= _RESEND_INTERVAL_SECONDS
        if not step_due and not time_due:
            return None
        if step is not None:
            self.last_resend_step = max(self.last_resend_step, step)
        self.pending_release_sent_at = now
        logger.warning(
            "Resending prefill alignment release %d; awaiting acks from ranks %s.",
            release.release_id,
            sorted(set(range(self.engine_count)) - self.pending_acks),
        )
        return release

    def _acknowledge(self, engine_index: int, release_id: int) -> None:
        release = self.pending_release
        if release is None or release_id != release.release_id:
            return
        self.pending_acks.add(engine_index)
        if len(self.pending_acks) == self.engine_count:
            self.current_release_id += 1
            self.pending_release = None
            self.pending_acks.clear()
            self.pending_release_sent_at = 0.0
            self.last_resend_step = -1
            self.snapshots.clear()
            self.incomplete_started_at.clear()


class EnginePrefillAlignment:
    """Engine-side state for the coordinator release protocol."""

    def __init__(self) -> None:
        self.release_queue = queue.Queue[ReleasePayload]()
        self.generation = 0
        self.release: ReleasePayload | None = None
        self.applied_release: tuple[int, int] | None = None
        self.last_ack: int | None = None
        self.last_schedule_sequence = 0

    @property
    def allows_prefill(self) -> bool:
        return self.applied_release is not None

    def enqueue(self, release: ReleasePayload) -> None:
        self.release_queue.put_nowait(release)

    def reset(self, schedule_sequence: int = 0) -> None:
        self.generation = 0
        self.release = None
        self.applied_release = None
        self.last_ack = None
        self.last_schedule_sequence = schedule_sequence
        while True:
            try:
                self.release_queue.get_nowait()
            except queue.Empty:
                break

    def prepare(self, wave: int, step: int, rank: int) -> bool:
        """Apply a due release and return whether a stale release needs an ack."""
        resend_ack = False
        while True:
            try:
                release_wave, release_id, target_step = self.release_queue.get_nowait()
            except queue.Empty:
                break
            if release_wave != wave:
                continue
            if release_id < self.generation:
                resend_ack |= release_id == self.last_ack
                continue
            if release_id > self.generation:
                logger.warning(
                    "DP%d resynchronizing prefill alignment release %d to %d.",
                    rank,
                    self.generation,
                    release_id,
                )
                self.generation = release_id
                self.applied_release = None
            self.release = release_wave, release_id, target_step

        if self.applied_release is not None or self.release is None:
            return resend_ack
        _, release_id, target_step = self.release
        if step < target_step:
            return resend_ack

        self.applied_release = release_id, target_step
        self.generation = release_id + 1
        self.release = None
        logger.debug(
            "DP%d applying prefill release wave=%d release_id=%d "
            "target_step=%d current_step=%d.",
            rank,
            wave,
            release_id,
            target_step,
            step,
        )
        return resend_ack

    def observe(
        self,
        wave: int,
        step: int,
        telemetry: PrefillAlignmentTelemetry | None,
        has_requests: bool,
    ) -> PrefillAlignmentObservation:
        fresh_walk = (
            telemetry is not None
            and telemetry.schedule_sequence != self.last_schedule_sequence
        )
        if fresh_walk:
            assert telemetry is not None
            self.last_schedule_sequence = telemetry.schedule_sequence
            if self.applied_release is not None:
                self._consume_release()
            candidate_deferred = telemetry.candidate_deferred
            force_allow = telemetry.force_allow
            running_batch = telemetry.running_batch
            max_prefill_batch = telemetry.max_prefill_batch
            max_running_requests = telemetry.max_running_requests
        else:
            if self.applied_release is not None and not has_requests:
                self._consume_release()
            candidate_deferred = force_allow = False
            running_batch = max_prefill_batch = max_running_requests = 0

        return PrefillAlignmentObservation(
            wave=wave,
            step=step,
            release_id=self.generation,
            candidate_deferred=candidate_deferred,
            force_allow=force_allow,
            running_batch=running_batch,
            max_prefill_batch=max_prefill_batch,
            max_running_requests=max_running_requests,
            ack_release_id=self.last_ack if self.last_ack is not None else -1,
        )

    def finish(self, wave: int, step: int) -> PrefillAlignmentObservation | None:
        if self.applied_release is not None:
            self._consume_release()
        return self.ack_observation(wave, step)

    def ack_observation(
        self, wave: int, step: int
    ) -> PrefillAlignmentObservation | None:
        if self.last_ack is None:
            return None
        return PrefillAlignmentObservation(
            wave=wave,
            step=step,
            release_id=self.generation,
            candidate_deferred=False,
            force_allow=False,
            running_batch=0,
            max_prefill_batch=0,
            max_running_requests=0,
            ack_release_id=self.last_ack,
            ack_only=True,
        )

    def consume_idle_release(self) -> None:
        if self.applied_release is not None:
            self._consume_release()

    def _consume_release(self) -> None:
        assert self.applied_release is not None
        self.last_ack = self.applied_release[0]
        self.applied_release = None
