# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.engine.coordinator import PrefillAlignmentCoordinator
from vllm.v1.metrics.stats import SchedulerStats


def observation(
    step: int,
    *,
    deferred: bool,
    candidate: bool | None = None,
    running: int = 1,
    max_prefill: int | None = None,
    max_running_requests: int = 8,
    force_allow: bool = False,
    wave: int = 0,
    generation: int = 0,
    ack_generation: int = -1,
    ack_target_step: int = -1,
    actual_requests: int = 0,
    actual_tokens: int = 0,
) -> SchedulerStats:
    candidate = deferred if candidate is None else candidate
    max_prefill = int(candidate) if max_prefill is None else max_prefill
    return SchedulerStats(
        step_counter=step,
        current_wave=wave,
        prefill_alignment_phase=1,
        prefill_alignment_generation=generation,
        prefill_alignment_ack_generation=ack_generation,
        prefill_alignment_ack_target_step=ack_target_step,
        prefillable=candidate,
        prefill_deferred=deferred,
        prefill_force_allow=force_allow,
        prefill_running_batch=running,
        prefill_max_batch=max_prefill,
        prefill_max_running_requests=max_running_requests,
        prefill_waiting_queue_len=int(candidate),
        actual_prefill_requests=actual_requests,
        actual_prefill_tokens=actual_tokens,
    )


def final_ack(
    release_generation: int,
    target_step: int,
    *,
    requests: int = 0,
    tokens: int = 0,
    wave: int = 0,
) -> SchedulerStats:
    return SchedulerStats(
        current_wave=wave,
        prefill_alignment_phase=2,
        prefill_alignment_generation=release_generation + 1,
        prefill_alignment_ack_generation=release_generation,
        prefill_alignment_ack_target_step=target_step,
        actual_prefill_requests=requests,
        actual_prefill_tokens=tokens,
    )


def update_step(
    coordinator: PrefillAlignmentCoordinator,
    step: int,
    deferred: list[bool],
    **kwargs,
):
    release = None
    for engine_index, value in enumerate(deferred):
        result = coordinator.update(
            engine_index,
            observation(step, deferred=value, **kwargs),
        )
        release = result or release
    return release


def test_all_prefillable_release_and_piggybacked_ack() -> None:
    coordinator = PrefillAlignmentCoordinator(4, target_step_lead=2)

    release = update_step(coordinator, 7, [True] * 4)

    assert release is not None
    assert release.release_id == 0
    assert release.target_step == 9
    assert release.reason == "all_prefillable"

    for engine_index in range(4):
        coordinator.update(
            engine_index,
            observation(
                10,
                deferred=False,
                generation=1,
                ack_generation=0,
                ack_target_step=9,
                actual_requests=1,
                actual_tokens=1024,
            ),
        )

    assert coordinator.pending_release is None
    assert coordinator.current_release_id == 1
    assert coordinator.last_actual_prefill == {
        engine_index: (1, 1024) for engine_index in range(4)
    }


@pytest.mark.parametrize("max_delay_passes", [1, 7, 30])
def test_mixed_prefill_is_bounded_by_fail_open_limit(
    max_delay_passes: int,
) -> None:
    coordinator = PrefillAlignmentCoordinator(
        4,
        max_delay_passes=max_delay_passes,
        target_step_lead=3,
    )

    for step in range(max_delay_passes - 1):
        assert update_step(coordinator, step, [True, False, False, False]) is None

    release = update_step(
        coordinator,
        max_delay_passes - 1,
        [True, False, False, False],
    )

    assert release is not None
    assert release.target_step == max_delay_passes + 2
    assert release.reason == "max_delay_fail_open"


def test_mixed_prefill_without_decode_still_waits() -> None:
    coordinator = PrefillAlignmentCoordinator(4, max_delay_passes=2)

    assert (
        update_step(
            coordinator,
            3,
            [True, False, False, False],
            running=0,
        )
        is None
    )
    release = update_step(
        coordinator,
        4,
        [True, False, False, False],
        running=0,
    )

    assert release is not None
    assert release.reason == "max_delay_fail_open"


def test_capacity_force_allow_is_global() -> None:
    coordinator = PrefillAlignmentCoordinator(4, target_step_lead=2)
    release = None
    for engine_index in range(4):
        result = coordinator.update(
            engine_index,
            observation(
                5,
                deferred=engine_index == 0,
                force_allow=engine_index == 0,
            ),
        )
        release = result or release

    assert release is not None
    assert release.reason == "capacity_force_allow"
    assert release.target_step == 7


def test_first_slot_limited_delay_is_skipped_then_bounded() -> None:
    coordinator = PrefillAlignmentCoordinator(2, max_delay_passes=2)
    kwargs = {"running": 8, "max_prefill": 1, "max_running_requests": 8}

    first = update_step(coordinator, 0, [True, True], **kwargs)
    assert first is not None
    assert first.reason == "first_delay_skip"
    for engine_index in range(2):
        coordinator.update(engine_index, final_ack(0, first.target_step))

    assert update_step(coordinator, 3, [True, True], generation=1, **kwargs) is None
    second = update_step(coordinator, 4, [True, True], generation=1, **kwargs)
    assert second is not None
    assert second.reason == "max_delay_fail_open"


def test_scheduled_candidate_is_not_false_positive_demand() -> None:
    coordinator = PrefillAlignmentCoordinator(2)

    release = update_step(
        coordinator,
        1,
        [False, False],
        candidate=True,
        actual_requests=1,
        actual_tokens=1024,
    )

    assert release is None
    assert coordinator.delayed_passes == 0


def test_missing_release_ack_broadcasts_fail_open_resync() -> None:
    coordinator = PrefillAlignmentCoordinator(
        2,
        max_delay_passes=3,
        target_step_lead=1,
    )
    release = update_step(coordinator, 0, [True, True])
    assert release is not None and release.target_step == 1

    coordinator.update(0, final_ack(0, 1, requests=1, tokens=512))
    assert coordinator.current_release_id == 0

    # Rank 1's ack is lost. Observations keep flowing, and the coordinator
    # advances after the bounded grace period instead of blocking DP progress.
    retry = coordinator.update(
        0,
        observation(4, deferred=False, generation=1),
    )

    assert coordinator.current_release_id == 1
    assert retry is not None
    assert retry.release_id == 1
    assert retry.target_step == 5
    assert retry.reason == "ack_timeout_resync"

    for engine_index in range(2):
        coordinator.update(engine_index, final_ack(1, 5))
    assert coordinator.pending_release is None
    assert coordinator.current_release_id == 2


def test_new_wave_resets_release_id_and_rejects_stale_stats() -> None:
    coordinator = PrefillAlignmentCoordinator(2)
    release = update_step(coordinator, 0, [True, True])
    assert release is not None

    coordinator.reset_wave(3)
    assert coordinator.current_wave == 3
    assert coordinator.current_release_id == 0
    assert coordinator.pending_release is None
    assert coordinator.skip_first_delay

    stale = observation(5, deferred=True, wave=2)
    assert coordinator.update(0, stale) is None
    assert coordinator.snapshots == {}
