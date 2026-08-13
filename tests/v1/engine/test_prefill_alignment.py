# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
from types import SimpleNamespace

import msgspec.msgpack
import pytest

from vllm.config import SchedulerConfig, VllmConfig
from vllm.v1.core.sched.interface import PrefillAlignmentTelemetry
from vllm.v1.engine import EngineCoreOutputs, PrefillAlignmentObservation
from vllm.v1.engine.coordinator import DPCoordinatorProc
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc
from vllm.v1.engine.prefill_alignment import (
    EnginePrefillAlignment,
    PrefillAlignmentCoordinator,
    PrefillAlignmentRelease,
)
from vllm.v1.engine.utils import launch_core_engines
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(autouse=True)
def _restore_prefill_alignment_clock(monkeypatch: pytest.MonkeyPatch):
    """Undo per-test clock monkeypatches before global fixture teardown."""
    yield
    monkeypatch.undo()


def observation(
    step: int,
    *,
    deferred: bool,
    release_id: int = 0,
    running: int = 1,
    max_prefill: int | None = None,
    max_running_requests: int = 8,
    force_allow: bool = False,
    wave: int = 0,
    ack_release_id: int = -1,
    ack_target_step: int = -1,
    actual_requests: int = 0,
    actual_tokens: int = 0,
    ack_only: bool = False,
) -> PrefillAlignmentObservation:
    max_prefill = int(deferred) if max_prefill is None else max_prefill
    return PrefillAlignmentObservation(
        wave=wave,
        step=step,
        release_id=release_id,
        candidate_deferred=deferred,
        force_allow=force_allow,
        running_batch=running,
        max_prefill_batch=max_prefill,
        max_running_requests=max_running_requests,
        ack_release_id=ack_release_id,
        ack_target_step=ack_target_step,
        actual_prefill_requests=actual_requests,
        actual_prefill_tokens=actual_tokens,
        ack_only=ack_only,
    )


def update_step(
    coordinator: PrefillAlignmentCoordinator,
    step: int,
    deferred: list[bool],
    **kwargs,
) -> PrefillAlignmentRelease | None:
    release = None
    for engine_index, value in enumerate(deferred):
        result = coordinator.update(
            engine_index,
            observation(step, deferred=value, **kwargs),
        )
        release = result or release
    return release


def ack(
    release: PrefillAlignmentRelease,
    *,
    step: int,
    requests: int = 1,
    tokens: int = 1024,
) -> PrefillAlignmentObservation:
    return observation(
        step,
        deferred=False,
        release_id=release.release_id + 1,
        wave=release.wave,
        ack_release_id=release.release_id,
        ack_target_step=release.target_step,
        actual_requests=requests,
        actual_tokens=tokens,
    )


def make_engine(
    *,
    step: int,
    release_id: int = 0,
    telemetry: PrefillAlignmentTelemetry | None = None,
    has_requests: bool = True,
) -> DPEngineCoreProc:
    engine = object.__new__(DPEngineCoreProc)
    engine.enable_adaptive_prefill_alignment = True
    engine.has_coordinator = True
    engine.current_wave = 2
    engine.step_counter = step
    engine.dp_rank = 0
    engine.prefill_alignment = EnginePrefillAlignment()
    engine.prefill_alignment.generation = release_id
    engine.output_queue = queue.Queue()
    telemetry = telemetry or PrefillAlignmentTelemetry()
    engine.scheduler = SimpleNamespace(
        get_prefill_alignment_telemetry=lambda: telemetry,
        has_requests=lambda: has_requests,
    )
    return engine


def test_all_rank_release_activates_exactly_at_s_plus_two() -> None:
    coordinator = PrefillAlignmentCoordinator(2)
    release = update_step(coordinator, 7, [True, True])

    assert release is not None
    assert release.release_id == 0
    assert release.target_step == 9

    engines = [make_engine(step=8), make_engine(step=8)]
    for engine in engines:
        engine.prefill_alignment.enqueue((2, 0, 9))
        engine._prepare_prefill_alignment_step()
        assert not engine.prefill_alignment.allows_prefill

        engine.step_counter = 9
        engine._prepare_prefill_alignment_step()
        assert engine.prefill_alignment.allows_prefill
        assert engine.prefill_alignment.generation == 1


def test_no_demand_and_mixed_readiness_are_silent() -> None:
    coordinator = PrefillAlignmentCoordinator(2)

    assert update_step(coordinator, 1, [False, False]) is None
    assert update_step(coordinator, 2, [True, False]) is None
    assert coordinator.pending_release is None


def test_force_allow_matches_measured_ungated_any_rank_semantics() -> None:
    coordinator = PrefillAlignmentCoordinator(2)
    assert (
        coordinator.update(
            0,
            observation(3, deferred=False, force_allow=True),
        )
        is None
    )

    release = coordinator.update(1, observation(3, deferred=True))

    assert release is not None
    assert release.target_step == 5
    assert release.reason == "capacity_force_allow"


def test_mixed_readiness_has_fixed_pass_and_time_bounds() -> None:
    coordinator = PrefillAlignmentCoordinator(2)

    for step in range(29):
        assert update_step(coordinator, step, [True, False]) is None
    release = update_step(coordinator, 29, [True, False])
    assert release is not None
    assert release.target_step == 31
    assert release.reason == "max_delay_passes_fail_open"

    coordinator = PrefillAlignmentCoordinator(2)
    assert update_step(coordinator, 0, [True, False]) is None
    assert coordinator.delay_started_at is not None
    coordinator.delay_started_at -= 6
    release = update_step(coordinator, 1, [True, False])
    assert release is not None
    assert release.reason == "max_delay_time_fail_open"


def test_mixed_wall_deadline_fires_when_one_rank_stops_publishing() -> None:
    coordinator = PrefillAlignmentCoordinator(2)
    assert update_step(coordinator, 0, [True, False]) is None
    assert coordinator.delay_started_at is not None
    coordinator.delay_started_at -= 6

    # Only rank 0 remains observable. The wall deadline must not require a
    # second complete all-rank snapshot.
    release = coordinator.update(0, observation(8, deferred=True))

    assert release is not None
    assert release.target_step == 10
    assert release.reason == "max_delay_time_fail_open"


def test_incomplete_snapshot_is_time_bounded_and_storage_bounded() -> None:
    coordinator = PrefillAlignmentCoordinator(2)

    # Rank 1 never publishes, including for the first observed step. Deferred
    # work on rank 0 must still arm the wall deadline.
    assert coordinator.update(0, observation(0, deferred=True)) is None
    assert coordinator.delay_started_at is None
    assert coordinator.incomplete_started_at

    # Partial snapshots cannot grow without bound while that rank is absent.
    for step in range(1, 29):
        assert coordinator.update(0, observation(step, deferred=True)) is None
    assert len(coordinator.snapshots) <= 30

    coordinator.incomplete_started_at[0] -= 6
    release = coordinator.deadline_action_due()

    assert release is not None
    assert release.target_step == 30
    assert release.reason == "max_delay_time_fail_open"


def test_incomplete_snapshot_has_pass_bound() -> None:
    coordinator = PrefillAlignmentCoordinator(2)

    for step in range(29):
        assert coordinator.update(0, observation(step, deferred=True)) is None
    release = coordinator.update(0, observation(29, deferred=True))

    assert release is not None
    assert release.target_step == 31
    assert release.reason == "max_delay_passes_fail_open"


def test_incomplete_no_demand_snapshot_remains_silent() -> None:
    coordinator = PrefillAlignmentCoordinator(2)

    for step in range(40):
        assert coordinator.update(0, observation(step, deferred=False)) is None

    assert coordinator.delay_started_at is None
    assert not coordinator.incomplete_started_at
    assert coordinator.deadline_action_due() is None
    assert len(coordinator.snapshots) <= 30


def test_complete_mixed_snapshot_starts_only_normal_delay() -> None:
    coordinator = PrefillAlignmentCoordinator(2)

    assert coordinator.update(0, observation(0, deferred=True)) is None
    assert coordinator.delay_started_at is None
    assert coordinator.incomplete_started_at

    assert coordinator.update(1, observation(0, deferred=False)) is None
    assert coordinator.delay_started_at is not None
    assert not coordinator.incomplete_started_at


def test_wave_reset_clears_incomplete_wait() -> None:
    coordinator = PrefillAlignmentCoordinator(2)
    assert coordinator.update(0, observation(0, deferred=True)) is None
    assert coordinator.incomplete_started_at

    assert coordinator.update(0, observation(0, deferred=False, wave=1)) is None
    assert coordinator.current_wave == 1
    assert not coordinator.incomplete_started_at


def test_one_lease_remains_until_all_ranks_ack_actual_work() -> None:
    coordinator = PrefillAlignmentCoordinator(2)
    release = update_step(coordinator, 4, [True, True])
    assert release is not None

    coordinator.update(0, ack(release, step=7, requests=2, tokens=2048))
    assert coordinator.pending_release == release
    coordinator.update(1, ack(release, step=7, requests=1, tokens=1024))

    assert coordinator.pending_release is None
    assert coordinator.current_release_id == 1
    assert coordinator.last_actual_prefill == {0: (2, 2048), 1: (1, 1024)}


def test_all_engines_apply_once_and_ack_their_real_scheduler_walk() -> None:
    coordinator = PrefillAlignmentCoordinator(2)
    release = update_step(coordinator, 4, [True, True])
    assert release is not None

    for rank, requests in enumerate((2, 1)):
        telemetry = PrefillAlignmentTelemetry(
            schedule_sequence=1,
            actual_prefill_requests=requests,
            actual_prefill_tokens=requests * 1024,
        )
        engine = make_engine(
            step=release.target_step,
            telemetry=telemetry,
        )
        engine.current_wave = release.wave
        engine.dp_rank = rank
        engine.prefill_alignment.enqueue(
            (release.wave, release.release_id, release.target_step)
        )
        engine._prepare_prefill_alignment_step()
        engine._prepare_prefill_alignment_step()
        assert engine.prefill_alignment.generation == 1

        engine.step_counter += 1
        engine._publish_prefill_alignment_observation()
        _, outputs = engine.output_queue.get_nowait()
        result = outputs.prefill_alignment_observation
        assert result is not None
        coordinator.update(rank, result)

    assert coordinator.pending_release is None
    assert coordinator.current_release_id == 1
    assert coordinator.last_actual_prefill == {0: (2, 2048), 1: (1, 1024)}


def test_dropped_release_is_retried_idempotently() -> None:
    coordinator = PrefillAlignmentCoordinator(2)
    release = update_step(coordinator, 0, [True, True])
    assert release is not None and release.target_step == 2

    coordinator.update(0, ack(release, step=3))
    retry = coordinator.update(1, observation(4, deferred=True))
    assert retry == release
    assert coordinator.current_release_id == 0

    engine = make_engine(step=4)
    engine.current_wave = retry.wave
    engine.prefill_alignment.enqueue((retry.wave, retry.release_id, retry.target_step))
    engine._prepare_prefill_alignment_step()
    assert engine.prefill_alignment.allows_prefill
    assert engine.prefill_alignment.applied_release == (0, 2, True, 0)


def test_duplicate_release_republishes_previous_ack() -> None:
    engine = make_engine(step=5, release_id=1)
    engine.prefill_alignment.last_ack = (0, 2, False, 1, 512)
    engine.prefill_alignment.enqueue((2, 0, 2))

    engine._prepare_prefill_alignment_step()

    _, outputs = engine.output_queue.get_nowait()
    result = outputs.prefill_alignment_observation
    assert result is not None and result.ack_only
    assert result.ack_release_id == 0
    assert result.actual_prefill_tokens == 512


def test_release_is_retained_across_batch_queue_pass() -> None:
    telemetry = PrefillAlignmentTelemetry(schedule_sequence=8)
    engine = make_engine(
        step=9,
        release_id=1,
        telemetry=telemetry,
    )
    engine.prefill_alignment.applied_release = (0, 9, False, 8)
    engine.prefill_alignment.last_schedule_sequence = 8

    engine._publish_prefill_alignment_observation()

    _, outputs = engine.output_queue.get_nowait()
    result = outputs.prefill_alignment_observation
    assert result is not None
    assert not result.candidate_deferred
    assert result.ack_release_id == -1
    assert engine.prefill_alignment.allows_prefill
    assert engine.prefill_alignment.applied_release == (0, 9, False, 8)


def test_idle_rank_observes_every_step_and_acks_dummy_spent_release() -> None:
    telemetry = PrefillAlignmentTelemetry(schedule_sequence=8)
    engine = make_engine(
        step=9,
        release_id=1,
        telemetry=telemetry,
        has_requests=False,
    )
    engine.prefill_alignment.last_schedule_sequence = 8
    engine.prefill_alignment.applied_release = (0, 9, False, 8)

    engine._publish_prefill_alignment_observation()
    _, first = engine.output_queue.get_nowait()
    first_observation = first.prefill_alignment_observation
    assert first_observation is not None
    assert first_observation.ack_release_id == 0
    assert first_observation.actual_prefill_requests == 0

    engine.step_counter = 10
    engine._publish_prefill_alignment_observation()
    _, second = engine.output_queue.get_nowait()
    second_observation = second.prefill_alignment_observation
    assert second_observation is not None
    assert second_observation.step == 10
    assert not second_observation.candidate_deferred


def test_idle_wave_end_ack_does_not_reuse_stale_actual_counters() -> None:
    telemetry = PrefillAlignmentTelemetry(
        schedule_sequence=8,
        actual_prefill_requests=3,
        actual_prefill_tokens=3072,
    )
    engine = make_engine(
        step=9,
        release_id=1,
        telemetry=telemetry,
        has_requests=False,
    )
    engine.prefill_alignment.last_schedule_sequence = 8
    engine.prefill_alignment.applied_release = (0, 9, False, 8)

    engine._publish_prefill_alignment_final_ack()

    _, outputs = engine.output_queue.get_nowait()
    result = outputs.prefill_alignment_observation
    assert result is not None and result.ack_only
    assert result.actual_prefill_requests == 0
    assert result.actual_prefill_tokens == 0


def test_wave_reset_baselines_lifetime_scheduler_sequence() -> None:
    telemetry = PrefillAlignmentTelemetry(
        schedule_sequence=8,
        candidate_deferred=True,
        force_allow=True,
    )
    engine = make_engine(step=0, telemetry=telemetry)
    engine.prefill_alignment.last_schedule_sequence = 7

    engine._reset_prefill_alignment()
    engine._publish_prefill_alignment_observation()

    _, outputs = engine.output_queue.get_nowait()
    result = outputs.prefill_alignment_observation
    assert result is not None
    assert engine.prefill_alignment.last_schedule_sequence == 8
    assert not result.candidate_deferred
    assert not result.force_allow


def test_fresh_walk_consumes_release_and_acks_actual_work() -> None:
    telemetry = PrefillAlignmentTelemetry(
        schedule_sequence=9,
        actual_prefill_requests=2,
        actual_prefill_tokens=2048,
    )
    engine = make_engine(step=10, release_id=1, telemetry=telemetry)
    engine.prefill_alignment.applied_release = (0, 9, True, 8)
    engine.prefill_alignment.last_schedule_sequence = 8

    engine._publish_prefill_alignment_observation()

    _, outputs = engine.output_queue.get_nowait()
    result = outputs.prefill_alignment_observation
    assert result is not None
    assert result.ack_release_id == 0
    assert result.ack_target_step == 9
    assert result.release_late
    assert result.actual_prefill_requests == 2
    assert not engine.prefill_alignment.allows_prefill


def test_server_lifetime_skip_first_survives_wave_reset() -> None:
    coordinator = PrefillAlignmentCoordinator(2)
    kwargs = {"running": 8, "max_prefill": 1, "max_running_requests": 8}
    release = update_step(coordinator, 0, [True, True], **kwargs)
    assert release is not None and release.reason == "first_delay_skip"

    coordinator.reset_wave(3)
    assert not coordinator.skip_first_delay
    assert update_step(coordinator, 0, [True, True], wave=3, **kwargs) is None


def test_observation_and_release_wire_formats_are_compact() -> None:
    expected = observation(3, deferred=True, release_id=7)
    encoded = MsgpackEncoder().encode(
        EngineCoreOutputs(
            engine_index=3,
            prefill_alignment_observation=expected,
        )
    )
    decoded = MsgpackDecoder(EngineCoreOutputs).decode(encoded)
    assert decoded.prefill_alignment_observation == expected

    sent = []
    socket = SimpleNamespace(send_multipart=lambda frames: sent.append(frames))
    DPCoordinatorProc._send_prefill_alignment_release(
        socket,
        PrefillAlignmentRelease(2, 7, 11, "all_prefillable"),
    )
    assert msgspec.msgpack.decode(sent[0][1]) == [2, 7, 11]


def test_engine_core_outputs_decodes_old_shorter_array_payload() -> None:
    # This is the complete pre-feature array layout, without the new trailing
    # prefill_alignment_observation field.
    old_payload = msgspec.msgpack.encode([3, [], None, 123.0, None, None, 4, 5])

    decoded = MsgpackDecoder(EngineCoreOutputs).decode(old_payload)

    assert decoded.engine_index == 3
    assert decoded.timestamp == 123.0
    assert decoded.wave_complete == 4
    assert decoded.start_wave == 5
    assert decoded.prefill_alignment_observation is None


def test_disabled_coordinator_uses_legacy_stats_poll_timeout() -> None:
    legacy_timeout = DPCoordinatorProc._legacy_stats_poll_timeout_ms

    # The legacy lockstep quiet window is at least 50ms.
    assert (
        legacy_timeout(
            now_ms=80,
            last_publish_time=0,
            stats_changed=True,
            has_step_snapshot=False,
            enable_wave_coordination=True,
            stats_update_interval_ms=100,
        )
        == 50
    )
    # A complete step snapshot removes that floor, including when publication
    # is already overdue.
    assert (
        legacy_timeout(
            now_ms=125,
            last_publish_time=0,
            stats_changed=True,
            has_step_snapshot=True,
            enable_wave_coordination=True,
            stats_update_interval_ms=100,
        )
        == 0
    )
    # Unchanged stats retain the original five-second heartbeat.
    assert (
        legacy_timeout(
            now_ms=4900,
            last_publish_time=0,
            stats_changed=False,
            has_step_snapshot=False,
            enable_wave_coordination=False,
            stats_update_interval_ms=100,
        )
        == 100
    )


def test_stats_deadline_survives_continuous_alignment_events() -> None:
    deadline = DPCoordinatorProc._stats_publish_deadline_ms(
        last_publish_time=0,
        stats_changed=True,
        stats_changed_at=1000,
        has_step_snapshot=False,
        enable_wave_coordination=True,
        stats_update_interval_ms=100,
    )

    # Event availability is intentionally absent from the deadline function:
    # even a continuously readable alignment socket cannot move this deadline.
    assert deadline == 1050
    continuous_event_times = range(1001, 1101)
    assert next(now for now in continuous_event_times if now >= deadline) == 1050

    # Once a complete step snapshot exists, the old 50ms collection wait no
    # longer delays an already-due normal stats publication.
    assert (
        DPCoordinatorProc._stats_publish_deadline_ms(
            last_publish_time=0,
            stats_changed=True,
            stats_changed_at=1000,
            has_step_snapshot=True,
            enable_wave_coordination=True,
            stats_update_interval_ms=100,
        )
        == 100
    )


def test_normal_request_count_stats_remain_separate() -> None:
    engine = make_engine(step=9)
    engine.publish_dp_lb_stats = True
    engine.last_counts = (0, 0)
    engine.scheduler = SimpleNamespace(
        get_request_counts=lambda: (7, 3),
        get_kv_cache_usage=lambda: 0.5,
    )

    engine._maybe_publish_request_counts()
    _, outputs = engine.output_queue.get_nowait()
    assert outputs.scheduler_stats is not None
    assert outputs.scheduler_stats.num_running_reqs == 7
    assert outputs.prefill_alignment_observation is None


def test_disabled_engine_allocates_no_alignment_runtime_state(monkeypatch) -> None:
    monkeypatch.setattr(EngineCoreProc, "__init__", lambda *args, **kwargs: None)
    config = SimpleNamespace(
        model_config=SimpleNamespace(is_moe=True),
        scheduler_config=SimpleNamespace(
            prefill_schedule_interval=1,
            enable_adaptive_prefill_alignment=False,
        ),
        parallel_config=SimpleNamespace(data_parallel_rank=0),
    )
    engine = DPEngineCoreProc(config, False, "", object, False)

    assert not engine.enable_adaptive_prefill_alignment
    assert not hasattr(engine, "prefill_alignment")


def make_scope_config(
    *, dp_size: int = 2, is_moe: bool = True, local_rank: int | None = None
) -> VllmConfig:
    config = object.__new__(VllmConfig)
    config.scheduler_config = SimpleNamespace(enable_adaptive_prefill_alignment=True)
    config.parallel_config = SimpleNamespace(
        data_parallel_size=dp_size,
        data_parallel_rank_local=local_rank,
        enable_elastic_ep=False,
    )
    config.model_config = SimpleNamespace(is_moe=is_moe)
    return config


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (make_scope_config(dp_size=1), "data-parallel-size"),
        (make_scope_config(is_moe=False), "MoE model"),
    ],
)
def test_adaptive_alignment_rejects_unsupported_scope(
    config: VllmConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        config._verify_adaptive_prefill_alignment()


def test_adaptive_alignment_accepts_online_moe_dp() -> None:
    make_scope_config()._verify_adaptive_prefill_alignment()


def test_adaptive_alignment_accepts_post_handshake_rank() -> None:
    # Online DP engines receive their local rank during the handshake, then
    # re-run VllmConfig.__post_init__. A populated local rank must therefore
    # remain valid after launch.
    make_scope_config(local_rank=0)._verify_adaptive_prefill_alignment()


def test_adaptive_alignment_rejects_offline_dp_launch() -> None:
    # The local rank identifies offline mode only at the launch boundary.
    # Keep that context-dependent rejection in launch_core_engines rather
    # than in VllmConfig.__post_init__, which also runs after online handshakes.
    config = make_scope_config(local_rank=0)
    config.parallel_config.data_parallel_size_local = 1
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.data_parallel_master_ip = "127.0.0.1"
    config.parallel_config.local_engines_only = True

    with (
        pytest.raises(ValueError, match="online MoE data parallel"),
        launch_core_engines(config, object, False, object()),
    ):
        pytest.fail("offline DP launch unexpectedly succeeded")


def test_adaptive_alignment_rejects_elastic_ep() -> None:
    config = make_scope_config()
    config.parallel_config.enable_elastic_ep = True
    with pytest.raises(ValueError, match="elastic expert parallelism"):
        config._verify_adaptive_prefill_alignment()


def test_adaptive_alignment_rejects_custom_scheduler() -> None:
    with pytest.raises(ValueError, match="scheduler-cls"):
        SchedulerConfig.default_factory(
            enable_adaptive_prefill_alignment=True,
            scheduler_cls="example.CustomScheduler",
        )


def test_custom_scheduler_remains_supported_when_disabled() -> None:
    config = SchedulerConfig.default_factory(scheduler_cls="example.CustomScheduler")
    assert config.scheduler_cls == "example.CustomScheduler"
