# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.offloader.runtime import PrefetchRuntimeController


def test_prefetch_runtime_rejects_nonpositive_prefetch_step():
    with pytest.raises(ValueError, match="prefetch_step must be >= 1"):
        PrefetchRuntimeController(unit_count=1, prefetch_step=0)


def _simulate_sequential_execution(
    unit_count: int,
    prefetch_step: int,
    execution_steps: int | None = None,
) -> list[tuple[int, int, int | None, int | None]]:
    runtime = PrefetchRuntimeController(
        unit_count=unit_count,
        prefetch_step=prefetch_step,
    )
    trace: list[tuple[int, int, int | None, int | None]] = []

    if unit_count == 0:
        return trace

    for runtime_unit in runtime.initial_prefetches():
        assert runtime.begin_prefetch(runtime_unit.unit_idx) is None

    total_steps = execution_steps or unit_count
    for step in range(total_steps):
        unit_idx = step % unit_count
        runtime_unit = runtime.get_unit(unit_idx)
        assert runtime.is_unit_resident(runtime_unit.unit_idx)

        next_unit = runtime.prefetch_after(runtime_unit.unit_idx)
        if next_unit is not None:
            runtime.begin_prefetch(next_unit.unit_idx)

        trace.append(
            (
                runtime_unit.unit_idx,
                runtime_unit.slot_idx,
                None if next_unit is None else next_unit.unit_idx,
                None if next_unit is None else next_unit.slot_idx,
            )
        )

    return trace


@pytest.mark.parametrize(
    ("unit_count", "prefetch_step", "expected_trace"),
    [
        (
            4,
            1,
            [
                (0, 0, 1, 0),
                (1, 0, 2, 0),
                (2, 0, 3, 0),
                (3, 0, 0, 0),
            ],
        ),
        (
            5,
            2,
            [
                (0, 0, 2, 0),
                (1, 1, 3, 1),
                (2, 0, 4, 0),
                (3, 1, 1, 1),
                (4, 0, 0, 0),
            ],
        ),
        (
            6,
            3,
            [
                (0, 0, 3, 0),
                (1, 1, 4, 1),
                (2, 2, 5, 2),
                (3, 0, 0, 0),
                (4, 1, 1, 1),
                (5, 2, 2, 2),
            ],
        ),
    ],
)
def test_prefetch_runtime_prefetch_step_schedule(
    unit_count: int,
    prefetch_step: int,
    expected_trace: list[tuple[int, int, int | None, int | None]],
):
    assert (
        _simulate_sequential_execution(
            unit_count=unit_count,
            prefetch_step=prefetch_step,
        )
        == expected_trace
    )


def test_prefetch_runtime_slot_reuse_wraps_across_multiple_passes():
    trace = _simulate_sequential_execution(
        unit_count=3,
        prefetch_step=2,
        execution_steps=6,
    )

    assert trace == [
        (0, 0, 2, 0),
        (1, 1, None, None),
        (2, 0, 0, 0),
        (0, 0, 2, 0),
        (1, 1, None, None),
        (2, 0, 0, 0),
    ]


def test_prefetch_runtime_does_not_reuse_unexecuted_slot_owner():
    runtime = PrefetchRuntimeController(unit_count=15, prefetch_step=8)

    for runtime_unit in runtime.initial_prefetches():
        assert runtime.begin_prefetch(runtime_unit.unit_idx) is None

    executed_units: set[int] = set()
    for unit_idx in range(runtime.unit_count):
        executed_units.add(unit_idx)
        next_unit = runtime.prefetch_after(unit_idx)
        if next_unit is None:
            continue

        previous_owner = runtime.begin_prefetch(next_unit.unit_idx)
        if previous_owner is not None:
            assert previous_owner.unit_idx in executed_units


def test_prefetch_runtime_tracks_capture_started_prefetches_until_waited():
    runtime = PrefetchRuntimeController(unit_count=4, prefetch_step=2)

    runtime.mark_prefetch_started(0, in_capture=False)
    runtime.mark_prefetch_started(1, in_capture=True)
    runtime.mark_prefetch_started(3, in_capture=True)

    assert runtime.is_pending_in_capture(0) is False
    assert runtime.is_pending_in_capture(1) is True
    assert runtime.is_pending_in_capture(2) is False
    assert runtime.is_pending_in_capture(3) is True
    assert tuple(
        (unit.unit_idx, unit.slot_idx) for unit in runtime.pending_capture_prefetches()
    ) == ((1, 1), (3, 1))

    runtime.mark_waited(1)

    assert tuple(
        (unit.unit_idx, unit.slot_idx) for unit in runtime.pending_capture_prefetches()
    ) == ((3, 1),)


def test_prefetch_runtime_non_capture_prefetch_clears_stale_pending_state():
    runtime = PrefetchRuntimeController(unit_count=3, prefetch_step=2)

    runtime.mark_prefetch_started(2, in_capture=True)
    assert runtime.is_pending_in_capture(2) is True

    runtime.mark_prefetch_started(2, in_capture=False)
    assert runtime.is_pending_in_capture(2) is False
    assert runtime.pending_capture_prefetches() == ()


def test_prefetch_runtime_begin_prefetch_returns_previous_slot_owner():
    runtime = PrefetchRuntimeController(unit_count=4, prefetch_step=2)

    assert runtime.begin_prefetch(0) is None
    assert runtime.begin_prefetch(1) is None

    previous_owner = runtime.begin_prefetch(2)
    assert previous_owner is not None
    assert (previous_owner.unit_idx, previous_owner.slot_idx) == (0, 0)

    previous_owner = runtime.begin_prefetch(3)
    assert previous_owner is not None
    assert (previous_owner.unit_idx, previous_owner.slot_idx) == (1, 1)


def test_prefetch_runtime_tracks_unit_residency():
    runtime = PrefetchRuntimeController(unit_count=4, prefetch_step=2)

    assert runtime.is_unit_resident(0) is False
    assert runtime.is_unit_resident(2) is False

    assert runtime.begin_prefetch(0) is None
    assert runtime.is_unit_resident(0) is True
    assert runtime.is_unit_resident(2) is False

    previous_owner = runtime.begin_prefetch(2)
    assert previous_owner is not None
    assert previous_owner.unit_idx == 0
    assert runtime.is_unit_resident(0) is False
    assert runtime.is_unit_resident(2) is True


def test_prefetch_runtime_reset_clears_slots_and_capture_state():
    runtime = PrefetchRuntimeController(unit_count=4, prefetch_step=2)

    runtime.begin_prefetch(0)
    runtime.begin_prefetch(1)
    runtime.mark_prefetch_started(0, in_capture=True)
    runtime.mark_prefetch_started(1, in_capture=True)

    assert runtime.is_unit_resident(0) is True
    assert runtime.is_unit_resident(1) is True
    assert runtime.pending_capture_prefetches()

    runtime.reset()

    assert runtime.is_unit_resident(0) is False
    assert runtime.is_unit_resident(1) is False
    assert runtime.pending_capture_prefetches() == ()
    assert runtime.begin_prefetch(0) is None
    assert runtime.is_unit_resident(0) is True


@pytest.mark.parametrize(
    ("unit_count", "prefetch_step", "expected_initial", "expected_after_zero"),
    [
        (0, 3, (), None),
        (1, 3, ((0, 0),), None),
        (2, 3, ((0, 0), (1, 1)), None),
    ],
)
def test_prefetch_runtime_handles_layer_count_edge_cases(
    unit_count: int,
    prefetch_step: int,
    expected_initial: tuple[tuple[int, int], ...],
    expected_after_zero: tuple[int, int] | None,
):
    runtime = PrefetchRuntimeController(
        unit_count=unit_count,
        prefetch_step=prefetch_step,
    )

    assert (
        tuple((unit.unit_idx, unit.slot_idx) for unit in runtime.initial_prefetches())
        == expected_initial
    )

    next_unit = runtime.prefetch_after(0)
    if expected_after_zero is None:
        assert next_unit is None
    else:
        assert next_unit is not None
        assert (next_unit.unit_idx, next_unit.slot_idx) == expected_after_zero
