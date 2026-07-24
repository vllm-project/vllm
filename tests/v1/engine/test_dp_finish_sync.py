# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DPEngineCoreProc finish-state sync cadence.

Covers the fix that removes the 32-step modulo gate so the DP finish-state
all-reduce (``ParallelConfig.sync_dp_state``) runs on *every* scheduler step.
The multi-node deadlock itself needs real DP ranks and cannot run in CI; these
tests exercise the CPU-only control logic of ``_has_global_unfinished_reqs``
by stubbing the collective, which is where the cadence regression would appear.
"""

from unittest.mock import patch

from vllm.v1.engine.core import DPEngineCoreProc


def _bare_proc(pending_pause: bool = False) -> DPEngineCoreProc:
    """A DPEngineCoreProc with only the attributes the method touches.

    Bypasses __init__ (which builds a real engine + DP process group) because
    the method under test only reads step_counter/dp_group/pending_pause and
    writes step_counter/pending_pause/ignore_start_dp_wave.
    """
    proc = object.__new__(DPEngineCoreProc)
    proc.step_counter = 0
    proc.pending_pause = pending_pause
    proc.ignore_start_dp_wave = False
    proc.dp_group = object()  # opaque; sync_dp_state is stubbed
    return proc


def test_sync_runs_every_step_not_every_32():
    """The fix: sync_dp_state fires once per call (regression guard for the
    old ``step_counter % 32`` gate that skipped 31 of every 32 syncs)."""
    proc = _bare_proc()
    with patch(
        "vllm.v1.engine.core.ParallelConfig.sync_dp_state",
        return_value=(True, False),
    ) as sync:
        for _ in range(35):
            proc._has_global_unfinished_reqs(local_unfinished=True)
    # Every-step cadence: 35 calls -> 35 syncs. The old gate would give 1.
    assert sync.call_count == 35
    assert proc.step_counter == 35


def test_returns_global_unfinished_from_sync():
    """Return value is the global has_unfinished from the all-reduce, not the
    local flag (a rank with no local work must still step if peers have work)."""
    proc = _bare_proc()
    with patch(
        "vllm.v1.engine.core.ParallelConfig.sync_dp_state",
        return_value=(True, False),
    ):
        # local_unfinished=False but global says True -> must return True.
        assert proc._has_global_unfinished_reqs(local_unfinished=False) is True
    with patch(
        "vllm.v1.engine.core.ParallelConfig.sync_dp_state",
        return_value=(False, False),
    ):
        assert proc._has_global_unfinished_reqs(local_unfinished=True) is False


def test_pause_consensus_sets_ignore_start_dp_wave():
    """When the all-reduce reports pause consensus, the rank latches
    ignore_start_dp_wave and clears pending_pause so stale START_DP_WAVE
    messages cannot re-wake it."""
    proc = _bare_proc(pending_pause=True)
    with patch(
        "vllm.v1.engine.core.ParallelConfig.sync_dp_state",
        return_value=(False, True),
    ):
        proc._has_global_unfinished_reqs(local_unfinished=False)
    assert proc.ignore_start_dp_wave is True
    assert proc.pending_pause is False


def test_no_pause_consensus_leaves_flags_untouched():
    """Without consensus, pause state is preserved across the step (the rank
    keeps waiting for peers rather than latching early)."""
    proc = _bare_proc(pending_pause=True)
    with patch(
        "vllm.v1.engine.core.ParallelConfig.sync_dp_state",
        return_value=(True, False),
    ):
        proc._has_global_unfinished_reqs(local_unfinished=True)
    assert proc.ignore_start_dp_wave is False
    assert proc.pending_pause is True


def test_sync_receives_current_pending_pause():
    """The local pending_pause flag is forwarded into the all-reduce so the
    consensus count reflects this rank's request."""
    proc = _bare_proc(pending_pause=True)
    with patch(
        "vllm.v1.engine.core.ParallelConfig.sync_dp_state",
        return_value=(False, False),
    ) as sync:
        proc._has_global_unfinished_reqs(local_unfinished=False)
    _, kwargs = sync.call_args
    assert kwargs["pending_pause"] is True
    assert kwargs["has_unfinished"] is False
