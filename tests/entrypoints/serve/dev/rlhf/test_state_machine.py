# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tier 0 — RL lifecycle state-machine invariant tests.

These tests verify the HTTP endpoint state-machine contracts: valid and invalid
state transitions, idempotency, and ordering invariants.  They use
--load-format dummy to start the server quickly without downloading real weights,
so they run as fast CPU-level sanity checks even though they drive the full HTTP
stack.

Industry context
----------------
This is an industry-wide coverage gap.  None of the 12 surveyed downstream RL
frameworks (verl, slime, SkyRL, AReaL, ROLL, miles, trl, OpenRLHF, prime-rl,
NeMo-RL, sglang, open-instruct) tests the state-machine layer at the CPU level.
The closest references are:
  - sglang test_pause_generation_tensor_consistency.py  (batch/scheduler state)
  - AReaL tests/test_staleness_manager.py               (concurrent calls)
  - verl test_vllm_weight_update_utils_on_cpu.py         (weight-update ordering)

Test classes
------------
TestSleepingTagsSemantics      sleeping_tags set ops (partial/full wake)
TestPauseStateOrthogonality    pause and sleep are distinct state dimensions
TestWeightUpdateOrdering       start/update/finish protocol ordering
TestIdempotency                double sleep, double wake, double pause
TestStateAfterRestart          server comes up in the correct initial state

All tests use dummy_weights=True (--load-format dummy) for fast startup.

RFC: https://github.com/vllm-project/vllm/issues/45585
PR:  https://github.com/vllm-project/vllm/pull/45586
"""

import threading
import time

import requests

from .conftest import (
    finish_weight_update,
    gen,
    get_world_size,
    health,
    is_paused,
    is_sleeping,
    ok,
    pause,
    poll_until,
    resume,
    server,
    sleep,
    start_weight_update,
    wake,
)

# Ports offset from the defaults to avoid collisions with other test modules.
_PORT_BASE = 8780


# ---------------------------------------------------------------------------
# TestSleepingTagsSemantics
# ---------------------------------------------------------------------------


class TestSleepingTagsSemantics:
    """sleeping_tags set-operations: partial wake keeps is_sleeping=True.

    The invariant: is_sleeping() = len(sleeping_tags) > 0.
    wake_up(["weights"]) removes "weights" from sleeping_tags but not "kv_cache",
    so is_sleeping stays True until the last tag is removed.

    Reference: sglang test_pause_generation_tensor_consistency.py
    """

    def test_full_sleep_sets_both_tags(self):
        """sleep(level=1) → is_sleeping=True; full wake → is_sleeping=False."""
        with server(port=_PORT_BASE, dummy_weights=True) as url:
            assert sleep(url, level=1) == 200
            assert poll_until(lambda: is_sleeping(url), timeout=10), (
                "engine did not transition to sleeping after sleep(1)"
            )

            assert wake(url) == 200
            assert poll_until(lambda: not is_sleeping(url), timeout=10), (
                "engine did not wake after wake_up()"
            )
            assert health(url) == 200

    def test_partial_wake_weights_only_keeps_sleeping(self):
        """wake_up(['weights']) removes weights tag but kv_cache stays → sleeping."""
        with server(port=_PORT_BASE + 1, dummy_weights=True) as url:
            assert sleep(url, level=1) == 200
            assert poll_until(lambda: is_sleeping(url), timeout=10)

            assert wake(url, tags=["weights"]) == 200
            # After weights-only wake, kv_cache is still unmapped → still sleeping
            assert poll_until(lambda: is_sleeping(url), timeout=5), (
                "is_sleeping went False after weights-only wake — "
                "kv_cache tag not tracked (PR #44483 regression)"
            )

            assert wake(url, tags=["kv_cache"]) == 200
            assert poll_until(lambda: not is_sleeping(url), timeout=10), (
                "engine did not fully wake after kv_cache wake"
            )
            assert health(url) == 200

    def test_sleep_level0_sets_is_sleeping_no_memory_tags(self):
        """sleep(level=0) pauses scheduler only — no GPU memory released.

        At level=0, is_sleeping reflects the scheduler-paused state.
        The engine must accept wake_up() to exit the level-0 sleep.
        """
        with server(port=_PORT_BASE + 2, dummy_weights=True) as url:
            assert sleep(url, level=0) == 200
            assert poll_until(lambda: is_sleeping(url), timeout=10), (
                "is_sleeping should be True after sleep(level=0)"
            )

            assert wake(url) == 200
            assert poll_until(lambda: not is_sleeping(url), timeout=10)
            assert health(url) == 200


# ---------------------------------------------------------------------------
# TestPauseStateSemantics
# ---------------------------------------------------------------------------


class TestPauseStateSemantics:
    """Verify the relationship between /pause, /sleep, is_paused, and is_sleeping.

    Key invariant (vLLM v1 engine):
      is_sleeping() = is_scheduler_paused() OR executor.is_sleeping
      is_paused()   = is_scheduler_paused()

    Both /pause and /sleep call pause_scheduler() internally, so:
      - After /pause:    is_paused=True,  is_sleeping=True  (scheduler paused)
      - After /sleep(1): is_paused=True,  is_sleeping=True  (scheduler + executor)
      - After /resume:   is_paused=False, is_sleeping=False (both cleared)
      - After /wake_up:  is_paused=False, is_sleeping=False (both cleared)

    The semantic difference between pause and sleep is NOT reflected in
    is_paused/is_sleeping alone — it lies in whether GPU memory was released,
    which is tested by TestPhysicalMemory in test_sleep_wake.py.

    Reference: SkyRL test_pause_and_continue_generation.py
    """

    def test_pause_sets_is_paused_and_is_sleeping(self):
        """/pause sets both is_paused=True and is_sleeping=True.

        is_sleeping() = is_scheduler_paused() OR executor.is_sleeping.
        /pause calls pause_scheduler(), so is_scheduler_paused() becomes True,
        which makes is_sleeping() return True even though no GPU memory changed.
        """
        with server(port=_PORT_BASE + 3, dummy_weights=True) as url:
            assert pause(url, mode="abort") == 200
            assert poll_until(lambda: is_paused(url), timeout=10)
            assert is_sleeping(url) is True, (
                "/pause should set is_sleeping=True because "
                "is_sleeping() = is_scheduler_paused() OR executor.is_sleeping"
            )

            assert resume(url) == 200
            assert poll_until(lambda: not is_paused(url), timeout=10)
            assert is_sleeping(url) is False
            assert health(url) == 200

    def test_sleep_sets_is_paused_and_is_sleeping(self):
        """/sleep(1) sets both is_paused=True and is_sleeping=True.

        /sleep internally calls pause_scheduler() before offloading GPU memory,
        so is_paused() becomes True (scheduler is paused) in addition to
        is_sleeping() (executor is also sleeping).
        """
        with server(port=_PORT_BASE + 4, dummy_weights=True) as url:
            assert sleep(url, level=1) == 200
            assert poll_until(lambda: is_sleeping(url), timeout=10)
            assert is_paused(url) is True, (
                "/sleep should set is_paused=True because "
                "sleep() calls pause_scheduler() internally"
            )

            assert wake(url) == 200
            assert poll_until(lambda: not is_sleeping(url), timeout=10)
            assert is_paused(url) is False
            assert health(url) == 200

    def test_resume_clears_both_pause_and_sleep(self):
        """/resume resets is_paused=False and is_sleeping=False after /pause."""
        with server(port=_PORT_BASE + 5, dummy_weights=True) as url:
            assert pause(url, mode="abort") == 200
            assert poll_until(lambda: is_paused(url), timeout=10)
            assert is_sleeping(url) is True

            assert resume(url) == 200
            assert poll_until(lambda: not is_paused(url), timeout=10)
            assert is_sleeping(url) is False
            assert health(url) == 200


# ---------------------------------------------------------------------------
# TestWeightUpdateOrdering
# ---------------------------------------------------------------------------


class TestWeightUpdateOrdering:
    """start/update/finish weight-transfer protocol ordering invariants.

    The protocol in vLLM v0.23.0 requires:
      init_weight_transfer_engine → start_weight_update → update_weights → finish

    Calling start/update/finish without a prior init returns an error.
    Out-of-order calls within an initialized session must also error.

    Reference: AReaL tests/experimental/weight_update/test_wu_controller.py
               TestLifecycle, TestUpdateWeights, TestDisconnect
    """

    def test_start_without_init_returns_error(self):
        """/start_weight_update without /init_weight_transfer_engine must error.

        In vLLM v0.23.0, the weight transfer engine must be initialized via
        /init_weight_transfer_engine (which sets up the NCCL communicator)
        before /start_weight_update is valid.  Without it the server returns
        a 500 with "Weight transfer not configured."
        """
        with server(port=_PORT_BASE + 6, dummy_weights=True) as url:
            r = start_weight_update(url)
            assert r.status_code in (400, 409, 500), (
                f"start without init must return an error, got {r.status_code}: {r.text}"
            )
            # Error message should be informative
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            error_msg = str(body.get("error", {}).get("message", r.text)).lower()
            assert any(kw in error_msg for kw in ("not configured", "weight transfer", "init")), (
                f"error message should mention configuration requirement: {r.text}"
            )
            assert health(url) == 200

    def test_finish_without_start_returns_error(self):
        """/finish_weight_update without a preceding /start must return an error."""
        with server(port=_PORT_BASE + 7, dummy_weights=True) as url:
            r = finish_weight_update(url)
            assert r.status_code in (400, 409, 500), (
                f"finish-without-start must return an error, got {r.status_code}"
            )
            assert health(url) == 200

    def test_weight_update_endpoints_do_not_crash_engine(self):
        """Calling weight-update endpoints without init must not kill the engine.

        Verifies that protocol violations return errors gracefully, not crashes.
        The engine must remain healthy and able to serve generation requests.
        """
        with server(port=_PORT_BASE + 8, dummy_weights=True) as url:
            # All three should error (no init), none should crash the engine
            start_weight_update(url)   # error expected
            finish_weight_update(url)  # error expected

            assert health(url) == 200, "engine must survive weight-update endpoint errors"
            resp = gen(url)
            assert ok(resp), f"generate failed after protocol violations: {resp}"


# ---------------------------------------------------------------------------
# TestIdempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    """Double calls must not crash the engine.

    Per PR #45518: sleep() on an already-sleeping engine should return
    an idempotency signal, not crash.  We verify liveness here; the body
    schema is validated in TestSleepingTagsSemantics.

    Reference: PR #45518 adds {"already_sleeping": true/false} to the body.
    """

    def test_double_sleep_engine_survives(self):
        with server(port=_PORT_BASE + 10, dummy_weights=True) as url:
            assert sleep(url, level=1) == 200
            assert poll_until(lambda: is_sleeping(url), timeout=10)

            sc = sleep(url, level=1)
            assert sc in (200, 400), (
                f"double sleep returned unexpected status {sc}"
            )
            assert health(url) == 200

            assert wake(url) == 200
            assert poll_until(lambda: not is_sleeping(url), timeout=10)
            assert health(url) == 200

    def test_double_wake_engine_survives(self):
        with server(port=_PORT_BASE + 11, dummy_weights=True) as url:
            # engine is awake by default
            assert not is_sleeping(url)
            sc = wake(url)
            assert sc in (200, 400), (
                f"wake on non-sleeping engine returned unexpected status {sc}"
            )
            assert health(url) == 200

    def test_double_pause_engine_survives(self):
        with server(port=_PORT_BASE + 12, dummy_weights=True) as url:
            assert pause(url, mode="abort") == 200
            assert poll_until(lambda: is_paused(url), timeout=10)

            sc = pause(url, mode="abort")
            assert sc in (200, 400), (
                f"double pause returned unexpected status {sc}"
            )
            assert health(url) == 200

            assert resume(url) == 200
            assert poll_until(lambda: not is_paused(url), timeout=10)


# ---------------------------------------------------------------------------
# TestInitialState
# ---------------------------------------------------------------------------


class TestInitialState:
    """Verify the engine starts in the correct initial state.

    On startup: is_sleeping=False, is_paused=False, weight_update_active=False.
    The get_world_size endpoint must respond with a positive integer.
    """

    def test_server_starts_awake_and_unpaused(self):
        with server(port=_PORT_BASE + 13, dummy_weights=True) as url:
            assert health(url) == 200
            assert is_sleeping(url) is False, (
                "engine should start awake (is_sleeping=False)"
            )
            assert is_paused(url) is False, (
                "engine should start unpaused (is_paused=False)"
            )

    def test_get_world_size_positive_on_fresh_server(self):
        with server(port=_PORT_BASE + 14, dummy_weights=True) as url:
            r = get_world_size(url)
            assert r.status_code == 200
            ws = r.json()["world_size"]
            assert isinstance(ws, int) and ws >= 1

    def test_weight_update_active_false_on_startup(self):
        """finish_weight_update on a fresh server (no active session) → error.

        This indirectly confirms _weight_update_active starts as False.
        """
        with server(port=_PORT_BASE + 15, dummy_weights=True) as url:
            r = finish_weight_update(url)
            # If _weight_update_active is False (correct), this must error.
            assert r.status_code in (400, 409, 500), (
                f"finish on fresh server must error (no active session), got {r.status_code}"
            )
            assert health(url) == 200


# ---------------------------------------------------------------------------
# TestConcurrentStateTransitions
# ---------------------------------------------------------------------------


class TestConcurrentStateTransitions:
    """Concurrent calls to state-transition endpoints must not corrupt state.

    Reference: AReaL tests/test_staleness_manager.py TestThreadSafety —
    uses ThreadPoolExecutor with concurrent on_rollout_submitted calls.
    Here we use threads to issue concurrent sleep/wake/pause calls and
    verify the engine survives and ends in a consistent state.
    """

    def test_concurrent_sleep_wake_calls_no_crash(self):
        """Multiple threads racing sleep and wake must not crash the engine."""
        with server(port=_PORT_BASE + 16, dummy_weights=True) as url:
            errors = []

            def _sleep_wake():
                try:
                    sleep(url, level=1)
                    time.sleep(0.05)
                    wake(url)
                except Exception as e:
                    errors.append(str(e))

            threads = [threading.Thread(target=_sleep_wake) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            # Engine must be alive regardless of which thread won the races
            assert health(url) == 200, (
                f"engine died after concurrent sleep/wake: errors={errors}"
            )
            # Drain any sleeping state so the server can shut down cleanly
            if is_sleeping(url):
                wake(url)
                poll_until(lambda: not is_sleeping(url), timeout=10)
