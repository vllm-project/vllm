# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end tests for the vLLM RL sleep/wake lifecycle.

Endpoint surface under test
---------------------------
sleep/api_router  : POST /sleep  POST /wake_up  GET /is_sleeping

All tests require:
  --enable-sleep-mode   KV cache allocated via CuMemAllocator; without this
                        flag sleep/wake are no-ops and the bug cannot trigger.
  VLLM_SERVER_DEV_MODE=1

Server/HTTP helpers come from the shared rlhf conftest — this module must
live under tests/entrypoints/serve/dev/rlhf/ for the import to resolve.

Runner matrix (MRV1 / MRV2)
---------------------------
Every test runs twice — once per model runner — via the ``use_v2``
parametrized fixture, which sets VLLM_USE_V2_MODEL_RUNNER explicitly for
each round (both values must be explicit: MRV2 is the default on recent
vLLM, so an unset variable would silently run both rounds on MRV2).

Mechanism-layer thresholds (TestPhysicalMemory, TestSleepWakeLatency,
TestMemoryLeakCycle) are MRV1-calibrated and shared across both rounds.
Triage the first MRV2 run before splitting any threshold: a contract-layer
failure on MRV2 is a bug (the sleep-mode contract lives in shared engine /
worker code), while a mechanism-layer failure needs a baseline comparison
first — split _THRESHOLDS per runner only for proven-legitimate shifts.

Test classes (ordered by increasing complexity)
------------------------------------------------
TestSleepWakeFlags              flag/metrics smoke test (supersedes test_sleep.py)
TestPhysicalMemory              GPU free-bytes per stage
TestSchedulerGate               scheduler must not dispatch during partial/full wake
TestOutputCorrectness           golden-output roundtrip across lifecycle
TestErrorPaths                  idempotency, wrong order, abort interaction
TestConcurrentRace              concurrent sleep + generate must not deadlock
TestMemoryLeakCycle             GPU memory stable across repeated cycles
TestAbortDuringParallelSampling abort with n>1 in flight must not crash
TestLogprobsPrecision           logprobs consistent across sleep/wake
TestSleepWakeLatency            sleep/wake completes within time bounds
TestStagedWakeCycles            staged wake keeps output stable across cycles

RFC: https://github.com/vllm-project/vllm/issues/45585
Guards against the regression introduced by: https://github.com/vllm-project/vllm/pull/44483
"""

import contextlib
import os
import threading
import time
from unittest.mock import patch

import pytest
import requests

from tests.entrypoints.serve.dev.rlhf.conftest import (
    gen,
    gpu_free_bytes,
    health,
    is_sleeping,
    ok,
    server,
    sleep,
    sleep_metrics,
    wake,
)

# ---------------------------------------------------------------------------
# Mechanism-layer thresholds
# ---------------------------------------------------------------------------

# MRV1-calibrated thresholds, shared across both runner rounds.  After the
# first MRV2 run has been triaged, any metric proven to shift legitimately
# between runners gets per-runner entries (keyed by use_v2) instead of a
# single shared value.
_THRESHOLDS = {
    "level1_freed_gib": 0.5,  # sleep(1) offloads weights (~1.2 GiB observed)
    "level2_freed_gib": 1.5,  # sleep(2) discards weights + KV cache
    "level0_freed_gib": 0.5,  # level 0 must NOT release more than this
    "wake_reallocated_gib": 0.4,  # full wake remaps weights
    "leak_tolerance_gib": 0.05,  # max drift across 10 cycles
    "full_roundtrip_s": 10.0,  # sleep(1) + full wake
    "staged_roundtrip_s": 15.0,  # sleep(1) + staged wake (weights → kv_cache)
    "five_cycles_s": 60.0,  # 5 full sleep/wake cycles
}


# ---------------------------------------------------------------------------
# MRV1/MRV2 dual-runner fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=[False, True], ids=["MRV1", "MRV2"])
def use_v2(request):
    """Run the whole suite twice: model runner v1 and v2."""
    return request.param


def _runner_env(use_v2: bool) -> dict:
    # Both values must be set explicitly: recent vLLM defaults to MRV2, so
    # an unset variable would silently run the "MRV1" round on MRV2 too.
    return {"VLLM_USE_V2_MODEL_RUNNER": "1" if use_v2 else "0"}


@pytest.fixture(scope="class")
def shared_server_url(use_v2):
    """One server per stateless class × runner round."""
    with patch.dict(os.environ, _runner_env(use_v2)), server() as url:
        yield url


@pytest.fixture
def isolated_server_url(use_v2):
    """Fresh server per test — for state-sensitive classes (memory/latency)."""
    with patch.dict(os.environ, _runner_env(use_v2)), server() as url:
        yield url


@pytest.fixture
def seeded_server_url(use_v2):
    """Fresh server with --seed 42 — deterministic staged-wake cycle test."""
    with (
        patch.dict(os.environ, _runner_env(use_v2)),
        server(extra_args=["--seed", "42"]) as url,
    ):
        yield url


class _SharedServerTests:
    """Base for stateless classes sharing one server per runner round.

    The autouse fixture guarantees every test starts and ends awake, so a
    mid-test failure cannot poison the next test on the shared server.
    """

    @pytest.fixture(autouse=True)
    def _restore_awake_state(self, shared_server_url):
        # 400 is acceptable: wake_up while already awake (cf. TestErrorPaths).
        assert wake(shared_server_url) in (200, 400)
        yield
        assert wake(shared_server_url) in (200, 400)


# ---------------------------------------------------------------------------
# TestSleepWakeFlags
# ---------------------------------------------------------------------------


class TestSleepWakeFlags(_SharedServerTests):
    """Smoke-level flag and metrics checks.

    These mirror the original test_sleep.py assertions so that file can
    eventually be removed.  Kept separate so a CI bisect can isolate quickly.
    """

    def test_sleep_sets_is_sleeping_and_metrics(self, shared_server_url):
        url = shared_server_url
        assert sleep(url, level=1) == 200
        assert is_sleeping(url) is True

        awake, weights_offloaded, discard_all = sleep_metrics(url)
        assert awake == 0 and weights_offloaded == 1 and discard_all == 0

        assert wake(url) == 200
        assert is_sleeping(url) is False

        awake, weights_offloaded, discard_all = sleep_metrics(url)
        assert awake == 1 and weights_offloaded == 0 and discard_all == 0

    def test_level1_sets_weights_offloaded_metric(self, shared_server_url):
        """sleep(level=1) offloads weights — weights_offloaded gauge must flip."""
        url = shared_server_url
        assert sleep(url, level=1) == 200
        _, weights_offloaded, _ = sleep_metrics(url)
        assert weights_offloaded == 1

        assert wake(url) == 200
        assert health(url) == 200

    def test_level2_sets_discard_all_metric(self, shared_server_url):
        url = shared_server_url
        assert sleep(url, level=2) == 200
        _, _, discard_all = sleep_metrics(url)
        assert discard_all == 1

        assert wake(url) == 200
        assert is_sleeping(url) is False
        assert health(url) == 200

    def test_level0_sets_nosleep_metrics(self, shared_server_url):
        """level=0 pauses scheduling only — it must not set sleep metrics."""
        url = shared_server_url
        assert sleep(url, level=0) == 200
        _, weights_offloaded, discard_all = sleep_metrics(url)
        assert weights_offloaded == 0 and discard_all == 0

        assert wake(url) == 200
        assert health(url) == 200

    def test_staged_wake_keeps_is_sleeping_true(self, shared_server_url):
        """wake_up(["weights"]) must leave is_sleeping True until kv_cache wakes."""
        url = shared_server_url
        assert sleep(url, level=1) == 200
        assert wake(url, tags=["weights"]) == 200
        assert is_sleeping(url) is True  # still partial

        assert wake(url, tags=["kv_cache"]) == 200
        assert is_sleeping(url) is False


# ---------------------------------------------------------------------------
# TestPhysicalMemory
# ---------------------------------------------------------------------------


class TestPhysicalMemory:
    """Assert GPU free bytes change, not just flags.

    Guards against regressions where CuMemAllocator.sleep() silently no-ops
    (e.g. missing stream-sync, wrong tag registration) while returning 200.
    """

    def test_sleep_level1_frees_gpu_memory(self, isolated_server_url):
        url = isolated_server_url
        gen(url)  # warm up — allocate KV blocks
        free_awake = gpu_free_bytes()

        assert sleep(url, level=1) == 200
        free_sleeping = gpu_free_bytes()
        freed_gib = (free_sleeping - free_awake) / 2**30

        assert freed_gib > _THRESHOLDS["level1_freed_gib"], (
            f"sleep(1) freed only {freed_gib:.2f} GiB — "
            "CuMemAllocator unmap may be a no-op"
        )
        assert wake(url) == 200
        free_awake2 = gpu_free_bytes()
        re_allocated_gib = (free_sleeping - free_awake2) / 2**30
        assert re_allocated_gib > _THRESHOLDS["wake_reallocated_gib"], (
            f"wake_up re-allocated only {re_allocated_gib:.2f} GiB — "
            "remap may be incomplete"
        )

    def test_sleep_level2_frees_weights_and_kv(self, isolated_server_url):
        url = isolated_server_url
        gen(url)
        free_awake = gpu_free_bytes()

        assert sleep(url, level=2) == 200
        freed_gib = (gpu_free_bytes() - free_awake) / 2**30
        assert freed_gib > _THRESHOLDS["level2_freed_gib"]

        assert wake(url) == 200
        assert health(url) == 200

    def test_sleep_level0_does_not_release_memory(self, isolated_server_url):
        """level=0 pauses scheduling only — no GPU memory change."""
        url = isolated_server_url
        gen(url)  # warm up — allocate KV blocks
        free_before = gpu_free_bytes()

        assert sleep(url, level=0) == 200
        freed_gib = (gpu_free_bytes() - free_before) / 2**30
        assert freed_gib < _THRESHOLDS["level0_freed_gib"], (
            f"sleep(0) released {freed_gib:.2f} GiB — "
            "it should only pause scheduling, not free GPU memory"
        )

        assert wake(url) == 200
        assert health(url) == 200

    def test_staged_wake_allocates_memory_per_tag(self, isolated_server_url):
        """Each staged-wake tag re-allocates a distinct chunk of GPU memory."""
        url = isolated_server_url
        gen(url)
        assert sleep(url, level=1) == 200

        assert wake(url, tags=["weights"]) == 200
        free_after_weights = gpu_free_bytes()

        assert wake(url, tags=["kv_cache"]) == 200
        free_after_kv = gpu_free_bytes()

        # waking kv_cache consumes more GPU memory than weights-only wake
        assert free_after_kv < free_after_weights, (
            "waking kv_cache should use more GPU memory than weights-only wake"
        )
        assert health(url) == 200


# ---------------------------------------------------------------------------
# TestSchedulerGate
# ---------------------------------------------------------------------------


class TestSchedulerGate(_SharedServerTests):
    """The scheduler must not dispatch work while memory is unmapped.

    The core of this class is the PR #44483 regression test.  Without the fix,
    wake_up(["weights"]) unconditionally called resume_scheduler(), allowing
    FlashAttention to launch on the still-unmapped kv_cache VA —
    flash_fwd_launch_template.h:199 TMA descriptor 700 illegal memory access.
    """

    def test_partial_wake_blocks_until_kv_resident(self, shared_server_url):
        """PR #44483 regression test.

        sleep(1) → wake_up(["weights"]) → generate must NOT execute FA on the
        unmapped kv_cache VA.  With the fix the scheduler holds the request
        until wake_up(["kv_cache"]) completes; without the fix the engine dies
        (health goes to 503) within 3 seconds.
        """
        url = shared_server_url
        assert sleep(url, level=1) == 200
        assert wake(url, tags=["weights"]) == 200
        assert is_sleeping(url) is True  # kv_cache still unmapped

        result: dict = {}

        def _bg():
            # short timeout — should block, not complete, while kv_cache is asleep
            result["resp"] = gen(url, timeout=8)

        t = threading.Thread(target=_bg)
        t.start()
        time.sleep(3)

        # engine must be alive — if it died the fix is missing
        assert health(url) == 200, (
            "engine died during partial-wake window — "
            "IMA from running FlashAttention on unmapped kv_cache "
            "(PR #44483 regression)"
        )

        assert wake(url, tags=["kv_cache"]) == 200
        assert is_sleeping(url) is False
        t.join(timeout=30)
        assert health(url) == 200

        resp = gen(url)
        assert resp and ok(resp)

    def test_sleep_abort_mode_blocks_new_requests(self, shared_server_url):
        """sleep(mode='abort') pauses the scheduler.

        New requests must not return successful completions while sleeping.
        They must also not hang forever (cf. #45326).
        """
        url = shared_server_url
        assert sleep(url, level=1, mode="abort") == 200
        assert is_sleeping(url) is True

        resp = gen(url, timeout=5)
        assert not ok(resp), (
            "generate returned a successful result while engine was sleeping "
            "(scheduler not paused — or request hangs without aborting)"
        )
        # engine still alive, not dead
        assert health(url) == 200

        assert wake(url) == 200
        assert ok(gen(url))

    def test_sleep_level0_blocks_new_requests(self, shared_server_url):
        """level=0 = pause only; same scheduler gate as sleep(1)."""
        url = shared_server_url
        assert sleep(url, level=0) == 200
        assert is_sleeping(url) is True

        resp = gen(url, timeout=5)
        assert not ok(resp)
        assert health(url) == 200

        assert wake(url) == 200
        assert ok(gen(url))


# ---------------------------------------------------------------------------
# TestOutputCorrectness
# ---------------------------------------------------------------------------


class TestOutputCorrectness(_SharedServerTests):
    """Output must be deterministic and self-consistent across the lifecycle."""

    def test_full_wake_restores_output(self, shared_server_url):
        """sleep(1) → wake_up() — weights restored; output matches golden."""
        url = shared_server_url
        golden = gen(url)
        assert golden
        golden_text = golden["choices"][0]["text"]

        assert sleep(url, level=1) == 200
        assert wake(url) == 200

        resp = gen(url)
        assert resp and resp["choices"][0]["text"] == golden_text, (
            "output changed after sleep/wake — weight restore broken"
        )

    def test_staged_wake_restores_output(self, shared_server_url):
        """sleep → wake(weights) → wake(kv_cache) — output matches golden."""
        url = shared_server_url
        golden_text = gen(url)["choices"][0]["text"]

        assert sleep(url, level=1) == 200
        assert wake(url, tags=["weights"]) == 200
        assert wake(url, tags=["kv_cache"]) == 200

        resp = gen(url)
        assert resp and resp["choices"][0]["text"] == golden_text

    def test_multiple_cycles_stable(self, shared_server_url):
        """3× sleep/wake cycles — output and engine stay stable.

        Guards against cumem bookkeeping corruption across repeated
        release+remap of the same physical pages.
        """
        url = shared_server_url
        golden_text = gen(url)["choices"][0]["text"]

        for i in range(3):
            assert sleep(url, level=1) == 200
            assert wake(url) == 200
            assert health(url) == 200

            resp = gen(url)
            assert resp and resp["choices"][0]["text"] == golden_text, (
                f"output drifted on cycle {i} — cumem bookkeeping corrupted"
            )

    def test_cached_prompt_generate_ok_afterwake(self, shared_server_url):
        """Generate with a previously-cached prompt must succeed after wake.

        wake_up() resets the prefix cache; if it were not flushed, a
        subsequent sleep cycle could reuse a stale entry pointing to an
        already-released physical page.
        """
        url = shared_server_url
        prompt = "The capital of France is"
        gen(url, prompt=prompt)  # populate prefix cache

        assert sleep(url, level=1) == 200
        assert wake(url) == 200
        assert health(url) == 200

        resp = gen(url, prompt=prompt)
        assert ok(resp), (
            "generate failed after wake with cached prompt — "
            "possible stale prefix-cache IMA"
        )


# ---------------------------------------------------------------------------
# TestErrorPaths
# ---------------------------------------------------------------------------


class TestErrorPaths(_SharedServerTests):
    """Protocol violations must not crash the engine."""

    def test_double_sleep_idempotent(self, shared_server_url):
        """sleep() while already sleeping must not crash.

        Per PR #45518, response body carries {"already_sleeping": true/false}.
        We assert only liveness here; the body contract is validated separately.
        """
        url = shared_server_url
        assert sleep(url, level=1) == 200
        sc = sleep(url, level=1)  # idempotent call
        assert sc in (200, 400), f"double sleep returned unexpected {sc}"

        assert wake(url) == 200
        assert health(url) == 200
        assert ok(gen(url))

    def test_wake_while_awake_idempotent(self, shared_server_url):
        url = shared_server_url
        gen(url)
        assert is_sleeping(url) is False
        sc = wake(url)
        assert sc in (200, 400)
        assert health(url) == 200
        assert ok(gen(url))

    def test_abort_then_sleepwake(self, shared_server_url):
        """Abort a mid-flight request, then sleep → wake.  Engine must survive.

        Simulates the colocate_async partial-rollout pattern.
        """
        url = shared_server_url

        def _bg():
            with contextlib.suppress(Exception):
                requests.post(
                    f"{url}/v1/completions",
                    json={
                        "model": "m",
                        "prompt": "x" * 200,
                        "max_tokens": 256,
                        "temperature": 0,
                    },
                    timeout=1,
                )

        threading.Thread(target=_bg).start()
        time.sleep(0.3)

        assert sleep(url, level=1) == 200
        assert wake(url) == 200
        assert health(url) == 200
        assert ok(gen(url))


# ---------------------------------------------------------------------------
# TestConcurrentRace
# ---------------------------------------------------------------------------


class TestConcurrentRace:
    """Concurrent sleep + generate threads must not deadlock or crash the engine.

    Reference: miles tests/fast/router/test_session_race_conditions.py
               TestSessionConcurrencyContracts (4 tests) +
               TestClosingRaceConditions (5 tests) — 8-thread ThreadPoolExecutor
    """

    def test_concurrent_sleep_and_generate_no_deadlock(self, isolated_server_url):
        """10 generate threads racing against 1 sleep/wake thread.

        The engine must survive and remain healthy after the race window.
        We don't assert that all generates succeed (some will be aborted by
        the sleep), but we do assert that:
          - no thread hangs indefinitely (join timeout 30 s)
          - engine is alive at the end
          - a fresh generate after the race succeeds
        """
        url = isolated_server_url
        results = []
        errors = []

        def _gen_thread():
            try:
                r = gen(url, max_tokens=8, timeout=10)
                results.append(r)
            except Exception as e:
                errors.append(str(e))

        def _sleep_wake_thread():
            try:
                sleep(url, level=1, mode="abort")
                time.sleep(0.2)
                wake(url)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=_gen_thread) for _ in range(10)]
        sw = threading.Thread(target=_sleep_wake_thread)

        for t in threads:
            t.start()
        time.sleep(0.1)  # let some generates get in-flight before sleep
        sw.start()

        sw.join(timeout=30)
        for t in threads:
            t.join(timeout=30)

        hung = [i for i, t in enumerate(threads) if t.is_alive()]
        assert not hung, f"generate threads {hung} hung after race (deadlock?)"
        assert not sw.is_alive(), "sleep/wake thread hung after race (deadlock?)"
        assert not errors, f"unexpected exceptions during race: {errors}"
        assert health(url) == 200, f"engine died after concurrent race; errors={errors}"
        # After the race the engine must be able to serve requests
        assert ok(gen(url)), "generate failed after concurrent race"


# ---------------------------------------------------------------------------
# TestMemoryLeakCycle
# ---------------------------------------------------------------------------


class TestMemoryLeakCycle:
    """sleep/wake cycles must not accumulate GPU memory leaks.

    Reference: ROLL tests/third_party/vllm/test_vllm_mem_oom.py
               generate_memory() — 20 iterations tracking memory growth.

    We measure GPU free bytes when the engine is awake (same lifecycle stage
    on every cycle) rather than host RSS, because:
      (a) the vLLM server is a subprocess so its RSS is not directly readable
          from the test process, and
      (b) GPU memory is what cumem manages — leaks manifest there first.
    """

    def test_no_gpu_memory_growth_over_10_cycles(self, isolated_server_url):
        """10 sleep/wake cycles: GPU free bytes (when awake) must be stable.

        After each wake, the engine should have remapped the same GPU pages.
        A growing delta (less free memory each cycle) indicates a leak in
        cumem bookkeeping or handle tracking.
        """
        url = isolated_server_url
        free_samples = []

        for i in range(10):
            gen(url)
            assert sleep(url, level=1) == 200
            assert wake(url) == 200
            assert health(url) == 200

            if i >= 2:  # skip warm-up cycles
                free_samples.append(gpu_free_bytes())

        baseline = free_samples[0]
        # Allow 50 MiB tolerance for KV block allocation jitter
        min_free = min(free_samples)
        leak_gib = (baseline - min_free) / 2**30

        assert leak_gib < _THRESHOLDS["leak_tolerance_gib"], (
            f"GPU free memory shrank by {leak_gib:.3f} GiB over 8 post-warmup "
            f"sleep/wake cycles (baseline={baseline / 2**30:.2f} GiB, "
            f"min={min_free / 2**30:.2f} GiB) — "
            "possible cumem handle leak or unmapped page accumulation"
        )


# ---------------------------------------------------------------------------
# TestAbortDuringParallelSampling
# ---------------------------------------------------------------------------


class TestAbortDuringParallelSampling:
    """sleep(mode='abort') while sampling_n > 1 is in-flight must not crash.

    Reference: ROLL tests/third_party/vllm/test_abort.py
               _check_vllm_sampling_n — cross-version abort semantics
               (CancelledError in V0, finish_reason='abort' in V1 >= 0.10.2)
    """

    def test_sampling_n4_abort_engine_survives(self, isolated_server_url):
        """sampling_n=4 in-flight request aborted by sleep — engine survives.

        With sampling_n=4, the scheduler allocates 4 sequence slots per
        request.  sleep(mode='abort') must drain all 4 cleanly without
        leaving dangling state that blocks the next wake.
        """
        url = isolated_server_url
        result: dict = {}

        def _bg():
            try:
                r = requests.post(
                    f"{url}/v1/completions",
                    json={
                        "model": "m",
                        "prompt": "Count from 1 to 100:",
                        "max_tokens": 64,
                        "temperature": 1.0,
                        "n": 4,
                    },
                    timeout=15,
                )
                result["resp"] = r.json()
            except Exception as e:
                result["err"] = str(e)

        t = threading.Thread(target=_bg)
        t.start()
        time.sleep(0.3)  # let the request start

        assert sleep(url, level=1, mode="abort") == 200
        t.join(timeout=15)

        assert "err" not in result, f"background thread raised: {result.get('err')}"
        assert health(url) == 200, "engine died after sampling_n=4 abort"
        assert wake(url) == 200
        # engine must recover and serve new requests
        assert ok(gen(url)), "generate failed after sampling_n abort + wake"


# ---------------------------------------------------------------------------
# TestLogprobsPrecision
# ---------------------------------------------------------------------------


class TestLogprobsPrecision(_SharedServerTests):
    """logprobs values must be consistent before and after a sleep/wake cycle.

    Reference: ROLL tests/distributed/strategy/log_probs/ (9 files)
               test_fsdp_log_probs_full, test_fsdp_log_probs_cp_rmpad, etc.

    After sleep(level=1)/wake, weights are remapped from CPU backup.
    Calibrated scales (FP8-KV, etc.) must be restored; logprobs must match
    the pre-sleep values within a tight tolerance.
    """

    def test_logprobs_stable_after_sleepwake(self, shared_server_url):
        """logprobs before and after sleep/wake must match within 1e-2.

        Reference: ROLL test_fsdp_log_probs_full — compares log_probs values
        across different parallelism configurations to within tight tolerance.
        """
        url = shared_server_url
        prompt = "The capital of France is Paris and the capital of Germany is"

        def _get_logprobs():
            r = requests.post(
                f"{url}/v1/completions",
                json={
                    "model": "m",
                    "prompt": prompt,
                    "max_tokens": 4,
                    "temperature": 0,
                    "logprobs": 5,
                },
                timeout=30,
            )
            resp = r.json()
            if "choices" not in resp or not resp["choices"]:
                return None
            choice = resp["choices"][0]
            lp = choice.get("logprobs", {})
            return lp.get("token_logprobs", [])

        before = _get_logprobs()
        assert before is not None, "failed to get logprobs before sleep"
        assert len(before) > 0

        assert sleep(url, level=1) == 200
        assert wake(url) == 200
        assert health(url) == 200

        after = _get_logprobs()
        assert after is not None, "failed to get logprobs after sleep/wake"
        assert len(after) == len(before), "logprobs length changed after sleep/wake"

        compared = 0
        for i, (b, a) in enumerate(zip(before, after)):
            if b is None or a is None:
                continue
            compared += 1
            diff = abs(b - a)
            # BF16 has ~3 significant decimal digits; 1e-2 is achievable
            # for identical greedy decodes across a sleep/wake cycle.
            assert diff < 1e-2, (
                f"logprob[{i}] drifted after sleep/wake: "
                f"before={b:.6f} after={a:.6f} diff={diff:.2e} — "
                "weight restore or KV-scale recalibration may be incorrect"
            )
        assert compared > 0, (
            "no non-None logprob pairs were compared — "
            "logprobs response may be empty or malformed"
        )


# ---------------------------------------------------------------------------
# TestSleepWakeLatency
# ---------------------------------------------------------------------------


class TestSleepWakeLatency:
    """sleep/wake must complete within reasonable wall-clock time.

    Reference: sglang test_update_weights_from_distributed.py
               assert time < 3s for weight sync operations.
    Here we use 10 s for the full sleep+wake roundtrip on a 0.6B model,
    which is generous but guards against regressions that cause multi-minute
    stalls (e.g. accidental full model re-download on wake).
    """

    def test_sleep_wake_roundtrip_under_10s(self, isolated_server_url):
        """sleep(1) + wake_up() roundtrip must complete in < 10 s.

        For a 0.6B bfloat16 model (~1.2 GB weights) the cumem unmap/remap
        should complete in < 2 s on modern hardware.  10 s is a conservative
        bound that would still catch pathological regressions.
        """
        url = isolated_server_url
        gen(url)  # warm up — ensure KV blocks are allocated

        t0 = time.perf_counter()
        assert sleep(url, level=1) == 200
        t_sleep = time.perf_counter()

        assert wake(url) == 200
        t_wake = time.perf_counter()

        sleep_elapsed = t_sleep - t0
        wake_elapsed = t_wake - t_sleep
        total = t_wake - t0

        assert total < _THRESHOLDS["full_roundtrip_s"], (
            f"sleep+wake roundtrip took {total:.2f}s "
            f"(sleep={sleep_elapsed:.2f}s, wake={wake_elapsed:.2f}s) — "
            "cumem unmap/remap regression?"
        )
        assert health(url) == 200
        assert ok(gen(url))

    def test_sleep_staged_wake_roundtrip_under_15s(self, isolated_server_url):
        """sleep(1) + staged wake (weights → kv_cache) must complete in < 15 s.

        Unlike the full-wake roundtrip test, this exercises the per-tag wake
        path used in colocate RL.  15 s is generous for a 0.6B model but
        catches pathological regressions (model re-download, page-fault storm).
        """
        url = isolated_server_url
        gen(url)  # warm up — ensure KV blocks are allocated

        t0 = time.perf_counter()
        assert sleep(url, level=1) == 200
        assert wake(url, tags=["weights"]) == 200
        assert wake(url, tags=["kv_cache"]) == 200
        elapsed = time.perf_counter() - t0

        assert elapsed < _THRESHOLDS["staged_roundtrip_s"], (
            f"sleep + staged wake took {elapsed:.2f}s — "
            "cumem unmap/remap latency regression?"
        )
        assert health(url) == 200
        assert ok(gen(url))

    def test_five_cycles_total_under_60s(self, isolated_server_url):
        """5 consecutive sleep/wake cycles must finish in < 60 s total.

        Guards against per-cycle overhead accumulation (handle leak,
        growing allocation list, etc.).
        """
        url = isolated_server_url
        gen(url)

        t0 = time.perf_counter()
        for _ in range(5):
            assert sleep(url, level=1) == 200
            assert wake(url) == 200
            assert health(url) == 200
        total = time.perf_counter() - t0

        assert total < _THRESHOLDS["five_cycles_s"], (
            f"5 sleep/wake cycles took {total:.2f}s — "
            "per-cycle overhead may be accumulating"
        )
        assert ok(gen(url))


# ---------------------------------------------------------------------------
# TestStagedWakeCycles
# ---------------------------------------------------------------------------


class TestStagedWakeCycles:
    """Staged-wake cycles: repeated sleep → staged wake keeps output stable.

    Reference: sglang test_release_memory_occupation.py
               test_release_and_resume_occupation_with_weights_cpu_backup —
               verifies golden output after CPU backup roundtrip.

    Unlike TestOutputCorrectness which tests 3 cycles, this specifically
    tests 5 cycles and focuses on the weights-only partial-wake path, which
    is the path used in colocate RL (trainer uses KV memory while engine
    keeps weights on CPU).
    """

    def test_staged_wake_5cycles_output_stable(self, seeded_server_url):
        """5× sleep(level=1) → wake(weights) → wake(kv_cache) output stable.

        This is the colocate RL pattern: sleep frees ALL GPU memory,
        wake(["weights"]) restores weights only (trainer has released GPU),
        wake(["kv_cache"]) re-allocates KV pool.
        Output must be non-empty and consistent across cycles.

        Uses --seed for deterministic greedy decoding so text comparison is
        a reliable weight-correctness check and not a nondeterminism false alarm.
        """
        url = seeded_server_url
        golden = gen(url)
        assert golden and ok(golden)
        golden_text = golden["choices"][0]["text"]
        assert golden_text.strip(), "golden output must be non-empty"

        for cycle in range(5):
            assert sleep(url, level=1) == 200

            # staged wake: weights first, then kv_cache
            assert wake(url, tags=["weights"]) == 200
            assert wake(url, tags=["kv_cache"]) == 200
            assert health(url) == 200

            resp = gen(url)
            assert resp and ok(resp), f"generate failed on staged-wake cycle {cycle}"
            cycle_text = resp["choices"][0]["text"]
            assert cycle_text.strip(), (
                f"empty output on staged-wake cycle {cycle} — "
                "weight restore may have corrupted model state"
            )
            assert cycle_text == golden_text, (
                f"output drifted on staged-wake cycle {cycle} — "
                "weight restore may be incomplete "
                f"(golden={golden_text!r}, got={cycle_text!r})"
            )
