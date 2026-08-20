# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the vLLM sleep-mode lifecycle (optimized, dual-runner).

Every test additionally runs twice via a module-scoped parametrized fixture:
once on MRV1 (V1 GPUModelRunner) and once on MRV2
(``VLLM_USE_V2_MODEL_RUNNER=1``, V2 GPUModelRunner).  The runners own the
KV-cache layout, attention backend and forward path — exactly the layers
sleep mode exercises — so each round gets a fresh set of servers.

Launch budget
-------------
Per runner round:
  TestSleepWakeCore       1 server   (3 low-risk methods share it)
  TestSchedulerGate       2 servers  (both high-risk: #44483 + abort/sampling_n)
  TestSleepWakeSafety     2 servers  (idempotency shares; concurrent+cumem isolated)
  TestSleepWakeStability  1 server   (2 low-risk methods share it)
                          --------
                          6 servers per round × 2 rounds = 12 total
                          (down from a naive 9 × 2 = 18)

"""

import os
import threading
import time
from unittest.mock import patch

import pytest
import requests

from .conftest import (
    gen,
    gen_with_logprobs,
    gpu_free_bytes,
    health,
    is_sleeping,
    ok,
    poll_until,
    server,
    sleep,
    sleep_metrics,
    wake,
)

# ---------------------------------------------------------------------------
# Port allocation
#
# Each class-scoped fixture and each isolated server binds a unique port.
# Layout (no overlaps, 10-port stride per consumer):
#   8800  TestSleepWakeCore        class-scoped
#   8810  TestSchedulerGate        method A (partial-wake, isolated)
#   8820  TestSchedulerGate        method B (abort/sampling_n, isolated)
#   8830  TestSleepWakeSafety      class-scoped (idempotency)
#   8840  TestSleepWakeSafety      concurrent+cumem (isolated)
#   8850  TestSleepWakeStability   class-scoped
# ---------------------------------------------------------------------------

_PORT = {
    "core": 8800,
    "gate_partial": 8810,
    "gate_blocks": 8820,
    "safety_idem": 8830,
    "safety_edge": 8840,
    "stability": 8850,
}


# ---------------------------------------------------------------------------
# MRV1 / MRV2 dual-round parametrization
#
# use_v2 fans every test item out into an [MRV1] and an [MRV2] copy.  When
# the parameter switches, pytest tears down all downstream fixtures and
# rebuilds them, so each round gets fresh servers.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=[False, True], ids=["MRV1", "MRV2"])
def use_v2(request):
    return request.param


@pytest.fixture(scope="module", autouse=True)
def v2_runner_env(use_v2):
    """Inject VLLM_USE_V2_MODEL_RUNNER for the whole round.

    autouse + module scope ensures BOTH the class-scoped shared servers AND
    the in-method isolated ``with server(...)`` launches inherit the env var
    via Popen's env snapshot in conftest.server().
    """
    env_vars = {"VLLM_USE_V2_MODEL_RUNNER": "1" if use_v2 else "0"}
    with patch.dict(os.environ, env_vars):
        yield


# ---------------------------------------------------------------------------
# Class-scoped server fixtures
#
# Only low-risk classes get one.  Each yields a URL to every test method in
# the class; the vLLM subprocess is started once and killed once.  The
# runner flavor (MRV1/MRV2) is inherited from the autouse v2_runner_env
# fixture active for the whole round.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def server_url_core():
    with server(port=_PORT["core"]) as url:
        yield url


@pytest.fixture(scope="class")
def server_url_safety():
    with server(port=_PORT["safety_idem"]) as url:
        yield url


@pytest.fixture(scope="class")
def server_url_stability():
    with server(port=_PORT["stability"]) as url:
        yield url


# ===========================================================================
# TestSleepWakeCore — flags / metrics / physical memory / output correctness
# ===========================================================================


class TestSleepWakeCore:
    """Core sleep/wake contract on ONE shared server.

    All three methods are deterministic and leave the engine fully awake on
    return, so they can safely share a single class-scoped server.
    """

    def test_flags_metrics_memory(self, server_url_core):
        """sleep(1)/sleep(2)/sleep(0) — flags, metrics, GPU free bytes.

        Covers: is_sleeping flag, vllm:engine_sleep_state metrics per level,
        GPU memory release (level 1/2) vs no-release (level 0), and wake
        re-allocating the freed memory.
        """
        url = server_url_core
        gen(url)  # warm up — allocate KV blocks
        free_awake = gpu_free_bytes()

        # --- sleep(level=1): offload weights only -------------------------
        assert sleep(url, level=1) == 200
        assert poll_until(lambda: is_sleeping(url), timeout=10)

        awake, wo, da = sleep_metrics(url)
        assert awake == 0 and wo == 1 and da == 0, (
            f"metrics inconsistent after sleep(1): "
            f"awake={awake} wo={wo} da={da}"
        )
        freed = (gpu_free_bytes() - free_awake) / 2**30
        assert freed > 0.5, (
            f"sleep(1) freed only {freed:.2f} GiB — "
            "CuMemAllocator unmap may be a no-op"
        )

        assert wake(url) == 200
        assert poll_until(lambda: not is_sleeping(url), timeout=10)
        # wake must re-allocate: free bytes should drop back close to baseline
        assert (gpu_free_bytes() - free_awake) / 2**30 < 0.5, (
            "wake did not re-allocate the memory freed by sleep(1)"
        )

        # --- sleep(level=2): discard all (weights + KV) -------------------
        assert sleep(url, level=2) == 200
        assert poll_until(lambda: is_sleeping(url), timeout=10)
        _, _, da = sleep_metrics(url)
        assert da == 1, f"level=2 must set discard_all=1, got da={da}"
        freed2 = (gpu_free_bytes() - free_awake) / 2**30
        assert freed2 > 1.5, (
            f"sleep(2) freed only {freed2:.2f} GiB — "
            "should discard weights + KV"
        )

        assert wake(url) == 200
        assert health(url) == 200

        # --- sleep(level=0): pause scheduling only, NO memory release -----
        free_before_l0 = gpu_free_bytes()
        assert sleep(url, level=0) == 200
        assert poll_until(lambda: is_sleeping(url), timeout=10)
        freed0 = (gpu_free_bytes() - free_before_l0) / 2**30
        assert freed0 < 0.5, (
            f"sleep(0) released {freed0:.2f} GiB — "
            "it should only pause scheduling, not free GPU memory"
        )
        _, wo, da = sleep_metrics(url)
        assert wo == 0 and da == 0, (
            "level=0 must not offload weights or discard all"
        )

        assert wake(url) == 200
        assert health(url) == 200

    def test_staged_wake_semantics(self, server_url_core):
        """Partial (staged) wake: tags set ops + is_sleeping invariant +
        per-step GPU memory change.

        Invariant: is_sleeping() = len(sleeping_tags) > 0.
        wake(['weights']) removes 'weights' but keeps is_sleeping True until
        'kv_cache' is also woken.
        """
        url = server_url_core
        gen(url)
        free_awake = gpu_free_bytes()

        assert sleep(url, level=1) == 200
        assert poll_until(lambda: is_sleeping(url), timeout=10)

        # wake weights only — kv_cache tag still present
        assert wake(url, tags=["weights"]) == 200
        assert is_sleeping(url) is True, (
            "is_sleeping went False after weights-only wake — "
            "kv_cache tag not tracked (#44483 regression)"
        )
        free_after_weights = gpu_free_bytes()

        # wake kv_cache — now fully awake
        assert wake(url, tags=["kv_cache"]) == 200
        assert poll_until(lambda: not is_sleeping(url), timeout=10)
        free_after_kv = gpu_free_bytes()

        # waking kv_cache consumes more GPU memory than weights-only wake
        assert free_after_kv < free_after_weights, (
            "waking kv_cache should use more GPU memory than weights-only wake"
        )
        assert health(url) == 200

    def test_output_correctness(self, server_url_core):
        """Output is deterministic across full wake, staged wake, and
        prefix-cache invalidation.

        Guards against: weight-restore corruption, stale prefix-cache entry
        pointing to a released physical page.
        """
        url = server_url_core
        golden = gen(url)
        assert golden and ok(golden)
        golden_text = golden["choices"][0]["text"]

        # full wake restores weights — output must match golden
        assert sleep(url, level=1) == 200
        assert wake(url) == 200
        assert gen(url)["choices"][0]["text"] == golden_text, (
            "output changed after full sleep/wake — weight restore broken"
        )

        # staged wake — same expectation
        assert sleep(url, level=1) == 200
        assert wake(url, tags=["weights"]) == 200
        assert wake(url, tags=["kv_cache"]) == 200
        assert gen(url)["choices"][0]["text"] == golden_text, (
            "output changed after staged sleep/wake"
        )

        # prefix cache must be cleared on wake; re-running a cached prompt
        # must not hit a stale entry pointing to a released page
        prompt = "The capital of France is"
        gen(url, prompt=prompt)
        assert sleep(url, level=1) == 200
        assert wake(url) == 200
        assert ok(gen(url, prompt=prompt)), (
            "generate failed after wake with cached prompt — "
            "possible stale prefix-cache IMA"
        )


# ===========================================================================
# TestSchedulerGate — #44483 partial-wake regression (isolated servers)
# ===========================================================================


class TestSchedulerGate:
    """The scheduler must not dispatch forward while KV memory is unmapped.

    Both methods are high-risk (they probe crash paths and abort semantics),
    so each launches its own isolated server.  No class-scoped fixture.

    Core of PR #44483: without the fix, wake_up(['weights']) unconditionally
    called resume_scheduler(), allowing FlashAttention to launch on the
    still-unmapped kv_cache VA — flash_fwd_launch_template.h:199 TMA
    descriptor 700 illegal memory access.
    """

    def test_partial_wake_blocks_until_kv_resident(self):
        """PR #44483 regression test.

        sleep(1) -> wake_up(['weights']) -> generate must NOT execute FA on
        the unmapped kv_cache VA.  With the fix the scheduler holds the
        request until wake_up(['kv_cache']) completes; without the fix the
        engine dies (health goes to 503) within 3 seconds.
        """
        with server(port=_PORT["gate_partial"]) as url:
            assert sleep(url, level=1) == 200
            assert wake(url, tags=["weights"]) == 200
            assert is_sleeping(url) is True  # kv_cache still unmapped

            result: dict = {}

            def _bg():
                # short timeout — should block, not complete, while kv_cache asleep
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
            assert poll_until(lambda: not is_sleeping(url), timeout=10)
            t.join(timeout=30)
            assert health(url) == 200

            assert ok(gen(url))

    def test_sleep_blocks_and_recovers(self):
        """sleep(abort)/sleep(level=0) pause the scheduler; new requests must
        neither succeed nor hang; engine recovers after wake.

        Also covers sampling_n=4 abort: sleep(mode='abort') must drain all 4
        sequence slots cleanly without blocking the next wake.
        """
        with server(port=_PORT["gate_blocks"]) as url:
            # --- sleep(level=1, abort) gates new requests ----------------
            assert sleep(url, level=1, mode="abort") == 200
            assert is_sleeping(url) is True

            assert not ok(gen(url, timeout=5)), (
                "generate returned a successful result while engine was "
                "sleeping (scheduler not paused, or request hangs without "
                "aborting) #45326"
            )
            assert health(url) == 200

            assert wake(url) == 200
            assert ok(gen(url))

            # --- sleep(level=0) gates the same way without releasing memory
            assert sleep(url, level=0) == 200
            assert is_sleeping(url) is True

            assert not ok(gen(url, timeout=5))
            assert health(url) == 200

            assert wake(url) == 200
            assert ok(gen(url))

            # --- sampling_n=4 abort drains all 4 slots cleanly ------------
            result: dict = {}

            def _bg_n():
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

            t = threading.Thread(target=_bg_n)
            t.start()
            time.sleep(0.3)

            assert sleep(url, level=1, mode="abort") == 200
            t.join(timeout=15)

            assert "err" not in result, (
                f"background thread raised: {result.get('err')}"
            )
            assert health(url) == 200, "engine died after sampling_n=4 abort"
            assert wake(url) == 200
            assert ok(gen(url)), "generate failed after sampling_n abort + wake"


# ===========================================================================
# TestSleepWakeSafety — idempotency (shared) + concurrent/cumem (isolated)
# ===========================================================================


class TestSleepWakeSafety:
    """Protocol-violation tolerance and CuMemAllocator edge cases.

    test_idempotency is deterministic and shares the class-scoped server.
    test_concurrent_and_cumem_edge races threads and probes crash paths, so
    it runs on its own isolated server.
    """

    def test_idempotency(self, server_url_safety):
        """Double sleep / double wake must not crash the engine (#45518).

        Also: a fresh server starts awake (is_sleeping=False).
        """
        url = server_url_safety
        # initial state
        assert is_sleeping(url) is False, (
            "engine should start awake (is_sleeping=False)"
        )

        # double sleep
        assert sleep(url, level=1) == 200
        assert poll_until(lambda: is_sleeping(url), timeout=10)

        sc = sleep(url, level=1)  # idempotent call
        assert sc in (200, 400), (
            f"double sleep returned unexpected status {sc} (#45518)"
        )
        assert health(url) == 200

        assert wake(url) == 200
        assert poll_until(lambda: not is_sleeping(url), timeout=10)
        assert ok(gen(url))

        # double wake on an already-awake engine
        assert is_sleeping(url) is False
        sc = wake(url)
        assert sc in (200, 400), (
            f"wake on non-sleeping engine returned unexpected status {sc}"
        )
        assert health(url) == 200
        assert ok(gen(url))

    def test_concurrent_and_cumem_edge(self):
        """Concurrent sleep+generate race + CuMem edge cases (isolated server).

        - 10 generate threads racing 1 sleep/wake thread: no deadlock, engine alive.
        - partial wake leaves KV unmapped (#44395 precondition).
        - sleep after partial wake must not double-unmap already-freed KV.
        - waking the same tag twice must be idempotent (no double create_and_map).
        - reverse-order wake (kv before weights) must not crash forward attempts.
        """
        with server(port=_PORT["safety_edge"]) as url:
            # --- concurrent race ----------------------------------------
            errors = []

            def _gen_thread():
                try:
                    gen(url, max_tokens=8, timeout=10)
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
            assert health(url) == 200, (
                f"engine died after concurrent race; errors={errors}"
            )
            assert ok(gen(url)), "generate failed after concurrent race"

            # --- partial wake leaves KV unmapped (#44395 precondition) ---
            gen(url)
            assert sleep(url, level=1) == 200
            assert wake(url, tags=["weights"]) == 200
            assert is_sleeping(url), "engine should still be sleeping (KV not woken)"

            # --- sleep again must not double-unmap already-freed KV ------
            assert sleep(url, level=1) == 200, (
                "sleep after partial wake crashed — "
                "double unmap of already-freed KV (#44395)"
            )
            assert wake(url) == 200
            assert ok(gen(url)), "generation failed after double-sleep cycle"

            # --- waking the same tag twice is idempotent -----------------
            assert sleep(url, level=1) == 200
            assert wake(url, tags=["weights"]) == 200
            assert wake(url, tags=["weights"]) == 200, (
                "double wake of same tag crashed — missing is_asleep guard"
            )
            assert wake(url) == 200
            assert ok(gen(url))

            # --- reverse-order wake (kv before weights) must not crash --
            assert sleep(url, level=1) == 200
            assert wake(url, tags=["kv_cache"]) == 200
            assert is_sleeping(url), "engine should still be sleeping"
            # attempt generation — must not crash (block/timeout/error are all OK)
            try:
                requests.post(
                    f"{url}/v1/completions",
                    json={
                        "model": "m",
                        "prompt": "Hi",
                        "max_tokens": 2,
                        "temperature": 0,
                    },
                    timeout=5,
                )
            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError):
                pass
            assert health(url) == 200, (
                "engine not healthy after reverse-order wake + generation attempt"
            )
            assert wake(url) == 200
            assert ok(gen(url))


# ===========================================================================
# TestSleepWakeStability — long-cycle leak / latency / precision (shared)
# ===========================================================================


class TestSleepWakeStability:
    """Long-running stability: no GPU memory leak, bounded latency, and
    numerical precision across repeated sleep/wake cycles.

    Both methods are deterministic and share one class-scoped server.  The
    multi-cycle method leaves the engine fully awake, so logprobs precision
    starts from a clean awake state.
    """

    def test_multi_cycle_leak_latency(self, server_url_stability):
        """5 sleep/wake cycles: per-cycle timing + GPU free-bytes stability +
        output determinism (CPU-backup weight restore path).

        Guards against: cumem handle leak, per-cycle overhead accumulation,
        weight-restore corruption over repeated offload/remap.
        """
        url = server_url_stability
        golden = gen(url)
        assert golden and ok(golden)
        golden_text = golden["choices"][0]["text"]

        free_samples = []
        cycle_times = []

        for i in range(5):
            gen(url)
            t0 = time.perf_counter()
            assert sleep(url, level=1) == 200
            # staged wake: weights first, then kv_cache (colocate RL path)
            assert wake(url, tags=["weights"]) == 200
            assert wake(url, tags=["kv_cache"]) == 200
            elapsed = time.perf_counter() - t0
            assert health(url) == 200

            if i >= 1:  # skip warm-up
                free_samples.append(gpu_free_bytes())
                cycle_times.append(elapsed)

            resp = gen(url)
            assert resp and ok(resp), f"generate failed on cycle {i}"
            assert resp["choices"][0]["text"] == golden_text, (
                f"output drifted on cycle {i} — cumem bookkeeping corrupted "
                "or weight restore incomplete"
            )

        # --- leak guard: GPU free bytes stable across cycles -----------
        if len(free_samples) >= 2:
            leak_gib = (free_samples[0] - min(free_samples)) / 2**30
            assert leak_gib < 0.05, (  # 50 MiB tolerance
                f"GPU free memory shrank by {leak_gib:.3f} GiB over "
                f"{len(free_samples)} cycles (baseline="
                f"{free_samples[0]/2**30:.2f} GiB, min="
                f"{min(free_samples)/2**30:.2f} GiB) — "
                "possible cumem handle leak or unmapped page accumulation"
            )

        # --- latency guard: each cycle bounded -------------------------
        if cycle_times:
            max_cycle = max(cycle_times)
            assert max_cycle < 10.0, (
                f"slowest sleep/wake cycle took {max_cycle:.2f}s — "
                "cumem unmap/remap regression?"
            )

    def test_logprobs_precision(self, server_url_stability):
        """logprobs must match within 1e-2 before vs after a sleep/wake cycle.

        After sleep(level=1)/wake, weights are remapped from CPU backup.
        Calibrated scales (FP8-KV, etc.) must be restored; logprobs must
        match the pre-sleep values within a tight tolerance.  Drift here
        would silently corrupt RL advantage computation.
        """
        url = server_url_stability
        prompt = ("The capital of France is Paris and the capital of "
                  "Germany is")

        def _get_logprobs():
            resp = gen_with_logprobs(url, prompt=prompt, max_tokens=4)
            if not resp or "choices" not in resp or not resp["choices"]:
                return None
            return (resp["choices"][0]
                    .get("logprobs", {})
                    .get("token_logprobs", []))

        before = _get_logprobs()
        assert before is not None, "failed to get logprobs before sleep"
        assert len(before) > 0

        assert sleep(url, level=1) == 200
        assert wake(url) == 200
        assert health(url) == 200

        after = _get_logprobs()
        assert after is not None, "failed to get logprobs after sleep/wake"
        assert len(after) == len(before), (
            "logprobs length changed after sleep/wake"
        )

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
