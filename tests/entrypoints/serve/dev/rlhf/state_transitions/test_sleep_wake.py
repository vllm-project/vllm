# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP sleep/wake gating, cache invalidation, and restored generation.

Consolidates serve/dev/test_sleep.py. Allocator memory/graph invariants and
offline level-2 reload remain in tests/basic_correctness/test_mem.py; hybrid
state and DP drain regressions remain in their model and distributed suites.
The small matrix covers both runners in eager mode and MRV2 with graphs.
"""

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import pytest
import requests

from tests.entrypoints.serve.dev.rlhf.conftest import (
    cached_tokens,
    completion_with_cache_details,
    golden_output,
    is_sleeping,
    reusable_server,
    sleep,
    sleep_metrics,
    wake,
)

PROMPT = "Paris is the capital of France. Berlin is the capital of Germany. " * 20

# Parent cleanup runs once after the shared server has fully shut down.
pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(
    scope="module",
    params=[(False, True), (True, True), (True, False)],
    ids=["MRV1-eager", "MRV2-eager", "MRV2-graph"],
)
def server_url(request):
    use_v2, eager = request.param
    with reusable_server(
        enforce_eager=eager,
        timeout=600,
        env_dict={"VLLM_USE_V2_MODEL_RUNNER": str(int(use_v2))},
        extra_args=[
            "--enable-prefix-caching",
            "--enable-prompt-tokens-details",
            "--enable-server-load-tracking",
        ],
    ) as url:
        yield url


@pytest.fixture(autouse=True)
def restore_awake(server_url):
    assert wake(server_url) == 200
    yield
    assert wake(server_url) == 200
    assert not is_sleeping(server_url)


def _completion(url):
    response = requests.post(
        f"{url}/v1/completions",
        json={
            "model": "m",
            "prompt": "The capital of France is",
            "temperature": 0,
            "max_tokens": 8,
            "logprobs": 1,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def assert_request_is_queued(url, future):
    """Observe server acceptance before testing that the same request is held."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if future.done():
            future.result()  # Surface the original HTTP error if it failed.
            pytest.fail("Request completed before wake")
        load = requests.get(f"{url}/load", timeout=5)
        load.raise_for_status()
        if load.json()["server_load"] == 1:
            break
        time.sleep(0.05)
    else:
        pytest.fail("Server did not accept the request while sleeping")
    with pytest.raises(TimeoutError):
        future.result(timeout=1)


@pytest.mark.parametrize("level", [0, 1])
def test_sleep_flags_metrics_and_idempotency(server_url, level):
    """Repeated commands preserve state; level zero only pauses scheduling."""
    # Metric tuple: (awake, weights_offloaded, discard_all).
    assert sleep_metrics(server_url) == (1, 0, 0)
    for _ in range(2):
        assert sleep(server_url, level=level) == 200
        assert is_sleeping(server_url)
        assert sleep_metrics(server_url) == ((0, 0, 0) if level == 0 else (0, 1, 0))
    for _ in range(2):
        assert wake(server_url) == 200
        assert not is_sleeping(server_url)
        assert sleep_metrics(server_url) == (1, 0, 0)
    assert _completion(server_url)["choices"]


@pytest.mark.parametrize("first_tag", ["weights", "kv_cache"])
def test_partial_wake_holds_and_recovers_same_request(server_url, first_tag):
    """Neither tag order may dispatch a forward onto unresident memory."""
    baseline = _completion(server_url)
    assert sleep(server_url) == 200
    assert wake(server_url, tags=[first_tag]) == 200
    assert is_sleeping(server_url)
    remaining = "kv_cache" if first_tag == "weights" else "weights"
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_completion, server_url)
        try:
            assert_request_is_queued(server_url, future)
            health = requests.get(f"{server_url}/health", timeout=5)
            health.raise_for_status()
        finally:
            assert wake(server_url, tags=[remaining]) == 200
        restored = future.result(timeout=30)
    assert not is_sleeping(server_url)
    assert golden_output(restored) == golden_output(baseline)


@pytest.mark.parametrize("level", [0, 1])
def test_sleep_queues_new_request_until_wake(server_url, level):
    """An accepted request must complete after wake, rather than time out."""
    assert sleep(server_url, level=level) == 200
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_completion, server_url)
        try:
            assert_request_is_queued(server_url, future)
        finally:
            assert wake(server_url) == 200
        response = future.result(timeout=30)
    assert response["choices"][0]["finish_reason"] in ("stop", "length")


def test_deep_sleep_restores_checkpoint_before_generation(server_url):
    """Level two discards weights; waking alone is not a valid roundtrip."""
    baseline = _completion(server_url)
    assert sleep(server_url, level=2) == 200
    try:
        assert sleep_metrics(server_url) == (0, 0, 1)
    finally:
        assert wake(server_url, tags=["weights"]) == 200
        response = requests.post(
            f"{server_url}/collective_rpc",
            json={"method": "reload_weights"},
            timeout=120,
        )
        response.raise_for_status()
        assert wake(server_url, tags=["kv_cache"]) == 200
    assert golden_output(_completion(server_url)) == golden_output(baseline)


@pytest.mark.parametrize("staged", [False, True])
def test_sleep_wake_preserves_output_and_clears_prefix_cache(server_url, staged):
    """Discarded KV must not be reused, and restored weights preserve output."""
    assert sleep(server_url) == 200
    assert wake(server_url) == 200
    baseline = completion_with_cache_details(server_url, PROMPT)
    assert cached_tokens(baseline) == 0
    assert cached_tokens(completion_with_cache_details(server_url, PROMPT)) > 0
    for _ in range(2):
        assert sleep(server_url) == 200
        if staged:
            assert wake(server_url, tags=["weights"]) == 200
            assert is_sleeping(server_url)
            assert wake(server_url, tags=["kv_cache"]) == 200
        else:
            assert wake(server_url) == 200
        restored = completion_with_cache_details(server_url, PROMPT)
        assert cached_tokens(restored) == 0
        assert golden_output(restored) == golden_output(baseline)
        before = baseline["choices"][0]["logprobs"]["token_logprobs"]
        after = restored["choices"][0]["logprobs"]["token_logprobs"]
        assert before and len(before) == len(after)
        assert all(value is not None for value in before + after)
        assert after == pytest.approx(before, abs=1e-2, rel=0)
