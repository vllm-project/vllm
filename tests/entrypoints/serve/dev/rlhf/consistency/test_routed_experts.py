# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing capture across cache reuse and RL memory lifecycle transitions.

Frontend serialization smoke tests stay in openai/test_return_routed_experts.py
and scale_out/token_in_token_out/test_return_routed_experts.py. This suite
checks the complete token-aligned capture, including cached prompt rows. It
does not exercise training-side replay or cross-batch numerical invariance.
"""

import io

import numpy as np
import pybase64 as base64
import pytest
import requests

from ..conftest import cached_tokens, server, sleep, wake

MODEL = "TitanML/tiny-mixtral"
PROMPT = "Paris is the capital of France. " * 24


@pytest.fixture(
    scope="module",
    params=[(False, True), (True, True), (True, False)],
    ids=["MRV1-eager", "MRV2-eager", "MRV2-graph"],
)
def server_url(request):
    use_v2, eager = request.param
    with server(
        model=MODEL,
        enforce_eager=eager,
        timeout=600,
        env_dict={"VLLM_USE_V2_MODEL_RUNNER": str(int(use_v2))},
        extra_args=[
            "--enable-return-routed-experts",
            "--enable-prefix-caching",
            "--enable-prompt-tokens-details",
            "--hf-overrides",
            '{"sliding_window": null}',
        ],
    ) as url:
        yield url


def completion(url):
    response = requests.post(
        f"{url}/v1/completions",
        json={
            "model": "m",
            "prompt": PROMPT,
            "max_tokens": 8,
            "temperature": 0,
            "ignore_eos": True,
            "return_token_ids": True,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def routing(response):
    choice = response["choices"][0]
    tokens = choice["token_ids"]
    assert len(tokens) == 8
    captured = np.load(
        io.BytesIO(base64.b64decode(choice["routed_experts"], validate=True)),
        allow_pickle=False,
    )
    # The last sampled token has not been forwarded through the model.
    assert captured.shape == (
        response["usage"]["prompt_tokens"] + len(tokens) - 1,
        2,
        2,
    )
    assert np.issubdtype(captured.dtype, np.integer)
    assert ((captured >= 0) & (captured < 8)).all()
    assert (captured[:, :, 0] != captured[:, :, 1]).all()
    return captured


def test_capture_survives_cache_hits_and_sleep_wake(server_url):
    """Cached prompt rows survive reuse; fresh post-wake rows match the baseline."""
    assert sleep(server_url, level=1) == 200
    assert wake(server_url) == 200
    baseline = completion(server_url)
    assert cached_tokens(baseline) == 0
    expected = routing(baseline)
    cached = completion(server_url)
    assert cached_tokens(cached) > 0
    assert cached["choices"][0]["token_ids"] == baseline["choices"][0]["token_ids"]
    np.testing.assert_array_equal(routing(cached), expected)

    assert sleep(server_url, level=1) == 200
    try:
        assert wake(server_url, tags=["weights"]) == 200
    finally:
        assert wake(server_url) == 200
    restored = completion(server_url)
    assert cached_tokens(restored) == 0
    assert restored["choices"][0]["token_ids"] == baseline["choices"][0]["token_ids"]
    np.testing.assert_array_equal(routing(restored), expected)


@pytest.mark.distributed
@pytest.mark.parametrize("use_v2", [False, True], ids=["MRV1", "MRV2"])
def test_parallel_capture_alignment(use_v2, num_gpus_available):
    """Preserve TP2/DP2 frontend coverage in an explicitly four-GPU lane."""
    if num_gpus_available < 4:
        pytest.skip("TP2/DP2 requires four GPUs")
    with server(
        model=MODEL,
        env_dict={
            "VLLM_USE_V2_MODEL_RUNNER": str(int(use_v2)),
            "VLLM_ENABLE_SCALE_OUT_ENDPOINTS": "1",
        },
        extra_args=[
            "--enable-return-routed-experts",
            "--hf-overrides",
            '{"sliding_window": null}',
            "--tensor-parallel-size",
            "2",
            "--data-parallel-size",
            "2",
            "--data-parallel-size-local",
            "2",
        ],
    ) as url:
        response = requests.post(
            f"{url}/inference/v1/generate",
            json={
                "model": "m",
                "token_ids": [1, 2, 3],
                "stream": False,
                "sampling_params": {
                    "max_tokens": 8,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            },
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        assert body["usage"]["prompt_tokens"] == 3
        routing(body)
