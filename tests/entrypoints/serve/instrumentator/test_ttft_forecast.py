# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from http import HTTPStatus

import pytest
import requests
from prometheus_client.parser import text_string_to_metric_families

from tests.utils import RemoteOpenAIServer
from vllm.v1.core.sched.forecast import curve_value

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_MODEL_LEN = 2048
SERVER_ARGS = [
    "--dtype",
    "bfloat16",
    "--max-model-len",
    str(MAX_MODEL_LEN),
    "--enforce-eager",
    "--max-num-seqs",
    "4",
]


@pytest.fixture(scope="module")
def server():
    with RemoteOpenAIServer(
        MODEL_NAME, SERVER_ARGS, env_dict={"VLLM_TTFT_FORECAST": "1"}
    ) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def forecast(server: RemoteOpenAIServer) -> dict:
    """The forecast of a server that has timed enough steps to publish one."""
    client = server.get_client()
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        # Steps of several sizes, so the step-time fit has more than one shape.
        for prompt_tokens in (16, 200, 1000):
            client.completions.create(
                model=MODEL_NAME,
                prompt=[1] * prompt_tokens,
                max_tokens=8,
                temperature=0.0,
            )
        response = requests.get(server.url_for("forecast"), params={"cached": "0,512"})
        response.raise_for_status()
        result = response.json()
        if result["ready"]:
            return result
    raise AssertionError("The engine did not publish a forecast.")


def at(curve: list, new_tokens: int) -> float:
    seconds = curve_value(curve, new_tokens)
    assert seconds is not None
    return seconds


def gauges(server: RemoteOpenAIServer) -> dict[str, list]:
    response = requests.get(server.url_for("metrics"))
    response.raise_for_status()
    return {
        family.name: family.samples
        for family in text_string_to_metric_families(response.text)
        if family.name.startswith("vllm:ttft_forecast")
    }


def test_forecast_curve_grows_with_new_tokens(forecast: dict):
    assert forecast["enabled"]
    assert set(forecast["curves"]) == {"0", "512"}
    for cached, curve in forecast["curves"].items():
        # The curve covers every prompt the model accepts with this prefix.
        assert curve[0][0] == 0
        assert curve[-1][2] == MAX_MODEL_LEN - int(cached)
        small, large = at(curve, 16), at(curve, 1024)
        assert 0.0 < small < large
        # An engine with nothing queued is never slower than the engine now.
        idle = forecast["floor"][cached]
        assert at(idle, 1024) <= large * 1.05


def test_single_forecasts_match_the_curve(server: RemoteOpenAIServer, forecast: dict):
    response = requests.post(
        server.url_for("forecast"),
        json={"cached": [0], "queries": [[16, 0], [MAX_MODEL_LEN, 0], [10**9, 0]]},
    )
    response.raise_for_status()
    result = response.json()
    small, whole, past_the_model = result["ttft"]
    assert 0.0 < small < whole
    # The curve ends at a prompt of the model length. Its segments are merged
    # within 2%.
    assert whole == pytest.approx(result["curves"]["0"][-1][3], rel=0.05)
    # A prompt longer than the model length is forecast at the model length.
    assert past_the_model == pytest.approx(whole, rel=0.01)


def test_forecast_gauges_follow_the_curve(server: RemoteOpenAIServer, forecast: dict):
    families = gauges(server)
    assert set(families) == {
        "vllm:ttft_forecast_seconds",
        "vllm:ttft_forecast_idle_seconds",
        "vllm:ttft_forecast_restore_seconds",
        "vllm:ttft_forecast_busy_share",
        "vllm:ttft_forecast_age_seconds",
    }
    by_knot = {
        (int(s.labels["new_tokens"]), int(s.labels["cached_tokens"])): s.value
        for s in families["vllm:ttft_forecast_seconds"]
    }
    # No prefix, a quarter and half of the model length, each up to the
    # longest prompt that fits.
    assert {cached for _, cached in by_knot} == {0, 512, 1024}
    assert max(new for new, cached in by_knot if cached == 0) == MAX_MODEL_LEN
    assert 0.0 < by_knot[(16, 0)] < by_knot[(1024, 0)]
    idle = {
        (int(s.labels["new_tokens"]), int(s.labels["cached_tokens"])): s.value
        for s in families["vllm:ttft_forecast_idle_seconds"]
    }
    assert idle.keys() == by_knot.keys()
    (age,) = families["vllm:ttft_forecast_age_seconds"]
    # The scrape that returned these values refreshed them.
    assert 0.0 <= age.value < 0.1


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"method": "GET", "params": {"cached": "many"}},
        {"method": "GET", "params": {"cached": "-1"}},
        {"method": "GET", "params": {"cached": ",".join(["0"] * 9)}},
        {"method": "POST", "json": {"queries": [[1, 0]] * 257}},
        {"method": "POST", "json": {"queries": [[1]]}},
    ],
)
def test_forecast_rejects_requests_it_does_not_bound(
    server: RemoteOpenAIServer, request_kwargs: dict
):
    response = requests.request(url=server.url_for("forecast"), **request_kwargs)
    assert response.status_code == HTTPStatus.BAD_REQUEST


def test_forecast_is_off_by_default():
    with RemoteOpenAIServer(MODEL_NAME, SERVER_ARGS) as remote_server:
        client = remote_server.get_client()
        client.completions.create(
            model=MODEL_NAME, prompt=[1] * 200, max_tokens=8, temperature=0.0
        )
        response = requests.get(remote_server.url_for("forecast"))
        response.raise_for_status()
        assert response.json() == {"enabled": False}
        assert gauges(remote_server) == {}
