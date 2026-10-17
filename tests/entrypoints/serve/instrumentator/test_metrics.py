# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from http import HTTPStatus

import openai
import pytest
import pytest_asyncio
import requests
from prometheus_client.parser import text_string_to_metric_families
from transformers import AutoTokenizer

from tests.conftest import LocalAssetServer
from tests.utils import RemoteOpenAIServer
from vllm import version
from vllm.utils.network_utils import get_open_port

MODELS = {
    "text": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "multimodal": "HuggingFaceTB/SmolVLM-256M-Instruct",
}
PREV_MINOR_VERSION = version._prev_minor_version()


@pytest.fixture(scope="module", params=list(MODELS.keys()))
def model_key(request):
    yield request.param


@pytest.fixture(scope="module")
def default_server_args():
    return [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "1024",
        "--enforce-eager",
        "--max-num-seqs",
        "128",
    ]


@pytest.fixture(
    scope="module",
    params=[
        "",
        "--enable-chunked-prefill",
        f"--show-hidden-metrics-for-version={PREV_MINOR_VERSION}",
    ],
)
def server(model_key, default_server_args, request):
    server_args = list(default_server_args)
    if request.param:
        server_args.append(request.param)

    model_name = MODELS[model_key]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as cl:
        yield cl


_PROMPT = "Hello my name is Robert and I love magic"


def _metric_totals(body: str, api_surface: str | None = None) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for family in text_string_to_metric_families(body):
        for sample in family.samples:
            if (
                api_surface is not None
                and sample.labels.get("api_surface") != api_surface
            ):
                continue
            if sample.name.endswith(("_total", "_sum", "_count")):
                totals[sample.name] += sample.value
    return totals


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "api_surface, expected",
    [
        (
            None,
            {
                "vllm:request_success_total": 9,
                "vllm:e2e_request_latency_seconds_count": 4,
                "vllm:e2e_request_latency_seconds_sum": 8,
            },
        ),
        (
            "messages",
            {
                "vllm:request_success_total": 4,
                "vllm:e2e_request_latency_seconds_count": 4,
                "vllm:e2e_request_latency_seconds_sum": 8,
            },
        ),
        ("completions", {"vllm:request_success_total": 5}),
        ("unknown", {}),
    ],
)
def test_metric_totals_sum_engines_and_finish_reasons_with_api_filter(
    api_surface, expected
):
    body = """# TYPE vllm:request_success counter
vllm:request_success_total{engine="0",api_surface="messages",finished_reason="stop"} 1
vllm:request_success_total{engine="0",api_surface="messages",finished_reason="length"} 2
vllm:request_success_total{engine="1",api_surface="messages",finished_reason="stop"} 1
vllm:request_success_total{api_surface="completions",finished_reason="stop"} 5
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_bucket{api_surface="messages",le="+Inf"} 4
vllm:e2e_request_latency_seconds_count{api_surface="messages"} 4
vllm:e2e_request_latency_seconds_sum{api_surface="messages"} 8
"""
    assert _metric_totals(body, api_surface) == expected


def _get_expected_values(
    num_requests: int,
    prompt_ids: list[int],
    max_tokens: int,
    num_cached_tokens: float = 0,
):
    num_prompt_tokens = len(prompt_ids)

    # {metric_family: [(suffix, expected_value)]}
    return {
        "vllm:time_to_first_token_seconds": [("_count", num_requests)],
        "vllm:inter_token_latency_seconds": [
            ("_count", num_requests * (max_tokens - 1))
        ],
        "vllm:e2e_request_latency_seconds": [("_count", num_requests)],
        "vllm:request_queue_time_seconds": [("_count", num_requests)],
        "vllm:request_inference_time_seconds": [("_count", num_requests)],
        "vllm:request_prefill_time_seconds": [("_count", num_requests)],
        "vllm:request_decode_time_seconds": [("_count", num_requests)],
        "vllm:request_num_preemptions": [("_count", num_requests)],
        "vllm:request_prompt_tokens": [
            ("_sum", num_requests * num_prompt_tokens),
            ("_count", num_requests),
        ],
        "vllm:request_generation_tokens": [
            ("_sum", num_requests * max_tokens),
            ("_count", num_requests),
        ],
        "vllm:request_params_n": [("_count", num_requests)],
        "vllm:request_params_max_tokens": [
            ("_sum", num_requests * max_tokens),
            ("_count", num_requests),
        ],
        "vllm:iteration_tokens_total": [
            (
                "_sum",
                num_requests * (num_prompt_tokens + max_tokens) - num_cached_tokens,
            ),
            ("_count", num_requests * max_tokens),
        ],
        "vllm:prompt_tokens": [("_total", num_requests * num_prompt_tokens)],
        "vllm:generation_tokens": [("_total", num_requests * max_tokens)],
        "vllm:request_success": [("_total", num_requests)],
    }


@pytest.mark.asyncio
async def test_metrics_counts(
    server: RemoteOpenAIServer,
    client: openai.AsyncClient,
    model_key: str,
):
    if model_key == "multimodal":
        pytest.skip("Unnecessary test")

    model_name = MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt_ids = tokenizer.encode(_PROMPT)
    num_requests = 2
    max_tokens = 10
    baseline = requests.get(server.url_for("metrics"))
    baseline.raise_for_status()
    before = _metric_totals(baseline.text)

    for _ in range(num_requests):
        await client.completions.create(
            model=model_name,
            prompt=prompt_ids,
            max_tokens=max_tokens,
            extra_body={"ignore_eos": True},
        )

    response = requests.get(server.url_for("metrics"))
    assert response.status_code == HTTPStatus.OK
    after = _metric_totals(response.text)
    cached_tokens = (
        after["vllm:prompt_tokens_cached_total"]
        - before["vllm:prompt_tokens_cached_total"]
    )
    expected_values = _get_expected_values(
        num_requests, prompt_ids, max_tokens, cached_tokens
    )
    for metric_family, suffix_values in expected_values.items():
        if (
            not server.show_hidden_metrics
            and metric_family in HIDDEN_DEPRECATED_METRICS
        ):
            continue
        for suffix, expected_value in suffix_values:
            sample_name = f"{metric_family}{suffix}"
            assert sample_name in after, f"Missing {sample_name} in metrics endpoint"
            assert after[sample_name] - before.get(sample_name, 0) == expected_value, (
                sample_name
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server, enable_api_surface_metrics",
    [("", False), ("--enable-api-surface-metrics", True)],
    indirect=["server"],
    scope="module",
)
@pytest.mark.parametrize("api_surface", ["messages", "chat_completions", "completions"])
@pytest.mark.parametrize("stream", [False, True])
async def test_metrics_counts_by_api_surface(
    server, client, model_key, api_surface, stream, enable_api_surface_metrics
):
    """Native JSON/SSE requests aggregate by default and split metrics on opt-in."""
    if model_key == "multimodal":
        pytest.skip("Covered by the text model")

    response = requests.get(server.url_for("metrics"))
    response.raise_for_status()
    if not enable_api_surface_metrics:
        assert all(
            "api_surface" not in sample.labels
            for family in text_string_to_metric_families(response.text)
            for sample in family.samples
        )
    surfaces = ("messages", "chat_completions", "completions", "unknown")
    before = _metric_totals(response.text)
    before_by_api = {
        surface: _metric_totals(response.text, surface) for surface in surfaces
    }
    kwargs = {"model": MODELS[model_key], "max_tokens": 4, "stream": stream}
    messages = [{"role": "user", "content": _PROMPT}]
    if api_surface == "messages":
        async with server.get_async_client_anthropic() as anthropic_client:
            result = await anthropic_client.messages.create(messages=messages, **kwargs)
            if stream:
                events = [event async for event in result]
                assert events[-1].type == "message_stop"
            else:
                assert result.type == "message"
    else:
        result = (
            await client.chat.completions.create(messages=messages, **kwargs)
            if api_surface == "chat_completions"
            else await client.completions.create(prompt=_PROMPT, **kwargs)
        )
        if stream:
            chunks = [chunk async for chunk in result]
            assert chunks[-1].choices[0].finish_reason is not None
        else:
            assert result.choices[0].finish_reason is not None

    response = requests.get(server.url_for("metrics"))
    response.raise_for_status()
    after = _metric_totals(response.text)
    after_by_api = {
        surface: _metric_totals(response.text, surface) for surface in surfaces
    }
    if enable_api_surface_metrics:
        for surface in surfaces:
            if surface != api_surface:
                assert after_by_api[surface] == before_by_api[surface], surface
        before, after = before_by_api[api_surface], after_by_api[api_surface]
    else:
        assert all(
            "api_surface" not in sample.labels
            for family in text_string_to_metric_families(response.text)
            for sample in family.samples
        )
    delta = {name: value - before.get(name, 0) for name, value in after.items()}
    assert delta["vllm:request_success_total"] == 1
    for metric in (
        "time_to_first_token_seconds",
        "request_time_per_output_token_seconds",
        "e2e_request_latency_seconds",
        "request_queue_time_seconds",
        "request_inference_time_seconds",
        "request_prefill_time_seconds",
        "request_decode_time_seconds",
        "request_prompt_tokens",
        "request_generation_tokens",
        "request_prefill_kv_computed_tokens",
    ):
        assert delta[f"vllm:{metric}_count"] == 1, metric
    prompt_tokens = delta["vllm:prompt_tokens_total"]
    generation_tokens = delta["vllm:generation_tokens_total"]
    assert prompt_tokens > 0
    assert generation_tokens > 0
    assert delta["vllm:request_prompt_tokens_sum"] == prompt_tokens
    assert delta["vllm:request_generation_tokens_sum"] == generation_tokens
    assert delta["vllm:inter_token_latency_seconds_count"] == generation_tokens - 1
    assert delta["vllm:prompt_tokens_by_source_total"] == prompt_tokens
    assert delta["vllm:request_prefill_kv_computed_tokens_sum"] == (
        prompt_tokens - delta["vllm:prompt_tokens_cached_total"]
    )


EXPECTED_METRICS_V1 = [
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_waiting_by_reason",
    "vllm:kv_cache_usage_perc",
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
    "vllm:num_preemptions_total",
    "vllm:admission_rejections_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:iteration_tokens_total",
    "vllm:cache_config_info",
    "vllm:request_success_total",
    "vllm:request_prompt_tokens_sum",
    "vllm:request_prompt_tokens_bucket",
    "vllm:request_prompt_tokens_count",
    "vllm:request_generation_tokens_sum",
    "vllm:request_generation_tokens_bucket",
    "vllm:request_generation_tokens_count",
    "vllm:request_params_n_sum",
    "vllm:request_params_n_bucket",
    "vllm:request_params_n_count",
    "vllm:request_params_max_tokens_sum",
    "vllm:request_params_max_tokens_bucket",
    "vllm:request_params_max_tokens_count",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_bucket",
    "vllm:time_to_first_token_seconds_count",
    "vllm:inter_token_latency_seconds_sum",
    "vllm:inter_token_latency_seconds_bucket",
    "vllm:inter_token_latency_seconds_count",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:e2e_request_latency_seconds_bucket",
    "vllm:e2e_request_latency_seconds_count",
    "vllm:request_queue_time_seconds_sum",
    "vllm:request_queue_time_seconds_bucket",
    "vllm:request_queue_time_seconds_count",
    "vllm:request_inference_time_seconds_sum",
    "vllm:request_inference_time_seconds_bucket",
    "vllm:request_inference_time_seconds_count",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_prefill_time_seconds_bucket",
    "vllm:request_prefill_time_seconds_count",
    "vllm:request_decode_time_seconds_sum",
    "vllm:request_decode_time_seconds_bucket",
    "vllm:request_decode_time_seconds_count",
    "vllm:request_num_preemptions_sum",
    "vllm:request_num_preemptions_bucket",
    "vllm:request_num_preemptions_count",
]

EXPECTED_METRICS_MM = [
    "vllm:mm_cache_queries",
    "vllm:mm_cache_hits",
]

HIDDEN_DEPRECATED_METRICS: list[str] = []


@pytest.mark.asyncio
async def test_metrics_exist(
    local_asset_server: LocalAssetServer,
    server: RemoteOpenAIServer,
    client: openai.AsyncClient,
    model_key: str,
):
    model_name = MODELS[model_key]

    # sending a request triggers the metrics to be logged.
    if model_key == "text":
        await client.completions.create(
            model=model_name,
            prompt="Hello, my name is",
            max_tokens=5,
            temperature=0.0,
        )
    else:
        # https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg
        await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": local_asset_server.url_for(
                                    "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                                ),
                            },
                        },
                        {"type": "text", "text": "What's in this image?"},
                    ],
                }
            ],
            max_tokens=5,
            temperature=0.0,
        )

    response = requests.get(server.url_for("metrics"))
    assert response.status_code == HTTPStatus.OK

    expected_metrics = EXPECTED_METRICS_V1
    if model_key == "multimodal":
        # NOTE: Don't use in-place assignment
        expected_metrics = expected_metrics + EXPECTED_METRICS_MM

    for metric in expected_metrics:
        if metric in HIDDEN_DEPRECATED_METRICS and not server.show_hidden_metrics:
            continue
        assert metric in response.text

    cache_config_samples = [
        sample
        for family in text_string_to_metric_families(response.text)
        if family.name == "vllm:cache_config_info"
        for sample in family.samples
    ]
    assert cache_config_samples
    for sample in cache_config_samples:
        assert sample.labels.get("kv_cache_size_tokens") not in (None, "None", "")
        assert sample.labels.get("kv_cache_max_concurrency") not in (None, "None", "")


@pytest.mark.asyncio
async def test_abort_metrics_reset(
    server: RemoteOpenAIServer,
    client: openai.AsyncClient,
    model_key: str,
):
    model_name = MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt_ids = tokenizer.encode(_PROMPT)

    running_requests, waiting_requests, kv_cache_usage = _get_running_metrics_from_api(
        server,
    )

    # Expect no running requests or kvcache usage
    assert running_requests == 0
    assert waiting_requests == 0
    assert kv_cache_usage == 0.0

    # Start some long-running requests that we can abort
    tasks = []
    for _ in range(3):
        task = asyncio.create_task(
            client.completions.create(
                model=model_name,
                prompt=prompt_ids,
                max_tokens=500,  # Long generation to give time to abort
                temperature=0.0,
            )
        )
        tasks.append(task)

    # Poll until we see running requests rather than using a fixed sleep,
    # since generation speed varies across hardware.
    try:
        await _poll_until(
            lambda: _get_running_metrics_from_api(server)[0] > 0,
            timeout=10.0,
            interval=0.1,
            description="running_requests > 0",
        )
    except TimeoutError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        pytest.fail("Requests never appeared as running in metrics")

    # Check that we have running requests
    running_requests, waiting_requests, kv_cache_usage = _get_running_metrics_from_api(
        server,
    )

    # Expect running requests and kvcache usage
    assert running_requests > 0
    assert kv_cache_usage > 0

    # Cancel all tasks to abort the requests
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # Poll until metrics reset rather than using a fixed sleep
    await _poll_until(
        lambda: _get_running_metrics_from_api(server) == (0, 0, 0),
        timeout=10.0,
        interval=0.2,
        description="gauge metrics back to zero",
    )

    # Verify running and waiting requests counts and KV cache usage are zero
    running_requests_after, waiting_requests_after, kv_cache_usage_after = (
        _get_running_metrics_from_api(server)
    )

    assert running_requests_after == 0, (
        f"Expected 0 running requests after abort, got {running_requests_after}"
    )
    assert waiting_requests_after == 0, (
        f"Expected 0 waiting requests after abort, got {waiting_requests_after}"
    )
    assert kv_cache_usage_after == 0, (
        f"Expected 0% KV cache usage after abort, got {kv_cache_usage_after}"
    )


async def _poll_until(
    predicate, *, timeout: float, interval: float = 0.5, description: str = "condition"
):
    """Poll until predicate() returns True, or raise TimeoutError."""
    start = time.time()
    while time.time() - start < timeout:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise TimeoutError(f"Timed out after {timeout}s waiting for: {description}")


def _get_running_metrics_from_api(server: RemoteOpenAIServer):
    """Return (running_count, waiting_count, kv_cache_usage)."""
    response = requests.get(server.url_for("metrics"))
    assert response.status_code == HTTPStatus.OK

    # Verify running and waiting requests counts and KV cache usage are zero
    running_requests, waiting_requests, kv_cache_usage = None, None, None

    kv_cache_usage_metric = "vllm:kv_cache_usage_perc"

    for family in text_string_to_metric_families(response.text):
        if family.name == "vllm:num_requests_running":
            for sample in family.samples:
                if sample.name == "vllm:num_requests_running":
                    running_requests = sample.value
                    break
        elif family.name == "vllm:num_requests_waiting":
            for sample in family.samples:
                if sample.name == "vllm:num_requests_waiting":
                    waiting_requests = sample.value
                    break
        elif family.name == kv_cache_usage_metric:
            for sample in family.samples:
                if sample.name == kv_cache_usage_metric:
                    kv_cache_usage = sample.value
                    break

    assert running_requests is not None
    assert waiting_requests is not None
    assert kv_cache_usage is not None

    return running_requests, waiting_requests, kv_cache_usage


def test_metrics_exist_run_batch():
    input_batch = """{"custom_id": "request-0", "method": "POST", "url": "/v1/embeddings", "body": {"model": "intfloat/multilingual-e5-small", "input": "You are a helpful assistant."}}"""  # noqa: E501

    base_url = "0.0.0.0"
    port = str(get_open_port())
    server_url = f"http://{base_url}:{port}"

    with (
        tempfile.NamedTemporaryFile("w") as input_file,
        tempfile.NamedTemporaryFile("r") as output_file,
    ):
        input_file.write(input_batch)
        input_file.flush()
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vllm.entrypoints.launchers.run_batch",
                "-i",
                input_file.name,
                "-o",
                output_file.name,
                "--model",
                "intfloat/multilingual-e5-small",
                "--enable-metrics",
                "--host",
                base_url,
                "--port",
                port,
            ],
        )

        try:

            def is_server_up(url):
                try:
                    response = requests.get(url)
                    return response.status_code == 200
                except requests.ConnectionError:
                    return False

            start = time.time()
            timeout = 120
            while not is_server_up(server_url):
                if proc.poll() is not None:
                    pytest.fail(
                        f"Batch process exited early with returncode={proc.returncode}"
                    )
                if time.time() - start > timeout:
                    pytest.fail("Batch server did not start within timeout")
                time.sleep(1)

            response = requests.get(server_url + "/metrics")
            assert response.status_code == HTTPStatus.OK
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
