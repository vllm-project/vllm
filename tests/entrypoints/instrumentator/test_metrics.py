# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import subprocess
import sys
import tempfile
import time
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
        "--disable-frontend-multiprocessing",
        f"--show-hidden-metrics-for-version={PREV_MINOR_VERSION}",
    ],
)
def server(model_key, default_server_args, request):
    if request.param:
        default_server_args.append(request.param)

    model_name = MODELS[model_key]
    with RemoteOpenAIServer(model_name, default_server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as cl:
        yield cl


_PROMPT = "Hello my name is Robert and I love magic"


def _get_expected_values(num_requests: int, prompt_ids: list[int], max_tokens: int):
    num_prompt_tokens = len(prompt_ids)

    # {metric_family: [(suffix, expected_value)]}
    return {
        "vllm:time_to_first_token_seconds": [("_count", num_requests)],
        "vllm:time_per_output_token_seconds": [
            ("_count", num_requests * (max_tokens - 1))
        ],
        "vllm:e2e_request_latency_seconds": [("_count", num_requests)],
        "vllm:request_queue_time_seconds": [("_count", num_requests)],
        "vllm:request_inference_time_seconds": [("_count", num_requests)],
        "vllm:request_prefill_time_seconds": [("_count", num_requests)],
        "vllm:request_decode_time_seconds": [("_count", num_requests)],
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
                num_requests * (num_prompt_tokens + max_tokens),
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
    num_requests = 10
    max_tokens = 10

    for _ in range(num_requests):
        # sending a request triggers the metrics to be logged.
        await client.completions.create(
            model=model_name,
            prompt=prompt_ids,
            max_tokens=max_tokens,
        )

    response = requests.get(server.url_for("metrics"))
    print(response.text)
    assert response.status_code == HTTPStatus.OK

    # Loop over all expected metric_families
    expected_values = _get_expected_values(num_requests, prompt_ids, max_tokens)
    for metric_family, suffix_values_list in expected_values.items():
        if metric_family not in EXPECTED_METRICS_V1 or (
            not server.show_hidden_metrics
            and metric_family in HIDDEN_DEPRECATED_METRICS
        ):
            continue

        found_metric = False

        # Check to see if the metric_family is found in the prom endpoint.
        for family in text_string_to_metric_families(response.text):
            if family.name == metric_family:
                found_metric = True

                # Check that each suffix is found in the prom endpoint.
                for suffix, expected_value in suffix_values_list:
                    metric_name_w_suffix = f"{metric_family}{suffix}"
                    found_suffix = False

                    for sample in family.samples:
                        if sample.name == metric_name_w_suffix:
                            found_suffix = True

                            # For each suffix, value sure the value matches
                            # what we expect.
                            assert sample.value == expected_value, (
                                f"{metric_name_w_suffix} expected value of "
                                f"{expected_value} did not match found value "
                                f"{sample.value}"
                            )
                            break
                    assert found_suffix, (
                        f"Did not find {metric_name_w_suffix} in prom endpoint"
                    )
                break

        assert found_metric, f"Did not find {metric_family} in prom endpoint"


EXPECTED_METRICS_V1 = [
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
    "vllm:num_preemptions_total",
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
    "vllm:time_per_output_token_seconds_sum",
    "vllm:time_per_output_token_seconds_bucket",
    "vllm:time_per_output_token_seconds_count",
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
]

EXPECTED_METRICS_MM = [
    "vllm:mm_cache_queries",
    "vllm:mm_cache_hits",
]

HIDDEN_DEPRECATED_METRICS: list[str] = [
    "vllm:gpu_cache_usage_perc",
    "vllm:gpu_prefix_cache_queries",
    "vllm:gpu_prefix_cache_hits",
    "vllm:time_per_output_token_seconds_sum",
    "vllm:time_per_output_token_seconds_bucket",
    "vllm:time_per_output_token_seconds_count",
]


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
                max_tokens=100,  # Long generation to give time to abort
                temperature=0.0,
            )
        )
        tasks.append(task)

    # Wait a bit for requests to start processing
    await asyncio.sleep(0.5)

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

    # Wait for cancellations to be processed
    await asyncio.sleep(1.0)

    # Check that metrics have reset to zero
    response = requests.get(server.url_for("metrics"))
    assert response.status_code == HTTPStatus.OK

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


def _get_running_metrics_from_api(server: RemoteOpenAIServer):
    """Return (running_count, waiting_count, kv_cache_usage)"""

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
    port = "8001"
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
                "vllm.entrypoints.openai.run_batch",
                "-i",
                input_file.name,
                "-o",
                output_file.name,
                "--model",
                "intfloat/multilingual-e5-small",
                "--enable-metrics",
                "--url",
                base_url,
                "--port",
                port,
            ],
        )

        def is_server_up(url):
            try:
                response = requests.get(url)
                return response.status_code == 200
            except requests.ConnectionError:
                return False

        while not is_server_up(server_url):
            time.sleep(1)

        response = requests.get(server_url + "/metrics")
        assert response.status_code == HTTPStatus.OK

        proc.wait()
