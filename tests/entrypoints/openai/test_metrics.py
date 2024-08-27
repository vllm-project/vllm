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

from ...utils import RemoteOpenAIServer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


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


@pytest.fixture(scope="module",
                params=[
                    "",
                    "--enable-chunked-prefill",
                    "--disable-frontend-multiprocessing",
                ])
def server(default_server_args, request):
    if request.param:
        default_server_args.append(request.param)
    with RemoteOpenAIServer(MODEL_NAME, default_server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as cl:
        yield cl


_PROMPT = "Hello my name is Robert and I love magic"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_TOKENIZED_PROMPT = tokenizer(_PROMPT)["input_ids"]

_NUM_REQUESTS = 10
_NUM_PROMPT_TOKENS_PER_REQUEST = len(_TOKENIZED_PROMPT)
_NUM_GENERATION_TOKENS_PER_REQUEST = 10

# {metric_family: [(suffix, expected_value)]}
EXPECTED_VALUES = {
    "vllm:time_to_first_token_seconds": [("_count", _NUM_REQUESTS)],
    "vllm:time_per_output_token_seconds":
    [("_count", _NUM_REQUESTS * (_NUM_GENERATION_TOKENS_PER_REQUEST - 1))],
    "vllm:e2e_request_latency_seconds": [("_count", _NUM_REQUESTS)],
    "vllm:request_prompt_tokens":
    [("_sum", _NUM_REQUESTS * _NUM_PROMPT_TOKENS_PER_REQUEST),
     ("_count", _NUM_REQUESTS)],
    "vllm:request_generation_tokens":
    [("_sum", _NUM_REQUESTS * _NUM_GENERATION_TOKENS_PER_REQUEST),
     ("_count", _NUM_REQUESTS)],
    "vllm:request_params_n": [("_count", _NUM_REQUESTS)],
    "vllm:request_params_best_of": [("_count", _NUM_REQUESTS)],
    "vllm:prompt_tokens": [("_total",
                            _NUM_REQUESTS * _NUM_PROMPT_TOKENS_PER_REQUEST)],
    "vllm:generation_tokens":
    [("_total", _NUM_REQUESTS * _NUM_PROMPT_TOKENS_PER_REQUEST)],
    "vllm:request_success": [("_total", _NUM_REQUESTS)],
}


@pytest.mark.asyncio
async def test_metrics_counts(client: openai.AsyncOpenAI):
    base_url = str(client.base_url)[:-3].strip("/")

    for _ in range(_NUM_REQUESTS):
        # sending a request triggers the metrics to be logged.
        await client.completions.create(
            model=MODEL_NAME,
            prompt=_TOKENIZED_PROMPT,
            max_tokens=_NUM_GENERATION_TOKENS_PER_REQUEST)

    response = requests.get(base_url + "/metrics")
    print(response.text)
    assert response.status_code == HTTPStatus.OK

    # Loop over all expected metric_families
    for metric_family, suffix_values_list in EXPECTED_VALUES.items():
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
                                f"{sample.value}")
                            break
                    assert found_suffix, (
                        f"Did not find {metric_name_w_suffix} in prom endpoint"
                    )
                break

        assert found_metric, (f"Did not find {metric_family} in prom endpoint")


EXPECTED_METRICS = [
    "vllm:num_requests_running",
    "vllm:num_requests_swapped",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:cpu_cache_usage_perc",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_bucket",
    "vllm:time_to_first_token_seconds_count",
    "vllm:time_per_output_token_seconds_sum",
    "vllm:time_per_output_token_seconds_bucket",
    "vllm:time_per_output_token_seconds_count",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:e2e_request_latency_seconds_bucket",
    "vllm:e2e_request_latency_seconds_count",
    "vllm:request_prompt_tokens_sum",
    "vllm:request_prompt_tokens_bucket",
    "vllm:request_prompt_tokens_count",
    "vllm:request_generation_tokens_sum",
    "vllm:request_generation_tokens_bucket",
    "vllm:request_generation_tokens_count",
    "vllm:request_params_n_sum",
    "vllm:request_params_n_bucket",
    "vllm:request_params_n_count",
    "vllm:request_params_best_of_sum",
    "vllm:request_params_best_of_bucket",
    "vllm:request_params_best_of_count",
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:cache_config_info",
    # labels in cache_config_info
    "block_size",
    "cache_dtype",
    "cpu_offload_gb",
    "enable_prefix_caching",
    "gpu_memory_utilization",
    "num_cpu_blocks",
    "num_gpu_blocks",
    "num_gpu_blocks_override",
    "sliding_window",
    "swap_space_bytes",
]


@pytest.mark.asyncio
async def test_metrics_exist(client: openai.AsyncOpenAI):
    base_url = str(client.base_url)[:-3].strip("/")

    # sending a request triggers the metrics to be logged.
    await client.completions.create(model=MODEL_NAME,
                                    prompt="Hello, my name is",
                                    max_tokens=5,
                                    temperature=0.0)

    response = requests.get(base_url + "/metrics")
    assert response.status_code == HTTPStatus.OK

    for metric in EXPECTED_METRICS:
        assert metric in response.text


def test_metrics_exist_run_batch():
    input_batch = """{"custom_id": "request-0", "method": "POST", "url": "/v1/embeddings", "body": {"model": "intfloat/e5-mistral-7b-instruct", "input": "You are a helpful assistant."}}"""  # noqa: E501

    base_url = "0.0.0.0"
    port = "8001"
    server_url = f"http://{base_url}:{port}"

    with tempfile.NamedTemporaryFile(
            "w") as input_file, tempfile.NamedTemporaryFile(
                "r") as output_file:
        input_file.write(input_batch)
        input_file.flush()
        proc = subprocess.Popen([
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.run_batch",
            "-i",
            input_file.name,
            "-o",
            output_file.name,
            "--model",
            "intfloat/e5-mistral-7b-instruct",
            "--enable-metrics",
            "--url",
            base_url,
            "--port",
            port,
        ], )

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
