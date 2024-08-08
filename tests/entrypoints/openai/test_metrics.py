from http import HTTPStatus

import openai
import pytest
import requests

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
        "--disable-frontend-multiprocessing"
    ]


@pytest.fixture(scope="module",
                params=["", "--disable-frontend-multiprocessing"])
def client(default_server_args, request):
    if request.param:
        default_server_args.append(request.param)
    with RemoteOpenAIServer(MODEL_NAME, default_server_args) as remote_server:
        yield remote_server.get_async_client()


EXPECTED_METRICS = [
    "vllm:num_requests_running",
    "vllm:num_requests_swapped",
    "vllm:num_requests_waiting",
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
