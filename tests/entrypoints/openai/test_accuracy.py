"""
This file test accuracy of the vLLM server via LMEval.
It uses local-completions, which interacts with vLLM
through the OAI API with N concurrent connections.
This simulates real work usage of the API and makes
sure that the zmq frontend mp RPC message passing and
AsyncLLMEngine are working correctly.
"""

import lm_eval
import pytest

from ...utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
NUM_CONCURRENT = 500
TASK = "gsm8k"
FILTER = "exact_match,strict-match"
RTOL = 0.03
EXPECTED_VALUE = 0.58


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len", "4096", "--enable-chunked-prefill",
        "--disable-log-requests", "--enforce-eager"
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def server_data(server):
    return {
        "url": f"{server.url_for('v1')}/completions",
    }


def test_lm_eval_accuracy(server_data):
    model_args = (f"model={MODEL_NAME},"
                  f"base_url={server_data['url']},"
                  f"num_concurrent={NUM_CONCURRENT},tokenized_requests=False")

    results = lm_eval.simple_evaluate(
        model="local-completions",
        model_args=model_args,
        tasks=TASK,
    )

    measured_value = results["results"][TASK][FILTER]
    assert (measured_value - RTOL < EXPECTED_VALUE
            and measured_value + RTOL > EXPECTED_VALUE
            ), f"Expected: {EXPECTED_VALUE} |  Measured: {measured_value}"
