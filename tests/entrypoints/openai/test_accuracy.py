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

from vllm.platforms import current_platform

from ...utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
NUM_CONCURRENT = 500
TASK = "gsm8k"
FILTER = "exact_match,strict-match"
RTOL = 0.03
EXPECTED_VALUE = 0.58
DEFAULT_ARGS = ["--max-model-len", "2048", "--disable-log-requests"]
MORE_ARGS_LIST = [
    [],  # Default
    ["--enable-chunked-prefill"],  # Chunked
    ["--num-scheduler-steps", "8"],  # MS
    ["--num-scheduler-steps", "8", "--multi-step-stream-outputs"]  # MS+Stream
]
MAX_WAIT_SECONDS = None

if current_platform.is_tpu():
    MORE_ARGS_LIST = [
        [],  # Default
        # ["--num-scheduler-steps", "8"], # Multi-step << currently fails
    ]
    MAX_WAIT_SECONDS = 600


@pytest.mark.parametrize("more_args", MORE_ARGS_LIST)
def test_lm_eval_accuracy(more_args):
    args = list(DEFAULT_ARGS)
    args.extend(more_args)

    print(f"Running with: {args}")

    with RemoteOpenAIServer(
            MODEL_NAME, args,
            max_wait_seconds=MAX_WAIT_SECONDS) as remote_server:
        url = f"{remote_server.url_for('v1')}/completions"

        model_args = (
            f"model={MODEL_NAME},"
            f"base_url={url},"
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
