# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file test accuracy of the vLLM server via LMEval.
It uses local-completions, which interacts with vLLM
through the OAI API with N concurrent connections.
This simulates real work usage of the API and makes
sure that the zmq frontend mp RPC message passing and
AsyncLLMEngine are working correctly.
"""

from tests.evals.gsm8k.gsm8k_eval import (
    assert_gsm8k_result,
    evaluate_gsm8k_lm_eval,
    get_gsm8k_eval_spec,
)
from vllm.platforms import current_platform

from ....utils import RemoteOpenAIServer

GSM8K_SPEC = get_gsm8k_eval_spec("openai_entrypoint", "qwen2-1.5b-instruct")
assert GSM8K_SPEC.model is not None
MODEL_NAME = GSM8K_SPEC.model
NUM_CONCURRENT = 500
DEFAULT_ARGS = ["--max-model-len", "4096"]
MORE_ARGS_LIST = [
    [],  # Default
    ["--enable-chunked-prefill"],  # Chunked
]
MAX_WAIT_SECONDS = None

if current_platform.is_tpu():
    MORE_ARGS_LIST = [
        [],  # Default
    ]
    MAX_WAIT_SECONDS = 600


def run_test(more_args):
    """Run the end to end accuracy test."""

    args = list(DEFAULT_ARGS)
    args.extend(more_args)
    print(f"Running with: {args}")

    with RemoteOpenAIServer(
        MODEL_NAME, args, max_wait_seconds=MAX_WAIT_SECONDS
    ) as remote_server:
        url = f"{remote_server.url_for('v1')}/completions"

        model_args = (
            f"model={MODEL_NAME},"
            f"base_url={url},"
            f"num_concurrent={NUM_CONCURRENT},tokenized_requests=False"
        )

        result = evaluate_gsm8k_lm_eval(
            model="local-completions",
            model_args=model_args,
            **GSM8K_SPEC.lm_eval_kwargs(),
        )

        assert_gsm8k_result(result, GSM8K_SPEC)


def test_lm_eval_accuracy_v1_engine():
    """Run with the V1 Engine."""

    more_args = []

    # Limit compilation time for V1 on TPU
    # Avoid OOM on XPU
    if current_platform.is_tpu() or current_platform.is_xpu():
        more_args = ["--max-num-seqs", "64"]

    run_test(more_args)
