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

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
NUM_CONCURRENT = 500
TASK = "gsm8k"
FILTER = "exact_match,strict-match"
RTOL = 0.03
EXPECTED_VALUE = 0.58


def run_test():
    """Run the end to end accuracy test."""

    model_args = f"pretrained={MODEL_NAME},max_model_len=2048"

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks="gsm8k",
        batch_size="auto",
    )

    measured_value = results["results"][TASK][FILTER]
    assert (measured_value - RTOL < EXPECTED_VALUE
            and measured_value + RTOL > EXPECTED_VALUE
            ), f"Expected: {EXPECTED_VALUE} |  Measured: {measured_value}"


@pytest.mark.skipif(not current_platform.is_cuda(),
                    reason="V1 is currently only supported on CUDA.")
def test_lm_eval_accuracy_v1_engine(monkeypatch):
    """Run with the V1 Engine."""

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        run_test()


def test_lm_eval_accuracy_v0_engine(monkeypatch):
    """Run with the V0 Engine."""

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "0")
        run_test()
