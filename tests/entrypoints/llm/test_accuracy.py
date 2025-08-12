# SPDX-License-Identifier: Apache-2.0
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


def run_test(more_args=None):
    """Run the end to end accuracy test."""

    model_args = f"pretrained={MODEL_NAME},max_model_len=4096"

    if more_args is not None:
        model_args = "{},{}".format(model_args, more_args)

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


# TODO: [AlexM] Fix it with new CI/CD tests
TPU_TP_TEST_STR = ""  #"tensor_parallel_size=4"


@pytest.mark.skipif(not current_platform.is_cuda()
                    and not current_platform.is_tpu(),
                    reason="V1 is currently only supported on CUDA and TPU")
def test_lm_eval_accuracy_v1_engine(monkeypatch: pytest.MonkeyPatch):
    """Run with the V1 Engine."""

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        more_args = None
        if current_platform.is_tpu():
            # Limit compilation time for TPU V1
            more_args = "max_model_len=2048,max_num_seqs=64"

            # Add TP test (if provided)
            if TPU_TP_TEST_STR:
                more_args += ",{}".format(TPU_TP_TEST_STR)

        run_test(more_args)


def test_lm_eval_accuracy_v0_engine(monkeypatch: pytest.MonkeyPatch):
    """Run with the V0 Engine."""

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "0")
        run_test()
