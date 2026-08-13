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

import pytest

from tests.evals.gsm8k.gsm8k_eval import (
    assert_min_accuracy,
    evaluate_gsm8k_lm_eval,
)
from vllm.platforms import current_platform

MODEL_NAMES = [
    "Qwen/Qwen3-1.7B",
    "google/gemma-3-1b-it",
]
FP8_KV_MODEL_NAMES = [
    "Qwen/Qwen3-1.7B",
]
NUM_CONCURRENT = 500
RTOL = 0.03
EXPECTED_VALUES = {
    "Qwen/Qwen3-1.7B": 0.68,
    "google/gemma-3-1b-it": 0.25,
}


def run_test(model_name, more_args=None):
    """Run the end to end accuracy test."""

    model_args = f"pretrained={model_name},max_model_len=4096"

    if more_args is not None:
        model_args = "{},{}".format(model_args, more_args)

    result = evaluate_gsm8k_lm_eval(
        model="vllm",
        model_args=model_args,
        batch_size="auto",
    )

    assert model_name in EXPECTED_VALUES, (
        f"Cannot find the expected value for the model {model_name=}"
    )
    assert_min_accuracy(
        result,
        EXPECTED_VALUES[model_name],
        tolerance=RTOL,
        context=model_name,
    )


# TODO: [AlexM] Fix it with new CI/CD tests
TPU_TP_TEST_STR = ""  # "tensor_parallel_size=4"


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_lm_eval_accuracy_v1_engine(model):
    """Run with the V1 Engine."""

    more_args = None
    if current_platform.is_tpu():
        # Limit compilation time for TPU V1

        more_args = "max_model_len=2048,max_num_seqs=64"

        # Add TP test (if provided)
        if TPU_TP_TEST_STR:
            more_args += ",{}".format(TPU_TP_TEST_STR)

    run_test(model, more_args)


@pytest.mark.parametrize("model", FP8_KV_MODEL_NAMES)
def test_lm_eval_accuracy_v1_engine_fp8_kv_cache(model):
    """Run with the V1 Engine."""

    more_args = None
    if current_platform.is_tpu():
        # Limit compilation time for TPU V1
        more_args = "max_model_len=2048,max_num_seqs=128,kv_cache_dtype=fp8"

        # Add TP test (if provided)
        if TPU_TP_TEST_STR:
            more_args += ",{}".format(TPU_TP_TEST_STR)

    run_test(model, more_args)
