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
    GSM8KEvalSpec,
    assert_gsm8k_result,
    evaluate_gsm8k_lm_eval,
    load_gsm8k_eval_specs,
)
from vllm.platforms import current_platform

GSM8K_SPECS = {spec.id: spec for spec in load_gsm8k_eval_specs("llm_entrypoint")}
MODEL_SPECS = [GSM8K_SPECS["qwen3-1.7b"], GSM8K_SPECS["gemma3-1b-it"]]
FP8_KV_MODEL_SPECS = [GSM8K_SPECS["qwen3-1.7b-fp8-kv"]]


def run_test(gsm8k_spec: GSM8KEvalSpec, more_args=None):
    """Run the end to end accuracy test."""
    assert gsm8k_spec.model is not None
    model_name = gsm8k_spec.model

    model_args = f"pretrained={model_name},max_model_len=4096"

    if more_args is not None:
        model_args = "{},{}".format(model_args, more_args)

    result = evaluate_gsm8k_lm_eval(
        model="vllm",
        model_args=model_args,
        batch_size="auto",
        **gsm8k_spec.lm_eval_kwargs(),
    )

    assert_gsm8k_result(result, gsm8k_spec, context=model_name)


# TODO: [AlexM] Fix it with new CI/CD tests
TPU_TP_TEST_STR = ""  # "tensor_parallel_size=4"


@pytest.mark.parametrize("gsm8k_spec", MODEL_SPECS, ids=lambda spec: spec.id)
def test_lm_eval_accuracy_v1_engine(gsm8k_spec: GSM8KEvalSpec):
    """Run with the V1 Engine."""

    more_args = None
    if current_platform.is_tpu():
        # Limit compilation time for TPU V1

        more_args = "max_model_len=2048,max_num_seqs=64"

        # Add TP test (if provided)
        if TPU_TP_TEST_STR:
            more_args += ",{}".format(TPU_TP_TEST_STR)

    run_test(gsm8k_spec, more_args)


@pytest.mark.parametrize("gsm8k_spec", FP8_KV_MODEL_SPECS, ids=lambda spec: spec.id)
def test_lm_eval_accuracy_v1_engine_fp8_kv_cache(gsm8k_spec: GSM8KEvalSpec):
    """Run with the V1 Engine."""

    more_args = None
    if current_platform.is_tpu():
        # Limit compilation time for TPU V1
        more_args = "max_model_len=2048,max_num_seqs=128,kv_cache_dtype=fp8"

        # Add TP test (if provided)
        if TPU_TP_TEST_STR:
            more_args += ",{}".format(TPU_TP_TEST_STR)

    run_test(gsm8k_spec, more_args)
