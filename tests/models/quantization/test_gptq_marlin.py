# SPDX-License-Identifier: Apache-2.0
"""Compares the outputs of gptq vs gptq_marlin.

Note: GPTQ and Marlin do not have bitwise correctness.
As a result, in this test, we just confirm that the top selected tokens of the
Marlin/GPTQ models are in the top 5 selections of each other.
Note: Marlin internally uses locks to synchronize the threads. This can
result in very slight nondeterminism for Marlin. As a result, we re-run the test
up to 3 times to see if we pass.
"""
import os

import pytest

from tests.quantization.utils import is_quant_method_supported
from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT

from ..utils import check_logprobs_close

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MAX_MODEL_LEN = 1024

MODELS = [
    # act_order==True, group_size=128
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "main"),

    # 8-bit, act_order==True, group_size=channelwise
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "gptq-8bit--1g-actorder_True"),

    # 4-bit, act_order==True, group_size=128
    ("TechxGenus/gemma-1.1-2b-it-GPTQ", "main")
]


@pytest.mark.flaky(reruns=3)
@pytest.mark.skipif(not is_quant_method_supported("gptq_marlin"),
                    reason="gptq_marlin is not supported on this GPU type.")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half", "bfloat16"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    vllm_runner,
    example_prompts,
    model,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    model_name, revision = model

    # Run marlin.
    with vllm_runner(model_name=model_name,
                     revision=revision,
                     dtype=dtype,
                     quantization="marlin",
                     max_model_len=MAX_MODEL_LEN,
                     tensor_parallel_size=1) as gptq_marlin_model:

        gptq_marlin_outputs = gptq_marlin_model.generate_greedy_logprobs(
            example_prompts[:-1], max_tokens, num_logprobs)
    _ROPE_DICT.clear()  # clear rope cache to avoid rope dtype error

    # Run gptq.
    # The naive gptq kernel doesn't support bf16 yet.
    # Here we always compare fp16/bf16 gpt marlin kernel
    # to fp16 gptq kernel.
    with vllm_runner(model_name=model_name,
                     revision=revision,
                     dtype="half",
                     quantization="gptq",
                     max_model_len=MAX_MODEL_LEN,
                     tensor_parallel_size=1) as gptq_model:
        gptq_outputs = gptq_model.generate_greedy_logprobs(
            example_prompts[:-1], max_tokens, num_logprobs)

    check_logprobs_close(
        outputs_0_lst=gptq_outputs,
        outputs_1_lst=gptq_marlin_outputs,
        name_0="gptq",
        name_1="gptq_marlin",
    )
