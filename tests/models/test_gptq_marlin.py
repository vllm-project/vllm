"""Compares the outputs of gptq vs gptq_marlin 
Note: GPTQ and Marlin do not have bitwise correctness.
As a result, in this test, we just confirm that the top selected tokens of the
Marlin/GPTQ models are in the top 3 selections of each other.
Note: Marlin internally uses locks to synchronize the threads. This can
result in very slight nondeterminism for Marlin. As a result, we re-run the test
up to 3 times to see if we pass.
Note: This test currently fails running with --forked with the following:
    RuntimeError: Cannot re-initialize CUDA in forked subprocess.
    To use CUDA with multiprocessing, you must use the 'spawn' start method
Run `pytest tests/models/test_gptq_marlin.py`.
"""
import os

import pytest
import torch

from tests.models.utils import check_logprobs_close
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MAX_MODEL_LEN = 1024

capability = torch.cuda.get_device_capability()
capability = capability[0] * 10 + capability[1]
gptq_marlin_not_supported = (
    capability < QUANTIZATION_METHODS["gptq_marlin"].get_min_capability())

MODELS = [
    # act_order==False, group_size=channelwise
    ("robertgshaw2/zephyr-7b-beta-channelwise-gptq", "main"),
    # act_order==False, group_size=128
    ("TheBloke/Llama-2-7B-GPTQ", "main"),

    # act_order==True, group_size=128
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "main"),
    # act_order==True, group_size=64
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "gptq-4bit-64g-actorder_True"),
    # act_order==True, group_size=32
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "gptq-4bit-32g-actorder_True"),

    # 8-bit, act_order==True, group_size=channelwise
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "gptq-8bit--1g-actorder_True"),
    # 8-bit, act_order==True, group_size=128
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "gptq-8bit-128g-actorder_True"),
    # 8-bit, act_order==True, group_size=32
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "gptq-8bit-32g-actorder_True"),
]


@pytest.mark.flaky(reruns=2)
@pytest.mark.skipif(gptq_marlin_not_supported,
                    reason="gptq_marlin is not supported on this GPU type.")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
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
    gptq_marlin_model = vllm_runner(model_name=model_name,
                                    revision=revision,
                                    dtype=dtype,
                                    quantization="marlin",
                                    max_model_len=MAX_MODEL_LEN,
                                    tensor_parallel_size=1)

    gptq_marlin_outputs = gptq_marlin_model.generate_greedy_logprobs(
        example_prompts, max_tokens, num_logprobs)
    del gptq_marlin_model

    # Run gptq.
    gptq_model = vllm_runner(model_name=model_name,
                             revision=revision,
                             dtype=dtype,
                             quantization="gptq",
                             max_model_len=MAX_MODEL_LEN,
                             tensor_parallel_size=1)
    gptq_outputs = gptq_model.generate_greedy_logprobs(example_prompts,
                                                       max_tokens,
                                                       num_logprobs)
    del gptq_model

    check_logprobs_close(
        outputs_0_lst=gptq_outputs,
        outputs_1_lst=gptq_marlin_outputs,
        name_0="gptq",
        name_1="gptq_marlin",
    )
