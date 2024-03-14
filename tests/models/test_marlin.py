"""Compare the outputs of a GPTQ model to a Marlin model.

Note: GPTQ and Marlin do not have bitwise correctness.
As a result, in this test, we just confirm that the top selected tokens of the
Marlin/GPTQ models are in the top 3 selections of each other.

Note: Marlin internally uses locks to synchronize the threads. This can
result in very slight nondeterminism for Marlin. As a result, we re-run the test
up to 3 times to see if we pass.

Note: This test currently fails running with --forked with the following:
    RuntimeError: Cannot re-initialize CUDA in forked subprocess.
    To use CUDA with multiprocessing, you must use the 'spawn' start method

Run `pytest tests/models/test_marlin.py`.
"""

import pytest
import torch
import gc
from compare_utils import check_logprobs_close
from dataclasses import dataclass
from vllm.model_executor.layers.quantization import _QUANTIZATION_CONFIG_REGISTRY

MAX_MODEL_LEN = 1024

capability = torch.cuda.get_device_capability()
capability = capability[0] * 10 + capability[1]
marlin_not_supported = (
    capability < _QUANTIZATION_CONFIG_REGISTRY["marlin"].get_min_capability())


@dataclass
class ModelPair:
    model_marlin: str
    model_gptq: str


model_pairs = [
    ModelPair(model_marlin="nm-testing/zephyr-beta-7b-marlin-g128",
              model_gptq="nm-testing/zephyr-beta-7b-gptq-g128"),
    ModelPair(model_marlin="robertgshaw2/zephyr-7b-beta-channelwise-marlin",
              model_gptq="robertgshaw2/zephyr-7b-beta-channelwise-gptq"),
    ModelPair(model_marlin="robertgshaw2/TinyLlama-1.1B-Chat-v1.0-g128-marlin",
              model_gptq="robertgshaw2/TinyLlama-1.1B-Chat-v1.0-g128-gptq")
]


@pytest.mark.flaky(reruns=2)
@pytest.mark.skipif(marlin_not_supported,
                    reason="Marlin is not supported on this GPU type.")
@pytest.mark.parametrize("model_pair", model_pairs)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    vllm_runner_nm,
    example_prompts,
    model_pair: ModelPair,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    marlin_model = vllm_runner_nm(model_pair.model_marlin,
                                  dtype=dtype,
                                  max_model_len=MAX_MODEL_LEN)
    marlin_outputs = marlin_model.generate_greedy_logprobs(
        example_prompts, max_tokens, num_logprobs)

    # vllm memory cleanup is poor. This seems to fix things.
    # NOTE: upstream sync should use downstream version.
    del marlin_model.model.llm_engine.driver_worker
    del marlin_model

    gc.collect()
    torch.cuda.empty_cache()

    gptq_model = vllm_runner_nm(model_pair.model_gptq,
                                dtype=dtype,
                                max_model_len=MAX_MODEL_LEN)
    gptq_outputs = gptq_model.generate_greedy_logprobs(example_prompts,
                                                       max_tokens,
                                                       num_logprobs)

    # vllm memory cleanup is poor. This seems to fix things.
    # NOTE: upstream sync should use downstream version.
    del gptq_model.model.llm_engine.driver_worker
    del gptq_model
    gc.collect()
    torch.cuda.empty_cache()

    # loop through the prompts
    # use logprobs or else this will consistently run out of memory
    check_logprobs_close(
        outputs_0_lst=gptq_outputs,
        outputs_1_lst=marlin_outputs,
        name_0="gptq",
        name_1="marlin",
    )
