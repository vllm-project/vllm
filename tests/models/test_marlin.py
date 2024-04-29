"""Compare the outputs of a GPTQ model to a Marlin model.

Note: GPTQ and Marlin do not have bitwise correctness.
As a result, in this test, we just confirm that the top selected tokens of the
Marlin/GPTQ models are in the top 3 selections of each other.

Note: Marlin internally uses locks to synchronize the threads. This can
result in very slight nondeterminism for Marlin. As a result, we re-run the test
up to 3 times to see if we pass.

Run `pytest tests/models/test_marlin.py`.
"""
from dataclasses import dataclass

import pytest
import torch

from tests.models.utils import check_logprobs_close
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

capability = torch.cuda.get_device_capability()
capability = capability[0] * 10 + capability[1]
marlin_not_supported = (capability <
                        QUANTIZATION_METHODS["marlin"].get_min_capability())


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
    vllm_runner,
    example_prompts,
    model_pair: ModelPair,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    marlin_model = vllm_runner(model_pair.model_marlin,
                               dtype=dtype,
                               quantization="marlin")
    marlin_outputs = marlin_model.generate_greedy_logprobs(
        example_prompts, max_tokens, num_logprobs)
    del marlin_model

    gptq_model = vllm_runner(model_pair.model_gptq,
                             dtype=dtype,
                             quantization="gptq")
    gptq_outputs = gptq_model.generate_greedy_logprobs(example_prompts,
                                                       max_tokens,
                                                       num_logprobs)
    del gptq_model

    check_logprobs_close(
        outputs_0_lst=gptq_outputs,
        outputs_1_lst=marlin_outputs,
        name_0="gptq",
        name_1="marlin",
    )
