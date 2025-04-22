# SPDX-License-Identifier: Apache-2.0
"""Compare the outputs of a GPTQ model to a bitblas model.

Note: GPTQ and bitblas do not have bitwise correctness.
As a result, in this test, we just confirm that the top selected tokens of the
bitblas/GPTQ models are in the top 3 selections of each other.

Note: bitblas internally uses locks to synchronize the threads. This can
result in very slight nondeterminism for bitblas. As a result, we re-run the 
test up to 3 times to see if we pass.

Run `pytest tests/models/test_bitblas.py`.
"""
from dataclasses import dataclass

import pytest

from .utils import check_logprobs_close


@dataclass
class ModelPair:
    model_gptq: str


model_pairs = [
    ModelPair(model_gptq="hxbgsyxh/opt-125m-4bit-128g"),
]


@pytest.mark.flaky(reruns=2)
@pytest.mark.skipif(True, reason="BitBLAS takes too much time for tuning.")
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
    with vllm_runner(model_pair.model_gptq,
                     dtype=dtype,
                     quantization="bitblas") as bitblas_model:
        bitblas_outputs = bitblas_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)

    with vllm_runner(model_pair.model_gptq, dtype=dtype,
                     quantization="gptq") as gptq_model:
        gptq_outputs = gptq_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)

    check_logprobs_close(
        outputs_0_lst=gptq_outputs,
        outputs_1_lst=bitblas_outputs,
        name_0="gptq",
        name_1="gptq_bitblas",
    )
