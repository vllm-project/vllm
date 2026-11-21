# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm import LLM, SamplingParams
from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig

MODEL = "vllm-project/tiny-random-LlamaForCausalLM"
VOCAB_SIZE = 32000
UNMASKED_TOKEN_ID = 1234


def test_masked_token_ids_never_sampled():
    """Masking all but one token leaves it as the only possible sample."""
    masked = [i for i in range(VOCAB_SIZE) if i != UNMASKED_TOKEN_ID]
    llm = LLM(MODEL, enforce_eager=True, masked_token_ids=masked)
    for params in (
        SamplingParams(temperature=0, max_tokens=8),
        SamplingParams(temperature=1.0, max_tokens=8, seed=0),
    ):
        output = llm.generate("Hello my name is", params)
        assert output[0].outputs[0].token_ids == [UNMASKED_TOKEN_ID] * 8


@pytest.mark.parametrize("token_id", [-1, VOCAB_SIZE])
def test_masked_token_ids_out_of_range(token_id: int):
    with pytest.raises(ValueError, match="masked_token_ids"):
        ModelConfig(MODEL, masked_token_ids=[token_id])


@pytest.mark.parametrize(
    "lossy_kwargs",
    [
        {"rejection_sample_method": "fly"},
        {"rejection_sample_method": "synthetic", "synthetic_acceptance_length": 1.5},
    ],
)
def test_masked_token_ids_rejects_lossy_spec_decode(lossy_kwargs: dict):
    target = ModelConfig(MODEL, masked_token_ids=[0])
    with pytest.raises(ValueError, match="masked_token_ids is not supported"):
        SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=2,
            target_model_config=target,
            target_parallel_config=ParallelConfig(),
            **lossy_kwargs,
        )
