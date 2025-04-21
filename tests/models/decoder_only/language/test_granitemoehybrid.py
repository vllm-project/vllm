# SPDX-License-Identifier: Apache-2.0

import pytest
from ...utils import check_logprobs_close
    
# Path of the checkpoints
MODELS = [
    "/block/granite/granite-4.0-tiny-base-pipecleaner-hf",
    # "/code/granite/granite-4_0-small-base-pipecleaner-hf",
    # "/code/granitegranite-4_0-medium-base-pipecleaner-hf",
]

@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_model_equivalence_to_hf_greedy(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
):
    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs)

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)
        
    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )

if __name__ == "__main__":
    pytest.main(["tests/models/decoder_only/language/test_granitemoehybrid.py"])
