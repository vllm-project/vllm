# SPDX-License-Identifier: Apache-2.0

import pytest
from ...utils import check_logprobs_close
    
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_model_equivalence_to_hf_greedy(
    hf_runner,
    vllm_runner,
    example_prompts,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
):
    # Path of the checkpoints
    DIR = '/block/granite/granite-hybridmoe-7b-a1b-base-pipecleaner-hf'
    
    with hf_runner(DIR, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs)

    with vllm_runner(DIR, dtype=dtype) as vllm_model:
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
