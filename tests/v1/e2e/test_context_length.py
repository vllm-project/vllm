# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
end-to-end tests for context length corner cases of vLLM v1 model runner
versus HuggingFace's transformers.

This test verifies the following behavior: allow a prefill that fills the
model's maximum context length and then request a single new token.

Test strategy
- Build a textual prompt that tokenizes to exactly ``max_model_len`` tokens.
- Run vLLM generation requesting a single new token (max_tokens=1).
- Run HF generation on the same prompt requesting a single token too.
- Assert both return the same number of generated tokens and the same ids.

"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from tests.models.utils import check_outputs_equal
from tests.utils import create_new_process_for_each_test
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


@create_new_process_for_each_test()
@pytest.mark.parametrize("model", ["JackFram/llama-160m"])
@pytest.mark.parametrize("max_model_len", [2048])
@pytest.mark.parametrize("max_tokens", [1])
def test_prefill_max_context_length(
    model: str,
    max_model_len: int,
    max_tokens: int,
) -> None:
    """Compare vLLM and HuggingFace when the prompt already fills the
    model's maximum context length and we request a single new token.

    The test ensures vLLM does not raise the "Sampled token IDs exceed the
    max model length" assertion and that both vLLM and HF produce the same
    single token when given the same inputs.
    """

    # Construct a prompt of size max_model_len
    prompt_ids = [[43] * max_model_len]

    # Generate max_tokens new tokens deterministically.
    sampling_params = [
        SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
    ]

    # --- vLLM generation ---
    llm = LLM(
        model=model,
        tokenizer=model,
        max_num_seqs=1,
        tensor_parallel_size=1,
    )

    vllm_token_prompts = [TokensPrompt(prompt_token_ids=prompt_ids[0])]
    vllm_results = llm.generate(vllm_token_prompts, sampling_params)

    vllm_output_ids = vllm_results[0].outputs[0].token_ids

    # --- HuggingFace generation ---
    with torch.no_grad():
        hf_model = AutoModelForCausalLM.from_pretrained(model)

        # HF expects a tensor of input ids shaped (batch, seq_len).
        hf_input_tokens = torch.tensor(prompt_ids[0]).unsqueeze(0)

        # Generate max_tokens new tokens deterministically.
        hf_generated = hf_model.generate(
            hf_input_tokens,
            do_sample=False,
            min_new_tokens=max_tokens,
            max_new_tokens=max_tokens,
        )

        # HF returns the prompt + generated tokens. Slice off the prompt.
        hf_output_ids = hf_generated.cpu().tolist()[0][len(prompt_ids[0]):]

    # check that vLLM outputs (token ids) match HF outputs
    # Note: for simplicity don't pass detokenized string
    check_outputs_equal(
        outputs_0_lst=[(hf_output_ids, "")],
        outputs_1_lst=[(vllm_output_ids, "")],
        name_0="hf",
        name_1="vllm",
    )
