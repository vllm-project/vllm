# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
end-to-end tests for context length corner cases of vLLM v1 model runner
versus HuggingFace's transformers.

This test verifies the following behavior: allow prefill and decodes on the
model's maximum context length ``max_model_len`` and get one more token.

Test strategy
- Build a prompt consisting of exactly ``prompt_len`` tokens.
- Run vLLM generation requesting ``max_tokens`` new tokens.
- Run HF generation on the same prompt requesting the same number of tokens.
- Assert both return the same number of generated tokens and the same ids.

Test cases
- Prefill a prompt of ``max_model_len`` (2048) and request a single token which
will be sampled after the prefill (context length ``max_model_len``).
- Prefill a prompt of ``max_model_len`` - 1 (2047) and request two tokens where
the 1st will be sampled after the prefill and the 2nd after the first decode
(context length ``max_model_len``).

"""

import pytest

from tests.conftest import HfRunner, VllmRunner
from tests.models.utils import check_outputs_equal
from tests.utils import create_new_process_for_each_test


@create_new_process_for_each_test()
@pytest.mark.parametrize("model", ["JackFram/llama-160m"])
@pytest.mark.parametrize(
    "prompt_len, max_tokens",
    [
        (2048, 1),  # prompt_len = max_model_len
        (2047, 2),  # prompt_len = max_model_len - 1
    ],
)
def test_max_context_length(
    model: str,
    vllm_runner: type[VllmRunner],
    hf_runner: type[HfRunner],
    prompt_len: int,
    max_tokens: int,
) -> None:
    """Compare vLLM and HuggingFace when the prompt already fills the
    model's maximum context length and we request a single new token.

    The test ensures vLLM does not raise the "Sampled token IDs exceed the
    max model length" assertion and that both vLLM and HF produce the same
    single token when given the same inputs.
    """

    # Construct a prompt of size prompt_len
    prompt_ids = [[43] * prompt_len]

    # --- vLLM generation ---
    with vllm_runner(
        model_name=model,
        tokenizer_name=model,
        max_model_len=2048,
        max_num_seqs=1,
        tensor_parallel_size=1,
    ) as vllm_model:
        # Generate max_tokens new tokens deterministically.
        vllm_outputs = vllm_model.generate_greedy(prompt_ids, max_tokens)

    # --- HuggingFace generation ---
    with hf_runner(
        model_name=model,
    ) as hf_model:
        hf_outputs = hf_model.generate_greedy(prompt_ids, max_tokens)

    # vLLM and HF runners return prompt + generated tokens. Slice off the prompt.
    vllm_output_ids = vllm_outputs[0][0][prompt_len:]
    hf_output_ids = hf_outputs[0][0][prompt_len:]

    # check that exactly max_tokens tokens were generated with vLLM and HF
    assert len(vllm_output_ids) == len(hf_output_ids) == max_tokens

    # check that vLLM outputs (token ids) match HF outputs
    # Note: for simplicity don't pass detokenized string
    check_outputs_equal(
        outputs_0_lst=[(hf_output_ids, "")],
        outputs_1_lst=[(vllm_output_ids, "")],
        name_0="hf",
        name_1="vllm",
    )
