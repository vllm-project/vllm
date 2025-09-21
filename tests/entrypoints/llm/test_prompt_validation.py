# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import LLM


def test_empty_prompt():
    llm = LLM(model="openai-community/gpt2", enforce_eager=True)
    with pytest.raises(ValueError, match='decoder prompt cannot be empty'):
        llm.generate([""])


@pytest.mark.skip_v1
def test_out_of_vocab_token():
    llm = LLM(model="openai-community/gpt2", enforce_eager=True)
    with pytest.raises(ValueError, match='out of vocabulary'):
        llm.generate({"prompt_token_ids": [999999]})
