# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.transformers_utils.tokenizer import get_tokenizer

TOKENIZER_NAMES = ["BAAI/bge-base-en"]


@pytest.mark.parametrize("tokenizer_name", TOKENIZER_NAMES)
@pytest.mark.parametrize("n_tokens", [510])
def test_special_tokens(tokenizer_name: str, n_tokens: int):
    tokenizer = get_tokenizer(tokenizer_name, revision="main")

    prompts = '[UNK]' * n_tokens
    prompt_token_ids = tokenizer.encode(prompts)
    assert len(prompt_token_ids) == n_tokens + 2
