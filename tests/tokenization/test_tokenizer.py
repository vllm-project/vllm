# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import PreTrainedTokenizerBase

from vllm.transformers_utils.tokenizer import get_tokenizer

TOKENIZER_NAMES = [
    "facebook/opt-125m",
    "gpt2",
]


@pytest.mark.parametrize("tokenizer_name", TOKENIZER_NAMES)
def test_tokenizer_revision(tokenizer_name: str):
    # Assume that "main" branch always exists
    tokenizer = get_tokenizer(tokenizer_name, revision="main")
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    # Assume that "never" branch always does not exist
    with pytest.raises(OSError, match="not a valid git identifier"):
        get_tokenizer(tokenizer_name, revision="never")
