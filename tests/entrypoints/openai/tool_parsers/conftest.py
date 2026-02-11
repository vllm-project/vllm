# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from vllm.tokenizers import TokenizerLike


@pytest.fixture(scope="function")
def default_tokenizer() -> TokenizerLike:
    return AutoTokenizer.from_pretrained("gpt2")
