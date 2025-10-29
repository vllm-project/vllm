# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from vllm.transformers_utils.tokenizer import AnyTokenizer


@pytest.fixture(scope="function")
def default_tokenizer() -> AnyTokenizer:
    return AutoTokenizer.from_pretrained("gpt2")
