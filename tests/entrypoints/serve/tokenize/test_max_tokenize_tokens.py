# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.entrypoints.serve.tokenize.protocol import TokenizeCompletionRequest
from vllm.exceptions import VLLMValidationError


class _CharTokenizer:
    max_chars_per_token = 1


def test_tokenize_rejects_prompt_over_configured_token_limit():
    """The /tokenize request's configured cap must reach renderer validation."""
    model_config = SimpleNamespace(max_tokenize_tokens=2, max_model_len=100)
    request = TokenizeCompletionRequest(model="test-model", prompt="too long")
    tok_params = request.build_tok_params(model_config)

    with pytest.raises(VLLMValidationError, match="maximum context length"):
        tok_params.apply_pre_tokenization(_CharTokenizer(), {"prompt": "abc"})
