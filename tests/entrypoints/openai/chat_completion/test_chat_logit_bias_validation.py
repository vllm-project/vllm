# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams

pytestmark = pytest.mark.skip_global_cleanup


class _FakeModelConfig:
    max_logprobs = 5
    logits_processors = None

    def __init__(self, vocab_size: int):
        self._vocab_size = vocab_size

    def get_vocab_size(self) -> int:
        return self._vocab_size


def _verify_logit_bias(logit_bias: dict[str, float], vocab_size: int) -> None:
    params = SamplingParams.from_optional(max_tokens=5, logit_bias=logit_bias)
    params.verify(
        model_config=_FakeModelConfig(vocab_size),
        speculative_config=None,
        structured_outputs_config=None,
        tokenizer=None,
    )


def test_chat_logit_bias_valid():
    """Test that valid logit_bias token IDs are accepted."""
    vocab_size = 10
    valid_token_id = vocab_size - 1

    _verify_logit_bias({str(valid_token_id): 1.0}, vocab_size)


def test_chat_logit_bias_invalid():
    """Test that invalid logit_bias token IDs are rejected."""
    vocab_size = 10
    invalid_token_id = vocab_size + 1

    with pytest.raises(VLLMValidationError) as excinfo:
        _verify_logit_bias({str(invalid_token_id): 1.0}, vocab_size)

    error = excinfo.value
    assert error.parameter == "logit_bias"
    assert error.value == [invalid_token_id]
    assert str(invalid_token_id) in str(error)
    assert str(vocab_size) in str(error)
