# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm import SamplingParams

pytestmark = pytest.mark.skip_global_cleanup


def _model_config(vocab_size: int = 10):
    return SimpleNamespace(
        max_logprobs=20,
        logits_processors=None,
        get_vocab_size=lambda: vocab_size,
    )


def _verify_without_tokenizer(params: SamplingParams) -> None:
    params.verify(
        _model_config(),
        speculative_config=None,
        structured_outputs_config=None,
        tokenizer=None,
    )


@pytest.mark.parametrize("stop", ["foo", ["foo"]])
def test_stop_strings_require_tokenizer(stop: str | list[str]):
    with pytest.raises(ValueError, match="skip_tokenizer_init"):
        _verify_without_tokenizer(SamplingParams(stop=stop))


def test_tokenizer_free_stop_params():
    _verify_without_tokenizer(SamplingParams())
    _verify_without_tokenizer(SamplingParams(stop_token_ids=[1]))
