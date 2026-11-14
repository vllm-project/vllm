# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.logprobs import Logprob


def _serving() -> OpenAIServingCompletion:
    serving = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
    serving.return_tokens_as_token_ids = False
    return serving


def test_completion_logprobs_handles_sampled_token_absent_from_step():
    """The sampled token can be missing from a step's top-logprobs dict (e.g.
    when the top logprobs are all special tokens). The completion path must
    not raise KeyError, mirroring the chat path guard added in #17637; that
    fix only landed on the chat surface.
    """
    serving = _serving()
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "<special>"

    # Sampled token 5 is not a key in the step's top-logprobs dict.
    top_logprobs: list[dict[int, Logprob] | None] = [
        {
            1: Logprob(logprob=-0.1, rank=1, decoded_token="a"),
            2: Logprob(logprob=-0.2, rank=2, decoded_token="b"),
        }
    ]

    result = serving._create_completion_logprobs(
        token_ids=[5],
        top_logprobs=top_logprobs,
        num_output_top_logprobs=2,
        tokenizer=tokenizer,
    )

    assert result.tokens == ["<special>"]
    assert result.token_logprobs == [None]
    assert result.top_logprobs == [None]
    tokenizer.decode.assert_called_once_with(5)
