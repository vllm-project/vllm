# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.generate.base.serving import align_step_logprobs
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.logprobs import Logprob


def _mock_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.decode = lambda ids: f"tok{ids[0] if isinstance(ids, (list, tuple)) else ids}"
    return tok


def _minimal_serving_chat() -> OpenAIServingChat:
    serving = OpenAIServingChat.__new__(OpenAIServingChat)
    serving.return_tokens_as_token_ids = False
    return serving


def _minimal_serving_completion() -> OpenAIServingCompletion:
    serving = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
    serving.return_tokens_as_token_ids = False
    return serving


@pytest.mark.parametrize(
    "top_logprobs",
    [
        [],
        [{10: Logprob(logprob=-0.1, decoded_token="a")}],
        [
            {10: Logprob(logprob=-0.1, decoded_token="a")},
            {11: Logprob(logprob=-0.2, decoded_token="b")},
            {12: Logprob(logprob=-0.3, decoded_token="c")},
        ],
    ],
    ids=["empty", "shorter", "longer"],
)
def test_align_step_logprobs_mismatched_lengths(top_logprobs):
    token_ids = [10, 11]
    aligned = align_step_logprobs(token_ids, top_logprobs)
    assert aligned == [None, None]


def test_align_step_logprobs_matching_length_passthrough():
    token_ids = [10, 11]
    top_logprobs = [
        {10: Logprob(logprob=-0.1, decoded_token="a")},
        None,
    ]
    assert align_step_logprobs(token_ids, top_logprobs) is top_logprobs


@pytest.mark.parametrize(
    "top_logprobs",
    [
        [],
        [{10: Logprob(logprob=-0.1, decoded_token="a")}],
        [
            {10: Logprob(logprob=-0.1, decoded_token="a")},
            {11: Logprob(logprob=-0.2, decoded_token="b")},
            {12: Logprob(logprob=-0.3, decoded_token="c")},
        ],
    ],
    ids=["empty", "shorter", "longer"],
)
def test_serving_logprobs_builders_handle_mismatched_lengths(top_logprobs):
    """Mismatched top_logprobs lengths must degrade to missing step logprobs
    rather than raising IndexError or misaligning tokens (#63751)."""
    token_ids = [10, 11]
    tok = _mock_tokenizer()

    chat_lp = _minimal_serving_chat()._create_chat_logprobs(
        token_ids=token_ids,
        top_logprobs=top_logprobs,
        tokenizer=tok,
        num_output_top_logprobs=1,
    )
    assert chat_lp.content is not None
    assert [c.token for c in chat_lp.content] == ["tok10", "tok11"]
    assert all(c.logprob == -9999.0 and c.top_logprobs == [] for c in chat_lp.content)

    comp_lp = _minimal_serving_completion()._create_completion_logprobs(
        token_ids=token_ids,
        top_logprobs=top_logprobs,
        tokenizer=tok,
        num_output_top_logprobs=1,
    )
    assert comp_lp.tokens == ["tok10", "tok11"]
    assert comp_lp.token_logprobs == [None, None]
    assert comp_lp.top_logprobs == [None, None]

    text_lp = ServingTokens._create_text_logprobs(
        None,
        token_ids=token_ids,
        top_logprobs=top_logprobs,
        tokenizer=tok,
        top_limit=1,
    )
    assert text_lp.content is not None
    assert [c.token for c in text_lp.content] == ["tok10", "tok11"]
    assert all(c.logprob == -9999.0 and c.top_logprobs == [] for c in text_lp.content)

    tokens_lp = ServingTokens._create_tokens_logprobs(
        None,
        token_ids=token_ids,
        top_logprobs=top_logprobs,
    )
    assert [c.token_id for c in tokens_lp.content] == [10, 11]
    assert all(c.logprob == -9999.0 for c in tokens_lp.content)
