# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.renderers.params import TokenizeParams


def test_tokenize_params_default_return_token_offsets_false():
    """Default value of return_token_offsets must be False so existing
    callers see zero behavior change."""
    params = TokenizeParams(max_total_tokens=None)
    assert params.return_token_offsets is False


def test_tokenize_params_return_token_offsets_constructible_true():
    """The new field must be constructible via kwarg."""
    params = TokenizeParams(max_total_tokens=None, return_token_offsets=True)
    assert params.return_token_offsets is True


def test_tokens_prompt_supports_offsets_field():
    """TokensPrompt accepts the new prompt_token_offsets field as a
    NotRequired TypedDict member. TypedDict has no runtime validation,
    so the assertion is structural: the field shows up in __annotations__."""
    from vllm.inputs.llm import TokensPrompt

    assert "prompt_token_offsets" in TokensPrompt.__annotations__


def test_tokens_input_supports_offsets_field():
    from vllm.inputs.engine import TokensInput

    assert "prompt_token_offsets" in TokensInput.__annotations__
