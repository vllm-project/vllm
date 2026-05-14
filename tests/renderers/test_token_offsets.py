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
