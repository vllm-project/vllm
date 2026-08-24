# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``TokenizeParams.with_kwargs`` overrides that are spelled as ``False``.

``padding`` and ``truncation`` accept the HuggingFace spellings documented at
https://huggingface.co/docs/transformers/en/pad_truncation, where ``False`` and
``"do_not_pad"`` / ``"do_not_truncate"`` mean the same thing. The two spellings
must therefore produce the same ``TokenizeParams``.
"""

import pytest

from vllm.renderers import TokenizeParams

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def params() -> TokenizeParams:
    return TokenizeParams(
        max_total_tokens=1024,
        max_output_tokens=16,
        pad_prompt_tokens=64,
        truncate_prompt_tokens=128,
    )


@pytest.mark.parametrize("truncation", [False, "do_not_truncate"])
def test_disabling_truncation_clears_truncate_prompt_tokens(params, truncation):
    assert params.with_kwargs(truncation=truncation).truncate_prompt_tokens is None


@pytest.mark.parametrize("padding", [False, "do_not_pad"])
def test_disabling_padding_clears_pad_prompt_tokens(params, padding):
    assert params.with_kwargs(padding=padding).pad_prompt_tokens is None


def test_enabling_truncation_still_uses_the_input_budget(params):
    child = params.with_kwargs(truncation=True)

    assert child.truncate_prompt_tokens == params.max_input_tokens


def test_padding_to_max_length_still_uses_the_input_budget(params):
    child = params.with_kwargs(padding="max_length")

    assert child.pad_prompt_tokens == params.max_input_tokens


def test_omitting_the_overrides_preserves_the_parent_values(params):
    child = params.with_kwargs(add_special_tokens=False)

    assert child.truncate_prompt_tokens == 128
    assert child.pad_prompt_tokens == 64
