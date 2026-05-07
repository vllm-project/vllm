# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.inputs.preprocess import InputPreprocessor

pytestmark = pytest.mark.skip_global_cleanup


class FakeTokenizationParams:
    def with_kwargs(self, **kwargs):
        return kwargs


class FakeRenderer:
    default_cmpl_tok_params = FakeTokenizationParams()

    def _tokenize_singleton_prompt(self, prompt, tokenization_params):
        truncate_prompt_tokens = tokenization_params.get("truncate_prompt_tokens")
        prompt_token_ids = prompt["prompt_token_ids"]
        if truncate_prompt_tokens is not None:
            prompt_token_ids = prompt_token_ids[-truncate_prompt_tokens:]
        return {"prompt_token_ids": prompt_token_ids}


def test_prompt_token_ids_respect_tokenization_kwargs():
    preprocessor = object.__new__(InputPreprocessor)
    preprocessor.renderer = FakeRenderer()

    inputs = preprocessor._prompt_to_llm_inputs(
        {"prompt_token_ids": [1, 2, 3, 4]},
        tokenization_kwargs={"truncate_prompt_tokens": 2},
    )

    assert inputs["prompt_token_ids"] == [3, 4]
