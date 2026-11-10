# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import ModelConfig, VllmConfig
from vllm.renderers.deepseek_v4 import DeepseekV4Renderer
from vllm.renderers.params import TokenizeParams
from vllm.tokenizers import get_tokenizer
from vllm.tokenizers.deepseek_v4 import get_deepseek_v4_tokenizer


def test_deepseek_v4_recipe_backend_gating(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The DeepSeek-V4 renderer routes plain chat prompts through the recipe
    helper and keeps specials and the builtin backend on the default path
    (same gating conditions as HfRenderer). The helper is spied on so this
    test runs without the optional `deepseek_recipe` package."""
    model = "facebook/opt-125m"
    recipe_renderer = DeepseekV4Renderer(
        VllmConfig(model_config=ModelConfig(model, tokenize_backend="recipe")),
        get_deepseek_v4_tokenizer(get_tokenizer(model)),
    )
    builtin_renderer = DeepseekV4Renderer(
        VllmConfig(model_config=ModelConfig(model, tokenize_backend="builtin")),
        get_deepseek_v4_tokenizer(get_tokenizer(model)),
    )
    prompt = {"prompt": "The transformer architecture processes sequences."}
    recipe_calls = []

    def fake_recipe(prompt_str, tokenizer):
        recipe_calls.append(prompt_str)
        return [1, 2, 3]

    monkeypatch.setattr("vllm.renderers.deepseek_v4._encode_prompt_with_recipe",
                        fake_recipe)

    # recipe + default chat path -> recipe helper, ids built from its output.
    tokens = recipe_renderer._tokenize_prompt(
        prompt, TokenizeParams(max_total_tokens=None, add_special_tokens=False))
    assert recipe_calls == [prompt["prompt"]]
    assert tokens["prompt_token_ids"] == [1, 2, 3]

    # recipe + add_special_tokens=True -> builtin path, helper untouched.
    recipe_calls.clear()
    recipe_renderer._tokenize_prompt(prompt,
                                     TokenizeParams(max_total_tokens=None, add_special_tokens=True))
    assert recipe_calls == []

    # builtin backend -> builtin path regardless of add_special_tokens.
    builtin_renderer._tokenize_prompt(prompt,
                                      TokenizeParams(max_total_tokens=None, add_special_tokens=False))
    assert recipe_calls == []

    # recipe helper returning None (slow-only fallback) -> builtin path.
    def none_recipe(prompt_str, tokenizer):
        return None

    monkeypatch.setattr("vllm.renderers.deepseek_v4._encode_prompt_with_recipe",
                        none_recipe)
    recipe_ids = recipe_renderer._tokenize_prompt(
        prompt, TokenizeParams(max_total_tokens=None, add_special_tokens=False))["prompt_token_ids"]
    builtin_ids = builtin_renderer._tokenize_prompt(
        prompt, TokenizeParams(max_total_tokens=None, add_special_tokens=False))["prompt_token_ids"]
    assert recipe_ids == builtin_ids
