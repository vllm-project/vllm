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
    recipe_renderer._tokenize_prompt(
        prompt,
        TokenizeParams(max_total_tokens=None, add_special_tokens=True),
    )
    assert recipe_calls == []

    # builtin backend -> builtin path regardless of add_special_tokens.
    builtin_renderer._tokenize_prompt(
        prompt,
        TokenizeParams(max_total_tokens=None, add_special_tokens=False),
    )
    assert recipe_calls == []

    # recipe helper returning None (slow-only fallback) -> builtin path.
    def none_recipe(prompt_str, tokenizer):
        return None

    monkeypatch.setattr("vllm.renderers.deepseek_v4._encode_prompt_with_recipe",
                        none_recipe)
    recipe_ids = recipe_renderer._tokenize_prompt(
        prompt,
        TokenizeParams(max_total_tokens=None, add_special_tokens=False),
    )["prompt_token_ids"]
    builtin_ids = builtin_renderer._tokenize_prompt(
        prompt,
        TokenizeParams(max_total_tokens=None, add_special_tokens=False),
    )["prompt_token_ids"]
    assert recipe_ids == builtin_ids


def test_deepseek_v4_recipe_official_ranges(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A prompt rendered by the official deepseek-recipe encoder carries its
    special ranges; tokenization then interleaves them with the plain regions
    without re-scanning the string."""
    from vllm.renderers.deepseek_recipe_encoder import DeepSeekRecipeEncoder

    model = "facebook/opt-125m"
    renderer = DeepseekV4Renderer(
        VllmConfig(model_config=ModelConfig(model, tokenize_backend="recipe")),
        get_deepseek_v4_tokenizer(get_tokenizer(model)),
    )
    prompt = {
        "prompt": "<|special|>hello",
        "_recipe_special_ranges": [(0, 10)],
    }
    seen = []

    class FakeEncoder:
        def encode(self, text, ranges):
            seen.append((text, list(ranges)))
            return [1, 2, 3]

    monkeypatch.setattr(
        DeepSeekRecipeEncoder, "get",
        classmethod(lambda cls, tokenizer, spec: FakeEncoder()),
    )
    tokens = renderer._tokenize_prompt(
        prompt, TokenizeParams(max_total_tokens=None, add_special_tokens=False))
    assert seen == [(prompt["prompt"], [(0, 10)])]
    assert tokens["prompt_token_ids"] == [1, 2, 3]


def test_deepseek_v4_recipe_tool_conversation(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool calls and tool definitions travel to the reference encoder."""
    from vllm.renderers import deepseek_v4 as dsv4

    model = "facebook/opt-125m"
    renderer = DeepseekV4Renderer(
        VllmConfig(model_config=ModelConfig(model, tokenize_backend="recipe")),
        get_deepseek_v4_tokenizer(get_tokenizer(model)),
    )
    calls = []

    class FakeEncoder:
        def render(self, messages, *, thinking, tools=None):
            calls.append((messages, thinking, tools))
            return "<|special|>ok", [(0, 10)]

    monkeypatch.setattr(
        dsv4.DeepSeekRecipeEncoder, "get",
        classmethod(lambda cls, backend, spec: FakeEncoder()),
    )
    tools = [{"type": "function", "function": {"name": "f"}}]
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "{}"},
    ]
    rendered = renderer._recipe_render(
        msgs, [], ChatParams(chat_template_kwargs={"tools": tools}))
    assert rendered == ("<|special|>ok", [(0, 10)])
    assert calls and calls[0][1] is False and calls[0][2] == tools


def test_deepseek_v4_recipe_skips_multimodal(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Multimodal messages stay on vLLM's own renderer."""
    model = "facebook/opt-125m"
    renderer = DeepseekV4Renderer(
        VllmConfig(model_config=ModelConfig(model, tokenize_backend="recipe")),
        get_deepseek_v4_tokenizer(get_tokenizer(model)),
    )
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
    ]}]
    assert renderer._recipe_render(msgs, [], ChatParams()) is None

