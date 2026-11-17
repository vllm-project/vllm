# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.renderers.hf import _encode_prompt_with_recipe


def _tiny_backend_tokenizer():
    """A byte-level tokenizer with the reference special-token spellings.

    Only ``tokenizers`` is required, so the segmented encoder can be tested
    without a model download.
    """
    from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.BPE(vocab={"<unk>": 0}, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.add_special_tokens(
        [
            AddedToken("<\uff5cbegin\u2581of\u2581sentence\uff5c>", special=True),
            AddedToken("<\uff5cSystem\uff5c>", special=True),
            AddedToken("<\uff5cUser\uff5c>", special=True),
            AddedToken("<\uff5cAssistant\uff5c>", special=True),
        ]
    )
    return tokenizer


class _Wrapper:
    """Mimics the served fast tokenizer's backend access path."""

    def __init__(self, backend):
        self.tokenizer = type("Inner", (), {"backend_tokenizer": backend})()


def test_segmented_encoding_matches_builtin() -> None:
    """The segmented encoder must reproduce the builtin token ids exactly."""
    tokenizer = _tiny_backend_tokenizer()
    text = (
        "<\uff5cbegin\u2581of\u2581sentence\uff5c>hello world"
        "<\uff5cUser\uff5c>\nhi there<\uff5cAssistant\uff5c>"
    )
    expected = tokenizer.encode(text, add_special_tokens=False).ids
    assert _encode_prompt_with_recipe(text, _Wrapper(tokenizer)) == expected


def test_recipe_encoder_matches_reference() -> None:
    """``render`` + segmented ``encode`` equals the reference encoder."""
    dr = pytest.importorskip("deepseek_recipe")
    from vllm.renderers.deepseek_recipe_encoder import DeepSeekRecipeEncoder

    backend = _tiny_backend_tokenizer()
    encoder = DeepSeekRecipeEncoder(backend, "dsv41")
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city": "B"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        {"role": "user", "content": "thanks"},
    ]
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get weather",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}}}}}]

    prompt, ranges = encoder.render(messages, thinking=False, tools=tools)
    # The rendered prompt introduces more reference spellings than the tiny
    # tokenizer knows; register them so ``encode`` can map them to ids.
    for start, end in ranges:
        spelling = prompt[start:end]
        if spelling not in encoder._added_ids:
            backend.add_tokens([spelling])
            encoder._added_ids[spelling] = backend.token_to_id(spelling)
    ours = encoder.encode(prompt, ranges)
    assert ours is not None

    dr_encoder = dr.DeepseekV41Encoding().with_tokenizer(
        dr.Tokenizer.from_str(backend.to_str()))
    conversation = dr.Conversation(
        messages=[
            dr.SystemMessage("you are helpful"),
            dr.UserMessage("hello"),
            dr.AssistantMessage("", tool_calls=[
                dr.ToolCall("call_1", "get_weather", '{"city": "B"}')]),
            dr.ToolMessage("{}", tool_call_id="call_1"),
            dr.UserMessage("thanks"),
        ],
        thinking_mode=False,
        tools=[dr.ToolDefinition("get_weather", "Get weather",
                                 {"type": "object",
                                  "properties": {"city": {"type": "string"}}})],
    )
    assert ours == dr_encoder.encode(conversation)
    assert ranges == [
        tuple(r) for r in dr_encoder.render_conversation(conversation).special_ranges
    ]
