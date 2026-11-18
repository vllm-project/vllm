# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.pooling.types import input_to_seq, is_conversations


def _msg(role: str, content: str = "hi") -> dict:
    """Build a plain OpenAI-style chat message dict."""
    return {"role": role, "content": content}


def _harmony_message(role_name: str, text: str):
    """Build an `openai_harmony.Message`; skip the test if harmony is absent."""
    harmony = pytest.importorskip("openai_harmony")
    role = getattr(harmony.Role, role_name.upper())
    return harmony.Message.from_role_and_content(role, text)


class TestIsConversations:
    @pytest.mark.parametrize(
        "value",
        [
            "",
            "hello",
            [1, 2, 3],  # token ids
            [],
            {"prompt": "hi"},  # TextPrompt
            {"prompt_token_ids": [1, 2, 3]},  # TokensPrompt
            {
                "encoder_prompt": "a",
                "decoder_prompt": "b",
            },  # ExplicitEncoderDecoderPrompt
            {"prompt": "hi", "cache_salt": "x"},  # prompt with options
            [{"prompt": "a"}, {"prompt": "b"}],  # batch of TextPrompts
            [_msg("user"), {"prompt": "x"}],  # mixed message + prompt
        ],
        ids=[
            "empty-string",
            "plain-string",
            "token-ids",
            "empty-list",
            "text-prompt",
            "tokens-prompt",
            "encoder-decoder-prompt",
            "prompt-with-options",
            "batch-of-text-prompts",
            "mixed-message-and-prompt",
        ],
    )
    def test_returns_false_for_non_conversation(self, value):
        # Note: `is_conversations([])` returns False because the helper
        # short-circuits on `len(...) == 0`, not because `[]` is treated as
        # an empty token list. `input_to_seq([])` separately returns `[]`
        # (empty batch); see `TestInputToSeq::test_empty_list`.
        assert is_conversations(value) is False

    @pytest.mark.parametrize(
        "value",
        [
            [_msg("user")],
            [_msg("user"), _msg("assistant")],
            [_msg("system"), _msg("user"), _msg("assistant")],
            [{"role": "user"}],  # only role is required
            [{"role": "developer", "tools": []}],
        ],
        ids=[
            "single-message",
            "user-assistant",
            "system-user-assistant",
            "role-only",
            "developer-with-tools",
        ],
    )
    def test_returns_true_for_dict_conversation(self, value):
        assert is_conversations(value) is True

    def test_returns_true_for_harmony_conversation(self):
        msgs = [
            _harmony_message("user", "hi"),
            _harmony_message("assistant", "yo"),
        ]
        assert is_conversations(msgs) is True

    def test_returns_true_for_mixed_dict_and_harmony(self):
        msgs = [
            _harmony_message("user", "hi"),
            _msg("assistant", "yo"),
        ]
        assert is_conversations(msgs) is True

    def test_batch_of_conversations_is_false(self):
        batch = [[_msg("user")], [_msg("user")]]
        assert is_conversations(batch) is False

    def test_batch_of_harmony_conversations_is_false(self):
        batch = [
            [_harmony_message("user", "hi")],
            [_harmony_message("user", "yo")],
        ]
        assert is_conversations(batch) is False


class TestInputToSeq:
    def test_single_string(self):
        assert input_to_seq("hello") == ["hello"]

    def test_single_prompt_dict(self):
        prompt = {"prompt": "hi"}
        assert input_to_seq(prompt) == [prompt]

    def test_single_tokens_prompt_dict(self):
        prompt = {"prompt_token_ids": [1, 2, 3]}
        assert input_to_seq(prompt) == [prompt]

    def test_single_token_ids(self):
        assert input_to_seq([1, 2, 3]) == [[1, 2, 3]]

    def test_single_token_ids_mixed_types_is_not_tokens(self):
        # A list of ints and non-ints falls through to the "batch" branch;
        # in practice the caller must pass homogeneous inputs, so this only
        # documents the current behaviour.
        assert input_to_seq([1, "a"]) == [1, "a"]

    def test_empty_list(self):
        assert input_to_seq([]) == []

    def test_single_conversation_dicts(self):
        conv = [_msg("user"), _msg("assistant")]
        assert input_to_seq(conv) == [conv]

    def test_single_conversation_harmony(self):
        conv = [
            _harmony_message("user", "hi"),
            _harmony_message("assistant", "yo"),
        ]
        assert input_to_seq(conv) == [conv]

    def test_batch_of_strings(self):
        assert input_to_seq(["a", "b"]) == ["a", "b"]

    def test_batch_of_prompt_dicts(self):
        batch = [{"prompt": "a"}, {"prompt": "b"}]
        assert input_to_seq(batch) == batch

    def test_batch_of_token_ids(self):
        batch = [[1, 2], [3, 4]]
        assert input_to_seq(batch) == batch

    def test_batch_of_conversations(self):
        batch = [
            [_msg("user")],
            [_msg("user")],
        ]
        assert input_to_seq(batch) == batch

    def test_batch_of_harmony_conversations(self):
        batch = [
            [_harmony_message("user", "hi")],
            [_harmony_message("user", "yo")],
        ]
        assert input_to_seq(batch) == batch


class TestConsistency:
    @pytest.mark.parametrize(
        "value",
        [
            "hello",
            [1, 2, 3],
            {"prompt": "hi"},
            [_msg("user"), _msg("assistant")],
        ],
        ids=["string", "token-ids", "prompt-dict", "conversation"],
    )
    def test_single_input_is_wrapped(self, value):
        # For any single input, `input_to_seq` must return exactly one element.
        result = input_to_seq(value)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == value

    def test_conversation_roundtrip(self):
        conv = [_msg("user"), _msg("assistant")]
        assert is_conversations(conv)
        seq = input_to_seq(conv)
        assert len(seq) == 1
        assert seq[0] == conv

    def test_batch_roundtrip(self):
        batch = [{"prompt": "a"}, {"prompt": "b"}]
        assert not is_conversations(batch)
        assert input_to_seq(batch) == batch

    def test_mixed_batch_roundtrip(self):
        # A batch that mixes a completion prompt and a chat conversation.
        # `input_to_seq` must preserve the batch shape; each element is then
        # dispatched independently by the offline `request_factory`
        # (`is_conversations(inp)` per element → EncodeChatRenderParams vs
        # EncodeCMPLRenderParams). If `is_conversations` ever grew a
        # batch-wide short-circuit, this test would catch it.
        batch = ["a plain prompt", [_msg("user"), _msg("assistant")]]
        assert not is_conversations(batch)
        assert input_to_seq(batch) == batch
