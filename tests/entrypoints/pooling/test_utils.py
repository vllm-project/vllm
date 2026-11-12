# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import numpy as np
import pybase64
import pytest
import torch

from vllm.entrypoints.pooling.embed.protocol import CohereEmbedRequest
from vllm.entrypoints.pooling.embed.serving import ServingEmbedding
from vllm.entrypoints.pooling.pooling.protocol import IOProcessorResponse
from vllm.entrypoints.pooling.pooling.serving import ServingPooling
from vllm.entrypoints.pooling.types import input_to_seq, is_conversations
from vllm.entrypoints.pooling.utils import encode_pooling_output_float_or_ndarray
from vllm.entrypoints.serve.utils.api_utils import create_json_response
from vllm.outputs import PoolingOutput, PoolingRequestOutput

pytestmark = pytest.mark.skip_global_cleanup


def _pooling_output(data):
    return SimpleNamespace(outputs=SimpleNamespace(data=data))


def test_encode_pooling_output_float_or_ndarray_returns_numpy_array():
    output = _pooling_output(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))

    encoded = encode_pooling_output_float_or_ndarray(output)

    assert isinstance(encoded, np.ndarray)
    np.testing.assert_allclose(encoded, [1.0, 2.0, 3.0])


def test_json_response_serializes_numpy_array():
    output = _pooling_output(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
    encoded = encode_pooling_output_float_or_ndarray(output)
    response = create_json_response({"embedding": encoded})
    assert json.loads(response.body)["embedding"] == pytest.approx([1.0, 2.0, 3.0])


def test_encode_pooling_output_float_or_ndarray_falls_back_to_list():
    class DataWithUnsupportedNumpy:
        def is_contiguous(self):
            return True

        def numpy(self):
            raise TypeError("unsupported dtype")

        def tolist(self):
            return [1.0, 2.0, 3.0]

    output = _pooling_output(DataWithUnsupportedNumpy())

    assert encode_pooling_output_float_or_ndarray(output) == [1.0, 2.0, 3.0]


@pytest.fixture
def pooling_outputs():
    return [
        PoolingRequestOutput(
            request_id="req",
            outputs=PoolingOutput(data=torch.tensor([1.25, -2.5, 0.0, 3.0])),
            prompt_token_ids=[1, 2],
            num_cached_tokens=0,
            finished=True,
        )
    ]


@pytest.mark.parametrize("encoding_format", ["float", "base64"])
@pytest.mark.parametrize("api", ["embedding", "pooling"])
def test_pooling_json_response_preserves_encoding(
    pooling_outputs, api, encoding_format
):
    if api == "embedding":
        build_response = object.__new__(ServingEmbedding)._openai_json_response
        data_key = "embedding"
    else:
        build_response = object.__new__(
            ServingPooling
        ).request_output_to_pooling_json_response
        data_key = "data"
    response = build_response(
        pooling_outputs, "req", 123, "model", encoding_format, "float32", "little"
    )
    body = json.loads(response.body)
    assert response.status_code == 200
    assert response.media_type == "application/json"
    assert body["id"] == "req"
    assert body["created"] == 123
    assert body["model"] == "model"
    assert body["usage"]["prompt_tokens"] == body["usage"]["total_tokens"] == 2
    encoded = body["data"][0][data_key]
    if encoding_format == "base64":
        encoded = np.frombuffer(pybase64.b64decode(encoded), dtype="<f4")
    np.testing.assert_array_equal(encoded, [1.25, -2.5, 0.0, 3.0])


def test_cohere_json_response_preserves_omitted_fields(pooling_outputs):
    serving = object.__new__(ServingEmbedding)
    ctx = SimpleNamespace(
        request=CohereEmbedRequest(texts=["hello"]),
        final_res_batch=pooling_outputs,
        request_id="req",
    )
    body = json.loads(serving._build_cohere_response_from_ctx(ctx).body)
    assert body["embeddings"] == {"float": [[1.25, -2.5, 0.0, 3.0]]}
    assert body["texts"] == ["hello"]
    assert body["meta"]["billed_units"]["input_tokens"] == 2


def test_pooling_plugin_json_response_preserves_payload():
    serving = object.__new__(ServingPooling)
    ctx = SimpleNamespace(
        response=IOProcessorResponse(
            request_id="req", created_at=123, data={"text": "中文", "scores": [1, 2]}
        )
    )
    body = json.loads(serving._build_response(ctx).body)
    assert body == ctx.response.model_dump()


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
            [],  # empty list is treated as an empty token list
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


# ---------------------------------------------------------------------------
# Cross-checks between the two helpers
# ---------------------------------------------------------------------------


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
