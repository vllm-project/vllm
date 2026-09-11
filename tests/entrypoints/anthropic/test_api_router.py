from vllm.entrypoints.anthropic.api_router import serialize_messages_response
from vllm.entrypoints.anthropic.protocol import AnthropicMessagesResponse


def test_serialize_messages_response_keeps_nullable_stop_fields():
    response = AnthropicMessagesResponse(
        id="msg_test",
        content=[],
        model="test-model",
        stop_reason="end_turn",
    )

    payload = serialize_messages_response(response)

    assert payload["stop_reason"] == "end_turn"
    assert payload["stop_sequence"] is None


def test_serialize_messages_response_keeps_null_stop_reason():
    response = AnthropicMessagesResponse(
        id="msg_test",
        content=[],
        model="test-model",
    )

    payload = serialize_messages_response(response)

    assert payload["stop_reason"] is None
    assert payload["stop_sequence"] is None
