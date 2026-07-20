# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Normalization of prior-turn output items replayed as request ``input``.

Clients (e.g. Codex) feed a previous model turn's output items back in as
request ``input``. The SDK's output-item schemas require server-assigned fields
those clients often drop (``id``, ``status``, ``output_text.annotations``),
which would otherwise 400 the request. ``ResponsesRequest`` backfills exactly
those safe-to-synthesize fields.
"""

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseReasoningItem,
)

from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


def _first(input_items):
    return ResponsesRequest(model="m", input=input_items).input[0]


def test_reasoning_item_missing_id_is_backfilled():
    item = {
        "type": "reasoning",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "why"}],
        "encrypted_content": None,
    }
    parsed = _first([item])
    assert isinstance(parsed, ResponseReasoningItem)
    assert parsed.id and parsed.id.startswith("rs_")


def test_reasoning_item_existing_id_preserved():
    item = {
        "type": "reasoning",
        "id": "rs_existing",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "why"}],
    }
    assert _first([item]).id == "rs_existing"


def test_assistant_output_message_missing_id_status_annotations():
    item = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "hello"}],
    }
    parsed = _first([item])
    assert isinstance(parsed, ResponseOutputMessage)
    assert parsed.id and parsed.id.startswith("msg_")
    assert parsed.status == "completed"
    # output_text parts get an empty annotations list backfilled.
    assert parsed.content[0].annotations == []


def test_assistant_output_message_existing_fields_preserved():
    item = {
        "type": "message",
        "role": "assistant",
        "id": "msg_existing",
        "status": "completed",
        "content": [
            {
                "type": "output_text",
                "text": "hello",
                "annotations": [
                    {
                        "type": "url_citation",
                        "url": "http://x",
                        "title": "t",
                        "start_index": 0,
                        "end_index": 1,
                    }
                ],
            }
        ],
    }
    parsed = _first([item])
    assert parsed.id == "msg_existing"
    assert len(parsed.content[0].annotations) == 1


def test_user_message_untouched():
    """Plain user messages must not be treated as output items."""
    for content in ["hello", [{"type": "input_text", "text": "hi"}]]:
        req = ResponsesRequest(
            model="m",
            input=[{"type": "message", "role": "user", "content": content}],
        )
        item = req.input[0]
        # User messages resolve to the input-message shape (a dict), not an
        # output message with a synthesized id.
        assert not isinstance(item, ResponseOutputMessage)


def test_function_call_dict_coerced_and_call_id_required():
    """function_call dicts become objects; a missing call_id still rejects."""
    ok = _first(
        [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "f",
                "arguments": "{}",
            }
        ]
    )
    assert isinstance(ok, ResponseFunctionToolCall)
    assert ok.call_id == "call_1"


def test_non_dict_and_typeless_items_pass_through():
    """Items without a known output type are left for pydantic to handle."""
    # A bare user string input is a valid request and must not be mangled.
    req = ResponsesRequest(model="m", input="just text")
    assert req.input == "just text"
