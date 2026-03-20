# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for structured output helpers in the Responses API."""

import json

import openai
import pytest
from pydantic import BaseModel

from vllm.entrypoints.openai.responses.serving import (
    _constraint_to_content_format,
)
from vllm.sampling_params import StructuredOutputsParams


@pytest.mark.asyncio
async def test_structured_output(client: openai.AsyncOpenAI):
    response = await client.responses.create(
        input=[
            {"role": "system", "content": "Extract the event information."},
            {
                "role": "user",
                "content": "Alice and Bob are going to a science fair on Friday.",
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "calendar_event",
                "schema": {
                    "type": "object",
                    "properties": {
                        "event_name": {"type": "string"},
                        "date": {"type": "string"},
                        "participants": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["event_name", "date", "participants"],
                    "additionalProperties": False,
                },
                "description": "A calendar event.",
                "strict": True,
            }
        },
    )
    print(response)

    # NOTE: The JSON schema is applied to the output text, not reasoning.
    output_text = response.output[-1].content[0].text
    event = json.loads(output_text)

    assert event["event_name"].lower() == "science fair"
    assert event["date"] == "Friday"
    participants = event["participants"]
    assert len(participants) == 2
    assert participants[0] == "Alice"
    assert participants[1] == "Bob"


@pytest.mark.asyncio
async def test_structured_output_with_parse(client: openai.AsyncOpenAI):
    class CalendarEvent(BaseModel):
        event_name: str
        date: str
        participants: list[str]

    response = await client.responses.parse(
        model=None,
        instructions="Extract the event information.",
        input="Alice and Bob are going to a science fair on Friday.",
        text_format=CalendarEvent,
    )
    print(response)

    # The output is successfully parsed.
    event = response.output_parsed
    assert event is not None

    # The output is correct.
    assert event.event_name.lower() == "science fair"
    assert event.date == "Friday"
    participants = event.participants
    assert len(participants) == 2
    assert participants[0] == "Alice"
    assert participants[1] == "Bob"


class TestConstraintToContentFormat:
    """Test _constraint_to_content_format helper."""

    def test_json_schema_string_is_parsed(self):
        """JSON schema passed as a string gets json.loads'd into a dict."""
        schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
        params = StructuredOutputsParams(json=json.dumps(schema))
        result = _constraint_to_content_format(params)

        assert result == {"type": "json_schema", "json_schema": schema}

    def test_json_schema_dict(self):
        """JSON schema passed as a dict is used directly."""
        schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
        params = StructuredOutputsParams(json=schema)
        result = _constraint_to_content_format(params)

        assert result == {"type": "json_schema", "json_schema": schema}

    def test_json_object(self):
        """json_object maps to minimal JSON schema."""
        params = StructuredOutputsParams(json_object=True)
        result = _constraint_to_content_format(params)

        assert result == {
            "type": "json_schema",
            "json_schema": {"type": "object"},
        }

    def test_regex(self):
        """Regex constraint is converted correctly."""
        params = StructuredOutputsParams(regex=r"\d+")
        result = _constraint_to_content_format(params)

        assert result == {"type": "regex", "pattern": r"\d+"}

    def test_grammar(self):
        """Grammar constraint is converted correctly."""
        params = StructuredOutputsParams(grammar="root ::= 'hello'")
        result = _constraint_to_content_format(params)

        assert result == {"type": "grammar", "grammar": "root ::= 'hello'"}

    def test_choice(self):
        """Choice constraint is converted correctly."""
        params = StructuredOutputsParams(choice=["yes", "no"])
        result = _constraint_to_content_format(params)

        assert result == {
            "type": "or",
            "elements": [
                {"type": "const_string", "value": "yes"},
                {"type": "const_string", "value": "no"},
            ],
        }

    def test_structural_tag_only_returns_none(self):
        """structural_tag is not a content constraint -- should return None."""
        params = StructuredOutputsParams(structural_tag='{"type": "structural_tag"}')
        result = _constraint_to_content_format(params)

        assert result is None
