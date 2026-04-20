# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import openai
import pytest
from pydantic import BaseModel


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
