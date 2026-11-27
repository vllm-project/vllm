# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import runpy
from functools import partial
from pathlib import Path

import httpx
import openai
import pytest

EXAMPLE = (
    Path(__file__).parents[2]
    / "examples/tool_calling/openai_responses_client_with_tools.py"
)


@pytest.mark.parametrize("locations", [[(48, 2)], [(48, 2), (35, 139)]])
def test_each_weather_call_has_a_matching_result(monkeypatch, locations):
    calls = [
        {
            "type": "function_call",
            "name": "get_weather",
            "call_id": f"call_{i}",
            "arguments": json.dumps({"latitude": lat, "longitude": lon}),
        }
        for i, (lat, lon) in enumerate(locations)
    ]
    model_output = list(calls)
    if len(calls) > 1:
        model_output.insert(
            0,
            {
                "id": "rs_weather",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Compare both locations."}],
            },
        )
    followup = []

    def respond(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200, json={"object": "list", "data": [{"id": "test"}]}
            )
        body = json.loads(request.content)
        if "tool_choice" in body:
            return httpx.Response(
                200, json={"object": "response", "output": model_output}
            )
        followup.extend(body["input"])
        return httpx.Response(200, json={"object": "response", "output": []})

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        monkeypatch.setattr(
            openai, "OpenAI", partial(openai.OpenAI, http_client=http_client)
        )
        runpy.run_path(str(EXAMPLE), run_name="__main__")

    assert followup[0]["role"] == "user"
    expected_outputs = [
        {
            "type": "function_call_output",
            "call_id": call["call_id"],
            "output": f"Current temperature at ({lat}, {lon}) is 20°C.",
        }
        for call, (lat, lon) in zip(calls, locations)
    ]
    assert followup[1:] == [*model_output, *expected_outputs]
