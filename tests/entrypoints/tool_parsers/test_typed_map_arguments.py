# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import regex as re
from pydantic import BaseModel

from tests.utils import RemoteOpenAIServer

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def test_gemma4_typed_map_argument_reaches_client():
    class Inventory(BaseModel):
        counts: dict[str, int] | None

    args = [
        "--enforce-eager",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "gemma4",
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "1",
    ]
    # Pin SmolLM2 to a native Gemma call so the test isolates argument parsing.
    output = "<|tool_call>call:inventory{counts:{bolts:7}}<tool_call|>"
    tool = {"name": "inventory", "parameters": Inventory.model_json_schema()}
    with RemoteOpenAIServer(MODEL, args) as server, server.get_client() as client:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Record inventory."}],
            tools=[{"type": "function", "function": tool}],
            tool_choice="auto",
            temperature=0,
            max_tokens=64,
            extra_body={"structured_outputs": {"regex": re.escape(output)}},
        )
    (call,) = response.choices[0].message.tool_calls
    assert json.loads(call.function.arguments) == {"counts": {"bolts": 7}}
