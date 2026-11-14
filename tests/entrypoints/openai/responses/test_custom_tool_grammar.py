# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grammar custom tools through the Qwen Responses serving path."""

import pytest

from tests.utils import RemoteOpenAIServer

MODEL = "Qwen/Qwen3.5-0.8B"


@pytest.fixture(scope="module")
def grammar_server():
    with RemoteOpenAIServer(
        MODEL,
        [
            "--tool-call-parser",
            "qwen3_coder",
            "--reasoning-parser",
            "qwen3",
            "--enable-auto-tool-choice",
            "--language-model-only",
            "--max-model-len",
            "4096",
            "--max-num-seqs",
            "1",
            "--enforce-eager",
            "--generation-config",
            "vllm",
            "--structured-outputs-config",
            '{"backend":"xgrammar"}',
        ],
    ) as server:
        yield server


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "syntax,definition,payload",
    [
        ("lark", 'start: "ok"', "ok"),
        ("regex", "ok", "ok"),
        ("lark", 'start: "\\n</parameter>\\n"', "\n</parameter>\n"),
    ],
)
def test_custom_tool_grammar_serving(
    grammar_server, stream, syntax, definition, payload
):
    with grammar_server.get_client() as client:
        tool = {
            "type": "custom",
            "name": "emit",
            "description": "Send the requested text as input.",
            "format": {"type": "grammar", "syntax": syntax, "definition": definition},
        }
        # Deliberately contradict the grammar: a description-only fallback fails.
        prompt = "Call emit with exactly wrong as its input."
        kwargs = dict(
            model=MODEL,
            input=prompt,
            tools=[tool],
            tool_choice={"type": "custom", "name": "emit"},
            temperature=0,
            max_output_tokens=256,
            store=False,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if stream:
            events = list(client.responses.create(**kwargs, stream=True))
            response = next(
                e.response for e in events if e.type == "response.completed"
            )
        else:
            response = client.responses.create(**kwargs)
        assert response.status == "completed"
        calls = [item for item in response.output if item.type == "custom_tool_call"]
        assert len(calls) == 1
        call = calls[0]
        assert (call.name, call.input) == ("emit", payload)

        if stream:
            deltas = [
                e.delta
                for e in events
                if e.type == "response.custom_tool_call_input.delta"
            ]
            assert "".join(deltas) == payload
            done = next(
                e for e in events if e.type == "response.custom_tool_call_input.done"
            )
            assert (done.item_id, done.input) == (call.id, payload)
            items = [
                e.item
                for e in events
                if e.type in {"response.output_item.added", "response.output_item.done"}
                and e.item.type == "custom_tool_call"
            ]
            assert len(items) == 2
            assert all(
                (item.id, item.call_id) == (call.id, call.call_id) for item in items
            )
        else:
            replay = client.responses.create(
                model=MODEL,
                tools=[tool],
                tool_choice="none",
                temperature=0,
                max_output_tokens=128,
                store=False,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                input=[
                    {"role": "user", "content": prompt},
                    *[item.model_dump() for item in response.output],
                    {
                        "type": "custom_tool_call_output",
                        "call_id": call.call_id,
                        "output": "GRAMMAR_REPLAY_OK",
                    },
                    {"role": "user", "content": "Repeat the tool output verbatim."},
                ],
            )
            assert replay.status == "completed"
            assert "GRAMMAR_REPLAY_OK" in replay.output_text
