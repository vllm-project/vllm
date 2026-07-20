# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Round-trip test: a model that emits the flattened ``namespace__name`` tool
call is reconstructed into a ResponseFunctionToolCall carrying the split
``name`` + ``namespace``.

Uses the real qwen3_coder tool parser + qwen3 reasoning parser against a small
cached tokenizer; no server / GPU required.
"""

from types import SimpleNamespace

import pytest
from openai.types.responses import NamespaceTool

from vllm.entrypoints.openai.parser.responses_parser import ResponsesParser
from vllm.parser import ParserManager

MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    except Exception as exc:  # pragma: no cover - depends on local cache
        pytest.skip(f"tokenizer {MODEL} unavailable: {exc}")


@pytest.fixture(scope="module")
def parser_manager():
    return ParserManager.get_parser(
        tool_parser_name="qwen3_coder",
        reasoning_parser_name="qwen3",
        enable_auto_tools=True,
        model_name="q",
    )


def _namespace_tool():
    return NamespaceTool(
        type="namespace",
        name="multi_agent_v1",
        description="agents",
        tools=[
            {
                "type": "function",
                "name": "spawn_agent",
                "description": "spawn",
                "parameters": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                },
            }
        ],
    )


def _run_parser(tokenizer, parser_manager, raw, tools):
    request = SimpleNamespace(request_id="r", tools=tools, tool_choice="auto")
    parser = ResponsesParser(
        tokenizer=tokenizer,
        reasoning_parser_cls=parser_manager.reasoning_parser_cls,
        response_messages=[],
        request=request,
        tool_parser_cls=parser_manager.tool_parser_cls,
    )
    parser.process(
        SimpleNamespace(
            text=raw,
            token_ids=tokenizer.encode(raw, add_special_tokens=False),
            finish_reason="stop",
            logprobs=None,
        )
    )
    items = parser.make_response_output_items_from_parsable_context()
    return [it for it in items if getattr(it, "type", None) == "function_call"]


def test_namespace_tool_call_split_on_output(tokenizer, parser_manager):
    """Model emits multi_agent_v1__spawn_agent -> name/namespace are split."""
    raw = (
        "<think>\nspawn\n</think>\n\n"
        "<tool_call>\n<function=multi_agent_v1__spawn_agent>\n"
        "<parameter=message>\ngo\n</parameter>\n</function>\n</tool_call>"
    )
    calls = _run_parser(tokenizer, parser_manager, raw, [_namespace_tool()])

    assert len(calls) == 1
    assert calls[0].name == "spawn_agent"
    assert calls[0].namespace == "multi_agent_v1"


def test_plain_tool_call_has_no_namespace(tokenizer, parser_manager):
    """A non-namespaced tool call keeps namespace=None."""
    from openai.types.responses import FunctionTool

    ft = FunctionTool(
        type="function",
        name="shell_command",
        description="s",
        strict=False,
        parameters={"type": "object", "properties": {"command": {"type": "string"}}},
    )
    raw = (
        "<think>\nx\n</think>\n\n"
        "<tool_call>\n<function=shell_command>\n"
        "<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"
    )
    calls = _run_parser(tokenizer, parser_manager, raw, [ft])

    assert len(calls) == 1
    assert calls[0].name == "shell_command"
    assert calls[0].namespace is None


def _run_streaming(tokenizer, parser_manager, raw, tools):
    """Drive the streaming event processor with `raw` fed token-by-token."""
    import asyncio

    from vllm.entrypoints.openai.responses.context import SimpleContext
    from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses

    srv = OpenAIServingResponses.__new__(OpenAIServingResponses)
    srv.parser = parser_manager
    srv.enable_auto_tools = True
    srv.tool_call_id_type = "random"
    request = SimpleNamespace(
        request_id="r",
        top_logprobs=0,
        is_include_output_logprobs=lambda: False,
        tools=tools,
        tool_choice="auto",
    )
    ids = tokenizer.encode(raw, add_special_tokens=False)

    async def gen():
        prev, acc = "", []
        for i, tid in enumerate(ids):
            acc.append(tid)
            full = tokenizer.decode(acc, skip_special_tokens=False)
            delta = full[len(prev) :]
            prev = full
            ctx = SimpleContext.__new__(SimpleContext)
            ctx.last_output = SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text=delta,
                        token_ids=[tid],
                        logprobs=None,
                        finish_reason=("stop" if i == len(ids) - 1 else None),
                    )
                ]
            )
            yield ctx

    async def run():
        events = []
        async for ev in srv._process_simple_streaming_events(
            request=request,
            sampling_params=None,
            result_generator=gen(),
            context=None,
            model_name="m",
            tokenizer=tokenizer,
            request_metadata=None,
            created_time=0,
            _increment_sequence_number_and_return=lambda e: e,
        ):
            events.append(ev)
        return events

    return asyncio.run(run())


def test_streaming_namespace_tool_call_split(tokenizer, parser_manager):
    raw = (
        "<think>\nspawn\n</think>\n\n"
        "<tool_call>\n<function=multi_agent_v1__spawn_agent>\n"
        "<parameter=message>\ngo\n</parameter>\n</function>\n</tool_call>"
    )
    events = _run_streaming(tokenizer, parser_manager, raw, [_namespace_tool()])

    added = [
        e
        for e in events
        if getattr(e, "type", "") == "response.output_item.added"
        and getattr(e.item, "type", "") == "function_call"
    ]
    assert len(added) == 1
    assert added[0].item.name == "spawn_agent"
    assert added[0].item.namespace == "multi_agent_v1"

    # No raw tool-call XML must leak into output_text deltas.
    text = "".join(
        getattr(e, "delta", "")
        for e in events
        if getattr(e, "type", "") == "response.output_text.delta"
    )
    assert "<tool_call>" not in text
    assert "<function=" not in text


def test_streaming_done_event_carries_namespace_and_args(tokenizer, parser_manager):
    raw = (
        "<think>\nspawn\n</think>\n\n"
        "<tool_call>\n<function=multi_agent_v1__spawn_agent>\n"
        "<parameter=message>\ngo\n</parameter>\n</function>\n</tool_call>"
    )
    events = _run_streaming(tokenizer, parser_manager, raw, [_namespace_tool()])

    done = [
        e
        for e in events
        if getattr(e, "type", "") == "response.output_item.done"
        and getattr(e.item, "type", "") == "function_call"
    ]
    assert len(done) == 1
    assert done[0].item.name == "spawn_agent"
    assert done[0].item.namespace == "multi_agent_v1"
    import json

    assert json.loads(done[0].item.arguments) == {"message": "go"}
