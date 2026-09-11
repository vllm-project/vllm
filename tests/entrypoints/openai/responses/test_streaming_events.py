# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from openai.types.responses import (
    CustomTool,
    ResponseCustomToolCall,
    response_text_delta_event,
)

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.entrypoints.openai.responses.streaming_events import (
    SimpleStreamingEventProcessor,
    _StateType,
    split_delta,
)
from vllm.entrypoints.openai.responses.utils import decode_reasoning_state


def _make_tool_call(
    index: int, name: str | None = None, arguments: str | None = None
) -> DeltaToolCall:
    fn = DeltaFunctionCall(name=name, arguments=arguments)
    return DeltaToolCall(index=index, function=fn)


class TestSplitDelta:
    def test_all_three_fields(self):
        tc = _make_tool_call(0, name="f")
        delta = DeltaMessage(reasoning="r", content="c", tool_calls=[tc])
        result = split_delta(delta)

        assert len(result) == 3
        assert result[0].reasoning == "r" and result[0].content is None
        assert result[1].content == "c" and result[1].reasoning is None
        assert len(result[2].tool_calls) == 1 and result[2].content is None

    def test_tool_calls_grouped_by_index(self):
        tc0 = _make_tool_call(0, name="f1")
        tc1 = _make_tool_call(1, name="f2")
        tc0b = _make_tool_call(0, arguments='{"a":1}')

        # Different indices → split
        result = split_delta(DeltaMessage(tool_calls=[tc0, tc1]))
        assert len(result) == 2
        assert result[0].tool_calls == [tc0]
        assert result[1].tool_calls == [tc1]

        # Same index → stays together
        delta = DeltaMessage(tool_calls=[tc0, tc0b])
        result = split_delta(delta)
        assert len(result) == 1
        assert result[0] is delta


def _run_through_processor(
    processor: SimpleStreamingEventProcessor,
    delta_message: DeltaMessage,
) -> list:
    """Simulate the streaming loop from serving.py for a single delta."""
    events = []
    for dm in split_delta(delta_message):
        target_state, tool_call = processor.resolve_target_state(dm)
        if target_state == _StateType.NONE:
            continue
        if processor.needs_transition(target_state, tool_call):
            events.extend(processor.close_current())
            events.extend(processor.open(target_state, tool_call))
        events.extend(processor.emit_delta(dm, None))
    return events


class TestProcessorCompoundDeltas:
    def test_all_three_states(self):
        tc = _make_tool_call(0, name="f", arguments="{}")
        delta = DeltaMessage(reasoning="r", content="c", tool_calls=[tc])

        processor = SimpleStreamingEventProcessor()
        events = _run_through_processor(processor, delta)

        types = [e.type for e in events]
        r_idx = types.index("response.reasoning_text.delta")
        c_idx = types.index("response.output_text.delta")
        fc_idx = types.index("response.function_call_arguments.delta")
        assert r_idx < c_idx < fc_idx

    def test_parallel_tool_calls(self):
        tc0 = _make_tool_call(0, name="f1", arguments='{"a":1}')
        tc1 = _make_tool_call(1, name="f2", arguments='{"b":2}')
        delta = DeltaMessage(tool_calls=[tc0, tc1])

        processor = SimpleStreamingEventProcessor()
        events = _run_through_processor(processor, delta)

        added = [e for e in events if e.type == "response.output_item.added"]
        deltas = [
            e for e in events if e.type == "response.function_call_arguments.delta"
        ]
        assert len(added) == 2
        assert len(deltas) == 2

    def test_split_name_and_args_same_index(self):
        """Regression: parsers like KimiK2 emit name and args as separate
        DeltaToolCalls at the same index within one DeltaMessage."""
        tc_name = _make_tool_call(0, name="get_weather")
        tc_args = _make_tool_call(0, arguments='{"city":"SF"}')
        delta = DeltaMessage(tool_calls=[tc_name, tc_args])

        processor = SimpleStreamingEventProcessor()
        events = _run_through_processor(processor, delta)

        deltas = [
            e for e in events if e.type == "response.function_call_arguments.delta"
        ]
        assert len(deltas) == 1
        assert deltas[0].delta == '{"city":"SF"}'

    def test_reasoning_to_content_transition(self):
        """Regression: the old special case in emit_delta handled this;
        now split_delta handles it generically."""
        processor = SimpleStreamingEventProcessor()
        _run_through_processor(processor, DeltaMessage(reasoning="think"))
        assert processor.state.current_state == _StateType.REASONING

        events = _run_through_processor(
            processor, DeltaMessage(reasoning="more", content="answer")
        )
        types = [e.type for e in events]
        assert "response.reasoning_text.delta" in types
        assert "response.output_text.delta" in types


class TestSimpleItemShapes:
    def test_done_items_match_final_shape(self):
        processor = SimpleStreamingEventProcessor()
        events = _run_through_processor(processor, DeltaMessage(reasoning="r"))
        events += _run_through_processor(processor, DeltaMessage(content="c"))
        events += processor.close_current()

        added = [e.item for e in events if e.type == "response.output_item.added"]
        done = [e.item for e in events if e.type == "response.output_item.done"]
        assert [item.id for item in done] == [item.id for item in added]
        assert done[0].id.startswith("rs_")
        assert done[1].id.startswith("msg_")
        assert "summary" not in done[1].model_dump()
        assert done[1].content[0].logprobs is None
        assert done[0].encrypted_content is None

    def test_content_logprobs_carried_to_done_item(self):
        top = response_text_delta_event.LogprobTopLogprob(token="hi", logprob=-0.5)
        logprob = response_text_delta_event.Logprob(
            token="hi", logprob=-0.5, top_logprobs=[top]
        )
        processor = SimpleStreamingEventProcessor()
        events = processor.open(_StateType.CONTENT)
        events += processor.emit_delta(
            DeltaMessage(content="hi"), None, lambda _: [logprob]
        )
        events += processor.close_current()

        delta = next(e for e in events if e.type == "response.output_text.delta")
        assert delta.logprobs == [logprob]
        part = events[-1].item.content[0]
        assert [lp.token for lp in part.logprobs] == ["hi"]
        assert part.logprobs[0].bytes == [104, 105]
        assert part.logprobs[0].top_logprobs[0].bytes == [104, 105]
        content_done = next(e for e in events if e.type == "response.content_part.done")
        assert content_done.part == part

    def test_encrypted_reasoning_on_done_item(self):
        processor = SimpleStreamingEventProcessor(encrypt_reasoning=True)
        events = _run_through_processor(processor, DeltaMessage(reasoning="secret"))
        events += processor.close_current()
        assert decode_reasoning_state(events[-1].item.encrypted_content) == "secret"

    def test_arguments_done_emitted_without_deltas(self):
        processor = SimpleStreamingEventProcessor()
        events = _run_through_processor(
            processor, DeltaMessage(tool_calls=[_make_tool_call(0, name="noop")])
        )
        events += processor.close_current()
        assert [e.type for e in events] == [
            "response.output_item.added",
            "response.function_call_arguments.done",
            "response.output_item.done",
        ]
        assert events[-1].item.id.startswith("fc_")


class TestCustomToolStreaming:
    def test_custom_tool_input_events(self):
        processor = SimpleStreamingEventProcessor(
            tools=[CustomTool(type="custom", name="emit_command")]
        )
        events = _run_through_processor(
            processor,
            DeltaMessage(
                tool_calls=[
                    _make_tool_call(0, name="emit_command", arguments='{"input": "ls ')
                ]
            ),
        )
        events += _run_through_processor(
            processor,
            DeltaMessage(tool_calls=[_make_tool_call(0, arguments='-la \\"x\\""}')]),
        )
        events += processor.close_current()

        assert [e.type for e in events] == [
            "response.output_item.added",
            "response.custom_tool_call_input.delta",
            "response.custom_tool_call_input.delta",
            "response.custom_tool_call_input.done",
            "response.output_item.done",
        ]
        done = events[-1].item
        assert isinstance(done, ResponseCustomToolCall)
        assert done.input == 'ls -la "x"'
        assert "".join(e.delta for e in events[1:3]) == done.input
        assert events[3].input == done.input
        assert events[0].item.type == "custom_tool_call"
        assert events[0].item.id == done.id
        assert done.id.startswith("ctc_")
        assert done.call_id.startswith("call_")
