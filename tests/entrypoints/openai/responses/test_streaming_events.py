# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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

    def test_tool_call_tail_emitted_before_content(self) -> None:
        """Verify that an in-flight tool call argument tail (name is None)
        is emitted before following reasoning and content."""
        tc_tail = _make_tool_call(0, arguments='{"city":"NYC"}')
        delta = DeltaMessage(reasoning="think", content="answer", tool_calls=[tc_tail])
        result = split_delta(delta)
        assert len(result) == 3
        assert result[0].tool_calls == [tc_tail]
        assert result[1].reasoning == "think"
        assert result[2].content == "answer"

    def test_new_tool_call_emitted_after_content(self) -> None:
        """Verify that a newly starting tool call (name is present)
        is emitted after preceding reasoning and content."""
        tc_new = _make_tool_call(0, name="get_weather", arguments='{"city":"NYC"}')
        delta = DeltaMessage(reasoning="think", content="answer", tool_calls=[tc_new])
        result = split_delta(delta)
        assert len(result) == 3
        assert result[0].reasoning == "think"
        assert result[1].content == "answer"
        assert result[2].tool_calls == [tc_new]

    def test_mixed_continuation_and_new_tool_calls_ordering(self) -> None:
        """Verify that when a DeltaMessage contains both a continuation group
        and a new tool call group, continuation precedes content and the new
        call follows content."""
        tc_tail = _make_tool_call(0, arguments='{"tail":1}')
        tc_new = _make_tool_call(1, name="calc", arguments='{"x":2}')
        delta = DeltaMessage(
            reasoning="think",
            content="answer",
            tool_calls=[tc_tail, tc_new],
        )
        result = split_delta(delta)
        assert len(result) == 4
        assert result[0].tool_calls == [tc_tail]
        assert result[1].reasoning == "think"
        assert result[2].content == "answer"
        assert result[3].tool_calls == [tc_new]


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

    def test_speculative_boundary_crossing_tool_call_tail_and_content(
        self,
    ) -> None:
        """Regression test for #55284: verify that multi-token/speculative
        decoding steps spanning a tool call finish and subsequent content
        do not trigger a nameless tool call open or crash with a Pydantic
        ValidationError."""
        processor = SimpleStreamingEventProcessor()

        # Step 1: Tool call opens
        step1 = DeltaMessage(
            tool_calls=[_make_tool_call(0, name="get_weather", arguments="")]
        )
        events1 = _run_through_processor(processor, step1)
        assert any(e.type == "response.output_item.added" for e in events1)
        assert processor.state.current_state == _StateType.TOOL_CALL

        # Step 2: Boundary crossing step (closing arguments + post-tool content)
        step2 = DeltaMessage(
            content="Here is the weather result.",
            tool_calls=[_make_tool_call(0, arguments='{"city":"NYC"}')],
        )
        events2 = _run_through_processor(processor, step2)

        types = [e.type for e in events2]
        assert "response.function_call_arguments.delta" in types
        assert "response.function_call_arguments.done" in types
        assert "response.output_text.delta" in types

        # Completed tool item must retain the original tool call name
        done_items = [e.item for e in events2 if e.type == "response.output_item.done"]
        assert done_items[0].name == "get_weather"
        assert done_items[0].arguments == '{"city":"NYC"}'

    def test_nameless_tool_call_open_fallback(self) -> None:
        """Verify that an unexpected nameless tool call open does not crash
        with a Pydantic ValidationError and uses a safe fallback."""
        processor = SimpleStreamingEventProcessor()
        tc = _make_tool_call(0, arguments='{"a":1}')
        # Force open without preceding state
        events = processor.open(_StateType.TOOL_CALL, tc)
        assert len(events) == 1
        assert events[0].item.name == "unknown"

    def test_closed_tool_call_does_not_leak_name_to_subsequent_nameless_tool_call(
        self,
    ) -> None:
        """Verify that when a tool call closes and content follows, a subsequent
        isolated nameless tool call open falls back to 'unknown' rather than
        leaking the previous tool call name."""
        processor = SimpleStreamingEventProcessor()

        # Step 1: Open and complete a named tool call
        step1 = DeltaMessage(
            tool_calls=[_make_tool_call(0, name="named_tool", arguments="{}")]
        )
        _run_through_processor(processor, step1)

        # Step 2: Content closes tool call and opens content
        step2 = DeltaMessage(content="Here is some text.")
        _run_through_processor(processor, step2)
        assert processor.state.current_state == _StateType.CONTENT
        assert processor.state.tool_call_name is None

        # Step 3: Isolated nameless tool call arrives
        tc_nameless = _make_tool_call(1, arguments='{"key":"val"}')
        events3 = processor.open(_StateType.TOOL_CALL, tc_nameless)
        assert len(events3) == 1
        assert events3[0].item.name == "unknown"
