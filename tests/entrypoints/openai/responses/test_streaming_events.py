# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.engine.protocol import (
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


class TestToolCallContinuationOrdering:
    def test_split_delta_replays_argument_tail_before_state_switch(self):
        """An argument-only group is the tail of the open tool call and must
        come before reasoning/content; a named group is a new call and stays
        after them."""
        tail = _make_tool_call(0, arguments='"}')
        new_call = _make_tool_call(1, name="g", arguments="{")
        delta = DeltaMessage(reasoning="r", content="c", tool_calls=[new_call, tail])

        result = split_delta(delta)

        assert len(result) == 4
        assert result[0].tool_calls == [tail]
        assert result[1].reasoning == "r"
        assert result[2].content == "c"
        assert result[3].tool_calls == [new_call]

    def test_argument_tail_interleaved_with_reasoning(self):
        """Regression: one engine step can span the end of a tool call and
        the start of the next reasoning segment (common with speculative
        decoding). The argument tail must extend the open call instead of
        reopening TOOL_CALL with name=None, which failed
        ResponseFunctionToolCallItem validation and aborted the SSE stream
        without a terminal response event."""
        processor = SimpleStreamingEventProcessor()
        _run_through_processor(
            processor,
            DeltaMessage(tool_calls=[_make_tool_call(0, name="f", arguments='{"a')]),
        )
        assert processor.state.current_state == _StateType.TOOL_CALL

        events = _run_through_processor(
            processor,
            DeltaMessage(
                reasoning="think",
                tool_calls=[_make_tool_call(0, arguments='":1}')],
            ),
        )

        types = [e.type for e in events]
        tail_idx = types.index("response.function_call_arguments.delta")
        reasoning_idx = types.index("response.reasoning_text.delta")
        assert tail_idx < reasoning_idx

        added = [e for e in events if e.type == "response.output_item.added"]
        assert [e.item.type for e in added] == ["reasoning"]
        done = [e for e in events if e.type == "response.function_call_arguments.done"]
        assert len(done) == 1
        assert done[0].arguments == '{"a":1}'
        assert processor.state.current_state == _StateType.REASONING

    def test_orphan_argument_delta_does_not_abort_stream(self):
        """Defensive: an argument-only delta arriving while no tool call is
        open must not crash item validation mid-stream."""
        processor = SimpleStreamingEventProcessor()
        events = _run_through_processor(
            processor,
            DeltaMessage(tool_calls=[_make_tool_call(0, arguments='{"a":1}')]),
        )

        added = [e for e in events if e.type == "response.output_item.added"]
        assert len(added) == 1
        assert added[0].item.type == "function_call"
        assert added[0].item.name == ""
        deltas = [
            e for e in events if e.type == "response.function_call_arguments.delta"
        ]
        assert len(deltas) == 1
        assert deltas[0].delta == '{"a":1}'
