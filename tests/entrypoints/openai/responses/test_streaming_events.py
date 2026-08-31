# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.entrypoints.openai.responses.streaming_events import (
    SimpleStreamingEventProcessor,
    StreamingParsedOutput,
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


class TestStreamingParsedOutput:
    """StreamingParsedOutput accumulates parsed delta components so the
    final response can be built without reparsing the full text."""

    def test_empty_has_no_output(self):
        acc = StreamingParsedOutput()
        assert not acc.has_any()

    def test_accumulates_reasoning_and_content(self):
        acc = StreamingParsedOutput()
        acc.update(DeltaMessage(reasoning="think "))
        acc.update(DeltaMessage(reasoning="hard"))
        acc.update(DeltaMessage(content="Hello "))
        acc.update(DeltaMessage(content="world"))

        assert acc.has_any()
        assert acc.reasoning_text == "think hard"
        assert acc.content_text == "Hello world"

    def test_merges_fragmented_tool_calls_by_index(self):
        acc = StreamingParsedOutput()
        acc.update(
            DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=0,
                        id="call_a",
                        function=DeltaFunctionCall(name="get_weather"),
                    )
                ]
            )
        )
        acc.update(
            DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=0, function=DeltaFunctionCall(arguments='{"city":')
                    )
                ]
            )
        )
        acc.update(
            DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=0, function=DeltaFunctionCall(arguments='Paris"}')
                    )
                ]
            )
        )
        # A second, distinct tool call.
        acc.update(
            DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=1,
                        id="call_b",
                        function=DeltaFunctionCall(name="get_time", arguments="{}"),
                    )
                ]
            )
        )

        assert len(acc.tool_calls) == 2
        first = acc.tool_calls[0]
        assert first.name == "get_weather"
        assert first.arguments == '{"city":Paris"}'
        assert first.id == "call_a"
        second = acc.tool_calls[1]
        assert second.name == "get_time"
        assert second.arguments == "{}"

    def test_tool_only_output(self):
        acc = StreamingParsedOutput()
        acc.update(
            DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=0,
                        function=DeltaFunctionCall(name="f", arguments="{}"),
                    )
                ]
            )
        )
        assert acc.has_any()
        assert acc.reasoning_text == ""
        assert acc.content_text == ""
