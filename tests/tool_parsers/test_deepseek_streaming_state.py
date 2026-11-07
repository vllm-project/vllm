# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.tool_parsers.deepseek_streaming_state import DeepSeekStreamingTokenState


class CountedList(list[int]):
    def __init__(self, values: list[int]):
        super().__init__(values)
        self.contains_calls = 0
        self.count_calls = 0

    def __contains__(self, value: object) -> bool:
        self.contains_calls += 1
        return super().__contains__(value)

    def count(self, value: int) -> int:
        self.count_calls += 1
        return super().count(value)


def test_cumulative_tokens_are_only_inspected_on_first_update():
    state = DeepSeekStreamingTokenState(1, 2, 3)
    previous = CountedList([9] * 1024)
    current = CountedList([9] * 1025)
    delta = CountedList([9])

    assert state.update(previous, current, delta) is None
    assert current.contains_calls == 1
    assert previous.count_calls == current.count_calls == 0

    previous = CountedList([9] * 1025)
    current = CountedList([9] * 1026)
    delta = CountedList([9])
    assert state.update(previous, current, delta) is None
    assert current.contains_calls == 0
    assert previous.count_calls == current.count_calls == 0


def test_first_update_preserves_cumulative_marker_detection():
    state = DeepSeekStreamingTokenState(1, 2, 3)

    assert state.update([1, 2], [1, 2, 3], [3]) == (1, 0, 1, 1)


def test_counts_are_updated_from_deltas_after_tool_calls_start():
    state = DeepSeekStreamingTokenState(1, 2, 3)
    assert state.update([], [1, 2], [1, 2]) == (0, 0, 1, 0)

    previous = CountedList([1, 2])
    current = CountedList([1, 2, 2, 2, 3])
    assert state.update(previous, current, [2, 2, 3]) == (1, 0, 3, 1)
    assert previous.count_calls == 0
    assert current.count_calls == 0
