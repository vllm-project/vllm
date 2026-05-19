# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.metrics.loggers import _inc_counter_nonnegative


class FakeCounter:
    def __init__(self):
        self.values: list[int] = []

    def inc(self, value: int):
        if value < 0:
            raise ValueError("Negative value not allowed")
        self.values.append(value)


def test_inc_counter_nonnegative_clamps_negative_values():
    counter = FakeCounter()

    _inc_counter_nonnegative(counter, -3)
    _inc_counter_nonnegative(counter, 4)

    assert counter.values == [0, 4]
