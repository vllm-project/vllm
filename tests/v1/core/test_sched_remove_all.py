# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.core.sched.utils import remove_all


def test_remove_all_drops_every_copy_of_one_item():
    first = object()
    second = object()
    running = [first, second, first]

    result = remove_all(running, {first})

    assert result is running
    assert result == [second]


def test_remove_all_multiple_items_returns_a_new_list():
    values = [1, 2, 3, 2]
    result = remove_all(values, {2, 3})
    assert result == [1]
    assert values == [1, 2, 3, 2]
