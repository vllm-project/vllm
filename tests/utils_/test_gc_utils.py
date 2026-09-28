# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

from vllm.utils.gc_utils import (
    GCDebugConfig,
    _compute_detailed_type,
    _compute_top_gc_collected_objects,
)


@dataclass
class Normal:
    v: int


@dataclass
class ListWrapper:
    vs: list[int]

    def __len__(self) -> int:
        return len(self.vs)


def test_compute_detailed_type():
    assert (
        _compute_detailed_type(Normal(v=8))
        == "<class 'tests.utils_.test_gc_utils.Normal'>"
    )

    assert _compute_detailed_type([1, 2, 3]) == "<class 'list'>(size:3)"
    assert _compute_detailed_type({4, 5}) == "<class 'set'>(size:2)"
    assert _compute_detailed_type({6: 7}) == "<class 'dict'>(size:1)"
    assert (
        _compute_detailed_type(ListWrapper(vs=[]))
        == "<class 'tests.utils_.test_gc_utils.ListWrapper'>(size:0)"
    )


def test_compute_top_gc_collected_objects():
    objects: list[Any] = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        {13, 14},
        {15: 16, 17: 18},
        Normal(v=19),
        Normal(v=20),
        Normal(v=21),
    ]
    assert _compute_top_gc_collected_objects(objects, top=-1) == ""
    assert _compute_top_gc_collected_objects(objects, top=0) == ""
    assert (
        _compute_top_gc_collected_objects(objects, top=1)
        == "    4:<class 'list'>(size:3)"
    )
    assert _compute_top_gc_collected_objects(objects, top=2) == "\n".join(
        [
            "    4:<class 'list'>(size:3)",
            "    3:<class 'tests.utils_.test_gc_utils.Normal'>",
        ]
    )
    assert _compute_top_gc_collected_objects(objects, top=3) == "\n".join(
        [
            "    4:<class 'list'>(size:3)",
            "    3:<class 'tests.utils_.test_gc_utils.Normal'>",
            "    1:<class 'set'>(size:2)",
        ]
    )


def test_gc_debug_config():
    assert not GCDebugConfig(None).enabled
    assert not GCDebugConfig("").enabled
    assert not GCDebugConfig("0").enabled

    config = GCDebugConfig("1")
    assert config.enabled
    assert config.top_objects == -1

    config = GCDebugConfig('{"top_objects":5}')
    assert config.enabled
    assert config.top_objects == 5
