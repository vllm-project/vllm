# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collectives dispatch through their custom ops only while traced; eager
calls run the implementation directly."""

import torch
from torch._inductor.pattern_matcher import fwd_only

from vllm.distributed.parallel_state import GroupCoordinator


def _coordinator(use_custom_op_call: bool, only_when_tracing: bool):
    group = GroupCoordinator.__new__(GroupCoordinator)
    group.use_custom_op_call = use_custom_op_call
    group.custom_op_only_when_tracing = only_when_tracing
    return group


def test_collectives_use_custom_op_only_when_tracing():
    group = _coordinator(True, True)
    assert not group._use_custom_op()
    assert torch.compile(group._use_custom_op, fullgraph=True, backend="eager")()

    # The all-reduce fusion passes trace their patterns with make_fx, outside
    # Dynamo: the pattern must still contain the all_reduce op.
    seen = []

    def pattern(x):
        seen.append(group._use_custom_op())
        return x + 1

    fwd_only(pattern, [torch.zeros(2)])
    assert seen == [True]


def test_collectives_keep_custom_op_where_platform_requires_it():
    assert _coordinator(True, False)._use_custom_op()
    assert not _coordinator(False, True)._use_custom_op()
