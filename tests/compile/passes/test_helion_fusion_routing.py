# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm.compilation.passes.fusion.helion_routing import (
    route_helion_fusion_ops,
)


def _native(*args, **kwargs):
    return None


def _routed(*args, **kwargs):
    return None


def _unrelated(*args, **kwargs):
    return None


def test_route_helion_fusion_ops_retargets_supported_calls():
    graph = torch.fx.Graph()
    direct = graph.call_function(_native)
    functionalized = graph.call_function(auto_functionalized, args=(_native,))
    unrelated = graph.call_function(_unrelated)
    graph.output((direct, functionalized, unrelated))

    count = route_helion_fusion_ops(graph, {_native: _routed})  # type: ignore[arg-type]

    assert count == 2
    assert direct.target is _routed
    assert functionalized.args[0] is _routed
    assert unrelated.target is _unrelated
