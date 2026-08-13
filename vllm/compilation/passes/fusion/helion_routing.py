# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Route fusion-produced native ops through CUDA-graph-aware Helion ops."""

from __future__ import annotations

import torch
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm.compilation.passes.inductor_pass import InductorPass
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.kernels.helion.routing import build_compiled_helion_op_map
from vllm.logger import init_logger

logger = init_logger(__name__)


def route_helion_fusion_ops(
    graph: fx.Graph,
    op_map: dict[torch._ops.OpOverload, torch._ops.OpOverload],
) -> int:
    """Retarget compatible direct and auto-functionalized native op calls."""
    count = 0
    for node in graph.nodes:
        if node.op != "call_function":
            continue

        if replacement := op_map.get(node.target):
            node.target = replacement
            count += 1
        elif node.target is auto_functionalized and node.args:
            if replacement := op_map.get(node.args[0]):
                node.args = (replacement, *node.args[1:])
                count += 1

    return count


class HelionFusionRoutingPass(VllmInductorPass):
    """Route compatible fusion-only ops to Helion during CUDA-graph capture."""

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.op_map = build_compiled_helion_op_map()

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = route_helion_fusion_ops(graph, self.op_map)
        routed_targets = set(self.op_map.values())
        routed_names = sorted(
            node.target._schema.name
            for node in graph.nodes
            if node.op == "call_function" and node.target in routed_targets
        )
        logger.debug(
            "HelionFusionRoutingPass routed %d ops: %s",
            self.matched_count,
            routed_names,
        )

    def uuid(self) -> str:
        return InductorPass.hash_dict(
            {
                "pass": type(self).__name__,
                "ops": sorted(str(op) for op in self.op_map),
            }
        )
