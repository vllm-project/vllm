# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from torch import fx

from vllm.compilation.vllm_inductor_pass import VllmInductorPass


class PostCleanupPass(VllmInductorPass):
    """
    This pass performs cleanup after custom passes.
    It topologically sorts the graph and removes unused nodes.
    This is needed because the pattern matcher does not guarantee producing
    a topologically sorted graph, and there may be unused nodes left around.
    """

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        from torch._inductor.pattern_matcher import stable_topological_sort

        stable_topological_sort(graph)
        graph.eliminate_dead_code()
