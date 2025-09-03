# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from torch import fx
from torch._inductor import pattern_matcher as pm

from vllm.logger import init_logger

logger = init_logger(__name__)


class MultiOutputMatch:
    """
    This class provides utilities to process multi-output matches.

    It edits the graph to move match input nodes before match output nodes,
    a workaround for an issue in the torch Inductor pattern matcher:

    This issue is expected to be fixed in torch==2.9:
    https://github.com/pytorch/pytorch/issues/162019

    Formerly, this class was used to manually insert all multi-output
    replacements as the capability was completely broken before torch==2.6:
    https://github.com/pytorch/pytorch/issues/137280
    """

    def __init__(self, match: pm.Match):
        self.match = match

    def process(self):
        """
        Process a multi-output match and move input nodes before output nodes.
        Returns True to allow direct use as the extra_check function in
        PatternMatcherPass.register() and enable automatic replacement.

        Example:
        input_1 = empty()
        output_1 = relu(input_1)
        input_2 = empty()
        output_2 = output_1 + input_2

        Pattern matcher inserts the replacement before the first output node,
        resulting in a use of input_2 before its definition.
        """

        nodes = list(self.graph.nodes)
        output_nodes = self.match.output_nodes()
        input_nodes = self.match.args + list(self.match.kwargs.values())
        assert len(set(input_nodes) & set(output_nodes)) == 0, \
            f"Overlapping inputs and outputs: " \
            f"{set(input_nodes) & set(output_nodes)}"

        node_indices = {
            n: nodes.index(n)  # works for graph inputs as well
            for n in (output_nodes + input_nodes)
        }

        def node_index(node):
            # lazy compute other nodes as we don't need it for all nodes,
            # but there might be inputs to inputs
            if node not in node_indices:
                node_indices[node] = nodes.index(node)
            return node_indices[node]

        first_output_node = min(output_nodes,
                                key=lambda node: node_index(node))
        first_output_node_idx = node_index(first_output_node)
        insertion_point = nodes[first_output_node_idx - 1]

        # During this worklist process, node indices change and
        # topological ordering can be temporarily broken.
        nodes_to_process = list(input_nodes)  # copy
        while nodes_to_process:
            # breadth-first: inputs of inputs get appended later
            # to end up higher up in the graph
            arg_node = nodes_to_process.pop()
            if not isinstance(arg_node, fx.Node):
                continue  # only nodes can have other inputs
            if node_index(arg_node) < first_output_node_idx:
                continue  # node already before output

            logger.debug("Moving %s before %s (inserting after %s)", arg_node,
                         first_output_node, insertion_point)
            # arg is after the first output, move it before it.
            insertion_point.append(arg_node)
            # Any inputs to arg_node should also be checked
            for arg2 in arg_node.args:
                if not isinstance(arg2, fx.Node):
                    continue
                if arg2 in output_nodes:
                    raise ValueError(f"an output node {arg2} is "
                                     f"an input to a pattern input {arg_node}")

                nodes_to_process.append(arg2)

        # match always succeeds
        return True

    @property
    def nodes(self) -> list[fx.Node]:
        return self.match.nodes

    @property
    def graph(self) -> fx.Graph:
        return self.match.graph
