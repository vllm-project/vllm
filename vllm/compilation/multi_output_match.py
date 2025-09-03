# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from torch import fx
from torch._inductor import pattern_matcher as pm


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

        print(self.graph.python_code(root_module="self").src)
        nodes = list(self.graph.nodes)
        output_nodes = self.match.output_nodes()
        arg_nodes = self.match.args + list(self.match.kwargs.values())
        node_indices = {
            node: nodes.index(node)
            for node in (output_nodes + arg_nodes)
        }
        print(node_indices)

        first_output_node = min(output_nodes,
                                key=lambda node: node_indices[node])
        first_output_node_idx = node_indices[first_output_node]
        print(f"first output node is {first_output_node} "
              f"at index {first_output_node_idx}")

        nodes_to_process = list(arg_nodes)  # copy
        print(f"{nodes_to_process=}")
        while nodes_to_process:
            arg_node = nodes_to_process.pop()
            print(arg_node)
            if not isinstance(arg_node, fx.Node):
                continue  # only nodes can have other inputs
            if node_indices[arg_node] < first_output_node_idx:
                continue  #

            print(f"moving {arg_node} before {first_output_node}")
            # arg is after the first output, move it before it.
            first_output_node.prepend(arg_node)
            # Any inputs to arg_node should also be checked
            for arg2 in arg_node.args:
                if not isinstance(arg2, fx.Node):
                    continue
                if arg2 in output_nodes:
                    raise ValueError(f"an output node {arg2} is "
                                     f"an input to a pattern input {arg_node}")

                # process arg2
                print(f"adding {arg2} to process list")
                nodes_to_process.append(arg2)

        print(self.graph.python_code(root_module="self").src)
        # match always succeeds
        return True

    @property
    def nodes(self) -> list[fx.Node]:
        return self.match.nodes

    @property
    def graph(self) -> fx.Graph:
        return self.match.graph
