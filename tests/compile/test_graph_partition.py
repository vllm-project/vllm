# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import operator

import pytest
import torch

from vllm.compilation.backends import split_graph


def test_getitem_moved_to_producer_subgraph():
    """
    Test that getitem operations are moved to the same subgraph as their input,
    preventing tuple inputs to submodules.
    """

    class TupleOutputModule(torch.nn.Module):
        """Tuple producer."""

        def forward(
            self, x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            a = x + 1
            b = y + 2
            return a, b

    class ConsumeTupleElementsModule(torch.nn.Module):
        """Tuple consumer."""

        def __init__(self):
            super().__init__()
            self.submod = TupleOutputModule()

        def forward(
            self, x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            t = self.submod(x, y)
            a = torch.nn.functional.relu(t[0])
            b = torch.nn.functional.relu(t[1])
            return a, b

    model = ConsumeTupleElementsModule()
    gm = torch.fx.symbolic_trace(model)

    split_ops = [torch.ops.aten.relu.default]
    split_gm, split_items = split_graph(gm, split_ops)

    assert len(split_items) > 0, "Graph should be split into submodules"

    for split_item in split_items:
        submodule = split_item.graph

        getitem_on_placeholder = []
        for node in submodule.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == operator.getitem
                and node.args[0].op == "placeholder"
            ):
                getitem_on_placeholder.append(node)

        assert len(getitem_on_placeholder) == 0, (
            f"Submodule {split_item.submod_name} has getitem operations on "
            f"placeholder nodes: {[n.name for n in getitem_on_placeholder]}. "
            "This means tuple inputs were not properly eliminated."
        )

    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    output_original = gm(x, y)
    output_split = split_gm(x, y)

    assert torch.allclose(output_original[0], output_split[0]), "First output mismatch"
    assert torch.allclose(output_original[1], output_split[1]), "Second output mismatch"


def test_no_tuple_inputs_with_multiple_consumers():
    """
    Test that when a tuple is consumed by multiple submodules,
    getitem operations are properly moved to avoid tuple inputs.
    """

    class TupleOutputModule(torch.nn.Module):
        """Tuple producer."""

        def forward(
            self, x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            a = x + 1
            b = y + 2
            return a, b

    class MultiConsumerModel(torch.nn.Module):
        """Tuple (multi-)consumer."""

        def __init__(self):
            super().__init__()
            self.submod = TupleOutputModule()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            t = self.submod(x + 1, x + 2)
            r1 = torch.nn.functional.sigmoid(t[0])
            r2 = torch.nn.functional.tanh(t[1])
            # Add relu so the last subgraph has real computation
            r2_relu = torch.nn.functional.relu(r2)
            return r1 + r2_relu

    model = MultiConsumerModel()
    gm = torch.fx.symbolic_trace(model)

    split_ops = [torch.ops.aten.sigmoid.default, torch.ops.aten.tanh.default]
    split_gm, split_items = split_graph(gm, split_ops)

    for split_item in split_items:
        submodule = split_item.graph

        for node in submodule.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == operator.getitem
                and node.args[0].op == "placeholder"
            ):
                pytest.fail(
                    f"Submodule {split_item.submod_name} has getitem on "
                    f"placeholder {node.args[0].name}, indicating it receives "
                    "a tuple input"
                )

    x = torch.randn(2, 3)
    output_original = gm(x)
    output_split = split_gm(x)

    assert torch.allclose(output_original, output_split), "Output mismatch after split"
