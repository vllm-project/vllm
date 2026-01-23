# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import operator

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from vllm.compilation.backends import (
    _is_sym_size_op,
    fold_sym_size_to_constants,
    split_graph,
)


def test_getitem_moved_to_producer_subgraph():
    """
    Test that getitem operations are moved to the same subgraph as their input,
    preventing tuple inputs to submodules.
    """

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        # torch.split returns a tuple, creating real getitem operations
        # Should become first submodule that produces tuple
        chunks = torch.split(x, x.shape[0] // 2, dim=0)

        # Following ops should become second submodule that consumes tuple
        result_0 = torch.relu(chunks[0])
        result_1 = torch.relu(chunks[1])
        return torch.cat([result_0, result_1], dim=0)

    x = torch.randn(4, 3)
    gm = make_fx(model_fn)(x)

    has_getitem = any(
        node.op == "call_function" and node.target == operator.getitem
        for node in gm.graph.nodes
    )
    assert has_getitem, "Test setup failed: graph should contain getitem operations"

    # Split on tuple producer aten::split
    split_ops = ["aten::split.Tensor"]
    split_gm, split_items = split_graph(gm, split_ops)
    assert len(split_items) == 2, "Graph should be split into 2 submodules"

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

    new_x = torch.randn(4, 3)
    output_original = gm(new_x)
    output_split = split_gm(new_x)

    assert torch.allclose(output_original, output_split), "Output mismatch"


def test_no_tuple_inputs_with_multiple_consumers():
    """
    Test that when a tuple is consumed by multiple split operations,
    getitem operations are properly moved to avoid tuple inputs.
    """

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        # torch.split returns a tuple, creating real getitem operations
        # Should become first submodule that produces tuple
        chunks = torch.split(x, x.shape[0] // 2, dim=0)

        # These should become second submodule consuming tuple
        result_1 = torch.relu(chunks[0])
        result_2 = torch.relu(chunks[1])

        # Artificial graph splitting point to create another
        # independent submodule that consumes tuple later
        # This would become the third submodule
        result_1 = torch.sigmoid(result_1)

        # Fourth submodule that consumes tuple
        result = torch.cat([chunks[0], chunks[1], result_1, result_2])
        return result

    x = torch.randn(4, 3)
    gm = make_fx(model_fn)(x)

    has_getitem = any(
        node.op == "call_function" and node.target == operator.getitem
        for node in gm.graph.nodes
    )
    assert has_getitem, "Test setup failed: graph should contain getitem operations"

    split_ops = ["aten::split.Tensor", "aten::sigmoid"]
    split_gm, split_items = split_graph(gm, split_ops)
    assert len(split_items) == 4, "Graph should be split into 4 submodules"

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

    new_x = torch.randn(4, 3)
    output_original = gm(new_x)
    output_split = split_gm(new_x)

    assert torch.allclose(output_original, output_split), "Output mismatch after split"


def test_fold_sym_size_to_constants():
    """
    Test fold_sym_size_to_constants: verifies sym_size ops are folded to
    constants, nodes remain as dead code, and graph produces correct output.
    """

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        feat = x.shape[1]
        last = x.shape[-1]  # negative index
        return x * batch + feat - last

    x = torch.randn(8, 16, 32)
    # Use symbolic tracing to preserve sym_size ops (concrete tracing)
    gm = make_fx(model_fn, tracing_mode="symbolic")(x)

    # Verify sym_size nodes exist before folding
    sym_size_nodes = [n for n in gm.graph.nodes if _is_sym_size_op(n)]
    assert len(sym_size_nodes) >= 3, (
        f"Should have sym_size ops, got {len(sym_size_nodes)}"
    )

    # Get expected output before folding
    expected_output = model_fn(x)

    # Build concrete_inputs using actual placeholder names from the graph
    concrete_inputs: dict[str, torch.Tensor] = {}
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            concrete_inputs[node.name] = x
            break

    # Fold sym_size to constants
    fold_sym_size_to_constants(gm, concrete_inputs)

    # Verify: nodes remain but have no users (dead code for reuse)
    sym_size_nodes_after = [n for n in gm.graph.nodes if _is_sym_size_op(n)]
    assert len(sym_size_nodes_after) == len(sym_size_nodes), (
        "sym_size nodes should remain as dead code"
    )
    for node in sym_size_nodes_after:
        assert len(node.users) == 0, f"{node.name} should have no users"

    # Verify: graph still produces correct output after recompile
    gm.recompile()
    actual_output = gm(x)
    assert torch.allclose(expected_output, actual_output), (
        "Output mismatch after folding sym_size to constants"
    )
