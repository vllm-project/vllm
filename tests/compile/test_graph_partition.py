# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import operator

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from vllm.compilation.backends import split_graph
from vllm.compilation.fx_utils import find_op_nodes

# This import automatically registers `torch.ops.silly.attention`
from . import silly_attention  # noqa: F401


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


def test_consecutive_ops_in_split():
    """
    Test that consecutive splitting operations are grouped into the same subgraph
    """

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        """
        Define a simple model where consecutive operations create opportunities
        for splitting subgraphs.
        """
        # Apply silly attention followed by consecutive operations
        intermediate = torch.relu(x)
        attn_inout = torch.sqrt(intermediate)
        torch.ops.silly.attention(intermediate, intermediate, attn_inout, attn_inout)
        final_result = torch.sigmoid(attn_inout)
        return final_result

    torch.set_default_device("cuda")

    # Create the traced FX graph for the model
    x = torch.randn(8, 4)

    gm = make_fx(model_fn)(x)

    # Assert presence of the expected operations in the setup
    assert (
        len(list(find_op_nodes(torch.ops.aten.relu, gm.graph))) == 1
        and len(list(find_op_nodes(torch.ops.aten.sqrt, gm.graph))) == 1
    ), "Test setup failed: Expected sqrt and relu operations in the graph."

    # Configure split operations to test
    splitting_ops = ["silly::attention", "aten::sqrt"]
    split_gm, split_items = split_graph(gm, splitting_ops)

    # Validate the number of partitions
    assert len(split_items) == 3, (
        "Consecutive splitting operations were not grouped correctly."
    )

    # Validate that correctness is preserved
    new_x = torch.randn(8, 4)
    output_original = gm(new_x)
    output_split = split_gm(new_x)
    assert torch.allclose(output_original, output_split), (
        "Output mismatch after splitting."
    )

    # Check the splitting item has 2 nodes exactly (relu and attn)
    splitting_items = list(s for s in split_items if s.is_splitting_graph)
    assert len(splitting_items) == 1, "Expecting a single splitting graph"
    print(splitting_items[0].graph.graph)
    splitting_gm = splitting_items[0].graph
    assert len(splitting_gm.graph.nodes) == 4, "Expecting 4 nodes in splitting graph"
    assert [node.op for node in splitting_gm.graph.nodes] == ["placeholder"] + 2 * [
        "call_function"
    ] + ["output"]



def test_sym_size_moved_across_split_boundary():
    """
    Test that sym_size operations (tensor.shape accesses) are moved to the same
    subgraph as their consumers when they would otherwise cross subgraph boundaries.

    This prevents issues where PT2 doesn't fully support torch.Size as submodule
    output when sym_size is in one subgraph and its consumer is in another.

    Pattern being tested:
        # Original order that causes issues:
        size = tensor_a.shape[0]       # subgraph 0
        some_cg_unsafe_op              # subgraph 1 (split point)
        tensor_b = tensor_b.view(size) # subgraph 2 (would fail without fix)

        # After fix, sym_size is moved:
        some_cg_unsafe_op              # subgraph 1 (split point)
        size = tensor_a.shape[0]       # moved to subgraph 2
        tensor_b = tensor_b.view(size) # subgraph 2 (works correctly)
    """

    def model_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Get shape before the split point - this creates sym_size ops
        batch_size = x.shape[0]
        hidden_size = x.shape[1]

        # This becomes a splitting operation
        z = torch.sigmoid(x)

        # Use the shape values after the split point
        # Without the fix, this would fail because batch_size/hidden_size
        # would be outputs of the first subgraph (as torch.Size/SymInt)
        reshaped_y = y.view(batch_size, hidden_size)

        return z + reshaped_y

    x = torch.randn(4, 8)
    y = torch.randn(32)  # Will be reshaped to (4, 8)
    # Use symbolic tracing to generate sym_size operations
    gm = make_fx(model_fn, tracing_mode="symbolic")(x, y)

    # Verify the graph contains sym_size operations
    sym_size_nodes = [
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and "sym_size" in str(node.target)
    ]
    assert (
        len(sym_size_nodes) > 0
    ), "Test setup failed: graph should contain sym_size operations"

    # Split on sigmoid which is the split point
    split_ops = ["aten::sigmoid"]
    split_gm, split_items = split_graph(gm, split_ops)

    # After the fix, we expect 2 submodules:
    # - subgraph 1: sigmoid (split point)
    # - subgraph 2: sym_size ops + view + add (consumer subgraph)
    # The original subgraph 0 becomes empty because sym_size ops are moved
    # to the consumer subgraph, so it's not created.
    assert len(split_items) == 2, f"Expected 2 submodules, got {len(split_items)}"

    # Verify that one is the splitting graph (sigmoid) and one is not
    splitting_items = [item for item in split_items if item.is_splitting_graph]
    non_splitting_items = [item for item in split_items if not item.is_splitting_graph]
    assert len(splitting_items) == 1, "Should have exactly 1 splitting subgraph"
    assert len(non_splitting_items) == 1, "Should have exactly 1 non-splitting subgraph"

    # The non-splitting subgraph should contain the sym_size operations
    # (they were moved from before the split to after)
    consumer_subgraph = non_splitting_items[0].graph
    sym_size_in_consumer = [
        node
        for node in consumer_subgraph.graph.nodes
        if node.op == "call_function" and "sym_size" in str(node.target)
    ]
    assert len(sym_size_in_consumer) > 0, (
        "sym_size operations should be in the consumer subgraph (after split)"
    )

    # Verify functional correctness with same-shaped inputs
    output_original = gm(x, y)
    output_split = split_gm(x, y)
    assert torch.allclose(output_original, output_split), "Output mismatch after split"


def test_sym_size_with_multiple_consumers_in_different_subgraphs():
    """
    Test that when a sym_size result is used by consumers in multiple different
    subgraphs, it's placed in the earliest consumer subgraph.
    """

    def model_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Get shape before any split points
        size = x.shape[0]

        # First split point
        z1 = torch.sigmoid(x)

        # Use size after first split
        y1 = y[:size]

        # Second split point
        z2 = torch.sigmoid(z1)

        # Use size again after second split
        y2 = y[:size]

        return z2 + y1 + y2

    x = torch.randn(4, 8)
    y = torch.randn(8, 8)
    # Use symbolic tracing to generate sym_size operations
    gm = make_fx(model_fn, tracing_mode="symbolic")(x, y)

    # Split on both sigmoid operations
    split_ops = ["aten::sigmoid"]
    split_gm, split_items = split_graph(gm, split_ops)

    # Verify functional correctness with same-shaped inputs
    output_original = gm(x, y)
    output_split = split_gm(x, y)
    assert torch.allclose(output_original, output_split), "Output mismatch after split"
