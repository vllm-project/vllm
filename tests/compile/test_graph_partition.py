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


<<<<<<< HEAD
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
=======
def test_sym_size_moved_across_split_boundary():
    """
    Test that sym_size operations (tensor.shape accesses) are moved to the same
    subgraph as their consumers when they would otherwise cross subgraph boundaries.

    This prevents issues where PT2 doesn't fully support torch.Size as submodule
    output when sym_size is in one subgraph and its consumer is in another.
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
    assert len(sym_size_nodes) > 0, (
        "Test setup failed: graph should contain sym_size operations"
    )

    # Split on sigmoid which is the split point
    split_ops = ["aten::sigmoid"]
    split_gm, split_items = split_graph(gm, split_ops)

    # Find the sigmoid (splitting) subgraph and the consumer subgraph
    splitting_items = [item for item in split_items if item.is_splitting_graph]
    assert len(splitting_items) == 1, "Should have exactly 1 splitting subgraph"

    # KEY VERIFICATION: sym_size operations should be in the same subgraph
    # as the view operation (their consumer), NOT in an earlier subgraph.
    # This prevents torch.Size from crossing subgraph boundaries.

    # Find which subgraph contains the view operation
    view_subgraph = None
    for item in split_items:
        for node in item.graph.graph.nodes:
            if node.op == "call_function" and "view" in str(node.target).lower():
                view_subgraph = item
                break
        if view_subgraph:
            break

    assert view_subgraph is not None, "Should have a subgraph with view operation"

    # Verify sym_size operations are in the SAME subgraph as view
    sym_size_in_view_subgraph = [
        node
        for node in view_subgraph.graph.graph.nodes
        if node.op == "call_function" and "sym_size" in str(node.target)
    ]
    assert len(sym_size_in_view_subgraph) > 0, (
        "sym_size operations should be in the same subgraph as their consumer "
        "(view). This ensures torch.Size doesn't cross subgraph boundaries."
    )

    # Verify ordering within the consumer subgraph: sym_size before view
    consumer_nodes = list(view_subgraph.graph.graph.nodes)
    # CRITICAL VERIFICATION: The sigmoid (splitting/unsafe op) subgraph must
    # have a LOWER graph_id than the consumer subgraph. Since subgraphs execute
    # in order of graph_id, this proves that:
    # 1. Sigmoid runs FIRST
    # 2. sym_size + view run SECOND (in consumer subgraph)
    # Therefore, sym_size now happens AFTER the unsafe op.
    sigmoid_subgraph = splitting_items[0]
    assert sigmoid_subgraph.graph_id < view_subgraph.graph_id, (
        f"Sigmoid subgraph (graph_id={sigmoid_subgraph.graph_id}) must execute "
        f"before consumer subgraph (graph_id={view_subgraph.graph_id}). "
        "This ensures sym_size happens AFTER the unsafe operation."
    )

    sym_size_indices = [
        i
        for i, node in enumerate(consumer_nodes)
        if node.op == "call_function" and "sym_size" in str(node.target)
    ]
    view_indices = [
        i
        for i, node in enumerate(consumer_nodes)
        if node.op == "call_function" and "view" in str(node.target).lower()
    ]

    max_sym_size_idx = max(sym_size_indices)
    min_view_idx = min(view_indices)
    assert max_sym_size_idx < min_view_idx, (
        f"sym_size (max index {max_sym_size_idx}) should come before "
        f"view (min index {min_view_idx}) in the consumer subgraph."
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
>>>>>>> 009f91613 ([compile][graph_partition]Add tensor size handling)
