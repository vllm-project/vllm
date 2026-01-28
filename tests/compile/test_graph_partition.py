# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import operator

import pytest
import torch
import torch._dynamo
import torch.fx as fx
from torch.fx.experimental.proxy_tensor import make_fx

from vllm.compilation.backends import (
    split_graph,
)
from vllm.compilation.fx_utils import find_op_nodes

# This import automatically registers `torch.ops.silly.attention`
from . import silly_attention  # noqa: F401
from vllm.compilation.fx_utils import find_op_nodes, is_func


@pytest.fixture
def vllm_compile_env(monkeypatch):
    """Set up vLLM compilation environment variables for testing."""
    monkeypatch.setenv("VLLM_ALL2ALL_BACKEND", "deepep_high_throughput")
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "debug")
    yield


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
    sym_size_nodes = list(find_op_nodes(torch.ops.aten.sym_size, gm.graph))
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
        view_nodes = list(find_op_nodes(torch.ops.aten.view, item.graph.graph))
        if view_nodes:
            view_subgraph = item
            break

    assert view_subgraph is not None, "Should have a subgraph with view operation"

    # Verify sym_size operations are in the SAME subgraph as view
    sym_size_in_view_subgraph = list(
        find_op_nodes(torch.ops.aten.sym_size, view_subgraph.graph.graph)
    )
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
        if is_func(node, torch.ops.aten.sym_size.int)
    ]
    view_indices = [
        i
        for i, node in enumerate(consumer_nodes)
        if is_func(node, torch.ops.aten.view.default)
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


def test_sym_size_with_torch_compile_and_mark_dynamic():
    """
    Test handling of SymInt placeholders from torch.compile with mark_dynamic
    across MULTIPLE split subgraphs.

    When using torch.compile + mark_dynamic, the captured graph has:
    - SymInt placeholders (e.g., s77) as separate inputs
    - Operations that use the SymInt directly (e.g., view([s77, 8]))

    standalone_compile / inductor expects only tensor inputs. split_graph must:
    1. Replace SymInt placeholder uses with sym_size calls on tensor inputs
    2. Replicate sym_size to ALL consumer subgraphs that need the dynamic size
    3. Remove unused SymInt placeholders from the final graph

    This test validates the complete SymInt -> sym_size pipeline with MULTIPLE
    split boundaries to ensure sym_size is correctly replicated across subgraphs:
    - Phase 1: SymInt placeholders exist in the captured graph
    - Phase 2 & 3: split_graph handles SymInt replacement and removal
    - Phase 4: sym_size.int exists in EACH consumer subgraph that needs it
    - Phase 5: Functional correctness with original input
    - Phase 6: Functional correctness with different batch size
    - Phase 7: Validate multiple split subgraphs exist
    """
    captured_graph = None

    def capturing_backend(gm: fx.GraphModule, example_inputs: list) -> fx.GraphModule:
        nonlocal captured_graph
        captured_graph = gm
        return gm

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        # Get the dynamic shape before any splits
        batch_size = x.shape[0]
        hidden_size = x.shape[1]

        # First split point - sigmoid #1
        x = torch.ops.aten.sigmoid.default(x)

        # Use dynamic size after first split - creates sym_size consumer
        x = x.clone().view(batch_size, hidden_size)

        # Second split point - sigmoid #2
        x = torch.ops.aten.sigmoid.default(x)

        # Use dynamic size again after second split - another sym_size consumer
        x = x.clone().view(batch_size, hidden_size)

        # Third split point - sigmoid #3
        x = torch.ops.aten.sigmoid.default(x)

        # Use dynamic size again after third split - yet another consumer
        x = x.clone().view(batch_size, hidden_size)

        return x

    x = torch.randn(4, 8)
    # Mark the first dimension as dynamic
    torch._dynamo.mark_dynamic(x, 0)

    compiled_fn = torch.compile(model_fn, backend=capturing_backend)
    compiled_fn(x)

    assert captured_graph is not None, "Graph should be captured by backend"

    # ===== PHASE 1: Validate SymInt placeholders exist in captured graph =====
    symint_placeholders = [
        node
        for node in captured_graph.graph.nodes
        if node.op == "placeholder"
        and node.meta.get("example_value") is not None
        and isinstance(node.meta.get("example_value"), torch.SymInt)
    ]
    assert len(symint_placeholders) > 0, (
        "Phase 1 FAILED: Captured graph should have SymInt placeholders from "
        "mark_dynamic. This is the prerequisite for testing the sym_size pipeline."
    )

    # Record original SymInt users for later validation
    original_symint_users = {}
    for symint_node in symint_placeholders:
        users = [u for u in symint_node.users if u.op != "output"]
        original_symint_users[symint_node.name] = [u.name for u in users]

    # ===== PHASE 2 & 3: split_graph handles SymInt replacement and removal =====
    # NOTE: split_graph modifies the input graph in-place!
    # With 3 sigmoid operations, we expect 7 subgraphs:
    # submod_0 (before sigmoid #1), submod_1 (sigmoid #1),
    # submod_2 (between sigmoid #1 and #2), submod_3 (sigmoid #2),
    # submod_4 (between sigmoid #2 and #3), submod_5 (sigmoid #3),
    # submod_6 (after sigmoid #3)
    split_gm, split_items = split_graph(captured_graph, ["aten::sigmoid"])

    # ===== PHASE 7: Validate multiple split subgraphs exist =====
    # Count splitting subgraphs (the sigmoid operations)
    splitting_subgraphs = [item for item in split_items if item.is_splitting_graph]

    assert len(splitting_subgraphs) == 3, (
        f"Phase 7 FAILED: Expected 3 splitting subgraphs (3 sigmoids), "
        f"got {len(splitting_subgraphs)}"
    )
    # Note: Total subgraphs can be 6 or 7 depending on whether there are
    # operations before the first sigmoid. With torch.compile, shape access
    # operations may be folded differently, resulting in 6 subgraphs:
    # submod_1 (sigmoid #1), submod_2 (compute), submod_3 (sigmoid #2),
    # submod_4 (compute), submod_5 (sigmoid #3), submod_6 (compute)
    assert len(split_items) >= 6, (
        f"Phase 7 FAILED: Expected at least 6 total subgraphs "
        f"(3 sigmoids + at least 3 compute blocks), got {len(split_items)}"
    )

    # ===== PHASE 3: Validate SymInt placeholders are removed from split_gm =====
    split_placeholders = [
        node for node in split_gm.graph.nodes if node.op == "placeholder"
    ]

    remaining_symint_placeholders = [
        node
        for node in split_placeholders
        if node.meta.get("example_value") is not None
        and isinstance(node.meta.get("example_value"), torch.SymInt)
    ]
    assert len(remaining_symint_placeholders) == 0, (
        f"Phase 3 FAILED: split_gm should not have SymInt placeholders after "
        f"_remove_symint_placeholders. Found: "
        f"{[n.name for n in remaining_symint_placeholders]}. "
        "This means SymInt would be passed as input which inductor doesn't support."
    )

    # ===== PHASE 4: Validate sym_size.int exists in consumer subgraphs =====
    # Each non-splitting subgraph that uses dynamic sizes should have sym_size.int
    # to compute the dynamic dimension locally from the tensor input.
    total_sym_size_nodes = 0
    subgraphs_with_sym_size = []

    for item in split_items:
        sym_size_nodes = list(find_op_nodes(torch.ops.aten.sym_size, item.graph.graph))

        if sym_size_nodes:
            total_sym_size_nodes += len(sym_size_nodes)
            subgraphs_with_sym_size.append(item.submod_name)

    assert total_sym_size_nodes > 0, (
        "Phase 4 FAILED: No sym_size.int nodes found in any subgraph. "
        "split_graph should replace SymInt placeholders with sym_size.int calls "
        "that compute dynamic sizes from tensor inputs."
    )

    # With 3 split boundaries and dynamic size usage after each split,
    # we expect sym_size to be replicated to multiple consumer subgraphs
    assert len(subgraphs_with_sym_size) >= 3, (
        f"Phase 4 FAILED: sym_size should exist in consumer subgraphs. "
        f"Found sym_size in {len(subgraphs_with_sym_size)} subgraphs: "
        f"{subgraphs_with_sym_size}"
    )

    # ===== PHASE 5: Validate functional correctness =====
    # split_gm should work with tensor-only input (no SymInt)
    output_split = split_gm(x)

    # Handle case where output is a tuple
    if isinstance(output_split, tuple):
        output_split = output_split[0]

    # For reference, run the model directly to get expected output
    expected_output = model_fn(x)

    assert torch.allclose(expected_output, output_split), (
        "Phase 5 FAILED: Output mismatch after split. The sym_size pipeline "
        "should preserve functional correctness."
    )

    # ===== PHASE 6: Validate with different batch size =====
    # The dynamic dimension should work with different sizes
    x_different = torch.randn(8, 8)  # Different batch size
    output_different = split_gm(x_different)
    if isinstance(output_different, tuple):
        output_different = output_different[0]
    expected_different = model_fn(x_different)
    assert torch.allclose(expected_different, output_different), (
        "Phase 6 FAILED: Output mismatch with different batch size. "
        "sym_size should correctly compute the dynamic dimension at runtime."
    )
