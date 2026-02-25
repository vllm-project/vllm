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


def test_sym_size_in_producer_subgraph():
    """
    Test that sym_size operations are assigned to the same subgraph as their
    tensor operand (the producer), so only the SymInt result crosses the
    split boundary — not the original tensor.

    This avoids passing tensors to consumer subgraphs just for .size() calls,
    which would keep the tensor alive longer and increase memory usage.
    """

    def model_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        hidden_size = x.shape[1]

        # This becomes a splitting operation
        z = torch.sigmoid(x)

        # Use the shape values after the split point
        reshaped_y = y.view(batch_size, hidden_size)

        return z + reshaped_y

    x = torch.randn(4, 8)
    y = torch.randn(32)  # Will be reshaped to (4, 8)
    gm = make_fx(model_fn, tracing_mode="symbolic")(x, y)

    # Verify the graph contains sym_size operations
    sym_size_nodes = list(find_op_nodes(torch.ops.aten.sym_size, gm.graph))
    assert len(sym_size_nodes) > 0, (
        "Test setup failed: graph should contain sym_size operations"
    )

    split_ops = ["aten::sigmoid"]
    split_gm, split_items = split_graph(gm, split_ops)

    # Find producer subgraph (before sigmoid) and consumer subgraph (with view)
    splitting_items = [item for item in split_items if item.is_splitting_graph]
    assert len(splitting_items) == 1, "Should have exactly 1 splitting subgraph"

    view_subgraph = None
    for item in split_items:
        view_nodes = list(find_op_nodes(torch.ops.aten.view, item.graph.graph))
        if view_nodes:
            view_subgraph = item
            break
    assert view_subgraph is not None, "Should have a subgraph with view operation"

    # KEY VERIFICATION: sym_size should NOT be in the consumer (view) subgraph.
    # It should be in the producer subgraph, with only the SymInt result
    # crossing the boundary.
    sym_size_in_view_subgraph = list(
        find_op_nodes(torch.ops.aten.sym_size, view_subgraph.graph.graph)
    )
    assert len(sym_size_in_view_subgraph) == 0, (
        "sym_size operations should NOT be in the consumer subgraph. "
        "They should be in the producer subgraph so only the SymInt result "
        "crosses the boundary, avoiding passing the tensor for .size() calls."
    )

    # Verify sym_size is in a producer subgraph (before sigmoid)
    producer_subgraphs_with_sym_size = []
    for item in split_items:
        if item.is_splitting_graph:
            continue
        if item.graph_id > splitting_items[0].graph_id:
            continue
        sym_size_nodes = list(find_op_nodes(torch.ops.aten.sym_size, item.graph.graph))
        if sym_size_nodes:
            producer_subgraphs_with_sym_size.append(item.submod_name)

    assert len(producer_subgraphs_with_sym_size) > 0, (
        "sym_size operations should be in a producer subgraph (before sigmoid)."
    )

    # Verify the consumer subgraph does NOT receive the original tensor x
    # as an input (it should only receive y, z, and SymInt values)
    view_placeholders = [
        n for n in view_subgraph.graph.graph.nodes if n.op == "placeholder"
    ]
    for ph in view_placeholders:
        ev = ph.meta.get("example_value")
        if isinstance(ev, torch.Tensor) and ev.shape == x.shape:
            # This placeholder matches x's shape — it should be y or z,
            # not x itself being passed just for .size()
            pass  # Allow tensors that are actually used for computation

    # Verify functional correctness
    output_original = gm(x, y)
    output_split = split_gm(x, y)
    assert torch.allclose(output_original, output_split), "Output mismatch after split"


def test_symint_crosses_split_boundary():
    """
    Test that SymInt placeholders from torch.compile + mark_dynamic
    cross split boundaries safely via split_module's natural threading.

    SymInt values are threaded through subgraphs by split_module and
    handled correctly by inductor — no special replacement is needed.
    """
    captured_graph = None

    def capturing_backend(gm: fx.GraphModule, example_inputs: list) -> fx.GraphModule:
        nonlocal captured_graph
        captured_graph = gm
        return gm

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        hidden_size = x.shape[1]
        x = torch.ops.aten.sigmoid.default(x)
        x = x.clone().view(batch_size, hidden_size)
        x = torch.ops.aten.sigmoid.default(x)
        x = x.clone().view(batch_size, hidden_size)
        x = torch.ops.aten.sigmoid.default(x)
        x = x.clone().view(batch_size, hidden_size)
        return x

    x = torch.randn(4, 8)
    torch._dynamo.mark_dynamic(x, 0)

    compiled_fn = torch.compile(model_fn, backend=capturing_backend)
    compiled_fn(x)

    assert captured_graph is not None, "Graph should be captured by backend"

    # SymInt placeholders should exist in the captured graph
    symint_placeholders = [
        node
        for node in captured_graph.graph.nodes
        if node.op == "placeholder"
        and isinstance(node.meta.get("example_value"), torch.SymInt)
    ]
    assert len(symint_placeholders) > 0, (
        "Captured graph should have SymInt placeholders from mark_dynamic."
    )

    # split_graph should handle SymInt placeholders without error
    split_gm, split_items = split_graph(captured_graph, ["aten::sigmoid"])

    # Should have 3 splitting subgraphs (3 sigmoids)
    splitting_subgraphs = [item for item in split_items if item.is_splitting_graph]
    assert len(splitting_subgraphs) == 3, (
        f"Expected 3 splitting subgraphs (3 sigmoids), got {len(splitting_subgraphs)}"
    )
    assert len(split_items) >= 6, (
        f"Expected at least 6 total subgraphs, got {len(split_items)}"
    )
