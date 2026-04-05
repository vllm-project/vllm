# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import operator

import pytest
import torch
import torch._dynamo
import torch.fx as fx
from torch.fx.experimental.proxy_tensor import make_fx

from vllm.compilation.backends import _is_empty_allocation_node, split_graph
from vllm.compilation.passes.fx_utils import find_op_nodes
from vllm.platforms import current_platform

# This import automatically registers `torch.ops.silly.attention`
from . import silly_attention  # noqa: F401

DEVICE_TYPE = current_platform.device_type


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

    torch.set_default_device(DEVICE_TYPE)

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


def _get_empty_nodes(split_item):
    return [
        node for node in split_item.graph.graph.nodes if _is_empty_allocation_node(node)
    ]


def _subgraphs_with_empty_nodes(split_items, *, is_splitting_graph):
    return [
        split_item
        for split_item in split_items
        if split_item.is_splitting_graph == is_splitting_graph
        and _get_empty_nodes(split_item)
    ]


def test_empty_only_partition_stays_separate_after_splitting_predecessor():
    """
    Empty-only subgraphs should not be merged when the only predecessor is
    a splitting-op subgraph.
    """

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        y = torch.sin(x)
        out = torch.empty_like(y)
        torch.ops.aten.cos.out(y, out=out)
        return out

    x = torch.randn(4, 3)
    gm = make_fx(model_fn)(x)

    split_ops = ["aten::sin", "aten::cos.out"]
    split_gm, split_items = split_graph(gm, split_ops)

    # Graph partitioning for this pattern is:
    # [sin], [empty_like], [cos.out].
    assert len(split_items) == 3, (
        "Empty-only partition should not merge into splitting-op subgraph"
    )

    splitting_with_empty = _subgraphs_with_empty_nodes(
        split_items, is_splitting_graph=True
    )
    assert len(splitting_with_empty) == 0, (
        "Splitting-op subgraphs should not contain empty allocation nodes: "
        f"{[item.submod_name for item in splitting_with_empty]}"
    )

    output_original = gm(x)
    output_split = split_gm(x)
    assert torch.allclose(output_original, output_split), "Output mismatch after split"


def test_empty_only_partition_is_merged():
    """
    Empty-only subgraphs should still be merged when a non-splitting predecessor
    exists. The merged empty node must remain outside splitting-op subgraphs.
    """

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        base = x + 1
        y = torch.sin(base)
        out = torch.empty_like(base)
        torch.ops.aten.cos.out(base, out=out)
        return out + y

    x = torch.randn(4, 3)
    gm = make_fx(model_fn)(x)
    split_gm, split_items = split_graph(gm, ["aten::sin", "aten::cos.out"])

    # Partitioning should be:
    # [add, empty_like], [sin], [cos.out], [add].
    assert len(split_items) == 4, (
        "Empty-only partition should be merged into non-splitting predecessor"
    )

    splitting_with_empty = _subgraphs_with_empty_nodes(
        split_items, is_splitting_graph=True
    )
    assert len(splitting_with_empty) == 0, (
        "Splitting-op subgraphs should not contain empty allocation nodes: "
        f"{[item.submod_name for item in splitting_with_empty]}"
    )

    non_splitting_with_empty = _subgraphs_with_empty_nodes(
        split_items, is_splitting_graph=False
    )
    assert len(non_splitting_with_empty) == 1, (
        "Exactly one non-splitting subgraph should contain the merged empty node"
    )
    assert len(_get_empty_nodes(non_splitting_with_empty[0])) == 1, (
        "Expected exactly one empty allocation node in merged subgraph"
    )

    output_original = gm(x)
    output_split = split_gm(x)
    assert torch.allclose(output_original, output_split), "Output mismatch after split"


def test_builtin_empty_only_partition_is_merged():
    """
    In Dynamo graphs, torch.empty/empty_like may appear as builtin call targets
    (not aten OpOverload). Ensure empty-only partitions are still merged.
    """

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        hidden = x + 1
        out1 = torch.empty_like(hidden)
        torch.ops.silly.attention(hidden, hidden, hidden, out1)
        out2 = torch.empty_like(hidden)
        torch.ops.silly.attention(out1, out1, hidden, out2)
        return out2 + hidden

    gm = torch.fx.symbolic_trace(model_fn)
    split_gm, split_items = split_graph(gm, ["silly::attention"])

    # Without empty-only merge, this graph would split into:
    # [add, empty_like], [attention], [empty_like], [attention], [add].
    assert len(split_items) == 4, "Builtin empty-only partition should be merged"

    splitting_with_empty = _subgraphs_with_empty_nodes(
        split_items, is_splitting_graph=True
    )
    assert len(splitting_with_empty) == 0, (
        "Splitting-op subgraphs should not contain empty allocation nodes: "
        f"{[item.submod_name for item in splitting_with_empty]}"
    )

    non_splitting_with_empty = _subgraphs_with_empty_nodes(
        split_items, is_splitting_graph=False
    )
    assert len(non_splitting_with_empty) == 1, (
        "Exactly one non-splitting subgraph should contain merged empty nodes"
    )
    assert len(_get_empty_nodes(non_splitting_with_empty[0])) == 2, (
        "Expected two builtin empty_like nodes in merged non-splitting subgraph"
    )

    x = torch.randn(2, 3, device=DEVICE_TYPE)
    output_original = gm(x)
    output_split = split_gm(x)
    assert torch.allclose(output_original, output_split), "Output mismatch after split"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sym_size_whole_shape_boundary():
    """
    Test that using x.size() (whole shape) across a split boundary can be
    compiled by standalone_compile.

    The dynamo graph looks like:
        shape = x.size()
        y = sigmoid(x)          # split point
        z = y.clone().view(shape)

    Which splits into:
        subgraph0(x) -> shape          # returns torch.Size — problematic
        subgraph1(x) -> y              # sigmoid
        subgraph2(y, shape) -> z       # view

    Two approaches to fix the torch.Size crossing:

    Approach 1 — move sym_size to consumer (memory implication: x passed to
    subgraph2 just for .size()):
        subgraph0(x) ->                # empty
        subgraph1(x) -> y
        subgraph2(y, x) -> z           # computes shape locally from x

    Approach 2 — decompose shape into individual int/SymInt values:
        subgraph0(x) -> s0, val        # returns individual scalars, not Size
        subgraph1(x) -> y
        subgraph2(y, s0, val) -> z     # reconstructs view args from scalars
    """
    from torch._inductor import standalone_compile

    captured_graph = None

    def capturing_backend(gm: fx.GraphModule, example_inputs: list) -> fx.GraphModule:
        nonlocal captured_graph
        captured_graph = gm
        return gm

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = torch.ops.aten.sigmoid.default(x)
        x = x.clone().view(shape)
        return x

    x = torch.randn(4, 8)
    torch._dynamo.mark_dynamic(x, 0)
    compiled_fn = torch.compile(model_fn, backend=capturing_backend)
    compiled_fn(x)

    split_gm, split_items = split_graph(captured_graph, ["aten::sigmoid"])
    assert len(split_items) == 3

    submod_0 = split_gm.submod_0
    example_input = torch.randn(4, 8)
    compiled = standalone_compile(
        submod_0, [example_input, 4], dynamic_shapes="from_example_inputs"
    )
    assert compiled is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shape_boundary_standalone_compile():
    """
    Repro for the original production bug:

        AssertionError: out_spec mismatch
        TreeSpec(tuple, None, [*, *, TreeSpec(Size, None, [*, *]), *])
        vs
        TreeSpec(tuple, None, [*, *, *, *])

    A subgraph outputs torch.Size (e.g. torch.Size([s72, 2048])) as one of
    its values when shape info crosses a split boundary. aot_autograd / inductor
    expect all submodule outputs to be flat tensors or scalars, not torch.Size.

    With the fix, x.size() is decomposed into individual sym_size.int calls
    so only scalar SymInts cross the boundary — not the torch.Size.
    """
    from torch._inductor import standalone_compile

    captured_graph = None

    def capturing_backend(gm: fx.GraphModule, example_inputs: list) -> fx.GraphModule:
        nonlocal captured_graph
        captured_graph = gm
        return gm

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = torch.ops.aten.sigmoid.default(x)
        x = x.clone().view(shape)
        return x

    x = torch.randn(4, 8)
    torch._dynamo.mark_dynamic(x, 0)
    torch.compile(model_fn, backend=capturing_backend)(x)

    split_gm, split_items = split_graph(captured_graph, ["aten::sigmoid"])
    assert len(split_items) == 3

    # Verify that the consumer subgraph only has a placeholder for the dynamic
    # dim (SymInt) — the static dim (8) should be inlined as a literal, not
    # threaded as a placeholder.
    consumer = split_items[-1]  # valid since len == 3: [producer, sigmoid, consumer]
    symint_placeholders = [
        n
        for n in consumer.graph.graph.nodes
        if n.op == "placeholder"
        and isinstance(n.meta.get("example_value"), torch.SymInt)
    ]
    static_int_placeholders = [
        n
        for n in consumer.graph.graph.nodes
        if n.op == "placeholder"
        and isinstance(n.meta.get("example_value"), int)
        and not isinstance(n.meta.get("example_value"), torch.SymInt)
    ]
    assert len(symint_placeholders) >= 1, (
        "Consumer should have a SymInt placeholder for the dynamic dim."
    )
    assert len(static_int_placeholders) == 0, (
        "Static dims should be inlined as literals, not threaded as placeholders."
    )

    submod_0 = split_gm.submod_0

    standalone_compile(
        submod_0, [torch.randn(4, 8), 4], dynamic_shapes="from_example_inputs"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_size_used_in_multiple_consumer_subgraphs():
    """
    Validates that x.size() (whole shape) used by multiple downstream subgraphs
    does not cause torch.Size to cross split boundaries.

    Model:
        shape = x.size()          # whole shape — must not cross as torch.Size
        z1 = sigmoid(x)           # split point 1
        y1 = y.view(shape)        # consumer 1 uses shape
        z2 = sigmoid(z1)          # split point 2
        y2 = y.view(shape)        # consumer 2 uses shape again

    Without the fix, torch.Size crosses the boundary as a submodule output,
    which aot_autograd / standalone_compile rejects.
    """
    captured_graph = None
    captured_inputs = None

    def capturing_backend(gm: fx.GraphModule, example_inputs: list) -> fx.GraphModule:
        nonlocal captured_graph, captured_inputs
        captured_graph = gm
        captured_inputs = example_inputs
        return gm

    def model_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        z1 = torch.ops.aten.sigmoid.default(x)
        y1 = y.view(shape)
        z2 = torch.ops.aten.sigmoid.default(z1)
        y2 = y.view(shape)
        return z2 + y1 + y2

    x = torch.randn(4, 8)
    y = torch.randn(4, 8)  # same shape as x so view(shape) doesn't specialize dim 0
    torch._dynamo.mark_dynamic(x, 0)
    torch._dynamo.mark_dynamic(y, 0)
    torch.compile(model_fn, backend=capturing_backend)(x, y)

    split_gm, split_items = split_graph(captured_graph, ["aten::sigmoid"])

    splitting_items = [item for item in split_items if item.is_splitting_graph]
    assert len(splitting_items) == 2

    # Verify functional correctness — fails without the fix because torch.Size
    # would cross a split boundary as a submodule output
    output_original = model_fn(x, y)
    output_split = split_gm(*captured_inputs)
    if isinstance(output_split, tuple):
        output_split = next(o for o in output_split if isinstance(o, torch.Tensor))
    assert torch.allclose(output_original, output_split), "Output mismatch after split"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sym_size_metadata_propagated():
    """
    Validates that new sym_size.int nodes created by the pre-pass have
    example_value metadata set. Without it, placeholder metadata in consumer
    subgraphs would be None, breaking any code that dynamically builds
    example inputs from metadata (e.g. standalone_compile per-submodule).
    """
    from torch._inductor import standalone_compile

    captured_graph = None

    def capturing_backend(gm: fx.GraphModule, example_inputs: list) -> fx.GraphModule:
        nonlocal captured_graph
        captured_graph = gm
        return gm

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = torch.ops.aten.sigmoid.default(x)
        x = x.clone().view(shape)
        return x

    x = torch.randn(4, 8)
    torch._dynamo.mark_dynamic(x, 0)
    torch.compile(model_fn, backend=capturing_backend)(x)

    split_gm, split_items = split_graph(captured_graph, ["aten::sigmoid"])

    # For each submodule, build example inputs purely from placeholder metadata.
    # This fails if example_value is None on any placeholder (i.e. metadata
    # was not propagated to the sym_size.int nodes we created).
    for item in split_items:
        submod = item.graph
        example_inputs = []
        for n in submod.graph.nodes:
            if n.op != "placeholder":
                continue
            ev = n.meta.get("example_value")
            assert ev is not None, (
                f"Placeholder '{n.name}' in {item.submod_name} has no "
                "example_value metadata. sym_size.int nodes must propagate "
                "metadata so consumer subgraphs can be introspected."
            )
            if isinstance(ev, torch.Tensor):
                example_inputs.append(torch.randn(*(int(d) for d in ev.shape)))
            else:
                example_inputs.append(int(ev))
        standalone_compile(submod, example_inputs, dynamic_shapes="from_example_inputs")
