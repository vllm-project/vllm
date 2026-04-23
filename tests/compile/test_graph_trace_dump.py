# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the hierarchical graph trace dump feature.
See https://github.com/vllm-project/vllm/issues/39215
"""

import tempfile
from collections import OrderedDict
from pathlib import Path

import torch

from vllm.compilation.graph_trace_dump import (
    dump_graph_hierarchy,
    dump_graph_hierarchy_to_file,
    trace_graph_structured,
)


def test_dump_graph_hierarchy_basic():
    """Test basic hierarchical output with manually constructed graph."""
    graph = torch.fx.Graph()

    x = graph.placeholder("x")

    # Node in outer module
    add_node = graph.call_function(torch.add, (x, x))
    add_node.meta["nn_module_stack"] = OrderedDict(
        [
            ("mod", ("", type("OuterModel", (), {}))),
        ]
    )

    # Node in nested module
    mul_node = graph.call_function(torch.mul, (add_node, add_node))
    mul_node.meta["nn_module_stack"] = OrderedDict(
        [
            ("mod", ("", type("OuterModel", (), {}))),
            ("mod.inner", ("inner", type("InnerModule", (), {}))),
        ]
    )

    # Another node in outer module (after leaving inner)
    neg_node = graph.call_function(torch.neg, (mul_node,))
    neg_node.meta["nn_module_stack"] = OrderedDict(
        [
            ("mod", ("", type("OuterModel", (), {}))),
        ]
    )

    graph.output(neg_node)

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    output = dump_graph_hierarchy(gm)

    lines = output.strip().split("\n")
    assert len(lines) >= 3, f"Expected at least 3 lines, got {len(lines)}: {output}"

    # Check module headers appear
    assert any("OuterModel" in line for line in lines), (
        f"Expected OuterModel header in output:\n{output}"
    )
    assert any("InnerModule" in line for line in lines), (
        f"Expected InnerModule header in output:\n{output}"
    )

    # Inner operations should be more indented than outer operations
    outer_op_lines = [
        line for line in lines if "torch.add" in line or "torch.neg" in line
    ]
    inner_op_lines = [line for line in lines if "torch.mul" in line]

    assert len(outer_op_lines) >= 2, (
        f"Expected >= 2 outer op lines, got: {outer_op_lines}"
    )
    assert len(inner_op_lines) >= 1, (
        f"Expected >= 1 inner op lines, got: {inner_op_lines}"
    )

    for inner in inner_op_lines:
        for outer in outer_op_lines:
            inner_indent = len(inner) - len(inner.lstrip())
            outer_indent = len(outer) - len(outer.lstrip())
            assert inner_indent > outer_indent, (
                f"Inner line should be more indented than outer.\n"
                f"Inner ({inner_indent}): {inner!r}\n"
                f"Outer ({outer_indent}): {outer!r}"
            )


def test_dump_graph_hierarchy_deep_nesting():
    """Test with 3 levels of nesting."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")

    OuterCls = type("OuterModel", (), {})
    MiddleCls = type("MiddleLayer", (), {})
    InnerCls = type("InnerLinear", (), {})

    # Depth 3 node
    deep_node = graph.call_function(torch.add, (x, x))
    deep_node.meta["nn_module_stack"] = OrderedDict(
        [
            ("mod", ("", OuterCls)),
            ("mod.layer", ("layer", MiddleCls)),
            ("mod.layer.linear", ("layer.linear", InnerCls)),
        ]
    )

    graph.output(deep_node)

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    output = dump_graph_hierarchy(gm)

    lines = output.strip().split("\n")
    # Should have 3 module headers + 1 op = 4 lines
    assert len(lines) >= 4, f"Expected >= 4 lines, got {len(lines)}:\n{output}"

    # The operation should be at depth 3 (6 spaces with indent_size=2)
    op_line = [line for line in lines if "torch.add" in line]
    assert len(op_line) == 1
    indent = len(op_line[0]) - len(op_line[0].lstrip())
    assert indent == 6, f"Expected indent of 6, got {indent}"


def test_dump_graph_hierarchy_no_module_stack():
    """Test nodes without nn_module_stack metadata."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")

    # Node without nn_module_stack
    add_node = graph.call_function(torch.add, (x, x))
    # No meta set

    graph.output(add_node)

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    output = dump_graph_hierarchy(gm)

    # Should still produce output (at top level, no indent)
    lines = output.strip().split("\n")
    assert len(lines) >= 1
    assert "torch.add" in lines[0]
    # No indentation for top-level nodes
    assert not lines[0].startswith(" ")


def test_dump_graph_hierarchy_indexed_layers():
    """Test that numeric fqn components produce [N] style display."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")

    LayerCls = type("DecoderLayer", (), {})

    node = graph.call_function(torch.add, (x, x))
    node.meta["nn_module_stack"] = OrderedDict(
        [
            ("mod.layers.0", ("layers.0", LayerCls)),
        ]
    )

    graph.output(node)

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    output = dump_graph_hierarchy(gm)

    assert "layers[0]: DecoderLayer" in output, (
        f"Expected 'layers[0]: DecoderLayer' in output:\n{output}"
    )


def test_dump_graph_hierarchy_layer_transition():
    """Test transitioning between layers (e.g., layer 0 to layer 1)."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")

    LayerCls = type("DecoderLayer", (), {})

    node0 = graph.call_function(torch.add, (x, x))
    node0.meta["nn_module_stack"] = OrderedDict(
        [
            ("mod.layers.0", ("layers.0", LayerCls)),
        ]
    )

    node1 = graph.call_function(torch.mul, (node0, node0))
    node1.meta["nn_module_stack"] = OrderedDict(
        [
            ("mod.layers.1", ("layers.1", LayerCls)),
        ]
    )

    graph.output(node1)

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    output = dump_graph_hierarchy(gm)

    assert "layers[0]: DecoderLayer" in output
    assert "layers[1]: DecoderLayer" in output


def test_dump_graph_hierarchy_to_file():
    """Test file output."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")

    node = graph.call_function(torch.add, (x, x))
    node.meta["nn_module_stack"] = OrderedDict(
        [
            ("mod", ("", type("Model", (), {}))),
        ]
    )

    graph.output(node)

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        tmppath = f.name

    dump_graph_hierarchy_to_file(gm, tmppath)

    content = Path(tmppath).read_text()
    assert len(content) > 0
    assert "Model" in content
    assert "torch.add" in content
    Path(tmppath).unlink()


def test_dump_graph_hierarchy_op_overload():
    """Test formatting of torch._ops.OpOverload targets."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")

    node = graph.call_function(torch.ops.aten.add.Tensor, (x, x))
    node.meta["nn_module_stack"] = OrderedDict(
        [
            ("mod", ("", type("Model", (), {}))),
        ]
    )

    graph.output(node)

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    output = dump_graph_hierarchy(gm)

    assert "aten.add.Tensor" in output, (
        f"Expected 'aten.add.Tensor' in output:\n{output}"
    )


def test_trace_graph_structured():
    """Test that trace_graph_structured emits both raw and hierarchy dumps."""
    from unittest.mock import patch

    graph = torch.fx.Graph()
    x = graph.placeholder("x")

    node = graph.call_function(torch.add, (x, x))
    node.meta["nn_module_stack"] = OrderedDict(
        [
            ("mod", ("", type("Model", (), {}))),
        ]
    )

    graph.output(node)

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    with patch("vllm.compilation.graph_trace_dump.trace_structured") as mock_ts:
        trace_graph_structured("test_graph", gm)

        # Should be called exactly twice: once for raw dump, once for hierarchy
        assert mock_ts.call_count == 2, (
            f"Expected 2 trace_structured calls, got {mock_ts.call_count}"
        )

        # Verify the metadata names
        first_call = mock_ts.call_args_list[0]
        second_call = mock_ts.call_args_list[1]

        assert first_call[0][0] == "graph_dump"
        assert first_call[1]["metadata_fn"]() == {"name": "test_graph"}

        assert second_call[0][0] == "graph_dump"
        assert second_call[1]["metadata_fn"]() == {
            "name": "test_graph_module_trace"
        }

        # Verify the hierarchy payload contains expected content
        hierarchy_payload = second_call[1]["payload_fn"]()
        assert "Model" in hierarchy_payload
        assert "torch.add" in hierarchy_payload


def test_dump_graph_hierarchy_source_context():
    """Test that stack_trace metadata produces '# via func_name' annotations."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")

    # Node with stack_trace pointing to a free function
    add_node = graph.call_function(torch.add, (x, x))
    add_node.meta["nn_module_stack"] = OrderedDict(
        [
            ("mod", ("", type("Model", (), {}))),
        ]
    )
    add_node.meta["stack_trace"] = (
        '  File "/home/user/project/model.py", line 10, in forward\n'
        '  File "/home/user/project/utils.py", line 5, in my_helper_fn\n'
    )

    # Node without stack_trace (should have no annotation)
    mul_node = graph.call_function(torch.mul, (add_node, add_node))
    mul_node.meta["nn_module_stack"] = OrderedDict(
        [
            ("mod", ("", type("Model", (), {}))),
        ]
    )

    graph.output(mul_node)

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    output = dump_graph_hierarchy(gm)

    # The add node should have the free function annotation
    assert "# via my_helper_fn" in output, (
        f"Expected '# via my_helper_fn' in output:\n{output}"
    )
    # The mul node should NOT have an annotation
    mul_lines = [line for line in output.split("\n") if "torch.mul" in line]
    assert len(mul_lines) == 1
    assert "# via" not in mul_lines[0], (
        f"Unexpected '# via' annotation on mul node: {mul_lines[0]}"
    )


def test_dump_graph_hierarchy_source_context_skips_torch_internals():
    """Test that torch internal frames are filtered out from source context."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")

    node = graph.call_function(torch.add, (x, x))
    node.meta["nn_module_stack"] = OrderedDict(
        [
            ("mod", ("", type("Model", (), {}))),
        ]
    )
    # Stack trace with only torch internals and forward
    node.meta["stack_trace"] = (
        '  File "/usr/lib/python3.12/torch/nn/modules/module.py",'
        ' line 1500, in _call_impl\n'
        '  File "/home/user/project/model.py", line 20, in forward\n'
    )

    graph.output(node)

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    output = dump_graph_hierarchy(gm)

    # No "# via" annotation because only forward and torch internals
    assert "# via" not in output, (
        f"Unexpected '# via' annotation in output:\n{output}"
    )
