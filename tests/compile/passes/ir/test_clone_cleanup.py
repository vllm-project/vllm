# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Comprehensive tests for UnsafeCloneEliminationPass.

This test suite exercises all possible valid FX graph patterns involving clones:
1. Clone with no users (dead code)
2. Clone with read-only users
3. Clone with mutation users
4. Clone of graph input
5. Clone with original used after mutation
6. Clone chains
"""

import pytest
import torch
from torch import fx
from torch.fx.experimental.proxy_tensor import make_fx

from vllm.compilation.passes.fx_utils import find_op_nodes
from vllm.compilation.passes.inductor_pass import get_pass_context, pass_context
from vllm.compilation.passes.ir.clone_elimination import (
    UnsafeCloneEliminationPass,
    user_writes_to_node,
)
from vllm.config import VllmConfig
from vllm.config.utils import Range


def count_clones(graph: fx.Graph) -> int:
    """Count clone nodes in a graph."""
    return len(list(find_op_nodes(torch.ops.aten.clone.default, graph)))


@pytest.fixture(scope="function")
def clone_cleanup_pass():
    return UnsafeCloneEliminationPass(VllmConfig())


@pytest.fixture(autouse=True)
def setup_pass_context():
    """Set up pass context for each test."""
    with pass_context(compile_range=Range(1, 8192)):
        yield


class TestCloneCleanup:
    """Test UnsafeCloneEliminationPass behavior on various graph patterns."""

    def test_remove_clone_readonly_users(self, clone_cleanup_pass):
        """Clone with only read-only users should be removed."""

        def f(x: torch.Tensor) -> torch.Tensor:
            x_clone = x.clone()
            return x_clone + 1

        inp = torch.randn(2, 3)
        graph_module = make_fx(f)(inp)
        assert count_clones(graph_module.graph) == 1

        expected = graph_module(inp)
        clone_cleanup_pass(graph_module.graph)
        graph_module.recompile()
        actual = graph_module(inp)

        assert count_clones(graph_module.graph) == 0
        torch.testing.assert_close(actual, expected)

    def test_keep_clone_with_mutation_and_original_used_after(self, clone_cleanup_pass):
        """Clone must be kept if it's mutated AND original is used after mutation."""

        def f(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x.relu()  # not a graph param
            x_clone = x.clone()
            x_clone.add_(1)
            return x, x_clone

        inp = torch.randn(2, 3)
        graph_module = make_fx(f)(inp)
        assert count_clones(graph_module.graph) == 1

        expected = graph_module(inp)
        clone_cleanup_pass(graph_module.graph)
        graph_module.recompile()
        actual = graph_module(inp)

        # Clone should be KEPT because original is used after mutation
        assert count_clones(graph_module.graph) == 1
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    def test_remove_clone_with_mutation_no_original_use(self, clone_cleanup_pass):
        """Clone can be removed if it's mutated but original is not used after."""

        def f(x: torch.Tensor) -> torch.Tensor:
            x = x.relu()  # not a graph param
            x_clone = x.clone()
            x_clone.add_(1)
            return x_clone

        inp = torch.randn(2, 3)
        graph_module = make_fx(f)(inp)
        assert count_clones(graph_module.graph) == 1

        expected = graph_module(inp)
        clone_cleanup_pass(graph_module.graph)
        graph_module.recompile()
        actual = graph_module(inp)

        assert count_clones(graph_module.graph) == 0
        torch.testing.assert_close(actual, expected)

    def test_clone_chain(self, clone_cleanup_pass):
        """Test handling of clone chains: x -> clone1 -> clone2."""

        def f(x: torch.Tensor) -> torch.Tensor:
            x = x.relu()  # not a graph param
            x1 = x.clone()
            x2 = x1.clone()
            return x2 + 1

        inp = torch.randn(2, 3)
        graph_module = make_fx(f)(inp)
        assert count_clones(graph_module.graph) == 2

        expected = graph_module(inp)
        clone_cleanup_pass(graph_module.graph)
        graph_module.recompile()
        actual = graph_module(inp)

        # Both clones should be removed
        assert count_clones(graph_module.graph) == 0
        torch.testing.assert_close(actual, expected)

    def test_multiple_clones_of_same_input(self, clone_cleanup_pass):
        """Test multiple independent clones of the same input."""

        def f(x: torch.Tensor) -> torch.Tensor:
            x1 = x.clone()
            x2 = x.clone()
            return x1 + x2

        inp = torch.randn(2, 3)
        graph_module = make_fx(f)(inp)
        assert count_clones(graph_module.graph) == 2

        expected = graph_module(inp)
        clone_cleanup_pass(graph_module.graph)
        graph_module.recompile()
        actual = graph_module(inp)

        # Both clones should be removed (only readonly uses)
        assert count_clones(graph_module.graph) == 0
        torch.testing.assert_close(actual, expected)

    def test_no_clones_in_graph(self, clone_cleanup_pass):
        """Test pass behavior when graph has no clones."""

        def f(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        inp = torch.randn(2, 3)
        graph_module = make_fx(f)(inp)
        assert count_clones(graph_module.graph) == 0

        expected = graph_module(inp)
        clone_cleanup_pass(graph_module.graph)
        graph_module.recompile()
        actual = graph_module(inp)

        assert count_clones(graph_module.graph) == 0
        torch.testing.assert_close(actual, expected)

    def test_multiple_passes(self, clone_cleanup_pass):
        """Test running the pass multiple times (should be idempotent)."""

        def f(x: torch.Tensor) -> torch.Tensor:
            x1 = x.clone()
            return x1 + 1

        inp = torch.randn(2, 3)
        graph_module = make_fx(f)(inp)
        assert count_clones(graph_module.graph) == 1

        expected = graph_module(inp)

        clone_cleanup_pass(graph_module.graph)
        assert count_clones(graph_module.graph) == 0
        graph_module.recompile()
        actual = graph_module(inp)
        torch.testing.assert_close(actual, expected)

        clone_cleanup_pass(graph_module.graph)
        assert count_clones(graph_module.graph) == 0
        graph_module.recompile()
        actual = graph_module(inp)
        torch.testing.assert_close(actual, expected)

    def test_output_node_no_write(self):
        """Output nodes never write to their inputs."""

        def f(x: torch.Tensor) -> torch.Tensor:
            return x

        graph_module = make_fx(f)(torch.randn(2, 3))
        x_node = [n for n in graph_module.graph.nodes if n.op == "placeholder"][0]
        output_node = [n for n in graph_module.graph.nodes if n.op == "output"][0]

        assert not user_writes_to_node(output_node, x_node)

    def test_readonly_op_no_write(self):
        """Readonly operations don't write to inputs."""

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        graph_module = make_fx(f)(torch.randn(2, 3), torch.randn(2, 3))
        placeholders = [n for n in graph_module.graph.nodes if n.op == "placeholder"]
        add_node = [
            n
            for n in graph_module.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.add.Tensor
        ][0]

        assert not user_writes_to_node(add_node, placeholders[0])
        assert not user_writes_to_node(add_node, placeholders[1])

    def test_inplace_op_writes(self):
        """Inplace operations write to first argument."""

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x.add_(y)
            return x

        graph_module = make_fx(f)(torch.randn(2, 3), torch.randn(2, 3))
        placeholders = [n for n in graph_module.graph.nodes if n.op == "placeholder"]
        add_node = [
            n
            for n in graph_module.graph.nodes
            if n.op == "call_function" and "add_" in str(n.target)
        ][0]

        # add_ writes to first arg but not second
        assert user_writes_to_node(add_node, placeholders[0])
        assert not user_writes_to_node(add_node, placeholders[1])

    def test_copy_writes(self):
        """copy_ operation writes to first argument."""

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x.copy_(y)
            return x

        graph_module = make_fx(f)(torch.randn(2, 3), torch.randn(2, 3))
        placeholders = [n for n in graph_module.graph.nodes if n.op == "placeholder"]
        copy_node = [
            n
            for n in graph_module.graph.nodes
            if n.op == "call_function" and "copy_" in str(n.target)
        ][0]

        assert user_writes_to_node(copy_node, placeholders[0])
        assert not user_writes_to_node(copy_node, placeholders[1])

    def test_auto_functionalized_not_a_write(self):
        """auto_functionalized ops are follow-up uses, not writes."""
        from torch._higher_order_ops.auto_functionalize import auto_functionalized

        def f(x: torch.Tensor) -> torch.Tensor:
            return x

        graph_module = make_fx(f)(torch.randn(2, 3))
        x_node = [n for n in graph_module.graph.nodes if n.op == "placeholder"][0]

        # Create an auto_functionalized node in the graph
        with graph_module.graph.inserting_before(None):
            af_node = graph_module.graph.call_function(
                auto_functionalized, kwargs={"input": x_node}
            )

        # auto_functionalized should not be treated as a write
        assert not user_writes_to_node(af_node, x_node)

    def test_higher_order_op_conservatively_writes(self):
        """Other higher-order operators are conservatively treated as writes."""
        from torch._ops import HigherOrderOperator

        def f(x: torch.Tensor) -> torch.Tensor:
            return x

        graph_module = make_fx(f)(torch.randn(2, 3))
        x_node = [n for n in graph_module.graph.nodes if n.op == "placeholder"][0]

        # Create a concrete higher-order operator subclass
        class MockHigherOrderOp(HigherOrderOperator):
            def __call__(self, *args, **kwargs):
                return args[0] if args else None

        mock_hoo = MockHigherOrderOp("mock_higher_order_op")

        with graph_module.graph.inserting_before(None):
            hoo_node = graph_module.graph.call_function(mock_hoo, args=(x_node,))

        # Should be conservative and assume it could write
        assert user_writes_to_node(hoo_node, x_node)


class TestCloneCleanupWithDonatedInputs:
    """Test UnsafeCloneEliminationPass with donated input tracking via PassContext."""

    @pytest.fixture(autouse=True)
    def setup_pass_context(self):
        """Set up pass context for each test."""
        with pass_context(compile_range=Range(1, 8192)):
            yield

    def test_donated_input_clone_removed(self, clone_cleanup_pass):
        """Clone of donated input should be removed."""

        def f(x: torch.Tensor) -> torch.Tensor:
            x_clone = x.clone()
            x_clone.add_(1)
            return x_clone

        inp = torch.randn(2, 3)
        graph_module = make_fx(f)(inp)
        assert count_clones(graph_module.graph) == 1

        # Mark first parameter as donated
        get_pass_context().donated_input_ids = {0}

        expected = graph_module(inp.clone())
        clone_cleanup_pass(graph_module.graph)
        graph_module.recompile()

        # Clone should be removed since input is donated
        assert count_clones(graph_module.graph) == 0

        # Input can be mutated (donated)
        inp_copy = inp.clone()
        actual = graph_module(inp_copy)
        torch.testing.assert_close(actual, expected)

    def test_non_donated_input_clone_kept(self, clone_cleanup_pass):
        """Clone of non-donated input with mutation should be kept."""

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x_clone = x.clone()
            x_clone.add_(1)
            return x, x_clone

        inp_x = torch.randn(2, 3)
        inp_y = torch.randn(2, 3)
        graph_module = make_fx(f)(inp_x, inp_y)
        assert count_clones(graph_module.graph) == 1

        # No donated inputs
        get_pass_context().donated_input_ids = set()

        expected = graph_module(inp_x.clone(), inp_y.clone())
        clone_cleanup_pass(graph_module.graph)
        graph_module.recompile()

        # Clone should be kept since input is not donated and original is used
        assert count_clones(graph_module.graph) == 1

        # Verify inputs are not mutated
        inp_x_before = inp_x.clone()
        inp_y_before = inp_y.clone()
        actual = graph_module(inp_x, inp_y)
        torch.testing.assert_close(
            inp_x, inp_x_before, msg="Input x should not be mutated"
        )
        torch.testing.assert_close(
            inp_y, inp_y_before, msg="Input y should not be mutated"
        )
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    def test_mixed_donated_inputs(self, clone_cleanup_pass):
        """Test with some inputs donated and some not."""

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x_clone = x.clone()
            x_clone.add_(1)
            y_clone = y.clone()
            y_clone.add_(2)
            return x_clone, y_clone

        inp_x = torch.randn(2, 3)
        inp_y = torch.randn(2, 3)
        graph_module = make_fx(f)(inp_x, inp_y)
        assert count_clones(graph_module.graph) == 2

        # Only x is donated
        get_pass_context().donated_input_ids = {0}

        expected = graph_module(inp_x.clone(), inp_y.clone())
        clone_cleanup_pass(graph_module.graph)
        graph_module.recompile()

        # x_clone removed (x is donated), y_clone kept (y is not donated)
        assert count_clones(graph_module.graph) == 1

        # Verify y is not mutated (x can be mutated since it's donated)
        inp_y_before = inp_y.clone()
        actual = graph_module(inp_x.clone(), inp_y)
        torch.testing.assert_close(
            inp_y, inp_y_before, msg="Input y should not be mutated"
        )
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
