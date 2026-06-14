# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch.fx._utils import lazy_format_graph_code as original_lazy_format_graph_code
from torch._logging._internal import LazyString

from vllm.compilation.passes.vllm_inductor_pass import format_graph_code


def _create_simple_graphmodule() -> torch.fx.GraphModule:
    """Create a simple GraphModule for testing."""

    class SimpleModel(torch.nn.Module):
        def forward(self, x):
            x = x + 1
            x = x.reshape(-1, 128)
            y = torch.sum(x, dim=-1)
            return torch.sigmoid(y)

    model = SimpleModel()
    return torch.fx.symbolic_trace(model)


class TestFormatGraphCodeIsLazy:
    """Tests for lazy evaluation behavior."""

    def test_returns_lazystring(self):
        """Verify format_graph_code returns a LazyString."""
        gm = _create_simple_graphmodule()
        result = format_graph_code("test", gm)
        assert isinstance(result, LazyString), (
            f"Expected LazyString return type, got {type(result)}"
        )

    def test_same_return_type_as_original(self):
        """Verify format_graph_code returns the same type as the original."""
        gm = _create_simple_graphmodule()
        original_result = original_lazy_format_graph_code("test", gm)
        new_result = format_graph_code("test", gm)
        assert type(original_result) == type(new_result), (
            f"Return types differ: original={type(original_result)}, "
            f"new={type(new_result)}"
        )


class TestFormatGraphCodeValidPython:
    """Tests for valid Python code output."""

    def test_formatted_output_compiles(self):
        """Verify formatted output is valid Python code."""
        gm = _create_simple_graphmodule()
        formatted = str(format_graph_code("test", gm))

        # Should compile without raising SyntaxError
        compile(formatted, "<string>", "exec")

    def test_original_output_invalid_python(self):
        """Verify original output is NOT valid Python code.

        This ensures our formatting actually solves the problem,
        rather than the original output already being valid Python.
        """
        gm = _create_simple_graphmodule()
        original_output = str(original_lazy_format_graph_code("test", gm))

        # Original version should contain header lines (non-Python syntax)
        assert "=====" in original_output, (
            "Expected header lines in original output"
        )

        # Original version should fail to compile
        with pytest.raises(SyntaxError):
            compile(original_output, "<string>", "exec")

    def test_header_lines_become_comments(self):
        """Verify title lines like '==== title ====' become comments."""
        gm = _create_simple_graphmodule()
        formatted = str(format_graph_code("test", gm))

        # No bare '=====' lines should remain (they should be comments)
        for line in formatted.split("\n"):
            stripped = line.strip()
            if "=====" in stripped and not stripped.startswith("#"):
                pytest.fail(
                    f"Header line not converted to comment: {stripped}"
                )

    def test_traced_graph_becomes_comment(self):
        """Verify 'TRACED GRAPH' header becomes a comment."""
        gm = _create_simple_graphmodule()
        formatted = str(format_graph_code("test", gm))

        # "TRACED GRAPH" should only appear as a comment
        for line in formatted.split("\n"):
            if "TRACED GRAPH" in line:
                assert line.strip().startswith("#"), (
                    f"'TRACED GRAPH' not converted to comment: {line}"
                )

    def test_eval_with_key_becomes_comment(self):
        """Verify '<eval_with_key>...' lines become comments."""
        gm = _create_simple_graphmodule()
        formatted = str(format_graph_code("test", gm))

        # "<eval_with_key>" should only appear as a comment
        for line in formatted.split("\n"):
            if "<eval_with_key>" in line:
                assert line.strip().startswith("#"), (
                    f"'<eval_with_key>' not converted to comment: {line}"
                )