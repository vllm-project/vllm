# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for get_num_decode_tokens utility."""

from types import SimpleNamespace
from unittest.mock import patch

import vllm.forward_context as fc_module
from vllm.forward_context import get_num_decode_tokens


class TestGetNumDecodeTokens:
    """Test the get_num_decode_tokens forward context utility."""

    def test_returns_default_when_context_unavailable(self):
        """No forward context set -> return default."""
        with patch.object(fc_module, "_forward_context", None):
            assert get_num_decode_tokens(42) == 42

    def test_returns_default_when_attn_metadata_is_none(self):
        ctx = SimpleNamespace(attn_metadata=None)
        with patch.object(fc_module, "_forward_context", ctx):
            assert get_num_decode_tokens(99) == 99

    def test_dict_layout_with_num_decode_tokens(self):
        """Standard v1 layout: dict[str, AttentionMetadata]."""
        layer_meta = SimpleNamespace(num_decode_tokens=7)
        ctx = SimpleNamespace(attn_metadata={"layer0": layer_meta})
        with patch.object(fc_module, "_forward_context", ctx):
            assert get_num_decode_tokens(100) == 7

    def test_dict_layout_without_num_decode_tokens_attr(self):
        """Backend metadata lacks the attribute -> return default."""
        layer_meta = SimpleNamespace()  # no num_decode_tokens
        ctx = SimpleNamespace(attn_metadata={"layer0": layer_meta})
        with patch.object(fc_module, "_forward_context", ctx):
            assert get_num_decode_tokens(50) == 50

    def test_list_layout_dbo(self):
        """DBO layout: list[dict[str, AttentionMetadata]]."""
        layer_meta = SimpleNamespace(num_decode_tokens=3)
        ctx = SimpleNamespace(attn_metadata=[{"layer0": layer_meta}])
        with patch.object(fc_module, "_forward_context", ctx):
            assert get_num_decode_tokens(100) == 3

    def test_empty_list_layout(self):
        """DBO layout with empty list -> return default."""
        ctx = SimpleNamespace(attn_metadata=[])
        with patch.object(fc_module, "_forward_context", ctx):
            assert get_num_decode_tokens(25) == 25

    def test_empty_dict_layout(self):
        """Empty dict metadata -> return default."""
        ctx = SimpleNamespace(attn_metadata={})
        with patch.object(fc_module, "_forward_context", ctx):
            assert get_num_decode_tokens(30) == 30

    def test_list_with_empty_first_dict(self):
        """DBO list where first microbatch dict is empty -> default."""
        ctx = SimpleNamespace(attn_metadata=[{}])
        with patch.object(fc_module, "_forward_context", ctx):
            assert get_num_decode_tokens(10) == 10

    def test_num_decode_tokens_zero(self):
        """Zero decode tokens is a valid value (all-prefill batch)."""
        layer_meta = SimpleNamespace(num_decode_tokens=0)
        ctx = SimpleNamespace(attn_metadata={"layer0": layer_meta})
        with patch.object(fc_module, "_forward_context", ctx):
            assert get_num_decode_tokens(100) == 0

    def test_multiple_layers_returns_first(self):
        """With multiple layers, returns the first one encountered."""
        meta_a = SimpleNamespace(num_decode_tokens=5)
        meta_b = SimpleNamespace(num_decode_tokens=5)
        ctx = SimpleNamespace(attn_metadata={"layer_a": meta_a, "layer_b": meta_b})
        with patch.object(fc_module, "_forward_context", ctx):
            # All layers in a batch have the same num_decode_tokens
            assert get_num_decode_tokens(100) == 5
