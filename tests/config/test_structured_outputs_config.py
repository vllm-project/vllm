# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config.structured_outputs import StructuredOutputsConfig


class TestDisableAnyWhitespace:
    """Tests for disable_any_whitespace config validation."""

    def test_auto_backend_allows_disable_any_whitespace(self):
        """backend='auto' should accept disable_any_whitespace=True,
        since auto resolves to xgrammar or guidance at runtime."""
        config = StructuredOutputsConfig(
            backend="auto",
            disable_any_whitespace=True,
        )
        assert config.disable_any_whitespace is True
        assert config.backend == "auto"

    def test_xgrammar_backend_allows_disable_any_whitespace(self):
        config = StructuredOutputsConfig(
            backend="xgrammar",
            disable_any_whitespace=True,
        )
        assert config.disable_any_whitespace is True

    def test_guidance_backend_allows_disable_any_whitespace(self):
        config = StructuredOutputsConfig(
            backend="guidance",
            disable_any_whitespace=True,
        )
        assert config.disable_any_whitespace is True

    def test_outlines_backend_rejects_disable_any_whitespace(self):
        with pytest.raises(ValueError, match="disable_any_whitespace"):
            StructuredOutputsConfig(
                backend="outlines",
                disable_any_whitespace=True,
            )

    def test_lm_format_enforcer_rejects_disable_any_whitespace(self):
        with pytest.raises(ValueError, match="disable_any_whitespace"):
            StructuredOutputsConfig(
                backend="lm-format-enforcer",
                disable_any_whitespace=True,
            )

    def test_default_disable_any_whitespace_is_false(self):
        config = StructuredOutputsConfig()
        assert config.disable_any_whitespace is False
