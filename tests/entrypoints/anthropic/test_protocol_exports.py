# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for Anthropic protocol exports used by serving.

Guards against Docker/nightly images shipping a stale protocol module that is
missing symbols imported by ``vllm.entrypoints.anthropic.serving`` (issue #44759).
"""

import pytest

from vllm.entrypoints.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicContextManagement,
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicDelta,
    AnthropicError,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicOutputConfig,
    AnthropicStreamEvent,
    AnthropicUsage,
)

pytestmark = pytest.mark.skip_global_cleanup

SERVING_PROTOCOL_EXPORTS = (
    AnthropicContentBlock,
    AnthropicContextManagement,
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicDelta,
    AnthropicError,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicOutputConfig,
    AnthropicStreamEvent,
    AnthropicUsage,
)


def test_serving_protocol_exports_are_importable():
    for export in SERVING_PROTOCOL_EXPORTS:
        assert export is not None


def test_anthropic_output_config_instantiation():
    config = AnthropicOutputConfig()
    assert config.effort is None
    assert config.format is None
