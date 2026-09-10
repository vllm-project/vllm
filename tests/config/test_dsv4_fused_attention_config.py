# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.config.attention import AttentionConfig


def test_dsv4_fused_attention_defaults():
    cfg = AttentionConfig()
    assert cfg.dsv4_fused_attention is None
    assert cfg.dsv4_fused_decode_min_tokens >= 0


def test_dsv4_fused_decode_min_tokens_rejects_negative():
    with pytest.raises(ValueError, match="dsv4_fused_decode_min_tokens"):
        AttentionConfig(dsv4_fused_decode_min_tokens=-1)
