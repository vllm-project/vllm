# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FP8 dynamic scaling in MMEncoderAttention."""

import contextlib

import pytest
import torch

from vllm.model_executor.layers.attention.mm_encoder_attention import (
    _FP8_AMAX_HISTORY_LEN,
    _FP8_MAX,
)
from vllm.utils.flashinfer import (
    is_flashinfer_cudnn_fp8_prefill_attn_supported,
)


@pytest.fixture
def _make_attention():
    """Create an MMEncoderAttention with dynamic FP8 scaling."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.multimodal import MultiModalConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if not is_flashinfer_cudnn_fp8_prefill_attn_supported():
        yield None
        return

    # Dynamic scaling is the default when no scale file is provided.
    mm_config = MultiModalConfig(mm_encoder_attn_dtype="fp8")
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(multimodal_config=mm_config)

    from vllm.model_executor.layers.attention.mm_encoder_attention import (
        MMEncoderAttention,
    )

    # Override backend to FlashInfer (platform supports it but may not
    # select it by default).
    attn = None
    with (
        set_current_vllm_config(vllm_config),
        contextlib.suppress(ValueError, ImportError),
        patch(
            "vllm.model_executor.layers.attention.mm_encoder_attention"
            ".get_vit_attn_backend",
            return_value=AttentionBackendEnum.FLASHINFER,
        ),
    ):
        attn = MMEncoderAttention(
            num_heads=16,
            head_size=80,
            prefix="visual.blocks.0.attn",
        )

    yield attn


def test_dynamic_scaling_updates_scales(_make_attention) -> None:
    """Verify that _record_amax_and_update_scales updates scale buffers."""
    attn = _make_attention
    if attn is None or not attn.fp8_enabled:
        pytest.skip("FP8 attention not available (FlashInfer backend required)")

    attn = attn.to("cuda")

    S, H, D = 32, 16, 80
    q = torch.full((S, H, D), 2.0, device="cuda", dtype=torch.bfloat16)
    k = torch.full((S, H, D), 3.0, device="cuda", dtype=torch.bfloat16)
    v = torch.full((S, H, D), 4.0, device="cuda", dtype=torch.bfloat16)

    attn._record_amax_and_update_scales(q, k, v)

    expected_q_scale = 2.0 / _FP8_MAX
    expected_k_scale = 3.0 / _FP8_MAX
    expected_v_scale = 4.0 / _FP8_MAX

    torch.testing.assert_close(
        attn._fp8_q_scale.item(), expected_q_scale, atol=1e-6, rtol=1e-4
    )
    torch.testing.assert_close(
        attn._fp8_k_scale.item(), expected_k_scale, atol=1e-6, rtol=1e-4
    )
    torch.testing.assert_close(
        attn._fp8_v_scale.item(), expected_v_scale, atol=1e-6, rtol=1e-4
    )


def test_circular_buffer_wraps(_make_attention) -> None:
    """Verify the amax circular buffer wraps at HISTORY_LEN."""
    attn = _make_attention
    if attn is None or not attn.fp8_enabled:
        pytest.skip("FP8 attention not available (FlashInfer backend required)")

    attn = attn.to("cuda")
    S, H, D = 16, 16, 80

    for i in range(_FP8_AMAX_HISTORY_LEN + 2):
        mag = float(i + 1)
        q = torch.full((S, H, D), mag, device="cuda", dtype=torch.bfloat16)
        k = torch.full((S, H, D), mag, device="cuda", dtype=torch.bfloat16)
        v = torch.full((S, H, D), mag, device="cuda", dtype=torch.bfloat16)
        attn._record_amax_and_update_scales(q, k, v)

    assert attn._fp8_amax_pos == 2

    expected_max = float(_FP8_AMAX_HISTORY_LEN + 2)
    expected_scale = expected_max / _FP8_MAX
    torch.testing.assert_close(
        attn._fp8_q_scale.item(), expected_scale, atol=1e-6, rtol=1e-4
    )
