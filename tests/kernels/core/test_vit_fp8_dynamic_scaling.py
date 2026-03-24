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


def _flashinfer_fp8_supported() -> bool:
    """Check if FlashInfer cuDNN FP8 is supported on this platform."""
    try:
        from vllm.platforms import current_platform
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        supported = current_platform.get_supported_vit_attn_backends()
        if AttentionBackendEnum.FLASHINFER not in supported:
            return False
    except (ImportError, AttributeError):
        return False

    # cuDNN FP8 requires >= 9.17.1
    try:
        import torch.backends.cudnn as cudnn

        if cudnn.is_available():
            ver = cudnn.version()
            if ver < 91701:
                return False
    except (ImportError, AttributeError):
        pass

    return True


@pytest.fixture
def _make_attention(monkeypatch, default_vllm_config):
    """Create an MMEncoderAttention with dynamic FP8 scaling."""
    from unittest.mock import patch

    from vllm.envs import disable_envs_cache
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if not _flashinfer_fp8_supported():
        yield None
        return

    monkeypatch.setenv("VLLM_MM_ENCODER_FP8_ATTN", "1")
    monkeypatch.delenv("VLLM_MM_ENCODER_FP8_ATTN_SCALE_PATH", raising=False)
    disable_envs_cache()

    from vllm.model_executor.layers.attention.mm_encoder_attention import (
        MMEncoderAttention,
    )

    # Override backend to FlashInfer (platform supports it but may not
    # select it by default).
    attn = None
    with (
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

    disable_envs_cache()


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
