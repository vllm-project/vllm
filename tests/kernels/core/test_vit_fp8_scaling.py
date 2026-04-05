# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FP8 scaling (dynamic and static) in MMEncoderAttention."""

import contextlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.attention.mm_encoder_attention import (
    _FP8_AMAX_HISTORY_LEN,
    _FP8_MAX,
)
from vllm.utils.flashinfer import (
    is_flashinfer_cudnn_fp8_prefill_attn_supported,
)

LAYER_0 = "visual.blocks.0.attn"
LAYER_1 = "visual.blocks.1.attn"
NUM_HEADS = 16
HEAD_DIM = 72


def _build_attention(mm_config):
    """Build an MMEncoderAttention with the given multimodal config.

    Returns None if FlashInfer cuDNN is not available.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.attention.mm_encoder_attention import (
        MMEncoderAttention,
    )
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if not is_flashinfer_cudnn_fp8_prefill_attn_supported():
        return None

    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(multimodal_config=mm_config)

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
            num_heads=NUM_HEADS,
            head_size=HEAD_DIM,
            prefix=LAYER_0,
        )
    return attn


@pytest.fixture
def _make_attention():
    """Create an MMEncoderAttention with dynamic FP8 scaling."""
    from vllm.config.multimodal import MultiModalConfig

    yield _build_attention(MultiModalConfig(mm_encoder_attn_dtype="fp8"))


@pytest.fixture
def _make_static_attention(tmp_path):
    """Create an MMEncoderAttention with static FP8 scales from a file."""
    from vllm.config.multimodal import MultiModalConfig

    scale_file = tmp_path / "scales.json"
    scale_file.write_text(
        json.dumps(
            {
                LAYER_0: {"q": 224.0, "k": 198.0, "v": 210.0},
                LAYER_1: {"q": 100.0, "k": 110.0, "v": 120.0},
            }
        )
    )
    yield _build_attention(
        MultiModalConfig(
            mm_encoder_attn_dtype="fp8",
            mm_encoder_fp8_scale_path=str(scale_file),
        )
    )


def test_dynamic_scaling_updates_scales(_make_attention) -> None:
    """Verify that _record_amax_and_update_scales updates scale buffers."""
    attn = _make_attention
    if attn is None or not attn.fp8_enabled:
        pytest.skip("FP8 attention not available (FlashInfer backend required)")

    attn = attn.to("cuda")

    S, H, D = 32, NUM_HEADS, HEAD_DIM
    q = torch.full((S, H, D), 2.0, device="cuda", dtype=torch.bfloat16)
    k = torch.full((S, H, D), 3.0, device="cuda", dtype=torch.bfloat16)
    v = torch.full((S, H, D), 4.0, device="cuda", dtype=torch.bfloat16)

    attn._record_amax_and_update_scales(q, k, v)

    expected_q_scale = 2.0 / _FP8_MAX
    expected_k_scale = 3.0 / _FP8_MAX
    expected_v_scale = 4.0 / _FP8_MAX

    torch.testing.assert_close(attn._fp8_q_scale.item(), expected_q_scale)
    torch.testing.assert_close(attn._fp8_k_scale.item(), expected_k_scale)
    torch.testing.assert_close(attn._fp8_v_scale.item(), expected_v_scale)


def test_circular_buffer_wraps(_make_attention) -> None:
    """Verify the amax circular buffer wraps at HISTORY_LEN."""
    attn = _make_attention
    if attn is None or not attn.fp8_enabled:
        pytest.skip("FP8 attention not available (FlashInfer backend required)")

    attn = attn.to("cuda")
    S, H, D = 16, NUM_HEADS, HEAD_DIM

    for i in range(_FP8_AMAX_HISTORY_LEN + 2):
        mag = float(i + 1)
        q = torch.full((S, H, D), mag, device="cuda", dtype=torch.bfloat16)
        k = torch.full((S, H, D), mag, device="cuda", dtype=torch.bfloat16)
        v = torch.full((S, H, D), mag, device="cuda", dtype=torch.bfloat16)
        attn._record_amax_and_update_scales(q, k, v)

    assert attn._fp8_amax_pos == 2

    expected_max = float(_FP8_AMAX_HISTORY_LEN + 2)
    expected_scale = expected_max / _FP8_MAX
    torch.testing.assert_close(attn._fp8_q_scale.item(), expected_scale)


def test_static_scales_loaded(_make_static_attention) -> None:
    """Verify static scales are loaded from the JSON file."""
    attn = _make_static_attention
    if attn is None or not attn.fp8_enabled:
        pytest.skip("FP8 attention not available (FlashInfer backend required)")

    assert attn.fp8_enabled
    assert not attn._fp8_dynamic_scale

    # Layer 0 scales (the layer this attention was created with).
    assert attn._fp8_q_scale.item() == 224.0
    assert attn._fp8_k_scale.item() == 198.0
    assert attn._fp8_v_scale.item() == 210.0

    assert not attn.skip_scale_q
    assert not attn.skip_scale_k
    assert not attn.skip_scale_v

    # No amax history buffers for static scaling.
    assert not hasattr(attn, "_fp8_q_amax")


def test_static_scales_missing_layer(tmp_path) -> None:
    """Verify error when requested layer is not in the scale file."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.multimodal import MultiModalConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if not is_flashinfer_cudnn_fp8_prefill_attn_supported():
        pytest.skip("FlashInfer cuDNN not available")

    scale_file = tmp_path / "wrong_layer.json"
    scale_file.write_text(
        json.dumps({"visual.blocks.99.attn": {"q": 1.0, "k": 1.0, "v": 1.0}})
    )
    mm_config = MultiModalConfig(
        mm_encoder_attn_dtype="fp8",
        mm_encoder_fp8_scale_path=str(scale_file),
    )
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(multimodal_config=mm_config)

    from vllm.model_executor.layers.attention.mm_encoder_attention import (
        MMEncoderAttention,
    )

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.model_executor.layers.attention.mm_encoder_attention"
            ".get_vit_attn_backend",
            return_value=AttentionBackendEnum.FLASHINFER,
        ),
        pytest.raises(ValueError, match="scales not found for layer"),
    ):
        MMEncoderAttention(
            num_heads=NUM_HEADS,
            head_size=HEAD_DIM,
            prefix=LAYER_0,
        )
