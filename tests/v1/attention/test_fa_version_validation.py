# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An explicitly requested FlashAttention version that the device or build does
not support must fail at backend selection, not later in attention forward.

The device capability is mocked, so no GPU is needed.
"""

from unittest.mock import MagicMock, patch

import pytest

from vllm.platforms.interface import DeviceCapability

fa_interface = pytest.importorskip(
    "vllm.vllm_flash_attn.flash_attn_interface",
    reason="requires compiled CUDA flash-attention extensions",
    exc_type=ImportError,
)

from vllm.v1.attention.backends.fa_utils import get_flash_attn_version  # noqa: E402


def _select(requested):
    """Select a version on SM89, where only FA2 is supported."""
    platform = MagicMock()
    platform.is_xpu.return_value = False
    platform.is_rocm.return_value = False
    platform.get_device_capability.return_value = DeviceCapability(8, 9)
    config = MagicMock()
    config.attention_config.flash_attn_version = requested
    config.model_config.is_diffusion = False
    with (
        patch("vllm.v1.attention.backends.fa_utils.current_platform", platform),
        patch.object(
            fa_interface, "is_fa_version_supported", side_effect=lambda v: v == 2
        ),
        patch.object(
            fa_interface, "fa_version_unsupported_reason", return_value="needs SM90"
        ),
        patch("vllm.config.get_current_vllm_config_or_none", return_value=config),
    ):
        return get_flash_attn_version()


@pytest.mark.parametrize("requested", [3, 4])
def test_unsupported_requested_version_raises(requested):
    with pytest.raises(ValueError, match=f"flash_attn_version={requested}.*\\[2\\]"):
        _select(requested)


def test_automatic_selection_unchanged():
    assert _select(None) == 2
