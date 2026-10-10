# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the VLLM_FORCE_ATTN_BACKEND auto-selection pin.

Components that do not plumb through the user's --attention-backend (the
spec-decode draft model in particular) reach get_attn_backend_cls with
selected_backend=None and auto-select. The env pins that path too; a forced
backend that is invalid for a component's configuration (or unknown) falls
back to auto-selection instead of failing the component.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.platforms import current_platform
from vllm.platforms.cuda import CudaPlatform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.selector import AttentionSelectorConfig

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-specific tests"
)

SELECTOR_CONFIG = AttentionSelectorConfig(
    head_size=64,
    dtype=torch.float16,
    kv_cache_dtype=None,
    block_size=16,
)

SM90 = DeviceCapability(major=9, minor=0)


class _AutoSelectionReached(Exception):
    """Sentinel: get_valid_backends (the auto-selection path) was entered."""


def _get_backend_cls():
    return CudaPlatform.get_attn_backend_cls(
        selected_backend=None,
        attn_selector_config=SELECTOR_CONFIG,
        num_heads=32,
    )


def test_forced_backend_bypasses_auto_selection(monkeypatch):
    monkeypatch.setenv("VLLM_FORCE_ATTN_BACKEND", "TRITON_ATTN")
    healthy = MagicMock()
    healthy.validate_configuration.return_value = []
    with (
        patch.object(CudaPlatform, "get_device_capability", return_value=SM90),
        patch("vllm.platforms.cuda._get_attn_backend_class", return_value=healthy),
        patch("vllm.platforms.cuda._backend_cls_path", return_value="forced.path"),
        patch.object(
            CudaPlatform, "get_valid_backends", side_effect=_AutoSelectionReached
        ),
    ):
        assert _get_backend_cls() == "forced.path"


def test_invalid_forced_backend_falls_back_to_auto_selection(monkeypatch):
    monkeypatch.setenv("VLLM_FORCE_ATTN_BACKEND", "TRITON_ATTN")
    unfit = MagicMock()
    unfit.validate_configuration.return_value = ["head_size not supported"]
    with (
        patch.object(CudaPlatform, "get_device_capability", return_value=SM90),
        patch("vllm.platforms.cuda._get_attn_backend_class", return_value=unfit),
        patch.object(
            CudaPlatform, "get_valid_backends", side_effect=_AutoSelectionReached
        ),
        pytest.raises(_AutoSelectionReached),
    ):
        _get_backend_cls()


def test_unknown_forced_backend_falls_back_to_auto_selection(monkeypatch):
    monkeypatch.setenv("VLLM_FORCE_ATTN_BACKEND", "NO_SUCH_BACKEND")
    with (
        patch.object(CudaPlatform, "get_device_capability", return_value=SM90),
        patch.object(
            CudaPlatform, "get_valid_backends", side_effect=_AutoSelectionReached
        ),
        pytest.raises(_AutoSelectionReached),
    ):
        _get_backend_cls()


def test_no_env_leaves_auto_selection_unchanged(monkeypatch):
    monkeypatch.delenv("VLLM_FORCE_ATTN_BACKEND", raising=False)
    with (
        patch.object(CudaPlatform, "get_device_capability", return_value=SM90),
        patch.object(
            CudaPlatform, "get_valid_backends", side_effect=_AutoSelectionReached
        ),
        pytest.raises(_AutoSelectionReached),
    ):
        _get_backend_cls()


def test_explicit_selected_backend_takes_precedence(monkeypatch):
    """A component that DOES receive an explicit backend is unaffected by the
    env: the selected_backend branch returns before the forced-env check."""
    monkeypatch.setenv("VLLM_FORCE_ATTN_BACKEND", "TRITON_ATTN")
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    healthy = MagicMock()
    healthy.validate_configuration.return_value = []
    with (
        patch.object(CudaPlatform, "get_device_capability", return_value=SM90),
        patch("vllm.platforms.cuda._get_attn_backend_class", return_value=healthy),
        patch("vllm.platforms.cuda._backend_cls_path", return_value="explicit.path"),
    ):
        result = CudaPlatform.get_attn_backend_cls(
            selected_backend=AttentionBackendEnum.FLASH_ATTN,
            attn_selector_config=SELECTOR_CONFIG,
            num_heads=32,
        )
    assert result == "explicit.path"
