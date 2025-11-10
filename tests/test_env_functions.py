# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for environment variable helper functions.

This module tests specific environment variable functions that have custom logic,
as opposed to test_env_utils.py which tests generic utility functions.
"""

import os
from unittest.mock import patch

import pytest

from vllm.envs import get_vllm_port, use_flashinfer_sampler
from vllm.platforms.interface import DeviceCapability


class TestGetVllmPort:
    """Test cases for get_vllm_port function."""

    def test_get_vllm_port_not_set(self):
        """Test when VLLM_PORT is not set."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_vllm_port() is None

    def test_get_vllm_port_valid(self):
        """Test when VLLM_PORT is set to a valid integer."""
        with patch.dict(os.environ, {"VLLM_PORT": "5678"}, clear=True):
            assert get_vllm_port() == 5678

    def test_get_vllm_port_invalid(self):
        """Test when VLLM_PORT is set to a non-integer value."""
        with (
            patch.dict(os.environ, {"VLLM_PORT": "abc"}, clear=True),
            pytest.raises(ValueError, match="must be a valid integer"),
        ):
            get_vllm_port()

    def test_get_vllm_port_uri(self):
        """Test when VLLM_PORT is set to a URI."""
        with (
            patch.dict(os.environ, {"VLLM_PORT": "tcp://localhost:5678"}, clear=True),
            pytest.raises(ValueError, match="appears to be a URI"),
        ):
            get_vllm_port()


class TestUseFlashinferSampler:
    """Test cases for use_flashinfer_sampler function."""

    def test_capability_below_sm75(self):
        with patch("vllm.platforms.current_platform") as mock_platform:
            mock_platform.get_device_capability.return_value = DeviceCapability(
                major=7, minor=0
            )
            with patch.dict(os.environ, {"VLLM_USE_FLASHINFER_SAMPLER": "1"}):
                assert use_flashinfer_sampler() is False
            with patch.dict(os.environ, {}, clear=True):
                assert use_flashinfer_sampler() is False

    def test_capability_equal_sm75(self):
        with patch("vllm.platforms.current_platform") as mock_platform:
            mock_platform.get_device_capability.return_value = DeviceCapability(
                major=7, minor=5
            )
            with patch.dict(os.environ, {"VLLM_USE_FLASHINFER_SAMPLER": "1"}):
                assert use_flashinfer_sampler() is True

    def test_capability_above_sm75(self):
        with patch("vllm.platforms.current_platform") as mock_platform:
            mock_platform.get_device_capability.return_value = DeviceCapability(
                major=8, minor=0
            )
            with patch.dict(os.environ, {"VLLM_USE_FLASHINFER_SAMPLER": "0"}):
                assert use_flashinfer_sampler() is False

    def test_capability_is_none(self):
        with patch("vllm.platforms.current_platform") as mock_platform:
            mock_platform.get_device_capability.return_value = None
            with patch.dict(os.environ, {}, clear=True):
                # When capability is None and env var is not set, should return None
                with patch.dict(os.environ, {}, clear=True):
                    assert use_flashinfer_sampler() is None
                # When capability is None, env var should be respected
                with patch.dict(os.environ, {"VLLM_USE_FLASHINFER_SAMPLER": "1"}):
                    assert use_flashinfer_sampler() is True
                with patch.dict(os.environ, {"VLLM_USE_FLASHINFER_SAMPLER": "0"}):
                    assert use_flashinfer_sampler() is False

    def test_env_var_not_set(self):
        with patch("vllm.platforms.current_platform") as mock_platform:
            mock_platform.get_device_capability.return_value = DeviceCapability(
                major=8, minor=0
            )
            with patch.dict(os.environ, {}, clear=True):
                assert use_flashinfer_sampler() is None
