# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from vllm.platforms.interface import DeviceCapability


@pytest.fixture
def mock_cuda_platform():
    """
    Fixture that returns a factory for creating mocked CUDA platforms.

    Usage:
        def test_something(mock_cuda_platform):
            with mock_cuda_platform(is_cuda=True, capability=(9, 0)):
                # test code
    """

    @contextmanager
    def _mock_platform(is_cuda: bool = True, capability: tuple[int, int] | None = None):
        mock_platform = MagicMock()
        mock_platform.is_cuda.return_value = is_cuda
        if capability is not None:
            mock_platform.get_device_capability.return_value = DeviceCapability(
                *capability
            )
        with patch("vllm.platforms.current_platform", mock_platform):
            yield mock_platform

    return _mock_platform
