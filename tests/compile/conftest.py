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
        device_capability = (
            DeviceCapability(*capability) if capability is not None else None
        )
        mock_platform.get_device_capability.return_value = device_capability

        def is_device_capability_family(
            requested_capability: int, device_id: int = 0
        ) -> bool:
            current_capability = mock_platform.get_device_capability(
                device_id=device_id
            )
            if current_capability is None:
                return False
            return current_capability.major == (requested_capability // 10)

        mock_platform.is_device_capability_family.side_effect = (
            is_device_capability_family
        )
        with patch("vllm.platforms.current_platform", mock_platform):
            yield mock_platform

    return _mock_platform
