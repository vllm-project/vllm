# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from tests.kernels.utils import override_backend_env_variable
from vllm.attention.selector import _cached_get_attn_backend, get_attn_backend
from vllm.platforms.rocm import RocmPlatform


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear lru cache to ensure each test case runs without caching.
    """
    _cached_get_attn_backend.cache_clear()


def test_selector(monkeypatch):
    """Test that the attention selector for ROCm.
    """
    override_backend_env_variable(monkeypatch, "ROCM_FLASH")

    with patch("vllm.attention.selector.current_platform", RocmPlatform()):
        backend = get_attn_backend(16, torch.float16, torch.float16, 16, False)
        assert backend.get_name() == "ROCM_FLASH"
        # mla test for deepseek related
        backend = get_attn_backend(576, torch.bfloat16, "auto", 16, False,
                                   False, True)
        assert backend.get_name() == "TRITON_MLA"
