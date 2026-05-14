# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import CacheConfig
from vllm.platforms import current_platform
from vllm.v1.attention.backend import MultipleOf


def test_update_block_size_for_backend_rejects_unsupported_user_block_size(
    monkeypatch: pytest.MonkeyPatch,
):
    class Backend:
        @staticmethod
        def get_supported_kernel_block_sizes():
            return [MultipleOf(16)]

        @classmethod
        def supports_block_size(cls, block_size: int | None) -> bool:
            return block_size is not None and block_size % 16 == 0

        @staticmethod
        def get_name():
            return "TEST_BACKEND"

    vllm_config = SimpleNamespace(
        cache_config=CacheConfig(block_size=8),
        model_config=SimpleNamespace(is_hybrid=False),
    )
    monkeypatch.setattr(
        type(current_platform),
        "_find_non_ssm_backend",
        classmethod(lambda cls, vllm_config: Backend),
    )

    with pytest.raises(ValueError, match=r"block_size \(8\).*TEST_BACKEND"):
        current_platform.update_block_size_for_backend(vllm_config)
