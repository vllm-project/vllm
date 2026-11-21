# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer whose kernel block size is fixed by the model's other layers.

Every layer of a model shares one kernel block size (select_common_block_size),
so a backend that only splits the KV-manager block into smaller kernel blocks
is not usable next to layers that support a single, larger one."""

import pytest
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionBackend, MultipleOf
from vllm.v1.attention.selector import AttentionSelectorConfig

pytestmark = pytest.mark.cpu_test


class _FixedSizes(AttentionBackend):
    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        return [16, 32]


class _AnyMultipleOf16(AttentionBackend):
    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        return [MultipleOf(16)]


class _Unconstrained(AttentionBackend):
    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        return []


@pytest.mark.parametrize(
    "backend, size, supported",
    [
        (_FixedSizes, 32, True),
        (_FixedSizes, 128, False),  # 128 splits into 32s, but is not a 32
        (_AnyMultipleOf16, 128, True),
        (_AnyMultipleOf16, 136, False),
        (_Unconstrained, 128, True),
    ],
)
def test_supports_kernel_block_size_is_exact(backend, size, supported):
    assert backend.supports_kernel_block_size(size) is supported


def test_supports_block_size_still_accepts_a_splittable_manager_block():
    # The KV-manager block check is unchanged: 128 = 4 kernel blocks of 32.
    assert _FixedSizes.supports_block_size(128) is True


def _reasons(backend, required):
    return backend.validate_configuration(
        **AttentionSelectorConfig(
            head_size=128,
            dtype=torch.bfloat16,
            kv_cache_dtype="auto",
            block_size=128,
            required_kernel_block_size=required,
        )._asdict(),
        device_capability=DeviceCapability(9, 4),
    )


def test_validate_configuration_rejects_an_unsupported_required_size():
    assert "kernel block size 128 not supported" in _reasons(_FixedSizes, 128)
    assert not any("kernel block size" in r for r in _reasons(_FixedSizes, None))
    assert not any("kernel block size" in r for r in _reasons(_AnyMultipleOf16, 128))


def test_selector_config_defaults_to_no_constraint():
    config = AttentionSelectorConfig(
        head_size=128, dtype=torch.bfloat16, kv_cache_dtype=None, block_size=None
    )
    assert config.required_kernel_block_size is None
    assert "required_kernel_block_size=None" in repr(config)
