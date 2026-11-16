# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the TurboQuant hybrid startup failure in #62558.

With mamba cache alignment, a hybrid model can resolve a manager block of
1712 tokens (16 * 107). The TQ backend used to declare only the exact sizes
16/32/64/128, so select_common_block_size picked 16 and tried to split the
manager block into 107 kernel blocks. create_kv_cache_views refuses to split
padded hybrid pages, which made engine startup fail.
"""

import pytest

from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionBackend
from vllm.v1.worker.utils import select_common_block_size


class _ExactSizeBackend:
    """A backend that accepts exactly one kernel block size."""

    @staticmethod
    def get_name() -> str:
        return "EXACT"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        return [64]


def test_turboquant_declares_multiple_of_16():
    sizes = TurboQuantAttentionBackend.get_supported_kernel_block_sizes()
    assert len(sizes) == 1
    assert isinstance(sizes[0], MultipleOf)
    assert sizes[0].base == 16


@pytest.mark.parametrize("manager_block_size", [16, 32, 64, 128, 256, 1712])
def test_turboquant_manager_block_is_used_unsplit(manager_block_size):
    """The chosen kernel size must be the manager block itself (ratio == 1)."""
    selected = select_common_block_size(
        manager_block_size, [TurboQuantAttentionBackend]
    )
    assert selected == manager_block_size
    assert manager_block_size // selected == 1


def test_turboquant_1712_is_the_reported_failure_case():
    # 1712 = 16 * 107 and 107 is prime, so no power-of-two size above 16
    # divides it. The old declaration forced a 107-way kernel block split.
    assert 1712 % 16 == 0
    assert 1712 % 32 != 0
    assert 1712 % 64 != 0
    assert 1712 % 128 != 0
    assert select_common_block_size(1712, [TurboQuantAttentionBackend]) == 1712


def test_turboquant_still_honors_an_exact_size_backend():
    """A mixed group must still fall back to a common exact divisor."""
    assert (
        select_common_block_size(128, [TurboQuantAttentionBackend, _ExactSizeBackend])
        == 64
    )


def test_turboquant_still_rejects_non_multiple_of_16():
    with pytest.raises(ValueError, match="No common block size"):
        select_common_block_size(24, [TurboQuantAttentionBackend])


def test_turboquant_supports_block_size_is_unchanged():
    # The relaxation must not change which manager blocks the backend
    # accepts: 16 is already the smallest declared divisor.
    assert TurboQuantAttentionBackend.supports_block_size(16)
    assert TurboQuantAttentionBackend.supports_block_size(64)
    assert TurboQuantAttentionBackend.supports_block_size(1712)
    assert not TurboQuantAttentionBackend.supports_block_size(24)
    assert not TurboQuantAttentionBackend.supports_block_size(17)
