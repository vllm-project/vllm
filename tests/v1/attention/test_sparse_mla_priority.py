# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests pinning the sparse-MLA backend priority order on Blackwell.

FLASH_ATTN_MLA_SPARSE_FA4 is the first sparse candidate for a bf16 KV cache at
the head counts it was measured on; the fp8 KV list must stay untouched. Pure
priority-list logic, so no GPU is required.
"""

from vllm.platforms.cuda import _get_backend_priorities
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum

SM100 = DeviceCapability(major=10, minor=0)

FA4 = AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE_FA4
FLASHINFER_SPARSE = AttentionBackendEnum.FLASHINFER_MLA_SPARSE
FLASHMLA_SPARSE = AttentionBackendEnum.FLASHMLA_SPARSE

SPARSE_BACKENDS = (FA4, FLASHINFER_SPARSE, FLASHMLA_SPARSE)


def _sparse_tail(priorities: list[AttentionBackendEnum]) -> list[AttentionBackendEnum]:
    """The trailing sparse-MLA candidates, in priority order."""
    return [b for b in priorities if b in SPARSE_BACKENDS]


def test_bf16_low_head_count_defaults_to_fa4():
    priorities = _get_backend_priorities(
        use_mla=True,
        device_capability=SM100,
        num_heads=16,
        kv_cache_dtype="auto",
    )
    assert _sparse_tail(priorities) == [FA4, FLASHINFER_SPARSE, FLASHMLA_SPARSE]


def test_bf16_high_head_count_unchanged():
    """Above 16 heads FA4 stays opt-in: FlashMLA's masked-MHA rows are tuned."""
    priorities = _get_backend_priorities(
        use_mla=True,
        device_capability=SM100,
        num_heads=32,
        kv_cache_dtype="auto",
    )
    assert _sparse_tail(priorities) == [FLASHMLA_SPARSE, FLASHINFER_SPARSE]


def test_fp8_kv_cache_excludes_fa4():
    priorities = _get_backend_priorities(
        use_mla=True,
        device_capability=SM100,
        num_heads=16,
        kv_cache_dtype="fp8_ds_mla",
    )
    assert _sparse_tail(priorities) == [FLASHINFER_SPARSE, FLASHMLA_SPARSE]
