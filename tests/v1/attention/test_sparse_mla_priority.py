# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sparse-MLA backend priority order on Blackwell. No GPU required."""

import pytest

from vllm.platforms.cuda import _get_backend_priorities
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum

FA4 = AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE_FA4
FLASHINFER = AttentionBackendEnum.FLASHINFER_MLA_SPARSE
FLASHMLA = AttentionBackendEnum.FLASHMLA_SPARSE


@pytest.mark.parametrize(
    "num_heads,kv_cache_dtype,expected",
    [
        (16, "auto", [FA4, FLASHINFER, FLASHMLA]),
        (32, "auto", [FLASHMLA, FLASHINFER]),
        (16, "fp8_ds_mla", [FLASHINFER, FLASHMLA]),
    ],
)
def test_sparse_backend_priority(num_heads, kv_cache_dtype, expected):
    priorities = _get_backend_priorities(
        use_mla=True,
        device_capability=DeviceCapability(major=10, minor=0),
        num_heads=num_heads,
        kv_cache_dtype=kv_cache_dtype,
    )

    assert [b for b in priorities if b in (FA4, FLASHINFER, FLASHMLA)] == expected
