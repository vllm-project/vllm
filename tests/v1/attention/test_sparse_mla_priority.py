# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sparse-MLA backend priority order on Blackwell. No GPU required."""

from unittest.mock import patch

import pytest
import torch

from vllm.platforms.cuda import CudaPlatform, _get_backend_priorities
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig

FA4 = AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE_FA4
FLASHINFER = AttentionBackendEnum.FLASHINFER_MLA_SPARSE
FLASHMLA = AttentionBackendEnum.FLASHMLA_SPARSE


@pytest.mark.parametrize(
    "num_heads,kv_cache_dtype,use_hisparse,expected",
    [
        (16, "auto", False, [FA4, FLASHINFER, FLASHMLA]),
        (16, "auto", True, [FLASHINFER, FA4, FLASHMLA]),
        (32, "auto", False, [FLASHMLA, FLASHINFER]),
        (32, "auto", True, [FLASHMLA, FLASHINFER]),
        (16, "fp8_ds_mla", False, [FLASHINFER, FLASHMLA]),
        (16, "fp8_ds_mla", True, [FLASHINFER, FLASHMLA]),
    ],
)
def test_sparse_backend_priority(num_heads, kv_cache_dtype, use_hisparse, expected):
    priorities = _get_backend_priorities(
        use_mla=True,
        device_capability=DeviceCapability(major=10, minor=0),
        num_heads=num_heads,
        kv_cache_dtype=kv_cache_dtype,
        use_hisparse=use_hisparse,
    )

    assert [b for b in priorities if b in (FA4, FLASHINFER, FLASHMLA)] == expected


@pytest.mark.parametrize("use_hisparse", [False, True])
def test_get_valid_backends_forwards_use_hisparse(use_hisparse):
    config = AttentionSelectorConfig(
        head_size=576,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=None,
        use_mla=True,
        use_sparse=True,
        use_hisparse=use_hisparse,
    )
    with patch(
        "vllm.platforms.cuda._get_backend_priorities", return_value=[]
    ) as priorities:
        CudaPlatform.get_valid_backends(
            device_capability=DeviceCapability(major=10, minor=0),
            attn_selector_config=config,
            num_heads=16,
        )
    assert priorities.call_args.kwargs["use_hisparse"] is use_hisparse
