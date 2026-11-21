# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms.xpu import XPUPlatform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig


def _config(**overrides) -> AttentionSelectorConfig:
    fields = dict(
        head_size=128, dtype=torch.bfloat16, kv_cache_dtype=None, block_size=16
    )
    return AttentionSelectorConfig(**{**fields, **overrides})


@pytest.mark.parametrize(
    "overrides", [{}, {"use_batch_invariant": True}, {"dtype": torch.float32}]
)
def test_xpu_honors_explicit_flex_attention(overrides):
    path = XPUPlatform.get_attn_backend_cls(
        selected_backend=AttentionBackendEnum.FLEX_ATTENTION,
        attn_selector_config=_config(**overrides),
    )
    assert path == AttentionBackendEnum.FLEX_ATTENTION.get_path()


def test_xpu_still_rejects_unsupported_explicit_backend():
    with pytest.raises(ValueError, match="Invalid attention backend for xpu"):
        XPUPlatform.get_attn_backend_cls(
            selected_backend=AttentionBackendEnum.FLASHINFER,
            attn_selector_config=_config(),
        )
