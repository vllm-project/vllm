# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


def test_flash_attn_varlen_requires_cu_seqlens_k(monkeypatch):
    from vllm.attention.utils import fa_utils

    def _dummy_flash_attn(*_args, **_kwargs):
        return torch.empty(0)

    monkeypatch.setattr(fa_utils, "_flash_attn_varlen_func", _dummy_flash_attn)

    q = torch.empty((1, 1), dtype=torch.float16)
    k = torch.empty((1, 1), dtype=torch.float16)
    v = torch.empty((1, 1), dtype=torch.float16)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32)

    with pytest.raises(ValueError, match="cu_seqlens_k"):
        fa_utils.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=None,
            max_seqlen_q=1,
            max_seqlen_k=1,
        )
