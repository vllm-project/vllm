# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Output layout of the FLASH_ATTN MLA prefill backend."""

from unittest.mock import patch

import torch

from vllm.v1.attention.backends.mla.prefill.flash_attn import (
    FlashAttnPrefillBackend,
)

_FA_MODULE = "vllm.v1.attention.backends.mla.prefill.flash_attn"


def test_upstream_flash_attn_head_major_output_is_made_token_major():
    """Upstream flash_attn on ROCm may return a head-major output, while
    merge_attn_states derives each partial's token stride from its head
    stride. The backend must hand back a token-major partial."""
    num_tokens, num_heads, head_size = 5, 3, 8
    expected = torch.randn(num_tokens, num_heads, head_size)
    lse = torch.randn(num_heads, num_tokens)

    def head_major_kernel(**kwargs):
        assert kwargs["return_attn_probs"]
        return expected.transpose(0, 1).contiguous().transpose(0, 1), lse

    backend = object.__new__(FlashAttnPrefillBackend)
    backend.requires_v_padding = False
    backend._is_vllm_fa = False
    backend.flash_attn_varlen_func = None
    q = torch.empty(num_tokens, num_heads, head_size)
    with patch(f"{_FA_MODULE}._FA4_MLA_PREFILL_KERNEL", head_major_kernel):
        attn_out, attn_lse = backend._flash_attn_varlen_diff_headdims(
            q=q, k=q, v=q, return_softmax_lse=True
        )

    assert attn_out.is_contiguous()
    torch.testing.assert_close(attn_out, expected)
    assert attn_lse is lse
