# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper


def test_h12_query_is_zero_padded_to_h16():
    q = torch.arange(2 * 12 * 4, dtype=torch.bfloat16).view(2, 12, 4)

    padded_q = AiterMLAHelper.get_mla_padded_q(12, q)

    assert padded_q.shape == (2, 16, 4)
    torch.testing.assert_close(padded_q[:, :12], q)
    assert torch.count_nonzero(padded_q[:, 12:]) == 0


def test_h12_output_discards_padding_heads():
    o = torch.arange(2 * 16 * 4, dtype=torch.bfloat16).view(2, 16, 4)

    unpadded_o = AiterMLAHelper.get_mla_unpadded_o(12, o)

    assert unpadded_o.shape == (2, 12, 4)
    torch.testing.assert_close(unpadded_o, o[:, :12])


def test_existing_divisor_head_mapping_is_unchanged():
    q = torch.arange(2 * 8 * 4, dtype=torch.bfloat16).view(2, 8, 4)

    padded_q = AiterMLAHelper.get_mla_padded_q(8, q)
    unpadded_o = AiterMLAHelper.get_mla_unpadded_o(8, padded_q)

    torch.testing.assert_close(padded_q, q.repeat_interleave(2, dim=1))
    torch.testing.assert_close(unpadded_o, q)


def test_h12_is_the_only_non_divisor_below_h16_supported():
    assert AiterMLAHelper.is_valid_num_heads(12)
    assert not AiterMLAHelper.is_valid_num_heads(10)
