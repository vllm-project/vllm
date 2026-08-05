# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import types

import pytest
import torch
import torch.nn.functional as F

from vllm._aiter_ops import is_aiter_found
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
    AiterMLADecodeMetadata,
    AiterMLAHelper,
    AiterMLAImpl,
    AiterMLAMetadata,
)

NUM_HEADS = 12
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
CONTEXT_LEN = 4096
SCALE = 1.0 / math.sqrt(QK_HEAD_DIM)


def _on_gfx950() -> bool:
    if not (current_platform.is_rocm() and is_aiter_found()):
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


def _make_h12_decode_metadata(device: torch.device) -> AiterMLAMetadata:
    page_indices = torch.arange(CONTEXT_LEN, dtype=torch.int32, device=device)
    decode = AiterMLADecodeMetadata(
        block_table=page_indices.view(1, -1),
        seq_lens=torch.tensor([CONTEXT_LEN], dtype=torch.int32, device=device),
        dcp_tot_seq_lens=None,
        paged_kv_indptr=torch.tensor(
            [0, CONTEXT_LEN], dtype=torch.int32, device=device
        ),
        paged_kv_indices=page_indices,
        paged_kv_last_page_len=torch.ones(1, dtype=torch.int32, device=device),
        qo_indptr=torch.tensor([0, 1], dtype=torch.int32, device=device),
        attn_out_dtype=torch.bfloat16,
        max_qo_len=1,
    )
    return AiterMLAMetadata(
        num_reqs=1,
        max_query_len=1,
        max_seq_len=CONTEXT_LEN,
        num_actual_tokens=1,
        query_start_loc=decode.qo_indptr,
        slot_mapping=torch.tensor([CONTEXT_LEN - 1], dtype=torch.int64, device=device),
        num_decodes=1,
        num_decode_tokens=1,
        num_prefills=0,
        decode=decode,
    )


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


def test_h12_uses_persistent_decode_while_other_small_heads_use_gluon():
    assert AiterMLAHelper.is_valid_num_heads(12)
    assert AiterMLAHelper.is_valid_num_heads(10)
    assert not AiterMLAHelper.use_gluon_decode(12, 1)
    assert AiterMLAHelper.use_gluon_decode(10, 1)


@pytest.mark.skipif(
    not _on_gfx950(),
    reason="12-head AITER MLA persistent decode is gfx950-only",
)
@torch.inference_mode()
def test_h12_aiter_mla_decode_matches_reference():
    """The actual 12-head AITER decode output must match attention reference."""
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    q = torch.randn(1, NUM_HEADS, QK_HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv_cache = torch.randn(
        CONTEXT_LEN,
        1,
        QK_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )

    impl = object.__new__(AiterMLAImpl)
    impl.num_heads = NUM_HEADS
    impl.kv_lora_rank = KV_LORA_RANK
    impl.scale = SCALE

    one = torch.ones(1, dtype=torch.float32, device=device)
    layer = types.SimpleNamespace(_q_scale=one, _k_scale=one)
    metadata = _make_h12_decode_metadata(device)

    out, _ = impl.forward_mqa(q, kv_cache, metadata, layer)

    key = kv_cache[:, 0].float().unsqueeze(0)
    value = key[..., :KV_LORA_RANK]
    out_ref = F.scaled_dot_product_attention(
        q[0].float().unsqueeze(1),
        key,
        value,
        scale=SCALE,
        enable_gqa=True,
    ).squeeze(1)

    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        out.float(),
        out_ref.unsqueeze(0),
        atol=1e-2,
        rtol=1e-2,
    )
