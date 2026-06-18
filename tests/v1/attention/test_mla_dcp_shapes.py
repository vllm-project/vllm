# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shape contract tests for MLA DCP attention paths."""

from types import SimpleNamespace

import torch

import vllm.v1.attention.backends.mla.flashmla as flashmla_mod


def test_flashmla_forward_mqa_flattens_lse_with_gathered_heads(monkeypatch):
    impl = object.__new__(flashmla_mod.FlashMLAImpl)
    impl.kv_cache_dtype = "auto"
    impl.kv_lora_rank = 512
    impl.scale = 1.0

    num_decodes = 2
    seq_len = 3
    num_heads = 64
    head_dim = 576

    q = torch.randn(num_decodes * seq_len, num_heads, head_dim)
    kv_cache = torch.randn(8, head_dim)
    expected_lse = torch.arange(
        num_decodes * num_heads * seq_len, dtype=torch.float32
    ).reshape(num_decodes, num_heads, seq_len)

    def fake_flash_mla_with_kvcache(**kwargs):
        assert kwargs["q"].shape == (num_decodes, seq_len, num_heads, head_dim)
        out = torch.zeros(num_decodes, seq_len, num_heads, impl.kv_lora_rank)
        return out, expected_lse

    monkeypatch.setattr(
        flashmla_mod, "flash_mla_with_kvcache", fake_flash_mla_with_kvcache
    )

    decode = flashmla_mod.FlashMLADecodeMetadata(
        block_table=torch.zeros(num_decodes, 1, dtype=torch.int32),
        seq_lens=torch.full((num_decodes,), 8, dtype=torch.int32),
        dcp_tot_seq_lens=None,
        scheduler_metadata=SimpleNamespace(),
    )
    metadata = flashmla_mod.FlashMLAMetadata(
        num_reqs=num_decodes,
        max_query_len=seq_len,
        max_seq_len=8,
        num_actual_tokens=num_decodes * seq_len,
        query_start_loc=torch.tensor([0, seq_len, 2 * seq_len], dtype=torch.int32),
        slot_mapping=torch.empty(0, dtype=torch.int64),
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes * seq_len,
        num_prefills=0,
        decode=decode,
    )

    out, lse = flashmla_mod.FlashMLAImpl.forward_mqa(
        impl,
        q,
        kv_cache,
        metadata,
        layer=SimpleNamespace(),
    )

    assert out.shape == (num_decodes * seq_len, num_heads, impl.kv_lora_rank)
    assert lse.shape == (num_decodes * seq_len, num_heads)
    torch.testing.assert_close(
        lse,
        expected_lse.permute(0, 2, 1)
        .reshape(num_decodes * seq_len, num_heads)
        .contiguous(),
    )
