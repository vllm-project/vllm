# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention.mla_attention import QueryLenSupport
from vllm.v1.attention.backends.mla import triton_mla


@pytest.mark.cpu_test
def test_causal_multi_token_decode_flattens_with_prefix_lengths(monkeypatch):
    assert (
        triton_mla.TritonMLAMetadataBuilder.query_len_support == QueryLenSupport.UNIFORM
    )

    captured = {}

    def decode_spy(*args, **kwargs):
        captured["block_table"] = args[5].clone()
        captured["seq_lens"] = args[6].clone()

    monkeypatch.setattr(triton_mla, "decode_attention_fwd", decode_spy)
    monkeypatch.setattr(triton_mla, "_compute_num_kv_splits", lambda *_: 1)

    impl = object.__new__(triton_mla.TritonMLAImpl)
    impl.kv_lora_rank = 2
    impl.scale = 1.0
    impl._sm_count = 1

    query_len = 3
    num_decodes = 3
    num_heads = 2
    head_size = 3
    block_table = torch.arange(12, dtype=torch.int32).view(num_decodes, 4)
    seq_lens = torch.tensor([10, 2, 0], dtype=torch.int32)
    metadata = SimpleNamespace(
        causal=True,
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes * query_len,
        max_seq_len=10,
        decode=SimpleNamespace(block_table=block_table, seq_lens=seq_lens),
    )
    q = torch.zeros(num_decodes * query_len, num_heads, head_size)
    kv_cache = torch.zeros(8, 4, head_size)

    output, lse = impl.forward_mqa(
        q,
        kv_cache,
        metadata,
        SimpleNamespace(_k_scale=1.0),
    )

    assert output.shape == (num_decodes * query_len, num_heads, 2)
    assert lse is not None
    assert lse.shape == (num_decodes * query_len, num_heads)
    torch.testing.assert_close(
        captured["block_table"], block_table.repeat_interleave(query_len, dim=0)
    )
    torch.testing.assert_close(
        captured["seq_lens"],
        torch.tensor([8, 9, 10, 0, 1, 2, 0, 0, 0], dtype=torch.int32),
    )
