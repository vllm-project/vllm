# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform

requires_flashinfer_mla = pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="FlashInfer MLA requires compute capability 10 or above.",
)


def test_mla_dcp_gathered_query_reserves_backend_head_storage():
    from vllm.v1.attention.ops.dcp import reserve_query_head_storage

    query = torch.randn(3, 24, 576, dtype=torch.bfloat16)

    padded = reserve_query_head_storage(query, 128)

    assert padded.shape == query.shape
    assert padded.stride() == query.stride()
    assert padded.untyped_storage().nbytes() >= 3 * 128 * 576 * 2
    assert torch.equal(padded, query)


@requires_flashinfer_mla
def test_flashinfer_mla_selects_backend_for_gathered_heads():
    from vllm.v1.attention.backends.mla.flashinfer_mla import (
        _select_mla_decode_backend,
    )

    assert _select_mla_decode_backend(6) is None
    assert _select_mla_decode_backend(24) == "cute-dsl"


@requires_flashinfer_mla
def test_flashinfer_mla_forward_uses_gathered_head_count(monkeypatch):
    import vllm.v1.attention.backends.mla.flashinfer_mla as flashinfer_mla

    impl = MagicMock()
    impl.bmm1_scale = 1.0
    impl.bmm2_scale = 1.0
    impl.need_to_return_lse_for_decode = True
    impl.dcp_world_size = 1
    impl.dcp_rank = 0
    impl.cp_kv_cache_interleave_size = 1
    impl.num_heads = 6
    impl.qk_nope_head_dim = 128
    impl.kv_lora_rank = 512
    impl.qk_rope_head_dim = 64
    impl.kv_cache_dtype = "auto"

    attn_metadata = MagicMock()
    attn_metadata.causal = True
    attn_metadata.num_decode_tokens = 2
    attn_metadata.num_decodes = 2
    attn_metadata.max_seq_len = 1
    attn_metadata.decode.block_table = torch.zeros(2, 1, dtype=torch.int32)
    attn_metadata.decode.seq_lens = torch.ones(2, dtype=torch.int32)

    query = torch.ones(2, 24, 576, dtype=torch.bfloat16)
    kv_cache = torch.ones(1, 128, 576, dtype=torch.bfloat16)
    kernel = MagicMock(
        return_value=(
            torch.ones(2, 1, 24, 512, dtype=torch.bfloat16),
            torch.ones(2, 24, dtype=torch.float32),
        )
    )
    monkeypatch.setattr(flashinfer_mla, "_get_workspace_buffer", MagicMock())
    monkeypatch.setattr(flashinfer_mla, "trtllm_batch_decode_with_kv_cache_mla", kernel)
    layer = MagicMock()

    output, lse = flashinfer_mla.FlashInferMLAImpl.forward_mqa(
        impl, query, kv_cache, attn_metadata, layer
    )

    assert kernel.call_args.kwargs["backend"] == "cute-dsl"
    assert output.shape == (2, 24, 512)
    assert lse is not None and lse.shape == (2, 24)


@requires_flashinfer_mla
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "noncausal"])
def test_flashinfer_mla_forward_uses_native_dcp_api(monkeypatch, causal):
    import vllm.v1.attention.backends.mla.flashinfer_mla as flashinfer_mla

    impl = MagicMock()
    impl.bmm1_scale = 1.0
    impl.bmm2_scale = 1.0
    impl.need_to_return_lse_for_decode = True
    impl.num_heads = 6
    impl.qk_nope_head_dim = 128
    impl.kv_lora_rank = 512
    impl.qk_rope_head_dim = 64
    impl.kv_cache_dtype = "auto"
    impl.dcp_world_size = 8
    impl.dcp_rank = 3
    impl.cp_kv_cache_interleave_size = 1

    num_reqs, query_len = 2, 3
    num_tokens = num_reqs * query_len
    block_table = torch.tensor([[10], [20]], dtype=torch.int32)
    seq_lens = torch.tensor([4, 5], dtype=torch.int32)
    global_causal_seq_lens = torch.tensor([31, 35], dtype=torch.int32)
    attn_metadata = MagicMock()
    attn_metadata.causal = causal
    attn_metadata.num_decode_tokens = num_tokens
    attn_metadata.num_decodes = num_reqs
    attn_metadata.max_seq_len = 9
    attn_metadata.decode.block_table = block_table
    attn_metadata.decode.seq_lens = seq_lens
    attn_metadata.decode.dcp_tot_seq_lens = global_causal_seq_lens

    query = torch.ones(num_tokens, 24, 576, dtype=torch.bfloat16)
    kv_cache = torch.ones(2, 128, 576, dtype=torch.bfloat16)
    kernel_batch = num_reqs if causal else num_tokens
    kernel_query_len = query_len if causal else 1
    kernel = MagicMock(
        return_value=(
            torch.ones(
                kernel_batch,
                kernel_query_len,
                24,
                512,
                dtype=torch.bfloat16,
            ),
            torch.ones(
                kernel_batch,
                kernel_query_len,
                24,
                dtype=torch.float32,
            ),
        )
    )
    monkeypatch.setattr(flashinfer_mla, "_get_workspace_buffer", MagicMock())
    monkeypatch.setattr(flashinfer_mla, "trtllm_batch_decode_with_kv_cache_mla", kernel)

    output, lse = flashinfer_mla.FlashInferMLAImpl.forward_mqa(
        impl, query, kv_cache, attn_metadata, MagicMock()
    )

    call = kernel.call_args.kwargs
    assert call["query"].shape == (kernel_batch, kernel_query_len, 24, 576)
    expected_block_table = (
        block_table if causal else block_table.repeat_interleave(query_len, dim=0)
    )
    expected_seq_lens = seq_lens if causal else seq_lens.repeat_interleave(query_len)
    expected_global_seq_lens = (
        global_causal_seq_lens
        if causal
        else global_causal_seq_lens.repeat_interleave(query_len)
    )
    torch.testing.assert_close(call["block_tables"], expected_block_table)
    torch.testing.assert_close(call["seq_lens"], expected_seq_lens)
    assert call["backend"] == "cute-dsl"
    assert call["enable_dcp"] is True
    assert call["cp_world"] == 8
    assert call["cp_rank"] == 3
    torch.testing.assert_close(
        call["causal_seqlens_kv_global"], expected_global_seq_lens
    )
    assert "multi_ctas_kv_counter_buffer" not in call
    assert flashinfer_mla.FlashInferMLAImpl.lse_base_on_e
    assert output.shape == (num_tokens, 24, 512)
    assert lse is not None and lse.shape == (num_tokens, 24)
