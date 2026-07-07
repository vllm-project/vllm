# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import torch

from vllm.model_executor.layers.attention.mla_attention import MLACommonImpl


def test_mla_prefill_quantized_weight_dtype():
    # Setup mock self object
    self_mock = MagicMock(spec=MLACommonImpl)
    self_mock.kv_lora_rank = 16
    self_mock.num_heads = 8
    self_mock.qk_nope_head_dim = 16
    self_mock.v_head_dim = 16
    self_mock.kv_cache_dtype = torch.bfloat16
    self_mock._use_flashinfer_concat_mla_k = False
    self_mock._concat_k_nope_k_pe.return_value = torch.zeros(2)

    # Mock kv_b_proj without weight attribute (like AWQ/GPTQ)
    mock_kv_b_proj = MagicMock()
    if hasattr(mock_kv_b_proj, "weight"):
        del mock_kv_b_proj.weight
    mock_kv_b_proj.params_dtype = torch.float16
    # Make kv_b_proj return a tensor when called
    mock_kv_b_proj.return_value = (torch.zeros(2, 8, 32),)
    self_mock.kv_b_proj = mock_kv_b_proj

    # Setup other mock arguments
    q = torch.zeros(2, 16)
    kv_c_and_k_pe_cache = torch.zeros(2, 32)
    k_scale = torch.tensor(1.0)

    attn_metadata = MagicMock()
    prefill_metadata = attn_metadata.prefill
    prefill_metadata.prefill_backend.run_prefill_context_chunk.return_value = (
        torch.zeros(2),
        torch.zeros(2),
    )
    prefill_metadata.q_data_type = torch.bfloat16
    prefill_metadata.chunked_context.seq_tot = [2]
    prefill_metadata.chunked_context.workspace = torch.zeros(10, 32)
    prefill_metadata.chunked_context.cu_seq_lens = [[0, 2]]
    prefill_metadata.chunked_context.token_to_seq = [[0, 1]]
    prefill_metadata.chunked_context.chunk_total_token = [2]
    prefill_metadata.chunked_context.starts = [0]
    prefill_metadata.chunked_context.seq_tot = [2]

    with patch("vllm.model_executor.layers.attention.mla_attention.ops"):
        # Call the method
        MLACommonImpl._compute_prefill_context(
            self_mock, q, kv_c_and_k_pe_cache, attn_metadata, k_scale
        )

        # Verify kv_b_proj was called
        mock_kv_b_proj.assert_called_once()

        # The input passed to kv_b_proj should have been cast to params_dtype (float16)
        called_args, _ = mock_kv_b_proj.call_args
        passed_tensor = called_args[0]
        assert passed_tensor.dtype == torch.float16
