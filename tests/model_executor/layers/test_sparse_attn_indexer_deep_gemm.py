# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla import indexer as mla_indexer


def test_sparse_attn_indexer_allows_cuda_paged_mqa_fallback(
    monkeypatch,
) -> None:
    assert hasattr(sparse_indexer, "is_deep_gemm_paged_mqa_supported")

    monkeypatch.setattr(sparse_indexer.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        sparse_indexer, "is_deep_gemm_paged_mqa_supported", lambda: False
    )
    monkeypatch.setattr(
        SparseAttnIndexer,
        "dispatch_forward",
        lambda *_args, **_kwargs: lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        sparse_indexer,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1,
                cp_kv_cache_interleave_size=1,
            ),
        ),
    )

    k_cache = SimpleNamespace(
        prefix="test_k_cache",
        kv_cache=torch.empty((1, 1, 1, 1), dtype=torch.uint8),
    )

    indexer = SparseAttnIndexer(
        k_cache=k_cache,
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=1,
        head_dim=128,
        max_model_len=16,
        max_total_seq_len=16,
        topk_indices_buffer=torch.empty((1, 1), dtype=torch.int32),
    )

    assert indexer.k_cache is k_cache


def test_deepseek_v32_indexer_disables_cudagraph_for_cuda_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setattr(mla_indexer.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(mla_indexer, "is_deep_gemm_paged_mqa_supported", lambda: False)

    assert (
        mla_indexer.DeepseekV32IndexerMetadataBuilder.get_cudagraph_support(
            SimpleNamespace(), SimpleNamespace()
        )
        is AttentionCGSupport.NEVER
    )

    monkeypatch.setattr(mla_indexer, "is_deep_gemm_paged_mqa_supported", lambda: True)
    assert (
        mla_indexer.DeepseekV32IndexerMetadataBuilder.get_cudagraph_support(
            SimpleNamespace(), SimpleNamespace()
        )
        is AttentionCGSupport.UNIFORM_BATCH
    )
