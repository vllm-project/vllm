# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.models.deepseek_v4.nvidia import flashmla as flashmla_mod
from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention


def _make_layer(compress_ratio: int) -> DeepseekV4FlashMLAAttention:
    layer = object.__new__(DeepseekV4FlashMLAAttention)
    layer.compress_ratio = compress_ratio
    layer.swa_cache_layer = SimpleNamespace(kv_cache=torch.empty(1, 1, 512))
    layer.n_local_heads = 2
    layer.scale = 1.0
    layer.attn_sink = torch.zeros(2)
    return layer


def _make_swa_metadata() -> SimpleNamespace:
    return SimpleNamespace(
        num_decodes=1,
        num_decode_tokens=1,
        decode_swa_indices=torch.zeros((1, 4), dtype=torch.int32),
        decode_swa_lens=torch.ones(1, dtype=torch.int32),
        is_valid_token=torch.ones(1, dtype=torch.bool),
        token_to_req_indices=torch.zeros(1, dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        block_size=64,
        seq_lens=torch.full((1,), 4, dtype=torch.int32),
        tile_sched_swaonly=None,
        tile_sched_c4a=None,
        tile_sched_c128a=None,
    )


def test_swa_decode_uses_triton_path_without_flashmla_tile_sched(monkeypatch):
    layer = _make_layer(compress_ratio=1)
    metadata = _make_swa_metadata()
    calls = []

    def fake_decode(
        cls,
        layer,
        q,
        swa_k_cache,
        swa_metadata,
        output,
    ):
        calls.append(
            (layer, q.shape, swa_k_cache.shape, swa_metadata, output.shape)
        )

    monkeypatch.setattr(flashmla_mod, "is_triton_sparse_mla_enabled", lambda _: True)
    monkeypatch.setattr(
        DeepseekV4FlashMLAAttention,
        "_forward_sparse_mla_swa_decode_triton",
        classmethod(fake_decode),
    )

    q = torch.empty(1, 2, 512)
    output = torch.empty_like(q)

    layer._forward_decode(
        q=q,
        kv_cache=None,
        swa_metadata=metadata,
        attn_metadata=None,
        swa_only=True,
        output=output,
    )

    assert calls == [(layer, (1, 1, 2, 512), (1, 1, 512), metadata, (1, 2, 512))]


def test_compressed_decode_uses_triton_path_without_flashmla_tile_sched(monkeypatch):
    layer = _make_layer(compress_ratio=128)
    metadata = _make_swa_metadata()
    attn_metadata = SimpleNamespace(
        block_size=256,
        c128a_global_decode_topk_indices=torch.zeros((1, 1, 2), dtype=torch.int32),
        c128a_decode_topk_lens=torch.ones(1, dtype=torch.int32),
    )
    kv_cache = torch.empty(1, 1, 512)
    calls = []

    def fake_decode(
        cls,
        layer,
        q,
        compressed_k_cache,
        swa_k_cache,
        topk_indices,
        topk_lens,
        swa_metadata,
        attn_metadata,
        output,
    ):
        calls.append(
            (
                layer,
                q.shape,
                compressed_k_cache.shape,
                swa_k_cache.shape,
                topk_indices.shape,
                topk_lens.shape,
                swa_metadata,
                attn_metadata,
                output.shape,
            )
        )

    monkeypatch.setattr(flashmla_mod, "is_triton_sparse_mla_enabled", lambda _: True)
    monkeypatch.setattr(
        DeepseekV4FlashMLAAttention,
        "_forward_sparse_mla_compressed_decode_triton",
        classmethod(fake_decode),
    )

    q = torch.empty(1, 2, 512)
    output = torch.empty_like(q)

    layer._forward_decode(
        q=q,
        kv_cache=kv_cache,
        swa_metadata=metadata,
        attn_metadata=attn_metadata,
        swa_only=False,
        output=output,
    )

    assert calls == [
        (
            layer,
            (1, 1, 2, 512),
            (1, 1, 512),
            (1, 1, 512),
            (1, 1, 2),
            (1,),
            metadata,
            attn_metadata,
            (1, 2, 512),
        )
    ]
