# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing checks for the mxfp4_mla KV cache.

``_use_rocm_sparse_triton`` guards a single ``rocm_sparse_attn_prefill`` call,
so prefill and single-token decode share ``_sparse_attn_prefill_ragged_kernel``,
the only kernel that reads an MXFP4 cache. CPU-only, except where marked.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

# GLM-5.3-Flash: kv_lora_rank=512, qk_rope_head_dim=0, so head_size == 512.
HEAD_SIZE = 512
KV_LORA_RANK = 512


def _route(kv_cache_dtype, head_size=HEAD_SIZE, **kw) -> bool:
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        _use_rocm_sparse_triton,
    )

    meta = dict(num_prefills=0, num_decodes=0, num_decode_tokens=0, max_query_len=1)
    meta.update(kw)
    return _use_rocm_sparse_triton(
        kv_cache_dtype=kv_cache_dtype,
        head_size=head_size,
        kv_lora_rank=KV_LORA_RANK,
        **meta,
    )


DECODE = dict(num_decodes=8, num_decode_tokens=8, max_query_len=1)
PREFILL = dict(num_prefills=2, max_query_len=128)


def test_mxfp4_decode_and_prefill_take_the_triton_route():
    for step in (DECODE, PREFILL):
        assert _route("mxfp4_mla", **step) is True
        assert _route("auto", **step) is True


def test_rope_bearing_geometry_is_refused():
    """DeepSeek shapes (576 = 512 + 64) must not take this route."""
    assert _route("mxfp4_mla", head_size=576, **DECODE) is False
    assert _route("auto", head_size=576, **PREFILL) is False


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="_forward_mla imports vllm.platforms.rocm, which queries the GPU",
)
def test_mxfp4_decode_takes_the_split_k_kernel(monkeypatch):
    """Long decodes split, and the split-K kernel must be told the cache is packed."""
    from types import SimpleNamespace

    from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as sparse_mod

    def no_single_pass(**kwargs):
        raise AssertionError("expected the split-K decode")

    captured: dict[str, Any] = {}
    monkeypatch.setattr(sparse_mod, "rocm_sparse_decode_bf16_num_splits", lambda *a: 4)
    monkeypatch.setattr(sparse_mod, "rocm_sparse_attn_decode_bf16", captured.update)
    monkeypatch.setattr(sparse_mod, "rocm_sparse_attn_prefill", no_single_pass)

    impl = object.__new__(sparse_mod.ROCMAiterMLASparseImpl)
    impl.num_heads = 16
    impl.kv_lora_rank = KV_LORA_RANK
    impl.kv_cache_dtype = "mxfp4_mla"
    impl.scale = KV_LORA_RANK**-0.5
    impl.sinks = None
    metadata = SimpleNamespace(
        attn_out_dtype=torch.bfloat16,
        num_prefills=0,
        num_decodes=2,
        num_decode_tokens=2,
        max_query_len=1,
        max_seq_len=8192,
        topk_tokens=2048,
        paged_kv_indices=torch.zeros(4, dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, 2, 4], dtype=torch.int32),
    )
    q = torch.zeros(2, 16, HEAD_SIZE, dtype=torch.bfloat16)
    cache = torch.zeros(4, 1, 272, dtype=torch.uint8)

    impl._forward_mla(SimpleNamespace(), q, cache, metadata)

    assert captured["kv_cache_dtype"] == "mxfp4_mla"
    assert captured["kv"].shape == (4, 1, 272)
    assert captured["num_splits"] == 4
