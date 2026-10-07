# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing checks for the mxfp4_mla KV cache.

``_use_rocm_sparse_triton`` guards a single ``rocm_sparse_attn_prefill`` call,
so prefill and single-token decode share ``_sparse_attn_prefill_ragged_kernel``,
the only kernel that reads an MXFP4 cache. CPU-only.
"""

from __future__ import annotations

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
