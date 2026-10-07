# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing and guard checks for the mxfp4_mla KV cache.

``_use_rocm_sparse_triton`` guards a single ``rocm_sparse_attn_prefill`` call,
so prefill and single-token decode share ``_sparse_attn_prefill_ragged_kernel``,
the only kernel with an MXFP4 branch. The ``_sparse_attn_decode_*`` kernels
address a 576-byte ``fp8_ds_mla`` paged row and are reached only from the
DeepSeek-V4 model path; a guard refuses a packed MXFP4 cache there.

CPU-only: these are predicate and guard facts, not kernel behaviour.
"""

from __future__ import annotations

import inspect

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


def test_guard_rejects_a_packed_mxfp4_cache():
    from vllm.v1.attention.ops.mxfp4_mla import row_bytes
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _reject_mxfp4_cache

    # 272 bytes for a 512-wide latent: 256 E2M1 + 16 E8M0.
    assert row_bytes(KV_LORA_RANK) == 272
    cache = torch.zeros((4, 64, row_bytes(KV_LORA_RANK)), dtype=torch.uint8)

    with pytest.raises(NotImplementedError, match="packed MXFP4"):
        _reject_mxfp4_cache(cache, KV_LORA_RANK, "extra")


def test_guard_allows_an_fp8_ds_mla_cache():
    """The guard must not break the path it is defending.

    fp8_ds_mla is also uint8, so a guard keyed on dtype would reject this and
    take out DeepSeek-V4. Only the MXFP4 row width may trip it.
    """
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _reject_mxfp4_cache

    for row in (576, 656):
        cache = torch.zeros((4, 64, row), dtype=torch.uint8)
        _reject_mxfp4_cache(cache, KV_LORA_RANK, "extra")

    # A bf16 cache is not uint8 and must pass through untouched.
    _reject_mxfp4_cache(
        torch.zeros((4, 64, KV_LORA_RANK), dtype=torch.bfloat16),
        KV_LORA_RANK,
        "extra",
    )


def test_decode_op_installs_the_guard_on_both_caches():
    """Both the SWA and the extra cache must be checked, not just one."""
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_decode

    body = inspect.getsource(rocm_sparse_attn_decode)
    assert body.count("_reject_mxfp4_cache") == 2
    assert "_reject_mxfp4_cache(swa_k_cache" in body
