# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the fuse_score_remap decode indexer hook."""

import torch

from vllm.model_executor.kernels.attention.dsa.fuse_score_remap.indexer import (
    can_use_fuse_score_remap,
    pack_parts,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    indexer_topk_is_physical,
    set_indexer_topk_physical,
    triton_convert_req_index_to_global_index,
)


def _q(batch: int = 1, next_n: int = 1, heads: int = 32, dim: int = 128):
    return torch.empty((batch, next_n, heads, dim))


def _kv(num_pages: int = 4, page: int = 64, width: int = 132):
    return torch.empty((num_pages, page, 1, width), dtype=torch.uint8)


def test_can_use_fuse_score_remap_accepts_sm90_decode_shapes():
    assert can_use_fuse_score_remap(
        _q(),
        _kv(),
        next_n=1,
        head_dim=128,
        topk_tokens=2048,
        use_fp4_cache=False,
        block_table=torch.zeros((1, 32), dtype=torch.int32),
        max_model_len=65536,
    )


def test_can_use_fuse_score_remap_rejects_unsupported_configs():
    q = _q()
    kv = _kv()
    base = dict(
        next_n=1,
        head_dim=128,
        topk_tokens=2048,
        use_fp4_cache=False,
    )
    assert not can_use_fuse_score_remap(q, kv, **{**base, "use_fp4_cache": True})
    assert not can_use_fuse_score_remap(q, kv, use_pcp=True, **base)
    assert not can_use_fuse_score_remap(q, kv, dcp_world_size=2, **base)
    assert not can_use_fuse_score_remap(q, kv, has_prefill=True, **base)
    assert not can_use_fuse_score_remap(q, kv, **{**base, "next_n": 2})
    assert not can_use_fuse_score_remap(q, kv, **{**base, "topk_tokens": 256})
    assert not can_use_fuse_score_remap(q, kv, **{**base, "max_model_len": 65537})


def test_pack_parts_matches_split_kv():
    table = torch.zeros((2, 128), dtype=torch.int32)  # 128 * 64 = 8192
    assert pack_parts(table) == 32
    table64k = torch.zeros((2, 1024), dtype=torch.int32)  # 65536
    assert pack_parts(table64k) == 256


def test_physical_topk_skips_page_table_remap():
    set_indexer_topk_physical(True)
    try:
        physical = torch.tensor([[64, 65, -1, -1]], dtype=torch.int32)
        req_id = torch.zeros((1,), dtype=torch.int32)
        block_table = torch.arange(8, dtype=torch.int32).reshape(1, 8)
        out, valid = triton_convert_req_index_to_global_index(
            req_id,
            block_table,
            physical,
            BLOCK_SIZE=64,
            NUM_TOPK_TOKENS=4,
            BLOCK_N=4,
            return_valid_counts=True,
        )
        assert torch.equal(out, physical)
        assert valid.tolist() == [2]
        assert indexer_topk_is_physical()
    finally:
        set_indexer_topk_physical(False)
