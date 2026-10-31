# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's CuTe kpool indexer kernels are bit-identical to the
stock Triton kernels they replace."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

DIM, POOL, GROUPS, WIDTH, BLOCK = 128, 4, 512, 2176, 64

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the CuTe kernels target SM100/SM103",
)


def _update_inputs(requests, tokens, seed, padded):
    g = torch.Generator(device="cuda").manual_seed(seed)
    blocks, page = requests + 4, 64
    cache = torch.randint(
        0, 255, (blocks, page, DIM + 4), dtype=torch.uint8, device="cuda", generator=g
    )
    tail = (
        torch.randn(blocks, 2, POOL, DIM, device="cuda", generator=g) * 3
    ).bfloat16()
    key = (
        torch.randn(requests, tokens, DIM, device="cuda", generator=g) * 2
    ).bfloat16()
    score = (
        torch.randn(requests, tokens, DIM, device="cuda", generator=g) * 2
    ).bfloat16()
    ape = torch.randn(POOL, DIM, device="cuda", generator=g)
    start = torch.randint(0, 8000, (requests, 1), device="cuda", generator=g)
    positions = start + torch.arange(tokens, device="cuda")
    owner = torch.arange(1, requests + 1, device="cuda")[:, None]
    tail_slots = (owner * POOL + positions % POOL).int()
    # Pool-granular cache slots: only a pool's last token writes, one page each.
    slots = torch.where(
        positions % POOL == POOL - 1,
        (owner + 1) * page + (positions // POOL) % page,
        -1,
    ).int()
    if padded:
        positions[0, -1], slots[0, -1], tail_slots[0, -1] = -1, -1, -1
    return [
        cache,
        tail,
        tail_slots,
        key,
        score,
        ape,
        slots,
        positions,
    ]


@pytest.mark.parametrize("requests,tokens", [(1, 1), (64, 1), (2, 7)])
@pytest.mark.parametrize("padded", [False, True])
def test_kpool_update_matches_stock(requests, tokens, padded):
    """Tail stashes, completed pools (FP8 values and UE8M0 scales) and the
    in-order read of a request's earlier stashes match bit for bit."""
    from vllm.models.glm5next.nvidia.ops.kpool_compress import (
        kpool_decode_update_and_maybe_write_cache_batched as stock,
    )
    from vllm.models.glm5next.nvidia.specialized_decode.kpool import kpool_update
    from vllm.utils.deep_gemm import mqa_logits_use_mx_scales

    assert mqa_logits_use_mx_scales()
    ref = _update_inputs(requests, tokens, 100 * requests + tokens, padded)
    out = [t.clone() for t in ref]
    before = ref[0].clone()
    stock(*ref, POOL, DIM, round_scale=True)
    kpool_update(*out, POOL)
    assert not torch.equal(ref[0], before) or requests * tokens < POOL
    assert torch.equal(out[0], ref[0])
    assert torch.equal(out[1].view(torch.int16), ref[1].view(torch.int16))


@pytest.mark.parametrize("rows", [1, 32])
@pytest.mark.parametrize("width", [2048, 32768])
def test_upcast_matches_stock(rows, width):
    from vllm.model_executor.layers.sparse_attn_indexer import as_fp32_logits
    from vllm.models.glm5next.nvidia.specialized_decode.kpool import upcast_logits

    logits = torch.randn(rows, width + 64, device="cuda").bfloat16()[:, :width]
    ends = torch.randint(1, width + 1, (rows,), dtype=torch.int32, device="cuda")
    ref, out = as_fp32_logits(logits, ends), upcast_logits(logits, ends)
    live = torch.arange(width, device="cuda") < ends[:, None]
    assert torch.equal(out.view(torch.int32)[live], ref.view(torch.int32)[live])


def _pool_ids(rows, seq, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randint(
        0, seq // POOL, (rows, GROUPS), dtype=torch.int32, device="cuda", generator=g
    )
    ids[torch.rand(rows, GROUPS, device="cuda", generator=g) < 0.2] = -1
    positions = torch.randint(seq - 8, seq, (rows,), device="cuda", generator=g)
    return ids, positions


@pytest.mark.parametrize("rows", [1, 64])
def test_expand_matches_stock(rows):
    from vllm.models.glm5next.nvidia.ops.kpool_compress import (
        expand_pools_and_append_tail,
    )
    from vllm.models.glm5next.nvidia.specialized_decode.kpool import expand_pools

    ids, positions = _pool_ids(rows, 8192, rows)
    ref = torch.full((rows, WIDTH), -1, dtype=torch.int32, device="cuda")
    out = ref.clone()
    expand_pools_and_append_tail(ids, positions, POOL, seq_offset=1, out=ref)
    expand_pools(ids, positions, POOL, seq_offset=1, out=out)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("rows,padding", [(64, 0), (4, 3)])
@pytest.mark.parametrize("stride_rows,max_blocks", [(BLOCK, 160), (3 * BLOCK, 50)])
def test_expand_and_remap_matches_stock(rows, padding, stride_rows, max_blocks):
    """Expansion plus the compacting remap equals the stock pair, including
    pools past the block table (dropped) and padded rows (all -1)."""
    from vllm.models.glm5next.nvidia.ops.kpool_compress import (
        expand_pools_and_append_tail,
    )
    from vllm.models.glm5next.nvidia.specialized_decode.kpool import expand_and_remap
    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_convert_req_index_to_global_index,
    )

    ids, positions = _pool_ids(rows, 8192, rows + max_blocks)
    total = rows + padding
    req_ids = torch.arange(total, dtype=torch.int32, device="cuda") % rows
    block_table = torch.randint(
        0, 5000, (rows, max_blocks), dtype=torch.int32, device="cuda"
    )
    logical = torch.full((total, WIDTH), -1, dtype=torch.int32, device="cuda")
    expand_pools_and_append_tail(ids, positions, POOL, seq_offset=1, out=logical)
    ref, ref_counts = triton_convert_req_index_to_global_index(
        req_ids, block_table, logical, BLOCK_SIZE=BLOCK,
        BLOCK_STRIDE_ROWS=stride_rows, NUM_TOPK_TOKENS=WIDTH, return_valid_counts=True,
    )  # fmt: skip
    out = torch.full_like(ref, 7)
    counts = torch.full_like(ref_counts, 7)
    expand_and_remap(
        ids, positions, POOL, req_ids, block_table, out, counts,
        block_size=BLOCK, stride_rows=stride_rows,
    )  # fmt: skip
    assert torch.equal(counts, ref_counts)
    assert torch.equal(out, ref)


def test_fused_remap_hook_matches_stock_conversion():
    """A deferred pool top-k converted through the index group's side stream
    equals the stock expansion followed by the stock group conversion."""
    import vllm.models.glm5next.nvidia.specialized_decode.kpool as kpool
    from vllm.v1.attention.backends.mla.index_group import SparseMLAIndexGroupBuilder

    rows = 4
    ids, positions = _pool_ids(rows, 8192, 0)
    req_ids = torch.arange(rows, dtype=torch.int32, device="cuda")
    block_table = torch.randint(0, 5000, (rows, 160), dtype=torch.int32, device="cuda")
    md = SimpleNamespace(
        req_id_per_token=req_ids, block_table=block_table, block_size=BLOCK
    )

    def group():
        logical = torch.full((16, WIDTH), -1, dtype=torch.int32, device="cuda")
        return SparseMLAIndexGroupBuilder(logical).register_layer(True)[0]

    stock = group()
    expanded = stock.logical_topk_indices[:rows]
    kpool.expand_pools_and_append_tail(ids, positions, POOL, seq_offset=1, out=expanded)
    stock.logical_topk_ready.record()
    ref = stock.convert_logical_to_physical_topk(
        0, expanded, md, block_stride_rows=None, return_valid_counts=True
    )

    fused = group()
    impl = SimpleNamespace(
        index_group=fused, group_conversion_decode_only=False, dcp_world_size=1
    )
    attention = SimpleNamespace(
        mla_attn=SimpleNamespace(mla_attn=SimpleNamespace(impl=impl)),
        indexer=SimpleNamespace(k_cache=SimpleNamespace(prefix="layers.0.k_cache")),
    )
    stage = kpool.KpoolStage([attention], 64, "cuda")
    assert stage.fuse and stage.defers(
        "layers.0.k_cache", rows, GROUPS * POOL, POOL, WIDTH
    )
    # Layers without a fused conversion (the MTP draft's) never defer.
    assert not stage.defers("mtp.k_cache", rows, GROUPS * POOL, POOL, WIDTH)
    pool_ids = stage.pool_ids(rows, GROUPS)
    pool_ids.copy_(ids)
    stage.defer(pool_ids, positions, POOL)
    fused.logical_topk_ready.record()
    # The logical rows are stale: the fused conversion must not read them.
    fused.logical_topk_indices.fill_(123)
    out = fused.convert_logical_to_physical_topk(
        0, fused.logical_topk_indices[:rows], md, block_stride_rows=None,
        return_valid_counts=True,
    )  # fmt: skip
    torch.accelerator.synchronize()
    assert stage.pending is None
    assert torch.equal(out[1], ref[1]) and torch.equal(out[0], ref[0])
