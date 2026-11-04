# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP ShardQ prefill: the gathered context against replicated (TP) caches.

Each DCP rank's paged MLA and indexer caches are built from one replicated
copy; pushing every rank's rows must rebuild that copy in the workspace byte
for byte, and the indexer and the FlashInfer sparse MLA kernels (trtllm-gen and
prims_ts) must read it exactly as the TP ShardQ path reads the replicated
caches.
"""

import dataclasses
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_prims_ts_sparse_mla
from vllm.utils.import_utils import has_cutedsl

pytestmark = pytest.mark.skipif(
    not (current_platform.is_device_capability_family(100) and has_cutedsl()),
    reason="CuTe DSL on SM100 is required",
)

PAGE = 64
# The MLA cache's kernel blocks span several pages, as in a packed block row.
MLA_BLOCK = 4 * PAGE
INDEX_HEAD_DIM = 128
INDEX_HEAD_SIZE = INDEX_HEAD_DIM + 4
# Value bytes per token of each indexer key format (each adds 4 B of scales):
# fp8 keys, or MXFP4 keys (E2M1 pairs; four UE8M0 scales per 128 values).
INDEX_ROW = {"fp8": INDEX_HEAD_DIM, "mxfp4": INDEX_HEAD_DIM // 2}
INDEX_FORMATS = list(INDEX_ROW)
TOPK = 2048
NUM_HEADS = 96
MLA_ROW = {"fp8": 576, "nvfp4_ds_mla": 352}
CACHE_DTYPES = list(MLA_ROW)
FILL = 0xA5
_needs_prims_ts = pytest.mark.skipif(
    not has_flashinfer_prims_ts_sparse_mla(),
    reason="needs FlashInfer's prims_ts sparse MLA",
)
# (cache dtype, FlashInfer sparse MLA kernel): every pairing the backend serves.
KERNEL_CASES = [
    pytest.param("fp8", "trtllm", id="fp8-trtllm"),
    pytest.param("fp8", "prims_ts", id="fp8-prims_ts", marks=_needs_prims_ts),
    pytest.param(
        "nvfp4_ds_mla", "prims_ts", id="nvfp4_ds_mla-prims_ts", marks=_needs_prims_ts
    ),
]

# (context, new tokens) per request.
SCENARIOS = {
    "fresh": [(0, 300)],
    "short_contexts": [(3, 5), (7, 1), (8, 64), (1000, 37)],
    "page_tails": [(4097, 2048), (513, 1), (0, 64), (1, 511)],
    "long": [(20000, 4096)],
}


def _layout(cache_dtype: str, index_format: str = "fp8"):
    from vllm.v1.attention.ops.dcp_shardq.context import (
        ContextLayout,
    )

    return ContextLayout(MLA_ROW[cache_dtype], INDEX_ROW[index_format], 4)


def _index_entry(layout) -> int:
    """Indexer cache bytes per token (values then scales, in pages)."""
    return layout.index_row_bytes + layout.index_scale_bytes


def _page_start(lens) -> np.ndarray:
    pages = np.array([-(-(c + n) // PAGE) for c, n in lens])
    return np.cumsum(pages) - pages


def _sections(layout):
    """(cache, offset in the page, bytes per row) of each per-token section."""
    return (
        ("mla", 0, layout.mla_row_bytes),
        ("index", 0, layout.index_row_bytes),
        ("index", PAGE * layout.index_row_bytes, layout.index_scale_bytes),
    )


def _row_bytes(page_base: torch.Tensor, rows: torch.Tensor, offset: int, row: int):
    """Flat byte offsets [len(rows), row] of rows in pages at ``page_base``."""
    base = page_base.long() + offset + rows.long() * row
    return base[:, None] + torch.arange(row, device=rows.device)


def _flat(cache: torch.Tensor) -> torch.Tensor:
    """The bytes under a paged cache view, from its first byte."""
    end = (cache.shape[0] - 1) * cache.stride(0) + cache[0].numel()
    return torch.as_strided(cache, (end,), (1,))


class Replicated:
    """Every token of every request, in dense pages from ``_page_start``."""

    def __init__(self, lens, layout, fill_random: bool = True):
        self.lens, self.layout = lens, layout
        num_pages = int(sum(-(-(c + n) // PAGE) for c, n in lens))
        torch.manual_seed(0)

        def cache(row):
            shape = (num_pages, PAGE, row)
            if fill_random:
                return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
            return torch.zeros(shape, dtype=torch.uint8, device="cuda")

        self.mla = cache(layout.mla_row_bytes)
        self.index = cache(_index_entry(layout))

    def token_bytes(self, req: int, tokens: torch.Tensor, section):
        """``section``'s cache and the flat byte offsets of request ``req``'s
        rows of ``tokens`` in it."""
        name, offset, row = section
        cache = getattr(self, name)
        pages = int(_page_start(self.lens)[req]) + tokens // PAGE
        return cache, _row_bytes(pages * cache.stride(0), tokens % PAGE, offset, row)


def _shard(
    rep: Replicated, world: int, rank: int, seed: int, mla_block: int = MLA_BLOCK
):
    """``rank``'s DCP-local caches of ``rep``: the MLA cache in ``mla_block``-
    token blocks and the indexer cache in pages, both padded apart (other
    layers' pages sit between them in a packed cache) and randomly permuted,
    with their block tables."""
    layout, lens = rep.layout, rep.lens
    gen = torch.Generator().manual_seed(seed)
    local_lens = [max(c + n - rank + world - 1, 0) // world for c, n in lens]

    def paged(block_tokens, row, pad):
        blocks = [-(-n // block_tokens) for n in local_lens]
        num_blocks = sum(blocks) + 2
        stride = block_tokens * row + pad
        raw = torch.randint(0, 256, (num_blocks * stride,), dtype=torch.uint8)
        cache = torch.as_strided(
            raw.cuda(), (num_blocks, block_tokens, row), (stride, row, 1)
        )
        perm = torch.randperm(num_blocks, generator=gen)
        table = torch.zeros(len(lens), max(max(blocks), 1), dtype=torch.int32)
        used = 0
        for q, n in enumerate(blocks):
            table[q, :n] = perm[used : used + n]
            used += n
        return cache, table.cuda()

    # Whole pages apart, as the sparse MLA kernels address the cache in pages
    # of rows.
    mla, mla_table = paged(mla_block, layout.mla_row_bytes, layout.mla_page_bytes)
    index, index_table = paged(PAGE, _index_entry(layout), 80)
    for q, n_local in enumerate(local_lens):
        local = torch.arange(n_local, device="cuda")
        token = local * world + rank
        for section in _sections(layout):
            src_cache, src = rep.token_bytes(q, token, section)
            if section[0] == "mla":
                cache, rows = mla, local % mla_block
                base = mla_table[q][local // mla_block].long() * mla.stride(0)
            else:
                cache, rows = index, local % PAGE
                base = index_table[q][local // PAGE].long() * index.stride(0)
            dst = _row_bytes(base, rows, section[1], section[2])
            _flat(cache)[dst.reshape(-1)] = _flat(src_cache)[src.reshape(-1)]
    return SimpleNamespace(
        mla=mla, mla_table=mla_table, index=index, index_table=index_table
    )


def _metadata(lens):
    context = np.array([c for c, _ in lens])
    new = np.array([n for _, n in lens])
    qsl = torch.from_numpy(np.r_[0, np.cumsum(new)]).to(torch.int32)
    seq_lens = torch.from_numpy(context + new).to(torch.int32)
    return SimpleNamespace(
        num_reqs=len(lens),
        query_start_loc_cpu=qsl,
        query_start_loc=qsl.cuda(),
        seq_lens_cpu_upper_bound=seq_lens,
        seq_lens=seq_lens.cuda(),
    )


def _build_context(lens, layout, world, rank, max_pages=None):
    from vllm.v1.attention.ops.dcp_shardq.context import (
        build_dcp_shardq_context,
    )

    total_pages = sum(-(-(c + n) // PAGE) for c, n in lens)
    return build_dcp_shardq_context(
        _metadata(lens),
        layout,
        total_pages if max_pages is None else max_pages,
        max_indexer_keys=1 << 20,
        max_logits_bytes=1 << 30,
        world=world,
        rank=rank,
        device=torch.device("cuda"),
    )


def _workspace(plan, layout):
    """A local stand-in for one rank's symmetric workspace and its views."""
    nbytes = plan.pages * (layout.mla_page_bytes + layout.index_page_bytes)
    raw = torch.full((nbytes,), FILL, dtype=torch.uint8, device="cuda")
    mla = torch.as_strided(
        raw,
        (plan.pages, PAGE, layout.mla_row_bytes),
        (layout.mla_page_bytes, layout.mla_row_bytes, 1),
    )
    index = torch.as_strided(
        raw,
        (plan.pages, PAGE, _index_entry(layout)),
        (layout.index_page_bytes, _index_entry(layout), 1),
        plan.pages * layout.mla_page_bytes,
    )
    return raw, mla, index


def _push_all(rep: Replicated, world: int, num_dst: int = 2):
    """Push every simulated rank's rows into ``num_dst`` local workspaces;
    returns them and the plans."""
    layout = rep.layout
    plans = [_build_context(rep.lens, layout, world, r) for r in range(world)]
    workspaces = [_workspace(plans[0], layout) for _ in range(num_dst)]
    # The kernel stores to ``world`` destinations; spread them over the copies.
    dst_addrs = torch.tensor(
        [workspaces[d % num_dst][0].data_ptr() for d in range(world)],
        dtype=torch.int64,
        device="cuda",
    )
    for rank, plan in enumerate(plans):
        shard = _shard(rep, world, rank, seed=rank)
        plan.push(shard.mla, shard.mla_table, shard.index, shard.index_table, dst_addrs)
    torch.accelerator.synchronize()
    return workspaces, plans


def _check_workspace(
    ws_mla, ws_index, rep: Replicated, new_mla_rows: bool, fill, num_decodes=0
):
    """The workspace holds every section of each prefill request's tokens (the
    requests from ``num_decodes`` on, in pages of their own; the MLA rows of
    its new tokens only with ``new_mla_rows``); with ``fill``, every other
    byte still holds it."""
    workspaces = {"mla": ws_mla, "index": ws_index}
    expected = {
        name: torch.full_like(_flat(ws), FILL) for name, ws in workspaces.items()
    }
    written = {
        name: torch.zeros_like(e, dtype=torch.bool) for name, e in expected.items()
    }
    page_start = _page_start(rep.lens[num_decodes:])
    for q, (context, new) in enumerate(rep.lens[num_decodes:], num_decodes):
        tokens = torch.arange(context + new, device="cuda")
        for section in _sections(rep.layout):
            name, offset, row = section
            keep = tokens if name == "index" or new_mla_rows else tokens[:context]
            cache, src = rep.token_bytes(q, keep, section)
            pages = int(page_start[q - num_decodes]) + keep // PAGE
            stride = workspaces[name].stride(0)
            dst = _row_bytes(pages * stride, keep % PAGE, offset, row).reshape(-1)
            expected[name][dst] = _flat(cache)[src.reshape(-1)]
            written[name][dst] = True
    for name, ws in (("mla", ws_mla), ("index", ws_index)):
        if fill:
            assert torch.equal(_flat(ws), expected[name])
        else:
            mask = written[name]
            assert torch.equal(_flat(ws)[mask], expected[name][mask])


# (MLA cache dtype, indexer key format): MXFP4 keys (the K3.1 default on SM100)
# with the per-tensor fp8 MLA cache they are deployed with.
LAYOUT_CASES = [
    pytest.param("fp8", "fp8", id="fp8"),
    pytest.param("nvfp4_ds_mla", "fp8", id="nvfp4_ds_mla"),
    pytest.param("fp8", "mxfp4", id="fp8-mxfp4_index"),
]


@pytest.mark.parametrize("cache_dtype,index_format", LAYOUT_CASES)
@pytest.mark.parametrize("world", [2, 8])
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_push_rebuilds_context_in_token_order(
    cache_dtype, index_format, world, scenario
):
    rep = Replicated(SCENARIOS[scenario], _layout(cache_dtype, index_format))
    workspaces, _ = _push_all(rep, world)
    for _, ws_mla, ws_index in workspaces:
        # Prior context in full, the new tokens' indexer rows, nothing else
        # (the new MLA rows and page tails stay untouched).
        _check_workspace(ws_mla, ws_index, rep, new_mla_rows=False, fill=True)


def test_plan_bounds_and_capacity():
    lens = SCENARIOS["page_tails"]
    layout = _layout("fp8")
    plan = _build_context(lens, layout, 8, 3)
    seq_lens = np.array([c + n for c, n in lens])
    page_start = _page_start(lens)
    # The workspace block table maps each request's pages to its region, in
    # rows of whole multiples of 16 pages (one Triton specialization).
    table = plan.block_table.cpu().numpy()
    assert table.shape[1] % 16 == 0
    for q, pages in enumerate(-(-seq_lens // PAGE)):
        np.testing.assert_array_equal(
            table[q, :pages], page_start[q] + np.arange(pages)
        )
    # New token i of request q lands at its logical position.
    expected_slots = np.concatenate(
        [page_start[q] * PAGE + np.arange(c, c + n) for q, (c, n) in enumerate(lens)]
    )
    np.testing.assert_array_equal(plan.slot_mapping.cpu().numpy(), expected_slots)
    # A step one page larger than the workspace keeps the DCP path.
    total_pages = int((-(-seq_lens // PAGE)).sum())
    assert _build_context(lens, layout, 8, 3, total_pages - 1) is None


@pytest.mark.parametrize("cache_dtype,index_format", LAYOUT_CASES)
def test_indexer_chunks_read_the_workspace_like_a_replicated_cache(
    cache_dtype, index_format
):
    """The prefill indexer's key gather (the same op for fp8 and MXFP4 keys, at
    the format's widths) reads the workspace exactly as the replicated cache:
    byte-identical keys and scales, hence identical logits and top-k."""
    from vllm import _custom_ops as ops

    lens, world = SCENARIOS["page_tails"], 8
    layout = _layout(cache_dtype, index_format)
    rep = Replicated(lens, layout)
    workspaces, plans = _push_all(rep, world, num_dst=1)
    _, _, ws_index = workspaces[0]
    seq_lens = [c + n for c, n in lens]
    for chunk in plans[0].indexer_chunks:
        # Every key of the chunk's requests, as the TP path gathers them.
        keys = torch.empty(
            chunk.total_seq_lens,
            layout.index_row_bytes,
            dtype=torch.uint8,
            device="cuda",
        )
        scales = torch.empty(
            chunk.total_seq_lens,
            layout.index_scale_bytes,
            dtype=torch.uint8,
            device="cuda",
        )
        ref_keys, ref_scales = torch.empty_like(keys), torch.empty_like(scales)
        ops.cp_gather_indexer_k_quant_cache(
            ws_index, keys, scales, chunk.block_table, chunk.cu_seq_lens
        )
        ops.cp_gather_indexer_k_quant_cache(
            rep.index, ref_keys, ref_scales, chunk.block_table, chunk.cu_seq_lens
        )
        assert torch.equal(keys, ref_keys) and torch.equal(scales, ref_scales)
        # Causal bounds over the request's whole sequence (no DCP-local form).
        ks, ke = chunk.cu_seqlen_ks.cpu(), chunk.cu_seqlen_ke.cpu()
        row = 0
        for q, (context, new) in enumerate(lens):
            start = int(chunk.cu_seq_lens[q])
            assert torch.equal(ks[row : row + new], torch.full((new,), start))
            assert torch.equal(
                ke[row : row + new],
                start + torch.arange(context + 1, context + new + 1),
            )
            row += new
        assert row == ks.numel() and chunk.total_seq_lens == sum(seq_lens)


def test_workspace_holds_a_max_model_len_request():
    """DCP ShardQ has no capacity knob: the workspace fits one request of
    ``max_model_len`` tokens, plus a page of rounding per request."""
    from vllm.v1.attention.ops.dcp_shardq import context as dcp

    config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=131072))
    assert dcp.workspace_max_tokens(config) == 131072
    assert dcp.capacity_pages(dcp.workspace_max_tokens(config), 8) == 2048 + 8


def test_stats_report_paths_since_last_report():
    from vllm.v1.attention.ops.dcp_shardq import context as dcp

    stats = dcp.DcpShardQStats(log=False)
    stats.record(dcp.SHARDQ, 16384)
    stats.record(dcp.SHARDQ, 8000)
    stats.record(dcp.DENSE_MHA, 600)
    stats.record(dcp.WORKSPACE, 15000)
    assert stats.total == {
        dcp.SHARDQ: [2, 24384],
        dcp.DENSE_MHA: [1, 600],
        dcp.WORKSPACE: [1, 15000],
    }
    assert stats.report() == (
        "DCP ShardQ prefill: 2/4 steps, 24384/39984 tokens (61.0%; 61.0% since "
        "start) on ShardQ; DCP path: dense MHA 1 steps / 600 tokens, "
        "workspace full 1 steps / 15000 tokens"
    )
    stats.recent.clear()
    stats.record(dcp.FEW_TOKENS, 900)
    assert stats.report() == (
        "DCP ShardQ prefill: 0/1 steps, 0/900 tokens (0.0%; 59.6% since start) "
        "on ShardQ; DCP path: few tokens 1 steps / 900 tokens"
    )


@pytest.mark.parametrize("num_spec", [0, 3])
def test_planner_follows_the_indexer_decode_threshold(num_spec):
    """A step's leading decode rows (at most 1 + num_speculative_tokens tokens,
    which the indexer scores on its decode path) keep the DCP path; the rest
    of the step is sharded, a later short row included. Steps without prefill
    rows are not counted; padded steps, zero-length rows included, are
    reported as such."""
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )
    from vllm.v1.attention.ops.dcp_shardq import context as dcp

    speculative_config = (
        SimpleNamespace(num_speculative_tokens=num_spec) if num_spec else None
    )
    threshold = DeepseekV32IndexerMetadataBuilder.get_decode_threshold(
        SimpleNamespace(speculative_config=speculative_config)
    )
    assert threshold == 1 + num_spec
    world, rank = 8, 3
    planner = dcp.DcpShardQPlanner(
        _layout("fp8"),
        1 << 12,
        1 << 20,
        1 << 30,
        64,
        threshold,
        world,
        rank,
        torch.device("cuda"),
        label="layers.0, 1 layers",
    )
    long = [(1000, 300), (0, 200)]

    def plan(lens, extra_tokens=0):
        table = torch.zeros(len(lens), 1, dtype=torch.int32, device="cuda")
        metadata = _common_metadata(lens, table)
        if extra_tokens:
            metadata = dataclasses.replace(
                metadata, num_actual_tokens=metadata.num_actual_tokens + extra_tokens
            )
        return planner.plan(metadata, use_dense_mha=False)

    decodes = [(500, threshold), (70, 1)]
    planned = plan([*decodes, *long])
    assert planned is not None
    split, context = planned
    assert context.num_decodes == 2
    assert context.num_decode_tokens == threshold + 1
    assert sum(split.counts) == 500
    assert context.world == world and context.rank == rank
    # A short row after the first prefill row is a prefill row.
    planned = plan([*long, (500, threshold)])
    assert planned is not None
    assert planned[1].num_decodes == 0
    assert sum(planned[0].counts) == 500 + threshold
    assert plan(decodes) is None
    assert plan([(500, 0), *long]) is None
    assert plan(long, extra_tokens=8) is None
    assert {path: c[0] for path, c in planner.stats.total.items()} == {
        dcp.SHARDQ: 2,
        dcp.PADDED: 2,
    }
    assert planner.stats.report().startswith(
        "DCP ShardQ prefill [layers.0, 1 layers]: 2/4 steps"
    )


def _flashinfer_impl(cache_dtype: str, kernel: str, dcp_world_size: int):
    """A FlashInfer sparse MLA impl at K3.1's MLA sizes on ``kernel``."""
    from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as fims

    impl = object.__new__(fims.FlashInferMLASparseImpl)
    impl.dcp_world_size = dcp_world_size
    impl.dcp_rank = 0
    impl.scale = 192**-0.5
    impl.kv_cache_dtype = cache_dtype
    impl.kv_lora_rank, impl.qk_nope_head_dim, impl.qk_rope_head_dim = 512, 128, 64
    impl.is_nope_mla = False
    impl.index_group = None
    impl.need_to_return_lse_for_decode = dcp_world_size > 1
    impl._workspace_buffer = impl.bmm1_scale = impl.bmm2_scale = None
    impl.use_prims_ts = kernel == "prims_ts"
    if impl.use_prims_ts:
        impl.lse_base_on_e = True
        impl.prims_ts_format = fims._PRIMS_TS_FORMATS[cache_dtype]
        impl.prims_ts_buckets = tuple(1 << i for i in range(13))
    return impl


# The attention layer's static per-tensor query and KV scales.
SCALES = SimpleNamespace(_q_scale_float=1.0, _k_scale_float=1.0)


def _fp8_query(num_tokens: int, num_heads: int) -> torch.Tensor:
    """A query as the layer quantizes it for the FlashInfer kernels."""
    q = torch.randn(num_tokens, num_heads, 576, device="cuda")
    return q.to(torch.float8_e4m3fn)


def _real_rows(rep: Replicated, cache_dtype: str):
    """Write real MLA and indexer rows of every token into ``rep``; returns
    the new tokens' latents (to write them as forward_impl does)."""
    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer import (
        indexer_k_norm_quant_insert,
    )

    seq_lens = [c + n for c, n in rep.lens]
    slots = torch.from_numpy(
        np.concatenate(
            [p * PAGE + np.arange(s) for p, s in zip(_page_start(rep.lens), seq_lens)]
        )
    ).cuda()
    torch.manual_seed(1)
    kv_c = torch.randn(len(slots), 512, device="cuda").bfloat16()
    k_pe = torch.randn(len(slots), 64, device="cuda").bfloat16()
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    ops.concat_and_cache_mla(kv_c, k_pe, rep.mla, slots, cache_dtype, scale)
    norm = torch.ones(INDEX_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    # The insert also clears that many top-k rows.
    scratch = torch.empty(len(slots), TOPK, dtype=torch.int32, device="cuda")
    indexer_k_norm_quant_insert(
        torch.randn(len(slots), INDEX_HEAD_DIM, device="cuda").bfloat16(),
        norm,
        torch.zeros_like(norm),
        1e-6,
        slots,
        rep.index,
        scratch,
        len(slots),
    )
    is_new = torch.from_numpy(
        np.concatenate([np.arange(s) >= c for (c, _), s in zip(rep.lens, seq_lens)])
    ).cuda()
    return kv_c[is_new], k_pe[is_new], scale


def _separable_decode_keys(rep: Replicated, num_decodes: int) -> torch.Tensor:
    """Rewrite the indexer keys of the first ``num_decodes`` (one-token)
    requests of ``rep`` so that their top-k is unique: TOPK random positions
    get the key +v and the others -v, so a query along v scores TOPK keys
    positive and the rest zero (after the ReLU), with no tie at the cut-off.
    Returns the sorted top-k positions, [num_decodes, TOPK]."""
    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer import (
        indexer_k_norm_quant_insert,
    )

    gen = torch.Generator().manual_seed(3)
    v = _decode_query_direction()
    expected = []
    for q, (context, new) in enumerate(rep.lens[:num_decodes]):
        seq_len = context + new
        chosen = torch.randperm(seq_len, generator=gen)[:TOPK].cuda()
        sign = torch.full((seq_len, 1), -1.0, device="cuda")
        sign[chosen] = 1.0
        page = int(_page_start(rep.lens)[q])
        slots = page * PAGE + torch.arange(seq_len, device="cuda")
        norm = torch.ones(INDEX_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        scratch = torch.empty(seq_len, TOPK, dtype=torch.int32, device="cuda")
        indexer_k_norm_quant_insert(
            (sign * v).bfloat16(),
            norm,
            torch.zeros_like(norm),
            1e-6,
            slots,
            rep.index,
            scratch,
            seq_len,
        )
        expected.append(torch.sort(chosen.int()).values)
    if not expected:
        return torch.empty((0, TOPK), dtype=torch.int32, device="cuda")
    return torch.stack(expected)


def _decode_query_direction() -> torch.Tensor:
    gen = torch.Generator().manual_seed(4)
    return torch.randn(INDEX_HEAD_DIM, generator=gen).cuda()


def _random_topk(lens) -> tuple[torch.Tensor, torch.Tensor]:
    """Random causal top-k of every new token (-1 pads short rows) and the
    request of each new token."""
    positions = np.concatenate([np.arange(c, c + n) for c, n in lens])
    req_id = np.repeat(np.arange(len(lens)), [n for _, n in lens])
    gen = torch.Generator().manual_seed(2)
    topk = torch.full((len(positions), TOPK), -1, dtype=torch.int32)
    for t, pos in enumerate(positions):
        valid = min(pos + 1, TOPK)
        topk[t, :valid] = torch.randperm(pos + 1, generator=gen)[:valid]
    return topk.cuda(), torch.from_numpy(req_id).to(torch.int32).cuda()


@pytest.mark.parametrize(("cache_dtype", "kernel"), KERNEL_CASES)
def test_attention_over_workspace_matches_replicated_cache(cache_dtype, kernel):
    """New MLA rows written through the plan's slot mapping complete the
    workspace; the kernel then attends it bit-identically to the replicated
    cache, through the same block table."""
    from vllm import _custom_ops as ops
    from vllm.v1.attention.backends.mla.sparse_utils import (
        flat_kv_row_view,
        triton_convert_req_index_to_global_index,
    )

    lens, world = SCENARIOS["page_tails"], 8
    rep = Replicated(lens, _layout(cache_dtype), fill_random=False)
    kv_c, k_pe, scale = _real_rows(rep, cache_dtype)
    workspaces, plans = _push_all(rep, world, num_dst=1)
    _, ws_mla, ws_index = workspaces[0]
    # The step's new tokens, as forward_impl writes them on every rank.
    ops.concat_and_cache_mla(
        kv_c, k_pe, ws_mla, plans[0].slot_mapping, cache_dtype, scale
    )
    _check_workspace(ws_mla, ws_index, rep, new_mla_rows=True, fill=True)

    topk, req_id = _random_topk(lens)
    q = _fp8_query(topk.shape[0], NUM_HEADS)
    impl = _flashinfer_impl(cache_dtype, kernel, 1)

    def attend(cache):
        _, block_stride_rows = flat_kv_row_view(cache, PAGE)
        physical_topk, valid_counts = triton_convert_req_index_to_global_index(
            req_id,
            plans[0].block_table,
            topk,
            BLOCK_SIZE=PAGE,
            BLOCK_STRIDE_ROWS=block_stride_rows,
            NUM_TOPK_TOKENS=TOPK,
            return_valid_counts=True,
        )
        out, lse = impl._forward_mqa_kernel(
            q,
            cache,
            physical_topk,
            valid_counts,
            layer=SCALES,
            block_size=PAGE,
            is_decode=False,
        )
        assert lse is None
        return out

    out = attend(ws_mla)
    assert torch.isfinite(out).all()
    assert torch.equal(out, attend(rep.mla))


def _distributed_worker(rank, world, init_file, cache_dtype, kernel):
    """One rank of a DCP = TP group: the production gather, indexer and
    attention over the gathered context against TP ShardQ over the replicated
    caches, bit for bit, for two layers through the same workspace."""
    from vllm import _custom_ops as ops
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.forward_context import create_forward_context, override_forward_context
    from vllm.model_executor.layers.attention.sparse_mla_attention import (
        SparseMLAShardQ,
    )
    from vllm.model_executor.layers.sparse_attn_indexer import sparse_attn_indexer
    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer import (
        indexer_q_quant_fold_weights,
    )
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadata,
        DeepseekV32IndexerPrefillMetadata,
        build_prefill_chunk_metadata,
    )
    from vllm.v1.attention.ops.shardq import indexer_rows, make_shardq_split
    from vllm.v1.worker.workspace import init_workspace_manager

    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    config = VllmConfig()
    config.parallel_config.tensor_parallel_size = world
    config.parallel_config.decode_context_parallel_size = world
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"file://{init_file}",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=world, decode_context_model_parallel_size=world
        )
        init_workspace_manager(device)
        # Requests within the top-k, so every row selects all its keys: the
        # top-k kernels break ties between equal scores nondeterministically.
        lens = [(1000, 300), (513, 64), (0, 700), (1, 63)]
        layout = _layout(cache_dtype)
        seq_lens = [c + n for c, n in lens]
        num_tokens = sum(n for _, n in lens)
        rep = Replicated(lens, layout, fill_random=False)
        kv_c, k_pe, scale = _real_rows(rep, cache_dtype)
        plan = _build_context(lens, layout, world, rank)
        tp_plan = _build_context(lens, layout, 1, 0)

        torch.manual_seed(4)
        q_fp8, weights = indexer_q_quant_fold_weights(
            torch.randn(num_tokens, 32, INDEX_HEAD_DIM, device="cuda").bfloat16(),
            torch.randn(num_tokens, 32, device="cuda").bfloat16(),
            INDEX_HEAD_DIM**-0.5 * 32**-0.5,
            32,
        )
        hidden = torch.randn(num_tokens, 256, device="cuda").bfloat16()
        q = _fp8_query(num_tokens, NUM_HEADS // world)
        qsl = np.r_[0, np.cumsum([n for _, n in lens])]
        split = make_shardq_split(qsl, world, rank, device)
        rows_cpu, rows = indexer_rows(split, 0)
        req_id = torch.from_numpy(
            np.repeat(np.arange(len(lens)), np.diff(qsl)).astype(np.int32)
        ).cuda()
        seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int32)
        topk = torch.full((num_tokens, TOPK), -1, dtype=torch.int32, device="cuda")

        def dcp_local_chunks(index_table):
            """The indexer builder's DCP-local chunk of the step, which a DCP
            ShardQ step replaces."""
            return [
                build_prefill_chunk_metadata(
                    0,
                    len(lens),
                    torch.from_numpy(qsl).int().cuda(),
                    torch.from_numpy(qsl).int(),
                    seq_lens_cpu.cuda(),
                    seq_lens_cpu.cuda(),
                    seq_lens_cpu,
                    index_table,
                    1,
                    dcp_rank=rank,
                    dcp_world_size=world,
                )
            ]

        def run(gathered_context, mla, mla_table, index, index_table, chunks, dcp_size):
            shardq = SparseMLAShardQ(
                split, req_id[split.local_idx], 0, rows_cpu, rows, gathered_context
            )
            mla_metadata = SimpleNamespace(
                block_size=mla.shape[1],
                block_table=mla_table,
                req_id_per_token=req_id,
                num_decode_tokens=0,
                num_actual_tokens=num_tokens,
                cp_kv_cache_interleave_size=1,
                shardq=shardq,
            )
            index_metadata = DeepseekV32IndexerMetadata(
                seq_lens=seq_lens_cpu.cuda(),
                max_seq_len=max(seq_lens),
                slot_mapping=torch.empty(0, dtype=torch.int64, device="cuda"),
                num_decodes=0,
                num_decode_tokens=0,
                num_prefills=len(lens),
                num_prefill_tokens=num_tokens,
                prefill=DeepseekV32IndexerPrefillMetadata(chunks),
                block_table=index_table,
            )
            config.compilation_config.static_forward_context["mla"] = SimpleNamespace(
                kv_cache=mla
            )
            context = create_forward_context(
                {"idx": index_metadata, "mla": mla_metadata}, config
            )
            with override_forward_context(context):
                sparse_attn_indexer(
                    hidden,
                    "idx",
                    index,
                    q_fp8,
                    None,
                    None,
                    weights,
                    INDEX_HEAD_DIM,
                    "ue8m0",
                    TOPK,
                    INDEX_HEAD_DIM,
                    1 << 20,
                    1 << 20,
                    topk,
                    True,
                    False,
                    False,
                    "",
                    False,
                    rank,
                    dcp_size,
                    1,
                    shardq_layer_name="mla",
                )
            # A canonical order, so that both attentions read the same rows.
            local_topk = torch.sort(topk[: split.num_local], dim=1).values
            topk[: split.num_local] = local_topk
            impl = _flashinfer_impl(cache_dtype, kernel, dcp_size)
            impl.topk_indices_buffer = topk
            if gathered_context is not None:
                # forward_impl's write of the new tokens' MLA rows.
                ops.concat_and_cache_mla(
                    kv_c,
                    k_pe,
                    gathered_context.mla_view(mla),
                    gathered_context.slot_mapping,
                    cache_dtype,
                    scale,
                )
            out = impl.forward_mqa_shardq(q, mla, mla_metadata, SCALES)
            return local_topk, out.clone()

        try:
            tp_topk, tp_out = run(
                None,
                rep.mla,
                tp_plan.block_table,
                rep.index,
                tp_plan.block_table,
                tp_plan.indexer_chunks,
                1,
            )
            for layer in range(2):  # The second layer reuses the workspace.
                shard = _shard(rep, world, rank, seed=rank + 7 * layer)
                dcp_topk, dcp_out = run(
                    plan,
                    shard.mla,
                    shard.mla_table,
                    shard.index,
                    shard.index_table,
                    dcp_local_chunks(shard.index_table),
                    world,
                )
                _check_workspace(
                    plan.mla_view(shard.mla),
                    plan.index_view(shard.index),
                    rep,
                    new_mla_rows=True,
                    fill=False,
                )
                assert torch.equal(dcp_topk, tp_topk)
                assert torch.equal(dcp_out, tp_out)
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.skipif(torch.accelerator.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize(("cache_dtype", "kernel"), KERNEL_CASES)
def test_gathered_context_step_matches_tp_shardq(cache_dtype, kernel, tmp_path):
    import torch.multiprocessing as mp

    world = min(torch.accelerator.device_count(), 4)
    mp.spawn(
        _distributed_worker,
        args=(world, str(tmp_path / "dist_init"), cache_dtype, kernel),
        nprocs=world,
    )


# A Kimi K3.1 DSA layer at DeepSeek-V2-Lite's MLA sizes (16 heads, NoPE latent
# rows with the 64-dim RoPE slot), on the model config the builders read.
LAYER_MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"
LAYER_HEADS = 16
LAYER_BLOCK = 4 * PAGE
# Requests above the decode threshold of 4-8 local heads (1024 query tokens),
# longer than the top-k so the dense-MHA prefill never applies.
LAYER_LENS = [(3000, 1100), (0, 2200), (700, 1500)]
# Prefill steps that run as DCP ShardQ steps. A reordered batch starts with
# its pieces of at most 1024 tokens, the leading ones of which the sparse MLA
# backend takes as (uniform) decodes; the indexer takes them as prefill rows,
# except one-token decode rows, which keep the DCP path in a mixed step.
# Every sequence is longer than the top-k, so dense MHA never applies.
LAYER_STEPS = {
    "long_pieces": LAYER_LENS,
    "short_and_long_pieces": [(900, 300), (2500, 64), *LAYER_LENS],
    "short_pieces_only": [(3000, 700), (2000, 500)],
    "decodes_and_pieces": [(6000, 1), (2100, 1), (900, 300), *LAYER_LENS],
}


def _leading(lens, max_query_len: int) -> int:
    """Requests before the step's first one of more than ``max_query_len``
    tokens."""
    return next((i for i, (_, n) in enumerate(lens) if n > max_query_len), len(lens))


def _dsa_layer(config, topk_buffer, layer_index=0):
    """A real KimiK3DSAAttention, its MLA layer's weights processed."""
    from vllm.models.kimi_k3.nvidia.dsa import KimiK3DSAAttention
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

    text_config = KimiLinearConfig(
        hidden_size=256,
        num_attention_heads=LAYER_HEADS,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_use_nope=True,
        dsa_config={
            "dsa_layers": [0],
            "index_topk": TOPK,
            "index_head_dim": INDEX_HEAD_DIM,
            "index_num_heads": 32,
            "index_rope_dim": 0,
            "index_k_norm_type": "layernorm",
        },
    )
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        attn = KimiK3DSAAttention(
            text_config,
            hidden_size=256,
            num_heads=LAYER_HEADS,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            q_lora_rank=None,
            kv_lora_rank=512,
            cache_config=config.cache_config,
            prefix=f"model.layers.{layer_index}.self_attn",
            topk_indices_buffer=topk_buffer,
        )
    torch.set_default_dtype(torch.float32)
    layer = attn.mla_attn.mla_attn
    torch.manual_seed(5)  # The same weights on every rank's slice.
    weight = torch.randn(
        LAYER_HEADS * 256, 512, device="cuda", dtype=torch.bfloat16
    ).chunk(layer.impl.dcp_world_size)[layer.impl.dcp_rank]
    layer.kv_b_proj.weight.data.copy_(weight * 0.05)
    layer.process_weights_after_loading(torch.bfloat16)
    return attn


def _common_metadata(lens, block_table, dcp=None):
    """The step's metadata, with the DCP-local sequence lengths that the runner
    adds on rank ``dcp[1]`` of a DCP group of ``dcp[0]``."""
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

    new = [n for _, n in lens]
    qsl = torch.tensor(np.r_[0, np.cumsum(new)], dtype=torch.int32)
    seq_lens = torch.tensor([c + n for c, n in lens], dtype=torch.int32)
    num_tokens = int(qsl[-1])
    local = None if dcp is None else get_dcp_local_seq_lens(seq_lens, *dcp)
    return CommonAttentionMetadata(
        query_start_loc=qsl.cuda(),
        query_start_loc_cpu=qsl,
        seq_lens=seq_lens.cuda(),
        num_reqs=len(lens),
        num_actual_tokens=num_tokens,
        max_query_len=max(new),
        max_seq_len=int(seq_lens.max()),
        block_table_tensor=block_table,
        # The layer's own cache writes are not under test.
        slot_mapping=torch.full((num_tokens,), -1, dtype=torch.int64, device="cuda"),
        seq_lens_cpu_upper_bound=seq_lens,
        dcp_local_seq_lens=None if local is None else local.cuda(),
        dcp_local_seq_lens_cpu_upper_bound=local,
    )


def _layer_forward(layer, q, kv_c, k_pe, kv_cache, metadata, q_dcp_replicated=None):
    out = torch.empty(q.shape[0], q.shape[1] * 128, device="cuda", dtype=torch.bfloat16)
    return layer.forward_impl(
        q,
        kv_c,
        k_pe.unsqueeze(1),
        kv_cache,
        metadata,
        out,
        q_dcp_replicated=q_dcp_replicated,
    )


def _min_topk_recall(topk: torch.Tensor, ref: torch.Tensor) -> float:
    """The smallest share of a row's ``ref`` keys that ``topk`` selects."""
    keys = torch.sort(topk, dim=1).values.contiguous()
    at = torch.searchsorted(keys, ref.contiguous()).clamp(max=keys.shape[1] - 1)
    found = ((keys.gather(1, at) == ref) & (ref >= 0)).sum(1).float()
    return float((found / (ref >= 0).sum(1).clamp(min=1)).min())


def _indexer_topk(
    indexer, mla_name, config, topk_buffer, inputs, mla_metadata, index_metadata
):
    """A DSA layer's indexer: the top-k of a step's rows (of this rank's
    tokens on a ShardQ step) into the buffer, which the fused prologue clears
    first."""
    from vllm.forward_context import create_forward_context, override_forward_context

    hidden, index_q, index_weights = inputs
    topk_buffer.fill_(-1)
    context = create_forward_context(
        {mla_name: mla_metadata, indexer.k_cache.prefix: index_metadata}, config
    )
    with override_forward_context(context):
        indexer.score(hidden, index_q, index_weights, index_k=None)


def _layer_worker(rank, world, init_file, cache_dtype, kernel):
    """One rank of a DCP = TP group running a real K3.1 DSA layer.

    The model and its metadata builders register their JIT kernels with a
    warmup registry, which warms them through the runner's path. The MLA
    builder then engages DCP ShardQ on prefill steps (1024-token MLA kernel
    blocks being whole pages), short pieces included. The layer's indexer
    (the sparse_attn_indexer op, on the indexer builder's metadata) gathers
    the context and scores this rank's tokens over it, and
    MLAAttention.forward_impl over the gathered context (with the new tokens'
    rows it writes there) matches the DCP path it replaces, top-k included,
    without compiling any kernel; a mixed step's decode rows keep the DCP
    path. Steps with too few prefill tokens or beyond the workspace keep the
    DCP path, the builder counts each path, steps without prefill rows are not
    counted, and a replicated query is ignored. Interleave-64 DCP and
    disaggregated decode instances keep DCP ShardQ off.
    """
    from tests.v1.attention.utils import create_vllm_config
    from vllm.config import CUDAGraphMode, KVTransferConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.utils import jit_monitor
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseTRTLLMBackend,
    )
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.ops.dcp_shardq import context as dcp
    from vllm.v1.worker.block_table import get_block_table_width
    from vllm.v1.worker.workspace import init_workspace_manager

    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    max_tokens = max(sum(c + n for c, n in lens) for lens in LAYER_STEPS.values())
    config = create_vllm_config(
        model_name=LAYER_MODEL,
        tensor_parallel_size=world,
        max_model_len=8192,
        block_size=LAYER_BLOCK,
        max_num_seqs=8,
        max_num_batched_tokens=8192,
        hf_config_override={"index_topk": TOPK},
    )
    config.parallel_config.decode_context_parallel_size = world
    config.cache_config.cache_dtype = cache_dtype
    config.attention_config.backend = AttentionBackendEnum.FLASHINFER_MLA_SPARSE
    config.kernel_config.flashinfer_sparse_mla_kernel = kernel
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    attention_config = config.attention_config
    attention_config.sparse_mla_shard_q = True
    dcp.workspace_max_tokens = lambda _: max_tokens
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"file://{init_file}",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=world, decode_context_model_parallel_size=world
        )
        init_workspace_manager(device)
        try:
            topk_buffer = torch.full((8192, TOPK), -1, dtype=torch.int32, device="cuda")
            # The runner collects registrations while it builds the model and
            # the metadata builders, then warms them.
            registry = JitWarmupRegistry(config)
            with registry.activate():
                attn = _dsa_layer(config, topk_buffer)
                layer, indexer = attn.mla_attn.mla_attn, attn.dsa_indexer
                assert layer.use_shardq
                assert indexer.shardq_layer_name == layer.layer_name
                spec = layer.get_kv_cache_spec(config)
                assert spec.block_size == LAYER_BLOCK
                builder = FlashInferMLASparseTRTLLMBackend.get_builder_cls()(
                    spec, [layer.layer_name], config, device
                )
                # The indexer builder sees its 64-token kernel blocks.
                index_spec = dataclasses.replace(
                    indexer.k_cache.get_kv_cache_spec(config), block_size=PAGE
                )
                index_builder = DeepseekV32IndexerMetadataBuilder(
                    index_spec,
                    [indexer.k_cache.prefix],
                    config,
                    device,
                    block_table_width=get_block_table_width(
                        index_spec.max_num_blocks_per_req(config, 8192),
                        index_spec.block_size,
                    ),
                )
            # The runner's warmup path (keyword registrations included).
            registry.warmup()
            assert builder.dcp_shardq is not None
            # A kernel that a DCP ShardQ step compiles after warmup, beyond
            # those the same step compiles on the DCP path (which the
            # runner's dummy runs also exercise), is a warmup miss.
            compiled: list[str] = []
            monitoring = [False]

            def record(*, backend, event, fn_name, detail=None):
                if monitoring[0]:
                    compiled.append(f"{backend} {event}: {fn_name}")

            jit_monitor._handle_jit_event = record
            jit_monitor.activate()

            # The DCP path on the same steps, from a builder whose workspace is
            # too small for them.
            dcp.workspace_max_tokens = lambda _: 1024
            small = FlashInferMLASparseTRTLLMBackend.get_builder_cls()(
                spec, [layer.layer_name], config, device
            )
            for name, lens in LAYER_STEPS.items():
                rep = Replicated(lens, _layout(cache_dtype), fill_random=False)
                kv_c, k_pe, _ = _real_rows(rep, cache_dtype)
                num_decodes = _leading(lens, 1)
                num_decode_tokens = sum(n for _, n in lens[:num_decodes])
                expected_decode_topk = _separable_decode_keys(rep, num_decodes)
                shard = _shard(rep, world, rank, seed=rank, mla_block=LAYER_BLOCK)
                step = _common_metadata(lens, shard.mla_table, (world, rank))
                num_tokens = kv_c.shape[0]
                torch.manual_seed(6)
                q_all = torch.randn(
                    num_tokens, LAYER_HEADS, 192, device="cuda", dtype=torch.bfloat16
                )
                q = q_all.chunk(world, dim=1)[rank].contiguous()
                forward = partial(_layer_forward, layer, q, kv_c, k_pe, shard.mla)
                # The indexer's inputs (its keys are in the cache already).
                index_q = torch.randn(
                    num_tokens, 32, INDEX_HEAD_DIM, device="cuda", dtype=torch.bfloat16
                )
                index_weights = torch.randn(
                    num_tokens, 32, device="cuda", dtype=torch.bfloat16
                )
                # The decode rows' queries along the separating direction.
                index_q[:num_decode_tokens] = _decode_query_direction().bfloat16()
                index_weights[:num_decode_tokens] = 1.0
                hidden = torch.empty(num_tokens, 256, device="cuda").bfloat16()
                layer.kv_cache = shard.mla
                indexer.k_cache.kv_cache = shard.index
                index_step = _common_metadata(lens, shard.index_table, (world, rank))

                score = partial(
                    _indexer_topk,
                    indexer,
                    layer.layer_name,
                    config,
                    topk_buffer,
                    (hidden, index_q, index_weights),
                )

                dcp_metadata = small.build(0, step)
                assert dcp_metadata.shardq is None
                score(dcp_metadata, index_builder.build(0, index_step))
                dcp_topk = topk_buffer[:num_tokens].clone()
                # The DCP path's kernels, which the runner's dummy runs warm.
                forward(dcp_metadata)

                monitoring[0] = True
                metadata = builder.build(0, step)
                index_metadata = index_builder.build(0, index_step)
                # Before admitting short pieces, MLA decodes kept the DCP path.
                assert metadata.num_decodes == _leading(lens, 1024), name
                # The indexer scores the one-token rows as decodes, the rest
                # as prefill rows.
                assert index_metadata.num_decode_tokens == num_decode_tokens, name
                shardq = metadata.shardq
                assert shardq is not None and shardq.gathered_context is not None, name
                assert shardq.gathered_context.num_decodes == num_decodes
                assert shardq.num_decode_tokens == num_decode_tokens
                assert sum(shardq.split.counts) == num_tokens - num_decode_tokens

                # DCP ShardQ: the indexer gathers the prefill rows' context
                # and writes the top-k rows of this rank's prefill tokens after
                # the decode rows, which it scores on the DCP path in place.
                score(metadata, index_metadata)
                local = num_decode_tokens + shardq.split.local_idx
                num_rows = num_decode_tokens + shardq.split.num_local
                decode_topk = topk_buffer[:num_decode_tokens].clone()
                local_topk = topk_buffer[num_decode_tokens:num_rows].clone()
                out = forward(metadata).clone()
                torch.accelerator.synchronize()
                monitoring[0] = False
                assert not compiled, (name, compiled)
                # The keys of the DCP path's cross-rank top-k merge, up to the
                # order of near-tied (bf16) scores at the top-k boundary.
                assert _min_topk_recall(local_topk, dcp_topk[local]) > 0.99, name
                # The decode rows' top-k, on both paths: exactly the keys that
                # score positive.
                for topk_rows in (decode_topk, dcp_topk[:num_decode_tokens]):
                    assert torch.equal(
                        torch.sort(topk_rows, dim=1).values, expected_decode_topk
                    ), name
                # The prefill rows' MLA rows are complete in the workspace.
                _check_workspace(
                    shardq.gathered_context.mla_view(shard.mla),
                    shardq.gathered_context.index_view(shard.index),
                    rep,
                    new_mla_rows=True,
                    fill=False,
                    num_decodes=num_decodes,
                )
                # A replicated query (DCP query replication) is left aside.
                q_rep = get_tp_group().all_gather(q, dim=1)
                assert torch.equal(forward(metadata, q_rep), out)

                # The DCP path over the same keys: every rank's top-k rows.
                topk = torch.zeros_like(dcp_topk)
                topk[local] = local_topk + 1
                torch.distributed.all_reduce(topk, group=get_tp_group().device_group)
                topk_buffer[:num_tokens] = topk - 1
                topk_buffer[:num_decode_tokens] = decode_topk
                expected = forward(dcp_metadata)
                # The FlashInfer kernels round the softmax weights to E4M3, the
                # DCP path once per rank's partial before the LSE merge: each
                # path is within 2e-2 of an fp32 reference
                # (test_flashinfer_mla_sparse_prims_ts), and they differ by
                # about 2.1e-2.
                err = (out.float() - expected.float()).norm() / expected.float().norm()
                assert err < 3e-2, (name, err)

            # The next layer's gather waits for every rank to finish reading
            # this one: rank 0 lags, and its copy must still hold this layer.
            context = shardq.gathered_context
            next_shard = _shard(
                Replicated(lens, rep.layout), world, rank, 1, LAYER_BLOCK
            )
            if rank == 0:
                torch.cuda._sleep(500_000_000)
            snapshot = [
                context.mla_view(shard.mla).clone(),
                context.index_view(shard.index).clone(),
            ]
            context.gather(
                next_shard.mla,
                next_shard.mla_table,
                next_shard.index,
                next_shard.index_table,
            )
            _check_workspace(
                *snapshot,
                rep,
                new_mla_rows=True,
                fill=False,
                num_decodes=context.num_decodes,
            )

            # Steps that keep the DCP path (each fits the workspace): fewer
            # than sparse_mla_shard_q_min_tokens prefill tokens (after a
            # decode row), and decode rows alone, which are not counted.
            for lens, path in (
                ([(100, 1), (3000, 600), (2900, 300)], dcp.FEW_TOKENS),
                ([(100, 1), (3000, 1)], None),
            ):
                rep = Replicated(lens, _layout(cache_dtype), fill_random=False)
                shard = _shard(rep, world, rank, seed=rank, mla_block=LAYER_BLOCK)
                before = dict(builder.dcp_shardq.stats.total)
                assert (
                    builder.build(
                        0, _common_metadata(lens, shard.mla_table, (world, rank))
                    ).shardq
                    is None
                )
                after = builder.dcp_shardq.stats.total
                if path is None:
                    assert after == before
                else:
                    assert after[path][0] == before.get(path, [0, 0])[0] + 1
            stats = builder.dcp_shardq.stats
            assert stats.total[dcp.SHARDQ][0] == len(LAYER_STEPS)
            assert small.dcp_shardq.stats.total[dcp.WORKSPACE][0] == len(LAYER_STEPS)
            assert set(stats.total) == {dcp.SHARDQ, dcp.FEW_TOKENS}

            # The gathered context holds DCP interleave-1 shards only.
            config.parallel_config.cp_kv_cache_interleave_size = 64
            next_attn = _dsa_layer(config, topk_buffer, layer_index=1)
            assert not next_attn.mla_attn.mla_attn.use_shardq
            # A disaggregated decode instance reserves no workspace.
            config.parallel_config.cp_kv_cache_interleave_size = 1
            config.kv_transfer_config = KVTransferConfig(
                kv_connector="NixlConnector", kv_role="kv_consumer"
            )
            decode_attn = _dsa_layer(config, topk_buffer, layer_index=2)
            assert not decode_attn.mla_attn.mla_attn.use_shardq
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.skipif(torch.accelerator.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize(("cache_dtype", "kernel"), KERNEL_CASES)
def test_dsa_layer_runs_dcp_prefill_as_shardq(cache_dtype, kernel, tmp_path):
    import torch.multiprocessing as mp

    world = min(torch.accelerator.device_count(), 4)
    mp.spawn(
        _layer_worker,
        args=(world, str(tmp_path / "dist_init"), cache_dtype, kernel),
        nprocs=world,
    )


def _profile_worker(rank, world, init_file, cache_dtype, kernel):
    """One rank of a DCP = TP prefill instance with DCP ShardQ on the a2a DCP
    backend, whose steps rarely keep the DCP path.

    Its memory profile run (the layer without attention metadata) runs the DCP
    path's collectives at the profile's size, beyond the direct workspace, so
    the NCCL A2A connects its peers and allocates its buffers there, and warms
    the A2A's Triton kernels: a later step that keeps the DCP path (here, one
    that overflows the gathered-context workspace) compiles none of them.
    """
    from tests.v1.attention.utils import create_vllm_config
    from vllm.config import CUDAGraphMode, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.utils import jit_monitor
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseTRTLLMBackend,
    )
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.ops.dcp import get_dcp_workspace_max_num_tokens
    from vllm.v1.attention.ops.dcp_shardq import context as dcp
    from vllm.v1.worker.block_table import get_block_table_width
    from vllm.v1.worker.workspace import init_workspace_manager

    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    config = create_vllm_config(
        model_name=LAYER_MODEL,
        tensor_parallel_size=world,
        max_model_len=8192,
        block_size=LAYER_BLOCK,
        max_num_seqs=8,
        max_num_batched_tokens=8192,
        hf_config_override={"index_topk": TOPK},
    )
    config.parallel_config.decode_context_parallel_size = world
    config.parallel_config.dcp_comm_backend = "a2a"
    config.cache_config.cache_dtype = cache_dtype
    config.attention_config.backend = AttentionBackendEnum.FLASHINFER_MLA_SPARSE
    config.kernel_config.flashinfer_sparse_mla_kernel = kernel
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    attention_config = config.attention_config
    attention_config.sparse_mla_shard_q = True
    # Too small for the step below, which keeps the DCP path.
    dcp.workspace_max_tokens = lambda _: 1024
    # Three requests: another power-of-two bucket than the profile run's one.
    lens = LAYER_LENS
    num_tokens = sum(n for _, n in lens)
    assert num_tokens > get_dcp_workspace_max_num_tokens(config)
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"file://{init_file}",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=world, decode_context_model_parallel_size=world
        )
        init_workspace_manager(device)
        a2a_sizes: list[int] = []
        nccl_a2a = torch.distributed.all_to_all_single

        def a2a(output, input, *args, **kwargs):
            a2a_sizes.append(input.numel())
            return nccl_a2a(output, input, *args, **kwargs)

        torch.distributed.all_to_all_single = a2a
        gather_sizes: list[int] = []
        try:
            topk_buffer = torch.full((8192, TOPK), -1, dtype=torch.int32, device="cuda")
            registry = JitWarmupRegistry(config)
            with registry.activate():
                attn = _dsa_layer(config, topk_buffer)
                layer, indexer = attn.mla_attn.mla_attn, attn.dsa_indexer
                assert layer.use_shardq
                spec = layer.get_kv_cache_spec(config)
                builder = FlashInferMLASparseTRTLLMBackend.get_builder_cls()(
                    spec, [layer.layer_name], config, device
                )
                # The indexer builder sees its 64-token kernel blocks.
                index_spec = dataclasses.replace(
                    indexer.k_cache.get_kv_cache_spec(config), block_size=PAGE
                )
                index_builder = DeepseekV32IndexerMetadataBuilder(
                    index_spec,
                    [indexer.k_cache.prefix],
                    config,
                    device,
                    block_table_width=get_block_table_width(
                        index_spec.max_num_blocks_per_req(config, 8192),
                        index_spec.block_size,
                    ),
                )
            registry.warmup()
            group = layer.dcp_manager.group
            group_all_gather = group.all_gather

            def all_gather(input_, *args, **kwargs):
                gather_sizes.append(input_.numel())
                return group_all_gather(input_, *args, **kwargs)

            group.all_gather = all_gather

            # The profile run, at the scheduler's token budget.
            profile_tokens = config.scheduler_config.max_num_batched_tokens
            local_heads = LAYER_HEADS // world
            q = torch.randn(profile_tokens, local_heads, 192, device="cuda")
            layer.forward_impl(
                q.bfloat16(),
                torch.randn(profile_tokens, 512, device="cuda").bfloat16(),
                torch.randn(profile_tokens, 1, 64, device="cuda").bfloat16(),
                torch.empty(0, device="cuda"),
                None,
                torch.empty(
                    profile_tokens,
                    local_heads * 128,
                    device="cuda",
                    dtype=torch.bfloat16,
                ),
            )
            # The NCCL query gather and A2A of the largest step that can still
            # take the DCP path (the scheduler keeps whole prefill steps within
            # the ShardQ workspace): every local head's query, and every head's
            # partial outputs (and LSEs), of every token.
            fallback_tokens = dcp.dcp_fallback_max_tokens(
                config, DeepseekV32IndexerMetadataBuilder.get_decode_threshold(config)
            )
            assert gather_sizes
            assert max(gather_sizes) >= fallback_tokens * local_heads * 512
            assert a2a_sizes and max(a2a_sizes) >= fallback_tokens * LAYER_HEADS * 512
            torch.accelerator.synchronize()

            compiled: list[str] = []
            monitoring = [False]

            def record(*, backend, event, fn_name, detail=None):
                if monitoring[0]:
                    compiled.append(f"{backend} {event}: {fn_name}")

            jit_monitor._handle_jit_event = record
            jit_monitor.activate()

            rep = Replicated(lens, _layout(cache_dtype), fill_random=False)
            kv_c, k_pe, _ = _real_rows(rep, cache_dtype)
            shard = _shard(rep, world, rank, seed=rank, mla_block=LAYER_BLOCK)
            q = torch.randn(
                num_tokens, local_heads, 192, device="cuda", dtype=torch.bfloat16
            )
            layer.kv_cache = shard.mla
            indexer.k_cache.kv_cache = shard.index
            metadata = builder.build(
                0, _common_metadata(lens, shard.mla_table, (world, rank))
            )
            assert metadata.shardq is None
            assert builder.dcp_shardq is not None
            assert set(builder.dcp_shardq.stats.total) == {dcp.WORKSPACE}
            index_metadata = index_builder.build(
                0, _common_metadata(lens, shard.index_table, (world, rank))
            )
            num_a2a = len(a2a_sizes)
            monitoring[0] = True
            _indexer_topk(
                indexer,
                layer.layer_name,
                config,
                topk_buffer,
                (
                    torch.empty(num_tokens, 256, device="cuda").bfloat16(),
                    torch.randn(
                        num_tokens, 32, INDEX_HEAD_DIM, device="cuda"
                    ).bfloat16(),
                    torch.randn(num_tokens, 32, device="cuda").bfloat16(),
                ),
                metadata,
                index_metadata,
            )
            _layer_forward(layer, q, kv_c, k_pe, shard.mla, metadata)
            # The step combined its outputs with the NCCL A2A.
            assert len(a2a_sizes) > num_a2a
            # Another number of requests and an odd number of tokens: with 4
            # local heads, the A2A buffers' per-rank stride is then not a
            # multiple of 16 elements.
            num_seqs, odd_tokens = 5, num_tokens - 1
            assert odd_tokens % 2 == 1 and odd_tokens > num_seqs
            layer.dcp_manager.combine(
                torch.zeros(odd_tokens, LAYER_HEADS, 512, device="cuda").bfloat16(),
                torch.zeros(odd_tokens, LAYER_HEADS, device="cuda"),
                seq_lens=torch.ones(num_seqs, dtype=torch.int32, device="cuda"),
                query_start_loc=torch.tensor(
                    [0, 1, 2, 3, 4, odd_tokens], dtype=torch.int32, device="cuda"
                ),
            )
            torch.accelerator.synchronize()
            monitoring[0] = False
            assert not [c for c in compiled if "_dcp_a2a" in c], compiled
        finally:
            torch.distributed.all_to_all_single = nccl_a2a
            cleanup_dist_env_and_memory()


@pytest.mark.skipif(torch.accelerator.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize(("cache_dtype", "kernel"), KERNEL_CASES)
def test_profile_run_warms_the_dcp_path(cache_dtype, kernel, tmp_path):
    import torch.multiprocessing as mp

    world = min(torch.accelerator.device_count(), 4)
    mp.spawn(
        _profile_worker,
        args=(world, str(tmp_path / "dist_init"), cache_dtype, kernel),
        nprocs=world,
    )


@pytest.mark.parametrize(
    "fp4,entry,split",
    [
        pytest.param(False, 132, (128, 4), id="fp8_index"),
        pytest.param(True, 68, (64, 4), id="mxfp4_index"),
        # index_head_dim 64: 32 B of values and 2 scales.
        pytest.param(True, 34, (32, 2), id="mxfp4_index_dim64"),
    ],
)
def test_context_layout_of_the_indexer_cache_format(fp4, entry, split):
    """The gathered indexer rows follow the indexer cache's format: fp8 keys (128 B
    + one 4 B scale) or MXFP4 keys (head_dim / 2 B of E2M1 values + one UE8M0
    scale per 32 values, the Kimi K3.1 default on SM100), values then scales."""
    from types import SimpleNamespace

    from vllm.v1.attention.ops.dcp_shardq.context import ContextLayout
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    spec = MLAAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=576, dtype=torch.uint8
    )
    indexer = SimpleNamespace(
        k_cache=SimpleNamespace(head_dim=entry),
        quant_block_size=128,
        use_fp4_cache=fp4,
    )
    layout = ContextLayout.from_layer(spec, indexer)
    assert layout is not None
    assert (layout.index_row_bytes, layout.index_scale_bytes) == split
    assert layout.index_page_bytes == 64 * entry
