# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP ShardQ prefill: the gathered context against replicated (TP) caches.

Each DCP rank's paged MLA and indexer caches are built from one replicated
copy; pushing every rank's rows must rebuild that copy in the workspace byte
for byte, and the indexer and the FlashInfer sparse MLA kernels (trtllm-gen and
prims_ts) must read it exactly as the TP ShardQ path reads the replicated
caches.
"""

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


def _split(lens, world, num_decodes=0):
    """ShardQ's token split over ``world`` ranks of the requests after the
    first ``num_decodes``."""
    from vllm.v1.attention.ops.shardq import ShardQSplit, zigzag_split

    new = np.array([n for _, n in lens])
    return ShardQSplit(
        int(new[:num_decodes].sum()), *zigzag_split(new[num_decodes:], world)
    )


def _build_context(lens, layout, world, rank, max_pages=None, split=None):
    from vllm.v1.attention.ops.dcp_shardq.context import (
        build_dcp_shardq_context,
    )

    total_pages = sum(-(-(c + n) // PAGE) for c, n in lens)
    return build_dcp_shardq_context(
        _metadata(lens),
        _split(lens, world) if split is None else split,
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
    # This rank's tokens, each at its position in its request.
    qsl = np.r_[0, np.cumsum([n for _, n in lens])]
    rows = plans[0].gather_indices.cpu().numpy()
    row_req = np.searchsorted(qsl[1:], rows, side="right")
    row_pos = rows - qsl[row_req] + np.array([c for c, _ in lens])[row_req]
    assert len(plans[0].indexer_chunks) == 1
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
        # Causal bounds of this rank's tokens over the request's whole
        # sequence (no DCP-local form).
        ks, ke = chunk.cu_seqlen_ks.cpu().numpy(), chunk.cu_seqlen_ke.cpu().numpy()
        starts = chunk.cu_seq_lens.cpu().numpy()[row_req]
        np.testing.assert_array_equal(ks, starts)
        np.testing.assert_array_equal(ke, starts + row_pos + 1)
        assert chunk.total_seq_lens == sum(seq_lens)


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
    stats.record(dcp.NO_SPLIT, 600)
    stats.record(dcp.WORKSPACE, 15000)
    assert stats.total == {
        dcp.SHARDQ: [2, 24384],
        dcp.NO_SPLIT: [1, 600],
        dcp.WORKSPACE: [1, 15000],
    }
    assert stats.report() == (
        "DCP ShardQ prefill: 2/4 steps, 24384/39984 tokens (61.0%; 61.0% since "
        "start) on ShardQ; DCP path: no split 1 steps / 600 tokens, "
        "workspace full 1 steps / 15000 tokens"
    )
    stats.recent.clear()
    stats.record(dcp.NO_SPLIT, 900)
    assert stats.report() == (
        "DCP ShardQ prefill: 0/1 steps, 0/900 tokens (0.0%; 59.6% since start) "
        "on ShardQ; DCP path: no split 1 steps / 900 tokens"
    )


@pytest.mark.parametrize("num_spec", [0, 3])
def test_planner_follows_the_split_decode_rows(num_spec):
    """The split's decode rows (at most 1 + num_speculative_tokens tokens,
    which the indexer scores on its decode path) keep the DCP path; the plan
    covers the rest of the step, a later short row included. Steps that
    ShardQ does not split keep the DCP path too, counted when they have
    prefill rows."""
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
        threshold,
        world,
        rank,
        torch.device("cuda"),
        label="layers.0, 1 layers",
    )
    long = [(1000, 300), (0, 200)]

    def plan(lens, num_decodes=None):
        table = torch.zeros(len(lens), 1, dtype=torch.int32, device="cuda")
        metadata = _common_metadata(lens, table)
        split = None if num_decodes is None else _split(lens, world, num_decodes)
        return planner.plan(metadata, split)

    decodes = [(500, threshold), (70, 1)]
    context = plan([*decodes, *long], num_decodes=2)
    assert context is not None
    assert context.num_decodes == 2
    assert context.num_decode_tokens == threshold + 1
    assert context.world == world and context.rank == rank
    # This rank's tokens follow the decode rows in place.
    rows = context.gather_indices.cpu().numpy()
    np.testing.assert_array_equal(rows[: threshold + 1], np.arange(threshold + 1))
    assert len(rows) - threshold - 1 == _split(long, world).num_tokens_per_rank[rank]
    # A short row after the first prefill row is a prefill row.
    context = plan([*long, (500, threshold)], num_decodes=0)
    assert context is not None and context.num_decodes == 0
    # Not split: decode rows alone are not counted, prefill rows are.
    assert plan(decodes) is None
    assert plan([*decodes, *long]) is None
    assert {path: c[0] for path, c in planner.stats.total.items()} == {
        dcp.SHARDQ: 2,
        dcp.NO_SPLIT: 1,
    }
    assert planner.stats.report().startswith(
        "DCP ShardQ prefill [layers.0, 1 layers]: 2/3 steps"
    )


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
