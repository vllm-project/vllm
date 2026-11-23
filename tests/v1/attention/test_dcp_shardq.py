# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP ShardQ prefill: the gathered context against replicated (TP) caches.

Each DCP rank's paged MLA and indexer caches are built from one replicated
copy; pushing every rank's KV shard must rebuild that copy in the workspace byte
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
# The MLA cache's kernel blocks span several pages, as in a packed KV cache.
MLA_BLOCK = 4 * PAGE
INDEX_HEAD_DIM = 128
# An fp8 indexer cache entry: the values, then one 4-byte scale.
INDEX_ENTRY = INDEX_HEAD_DIM + 4
TOPK = 2048
NUM_HEADS = 96
MLA_TOKEN_BYTES = {"fp8": 576, "nvfp4_ds_mla": 352}
FILL = 0xA5
_needs_prims_ts = pytest.mark.skipif(
    not has_flashinfer_prims_ts_sparse_mla(),
    reason="needs FlashInfer's prims_ts sparse MLA",
)
# (cache dtype, FlashInfer sparse MLA kernel): every pairing the backend serves.
KERNEL_CASES = [
    pytest.param("fp8", "trtllm"),
    pytest.param("fp8", "prims_ts", marks=_needs_prims_ts),
    pytest.param("nvfp4_ds_mla", "prims_ts", marks=_needs_prims_ts),
]

# (context, new tokens) per request.
SCENARIOS = {
    "short_contexts": [(3, 5), (7, 1), (8, 64), (1000, 37)],
    "page_tails": [(4097, 2048), (513, 1), (0, 64), (1, 511)],
}


def _num_pages(lens) -> int:
    return sum(-(-(c + n) // PAGE) for c, n in lens)


def _page_start(lens) -> np.ndarray:
    pages = np.array([-(-(c + n) // PAGE) for c, n in lens])
    return np.cumsum(pages) - pages


def _index_tokens(pages: torch.Tensor) -> torch.Tensor:
    """Indexer cache pages (each page's values, then its scales) as each
    token's entry, [n, PAGE, INDEX_ENTRY]."""
    flat, split = pages.flatten(1), PAGE * INDEX_HEAD_DIM
    values = flat[:, :split].reshape(-1, PAGE, INDEX_HEAD_DIM)
    return torch.cat((values, flat[:, split:].reshape(-1, PAGE, 4)), dim=2)


def _index_pages(tokens: torch.Tensor) -> torch.Tensor:
    """The inverse of `_index_tokens`."""
    values, scales = tokens.split([INDEX_HEAD_DIM, 4], dim=2)
    flat = torch.cat((values.flatten(1), scales.flatten(1)), dim=1)
    return flat.view(-1, PAGE, INDEX_ENTRY)


class Replicated:
    """Every token of every request, in dense pages from `_page_start`."""

    def __init__(self, lens, mla_bytes: int, fill_random: bool = True):
        self.lens, self.mla_bytes = lens, mla_bytes
        torch.manual_seed(0)

        def cache(token_bytes):
            shape = (_num_pages(lens), PAGE, token_bytes)
            if fill_random:
                return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
            return torch.zeros(shape, dtype=torch.uint8, device="cuda")

        self.mla = cache(mla_bytes)
        self.index = cache(INDEX_ENTRY)


def _shard(rep: Replicated, world: int, rank: int, seed: int):
    """`rank`'s DCP-local (MLA cache, its block table, indexer cache, its
    block table) of `rep`: MLA blocks of MLA_BLOCK tokens and indexer pages,
    padded apart (as other layers' pages in a packed cache) and permuted."""
    lens = rep.lens
    gen = torch.Generator().manual_seed(seed)
    local_lens = [max(c + n - rank + world - 1, 0) // world for c, n in lens]

    def paged(block_tokens, token_bytes, pad):
        blocks = [-(-n // block_tokens) for n in local_lens]
        num_blocks = sum(blocks) + 2
        stride = block_tokens * token_bytes + pad
        raw = torch.randint(0, 256, (num_blocks * stride,), dtype=torch.uint8)
        cache = torch.as_strided(
            raw.cuda(),
            (num_blocks, block_tokens, token_bytes),
            (stride, token_bytes, 1),
        )
        perm = torch.randperm(num_blocks, generator=gen)
        table = torch.zeros(len(lens), max(max(blocks), 1), dtype=torch.int32)
        used = 0
        for q, n in enumerate(blocks):
            table[q, :n] = perm[used : used + n]
            used += n
        return cache, table.cuda()

    # Whole pages apart, as the sparse MLA kernels address 64-token pages.
    mla_bytes = rep.mla_bytes
    mla, mla_table = paged(MLA_BLOCK, mla_bytes, PAGE * mla_bytes)
    index, index_table = paged(PAGE, INDEX_ENTRY, 80)
    rep_mla = rep.mla.view(-1, mla_bytes)
    rep_index = _index_tokens(rep.index).view(-1, INDEX_ENTRY)
    for q, n_local in enumerate(local_lens):
        # Local token i is the request's token world * i + rank.
        start = int(_page_start(lens)[q]) * PAGE
        tokens = start + torch.arange(n_local, device="cuda") * world + rank
        blocks = mla_table[q, : -(-n_local // MLA_BLOCK)]
        local = mla[blocks].view(-1, mla_bytes)
        local[:n_local] = rep_mla[tokens]
        mla[blocks] = local.view(-1, MLA_BLOCK, mla_bytes)
        pages = index_table[q, : -(-n_local // PAGE)]
        local = _index_tokens(index[pages]).view(-1, INDEX_ENTRY)
        local[:n_local] = rep_index[tokens]
        index[pages] = _index_pages(local.view(-1, PAGE, INDEX_ENTRY))
    return mla, mla_table, index, index_table


def _common_metadata(lens, block_table=None, dcp=None):
    """The step's metadata, with the DCP-local sequence lengths that the runner
    adds on rank `dcp[1]` of a DCP group of `dcp[0]`."""
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

    new = [n for _, n in lens]
    qsl = torch.tensor(np.r_[0, np.cumsum(new)], dtype=torch.int32)
    seq_lens = torch.tensor([c + n for c, n in lens], dtype=torch.int32)
    num_tokens = int(qsl[-1])
    local = None if dcp is None else get_dcp_local_seq_lens(seq_lens, *dcp)
    if block_table is None:
        width = -(-int(seq_lens.max()) // PAGE)
        block_table = torch.zeros(len(lens), width, dtype=torch.int32, device="cuda")
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


def _split(lens, world, num_decodes=0):
    """ShardQ's token split over `world` ranks of the requests after the
    first `num_decodes`."""
    from vllm.v1.attention.ops.shardq import ShardQSplit, zigzag_split

    new = np.array([n for _, n in lens])
    return ShardQSplit(
        int(new[:num_decodes].sum()), *zigzag_split(new[num_decodes:], world)
    )


def _build_context(lens, mla_bytes, world, rank, max_pages=None, split=None):
    from vllm.v1.attention.ops.dcp_shardq.context import build_dcp_shardq_context

    return build_dcp_shardq_context(
        _common_metadata(lens),
        _split(lens, world) if split is None else split,
        mla_token_bytes=mla_bytes,
        index_token_bytes=INDEX_ENTRY,
        max_pages=_num_pages(lens) if max_pages is None else max_pages,
        max_indexer_keys=1 << 20,
        max_logits_bytes=1 << 30,
        world=world,
        rank=rank,
        device=torch.device("cuda"),
    )


def _local_workspace(plan):
    """A local stand-in for one rank's workspace: (bytes, MLA, indexer view)."""
    mla_page, index_page = PAGE * plan.mla_token_bytes, PAGE * INDEX_ENTRY
    raw = torch.full(
        (plan.pages * (mla_page + index_page),), FILL, dtype=torch.uint8, device="cuda"
    )
    shape = (plan.pages, PAGE)
    mla = torch.as_strided(
        raw, (*shape, plan.mla_token_bytes), (mla_page, plan.mla_token_bytes, 1)
    )
    index = torch.as_strided(
        raw, (*shape, INDEX_ENTRY), (index_page, INDEX_ENTRY, 1), plan.pages * mla_page
    )
    return raw, mla, index


def _push_all(rep: Replicated, world: int, num_dst: int = 2):
    """Push every simulated rank's KV shard into `num_dst` local workspaces;
    returns them and the plans."""
    plans = [_build_context(rep.lens, rep.mla_bytes, world, r) for r in range(world)]
    workspaces = [_local_workspace(plans[0]) for _ in range(num_dst)]
    # The kernel stores to `world` destinations; spread them over the copies.
    dst_addrs = torch.tensor(
        [workspaces[d % num_dst][0].data_ptr() for d in range(world)],
        dtype=torch.int64,
        device="cuda",
    )
    for rank, plan in enumerate(plans):
        plan.push(*_shard(rep, world, rank, seed=rank), dst_addrs)
    torch.accelerator.synchronize()
    return workspaces, plans


def _check_workspace(
    ws_mla, ws_index, rep: Replicated, new_mla_tokens: bool, fill, num_decodes=0
):
    """The workspace holds the tokens of the requests from `num_decodes` on as
    `rep` does (the MLA entries of their new tokens only with
    `new_mla_tokens`); with `fill`, every other byte still holds FILL."""
    lens = rep.lens[num_decodes:]
    first, num_pages = _num_pages(rep.lens[:num_decodes]), _num_pages(lens)
    ref = slice(first, first + num_pages)

    def written(num_tokens):
        """[num_pages, PAGE]: whether each workspace slot holds a token."""
        slots = [torch.arange(-(-(c + n) // PAGE) * PAGE) for c, n in lens]
        mask = torch.cat([s < k for s, k in zip(slots, num_tokens)])
        return mask.view(-1, PAGE).cuda()

    seq_lens = [c + n for c, n in lens]
    mla_tokens = seq_lens if new_mla_tokens else [c for c, _ in lens]
    for ws, expected, mask in (
        (ws_mla, rep.mla[ref], written(mla_tokens)),
        (_index_tokens(ws_index), _index_tokens(rep.index[ref]), written(seq_lens)),
    ):
        assert torch.equal(ws[:num_pages][mask], expected[mask])
        if fill:
            assert (ws[:num_pages][~mask] == FILL).all()
            assert (ws[num_pages:] == FILL).all()


@pytest.mark.parametrize("cache_dtype", list(MLA_TOKEN_BYTES))
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_push_rebuilds_the_replicated_cache(cache_dtype, scenario):
    """The pushes rebuild the replicated caches in token order and write
    nothing else; the indexer's prefill key gather then reads the workspace
    exactly as the replicated cache, with causal bounds over whole sequences."""
    from vllm import _custom_ops as ops

    lens = SCENARIOS[scenario]
    rep = Replicated(lens, MLA_TOKEN_BYTES[cache_dtype])
    workspaces, plans = _push_all(rep, 8)
    for _, ws_mla, ws_index in workspaces:
        # Prior context in full and the new tokens' indexer entries; the new
        # MLA entries and page tails stay untouched.
        _check_workspace(ws_mla, ws_index, rep, new_mla_tokens=False, fill=True)

    (chunk,) = plans[0].indexer_chunks
    assert chunk.total_seq_lens == sum(c + n for c, n in lens)

    def gather_keys(cache):
        keys = torch.empty(chunk.total_seq_lens, INDEX_HEAD_DIM, dtype=torch.uint8)
        keys = keys.cuda()
        scales = keys.new_empty(chunk.total_seq_lens, 4)
        ops.cp_gather_indexer_k_quant_cache(
            cache, keys, scales, chunk.block_table, chunk.cu_seq_lens
        )
        return torch.cat((keys, scales), dim=1)

    assert torch.equal(gather_keys(workspaces[0][2]), gather_keys(rep.index))
    # Rank 0's tokens attend causally over their whole sequences.
    qsl = np.r_[0, np.cumsum([n for _, n in lens])]
    scored = plans[0].gather_indices.cpu().numpy()
    token_req = np.searchsorted(qsl[1:], scored, side="right")
    token_pos = scored - qsl[token_req] + np.array([c for c, _ in lens])[token_req]
    starts = chunk.cu_seq_lens.cpu().numpy()[token_req]
    np.testing.assert_array_equal(chunk.cu_seqlen_ks.cpu().numpy(), starts)
    np.testing.assert_array_equal(
        chunk.cu_seqlen_ke.cpu().numpy(), starts + token_pos + 1
    )


def test_plan_bounds_and_capacity():
    lens = SCENARIOS["page_tails"]
    mla_bytes = MLA_TOKEN_BYTES["fp8"]
    plan = _build_context(lens, mla_bytes, 8, 3)
    seq_lens = np.array([c + n for c, n in lens])
    page_start = _page_start(lens)
    # Each request's pages map to its region, in tables whose width is a
    # multiple of 16 pages (one Triton specialization).
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
    assert _build_context(lens, mla_bytes, 8, 3, _num_pages(lens) - 1) is None

    # The split's leading decodes keep the DCP path, their tokens scored in
    # place first; a later short request is a prefill request.
    lens = [(500, 4), (70, 1), *lens, (500, 4)]
    plan = _build_context(lens, mla_bytes, 8, 3, split=_split(lens, 8, 2))
    assert (plan.num_decodes, plan.num_decode_tokens) == (2, 5)
    np.testing.assert_array_equal(plan.gather_indices.cpu()[:5], np.arange(5))


def _flashinfer_impl(cache_dtype: str, kernel: str):
    """A FlashInfer sparse MLA impl at K3.1's MLA sizes on `kernel`."""
    from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as fims

    impl = object.__new__(fims.FlashInferMLASparseImpl)
    impl.dcp_world_size = 1
    impl.dcp_rank = 0
    impl.scale = 192**-0.5
    impl.kv_cache_dtype = cache_dtype
    impl.kv_lora_rank, impl.qk_nope_head_dim, impl.qk_rope_head_dim = 512, 128, 64
    impl.is_nope_mla = False
    impl.index_group = None
    impl.need_to_return_lse_for_decode = False
    impl._workspace_buffer = impl.bmm1_scale = impl.bmm2_scale = None
    impl.use_prims_ts = kernel == "prims_ts"
    if impl.use_prims_ts:
        impl.lse_base_on_e = True
        impl.prims_ts_format = fims._PRIMS_TS_FORMATS[cache_dtype]
        impl.prims_ts_buckets = tuple(1 << i for i in range(13))
    return impl


def _real_kv(rep: Replicated, cache_dtype: str):
    """Write real MLA and indexer entries of every token into `rep`; returns
    the new tokens' latents (to write them as forward_impl does)."""
    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer import (
        indexer_k_norm_quant_insert,
    )

    seq_lens = [c + n for c, n in rep.lens]
    page_start = _page_start(rep.lens)
    slots = torch.cat(
        [p * PAGE + torch.arange(s) for p, s in zip(page_start, seq_lens)]
    )
    slots = slots.cuda()
    torch.manual_seed(1)
    kv_c = torch.randn(len(slots), 512, device="cuda").bfloat16()
    k_pe = torch.randn(len(slots), 64, device="cuda").bfloat16()
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    ops.concat_and_cache_mla(kv_c, k_pe, rep.mla, slots, cache_dtype, scale)
    norm = torch.ones(INDEX_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    # The insert also clears that many top-k.
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


def _random_topk(lens) -> tuple[torch.Tensor, torch.Tensor]:
    """Random causal top-k of every new token (-1 pads short requests) and the
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
    """New MLA entries written through the plan's slot mapping complete the
    workspace; the kernel then attends it bit-identically to the replicated
    cache, through the same block table."""
    from vllm import _custom_ops as ops
    from vllm.v1.attention.backends.mla.sparse_utils import (
        flat_kv_row_view,
        triton_convert_req_index_to_global_index,
    )

    lens, world = SCENARIOS["page_tails"], 8
    rep = Replicated(lens, MLA_TOKEN_BYTES[cache_dtype], fill_random=False)
    kv_c, k_pe, scale = _real_kv(rep, cache_dtype)
    workspaces, plans = _push_all(rep, world, num_dst=1)
    _, ws_mla, ws_index = workspaces[0]
    # The step's new tokens, as forward_impl writes them on every rank.
    ops.concat_and_cache_mla(
        kv_c, k_pe, ws_mla, plans[0].slot_mapping, cache_dtype, scale
    )
    _check_workspace(ws_mla, ws_index, rep, new_mla_tokens=True, fill=True)

    topk, req_id = _random_topk(lens)
    # A query as the layer quantizes it for the FlashInfer kernels.
    q = torch.randn(topk.shape[0], NUM_HEADS, 576, device="cuda")
    q = q.to(torch.float8_e4m3fn)
    impl = _flashinfer_impl(cache_dtype, kernel)
    # The attention layer's static per-tensor query and KV scales.
    scales = SimpleNamespace(_q_scale_float=1.0, _k_scale_float=1.0)

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
        out, _ = impl._forward_mqa_kernel(
            q,
            cache,
            physical_topk,
            valid_counts,
            layer=scales,
            block_size=PAGE,
            is_decode=False,
        )
        return out

    out = attend(ws_mla)
    assert torch.isfinite(out).all()
    assert torch.equal(out, attend(rep.mla))


# A Kimi K3.1 DSA layer at DeepSeek-V2-Lite's MLA sizes (16 heads, NoPE latent
# with the 64-dim RoPE slot), on the model config the builders read.
LAYER_MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"
LAYER_HEADS = 16
# Requests above the decode threshold of 4-8 local heads (1024 query tokens),
# longer than the top-k so the dense-MHA prefill never applies.
LAYER_LENS = [(3000, 1100), (0, 2200), (700, 1500)]
# DCP ShardQ steps. The sparse MLA backend takes leading pieces of at most 1024
# tokens as decodes, the indexer only one-token ones, which keep the DCP path.
LAYER_STEPS = {
    "long_pieces": LAYER_LENS,
    "short_and_long_pieces": [(900, 300), (2500, 64), *LAYER_LENS],
    "short_pieces_only": [(3000, 700), (2000, 500)],
    "decodes_and_pieces": [(6000, 1), (2100, 1), (900, 300), *LAYER_LENS],
}


def _leading(lens, max_query_len: int) -> int:
    """Requests before the step's first one of more than `max_query_len`
    tokens."""
    return next((i for i, (_, n) in enumerate(lens) if n > max_query_len), len(lens))


def _dsa_layer(config, topk_buffer, layer_index=0):
    """A real KimiK3DSAAttention, its MLA layer's weights processed."""
    from vllm.models.kimi_k3.nvidia.dsa import KimiK3DSAAttention
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

    mla_dims = dict(qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128)
    text_config = KimiLinearConfig(
        hidden_size=256,
        num_attention_heads=LAYER_HEADS,
        kv_lora_rank=512,
        mla_use_nope=True,
        dsa_config={
            "dsa_layers": [0],
            "index_topk": TOPK,
            "index_head_dim": INDEX_HEAD_DIM,
            "index_num_heads": 32,
            "index_rope_dim": 0,
            "index_k_norm_type": "layernorm",
        },
        **mla_dims,
    )
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        attn = KimiK3DSAAttention(
            text_config,
            hidden_size=256,
            num_heads=LAYER_HEADS,
            q_lora_rank=None,
            kv_lora_rank=512,
            cache_config=config.cache_config,
            prefix=f"model.layers.{layer_index}.self_attn",
            topk_indices_buffer=topk_buffer,
            **mla_dims,
        )
    torch.set_default_dtype(torch.float32)
    layer = attn.mla_attn.mla_attn
    torch.manual_seed(5)  # The same weights on every rank's slice.
    weight = torch.randn(
        LAYER_HEADS * 256, 512, device="cuda", dtype=torch.bfloat16
    ).chunk(layer.impl.dcp_world_size)[layer.impl.dcp_rank]
    layer.kv_b_proj.weight.data.copy_(weight * 0.05)
    layer.process_weights_after_loading()
    return attn


def _layer_forward(layer, q, kv_c, k_pe, kv_cache, metadata, q_rep=None):
    out = torch.empty(q.shape[0], q.shape[1] * 128, device="cuda", dtype=torch.bfloat16)
    args = (q, kv_c, k_pe.unsqueeze(1), kv_cache, metadata, out)
    return layer.forward_impl(*args, q_dcp_replicated=q_rep)


def _min_topk_recall(topk: torch.Tensor, ref: torch.Tensor) -> float:
    """The smallest share of a token's `ref` keys that `topk` selects."""
    keys = torch.sort(topk, dim=1).values.contiguous()
    at = torch.searchsorted(keys, ref.contiguous()).clamp(max=keys.shape[1] - 1)
    found = ((keys.gather(1, at) == ref) & (ref >= 0)).sum(1).float()
    return float((found / (ref >= 0).sum(1).clamp(min=1)).min())


def _layer_worker(rank, world, init_file, cache_dtype, kernel, comm_backend):
    """One rank of a DCP = TP group running a real K3.1 DSA layer.

    The profile run allocates the workspace and runs one full-size DCP-path
    step. DCP ShardQ steps (short pieces and leading decodes included) then
    match the DCP path, top-k included, without compiling any kernel after
    warmup. Small steps, steps beyond the workspace, interleave-64 DCP and
    disaggregated decode instances keep the DCP path.
    """
    from tests.v1.attention.utils import create_vllm_config
    from vllm.config import CUDAGraphMode, KVTransferConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.forward_context import create_forward_context, override_forward_context
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
    from vllm.v1.attention.ops.dcp_shardq.gather import DCP_CONTEXT_PUSH_KERNEL
    from vllm.v1.worker.workspace import init_workspace_manager

    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    config = create_vllm_config(
        model_name=LAYER_MODEL,
        tensor_parallel_size=world,
        max_model_len=8192,
        block_size=MLA_BLOCK,
        max_num_seqs=8,
        max_num_batched_tokens=8192,
        hf_config_override={"index_topk": TOPK},
    )
    config.parallel_config.decode_context_parallel_size = world
    config.parallel_config.dcp_comm_backend = comm_backend
    config.cache_config.cache_dtype = cache_dtype
    config.attention_config.backend = AttentionBackendEnum.FLASHINFER_MLA_SPARSE
    config.attention_config.sparse_mla_shard_q = True
    config.kernel_config.flashinfer_sparse_mla_kernel = kernel
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    # The profile run sizes the workspace once, for every step below.
    max_pages = max(_num_pages(lens) for lens in LAYER_STEPS.values())
    dcp.workspace_pages = lambda _: max_pages
    mla_bytes = MLA_TOKEN_BYTES[cache_dtype]
    randn = partial(torch.randn, device="cuda", dtype=torch.bfloat16)
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
            # Registrations made while building the model and builders, and
            # binding the KV cache (the top-k remap, onto the workspace too).
            registry = JitWarmupRegistry(config)
            with registry.activate():
                attn = _dsa_layer(config, topk_buffer)
                layer, indexer = attn.mla_attn.mla_attn, attn.dsa_indexer
                assert layer.use_shardq
                assert layer.shardq_index_entry_bytes == INDEX_ENTRY
                assert indexer.k_cache.shardq_layer_name == layer.layer_name
                spec = layer.get_kv_cache_config(config).spec
                builder = FlashInferMLASparseTRTLLMBackend.get_builder_cls()(
                    spec, [layer.layer_name], config, device
                )
                # The indexer builder sees its 64-token kernel blocks.
                index_builder = DeepseekV32IndexerMetadataBuilder(
                    dataclasses.replace(
                        indexer.k_cache.get_kv_cache_config(config).spec,
                        block_size=PAGE,
                    ),
                    [indexer.k_cache.prefix],
                    config,
                    device,
                    block_table_width=8192 // PAGE,
                )
                layer.bind_kv_cache(
                    torch.empty(1, 1, MLA_BLOCK, mla_bytes, dtype=torch.uint8).cuda()
                )
            assert builder.dcp_shardq is not None
            # The push kernel's key holds the whole indexer entry.
            push_key = dict(
                world=world, mla_token_bytes=mla_bytes, index_token_bytes=INDEX_ENTRY
            )
            assert registry._registrations[DCP_CONTEXT_PUSH_KERNEL] == [((), push_key)]
            registry.warmup()

            # The profile run (no attention metadata), twice: the workspace and
            # one DCP-path query gather and combine at the token budget, beyond
            # the direct workspaces, once per process.
            manager = layer.dcp_manager
            calls: list[tuple[str, int, int]] = []

            def recorded(name, fn):
                def call(x, *args, **kwargs):
                    calls.append((name, x.shape[0], x.shape[1]))
                    return fn(x, *args, **kwargs)

                return call

            manager.query_gather = recorded("gather", manager.query_gather)
            manager.combine = recorded("combine", manager.combine)
            profile_tokens = config.scheduler_config.max_num_batched_tokens
            local_heads = LAYER_HEADS // world
            for _ in range(2):
                _layer_forward(
                    layer,
                    randn(profile_tokens, local_heads, 192),
                    randn(profile_tokens, 512),
                    randn(profile_tokens, 64),
                    torch.empty(0, device="cuda"),
                    None,
                )
            assert calls == [
                ("gather", profile_tokens, local_heads),
                ("combine", profile_tokens, LAYER_HEADS),
            ], calls
            assert dcp._workspace_buffer is not None
            assert dcp._workspace_buffer.storage.numel() == (
                max_pages * PAGE * (mla_bytes + INDEX_ENTRY)
            )

            # A kernel that a ShardQ step compiles after warmup, beyond those of
            # the DCP path (which the runner's dummy runs warm), is a miss.
            compiled: list[str] = []
            monitoring = [False]

            def record(*, backend, event, fn_name, detail=None):
                if monitoring[0]:
                    compiled.append(f"{backend} {event}: {fn_name}")

            jit_monitor._handle_jit_event = record
            jit_monitor.activate()

            def score(inputs, mla_metadata, index_metadata):
                """The indexer's top-k of the step's tokens (of this rank's
                prefill tokens on a ShardQ step) into the cleared buffer."""
                topk_buffer.fill_(-1)
                metadata = {
                    layer.layer_name: mla_metadata,
                    indexer.k_cache.prefix: index_metadata,
                }
                with override_forward_context(create_forward_context(metadata, config)):
                    indexer.score(*inputs, index_k=None)

            # The prefill requests' tokens, new MLA entries included.
            check_workspace = partial(_check_workspace, new_mla_tokens=True, fill=False)
            # The DCP path on the same steps, from a builder whose workspace is
            # too small for them.
            dcp.workspace_pages = lambda _: 16
            small = FlashInferMLASparseTRTLLMBackend.get_builder_cls()(
                spec, [layer.layer_name], config, device
            )
            for name, lens in LAYER_STEPS.items():
                rep = Replicated(lens, mla_bytes, fill_random=False)
                kv_c, k_pe, _ = _real_kv(rep, cache_dtype)
                num_decodes = _leading(lens, 1)
                num_decode_tokens = sum(n for _, n in lens[:num_decodes])
                mla_cache, mla_table, index_cache, index_table = _shard(
                    rep, world, rank, seed=rank
                )
                layer.kv_cache, indexer.k_cache.kv_cache = mla_cache, index_cache
                step = _common_metadata(lens, mla_table, (world, rank))
                index_step = _common_metadata(lens, index_table, (world, rank))
                num_tokens = kv_c.shape[0]
                torch.manual_seed(6)
                q = randn(num_tokens, LAYER_HEADS, 192).chunk(world, dim=1)[rank]
                q = q.contiguous()
                forward = partial(_layer_forward, layer, q, kv_c, k_pe, mla_cache)
                # The indexer's inputs (its keys are in the cache already).
                inputs = (
                    randn(num_tokens, 256),
                    randn(num_tokens, 32, INDEX_HEAD_DIM),
                    randn(num_tokens, 32),
                )

                dcp_metadata = small.build(0, step)
                assert dcp_metadata.shardq is None
                score(inputs, dcp_metadata, index_builder.build(0, index_step))
                dcp_topk = topk_buffer[:num_tokens].clone()
                # The DCP path's kernels, which the runner's dummy runs warm.
                forward(dcp_metadata)

                monitoring[0] = True
                metadata = builder.build(0, step)
                index_metadata = index_builder.build(0, index_step)
                # Before admitting short pieces, MLA decodes kept the DCP path.
                assert metadata.num_decodes == _leading(
                    lens, builder.reorder_batch_threshold
                ), name
                # The indexer scores the one-token requests as decodes.
                assert index_metadata.num_decode_tokens == num_decode_tokens, name
                shardq = metadata.shardq
                assert shardq is not None, name
                context = shardq.gathered_context
                assert context is not None, name
                assert context.num_decodes == num_decodes
                assert shardq.num_decode_tokens == num_decode_tokens
                assert sum(shardq.num_tokens_per_rank) == num_tokens - num_decode_tokens

                # The indexer gathers the context and writes the top-k of this
                # rank's prefill tokens after the decode tokens, which it
                # scores on the DCP path in place.
                score(inputs, metadata, index_metadata)
                local = num_decode_tokens + shardq.local_token_ids
                num_scored = num_decode_tokens + len(shardq.local_token_ids)
                decode_topk = topk_buffer[:num_decode_tokens].clone()
                local_topk = topk_buffer[num_decode_tokens:num_scored].clone()
                out = forward(metadata).clone()
                torch.accelerator.synchronize()
                monitoring[0] = False
                # ShardQ's exchange kernels (vllm.v1.attention.ops.shardq) are
                # not warmup-registered, under DCP as under TP.
                compiled[:] = [
                    c for c in compiled if "shard_q" not in c and "unshard_out" not in c
                ]
                assert not compiled, (name, compiled)
                # The keys of the DCP path's cross-rank top-k merge, up to the
                # order of near-tied (bf16) scores at the top-k boundary.
                assert _min_topk_recall(local_topk, dcp_topk[local]) > 0.99, name
                # The decode tokens' top-k (every key valid), as on the DCP path.
                assert (decode_topk >= 0).all(), name
                assert torch.equal(
                    torch.sort(decode_topk, dim=1).values,
                    torch.sort(dcp_topk[:num_decode_tokens], dim=1).values,
                ), name
                # The prefill tokens' MLA entries are complete in the workspace.
                ws_mla = context.mla_view(mla_cache)
                ws_index = context.index_view(index_cache)
                check_workspace(ws_mla, ws_index, rep, num_decodes=num_decodes)
                # A replicated query (DCP query replication) is left aside.
                q_rep = get_tp_group().all_gather(q, dim=1)
                assert torch.equal(forward(metadata, q_rep), out)

                # The DCP path over the same keys: every rank's top-k.
                topk = torch.zeros_like(dcp_topk)
                topk[local] = local_topk + 1
                torch.distributed.all_reduce(topk, group=get_tp_group().device_group)
                topk_buffer[:num_tokens] = topk - 1
                topk_buffer[:num_decode_tokens] = decode_topk
                expected = forward(dcp_metadata)
                # Both round the softmax weights to E4M3 (the DCP path per rank's
                # partial): each is within 2e-2 of fp32, apart by about 2.1e-2.
                err = (out.float() - expected.float()).norm() / expected.float().norm()
                assert err < 3e-2, (name, err)

            # The next layer's gather waits for every rank to finish reading
            # this one: rank 0 lags, and its copy must still hold this layer.
            next_shard = _shard(Replicated(lens, mla_bytes), world, rank, seed=1)
            if rank == 0:
                torch.cuda._sleep(500_000_000)
            ws_mla, ws_index = ws_mla.clone(), ws_index.clone()
            assert context is not None
            context.gather(*next_shard)
            check_workspace(ws_mla, ws_index, rep, num_decodes=num_decodes)

            # Steps that keep the DCP path (each fits the workspace): fewer
            # than sparse_mla_shard_q_min_tokens prefill tokens (after a
            # decode token), which ShardQ does not split, and decodes alone.
            for lens in ([(100, 1), (3000, 600), (2900, 300)], [(100, 1), (3000, 1)]):
                step = _common_metadata(lens, dcp=(world, rank))
                assert builder.build(0, step).shardq is None

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
@pytest.mark.parametrize(
    ("cache_dtype", "kernel", "comm_backend"),
    [
        pytest.param("fp8", "trtllm", "a2a"),
        pytest.param("nvfp4_ds_mla", "prims_ts", "ag_rs", marks=_needs_prims_ts),
    ],
)
def test_dsa_layer_runs_dcp_prefill_as_shardq(
    cache_dtype, kernel, comm_backend, tmp_path
):
    import torch.multiprocessing as mp

    world = min(torch.accelerator.device_count(), 4)
    mp.spawn(
        _layer_worker,
        args=(world, str(tmp_path / "dist_init"), cache_dtype, kernel, comm_backend),
        nprocs=world,
    )
