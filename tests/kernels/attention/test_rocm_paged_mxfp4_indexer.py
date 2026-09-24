# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1's ROCm MXFP4 sparse indexer against a torch reference.

The indexer K cache is a view into vLLM's block-major KV pool: its pages are
a block stride apart with other layers' pages in between, so the tests put
it there. The reference reads the keys back from a second cache the same
writer fills in natural order, so it shares none of the preshuffled
addressing it checks.
"""

import itertools
import types

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm-only", allow_module_level=True)

from vllm.config import CUDAGraphMode
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerPrefillChunkMetadata,
)
from vllm.v1.attention.backends.mla.rocm_paged_mxfp4_indexer import (
    DeepseekV41RocmMxfp4IndexerMetadata,
    native_decode,
    plan_gather_launches,
    plan_prefill_chunks,
)
from vllm.v1.attention.ops import rocm_paged_mxfp4_indexer as ops
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

if (_reason := ops.rocm_mxfp4_indexer_unsupported_reason()) is not None:
    pytest.skip(_reason, allow_module_level=True)

HEADS, HEAD_DIM = 32, 128
WIDTH = HEAD_DIM // 2 + HEAD_DIM // 32
# V4.1's 8-token blocks, with the pool (2048 blocks) and top-k (512) scaled
# down so short test contexts still select.
CAND_BLOCK, POOL_BLOCKS, TOPK = 8, 16, 32
MAX_LEN = 4096
DEVICE = "cuda"

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1 = torch.cat([_E2M1, -_E2M1])


def _dequant(packed, e8m0):
    """[..., D/2] e2m1 pairs and [..., D/32] e8m0 to [..., D] fp32."""
    table = E2M1.to(packed.device)
    vals = torch.stack([table[(packed & 0xF).long()], table[(packed >> 4).long()]], -1)
    scale = torch.pow(2.0, e8m0.float() - 127)
    return (vals.flatten(-2).unflatten(-1, (-1, 32)) * scale[..., None]).flatten(-2)


@pytest.fixture(autouse=True)
def _workspace():
    init_workspace_manager(torch.device(DEVICE))
    yield
    reset_workspace_manager()


class _Case:
    """Requests with their keys written into a block-major pool (ratio-2 and
    ratio-1 pages side by side in every block) and into natural-order caches
    the reference reads."""

    def __init__(self, seq_lens, block, seed=0):
        torch.manual_seed(seed)
        self.seq_lens = seq_lens
        num_blocks = sum(cdiv(n, block) for n in seq_lens) + 3
        perm = torch.randperm(num_blocks)
        width = max(cdiv(n, block) for n in seq_lens)
        self.block_table = torch.zeros(len(seq_lens), width, dtype=torch.int32)
        used = 0
        for req, n in enumerate(seq_lens):
            self.block_table[req, : cdiv(n, block)] = perm[used : used + cdiv(n, block)]
            used += cdiv(n, block)
        self.block_table = self.block_table.to(DEVICE)

        def align(n):
            return cdiv(n, 576) * 576

        page2, page1 = block // 2 * WIDTH, block * WIDTH
        stride = align(page2) + align(page1) + 576
        self.pool = torch.zeros(num_blocks * stride, dtype=torch.uint8, device=DEVICE)
        self.page_bytes = torch.zeros_like(self.pool, dtype=torch.bool)
        for i in range(num_blocks):
            base = i * stride
            self.page_bytes[base : base + page2] = True
            base += align(page2)
            self.page_bytes[base : base + page1] = True
        self.cache = {
            2: torch.as_strided(
                self.pool, (num_blocks, block // 2, WIDTH), (stride, WIDTH, 1), 0
            ),
            1: torch.as_strided(
                self.pool,
                (num_blocks, block, WIDTH),
                (stride, WIDTH, 1),
                align(page2),
            ),
        }
        self.offsets = [0]
        for n in seq_lens:
            self.offsets.append(self.offsets[-1] + n)
        # the same keys in natural order, one entry per page
        self.natural = {
            r: torch.zeros(self.offsets[-1], 1, WIDTH, dtype=torch.uint8, device=DEVICE)
            for r in (1, 2)
        }
        cos_sin = torch.randn(MAX_LEN, 64, device=DEVICE)
        norm = torch.rand(HEAD_DIM, device=DEVICE) + 0.5
        for req, n in enumerate(seq_lens):
            if n == 0:
                continue
            k_pre = torch.randn(n, HEAD_DIM, device=DEVICE).to(torch.bfloat16)
            pos = torch.arange(n, device=DEVICE)
            for ratio, cache in self.cache.items():
                comp = pos // ratio
                boundary = (pos + 1) % ratio == 0
                entries = cache.shape[1]
                page = self.block_table[req, comp // entries].long()
                slot = torch.where(boundary, page * entries + comp % entries, -1)
                nat = torch.where(boundary, self.offsets[req] + comp, -1)
                ops.rocm_mxfp4_indexer_k_store(
                    k_pre,
                    pos,
                    cos_sin,
                    norm,
                    1e-6,
                    cache,
                    slot,
                    ratio,
                    True,
                    num_heads=HEADS,
                )
                ops._aiter_cache().indexer_k_norm_rope_mxfp4_cache(
                    k_pre, pos, cos_sin, norm, 1e-6, self.natural[ratio], nat, ratio
                )

    def keys(self, ratio, req, n):
        nat = self.natural[ratio][self.offsets[req] : self.offsets[req] + n, 0]
        return _dequant(nat[:, : HEAD_DIM // 2], nat[:, HEAD_DIM // 2 :])


def _queries(num_rows):
    q = torch.randint(0, 256, (num_rows, HEADS, HEAD_DIM // 2), dtype=torch.uint8)
    scales = torch.randint(122, 130, (num_rows, HEADS, HEAD_DIM // 32))
    q, scales = q.to(DEVICE), scales.to(torch.uint8).to(DEVICE)
    weights = torch.randn(num_rows, HEADS, device=DEVICE)
    # the fused Q kernel returns the scales as one int32 per head
    return q, scales.view(torch.int32).squeeze(-1), weights, _dequant(q, scales)


def _scores(q, w, keys):
    return (torch.relu(q @ keys.T) * w[:, None]).sum(0)


def _assert_topk(row, scores, k):
    """``row`` is a top-k of the entries of ``scores`` above -inf, up to the
    fp32 reduce order."""
    valid = scores > float("-inf")
    want = min(k, int(valid.sum()))
    sel = row[row >= 0].long()
    assert sel.numel() == want and sel.unique().numel() == want
    if want:
        assert bool(valid[sel].all())
        kth = scores[valid].topk(want).values[-1]
        finite = scores[torch.isfinite(scores)]
        tol = 1e-4 * float(finite.abs().max()) if finite.numel() else 0.0
        assert bool((scores[sel] >= kth - tol).all())


def _block_maxima(scores):
    end = scores.numel()
    if end == 0:
        return scores
    padded = torch.full((cdiv(end, CAND_BLOCK) * CAND_BLOCK,), float("-inf"))
    padded[:end] = scores.cpu()
    out = padded.view(-1, CAND_BLOCK).amax(1)
    out[(end - 1) // CAND_BLOCK] = float("inf")
    return out.to(scores.device)


def _forward_context(monkeypatch, metadata):
    context = types.SimpleNamespace(
        attn_metadata={"indexer": metadata},
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )
    monkeypatch.setattr(ops, "get_forward_context", lambda: context)


def _decode_metadata(case, rows, ratio, query_lens):
    lens = torch.tensor([(p + 1) // ratio for _, p in rows], dtype=torch.int32)
    lens = lens.to(DEVICE)
    block_table = case.block_table[[req for req, _ in rows]].contiguous()
    entries = case.cache[ratio].shape[1]
    next_n = max(query_lens)
    context_lens = (torch.tensor(case.seq_lens, dtype=torch.int32) // ratio).to(DEVICE)
    native = native_decode(lens, block_table, query_lens, next_n, context_lens)
    schedule = torch.empty(
        ops.rocm_mxfp4_decode_schedule_words(HEADS, HEAD_DIM, entries, next_n),
        dtype=torch.int32,
        device=DEVICE,
    )
    return DeepseekV41RocmMxfp4IndexerMetadata(
        seq_lens=None,
        max_seq_len=max(case.seq_lens),
        slot_mapping=None,
        num_decodes=len(case.seq_lens),
        num_decode_tokens=len(rows),
        num_prefills=0,
        num_prefill_tokens=0,
        decode=types.SimpleNamespace(block_table=block_table, seq_lens=lens[:, None]),
        decode_row_lens=lens,
        decode_block_ends=(lens + CAND_BLOCK - 1) // CAND_BLOCK,
        decode_native=native,
        decode_schedule=ops.build_rocm_mxfp4_decode_schedule(
            lens, HEADS, HEAD_DIM, entries, schedule, MAX_LEN // ratio, native
        ),
    )


def _prefill_metadata(
    case, rows, ratio, chunk_bounds, query_start_loc, gather_rows, gather
):
    chunks = []
    for req_lo, req_hi, t0, t1 in chunk_bounds:
        ends = torch.tensor(
            [(rows[t][1] + 1) // ratio for t in range(t0, t1)], dtype=torch.int32
        ).to(DEVICE)
        chunks.append(
            DeepseekV32IndexerPrefillChunkMetadata(
                block_table=case.block_table[req_lo:req_hi],
                cu_seqlen_ks=torch.zeros_like(ends),
                cu_seqlen_ke=ends,
                cu_seq_lens=torch.zeros(1, dtype=torch.int32, device=DEVICE),
                token_to_seq=torch.zeros(1, dtype=torch.int32, device=DEVICE),
                total_seq_lens=1,
                token_start=t0,
                token_end=t1,
                num_reqs=req_hi - req_lo,
            )
        )
    seq_lens = torch.tensor(case.seq_lens)
    context_lens = (seq_lens // ratio).int().to(DEVICE)
    plans = plan_prefill_chunks(
        chunks,
        query_start_loc,
        seq_lens,
        context_lens,
        ratio,
        CAND_BLOCK,
        0.0 if gather else None,
        torch.tensor(query_start_loc, dtype=torch.int32, device=DEVICE),
    )
    return DeepseekV41RocmMxfp4IndexerMetadata(
        seq_lens=None,
        max_seq_len=max(case.seq_lens),
        slot_mapping=None,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=len(case.seq_lens),
        num_prefill_tokens=len(rows),
        prefill=types.SimpleNamespace(chunks=chunks),
        prefill_plans=plans,
        gather_launches=plan_gather_launches(chunks, plans, gather_rows),
    )


# 128-token pages are what vLLM allocates for this kernel; 64 is the default.
BLOCKS = pytest.mark.parametrize("block", [128, 64])


@BLOCKS
@pytest.mark.parametrize("ratio", [2, 1])
def test_k_cache_is_the_kernels_preshuffled_order(ratio, block):
    """The writer stores what aiter's preshuffle_cache makes of the natural
    bytes, page by page, inside the pool and nowhere else."""
    from aiter.ops.triton.attention.pa_mqa_logits_mxfp4 import preshuffle_cache

    case = _Case([700, 333], block)
    cache = case.cache[ratio]
    entries = cache.shape[1]
    for req, n in enumerate(case.seq_lens):
        ctx = n // ratio
        natural = case.natural[ratio][case.offsets[req] : case.offsets[req] + ctx, 0]
        for p in range(cdiv(ctx, entries)):
            page = torch.zeros(entries, WIDTH, dtype=torch.uint8, device=DEVICE)
            chunk = natural[p * entries : (p + 1) * entries]
            page[: chunk.shape[0]] = chunk
            values, scales = preshuffle_cache(
                page[None, :, : HEAD_DIM // 2],
                page[None, :, HEAD_DIM // 2 :],
                HEADS,
                HEAD_DIM,
            )
            stored = cache[int(case.block_table[req, p])].reshape(-1)
            torch.testing.assert_close(
                stored, torch.cat([values.reshape(-1), scales.reshape(-1)])
            )
    assert int(case.pool[~case.page_bytes].count_nonzero()) == 0


def _run_layers(monkeypatch, case, rows, metadata):
    """Ratio-2 dense layer, the candidate source, and a consumer both ways;
    each checked against the torch reference."""
    num_rows = len(rows)
    q, q_scale, weights, q_ref = _queries(num_rows)
    hidden = torch.empty(num_rows, 1, device=DEVICE)

    def topk_buffer():
        return torch.full((num_rows + 2, TOPK), 7, dtype=torch.int32, device=DEVICE)

    _forward_context(monkeypatch, metadata(2))
    out = topk_buffer()
    ops.rocm_mxfp4_sparse_attn_indexer(
        hidden,
        "indexer",
        case.cache[2],
        q,
        q_scale,
        weights,
        TOPK,
        HEAD_DIM,
        MAX_LEN // 2,
        out,
        compress_ratio=2,
    )
    for i, (req, pos) in enumerate(rows):
        end = (pos + 1) // 2
        keys = case.keys(2, req, end)
        _assert_topk(out[i], _scores(q_ref[i], weights[i], keys), TOPK)

    source = metadata(1)
    _forward_context(monkeypatch, source)
    pool = torch.full((num_rows + 2, POOL_BLOCKS), 5, dtype=torch.int32, device=DEVICE)
    out = topk_buffer()
    ops.rocm_mxfp4_sparse_attn_indexer(
        hidden,
        "indexer",
        case.cache[1],
        q,
        q_scale,
        weights,
        TOPK,
        HEAD_DIM,
        MAX_LEN,
        out,
        compress_ratio=1,
        candidate_blocks=pool,
        candidate_block_size=CAND_BLOCK,
        candidate_write=True,
    )
    allowed = []
    for i, (req, pos) in enumerate(rows):
        scores = _scores(q_ref[i], weights[i], case.keys(1, req, pos + 1))
        _assert_topk(out[i], scores, TOPK)
        _assert_topk(pool[i], _block_maxima(scores), POOL_BLOCKS)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        for block in pool[i][pool[i] >= 0].tolist():
            mask[block * CAND_BLOCK : (block + 1) * CAND_BLOCK] = True
        allowed.append(torch.where(mask, scores, float("-inf")))

    for gather in (True, False):
        consumer = metadata(1, gather)
        consumer.decode_use_gather = gather
        _forward_context(monkeypatch, consumer)
        out = topk_buffer()
        ops.rocm_mxfp4_sparse_mqa_indexer(
            hidden,
            "indexer",
            case.cache[1],
            q,
            q_scale,
            weights,
            TOPK,
            HEAD_DIM,
            MAX_LEN,
            out,
            1,
            pool,
            CAND_BLOCK,
            POOL_BLOCKS * CAND_BLOCK,
        )
        for i in range(num_rows):
            _assert_topk(out[i], allowed[i], TOPK)


@BLOCKS
@pytest.mark.parametrize(
    "query_lens",
    # Uniform steps launch the dense layers on next_n-row sequences, also with
    # cudagraph padding (query length 0) after them; a ragged step keeps a row
    # per token.
    [[6, 6, 6, 6], [2, 2, 2], [2, 2, 2, 0], [2, 1, 2]],
    ids=["native6", "native2", "padded", "ragged"],
)
def test_decode_layers_match_reference(monkeypatch, block, query_lens):
    next_n = max(query_lens)
    case = _Case([q and n for q, n in zip(query_lens, [900, 333, 610, 1200])], block)
    # a padding request has next_n rows of its own and no context
    rows = [
        (req, n - q + j if q else -1)
        for req, (n, q) in enumerate(zip(case.seq_lens, query_lens))
        for j in range(q or next_n)
    ]
    _run_layers(
        monkeypatch,
        case,
        rows,
        lambda r, gather=False: _decode_metadata(case, rows, r, query_lens),
    )


@pytest.mark.parametrize(
    "seq_lens,new_tokens,chunks",
    [
        # one request past a cached prefix, then one sliced into two chunks
        ([760, 500], [300, 500], [(0, 1, 0, 300), (1, 2, 300, 560), (1, 2, 560, 800)]),
        # three requests in one chunk, the first two with equal rows
        ([600, 450, 900], [200, 200, 120], [(0, 3, 0, 520)]),
        # four requests with four different query lengths in one chunk
        ([900, 600, 450, 700], [150, 90, 200, 60], [(0, 4, 0, 500)]),
    ],
    ids=["sliced", "batched", "ragged"],
)
# The consumers rejoin a request's rows across chunks, then cut them at most
# gather_rows a launch.
@pytest.mark.parametrize("gather_rows", [1 << 20, 128], ids=["joined", "cut"])
@BLOCKS
def test_prefill_layers_match_reference(
    monkeypatch, seq_lens, new_tokens, chunks, gather_rows, block
):
    case = _Case(seq_lens, block)
    rows = [
        (req, n - q + i)
        for req, (n, q) in enumerate(zip(seq_lens, new_tokens))
        for i in range(q)
    ]
    query_start_loc = [0, *itertools.accumulate(new_tokens)]
    _run_layers(
        monkeypatch,
        case,
        rows,
        lambda r, gather=True: _prefill_metadata(
            case, rows, r, chunks, query_start_loc, gather_rows, gather
        ),
    )
