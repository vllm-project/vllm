# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 MXFP4 K/V and index cache on ROCm gfx950: the fused write, the
index scores + top-k and the sparse attend, each against a torch reference on
the dequantized pages.

Needs VLLM_ROCM_USE_AITER=1 and an aiter with the MXFP4 FlyDSL ops
(ROCm/aiter#7056, #7057, #7060, #7061).
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import PreTrainedConfig

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm (gfx950) FlyDSL kernels", allow_module_level=True)

from vllm.platforms.rocm import on_gfx950  # noqa: E402

if not on_gfx950():
    pytest.skip("gfx950 instructions", allow_module_level=True)

from vllm._aiter_ops import rocm_aiter_ops  # noqa: E402
from vllm.models.minimax_m3.amd.indexer_mxfp4 import (  # noqa: E402
    _pin_forced_blocks,
    mxfp4_index_topk_decode,
    mxfp4_index_topk_prefill,
    rocm_mxfp4_aiter_unavailable_reason,
)
from vllm.models.minimax_m3.amd.model import (  # noqa: E402
    MiniMAXGemmaRMSNorm,
    MiniMaxM3SparseAttention,
    _build_rotary_emb,
)
from vllm.models.minimax_m3.amd.ops.mxfp4_pages import (  # noqa: E402
    PAGE_BYTES,
    mxfp4_page_views,
)
from vllm.models.minimax_m3.amd.ops.sparse_pa import (  # noqa: E402
    minimax_m3_build_sparse_block_table_decode,
    minimax_m3_build_sparse_block_table_prefill,
)
from vllm.models.minimax_m3.amd.sparse_attention_mxfp4 import (  # noqa: E402
    mxfp4_sparse_attend,
)
from vllm.models.minimax_m3.common.sparse_attention import (  # noqa: E402
    minimax_m3_query_token_positions,
)

reason = rocm_mxfp4_aiter_unavailable_reason()
if reason is not None:
    pytest.skip(f"MXFP4 ops: {reason}", allow_module_level=True)

BLOCK = 128
D = 128
TOPK = 16
NQ = 16  # query heads per rank at TP4
ROT = 64
SCALE = D**-0.5
SENTINEL = 0xA5
SLACK = 0.04
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def quantize_pages(rows: torch.Tensor) -> torch.Tensor:
    """uint8 pages ``[P, 8704]`` of float rows ``[P, 128, 128]``: per 32 values
    the e8m0 scale ceil_pow2(amax / 6) and the nearest e2m1 code."""
    x = rows.float().unflatten(-1, (4, 32))
    amax = x.abs().amax(-1).clamp_min(2.0**-100)
    exp = torch.ceil(torch.log2(amax / 6.0)).clamp(-100, 100)
    y = x / torch.exp2(exp)[..., None]
    grid = torch.tensor(E2M1, device=rows.device)
    code = (y.abs()[..., None] - grid).abs().argmin(-1) | ((y < 0) << 3)
    code = code.flatten(-2)
    data = (code[..., 0::2] | (code[..., 1::2] << 4)).to(torch.uint8)
    scale = (exp + 127).to(torch.uint8)
    p = rows.shape[0]
    return torch.cat([data.reshape(p, -1), scale.reshape(p, -1)], dim=-1)


def dequantize_pages(pages: torch.Tensor) -> torch.Tensor:
    """float32 rows ``[..., 128, 128]`` of uint8 pages ``[..., 8704]``."""
    data, scale = mxfp4_page_views(pages)
    d = data.int()
    code = torch.stack((d & 15, d >> 4), dim=-1).flatten(-2)
    grid = torch.tensor(E2M1, device=pages.device)
    value = grid[code & 7] * torch.where(code & 8 != 0, -1.0, 1.0)
    step = torch.exp2(scale.float() - 127.0)[..., None]
    return (value.unflatten(-1, (4, 32)) * step).flatten(-2)


def group_steps(pages: torch.Tensor) -> torch.Tensor:
    """Per-element value of one e8m0 step, ``[..., 128, 128]``."""
    _, scale = mxfp4_page_views(pages)
    step = torch.exp2(scale.float() - 127.0)
    return step.repeat_interleave(32, dim=-1)


def block_table(seq_lens, extra=2, seed=0):
    widths = [-(-s // BLOCK) for s in seq_lens]
    num_pages = sum(widths) + extra
    g = torch.Generator(device="cuda").manual_seed(seed)
    perm = torch.randperm(num_pages, device="cuda", generator=g).to(torch.int32)
    table = torch.zeros(len(seq_lens), max(widths), dtype=torch.int32, device="cuda")
    start = 0
    for r, w in enumerate(widths):
        table[r, :w] = perm[start : start + w]
        start += w
    return table, num_pages


# ---- the fused write -------------------------------------------------------


def _write_inputs(num_tokens, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    row = (NQ + 2 + 1 + 1) * D
    qkv = torch.randn(num_tokens, row, dtype=dtype, device="cuda", generator=g)
    w = (torch.randn(D, dtype=dtype, device="cuda", generator=g) * 0.1).to(dtype)
    cos_sin = torch.randn(
        max(num_tokens, 64), ROT, dtype=dtype, device="cuda", generator=g
    )
    return qkv, w, cos_sin, torch.arange(num_tokens, device="cuda")


@pytest.mark.parametrize("num_tokens", [1, 7, 300, 1031])
@pytest.mark.parametrize("skip_index", [False, True])
def test_write_matches_bf16_cache(num_tokens, skip_index):
    """K, V, index-K and the e4m3 q written through the vLLM wrapper are the
    bf16 rows of the "auto" cache to within one e8m0 step of their group."""
    import aiter

    qkv, w, cos_sin, pos = _write_inputs(num_tokens, seed=num_tokens)
    slots = torch.randperm(((num_tokens + 127) // 128 + 1) * 128, device="cuda")
    slots = slots[:num_tokens].to(torch.int64)
    if num_tokens > 5:
        slots[::5] = -1
    num_pages = (int(slots.max()) + 128) // 128 + 1

    k_ref = torch.zeros(num_pages, 128, 1, D, dtype=qkv.dtype, device="cuda")
    v_ref = torch.zeros_like(k_ref)
    i_ref = torch.zeros(num_pages, 128, D, dtype=qkv.dtype, device="cuda")
    q_ref = torch.empty(num_tokens, NQ * D, dtype=qkv.dtype, device="cuda")
    iq_ref = torch.empty(num_tokens, D, dtype=qkv.dtype, device="cuda")
    aiter.fused_qknorm_idxrqknorm(
        qkv.clone(), w, w, cos_sin, pos, NQ, 1, ROT, 1e-6, w, w, 1,
        slots, k_ref, v_ref, i_ref, 128, q_ref, iq_ref, slots,
        kv_cache_dtype="auto", skip_index_branch=skip_index,
    )  # fmt: skip

    # [pages, 2 * Hkv, 128, 68] as the model's KV cache, written per K / V head.
    kv = torch.full((num_pages, 2, 128, 68), SENTINEL, dtype=torch.uint8, device="cuda")
    index_pages = torch.full(
        (num_pages, PAGE_BYTES), SENTINEL, dtype=torch.uint8, device="cuda"
    )
    q8 = torch.empty(num_tokens, NQ * D, dtype=torch.float8_e4m3fn, device="cuda")
    iq = torch.empty_like(iq_ref)
    pages = kv.view(num_pages, 2, -1)
    rocm_aiter_ops.fused_qknorm_idxrqknorm_mxfp4(
        qkv.clone(), w, w, cos_sin, pos, NQ, 1, ROT, 1e-6, slots,
        pages[:, :1], pages[:, 1:], q_out=q8,
        index_q_norm_weight=w, index_k_norm_weight=w, num_index_heads=1,
        index_cache=None if skip_index else index_pages,
        index_q_out=None if skip_index else iq,
        index_slot_mapping=None if skip_index else slots,
        skip_index_branch=skip_index,
    )  # fmt: skip

    ok = slots >= 0
    page, tok = slots[ok] // 128, slots[ok] % 128

    def close(written_pages, ref_rows):
        got = dequantize_pages(written_pages)[page, ..., tok, :]
        step = group_steps(written_pages)[page, ..., tok, :]
        want = ref_rows[page, tok].reshape(got.shape).float()
        assert ((got - want).abs() <= step * 1.001).all()
        assert (got != 0).any()

    close(pages[:, :1], k_ref)
    close(pages[:, 1:], v_ref)
    # Slots that were not written keep their fill byte.
    unwritten = torch.ones(num_pages * 128, dtype=torch.bool, device="cuda")
    unwritten[slots[ok]] = False
    data, _ = mxfp4_page_views(pages[:, :1])
    assert (data[:, 0].reshape(-1, 64)[unwritten] == SENTINEL).all()
    if not skip_index:
        close(index_pages, i_ref)
        torch.testing.assert_close(iq, iq_ref)
    # The e4m3 q is the bf16 q to e4m3 precision (3 mantissa bits).
    q_diff = (q8.float() - q_ref.float()).abs()
    assert (q_diff <= 0.07 * q_ref.float().abs() + 2**-9).all()


@pytest.mark.parametrize("skip_index", [False, True])
def test_attention_layer_insert(skip_index):
    """``MiniMaxM3SparseAttention._norm_rope_mxfp4_insert`` fills the K/V pool
    and the index cache it is bound to with the normed, rotated rows.

    The reference rounds the norm and the rotation through bf16 on its own, so
    it differs from the kernel's fp32 chain by up to ``SLACK``."""
    from vllm.config import VllmConfig, set_current_vllm_config

    num_tokens, num_pages = 200, 3
    attention = object.__new__(MiniMaxM3SparseAttention)
    torch.nn.Module.__init__(attention)
    attention.num_heads, attention.num_kv_heads, attention.num_idx_heads = NQ, 1, 1
    attention.head_dim = attention.idx_head_dim = D
    attention.q_size, attention.kv_size, attention.index_q_size = NQ * D, D, D
    attention.skip_index_topk = skip_index
    for name in ("q_norm", "k_norm", "index_q_norm", "index_k_norm"):
        norm = MiniMAXGemmaRMSNorm(D).to("cuda", torch.bfloat16)
        norm.weight.data.normal_(0, 0.1)
        setattr(attention, name, norm)
    config = PreTrainedConfig(
        rope_theta=10000.0, partial_rotary_factor=0.5, max_position_embeddings=512
    )
    torch.set_default_dtype(torch.bfloat16)
    try:
        with set_current_vllm_config(VllmConfig()):
            attention.rotary_emb = _build_rotary_emb(config, D).to("cuda")
    finally:
        torch.set_default_dtype(torch.float32)
    attention.index_rotary_emb = attention.rotary_emb
    attention.kv_cache = torch.full(
        (num_pages, 2, 128, 68), SENTINEL, dtype=torch.uint8, device="cuda"
    )
    index_cache = torch.full(
        (num_pages, 1, 128, 68), SENTINEL, dtype=torch.uint8, device="cuda"
    )
    attention.indexer = SimpleNamespace(
        index_cache=SimpleNamespace(
            query_dtype=torch.bfloat16, kv_cache=index_cache.squeeze(1)
        )
    )

    width = attention.q_size + 2 * D + D + D
    qkv = torch.randn(num_tokens, width, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(num_tokens, device="cuda")
    slots = torch.randperm(num_pages * 128, device="cuda")[:num_tokens]
    index_slots = torch.randperm(num_pages * 128, device="cuda")[:num_tokens]
    q_raw, k_raw, v_raw, iq_raw, ik_raw = qkv.clone().split(
        [attention.q_size, D, D, D, D], dim=-1
    )
    q_ref = attention.q_norm(q_raw.view(num_tokens, NQ, D))
    k_ref = attention.k_norm(k_raw.view(num_tokens, 1, D))
    q_ref, k_ref = attention.rotary_emb.forward_native(positions, q_ref, k_ref)
    iq_ref = attention.index_q_norm(iq_raw.view(num_tokens, 1, D))
    ik_ref = attention.index_k_norm(ik_raw.view(num_tokens, D))
    iq_ref, ik_ref = attention.rotary_emb.forward_native(positions, iq_ref, ik_ref)

    q, index_q = attention._norm_rope_mxfp4_insert(qkv, positions, slots, index_slots)

    assert q.dtype == torch.float8_e4m3fn and q.shape == (num_tokens, NQ * D)
    q_err = (q.float() - q_ref.reshape(num_tokens, -1).float()).abs()
    assert (q_err <= 0.07 * q_ref.reshape(num_tokens, -1).float().abs() + SLACK).all()
    assert (index_q is None) == skip_index

    def written(pages, rows, slot):
        got = dequantize_pages(pages)[slot // 128, ..., slot % 128, :]
        step = group_steps(pages)[slot // 128, ..., slot % 128, :]
        assert (
            (got - rows.reshape(got.shape).float()).abs() <= step * 1.001 + SLACK
        ).all()

    pages = attention.kv_cache.view(num_pages, 2, -1)
    written(pages[:, :1], k_ref, slots)
    written(pages[:, 1:], v_raw.view(num_tokens, 1, D), slots)
    if not skip_index:
        written(index_cache.view(num_pages, PAGE_BYTES), ik_ref, index_slots)
        torch.testing.assert_close(
            index_q, iq_ref.reshape(num_tokens, -1), atol=SLACK, rtol=0.05
        )
    else:  # a layer reusing another layer's top-k leaves the index cache alone
        assert (index_cache == SENTINEL).all()


# ---- index scores + top-k --------------------------------------------------


def _keys_pages(num_pages, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = torch.randn(num_pages, BLOCK, D, device="cuda", generator=g) * 2
    return quantize_pages(rows)


def _reference_scores(q, keys, table, req, causal, init, local):
    """[num_blocks] block scores of one query row, pins applied."""
    nb = -(-causal // BLOCK)
    score = torch.empty(nb, device="cuda")
    for j in range(nb):
        valid = min(BLOCK, causal - j * BLOCK)
        score[j] = (keys[table[req, j]][:valid] @ q).max() * SCALE
    score[: min(init, nb)] = 1e30
    score[max(nb - local, 0) :] = 1e29
    return score


def _check_topk(got, score):
    """``got`` is a top-16 of ``score`` up to rounding of near ties."""
    nb = score.numel()
    ids = got[got >= 0].tolist()
    assert len(ids) == min(TOPK, nb) and len(set(ids)) == len(ids)
    assert all(0 <= i < nb for i in ids)
    assert (got[len(ids) :] == -1).all()
    kth = score.topk(min(TOPK, nb)).values[-1]
    tol = 1e-3 * score[score < 1e28].abs().max().clamp_min(1.0) if nb > 1 else 0
    assert all(score[i] >= kth - tol for i in ids)
    assert nb - 1 in ids  # the local (self) block is always selected


def _pinned(score, num_blocks, init, local):
    want = score.clone()
    for r, n in enumerate(num_blocks):
        want[0, r, : min(init, n)] = 1e30
        want[0, r, max(n - local, 0) : n] = 1e29
    return want


@pytest.mark.parametrize("init,local", [(0, 1), (2, 3)])
def test_pin_forced_blocks_prefill(init, local):
    nb = torch.randint(0, 20, (40,), dtype=torch.int32, device="cuda")
    score = torch.randn(1, 40, 64, device="cuda")
    want = _pinned(score, nb.tolist(), init, local)
    _pin_forced_blocks(score, init_blocks=init, local_blocks=local, num_valid_pages=nb)
    assert torch.equal(score, want)


@pytest.mark.parametrize("init,local", [(0, 1), (2, 3)])
def test_pin_forced_blocks_decode(init, local):
    """Rows of 4-token requests; a row sees seq_len - 4 + t + 1 tokens."""
    qlen = 4
    seq_lens = torch.randint(1, 3000, (10,), dtype=torch.int32, device="cuda")
    score = torch.randn(1, 10 * qlen, 64, device="cuda")
    causal = [max(int(seq_lens[r // qlen]) - qlen + r % qlen + 1, 0) for r in range(40)]
    want = _pinned(score, [-(-c // BLOCK) for c in causal], init, local)
    _pin_forced_blocks(
        score, init_blocks=init, local_blocks=local, seq_lens=seq_lens, query_len=qlen
    )
    assert torch.equal(score, want)


@pytest.mark.parametrize("use_work_map", [False, True])
@pytest.mark.parametrize("query_len", [1, 4])
@pytest.mark.parametrize("seq_lens", [[300, 1500, 9000], [128, 129, 4096, 2500]])
def test_decode_index_topk(seq_lens, query_len, use_work_map):
    from aiter.ops.flydsl import minimax_m3_index_score_work_map

    init, local = 0, 1
    table, num_pages = block_table(seq_lens)
    index_pages = _keys_pages(num_pages)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    rows = len(seq_lens) * query_len
    g = torch.Generator(device="cuda").manual_seed(1)
    q = torch.randn(rows, 1, D, device="cuda", generator=g).bfloat16()
    topk = torch.empty(1, rows, TOPK, dtype=torch.int32, device="cuda")
    max_seq_len = max(seq_lens)
    work_map = None
    if use_work_map:
        work_map = minimax_m3_index_score_work_map(
            lens, -(-max_seq_len // BLOCK), query_len, 1
        )
    mxfp4_index_topk_decode(
        q,
        index_pages,
        table,
        lens,
        work_map,
        topk,
        max_seq_len=max_seq_len,
        query_len=query_len,
        scale=SCALE,
        init_blocks=init,
        local_blocks=local,
    )
    keys = dequantize_pages(index_pages)
    for r in range(rows):
        req, t = divmod(r, query_len)
        causal = seq_lens[req] - query_len + t + 1
        score = _reference_scores(
            q[r, 0].float(), keys, table, req, causal, init, local
        )
        _check_topk(topk[0, r], score)


@pytest.mark.parametrize(
    "contexts,queries",
    [([0, 200], [150, 300]), ([1000], [700]), ([0, 0, 129], [5, 1, 3])],
)
def test_prefill_index_topk(contexts, queries):
    init, local = 0, 1
    seq_lens = [c + n for c, n in zip(contexts, queries)]
    table, num_pages = block_table(seq_lens)
    index_pages = _keys_pages(num_pages, seed=3)
    total = sum(queries)
    g = torch.Generator(device="cuda").manual_seed(2)
    q = torch.randn(total, 1, D, device="cuda", generator=g).bfloat16()
    seq = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    ctx = torch.tensor(contexts, dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, *queries], device="cuda").cumsum(0).to(torch.int32)
    req_id, abs_pos = minimax_m3_query_token_positions(cu, ctx, total)
    num_valid = (abs_pos // BLOCK + 1).to(torch.int32)
    topk = torch.empty(1, total, TOPK, dtype=torch.int32, device="cuda")
    mxfp4_index_topk_prefill(
        q,
        index_pages,
        table,
        cu,
        seq,
        ctx,
        num_valid,
        topk,
        max_query_len=max(queries),
        max_seq_len=max(seq_lens),
        scale=SCALE,
        init_blocks=init,
        local_blocks=local,
    )
    keys = dequantize_pages(index_pages)
    for r in range(total):
        score = _reference_scores(
            q[r, 0].float(),
            keys,
            table,
            int(req_id[r]),
            int(abs_pos[r]) + 1,
            init,
            local,
        )
        _check_topk(topk[0, r], score)


# ---- the sparse attend -----------------------------------------------------


def _kv_pool(num_pages, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    k = torch.randn(num_pages, BLOCK, D, device="cuda", generator=g)
    v = torch.randn(num_pages, BLOCK, D, device="cuda", generator=g)
    pool = torch.stack([quantize_pages(k), quantize_pages(v)], dim=1)
    return pool.view(num_pages, 2, BLOCK, 68)


def _selection(causal_lens, seed=0):
    """[rows, 16] random block ids, the tail block among them, -1 padded."""
    g = torch.Generator().manual_seed(seed)
    out = torch.full((len(causal_lens), TOPK), -1, dtype=torch.int32)
    for r, causal in enumerate(causal_lens):
        nb = -(-causal // BLOCK)
        pick = torch.randperm(nb - 1, generator=g)[: TOPK - 1].tolist() + [nb - 1]
        order = torch.randperm(len(pick), generator=g)
        out[r, : len(pick)] = torch.tensor(pick, dtype=torch.int32)[order]
    return out.cuda()


def _reference_attend(q8, pool, table, req_of_row, causal_lens, selection):
    keys = dequantize_pages(pool[:, 0].reshape(pool.shape[0], PAGE_BYTES))
    vals = dequantize_pages(pool[:, 1].reshape(pool.shape[0], PAGE_BYTES))
    out = torch.empty(q8.shape, dtype=torch.float32, device="cuda")
    for r in range(q8.shape[0]):
        blocks = [b for b in selection[r].tolist() if b >= 0]
        idx = [
            (int(table[req_of_row[r], b]), min(BLOCK, causal_lens[r] - b * BLOCK))
            for b in blocks
        ]
        k = torch.cat([keys[p][:n] for p, n in idx])
        v = torch.cat([vals[p][:n] for p, n in idx])
        logits = (q8[r].float() @ k.T) * SCALE
        out[r] = torch.softmax(logits, dim=-1) @ v
    return out


def _fp8_query(rows, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(rows, NQ, D, device="cuda", generator=g) * 2).to(
        torch.float8_e4m3fn
    )


def _attend_error(out, ref):
    return (out.float() - ref).abs().max().item(), ref.abs().max().item()


@pytest.mark.parametrize("query_len", [1, 4])
@pytest.mark.parametrize("seq_lens", [[300, 1500], [129, 5000, 640]])
def test_decode_sparse_attend(seq_lens, query_len):
    table, num_pages = block_table(seq_lens)
    pool = _kv_pool(num_pages)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    rows = len(seq_lens) * query_len
    req_of_row = [r // query_len for r in range(rows)]
    causal = [
        seq_lens[r // query_len] - query_len + r % query_len + 1 for r in range(rows)
    ]
    selection = _selection(causal)
    q8 = _fp8_query(rows)
    sparse_bt, sparse_ctx = minimax_m3_build_sparse_block_table_decode(
        selection[None],
        table,
        lens,
        query_len,
        block_page_stride=1,
        pages_per_block=1,
    )
    assert sparse_bt.shape == (rows, TOPK)
    assert sparse_ctx.tolist() == [
        sum(min(BLOCK, c - b * BLOCK) for b in selection[r].tolist() if b >= 0)
        for r, c in enumerate(causal)
    ]
    out = torch.empty(rows, NQ, D, dtype=torch.bfloat16, device="cuda")
    mxfp4_sparse_attend(q8, pool, sparse_bt, sparse_ctx, out, SCALE)
    ref = _reference_attend(q8, pool, table, req_of_row, causal, selection)
    err, mag = _attend_error(out, ref)
    assert err <= 2e-2 * max(mag, 1.0), (err, mag)


@pytest.mark.parametrize("contexts,queries", [([0, 200], [150, 300]), ([1000], [260])])
def test_prefill_sparse_attend(contexts, queries):
    seq_lens = [c + n for c, n in zip(contexts, queries)]
    table, num_pages = block_table(seq_lens)
    pool = _kv_pool(num_pages, seed=1)
    total = sum(queries)
    ctx = torch.tensor(contexts, dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, *queries], device="cuda").cumsum(0).to(torch.int32)
    req_id, abs_pos = minimax_m3_query_token_positions(cu, ctx, total)
    causal = (abs_pos + 1).tolist()
    selection = _selection(causal, seed=1)
    q8 = _fp8_query(total, seed=1)
    sparse_bt, sparse_ctx = minimax_m3_build_sparse_block_table_prefill(
        selection[None],
        table,
        req_id,
        abs_pos,
        block_page_stride=1,
        pages_per_block=1,
    )
    out = torch.empty(total, NQ, D, dtype=torch.bfloat16, device="cuda")
    mxfp4_sparse_attend(q8, pool, sparse_bt, sparse_ctx, out, SCALE)
    ref = _reference_attend(q8, pool, table, req_id.tolist(), causal, selection)
    err, mag = _attend_error(out, ref)
    assert err <= 2e-2 * max(mag, 1.0), (err, mag)
