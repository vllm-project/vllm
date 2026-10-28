# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DASH paged KV kernels and segmented FlashAttention/FlashInfer execution."""

import math
from functools import cache

import torch

from vllm.model_executor.layers.unlimited_ocr2 import (
    page_coordinates,
    page_rope,
    route_blocks,
)
from vllm.triton_utils import tl, triton


@cache
def _rope_frequencies(device, dim, pairs, base, page_base, truncate):
    ntok = dim // 2 - pairs
    exponent = torch.arange(ntok, device=device).float()
    exponent = exponent / (dim / 2 if truncate else ntok)
    inv_tok = 1.0 / (base**exponent)
    inv_page = 1.0 / (page_base ** (torch.arange(pairs, device=device).float() / pairs))
    return torch.cat((inv_tok, inv_page))


@triton.jit
def _rope(
    X,
    Tok,
    Page,
    Inv,
    Out,
    XS: tl.constexpr,
    XH: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    CS: tl.constexpr,
    NTOK: tl.constexpr,
    MOD: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, D)
    pair = d % (D // 2)
    tok = tl.load(Tok + row * CS).to(tl.float32)
    page = tl.load(Page + row * CS)
    if MOD > 0:
        page = page % MOD
    coord = tl.where(pair < NTOK, tok, page.to(tl.float32))
    angle = coord * tl.load(Inv + pair)
    dtype = X.dtype.element_ty
    cos = tl.extra.cuda.libdevice.cos(angle).to(dtype).to(tl.float32)
    sin = tl.extra.cuda.libdevice.sin(angle).to(dtype).to(tl.float32)
    x = tl.load(X + row * XS + head * XH + d).to(tl.float32)
    other = tl.load(X + row * XS + head * XH + (d + D // 2) % D)
    other = tl.where(d < D // 2, -other.to(tl.float32), other.to(tl.float32))
    a = (x * cos).to(dtype).to(tl.float32)
    b = (other * sin).to(dtype).to(tl.float32)
    tl.store(Out + (row * HEADS + head) * D + d, a + b)


def decode_page_rope(x, tok, page, sp, *, main):
    dim = x.shape[-1]
    pairs = int(sp.get("sparse_rope_page_pairs", 1))
    mode = "sparse_main_rope_tok_freq_mode" if main else "sparse_rope_tok_freq_mode"
    inv = _rope_frequencies(
        x.device,
        dim,
        pairs,
        float(sp.get("sparse_rope_tok_theta", 10000.0)),
        float(sp.get("sparse_rope_page_theta", 256.0)),
        sp.get(mode) == "truncate",
    )
    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    heads = x.shape[1] if x.ndim == 3 else 1
    _rope[(x.shape[0], heads)](
        x,
        tok,
        page,
        inv,
        y,
        x.stride(0),
        x.stride(1) if x.ndim == 3 else 0,
        heads,
        dim,
        tok.stride(0),
        dim // 2 - pairs,
        int(sp.get("sparse_rope_page_mod", 0)),
        enable_fp_fusion=False,
    )
    return y


@triton.jit
def _meta(
    C,
    T,
    row,
    pos,
    field: tl.constexpr,
    valid,
    CB: tl.constexpr,
    CH: tl.constexpr,
    CT: tl.constexpr,
    TS: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
):
    # A large per-layer KV allocation exceeds the signed int32 element range.
    block = tl.load(T + row * TS + pos // B, valid, 0).to(tl.int64)
    ptr = C + block * CB + H * CH + pos % B * CT + D + field
    b0 = tl.load(ptr, valid, 0).to(tl.uint32)
    b1 = tl.load(ptr + 1, valid, 0).to(tl.uint32)
    b2 = tl.load(ptr + 2, valid, 0).to(tl.uint32)
    b3 = tl.load(ptr + 3, valid, 0).to(tl.uint32)
    return (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)).to(tl.int32, bitcast=True)


@triton.jit
def _coordinates(
    C,
    T,
    Lens,
    Slots,
    Roles,
    Coords,
    CB: tl.constexpr,
    CH: tl.constexpr,
    CT: tl.constexpr,
    TS: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    MAX_LEN: tl.constexpr,
):
    row = tl.program_id(0)
    pos = tl.load(Lens + row) - 1
    valid = (pos > 0) & (tl.load(Slots + row) >= 0)
    old_tok = _meta(C, T, row, pos - 1, 2, valid, CB, CH, CT, TS, B, H, D)
    old_page = _meta(C, T, row, pos - 1, 6, valid, CB, CH, CT, TS, B, H, D)
    pages = _meta(C, T, row, pos - 1, 10, valid, CB, CH, CT, TS, B, H, D)
    old_query = _meta(C, T, row, pos - 1, 14, valid, CB, CH, CT, TS, B, H, D)
    pages = tl.minimum(tl.maximum(pages, 1), MAX_LEN - 1)
    old_page = tl.minimum(tl.maximum(old_page, 0), pages)
    is_page = tl.load(Roles + row) == 2
    page = tl.where(is_page, tl.minimum(old_page + 1, pages), old_page)
    count = _meta(
        C, T, row, page, 18, valid & is_page & (page <= pos), CB, CH, CT, TS, B, H, D
    )
    tok = tl.where(is_page, count, tl.where(valid, old_tok + 1, 0))
    query = tl.where(valid, old_query, -1) + is_page.to(tl.int32)
    tl.store(Coords + row * 4, tok)
    tl.store(Coords + row * 4 + 1, page)
    tl.store(Coords + row * 4 + 2, pages)
    tl.store(Coords + row * 4 + 3, query)


@triton.jit
def _write(
    C,
    K,
    V,
    IK,
    Roles,
    Coords,
    Slots,
    CB: tl.constexpr,
    CH: tl.constexpr,
    CT: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    IS: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    slot = tl.load(Slots + row).to(tl.int64)
    d = tl.arange(0, D)
    ptr = C + slot // B * CB + head * CH + slot % B * CT
    if head < H:
        k = tl.load(K + row * KS + head * KH + d)
        v = tl.load(V + row * VS + head * VH + d)
        tl.store(ptr + d, k, slot >= 0)
        tl.store(ptr + D + d, v, slot >= 0)
    else:
        ik = tl.load(IK + row * IS + d)
        tl.store(ptr + d, ik, slot >= 0)
        role = tl.load(Roles + row)
        tl.store(ptr + D, role, slot >= 0)
        fields = tl.arange(0, 4)
        values = tl.load(Coords + row * 4 + fields).to(tl.uint32, bitcast=True)
        byte = tl.arange(0, 4)
        encoded = (values[:, None] >> (byte[None, :] * 8)) & 255
        tl.store(ptr + D + 2 + fields[:, None] * 4 + byte[None, :], encoded, slot >= 0)


@triton.jit
def _summarize(keys, cls, D: tl.constexpr):
    logits = tl.sum(keys * cls[None, :], 1) * (1.0 / math.sqrt(D))
    logits = tl.minimum(tl.maximum(logits, -1e4), 1e4)
    weight = tl.exp(logits - tl.max(logits, 0))
    weight = weight / tl.maximum(tl.sum(weight, 0), 1e-6)
    return tl.sum(weight[:, None] * keys, 0)


@triton.jit
def _cache_summaries(
    C,
    Slots,
    HeadCls,
    CB: tl.constexpr,
    CH: tl.constexpr,
    CT: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
):
    block_id = tl.program_id(0)
    p = block_id * BS + tl.arange(0, BS)
    d = tl.arange(0, D)
    slot = tl.load(Slots + p).to(tl.int64)
    ptr = C + slot // B * CB + H * CH + slot % B * CT
    keys = tl.load(ptr[:, None] + d[None, :]).to(tl.float32)
    cls = tl.load(HeadCls + d).to(tl.float32)
    summary = _summarize(keys, cls, D)
    bits = summary.to(tl.uint32, bitcast=True)
    byte = tl.arange(0, 4)
    index = d[:, None] * 4 + byte[None, :]
    slot = tl.load(Slots + block_id * BS + index // (D - 32)).to(tl.int64)
    dest = C + slot // B * CB + H * CH + slot % B * CT
    # Four exactly represented bytes preserve FP32 summaries in BF16 storage.
    tl.store(
        dest + D + 32 + index % (D - 32), (bits[:, None] >> (byte[None, :] * 8)) & 255
    )


@triton.jit
def _scores(
    C,
    T,
    Lens,
    Prefix,
    IQ,
    HeadCls,
    Coords,
    Scores,
    Candidate,
    CB: tl.constexpr,
    CH: tl.constexpr,
    CT: tl.constexpr,
    TS: tl.constexpr,
    IS: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    LOCAL_BLOCKS: tl.constexpr,
    NB: tl.constexpr,
    RSWA: tl.constexpr,
    CACHED: tl.constexpr,
):
    row, block_id = tl.program_id(0), tl.program_id(1)
    length = tl.load(Lens + row)
    pos = length - 1
    score = -1.0e9
    candidate = False
    prefix = length
    if RSWA:
        prefix = tl.load(Prefix + row)
    if (block_id <= pos // BS - LOCAL_BLOCKS) & (block_id * BS < prefix):
        p = block_id * BS + tl.arange(0, BS)
        d = tl.arange(0, D)
        block = tl.load(T + row * TS + p // B, p < prefix, 0).to(tl.int64)
        ptr = C + block * CB + H * CH + p % B * CT
        if CACHED and RSWA and (block_id + 1) * BS <= prefix:
            byte = tl.arange(0, 4)
            index = d[:, None] * 4 + byte[None, :]
            summary_pos = block_id * BS + index // (D - 32)
            physical = tl.load(T + row * TS + summary_pos // B).to(tl.int64)
            summary_ptr = C + physical * CB + H * CH + summary_pos % B * CT
            encoded = tl.load(summary_ptr + D + 32 + index % (D - 32))
            bits = tl.sum(encoded.to(tl.uint32) << (byte[None, :] * 8), 1)
            summary = bits.to(tl.float32, bitcast=True)
        else:
            keys = tl.load(ptr[:, None] + d[None, :], (p < prefix)[:, None], 0).to(
                tl.float32
            )
            cls = tl.load(HeadCls + d).to(tl.float32)
            logits = tl.sum(keys * cls[None, :], 1) * (1.0 / math.sqrt(D))
            logits = tl.minimum(tl.maximum(logits, -1e4), 1e4)
            logits = tl.where(p < prefix, logits, -float("inf"))
            weight = tl.exp(logits - tl.max(logits, 0))
            weight = weight / tl.maximum(tl.sum(weight, 0), 1e-6)
            summary = tl.sum(weight[:, None] * keys, 0)
        iq = tl.load(IQ + row * IS + d).to(tl.float32)
        score = tl.sum(summary * iq, 0) * (1.0 / math.sqrt(D))
        score = tl.minimum(tl.maximum(score, -1e4), 1e4)
        role = tl.load(ptr + D, p < prefix, 0)
        image_page = _meta(C, T, row, p, 14, p < prefix, CB, CH, CT, TS, B, H, D)
        qp = tl.load(Coords + row * 4 + 3)
        candidate = (
            tl.sum(((role == 1) & (image_page == qp) & (qp >= 0)).to(tl.int32), 0) > 0
        )
    tl.store(Scores + row * NB + block_id, score)
    tl.store(Candidate + row * NB + block_id, candidate)


def cache_dash_prefill(k, v, ik, roles, slots, cache, sp, head_cls=None):
    """Store rotated keys and integer page state in one auxiliary KV head."""
    tok, page, _, query_page = page_coordinates(roles)
    num_pages = int(page.max())
    last_image = torch.where(roles == 1)[0][-1]
    prefill = torch.arange(len(roles), device=roles.device) <= last_image
    prompt_counts = torch.zeros(num_pages + 1, device=roles.device, dtype=torch.long)
    prompt_counts.scatter_add_(0, page, prefill.long())
    state = torch.stack(
        (tok, page, torch.full_like(tok, num_pages), query_page), -1
    ).to(torch.int32)
    shifts = torch.arange(4, device=k.device) * 8
    encoded = ((state[:, :, None] >> shifts) & 255).flatten(1)
    counts = ((prompt_counts[:, None] >> shifts) & 255).to(k.dtype)
    h, dim = k.shape[1:]
    packed = torch.zeros((len(k), h + 1, 2 * dim), device=k.device, dtype=k.dtype)
    packed[:, :h, :dim] = page_rope(k, tok, page, sp, main=True)
    packed[:, :h, dim:] = v
    packed[:, h, :dim] = page_rope(ik, tok, page, sp, main=False)
    packed[:, h, dim] = roles
    packed[:, h, dim + 2 : dim + 18] = encoded
    packed[: num_pages + 1, h, dim + 18 : dim + 22] = counts
    b = cache.shape[2]
    cache[slots // b, :, slots % b] = packed
    if head_cls is not None:
        bs = int(sp["sparse_block_size"])
        cb, ch, ct, _ = cache.stride()
        _cache_summaries[(len(k) // bs,)](
            cache,
            slots,
            head_cls,
            cb,
            ch,
            ct,
            b,
            h,
            dim,
            bs,
            num_warps=4,
        )


@triton.jit
def _gather_segments(
    C,
    T,
    L,
    P,
    S,
    INDICES,
    VALID,
    GK,
    GV,
    USED,
    CB: tl.constexpr,
    CH: tl.constexpr,
    CT: tl.constexpr,
    TS: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    LOCAL: tl.constexpr,
    K: tl.constexpr,
    WIDTH: tl.constexpr,
    RSWA: tl.constexpr,
):
    row, group, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(L + row)
    active = (length > 0) & (tl.load(S + row) >= 0)
    prefix = tl.load(P + row) if RSWA else length
    if group < K:
        start = tl.load(INDICES + row * K + group).to(tl.int32) * BS
        valid = tl.load(VALID + row * K + group)
        used = tl.maximum(
            0, tl.minimum(BS, tl.minimum(length - LOCAL - start, prefix - start))
        )
        used = tl.where(active & valid, used, 0)
        size = BS
        offset = group * BS
    else:
        start = tl.maximum(0, length - LOCAL)
        used = tl.where(active, tl.minimum(LOCAL, length), 0)
        size = LOCAL
        offset = K * BS
    if head == 0:
        tl.store(USED + row * (K + 1) + group, used)
    x = tl.arange(0, WIDTH)
    d = tl.arange(0, D)
    pos = start + x
    live = x < used
    block = tl.load(T + row * TS + pos // B, live, 0).to(tl.int64)
    ptr = C + block[:, None] * CB + head * CH + (pos % B)[:, None] * CT + d[None]
    key = tl.load(ptr, live[:, None], 0)
    value = tl.load(ptr + D, live[:, None], 0)
    dest = ((row * (K * BS + LOCAL) + offset + x[:, None]) * H + head) * D + d[None]
    tl.store(GK + dest, key, (x < size)[:, None])
    tl.store(GV + dest, value, (x < size)[:, None])


def paged_dash_decode(
    q,
    k,
    v,
    iq,
    ik,
    roles,
    cache,
    block_table,
    seq_lens,
    slots,
    head_cls,
    sp,
    layer_idx,
    max_model_len,
    prefix_lens=None,
    backend="FLASH_ATTN",
    cached_summaries=False,
):
    """One query per request; all sequence-dependent decisions stay on GPU."""
    n, nh, dim = q.shape
    h = k.shape[1]
    b = cache.shape[2]
    cb, ch, ct, _ = cache.stride()
    ts = block_table.stride(0)
    coords = torch.empty((n, 4), dtype=torch.int32, device=q.device)
    _coordinates[(n,)](
        cache,
        block_table,
        seq_lens,
        slots,
        roles,
        coords,
        cb,
        ch,
        ct,
        ts,
        b,
        h,
        dim,
        max_model_len,
    )
    q = decode_page_rope(q, coords[:, 0], coords[:, 1], sp, main=True)
    k = decode_page_rope(k, coords[:, 0], coords[:, 1], sp, main=True)
    iq = decode_page_rope(iq, coords[:, 0], coords[:, 1], sp, main=False)
    ik = decode_page_rope(ik, coords[:, 0], coords[:, 1], sp, main=False)
    _write[(n, h + 1)](
        cache,
        k,
        v,
        ik,
        roles,
        coords,
        slots,
        cb,
        ch,
        ct,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        ik.stride(0),
        b,
        h,
        dim,
    )
    bs = int(sp["sparse_block_size"])
    nb = triton.cdiv(max_model_len, bs)
    scores = torch.empty((n, nb), device=q.device, dtype=torch.float32)
    candidate = torch.empty((n, nb), device=q.device, dtype=torch.bool)
    _scores[(n, nb)](
        cache,
        block_table,
        seq_lens,
        prefix_lens if prefix_lens is not None else seq_lens,
        iq,
        head_cls,
        coords,
        scores,
        candidate,
        cb,
        ch,
        ct,
        ts,
        iq.stride(0),
        b,
        h,
        dim,
        bs,
        int(sp["sparse_local_block"]),
        nb,
        prefix_lens is not None,
        cached_summaries,
        num_warps=4,
    )
    indices, bias, valid = route_blocks(scores, candidate, sp, layer_idx)
    indices = indices.contiguous()
    local = int(sp["sparse_local_block"]) * bs
    topk = indices.shape[-1]
    keys = torch.empty(
        (n * (topk * bs + local), h, dim), device=q.device, dtype=q.dtype
    )
    values = torch.empty_like(keys)
    used = torch.empty(n * (topk + 1), device=q.device, dtype=torch.int32)
    _gather_segments[(n, topk + 1, h)](
        cache,
        block_table,
        seq_lens,
        prefix_lens if prefix_lens is not None else seq_lens,
        slots,
        indices,
        valid,
        keys,
        values,
        used,
        cb,
        ch,
        ct,
        ts,
        b,
        h,
        dim,
        bs,
        local,
        topk,
        triton.next_power_of_2(max(bs, local)),
        prefix_lens is not None,
    )
    return flash_dash_attention(q, keys, values, used, bias, backend, bs, local)


@cache
def _indptr(device, n, topk, bs, local):
    lengths = torch.tensor([bs] * topk + [local], dtype=torch.int32)
    kv = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.repeat(n).cumsum(0)))
    qo = torch.arange(n * (topk + 1) + 1, dtype=torch.int32)
    return qo, kv.to(torch.int32), qo.to(device), kv.to(device, dtype=torch.int32)


@cache
def _workspace(device):
    return torch.empty(128 * 1024**2, dtype=torch.uint8, device=device)


@cache
def _flashinfer_plan(device, n, topk, bs, local, heads, kv_heads, dim, dtype):
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("DASH FlashInfer plans must be warmed up before capture")
    qo, kv, qo_gpu, kv_gpu = _indptr(device, n, topk, bs, local)
    total = n * (topk * bs + local)
    packed_mask = torch.empty(total // 8, dtype=torch.uint8, device=device)
    mask_indptr = torch.empty_like(qo_gpu)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(
        _workspace(device),
        kv_layout="NHD",
        use_cuda_graph=True,
        qo_indptr_buf=qo_gpu,
        kv_indptr_buf=kv_gpu,
        custom_mask_buf=packed_mask,
        mask_indptr_buf=mask_indptr,
        backend="fa2",
    )
    wrapper.plan(
        qo,
        kv,
        heads,
        kv_heads,
        dim,
        custom_mask=torch.ones(total, dtype=torch.bool, device=device),
        causal=False,
        q_data_type=dtype,
        kv_data_type=dtype,
        sm_scale=dim**-0.5,
        disable_split_kv=True,
    )
    return wrapper, packed_mask


def flash_dash_attention(q, k, v, used, bias, backend, bs, local):
    """Attend gathered segments and combine normalized partial outputs.

    k/v are contiguous [N*(topk*bs+local), Hkv, D]; used contains the
    valid prefix length of each segment, including zero for inactive rows.
    """
    n, heads, dim = q.shape
    topk = bias.shape[1]
    groups = topk + 1
    device = q.device
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("DASH FlashAttention/FlashInfer require FP16 or BF16")
    if bs % 8 or local % 8:
        raise ValueError("DASH Flash backends require byte-aligned segments")
    _, _, cuq, cuk = _indptr(device, n, topk, bs, local)
    queries = q[:, None].expand(n, groups, heads, dim).reshape(-1, heads, dim)
    if backend == "FLASH_ATTN":
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

        version = get_flash_attn_version(head_size=dim)
        if version is None:
            raise RuntimeError(
                "DASH FLASH_ATTN requires a supported CUDA FlashAttention"
            )
        out, lse = flash_attn_varlen_func(
            queries,
            k,
            v,
            max_seqlen_q=1,
            cu_seqlens_q=cuq,
            max_seqlen_k=max(bs, local),
            cu_seqlens_k=cuk,
            seqused_k=used,
            softmax_scale=dim**-0.5,
            causal=False,
            return_softmax_lse=True,
            fa_version=version,
        )
        lse = lse.T
    elif backend == "FLASHINFER":
        wrapper, packed = _flashinfer_plan(
            device, n, topk, bs, local, heads, k.shape[1], dim, q.dtype
        )
        remote = (
            torch.arange(bs, device=device)[None, None]
            < used.view(n, groups)[:, :topk, None]
        )
        nearby = (
            torch.arange(local, device=device)[None] < used.view(n, groups)[:, -1, None]
        )
        mask = torch.cat((remote.flatten(1), nearby), -1).reshape(-1, 8)
        bits = torch.arange(8, device=device)
        packed.copy_((mask.to(torch.int32) << bits).sum(-1).to(torch.uint8))
        out, lse = wrapper.run(
            queries, k, v, return_lse=True, lse_base="ln", lse_layout="NH"
        )
    else:
        raise ValueError(f"Unsupported DASH Flash backend: {backend}")
    live = used.view(n, groups, 1) > 0
    prior = torch.cat((bias, torch.zeros_like(bias[:, :1])), -1)
    logits = lse.reshape(n, groups, heads).float() + prior[:, :, None]
    weights = logits.masked_fill(~live, -torch.inf).softmax(1).nan_to_num()
    partial = out.view(n, groups, heads, dim).float().masked_fill(~live[..., None], 0)
    return (partial * weights[..., None]).sum(1).to(q.dtype)
