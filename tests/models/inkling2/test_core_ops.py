# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 core kernels against plain torch references of the same math."""

from collections.abc import Iterator
from typing import Any

import pytest
import torch

from tests.models.inkling2.test_attention import shear_rel_logits
from vllm.models.inkling2.ops.attn_res import attn_res
from vllm.models.inkling2.ops.conv_mix import conv_mix
from vllm.models.inkling2.ops.norm import rms_norm
from vllm.models.inkling2.ops.sconv import (
    attention_prologue,
    shift_rel_proj,
    short_conv,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEVICE = "cuda"


@pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [4608, 128])
def test_rms_norm_rounds_once_to_bf16(in_dtype, width):
    x = torch.randn(37, width, device=DEVICE).to(in_dtype) * 3
    weight = torch.randn(width, device=DEVICE).bfloat16()
    ref = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    ref = ref * weight.float()
    out = rms_norm(x, weight, 1e-6, torch.bfloat16)
    torch.testing.assert_close(out, ref.bfloat16(), atol=0, rtol=1e-2)


def _clip(x, threshold, eps=1e-6):
    norm = x.norm(dim=-1, keepdim=True)
    return x * (threshold / (norm + eps)).clamp(max=1.0)


def _prologue_qr(
    q,
    r,
    q_scale,
    r_scale,
    tau,
    rel_proj,
    kv_heads,
    seqs=None,
    is_local=False,
):
    """Run the attention prologue for q/r only: zero K/V, no KV-cache writes.

    Returns the sheared relative logits of requests `seqs` (`(q_len,
    kv_len)` each). By default the tokens form one request after 2048 cached
    ones, so that every distance is present, and the logits come unsheared.
    """
    tokens, heads, dim = q.shape
    extent = rel_proj.shape[1]
    out_q = q.new_empty(tokens, heads, dim)
    rel = q.new_empty(tokens + 128, heads, extent + 256)
    past = 2048
    q_lens = [tokens] if seqs is None else [q_len for q_len, _ in seqs]
    kv_lens = [tokens + past] if seqs is None else [kv_len for _, kv_len in seqs]
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32, device=DEVICE
    )
    seq_idx = torch.repeat_interleave(
        torch.arange(len(q_lens), device=DEVICE), torch.tensor(q_lens, device=DEVICE)
    ).int()
    kv = q.new_zeros(tokens, kv_heads * dim)
    kv_cache = q.new_zeros(1, 16, kv_heads, 2 * dim)
    cache, table = _conv_cache_and_table(kv_heads, 4, 1024, tokens // 4 + 1)
    idx = torch.arange(tokens, device=DEVICE)
    no_slot = torch.full_like(idx, -1)
    key_cache, value_cache = kv_cache.split(dim, dim=-1)
    attention_prologue(
        q,
        r,
        kv,
        kv,
        q_scale,
        r_scale,
        tau,
        shift_rel_proj(rel_proj, is_local),
        q.new_zeros(kv_heads * dim, 4),
        q.new_zeros(kv_heads * dim, 4),
        None,
        None,
        cache,
        idx,
        table[:1],
        seq_idx,
        no_slot,
        torch.zeros_like(idx).int(),
        key_cache,
        value_cache,
        no_slot,
        out_q,
        rel,
        cu_seqlens_q,
        torch.tensor(kv_lens, dtype=torch.int32, device=DEVICE),
        clip_threshold=22.6,
        rel_threshold=8.0,
        norm_eps=1e-6,
    )
    if seqs is not None:
        return out_q, rel
    tile_end = ((idx // 128 + 1) * 128).clamp(max=tokens)
    diag = past + idx - ((past + tile_end + 127) // 128 - 10) * 128
    cols = diag[:, None] - torch.arange(extent, device=DEVICE)
    rel = rel[:tokens].gather(2, cols[:, None].expand(-1, heads, -1))
    return out_q, rel


@pytest.mark.parametrize("heads,kv_heads", [(12, 2), (6, 1), (48, 8)])
@pytest.mark.parametrize("with_tau", [True, False])
def test_qr_clip_rel_rounds_before_tau(heads, kv_heads, with_tau):
    torch.manual_seed(0)
    tokens, dim, d_rel, extent = 33, 128, 16, 1024
    # q and r are strided column slices of the fused projection.
    proj = torch.randn(tokens, heads * (dim + d_rel) + 64, device=DEVICE) * 4
    proj = proj.bfloat16()
    q = proj[:, : heads * dim].view(tokens, heads, dim)
    r = proj[:, heads * dim : heads * (dim + d_rel)].view(tokens, heads, d_rel)
    q_scale = torch.rand(heads, dim, device=DEVICE).bfloat16() * 2
    r_scale = torch.rand(heads, d_rel, device=DEVICE).bfloat16() * 2
    rel_proj = torch.randn(d_rel, extent, device=DEVICE).bfloat16()
    tau = 1 + torch.rand(tokens, device=DEVICE) if with_tau else None

    def ref_clip(x, scale, threshold):
        y = _clip(x.float() * scale.float(), threshold).bfloat16()
        if tau is None:
            return y
        return (y.float() * tau[:, None, None]).bfloat16()

    ref_q = ref_clip(q, q_scale, 22.6)
    ref_r = ref_clip(r, r_scale, 8.0)
    ref_rel = torch.einsum("thd,de->the", ref_r, rel_proj)
    out_q, rel = _prologue_qr(q, r, q_scale, r_scale, tau, rel_proj, kv_heads)
    torch.testing.assert_close(out_q, ref_q, atol=0, rtol=1e-2)
    # A 1-ulp difference in a clipped r element shifts its 1024 logits by
    # about ulp(r) * |rel_proj|.
    torch.testing.assert_close(rel, ref_rel, atol=0.25, rtol=1e-2)


@pytest.mark.parametrize("is_local", [False, True])
def test_prologue_shears_rel_logits(is_local):
    """The relative logits land in tml_fa4's sheared layout, padding
    included, for decode, verify, chunked and fresh prefill requests."""
    torch.manual_seed(0)
    heads, kv_heads, dim, d_rel, extent = 12, 2, 128, 16, 1024
    seqs = [(1, 3000), (8, 1500), (300, 300), (200, 1700), (130, 1029)]
    tokens = sum(q_len for q_len, _ in seqs)
    q = torch.randn(tokens, heads, dim, device=DEVICE).bfloat16()
    r = (torch.randn(tokens, heads, d_rel, device=DEVICE) * 4).bfloat16()
    q_scale = q.new_ones(heads, dim)
    r_scale = q.new_ones(heads, d_rel)
    rel_proj = torch.randn(d_rel, extent, device=DEVICE).bfloat16()
    args = (q, r, q_scale, r_scale, None, rel_proj, kv_heads)
    _, rel = _prologue_qr(*args, seqs=seqs, is_local=is_local)
    _, compact = _prologue_qr(*args)
    ref = shear_rel_logits(compact, seqs, 128, is_local)
    torch.testing.assert_close(rel[:tokens], ref[:tokens], atol=0, rtol=0)


@pytest.mark.parametrize("num_stack,num_q,with_addend", [(1, 1, False), (5, 3, True)])
@pytest.mark.parametrize("normalize", [True, False])
# Small batches keep rows in registers; large ones loop over D chunks.
@pytest.mark.parametrize("tokens", [29, 300])
def test_attn_res_matches_softmax_mix_then_norm(
    num_stack, num_q, with_addend, normalize, tokens
):
    dim = 4608
    stack = torch.randn(num_stack, tokens, dim, device=DEVICE).bfloat16()
    last = torch.randn(tokens, dim, device=DEVICE).bfloat16()
    addend = torch.randn(tokens, dim, device=DEVICE).bfloat16() if with_addend else None
    queries = torch.randn(num_q, dim, device=DEVICE) * dim**-0.5
    weight = torch.randn(num_q, dim, device=DEVICE).bfloat16()

    values = torch.cat([stack, last[None]]).float()
    if addend is not None:
        values[-1] += addend.float()
    inv_rms = torch.rsqrt(values.square().mean(-1) + 1e-6)
    probs = torch.einsum("ntd,qd,nt->qtn", values, queries, inv_rms).softmax(-1)
    if normalize:
        values = values * inv_rms[..., None]
    mix = torch.einsum("qtn,ntd->qtd", probs, values)
    ref = mix * torch.rsqrt(mix.square().mean(-1, keepdim=True) + 1e-6)
    ref = (ref * weight.float()[:, None]).bfloat16()

    out = attn_res(stack, last, addend, queries, weight, 1e-6, normalize)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-2)


def _dense_conv(x, weight, dyn):
    """Torch reference over whole sequences; x: (L, C)."""
    width = weight.shape[0] if dyn is not None else weight.shape[1]
    padded = torch.cat([x.new_zeros(width - 1, x.shape[1]), x]).float()
    acc = torch.zeros_like(x, dtype=torch.float32)
    for lag in range(width):
        value = padded[width - 1 - lag : width - 1 - lag + x.shape[0]]
        if dyn is None:
            w = weight[:, width - 1 - lag].float()
        else:
            head_dim = x.shape[1] // dyn.shape[1]
            w = weight[lag].float() + dyn[:, :, lag].float().repeat_interleave(
                head_dim, dim=1
            )
        acc += value * w
    scale = (width * (2 if dyn is not None else 1) + 1) ** -0.5
    return ((acc + x.float()) * scale).to(x.dtype)


_ConvMeta = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _chunked_steps(
    lengths_steps, block_table, block
) -> Iterator[tuple[Any, Any, Any, _ConvMeta]]:
    """Yield per-step conv metadata for chunked prefill + decode."""
    done = [0] * len(lengths_steps[0])
    for step in lengths_steps:
        reqs = [i for i in range(len(step)) if step[i]]
        rows = [torch.arange(done[i], done[i] + step[i], device=DEVICE) for i in reqs]
        starts = torch.tensor([0, *[step[i] for i in reqs]]).cumsum(0)
        seq_idx = torch.repeat_interleave(
            torch.arange(len(reqs)), torch.tensor([step[i] for i in reqs])
        )
        slots = torch.cat(
            [
                block_table[i][r // block].long() * block + r % block
                for i, r in zip(reqs, rows)
            ]
        )
        meta = (
            torch.cat(rows),
            block_table[reqs],
            seq_idx.int().to(DEVICE),
            slots,
            starts[seq_idx].int().to(DEVICE),
        )
        yield reqs, rows, starts, meta
        for i in reqs:
            done[i] += step[i]


STEPS = [[3, 7, 2, 9], [2, 6, 0, 0], [1, 1, 1, 1]]
# Steps of over 256 tokens take the prefill tiling.
LONG_STEPS = [[3, 150, 2, 140], [200, 6, 90, 0], [1, 1, 1, 1]]


def _conv_cache_and_table(heads, block, row, blocks_per_req=8):
    num_blocks = 4 * blocks_per_req
    cache = torch.zeros(2 * num_blocks, heads, block, row, device=DEVICE).bfloat16()
    perm = torch.randperm(2 * num_blocks - 1, device=DEVICE)[:num_blocks] + 1
    return cache, perm.view(4, blocks_per_req).int()


@pytest.mark.parametrize("fused_residual", [False, True])
@pytest.mark.parametrize("steps", [STEPS, LONG_STEPS])
def test_short_conv_chunked_matches_full_sequence(fused_residual, steps):
    """Chunked prefill + decode through the paged conv state equals one pass;
    the residual update `(res + addend) + y` is fused in fp32."""
    width, block, heads, stream_width, offset, row = 4, 4, 2, 128, 128, 1024
    channels = heads * stream_width
    totals = [sum(s[i] for s in steps) for i in range(4)]
    xs = [torch.randn(n, channels, device=DEVICE).bfloat16() for n in totals]
    adds = [torch.randn(n, channels, device=DEVICE).bfloat16() for n in totals]
    ress = [torch.randn(n, channels, device=DEVICE).bfloat16() for n in totals]
    weight = torch.randn(channels, width, device=DEVICE).bfloat16()
    cache, block_table = _conv_cache_and_table(heads, block, row, 64)

    outs: list[list[torch.Tensor]] = [[] for _ in totals]
    for reqs, rows, starts, meta in _chunked_steps(steps, block_table, block):
        x = torch.cat([xs[i][r] for i, r in zip(reqs, rows)])
        kwargs: dict[str, torch.Tensor] = {}
        if fused_residual:
            res = torch.cat([ress[i][r] for i, r in zip(reqs, rows)])
            add = torch.cat([adds[i][r] for i, r in zip(reqs, rows)])
            kwargs = dict(addend=add, residual=res, out=res)
        out = short_conv(x, weight, cache, *meta, offset, stream_width, **kwargs)
        for j, i in enumerate(reqs):
            outs[i].append(out[starts[j] : starts[j + 1]])
    for i in range(4):
        ref = _dense_conv(xs[i], weight, None)
        if fused_residual:
            ref = (ress[i].float() + adds[i].float() + ref.float()).bfloat16()
        torch.testing.assert_close(torch.cat(outs[i]), ref, atol=0, rtol=1e-2)


@pytest.mark.parametrize(
    "num_stack,num_q,with_last,with_residual",
    [
        (1, 1, True, False),
        (2, 1, False, False),
        (3, 1, False, True),
        (8, 3, False, True),
    ],
)
def test_conv_mix_matches_conv_then_attn_res(
    num_stack, num_q, with_last, with_residual
):
    """The fused prefill conv + mix kernel equals short_conv then attn_res over
    chunked steps (bitwise conv and cache; the mix up to its fp32 summation
    order)."""
    block, heads, stream_width, offset, row = 4, 2, 2304, 256, 2560
    dim = heads * stream_width
    weight = torch.randn(dim, 4, device=DEVICE).bfloat16()
    queries = torch.randn(num_q, dim, device=DEVICE) * dim**-0.5
    norm_w = torch.randn(num_q, dim, device=DEVICE).bfloat16()
    cache, block_table = _conv_cache_and_table(heads, block, row, 64)
    ref_cache = cache.clone()
    for _, _, _, meta in _chunked_steps(LONG_STEPS, block_table, block):
        tokens = meta[0].shape[0]
        x = torch.randn(tokens, dim, device=DEVICE).bfloat16()
        stack = torch.randn(num_stack, tokens, dim, device=DEVICE).bfloat16()
        last = torch.randn(tokens, dim, device=DEVICE).bfloat16()
        addend = torch.randn(tokens, dim, device=DEVICE).bfloat16()
        residual = torch.randn(tokens, dim, device=DEVICE).bfloat16()
        ref_conv = residual.clone() if with_residual else None
        ref_conv = short_conv(
            x,
            weight,
            ref_cache,
            *meta,
            offset,
            stream_width,
            addend if with_residual else None,
            ref_conv,
            ref_conv,
        )
        if with_last:
            ref = attn_res(stack, last, ref_conv, queries, norm_w, 1e-6, True)
        else:
            ref = attn_res(stack, ref_conv, None, queries, norm_w, 1e-6, True)
        conv = residual.clone() if with_residual else torch.empty_like(x)
        out = conv_mix(
            x,
            weight,
            cache,
            *meta,
            offset,
            stream_width,
            conv,
            conv if with_residual else None,
            addend if with_residual else None,
            stack,
            last if with_last else None,
            queries,
            norm_w,
            1e-6,
        )
        torch.testing.assert_close(conv, ref_conv, atol=0, rtol=0)
        torch.testing.assert_close(cache, ref_cache, atol=0, rtol=0)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dynamic", [False, True])
def test_attention_prologue_writes_clipped_k_and_normed_v(dynamic):
    """K/V convs over chunked steps, then K L2 clipping and V RMSNorm, land in
    the paged KV cache at each token's slot; padded tokens (slot -1, as in
    CUDA graph padding) write nothing."""
    width, block, heads, dim, kv_block = 4, 4, 2, 128, 16
    channels = heads * dim
    totals = [sum(s[i] for s in STEPS) for i in range(4)]
    ks, vs = (
        [torch.randn(n, channels, device=DEVICE).bfloat16() * 3 for n in totals]
        for _ in range(2)
    )
    dks, dvs = (
        [torch.randn(n, heads * width, device=DEVICE).bfloat16() for n in totals]
        for _ in range(2)
    )
    shape = (width, channels) if dynamic else (channels, width)
    wk, wv = (torch.randn(shape, device=DEVICE).bfloat16() for _ in range(2))
    cache, block_table = _conv_cache_and_table(heads, block, 1024)
    kv = torch.zeros(32, heads, kv_block, 2 * dim, device=DEVICE).bfloat16()
    key_cache, value_cache = kv.transpose(1, 2).split(dim, dim=-1)
    perm = torch.randperm(32 * kv_block, device=DEVICE)[: sum(totals)]
    kv_slots = list(perm.split(totals))

    for reqs, rows, starts, meta in _chunked_steps(STEPS, block_table, block):

        def cat(ts, reqs=reqs, rows=rows, pad=None):
            x = torch.cat([ts[i][r] for i, r in zip(reqs, rows)])
            pad = torch.full_like(x[:2], -1) if pad is None else pad(x[:2])
            return torch.cat([x, pad])

        positions, table, seq_idx, slots, qstart = meta
        num_tokens = positions.shape[0] + 2
        q = torch.zeros(num_tokens, 6, dim, device=DEVICE).bfloat16()
        r = q.new_zeros(num_tokens, 6, 16)
        attention_prologue(
            q,
            r,
            cat(ks, pad=torch.randn_like),
            cat(vs, pad=torch.randn_like),
            q.new_ones(6, dim),
            q.new_ones(6, 16),
            None,
            shift_rel_proj(q.new_zeros(16, 1024), True),
            wk,
            wv,
            cat(dks, pad=torch.randn_like) if dynamic else None,
            cat(dvs, pad=torch.randn_like) if dynamic else None,
            cache,
            torch.cat([positions, positions[:2]]),
            table,
            torch.cat([seq_idx, seq_idx[:2]]),
            torch.cat([slots, torch.full_like(slots[:2], -1)]),
            torch.cat([qstart, qstart[:2]]),
            key_cache,
            value_cache,
            cat(kv_slots),
            torch.empty_like(q),
            q.new_empty(num_tokens + 128, 6, 1024 + 256),
            starts.int().to(DEVICE),
            starts.diff().int().to(DEVICE) + 4096,
            clip_threshold=22.6,
            rel_threshold=8.0,
            norm_eps=1e-6,
        )
    for i in range(4):
        dyn = (
            (dks[i].view(-1, heads, width), dvs[i].view(-1, heads, width))
            if dynamic
            else (None, None)
        )
        k = _dense_conv(ks[i], wk, dyn[0]).float().view(-1, heads, dim)
        v = _dense_conv(vs[i], wv, dyn[1]).float().view(-1, heads, dim)
        ref_k = _clip(k, 22.6).bfloat16()
        ref_v = (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
        slot = kv_slots[i]
        got_k = key_cache[slot // kv_block, slot % kv_block]
        got_v = value_cache[slot // kv_block, slot % kv_block]
        torch.testing.assert_close(got_k, ref_k, atol=0, rtol=1e-2)
        torch.testing.assert_close(got_v, ref_v, atol=0, rtol=1e-2)
    num_tokens = sum(totals)
    assert (kv != 0).any(-1).sum() == num_tokens * heads
    assert (cache[..., : 2 * dim] != 0).any(-1).sum() == num_tokens * heads
