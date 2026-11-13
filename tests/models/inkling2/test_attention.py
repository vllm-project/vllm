# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the Inkling2 attention ops against fp32 references.

The attention reference uses scale-after-add relative bias::

    score(i, j, h) = (q[i, h] . k[j, h] + rel_bias(i, j, h) / scale) * scale
    rel_bias(i, j, h) = rel_logits[i, h, i - j] if 0 <= i - j < rel_extent else 0
"""

import pytest
import torch

from vllm.models.inkling.nvidia.ops.fa4_rel_attention import bucket_max_seqlen_q
from vllm.platforms import current_platform

_cap = current_platform.get_device_capability() if current_platform.is_cuda() else None
requires_sm100 = pytest.mark.skipif(
    _cap is None or _cap.major not in (10, 11),
    reason="tml_fa4 sheared relative bias requires SM100/SM110",
)

HEAD_DIM = 128
REL_EXTENT = 1024
SCENARIOS = {
    "prefill": [(1500, 1500), (37, 37), (300, 300)],
    "chunked_prefill": [(200, 1700), (64, 64), (513, 2100)],
    "decode": [(1, 1), (1, 900), (1, 3000), (1, 1025)],
}


def shear_rel_logits(rel_logits, seqs, tile, is_local):
    """Torch reference of the sheared layout `attention_prologue` writes.

    Token `i` of a request with `past` cached keys sits in a tile of
    `tile` query tokens whose key blocks end at `n_max`; column `c` holds
    key `c + (n_max - 10) * 128`, padded with -inf right of the diagonal and
    -inf (window) or 0 past the extent (also for keys before the sequence,
    which no kernel reads).
    """
    tokens, heads, extent = rel_logits.shape
    out = rel_logits.new_full((tokens + 128, heads, extent + 256), float("nan"))
    cols = torch.arange(extent + 256, device=rel_logits.device)
    start = 0
    for q_len, kv_len in seqs:
        idx = torch.arange(q_len, device=rel_logits.device)
        pos = kv_len - q_len + idx
        tile_end = ((idx // tile + 1) * tile).clamp(max=q_len)
        n_max = (kv_len - q_len + tile_end + 127) // 128
        dist = pos[:, None] - (cols[None] + (n_max[:, None] - 10) * 128)
        valid = (dist >= 0) & (dist < extent)
        rel = rel_logits[start : start + q_len]
        vals = rel.gather(2, dist.clamp(0, extent - 1)[:, None].expand(-1, heads, -1))
        pad = torch.where((dist < 0) | is_local, float("-inf"), 0.0).to(
            rel_logits.dtype
        )
        out[start : start + q_len] = torch.where(valid[:, None], vals, pad[:, None])
        start += q_len
    return out


def _ref_attention(
    q, key_cache, value_cache, rel_logits, seqs, block_table, scale, window_left
):
    num_heads = q.shape[1]
    num_kv_heads = key_cache.shape[2]
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    start = 0
    for i, (q_len, kv_len) in enumerate(seqs):
        blocks = block_table[i].long()
        k, v = (
            cache[blocks]
            .flatten(0, 1)[:kv_len]
            .float()
            .repeat_interleave(num_heads // num_kv_heads, dim=1)
            for cache in (key_cache, value_cache)
        )
        qpos = torch.arange(kv_len - q_len, kv_len, device=q.device)
        dist = qpos[:, None] - torch.arange(kv_len, device=q.device)[None]
        rel = rel_logits[start : start + q_len].float().transpose(0, 1)
        idx = dist.clamp(0, REL_EXTENT - 1).expand(num_heads, -1, -1)
        bias = rel.gather(2, idx).masked_fill(~((dist >= 0) & (dist < REL_EXTENT)), 0)
        qk = torch.einsum("qhd,khd->hqk", q[start : start + q_len].float(), k)
        scores = (qk + bias / scale) * scale
        mask = dist < 0
        if window_left is not None:
            mask |= dist > window_left
        scores.masked_fill_(mask, float("-inf"))
        lse[start : start + q_len] = scores.logsumexp(-1).T
        out[start : start + q_len] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)
        start += q_len
    return out, lse


@requires_sm100
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,block_size,window_left,scenario",
    [
        (12, 2, 16, None, "prefill"),
        (48, 8, 16, None, "prefill"),
        (12, 2, 128, REL_EXTENT - 1, "prefill"),
        (12, 2, 128, None, "chunked_prefill"),
        (48, 8, 16, REL_EXTENT - 1, "chunked_prefill"),
        (48, 8, 128, None, "chunked_prefill"),
        (12, 2, 16, None, "decode"),
        (48, 8, 128, REL_EXTENT - 1, "decode"),
    ],
)
@pytest.mark.parametrize(
    "num_splits,use_scheduler", [(1, True), (8, True), (8, False), (32, False)]
)
@torch.inference_mode()
def test_inkling2_rel_attention(
    num_heads,
    num_kv_heads,
    block_size,
    window_left,
    scenario,
    num_splits,
    use_scheduler,
):
    from vllm.models.inkling2.ops.attention import _INKLING2_FA4_REL_ATTENTION_KERNEL

    torch.manual_seed(0)
    device = "cuda"
    seqs = SCENARIOS[scenario]
    q_lens = [q_len for q_len, _ in seqs]
    num_tokens = sum(q_lens)
    max_blocks = -(-max(kv_len for _, kv_len in seqs) // block_size)
    num_blocks = len(seqs) * max_blocks + 1

    # Packed [blocks, kv_heads, block_size, 2 * D] cache split like
    # InklingAttention._split_kv_cache; q/k are L2-normalized as in the model.
    kv = torch.randn(num_blocks, block_size, num_kv_heads, 2 * HEAD_DIM, device=device)
    kv[..., :HEAD_DIM] = torch.nn.functional.normalize(kv[..., :HEAD_DIM], dim=-1) * 2
    kv = kv.bfloat16().transpose(1, 2)
    key_cache, value_cache = kv.transpose(1, 2).split(HEAD_DIM, dim=-1)
    block_table = (
        (torch.randperm(num_blocks - 1, device=device) + 1)[: len(seqs) * max_blocks]
        .view(len(seqs), max_blocks)
        .int()
    )
    q = torch.randn(num_tokens, num_heads, HEAD_DIM, device=device)
    q = (torch.nn.functional.normalize(q, dim=-1) * 3).bfloat16()
    rel_logits = torch.randn(
        num_tokens, num_heads, REL_EXTENT, device=device, dtype=torch.bfloat16
    )
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    seqused_k = torch.tensor(
        [kv_len for _, kv_len in seqs], dtype=torch.int32, device=device
    )
    scale = HEAD_DIM**-0.5
    rel_in = shear_rel_logits(rel_logits, seqs, 128, window_left is not None)
    out = torch.empty(q.shape, device=device)
    lse = torch.empty(q.shape[1::-1], device=device)

    _INKLING2_FA4_REL_ATTENTION_KERNEL(
        q,
        key_cache,
        value_cache,
        block_table=block_table,
        cache_seqlens=seqused_k,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=bucket_max_seqlen_q(max(q_lens)),
        softmax_scale=scale,
        window_size=(-1, -1) if window_left is None else (window_left, 0),
        sheared_bias=rel_in,
        num_splits=num_splits,
        use_scheduler=use_scheduler,
        out=out,
        lse=lse,
    )
    lse = lse.t()

    ref_out, ref_lse = _ref_attention(
        q, key_cache, value_cache, rel_logits, seqs, block_table, scale, window_left
    )
    assert lse.shape == (num_tokens, num_heads) and lse.dtype == torch.float32
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-4, rtol=1e-5)


@requires_sm100
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,window_left",
    [(12, 2, None), (6, 1, REL_EXTENT - 1), (12, 2, REL_EXTENT - 1)],
)
@pytest.mark.parametrize(
    "query_lens,kv_lens",
    [
        ([1], [5000]),
        ([1] * 4, [1, 200, 1025, 3000]),
        ([2, 1, 2], [2, 1500, 2100]),
        ([8, 8, 3], [8, 1500, 2100]),
    ],
)
@pytest.mark.parametrize(
    "block_size,num_splits", [(128, 1), (128, 9), (128, 64), (16, 9)]
)
@torch.inference_mode()
def test_inkling2_rel_decode_attention(
    num_heads, num_kv_heads, window_left, query_lens, kv_lens, block_size, num_splits
):
    """Decode and verify on the Gluon kernel match the reference, gate and
    sink included, with and without split-KV."""
    from vllm.models.inkling2.ops.attention import rel_decode_attention

    torch.manual_seed(0)
    device = "cuda"
    seqs = list(zip(query_lens, kv_lens))
    num_tokens = sum(query_lens)
    max_blocks = -(-max(kv_lens) // block_size)
    num_blocks = len(seqs) * max_blocks + 1
    # The model's [blocks, block_size, kv_heads, (K | V)] cache layout.
    kv = torch.randn(num_blocks, block_size, num_kv_heads, 2 * HEAD_DIM, device=device)
    kv[..., :HEAD_DIM] = torch.nn.functional.normalize(kv[..., :HEAD_DIM], dim=-1) * 2
    key_cache, value_cache = kv.bfloat16().split(HEAD_DIM, dim=-1)
    block_table = (
        (torch.randperm(num_blocks - 1, device=device) + 1)
        .view(len(seqs), max_blocks)
        .int()
    )
    q = torch.randn(num_tokens, num_heads, HEAD_DIM, device=device)
    q = (torch.nn.functional.normalize(q, dim=-1) * 3).bfloat16()
    rel_logits = torch.randn(
        num_tokens, num_heads, REL_EXTENT, device=device, dtype=torch.bfloat16
    )
    width = num_heads * HEAD_DIM
    proj = torch.randn(num_tokens + 1, width + num_heads + 4, device=device).bfloat16()
    gate, sinks = proj[:, :width], proj[:, width : width + num_heads]
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    scale = HEAD_DIM**-0.5
    # One padding token, which must come out zero.
    q_in = torch.cat([q, q[:1]])
    rel_in = shear_rel_logits(rel_logits, seqs, 128, window_left is not None)
    out = torch.full_like(gate, float("nan"))
    rel_decode_attention(
        q_in,
        key_cache,
        value_cache,
        rel_in,
        gate,
        sinks,
        out,
        block_table=block_table,
        cache_seqlens=seqused_k,
        cu_seqlens_q=cu_seqlens_q,
        max_query_len=max(query_lens),
        num_actual_tokens=num_tokens,
        softmax_scale=scale,
        is_local=window_left is not None,
        num_splits=num_splits,
    )

    ref_out, ref_lse = _ref_attention(
        q, key_cache, value_cache, rel_logits, seqs, block_table, scale, window_left
    )
    gate, sinks = gate[:num_tokens].float(), sinks[:num_tokens].float()
    expected = ref_out.view(num_tokens, -1) * torch.sigmoid(gate)
    expected *= torch.sigmoid(ref_lse - sinks).repeat_interleave(HEAD_DIM, dim=1)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out[:num_tokens].float(), expected, atol=1e-2, rtol=1e-2)
    assert (out[num_tokens:] == 0).all()


@requires_sm100
@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("num_splits", [1, 9])
@torch.inference_mode()
def test_inkling2_decode_attention_after_prologue(is_local, num_splits):
    """Back to back under PDL (CUDA graph replay), the decode kernel loads
    cached keys while the prologue still writes the new ones: it must match a
    run with the two launches serialized."""
    from vllm.models.inkling2.ops.attention import rel_decode_attention
    from vllm.models.inkling2.ops.sconv import attention_prologue, shift_rel_proj

    torch.manual_seed(0)
    device = "cuda"
    heads, kv_heads, d_rel, page = 12, 2, 16, 128
    q_lens, kv_lens = [8, 1], [3000, 300]
    num_reqs, num_tokens = len(q_lens), sum(q_lens)
    max_blocks = -(-max(kv_lens) // page)
    req = torch.arange(num_reqs, device=device).repeat_interleave(
        torch.tensor(q_lens, device=device)
    )
    cu_seqlens_q = torch.tensor([0, *q_lens], device=device).cumsum(0).int()
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    pos = seq_lens[req] - cu_seqlens_q[1:][req] + torch.arange(num_tokens).to(device)
    kv = torch.randn(num_reqs * max_blocks + 1, page, kv_heads, 2 * HEAD_DIM)
    kv = kv.to(device, torch.bfloat16)
    key_cache, value_cache = kv.split(HEAD_DIM, dim=-1)
    block_table = (torch.randperm(kv.shape[0] - 1, device=device) + 1).int()
    block_table = block_table.view(num_reqs, max_blocks)
    kv_slots = block_table[req, pos // page].long() * page + pos % page
    # Paged short-conv state: 4 positions per block, distinct blocks per request.
    conv_blocks = max(kv_lens) // 4 + 1
    conv_cache = torch.randn(num_reqs * conv_blocks + 1, kv_heads, 4, 1024)
    conv_cache = conv_cache.to(device, torch.bfloat16)
    conv_table = torch.arange(1, num_reqs * conv_blocks + 1, device=device).int()
    conv_table = conv_table.view(num_reqs, conv_blocks)
    conv_slots = conv_table[req, pos // 4].long() * 4 + pos % 4
    width = heads * (2 * HEAD_DIM + d_rel + 1)
    proj = torch.randn(num_tokens, width, device=device).bfloat16()
    q = proj[:, : heads * HEAD_DIM].view(num_tokens, heads, HEAD_DIM)
    r = proj[:, heads * HEAD_DIM : heads * (HEAD_DIM + d_rel)]
    gate = proj[:, heads * (HEAD_DIM + d_rel) : -heads]
    k, v = gate[:, : kv_heads * HEAD_DIM], gate[:, -kv_heads * HEAD_DIM :]
    sinks = proj[:, -heads:]
    conv_weight = torch.randn(kv_heads * HEAD_DIM, 4, device=device).bfloat16()
    rel_proj = torch.randn(d_rel, REL_EXTENT, device=device).bfloat16()
    rel_proj = shift_rel_proj(rel_proj, is_local)
    clipped_q = torch.empty_like(q)
    rel = q.new_empty(num_tokens + 128, heads, REL_EXTENT + 256)
    out = torch.empty(gate.shape, device=device, dtype=torch.bfloat16)
    kv_init, conv_init = kv.clone(), conv_cache.clone()

    def run():
        attention_prologue(
            q,
            r.view(num_tokens, heads, d_rel),
            k,
            v,
            q.new_ones(heads, HEAD_DIM),
            q.new_ones(heads, d_rel),
            None,
            rel_proj,
            conv_weight,
            conv_weight,
            None,
            None,
            conv_cache,
            pos,
            conv_table,
            req.int(),
            conv_slots,
            cu_seqlens_q[req],
            key_cache,
            value_cache,
            kv_slots,
            clipped_q,
            rel,
            cu_seqlens_q,
            seq_lens,
            clip_threshold=22.6,
            rel_threshold=8.0,
            norm_eps=1e-6,
        )
        if sync:
            torch.accelerator.synchronize()
        rel_decode_attention(
            clipped_q,
            key_cache,
            value_cache,
            rel,
            gate,
            sinks,
            out,
            block_table=block_table,
            cache_seqlens=seq_lens,
            cu_seqlens_q=cu_seqlens_q,
            max_query_len=max(q_lens),
            num_actual_tokens=num_tokens,
            softmax_scale=HEAD_DIM**-0.5,
            is_local=is_local,
            num_splits=num_splits,
        )

    sync = True
    run()
    expected = out.clone()
    sync = False
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        kv.copy_(kv_init)
        conv_cache.copy_(conv_init)
        out.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 300])
def test_gate_sink(num_tokens):
    from vllm.models.inkling2.ops.attention import gate_sink

    torch.manual_seed(0)
    num_heads, head_dim = 12, 128
    width = num_heads * head_dim
    output = torch.randn(num_tokens, num_heads, head_dim, device="cuda")
    # gate and sinks are strided column slices of a fused projection.
    proj = torch.randn(num_tokens, width + num_heads + 4, device="cuda").bfloat16()
    gate, sinks = proj[:, :width], proj[:, width : width + num_heads]
    lse = (torch.randn(num_heads, num_tokens, device="cuda") * 5).t()

    actual = torch.empty(gate.shape, device=gate.device, dtype=torch.bfloat16)
    gate_sink(output, gate, lse, sinks, actual)

    scale = torch.sigmoid(lse - sinks.float()).repeat_interleave(head_dim, dim=1)
    expected = output.view(num_tokens, -1) * torch.sigmoid(gate.float()) * scale
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=1e-2)
