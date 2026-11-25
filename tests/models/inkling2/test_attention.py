# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the Inkling2 attention ops against fp32 references.

The attention reference uses scale-after-add relative bias::

    score(i, j, h) = (q[i, h] . k[j, h] + rel_bias(i, j, h) / scale) * scale
    rel_bias(i, j, h) = rel_logits[i, h, i - j] if 0 <= i - j < rel_extent else 0
"""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from vllm.models.inkling.nvidia.ops.fa4_rel_attention import bucket_max_seqlen_q
from vllm.platforms import current_platform

_cap = current_platform.get_device_capability() if current_platform.is_cuda() else None
_sm90 = _cap is not None and _cap.major in (9, 10, 11, 12)
requires_sm90 = pytest.mark.skipif(
    not _sm90, reason="Inkling2 attention requires Hopper or Blackwell"
)
_is_rocm = current_platform.is_rocm()
_gfx950 = _is_rocm and (
    current_platform.is_device_capability(94)
    or current_platform.is_device_capability(95)
)
requires_rocm = pytest.mark.skipif(
    not _gfx950, reason="ROCm Triton attention requires gfx942 or gfx950"
)
requires_rel_attention = pytest.mark.skipif(
    not (_sm90 or _gfx950),
    reason="Inkling2 attention requires Hopper, Blackwell, gfx942 or gfx950",
)

HEAD_DIM = 128
REL_EXTENT = 1024
SCENARIOS = {
    "prefill": [(1500, 1500), (37, 37), (300, 300)],
    "chunked_prefill": [(200, 1700), (64, 64), (513, 2100)],
    "decode": [(1, 1), (1, 900), (1, 3000), (1, 1025)],
    # Requests without queries or keys between others.
    "empty_requests": [(1, 900), (0, 0), (1, 3000), (0, 0)],
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


@requires_sm90
def test_e4m3_kv_load_preserves_all_byte_values_and_masks():
    from vllm.models.inkling2.nvidia.ops.attention_sm90 import _load_fp8_kv
    from vllm.triton_utils import tl, triton

    @triton.jit
    def load_values(source, output):
        rows = tl.arange(0, 2)
        values = _load_fp8_kv(source, rows * 512, rows == 0, tl.bfloat16, 256)
        indices = rows[:, None] * 256 + tl.arange(0, 256)[None, :]
        tl.store(output + indices, values)

    source = torch.arange(256, device="cuda").to(torch.uint8).view(torch.float8_e4m3fn)
    output = torch.empty(512, device="cuda", dtype=torch.bfloat16)
    load_values[(1,)](source, output)
    expected = torch.cat((source.bfloat16(), torch.zeros_like(output[256:])))
    nan = expected.isnan()
    assert torch.equal(output.isnan(), nan)
    # Compare bits as well as values, including the sign of zero.
    assert torch.equal(output[~nan].view(torch.int16), expected[~nan].view(torch.int16))


@requires_sm90
@pytest.mark.parametrize("cache_dtype", [torch.float16, torch.float8_e4m3fn])
@pytest.mark.parametrize("query_len,splits", [(37, 1), (1, 1), (1, 3), (8, 1), (8, 3)])
@pytest.mark.parametrize("window", [0, 1024])
@torch.inference_mode()
def test_portable_attention_dequantizes_kv_before_mma(
    cache_dtype, query_len, splits, window
):
    """Stored FP16/FP8 values match explicit BF16 dequantization in all phases."""
    from vllm.models.inkling2.nvidia.ops.attention_sm90 import (
        MODE_OUT_LSE,
        MODE_PARTIAL,
        rel_attention,
    )

    torch.manual_seed(42)
    heads, kv_heads, page, dim = 24, 8, 16, 128
    q = torch.randn(query_len, heads, dim, device="cuda").bfloat16()
    kv = torch.randn(80, page, kv_heads, 2 * dim, device="cuda").to(cache_dtype)
    table = torch.randperm(80, device="cuda").view(1, -1).int()
    lengths = torch.tensor([1200], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, query_len], device="cuda", dtype=torch.int32)
    rel = torch.randn(query_len, heads, 1024, device="cuda").bfloat16()
    sheared = shear_rel_logits(rel, [(query_len, 1200)], 128, bool(window))

    def run(cache):
        if splits == 1:
            out = torch.empty_like(q, dtype=torch.float32)
            lse = torch.empty(heads, query_len, device="cuda")
        else:
            out = torch.empty(splits, *q.shape, device="cuda")
            lse = torch.empty(splits, query_len, heads, device="cuda")
        k, v = cache.split(dim, dim=-1)
        rel_attention(
            q,
            k,
            v,
            out,
            lse,
            sheared,
            block_table=table,
            cache_seqlens=lengths,
            cu_seqlens_q=cu,
            max_query_len=query_len,
            softmax_scale=dim**-0.5,
            window=window,
            num_splits=splits,
            mode=MODE_OUT_LSE if splits == 1 else MODE_PARTIAL,
        )
        return out, lse

    actual = run(kv)
    expected = run(kv.bfloat16())
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed, reference, atol=0, rtol=0)


@requires_rel_attention
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,block_size,window_left,scenario",
    [
        (24, 8, 16, None, "prefill"),
        (24, 8, 128, REL_EXTENT - 1, "chunked_prefill"),
        (12, 2, 16, None, "prefill"),
        (48, 8, 16, None, "prefill"),
        (12, 2, 128, REL_EXTENT - 1, "prefill"),
        (12, 2, 128, None, "chunked_prefill"),
        (48, 8, 16, REL_EXTENT - 1, "chunked_prefill"),
        (48, 8, 128, None, "chunked_prefill"),
        (12, 2, 16, None, "decode"),
        (48, 8, 128, REL_EXTENT - 1, "decode"),
        pytest.param(48, 8, 16, None, "empty_requests", marks=requires_rocm),
        pytest.param(48, 8, 16, REL_EXTENT - 1, "empty_requests", marks=requires_rocm),
    ],
)
# FA4's KV splits and split scheduler: ROCm's Triton kernel takes neither.
@pytest.mark.parametrize(
    "num_splits,use_scheduler",
    [(1, False)] if _is_rocm else [(1, True), (8, True), (8, False), (32, False)],
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
    from vllm.models.inkling2.nvidia.ops import use_sm100_kernels

    sm100 = use_sm100_kernels()
    if not (_is_rocm or sm100) and (num_splits, use_scheduler) != (1, True):
        pytest.skip("the Hopper kernel does not split prefill")
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

    if _is_rocm:
        from vllm.models.inkling2.amd.ops.attention import (
            rel_attention as amd_rel_attention,
        )

        amd_rel_attention(
            q,
            key_cache,
            value_cache,
            block_table=block_table,
            cache_seqlens=seqused_k,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max(q_lens),
            softmax_scale=scale,
            window=-1 if window_left is None else window_left,
            sheared_bias=rel_in,
            out=out,
            lse=lse,
        )
    elif sm100:
        from vllm.models.inkling2.nvidia.ops.attention import (
            _INKLING2_FA4_REL_ATTENTION_KERNEL,
        )

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
    else:
        from vllm.models.inkling2.nvidia.ops.attention_sm90 import (
            MODE_OUT_LSE,
            rel_attention,
        )

        rel_attention(
            q,
            key_cache,
            value_cache,
            out,
            lse,
            rel_in,
            block_table=block_table,
            cache_seqlens=seqused_k,
            cu_seqlens_q=cu_seqlens_q,
            max_query_len=max(q_lens),
            softmax_scale=scale,
            window=0 if window_left is None else window_left + 1,
            num_splits=1,
            mode=MODE_OUT_LSE,
        )
    lse = lse.t()

    ref_out, ref_lse = _ref_attention(
        q, key_cache, value_cache, rel_logits, seqs, block_table, scale, window_left
    )
    assert lse.shape == (num_tokens, num_heads) and lse.dtype == torch.float32
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-4, rtol=1e-5)


@requires_rel_attention
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,window_left",
    [
        (24, 8, None),
        (24, 8, REL_EXTENT - 1),
        (12, 2, None),
        (6, 1, REL_EXTENT - 1),
        (12, 2, REL_EXTENT - 1),
    ],
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
    """Decode and verify on the Gluon kernel (ROCm: the Triton kernel) match
    the reference, gate and sink included, with and without split-KV."""
    from vllm.models.inkling2 import rel_decode_attention

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


@requires_rel_attention
@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("num_splits", [1, 9])
@torch.inference_mode()
def test_inkling2_decode_attention_after_prologue(is_local, num_splits):
    """Back to back under PDL (CUDA graph replay), the decode kernel loads
    cached keys while the prologue still writes the new ones: it must match a
    run with the two launches serialized. (ROCm has no PDL: this checks the
    graph replay.)"""
    from vllm.models.inkling2 import (
        attention_prologue,
        rel_decode_attention,
        shift_rel_proj,
    )

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


@pytest.mark.skipif(
    not _gfx950,
    reason="ROCm's split merge reads the token count at replay (CUDA zeroes "
    "padding rows by the captured count)",
)
@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("num_splits", [1, 9])
@torch.inference_mode()
def test_inkling2_decode_graph_replays_smaller_batch(is_local, num_splits):
    """A full CUDA graph captured on a decode batch replays a smaller one as
    the runner pads it: the requests after the last real one have no queries,
    and the tokens after the last real one (NaN here) come out zero."""
    from vllm.models.inkling2 import rel_decode_attention

    torch.manual_seed(0)
    device = "cuda"
    heads, kv_heads, page, max_reqs, max_tokens = 12, 2, 16, 4, 8
    max_blocks = -(-3000 // page)
    kv = torch.randn(max_reqs * max_blocks + 1, page, kv_heads, 2 * HEAD_DIM)
    key_cache, value_cache = kv.to(device, torch.bfloat16).split(HEAD_DIM, dim=-1)
    block_table = (torch.randperm(kv.shape[0] - 1, device=device) + 1).int()
    block_table = block_table.view(max_reqs, max_blocks)
    width = heads * HEAD_DIM
    q = torch.empty(max_tokens, heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
    rel = q.new_empty(max_tokens + 128, heads, REL_EXTENT + 256)
    proj = q.new_empty(max_tokens, width + heads)
    gate, sinks = proj[:, :width], proj[:, width:]
    out = q.new_empty(max_tokens, width)
    cu_seqlens_q = torch.empty(max_reqs + 1, dtype=torch.int32, device=device)
    seq_lens = torch.empty(max_reqs, dtype=torch.int32, device=device)

    def run():
        rel_decode_attention(
            q,
            key_cache,
            value_cache,
            rel,
            gate,
            sinks,
            out,
            block_table=block_table,
            cache_seqlens=seq_lens,
            cu_seqlens_q=cu_seqlens_q,
            max_query_len=2,
            num_actual_tokens=max_tokens,
            softmax_scale=HEAD_DIM**-0.5,
            is_local=is_local,
            num_splits=num_splits,
        )

    def load(seqs, stale_seq_len):
        """Fill the buffers with a batch, padded as the runner pads it."""
        num_tokens = sum(q_len for q_len, _ in seqs)
        q.fill_(float("nan"))
        proj.fill_(float("nan"))
        rel.fill_(float("nan"))
        q[:num_tokens] = torch.randn(num_tokens, heads, HEAD_DIM)
        proj[:num_tokens] = torch.randn(num_tokens, width + heads)
        rel_logits = torch.randn(num_tokens, heads, REL_EXTENT).to(rel)
        rel[: num_tokens + 128] = shear_rel_logits(rel_logits, seqs, 128, is_local)
        starts = torch.tensor([0, *[q_len for q_len, _ in seqs]]).cumsum(0)
        cu_seqlens_q.fill_(num_tokens)
        cu_seqlens_q[: len(starts)] = starts
        seq_lens.fill_(stale_seq_len)
        seq_lens[: len(seqs)] = torch.tensor([kv_len for _, kv_len in seqs])
        out.fill_(float("nan"))
        return num_tokens

    load([(2, 3000), (2, 900), (2, 1025), (2, 2)], 0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # Padding requests keep the previous batch's KV lengths.
    for seqs, stale in (([(2, 2100), (1, 300)], 1025), ([(1, 1)], 2100)):
        num_tokens = load(seqs, stale)
        run()
        expected = out.clone()
        out.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        assert not out[:num_tokens].isnan().any()
        assert (out[num_tokens:] == 0).all()


@pytest.mark.skipif(
    not (current_platform.is_cuda() or _gfx950),
    reason="requires CUDA, gfx942 or gfx950",
)
@pytest.mark.parametrize("num_tokens", [1, 300])
def test_gate_sink(num_tokens):
    from vllm.models.inkling2 import gate_sink

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


def _paged_batch(seqs, num_heads, num_kv_heads, block_size, num_padding=0):
    """Random inputs of `(query_len, kv_len)` requests in the model's layouts,
    followed by `num_padding` padding tokens (NaN)."""
    device = "cuda"
    num_tokens = sum(q_len for q_len, _ in seqs)
    max_blocks = -(-max(kv_len for _, kv_len in seqs) // block_size)
    num_blocks = len(seqs) * max_blocks + 1
    kv = torch.randn(num_blocks, block_size, num_kv_heads, 2 * HEAD_DIM, device=device)
    kv[..., :HEAD_DIM] = torch.nn.functional.normalize(kv[..., :HEAD_DIM], dim=-1) * 2
    key_cache, value_cache = kv.bfloat16().split(HEAD_DIM, dim=-1)
    block_table = (torch.randperm(num_blocks - 1, device=device) + 1).int()
    q = torch.randn(num_tokens + num_padding, num_heads, HEAD_DIM, device=device)
    q = (torch.nn.functional.normalize(q, dim=-1) * 3).bfloat16()
    width = num_heads * HEAD_DIM
    proj = torch.randn(num_tokens + num_padding, width + num_heads + 4, device=device)
    proj = proj.bfloat16()
    q[num_tokens:] = proj[num_tokens:] = float("nan")
    q_lens = torch.tensor([0, *[q_len for q_len, _ in seqs]], device=device)
    return SimpleNamespace(
        q=q,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table.view(len(seqs), max_blocks),
        rel_logits=torch.randn(
            num_tokens, num_heads, REL_EXTENT, device=device, dtype=torch.bfloat16
        ),
        gate=proj[:, :width],
        sinks=proj[:, width : width + num_heads],
        cu_seqlens_q=q_lens.cumsum(0).int(),
        seq_lens=torch.tensor(
            [kv_len for _, kv_len in seqs], dtype=torch.int32, device=device
        ),
        num_tokens=num_tokens,
    )


def _gated_reference(batch, seqs, window_left):
    """`_ref_attention` of a `_paged_batch`, gated and with the sink."""
    nt = batch.num_tokens
    out, lse = _ref_attention(
        batch.q[:nt],
        batch.key_cache,
        batch.value_cache,
        batch.rel_logits,
        seqs,
        batch.block_table,
        HEAD_DIM**-0.5,
        window_left,
    )
    gate, sinks = batch.gate[:nt].float(), batch.sinks[:nt].float()
    expected = out.view(nt, -1) * torch.sigmoid(gate)
    return expected * torch.sigmoid(lse - sinks).repeat_interleave(HEAD_DIM, dim=1)


@requires_rocm
@pytest.mark.parametrize("is_local", [False, True])
@torch.inference_mode()
def test_inkling2_reload_refills_rel_proj(is_local):
    """Full CUDA graphs capture the address of `rel_proj_shifted`: a weight
    reload refills it in place."""
    from vllm.models.inkling2 import shift_rel_proj
    from vllm.models.inkling2.amd.attention import Inkling2Attention

    proj = torch.randn(16, REL_EXTENT, device="cuda").bfloat16()
    layer = cast(
        Inkling2Attention, SimpleNamespace(rel_logits_proj=proj, is_local=is_local)
    )
    Inkling2Attention.process_weights_after_loading(layer)
    captured = layer.rel_proj_shifted
    proj.copy_(torch.randn_like(proj))
    Inkling2Attention.process_weights_after_loading(layer)
    assert layer.rel_proj_shifted is captured
    expected = shift_rel_proj(proj, is_local)
    torch.testing.assert_close(captured, expected, atol=0, rtol=0)


@requires_rocm
@pytest.mark.parametrize(
    "seqs,num_splits",
    [
        ([(1, 3000), (1, 1500)], 1),
        ([(1, 3000), (1, 1500)], 9),
        # 0: the query tiles of `gated_rel_attention`.
        ([(200, 3000), (37, 2100)], 0),
    ],
)
@torch.inference_mode()
def test_inkling2_window_skips_expired_pages(seqs, num_splits):
    """A local layer's window starts mid key block here. The pages before it
    may have been freed (to the null block, or reused): their NaNs must not
    reach the output."""
    from vllm.models.inkling2.amd.ops.attention import (
        gated_rel_attention,
        rel_decode_attention,
    )

    torch.manual_seed(0)
    block_size, window = 16, REL_EXTENT - 1
    batch = _paged_batch(seqs, 24, 8, block_size)
    expected = _gated_reference(batch, seqs, window)
    # Whole expired pages map to the null block; the window's first page
    # keeps its live keys only.
    block_table = batch.block_table.clone()
    for cache in (batch.key_cache, batch.value_cache):
        cache[0] = float("nan")
    for i, (q_len, kv_len) in enumerate(seqs):
        lo = max(kv_len - q_len - window, 0)
        assert lo % 64, "the window must start mid key block"
        block_table[i, : lo // block_size] = 0
        for cache in (batch.key_cache, batch.value_cache):
            page = batch.block_table[i, lo // block_size]
            cache[page, : lo % block_size] = float("nan")
    rel_in = shear_rel_logits(batch.rel_logits, seqs, 128, True)
    out = torch.empty_like(batch.gate)
    if num_splits:
        rel_decode_attention(
            batch.q,
            batch.key_cache,
            batch.value_cache,
            rel_in,
            batch.gate,
            batch.sinks,
            out,
            block_table=block_table,
            cache_seqlens=batch.seq_lens,
            cu_seqlens_q=batch.cu_seqlens_q,
            max_query_len=max(q_len for q_len, _ in seqs),
            num_actual_tokens=batch.num_tokens,
            softmax_scale=HEAD_DIM**-0.5,
            is_local=True,
            num_splits=num_splits,
        )
    else:
        gated_rel_attention(
            batch.q,
            batch.key_cache,
            batch.value_cache,
            rel_in,
            batch.gate,
            batch.sinks,
            out,
            block_table=block_table,
            cache_seqlens=batch.seq_lens,
            cu_seqlens_q=batch.cu_seqlens_q,
            max_seqlen_q=max(q_len for q_len, _ in seqs),
            num_actual_tokens=batch.num_tokens,
            softmax_scale=HEAD_DIM**-0.5,
            window=window,
        )
    torch.testing.assert_close(out.float(), expected, atol=1e-2, rtol=1e-2)


@requires_rocm
@pytest.mark.parametrize("window_left", [None, REL_EXTENT - 1])
@torch.inference_mode()
def test_inkling2_gated_rel_attention_zeroes_padding(window_left):
    """Padding tokens come out zero even with a NaN gate and sink, as in a
    batch padded to a CUDA graph's size."""
    from vllm.models.inkling2.amd.ops.attention import gated_rel_attention

    torch.manual_seed(0)
    seqs = [(37, 37), (200, 1700)]
    batch = _paged_batch(seqs, 24, 8, 16, num_padding=5)
    out = torch.full_like(batch.gate, float("nan"))
    gated_rel_attention(
        batch.q,
        batch.key_cache,
        batch.value_cache,
        shear_rel_logits(batch.rel_logits, seqs, 128, window_left is not None),
        batch.gate,
        batch.sinks,
        out,
        block_table=batch.block_table,
        cache_seqlens=batch.seq_lens,
        cu_seqlens_q=batch.cu_seqlens_q,
        max_seqlen_q=200,
        num_actual_tokens=batch.num_tokens,
        softmax_scale=HEAD_DIM**-0.5,
        window=-1 if window_left is None else window_left,
    )
    nt = batch.num_tokens
    expected = _gated_reference(batch, seqs, window_left)
    torch.testing.assert_close(out[:nt].float(), expected, atol=1e-2, rtol=1e-2)
    assert (out[nt:] == 0).all()


@requires_rocm
@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,verify_len,split",
    [
        (48, 8, 1, True),
        (48, 8, 8, True),
        # 8 queries of 16 heads per KV head exceed a split-KV program.
        (24, 2, 8, False),
    ],
)
@torch.inference_mode()
def test_inkling2_mixed_batch_splits_leading_decodes(
    monkeypatch, is_local, num_heads, num_kv_heads, verify_len, split
):
    """A batch of decode/verify requests followed by prefills runs the former
    on split-KV when a program holds their queries, and the rest on the query
    tiles; padding tokens come out zero."""
    from vllm.models.inkling2.amd import attention as amd_attention
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    decode_tokens: list[int] = []
    rel_decode_attention = amd_attention.rel_decode_attention

    def record(*args, **kwargs):
        decode_tokens.append(kwargs["num_actual_tokens"])
        rel_decode_attention(*args, **kwargs)

    monkeypatch.setattr(amd_attention, "rel_decode_attention", record)
    torch.manual_seed(0)
    seqs = [(verify_len, 3000), (1, 900), (200, 1700), (37, 37)]
    window_left = REL_EXTENT - 1 if is_local else None
    batch = _paged_batch(seqs, num_heads, num_kv_heads, 16, num_padding=3)
    layer = SimpleNamespace(
        is_local=is_local,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        rel_extent=REL_EXTENT,
        scaling=HEAD_DIM**-0.5,
        window=-1 if window_left is None else window_left,
        max_decode_query_len=8,
    )
    md = SimpleNamespace(
        seq_lens=batch.seq_lens,
        max_query_len=200,
        block_table=batch.block_table,
        query_start_loc=batch.cu_seqlens_q,
        num_actual_tokens=batch.num_tokens,
        num_decode_reqs=2,
        num_decode_tokens=verify_len + 1,
    )
    out = torch.full_like(batch.gate, float("nan"))
    amd_attention.Inkling2Attention._rel_attention(
        cast(amd_attention.Inkling2Attention, layer),
        cast(FlashAttentionMetadata, md),
        batch.q,
        batch.key_cache,
        batch.value_cache,
        shear_rel_logits(batch.rel_logits, seqs, 128, is_local),
        batch.gate,
        batch.sinks,
        out,
    )
    assert decode_tokens == ([verify_len + 1] if split else [])
    nt = batch.num_tokens
    expected = _gated_reference(batch, seqs, window_left)
    torch.testing.assert_close(out[:nt].float(), expected, atol=1e-2, rtol=1e-2)
    assert (out[nt:] == 0).all()
