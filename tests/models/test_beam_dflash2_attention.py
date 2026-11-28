# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-KV sliding-window attention used by the ROCm Beam DFlash2 drafter."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")


def _reference(q, kv, query_start_loc, seq_lens, block_table, scale, window, causal):
    num_heads, head_size = q.shape[1], q.shape[2]
    page_size, num_kv_heads = kv.shape[2], kv.shape[1]
    out = torch.zeros_like(q, dtype=torch.float32)
    for seq in range(seq_lens.numel()):
        start, end = int(query_start_loc[seq]), int(query_start_loc[seq + 1])
        seq_len = int(seq_lens[seq])
        pos = torch.arange(seq_len, device=q.device)
        pages = block_table[seq, pos // page_size].long()
        rows = kv[pages, :, pos % page_size].float()  # [seq_len, Hkv, 2D]
        rep = num_heads // num_kv_heads
        k = rows[..., :head_size].repeat_interleave(rep, 1)
        v = rows[..., head_size:].repeat_interleave(rep, 1)
        q_pos = seq_len - (end - start) + torch.arange(end - start, device=q.device)
        dist = q_pos[:, None] - pos[None, :]
        visible = torch.ones_like(dist, dtype=torch.bool)
        if causal:
            visible &= dist >= 0
        if window:
            visible &= dist.abs() < window
        s = torch.einsum("qhd,khd->hqk", q[start:end].float(), k) * scale
        s = s.masked_fill(~visible[None], float("-inf"))
        out[start:end] = torch.einsum("hqk,khd->qhd", s.softmax(-1), v)
    return out


def _inputs(query_lens, heads, page_size, hnd):
    torch.manual_seed(0)
    device = torch.device("cuda")
    (num_heads, num_kv_heads), head_size = heads, 128
    context_lens = [9000, 30, 5000, 0][: len(query_lens)]
    seq_lens = [c + q for c, q in zip(context_lens, query_lens)]
    max_pages = (max(seq_lens) + page_size - 1) // page_size
    num_blocks = len(seq_lens) * max_pages + 1
    shape = (num_blocks, page_size, num_kv_heads, 2 * head_size)
    kv = torch.randn(shape, device=device, dtype=torch.bfloat16).transpose(1, 2)
    if hnd:
        kv = kv.contiguous()
    block_table = (
        (torch.randperm(num_blocks - 1, device=device)[: len(seq_lens) * max_pages] + 1)
        .view(len(seq_lens), max_pages)
        .int()
    )
    lens = torch.tensor(query_lens, device=device)
    query_start_loc = torch.zeros(len(query_lens) + 1, device=device, dtype=torch.int32)
    query_start_loc[1:] = lens.cumsum(0)
    seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    q = torch.randn(
        sum(query_lens), num_heads, head_size, device=device, dtype=torch.bfloat16
    )
    return q, kv, query_start_loc, seq_lens_t, block_table


def _run(q, kv, query_start_loc, seq_lens, block_table, window, causal, **scales):
    from vllm.models.beam.amd.dflash2_attention_ops import (
        dflash2_split_kv_attention,
        split_kv_workspace,
    )

    num_heads, head_size = q.shape[1], q.shape[2]
    out = torch.full_like(q, float("nan"))
    dflash2_split_kv_attention(
        q,
        kv,
        out,
        query_start_loc,
        seq_lens,
        block_table,
        seq_lens.numel(),
        int((query_start_loc[1:] - query_start_loc[:-1]).max()),
        head_size**-0.5,
        window,
        causal,
        split_kv_workspace(4096, num_heads, head_size, q.device),
        512,
        **scales,
    )
    return out


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [0, 4096, 37])
@pytest.mark.parametrize("query_lens", [[6, 6, 6], [1, 16, 3, 40]])
@pytest.mark.parametrize("hnd", [True, False])
# Per-rank drafter heads at TP8 and TP4.
@pytest.mark.parametrize("heads", [(5, 1), (10, 2)])
# 64 is vLLM's default block size.
@pytest.mark.parametrize("page_size", [16, 64])
def test_split_kv_matches_reference(causal, window, query_lens, hnd, heads, page_size):
    q, kv, query_start_loc, seq_lens, block_table = _inputs(
        query_lens, heads, page_size, hnd
    )
    out = _run(q, kv, query_start_loc, seq_lens, block_table, window, causal)
    expected = _reference(
        q,
        kv,
        query_start_loc,
        seq_lens,
        block_table,
        q.shape[2] ** -0.5,
        window,
        causal,
    )
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)


def test_split_kv_graph_replay():
    """One captured graph replayed with new device inputs: lengths, block
    table, Q and KV change, and a request turns into padding. The host-side
    launch parameters (num_seqs, max_query_len) stay as captured."""
    from vllm.models.beam.amd.dflash2_attention_ops import (
        dflash2_split_kv_attention,
        split_kv_workspace,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_heads, num_kv_heads, head_size, page_size = 5, 1, 128, 64
    num_seqs, q_len, max_pages, window = 4, 6, 160, 4096
    num_blocks = num_seqs * max_pages + 1
    scale = head_size**-0.5
    kv = torch.empty(
        num_blocks,
        num_kv_heads,
        page_size,
        2 * head_size,
        device=device,
        dtype=torch.bfloat16,
    )
    q = torch.empty(
        num_seqs * q_len, num_heads, head_size, device=device, dtype=torch.bfloat16
    )
    out = torch.empty_like(q)
    query_start_loc = torch.empty(num_seqs + 1, device=device, dtype=torch.int32)
    seq_lens = torch.empty(num_seqs, device=device, dtype=torch.int32)
    block_table = torch.empty(num_seqs, max_pages, device=device, dtype=torch.int32)
    workspace = split_kv_workspace(4096, num_heads, head_size, device)

    def fill(context_lens, query_lens):
        kv.normal_()
        q.normal_()
        out.fill_(float("nan"))
        query_start_loc.copy_(torch.tensor([0, *query_lens]).cumsum(0))
        seq_lens.copy_(torch.tensor([c + n for c, n in zip(context_lens, query_lens)]))
        pages = torch.randperm(num_blocks - 1, device=device)[: num_seqs * max_pages]
        block_table.copy_((pages + 1).view(num_seqs, max_pages))

    def run():
        dflash2_split_kv_attention(
            q,
            kv,
            out,
            query_start_loc,
            seq_lens,
            block_table,
            num_seqs,
            q_len,
            scale,
            window,
            False,
            workspace,
            512,
        )

    fill([9000, 30, 5000, 700], [q_len] * num_seqs)
    run()  # compile outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    for context_lens, query_lens in [
        ([9000, 30, 5000, 700], [q_len] * num_seqs),
        ([4100, 10000, 70, 0], [q_len, q_len, q_len, 0]),
    ]:
        fill(context_lens, query_lens)
        graph.replay()
        torch.cuda.synchronize()
        num_tokens = sum(query_lens)
        expected = _reference(
            q, kv, query_start_loc, seq_lens, block_table, scale, window, False
        )
        torch.testing.assert_close(
            out[:num_tokens].float(), expected[:num_tokens], rtol=2e-2, atol=2e-2
        )


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [0, 4096])
@pytest.mark.parametrize("page_size", [16, 64])
def test_split_kv_fp8_cache_matches_reference(causal, window, page_size):
    q, kv, query_start_loc, seq_lens, block_table = _inputs(
        [6, 6, 6], (5, 1), page_size, hnd=True
    )
    head_size = q.shape[2]
    fp8 = current_platform.fp8_dtype()
    k_scale = torch.tensor([0.05], device=q.device)
    v_scale = torch.tensor([0.02], device=q.device)
    kv_fp8 = torch.cat(
        [
            (kv[..., :head_size].float() / k_scale).to(fp8),
            (kv[..., head_size:].float() / v_scale).to(fp8),
        ],
        dim=-1,
    )
    dequant = torch.cat(
        [
            kv_fp8[..., :head_size].float() * k_scale,
            kv_fp8[..., head_size:].float() * v_scale,
        ],
        dim=-1,
    )

    out = _run(
        q,
        kv_fp8,
        query_start_loc,
        seq_lens,
        block_table,
        window,
        causal,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    expected = _reference(
        q,
        dequant,
        query_start_loc,
        seq_lens,
        block_table,
        head_size**-0.5,
        window,
        causal,
    )
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
