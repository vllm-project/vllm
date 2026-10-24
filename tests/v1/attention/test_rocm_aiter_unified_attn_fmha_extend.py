# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_AITER_UNIFIED_ATTN: long extend-prefill rows on AITER's fp8 FMHA v3."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backend import CommonAttentionMetadata

BLOCK = 128
HEAD = 128


def _metadata(q_lens, seq_lens, device, num_blocks):
    qsl = [0]
    for q in q_lens:
        qsl.append(qsl[-1] + q)
    max_blocks = max(-(-s // BLOCK) for s in seq_lens)
    # every request owns a disjoint, shuffled run of physical blocks
    perm = torch.randperm(num_blocks)[: len(q_lens) * max_blocks]
    block_table = perm.view(len(q_lens), max_blocks).to(torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor(qsl, dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.tensor(qsl, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        num_reqs=len(q_lens),
        num_actual_tokens=qsl[-1],
        max_query_len=max(q_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=block_table.to(device),
        slot_mapping=torch.zeros(qsl[-1], dtype=torch.long, device=device),
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
    )


def test_row_split(monkeypatch):
    """Long requests go to the FMHA rows, the rest to a compact batch."""
    from vllm.v1.attention.backends import rocm_aiter_unified_attn as mod

    monkeypatch.setenv("VLLM_ROCM_UNIFIED_ATTN_FMHA_EXTEND_MIN_Q", "64")
    monkeypatch.setattr(mod, "on_gfx950", lambda: True)
    cm = _metadata([4, 100, 1, 64], [1000, 300, 50, 64], "cpu", 64)
    ext = mod._build_fmha_extend(cm, BLOCK, 1)
    assert ext is not None
    # query_start_loc = [0, 4, 104, 105, 169]
    assert ext.long_tok.tolist() == list(range(4, 104)) + list(range(105, 169))
    assert ext.long_cu_q.tolist() == [0, 100, 164]
    assert ext.long_cu_k.tolist() == [0, 300, 364]
    assert (ext.long_max_q, ext.long_max_k) == (100, 300)
    # context rows of request 1 then request 3, each in token order
    bt = cm.block_table_tensor.long()
    t1, t3 = torch.arange(300), torch.arange(64)
    want = torch.cat(
        [bt[1, t1 // BLOCK] * BLOCK + t1 % BLOCK, bt[3, t3 // BLOCK] * BLOCK + t3]
    )
    assert torch.equal(ext.kv_rows, want)
    assert ext.short_tok.tolist() == [0, 1, 2, 3, 104]
    assert ext.short_cu_q.tolist() == [0, 4, 5]
    assert ext.short_seq_lens.tolist() == [1000, 50]
    assert torch.equal(ext.short_block_table, cm.block_table_tensor[[0, 2]])

    # spec decode rows (1 + 3 tokens) stay on unified attention at any threshold
    monkeypatch.setenv("VLLM_ROCM_UNIFIED_ATTN_FMHA_EXTEND_MIN_Q", "4")
    assert mod._build_fmha_extend(cm, BLOCK, 1).long_tok[0].item() == 0
    ext = mod._build_fmha_extend(cm, BLOCK, 1, max_decode_q=4)
    assert ext.short_tok.tolist() == [0, 1, 2, 3, 104]

    monkeypatch.setenv("VLLM_ROCM_UNIFIED_ATTN_FMHA_EXTEND_MIN_Q", "0")
    assert mod._build_fmha_extend(cm, BLOCK, 1) is None
    monkeypatch.setenv("VLLM_ROCM_UNIFIED_ATTN_FMHA_EXTEND_MIN_Q", "200")
    assert mod._build_fmha_extend(cm, BLOCK, 1) is None


@pytest.mark.skipif(
    not current_platform.is_rocm()
    or current_platform.get_device_capability() != (9, 5),
    reason="gfx950 only",
)
@pytest.mark.parametrize("num_kv_heads", [1, 2])
def test_fmha_extend_matches_unified_attention(monkeypatch, num_kv_heads):
    """The FMHA rows and the compact unified rows reproduce one unified call."""
    from aiter.ops.triton.unified_attention import unified_attention

    from vllm.v1.attention.backends import rocm_aiter_unified_attn as mod

    monkeypatch.setenv("VLLM_ROCM_UNIFIED_ATTN_FMHA_EXTEND_MIN_Q", "512")
    torch.manual_seed(0)
    dev = torch.device("cuda")
    fp8 = current_platform.fp8_dtype()
    q_heads = 8 * num_kv_heads
    q_lens, seq_lens = [4, 1024, 4, 2048], [3000, 4096, 700, 2048]
    num_blocks = 256
    cm = _metadata(q_lens, seq_lens, dev, num_blocks)
    kv_cache = (
        torch.randn(num_blocks, num_kv_heads, BLOCK, 2 * HEAD, device=dev) * 0.5
    ).to(fp8)
    key_cache, value_cache = kv_cache.transpose(1, 2).split(HEAD, dim=-1)
    query = (torch.randn(cm.num_actual_tokens, q_heads, HEAD, device=dev) * 0.5).to(fp8)
    one = torch.ones((), dtype=torch.float32, device=dev)
    layer = SimpleNamespace(_q_scale=one, _k_scale=one, _v_scale=one)
    scale = HEAD**-0.5

    ref = torch.empty(query.shape, dtype=torch.bfloat16, device=dev)
    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=ref,
        cu_seqlens_q=cm.query_start_loc,
        max_seqlen_q=cm.max_query_len,
        seqused_k=cm.seq_lens,
        max_seqlen_k=cm.max_seq_len,
        softmax_scale=scale,
        causal=True,
        alibi_slopes=None,
        window_size=(-1, -1),
        block_table=cm.block_table_tensor,
        softcap=0,
        q_descale=one.view(1),
        k_descale=one.view(1),
        v_descale=one.view(1),
        sinks=None,
        output_scale=None,
    )

    ext = mod._build_fmha_extend(cm, BLOCK, num_kv_heads)
    assert ext is not None and ext.short_tok is not None
    impl = SimpleNamespace(head_size=HEAD, fp8_dtype=fp8, scale=scale)
    out = torch.zeros_like(ref)
    mod.RocmAiterUnifiedAttentionImpl._forward_fmha_extend(
        impl, layer, query, kv_cache, out, ext
    )
    long_rows = ext.long_tok
    cos = torch.nn.functional.cosine_similarity(
        out[long_rows].float().flatten(), ref[long_rows].float().flatten(), dim=0
    )
    assert cos > 0.999, cos
    torch.testing.assert_close(out[long_rows], ref[long_rows], atol=2e-2, rtol=0)
