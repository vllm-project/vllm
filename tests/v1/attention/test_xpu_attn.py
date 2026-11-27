# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU_ATTN decode/prefill split, run on CPU with a reference paged attention
standing in for the vllm_xpu_kernels varlen call."""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.v1.attention.backends import xpu_attn
from vllm.v1.kv_cache_interface import AttentionSpec

HQ, HKV, D, BS, NUM_BLOCKS = 8, 2, 64, 16, 64
LAYER = SimpleNamespace(
    _q_scale=torch.ones(()), _k_scale=torch.ones(()), _v_scale=torch.ones(())
)


def ref_varlen(
    calls,
    *,
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    block_table,
    softmax_scale,
    causal,
    is_mix_batch=True,
    **kwargs,
):
    """Paged varlen attention, one sequence at a time."""
    query_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    assert int(cu_seqlens_q[-1]) == q.shape[0] == out.shape[0]
    assert max_seqlen_q >= int(query_lens.max())
    calls.append((cu_seqlens_q.numel() - 1, max_seqlen_q, is_mix_batch))
    group = q.shape[1] // k.shape[2]
    for i in range(cu_seqlens_q.numel() - 1):
        s, e = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        kv_len = int(seqused_k[i])
        blocks = block_table[i, : (kv_len + BS - 1) // BS].long()
        keys = k[blocks].reshape(-1, k.shape[2], D)[:kv_len]
        values = v[blocks].reshape(-1, v.shape[2], D)[:kv_len]
        keys = keys.repeat_interleave(group, 1).double()
        values = values.repeat_interleave(group, 1).double()
        scores = torch.einsum("qhd,khd->hqk", q[s:e].double(), keys) * softmax_scale
        if causal:
            q_pos = torch.arange(kv_len - (e - s), kv_len).view(-1, 1)
            scores = scores.masked_fill(
                torch.arange(kv_len).view(1, -1) > q_pos, float("-inf")
            )
        out[s:e] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), values).to(
            out.dtype
        )
    return out


def run_split(monkeypatch, query_lens, seq_lens, num_speculative_tokens=None):
    """Return (kernel calls, output, reference output) for one forward."""
    calls: list[tuple[int, int, bool]] = []
    monkeypatch.setattr(
        xpu_attn,
        "xpu_flash_attn_varlen_func",
        lambda **kw: ref_varlen(calls, **kw),
    )
    speculative_config = (
        None
        if num_speculative_tokens is None
        else SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens,
            parallel_drafting=False,
            max_num_new_slots_for_drafting=0,
        )
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, cp_kv_cache_interleave_size=1
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=64),
    )
    spec = AttentionSpec(
        block_size=BS, num_kv_heads=HKV, head_size=D, dtype=torch.float32
    )
    builder = xpu_attn.XpuAttentionMetadataBuilder(
        spec, [], vllm_config, torch.device("cpu")
    )
    impl = xpu_attn.XpuAttentionImpl(
        num_heads=HQ,
        head_size=D,
        scale=D**-0.5,
        num_kv_heads=HKV,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    torch.manual_seed(0)
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        BS,
        torch.device("cpu"),
        max_block_idx=NUM_BLOCKS,
    )
    attn_metadata = builder.build(0, common)
    q = torch.randn(sum(query_lens), HQ, D)
    kv_cache = torch.randn(NUM_BLOCKS, HKV, BS, 2 * D)
    out = torch.full_like(q, float("nan"))
    impl.forward(LAYER, q, q[:, :HKV], q[:, :HKV], kv_cache, attn_metadata, out)
    kernel_calls = list(calls)

    key_cache, value_cache = kv_cache.transpose(1, 2).split(D, dim=-1)
    ref = ref_varlen(
        calls,
        q=q,
        k=key_cache,
        v=value_cache,
        out=torch.zeros_like(q),
        cu_seqlens_q=common.query_start_loc,
        max_seqlen_q=common.max_query_len,
        seqused_k=common.seq_lens,
        block_table=common.block_table_tensor,
        softmax_scale=D**-0.5,
        causal=True,
    )
    return kernel_calls, out, ref


def test_mixed_batch_runs_decode_and_prefill_calls(monkeypatch):
    """Decodes run as a decode-only call (max_seqlen_q 1) and the prefills as a
    prefill-only call; no call is a mixed batch."""
    calls, out, ref = run_split(monkeypatch, [1, 1, 1, 7, 13], [20, 33, 5, 40, 13])
    assert calls == [(3, 1, False), (2, 13, False)]
    assert torch.equal(out, ref)


@pytest.mark.parametrize(
    ("query_lens", "seq_lens", "max_seqlen_q"),
    [([1, 1, 1], [9, 30, 17], 1), ([5, 9], [5, 40], 9)],
    ids=["decode_only", "prefill_only"],
)
def test_uniform_batch_runs_one_call(monkeypatch, query_lens, seq_lens, max_seqlen_q):
    calls, out, ref = run_split(monkeypatch, query_lens, seq_lens)
    assert calls == [(len(query_lens), max_seqlen_q, False)]
    assert torch.equal(out, ref)


def test_verification_rows_run_with_decodes(monkeypatch):
    """With 2 speculative tokens, rows of up to 3 query tokens are decodes. The
    decode call is bounded by that threshold and runs as a mixed batch, so its
    single-token rows still take the paged decode kernel."""
    calls, out, ref = run_split(
        monkeypatch, [3, 2, 1, 9], [30, 12, 8, 21], num_speculative_tokens=2
    )
    assert calls == [(3, 3, True), (1, 9, False)]
    assert torch.equal(out, ref)
