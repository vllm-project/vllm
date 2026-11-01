# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The FA2 packed-GQA context path must match one causal FA2 call over paged KV."""

import pytest
import torch

from tests.kernels.attention.test_flash_attn import ref_paged_attn
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops import fa2_packed_gqa_context as packed

try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func, is_fa_version_supported
except ImportError:
    pytest.skip("vllm_flash_attn is not available", allow_module_level=True)

if not current_platform.is_cuda() or not is_fa_version_supported(2):
    pytest.skip("requires CUDA with FlashAttention-2", allow_module_level=True)

BLOCK_SIZE = 16


def _run(
    query_lens: list[int],
    context_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    context_slack: list[int] | None = None,
    num_sms: int = 108,
):
    set_random_seed(0)
    num_q_heads, num_kv_heads = num_heads
    kv_lens = [c + q for c, q in zip(context_lens, query_lens)]
    num_seqs = len(query_lens)
    blocks_per_seq = [(k + BLOCK_SIZE - 1) // BLOCK_SIZE for k in kv_lens]
    num_blocks = sum(blocks_per_seq) + 4
    dtype = torch.bfloat16

    query = torch.randn(sum(query_lens), num_q_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, BLOCK_SIZE, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    perm = torch.randperm(num_blocks, dtype=torch.int32)
    block_table = torch.zeros(num_seqs, max(blocks_per_seq), dtype=torch.int32)
    offset = 0
    for i, n in enumerate(blocks_per_seq):
        block_table[i, :n] = perm[offset : offset + n]
        offset += n

    # New tokens' K/V as stored in the cache.
    new_k, new_v = [], []
    for i in range(num_seqs):
        pos = torch.arange(context_lens[i], kv_lens[i])
        blocks = block_table[i, pos // BLOCK_SIZE].long()
        new_k.append(key_cache[blocks, pos % BLOCK_SIZE])
        new_v.append(value_cache[blocks, pos % BLOCK_SIZE])
    key = torch.cat(new_k)
    value = torch.cat(new_v)

    query_start_loc_cpu = torch.zeros(num_seqs + 1, dtype=torch.int32)
    query_start_loc_cpu[1:] = torch.tensor(query_lens).cumsum(0)
    slack = context_slack or [0] * num_seqs
    seq_lens_ub_cpu = torch.tensor([k + s for k, s in zip(kv_lens, slack)])
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32)
    scale = head_size**-0.5

    plan = packed.build_packed_gqa_context_plan(
        buffers=packed.PackedGQAContextBuffers(
            max_num_reqs=num_seqs,
            max_num_tokens=sum(query_lens),
            group_size=num_q_heads // num_kv_heads,
            num_kv_heads=num_kv_heads,
            num_sms=num_sms,
            device=torch.device("cuda"),
        ),
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens_cpu_upper_bound=seq_lens_ub_cpu,
        query_start_loc=query_start_loc_cpu.cuda(),
        seq_lens=seq_lens.cuda(),
        block_table=block_table.cuda(),
        num_reqs=num_seqs,
        num_kv_heads=num_kv_heads,
        group_size=num_q_heads // num_kv_heads,
        block_size=BLOCK_SIZE,
        num_sms=num_sms,
    )
    assert plan is not None

    output = torch.empty_like(query).cuda()
    packed.packed_gqa_context_attention(
        flash_attn_varlen_func,
        query=query.cuda(),
        key=key.cuda(),
        value=value.cuda(),
        key_cache=key_cache.cuda(),
        value_cache=value_cache.cuda(),
        output=output,
        query_start_loc=query_start_loc_cpu.cuda(),
        max_query_len=max(query_lens),
        plan=plan,
        softmax_scale=scale,
        fa_version=2,
    )

    baseline = flash_attn_varlen_func(
        q=query.cuda(),
        k=key_cache.cuda(),
        v=value_cache.cuda(),
        cu_seqlens_q=query_start_loc_cpu.cuda(),
        max_seqlen_q=max(query_lens),
        seqused_k=seq_lens.cuda(),
        max_seqlen_k=max(kv_lens),
        softmax_scale=scale,
        causal=True,
        block_table=block_table.cuda(),
        fa_version=2,
    )
    ref = ref_paged_attn(
        query=query.clone(),
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_table,
        scale=scale,
    )
    torch.testing.assert_close(output.cpu(), ref, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(output, baseline, atol=2e-2, rtol=1e-2)
    return plan


@pytest.fixture(autouse=True)
def _small_thresholds(monkeypatch):
    monkeypatch.setattr(packed, "MIN_CONTEXT_LEN", 0)


@torch.inference_mode()
@pytest.mark.parametrize("num_heads", [(8, 2), (28, 4), (32, 8)])
@pytest.mark.parametrize("head_size", [64, 128, 256])
@pytest.mark.parametrize(
    "query_lens,context_lens",
    [
        ([4], [3000]),
        ([1, 6, 3], [4096, 100, 2000]),
        ([2, 2, 2, 2], [17, 1029, 0, 6000]),
        ([16], [1]),
    ],
)
def test_matches_single_fa2_call(num_heads, head_size, query_lens, context_lens):
    _run(query_lens, context_lens, num_heads, head_size)


@torch.inference_mode()
def test_many_chunks_per_request():
    """A tiny SM count forces the minimum chunk size and many chunks."""
    plan = _run([3], [8000], (28, 4), 128, num_sms=4096)
    assert plan.req_num_chunks.max().item() > 1


@torch.inference_mode()
def test_cpu_upper_bound_longer_than_actual():
    """Chunks past the real context (CPU lengths are an upper bound) stay empty."""
    plan = _run(
        [4, 4], [5000, 300], (32, 8), 128, context_slack=[3, 2000], num_sms=4096
    )
    kv_lens = plan.chunk_kv_lens.tolist()
    assert kv_lens.count(0) >= 2
    assert sum(kv_lens) == 5000 + 300
