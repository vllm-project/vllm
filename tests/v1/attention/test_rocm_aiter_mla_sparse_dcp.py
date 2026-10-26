# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode context parallelism for ROCM_AITER_MLA_SPARSE.

The tests simulate the DCP ranks in one process: they shard a request's KV
cache the way DCP interleaves it, run each rank's part, and merge the partial
results with their LSE. The merged result must match attention over the whole
cache.
"""

import pytest
import torch

from vllm.model_executor.kernels.attention.dsa.dcp_topk_torch import (
    pack_dcp_topk_keys,
    select_topk_from_keys,
)
from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm-only tests", allow_module_level=True)

DEVICE = torch.device("cuda")


def _owner(token: torch.Tensor, world: int, interleave: int) -> torch.Tensor:
    return (token // interleave) % world


def _local_position(token: torch.Tensor, world: int, interleave: int):
    return (token // (world * interleave)) * interleave + token % interleave


def _merge_like_dcp(logits_full, topk, world, interleave):
    """Run the local top-k on each simulated rank, then the key merge."""
    num_rows, seq_len = logits_full.shape
    tokens = torch.arange(seq_len, device=DEVICE)
    keys = []
    for rank in range(world):
        owned = tokens[_owner(tokens, world, interleave) == rank]
        local_logits = logits_full[:, owned]
        k = min(topk, owned.numel())
        local_topk = torch.full((num_rows, topk), -1, dtype=torch.int32, device=DEVICE)
        if k > 0:
            # Highest score first, the lower position wins a tie. Local
            # positions keep the order of the global ids, so this matches the
            # tie rule of the reference. The production top-k kernels break
            # ties arbitrarily, as they do without DCP.
            n = owned.numel()
            key = -local_logits.double() * n * 2 + torch.arange(
                n, device=DEVICE, dtype=torch.float64
            )
            local_topk[:, :k] = key.argsort(dim=1)[:, :k].to(torch.int32)
        keys.append(
            pack_dcp_topk_keys(local_logits, local_topk, rank, world, interleave)
        )
    gathered = torch.cat(keys, dim=1)
    out = torch.empty((num_rows, topk), dtype=torch.int32, device=DEVICE)
    select_topk_from_keys(gathered, topk, out)
    return out


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("interleave", [1, 16])
@pytest.mark.parametrize("seq_len", [100, 3000, 20000])
def test_dcp_topk_merge_matches_global_topk(world, interleave, seq_len):
    torch.manual_seed(0)
    topk = 2048
    num_rows = 5
    # Integer-valued scores make ties common, so the tie rule is tested too.
    logits = torch.randint(-50, 50, (num_rows, seq_len), device=DEVICE).float()
    merged = _merge_like_dcp(logits, topk, world, interleave)

    k = min(topk, seq_len)
    # Reference: highest score first, the lower token id wins a tie.
    order = torch.argsort(
        -logits.double() * seq_len * 2
        + torch.arange(seq_len, device=DEVICE, dtype=torch.float64),
        dim=1,
    )
    expected = order[:, :k]
    for row in range(num_rows):
        got = merged[row]
        assert int((got[:k] >= 0).sum()) == k
        assert bool((got[k:] == -1).all())
        assert set(got[:k].tolist()) == set(expected[row].tolist())


def test_dcp_topk_merge_rank_with_no_tokens():
    # With 8 ranks and 3 tokens, ranks 3 to 7 own nothing.
    logits = torch.tensor([[0.5, -1.0, 2.0]], device=DEVICE)
    merged = _merge_like_dcp(logits, 2048, 8, 1)
    assert sorted(merged[0, :3].tolist()) == [0, 1, 2]
    assert bool((merged[0, 3:] == -1).all())


def test_dcp_topk_merge_ignores_nan():
    logits = torch.tensor([[float("nan"), 1.0, 0.0, 3.0]], device=DEVICE)
    merged = _merge_like_dcp(logits, 2, 2, 1)
    assert sorted(merged[0].tolist()) == [1, 3]


@pytest.mark.parametrize("world", [2, 8])
@pytest.mark.parametrize("interleave", [1, 16])
def test_convert_and_compact_keep_only_owned_slots(world, interleave):
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        compact_dcp_kv_indptr,
        triton_convert_req_index_to_global_index,
    )

    torch.manual_seed(1)
    topk = 2048
    block_size = 64
    seq_lens = [5000, 300, 2048]
    num_tokens = len(seq_lens)
    max_blocks = 64
    block_table = (
        torch.randperm(num_tokens * max_blocks, device=DEVICE)
        .to(torch.int32)
        .view(num_tokens, max_blocks)
    )
    req_id = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
    token_indices = torch.full((num_tokens, topk), -1, dtype=torch.int32, device=DEVICE)
    for t, n in enumerate(seq_lens):
        k = min(topk, n)
        token_indices[t, :k] = torch.randperm(n, device=DEVICE)[:k].to(torch.int32)
    sparse_lens = torch.tensor(
        [min(n, topk) for n in seq_lens], dtype=torch.int32, device=DEVICE
    )

    for rank in range(world):
        indptr = torch.zeros(num_tokens + 4, dtype=torch.int32, device=DEVICE)
        torch.cumsum(sparse_lens, dim=0, out=indptr[1 : num_tokens + 1])
        indptr[num_tokens + 1 :] = indptr[num_tokens]
        indices = torch.full((num_tokens * topk,), -7, dtype=torch.int32, device=DEVICE)
        counts = torch.zeros(num_tokens, dtype=torch.int32, device=DEVICE)
        scratch = torch.empty(
            num_tokens * (topk + 1) + 1, dtype=torch.int32, device=DEVICE
        )
        triton_convert_req_index_to_global_index(
            req_id,
            block_table,
            token_indices,
            indptr,
            indices,
            BLOCK_STRIDE_ROWS=block_size,
            BLOCK_SIZE=block_size,
            NUM_TOPK_TOKENS=topk,
            dcp_size=world,
            dcp_rank=rank,
            cp_kv_cache_interleave_size=interleave,
            valid_counts=counts,
        )
        compact_dcp_kv_indptr(indptr, indices, counts, topk, scratch)

        for t in range(num_tokens):
            sel = token_indices[t]
            sel = sel[sel >= 0]
            mine = sel[_owner(sel, world, interleave) == rank]
            local = _local_position(mine, world, interleave)
            expected = (
                block_table[t, local // block_size] * block_size + local % block_size
            )
            start, end = int(indptr[t]), int(indptr[t + 1])
            assert end - start == mine.numel()
            assert sorted(indices[start:end].tolist()) == sorted(expected.tolist())
        # The padding rows after num_tokens keep their original offsets.


def _reference_attention(q, kv, kv_lora_rank, scale):
    # q: [heads, d], kv: [n, d] -> out [heads, kv_lora_rank], lse [heads]
    scores = (q.float() @ kv.float().T) * scale
    lse = torch.logsumexp(scores, dim=-1)
    out = torch.softmax(scores, dim=-1) @ kv[:, :kv_lora_rank].float()
    return out, lse


@pytest.mark.parametrize(
    "dtype,num_heads",
    [
        (torch.bfloat16, 16),
        (torch.bfloat16, 32),
        (torch.bfloat16, 64),
        (torch.bfloat16, 128),
        # The DCP path pads FP8 queries to the 128-head kernel. The 16-head
        # FP8 split-KV kernel is not used under DCP.
        ("fp8", 64),
        ("fp8", 128),
    ],
)
@pytest.mark.parametrize("num_tokens", [1, 7, 256, 1024])
def test_mla_decode_fwd_lse_matches_reference(dtype, num_heads, num_tokens):
    """The LSE wrapper must return a correct LSE for every split count.

    Before the fix, BF16 with 64 heads and a batch that fills the GPU got one
    KV split and an LSE buffer that was never written.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    torch.manual_seed(2)
    kv_lora_rank, rope = 512, 64
    d = kv_lora_rank + rope
    topk = 2048
    scale = d**-0.5
    num_slots = 8192
    kv = torch.randn(num_slots, d, device=DEVICE, dtype=torch.bfloat16) * 0.5
    q = torch.randn(num_tokens, num_heads, d, device=DEVICE, dtype=torch.bfloat16)
    lens = torch.randint(1, topk + 1, (num_tokens,), device=DEVICE)
    lens[0] = topk
    indptr = torch.zeros(num_tokens + 1, dtype=torch.int32, device=DEVICE)
    torch.cumsum(lens, dim=0, out=indptr[1:])
    indices = torch.cat(
        [torch.randperm(num_slots, device=DEVICE)[: int(n)] for n in lens]
    ).to(torch.int32)
    qo_indptr = torch.arange(num_tokens + 1, dtype=torch.int32, device=DEVICE)
    last_page = torch.ones(num_tokens, dtype=torch.int32, device=DEVICE)

    if dtype == "fp8":
        fp8 = current_platform.fp8_dtype()
        one = torch.ones(1, device=DEVICE, dtype=torch.float32)
        q_in, kv_in = q.to(fp8), kv.to(fp8)
        q_scale = kv_scale = one
        q_ref, kv_ref = q_in.to(torch.bfloat16), kv_in.to(torch.bfloat16)
    else:
        q_in, kv_in, q_scale, kv_scale = q, kv, None, None
        q_ref, kv_ref = q, kv

    out = torch.empty(
        num_tokens, num_heads, kv_lora_rank, device=DEVICE, dtype=torch.bfloat16
    )
    lse = rocm_aiter_ops.mla_decode_fwd_lse(
        q_in,
        kv_in.view(-1, 1, d),
        out,
        scale,
        qo_indptr,
        1,
        indptr,
        indices,
        last_page,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    for t in range(num_tokens):
        sel = indices[int(indptr[t]) : int(indptr[t + 1])].long()
        ref_out, ref_lse = _reference_attention(
            q_ref[t], kv_ref[sel], kv_lora_rank, scale
        )
        torch.testing.assert_close(lse[t].float(), ref_lse, atol=2e-2, rtol=0)
        torch.testing.assert_close(out[t].float(), ref_out, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("world", [2, 8])
def test_sharded_decode_merges_to_full_attention(world):
    """Each rank attends over its owned selected tokens, and the LSE merge of
    the partial outputs equals attention over all selected tokens."""
    from vllm._aiter_ops import rocm_aiter_ops

    torch.manual_seed(3)
    kv_lora_rank, rope, heads = 512, 64, 64
    d = kv_lora_rank + rope
    scale = d**-0.5
    seq_len, topk, num_tokens = 50000, 2048, 4
    kv = torch.randn(seq_len, d, device=DEVICE, dtype=torch.bfloat16) * 0.5
    q = torch.randn(num_tokens, heads, d, device=DEVICE, dtype=torch.bfloat16)
    selected = torch.stack(
        [torch.randperm(seq_len, device=DEVICE)[:topk] for _ in range(num_tokens)]
    )
    qo_indptr = torch.arange(num_tokens + 1, dtype=torch.int32, device=DEVICE)
    last_page = torch.ones(num_tokens, dtype=torch.int32, device=DEVICE)

    outs, lses = [], []
    for rank in range(world):
        rows = []
        for t in range(num_tokens):
            sel = selected[t]
            rows.append(sel[_owner(sel, world, 1) == rank])
        lens = torch.tensor([r.numel() for r in rows], device=DEVICE)
        indptr = torch.zeros(num_tokens + 1, dtype=torch.int32, device=DEVICE)
        torch.cumsum(lens, dim=0, out=indptr[1:])
        # The test keeps the global cache layout, so a token's slot is its id.
        indices = torch.cat(rows).to(torch.int32)
        out = torch.empty(
            num_tokens, heads, kv_lora_rank, device=DEVICE, dtype=torch.bfloat16
        )
        lse = rocm_aiter_ops.mla_decode_fwd_lse(
            q, kv.view(-1, 1, d), out, scale, qo_indptr, 1, indptr, indices,
            last_page,
        )  # fmt: skip
        outs.append(out.float())
        lses.append(lse.float())

    lse_all = torch.stack(lses)  # [world, tokens, heads]
    merged_lse = torch.logsumexp(lse_all, dim=0)
    weights = torch.exp(lse_all - merged_lse).unsqueeze(-1)
    merged = (weights * torch.stack(outs)).sum(dim=0)

    for t in range(num_tokens):
        ref_out, ref_lse = _reference_attention(
            q[t], kv[selected[t]], kv_lora_rank, scale
        )
        torch.testing.assert_close(merged_lse[t], ref_lse, atol=2e-2, rtol=0)
        torch.testing.assert_close(merged[t], ref_out, atol=3e-2, rtol=3e-2)
