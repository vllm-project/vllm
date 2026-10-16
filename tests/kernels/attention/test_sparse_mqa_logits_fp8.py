# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepGEMM sparse MQA logits + top-k on the FP8 indexer cache (SM100).

The sparse path scores only the candidate blocks. With the FP8 cache, K comes
as e4m3 values plus one power-of-two scale per token stored as four equal
UE8M0 bytes, and FP8 Q carries its scale in the (fp32) weights; the selected
keys must match the exact top-k over the candidate tokens up to the bf16
rounding of the kernels' logits.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not (
    torch.cuda.is_available() and current_platform.is_device_capability_family(100)
):
    pytest.skip("Requires an SM100-class GPU", allow_module_level=True)

from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (  # noqa: E402
    check_deep_select_layout,
    has_deep_select,
    pick_sparse_block_kv,
    sparse_mqa_logits_paged_decode,
    sparse_mqa_logits_prefill_chunk,
)
from vllm.model_executor.layers.sparse_attn_indexer import (  # noqa: E402
    kv_cache_as_quant_view,
)
from vllm.utils.deep_gemm import (  # noqa: E402
    has_deep_gemm_sparse_mqa,
    mqa_logits_mx_inputs,
)

if not (has_deep_gemm_sparse_mqa() and has_deep_select()):
    pytest.skip(
        "Requires DeepGEMM sparse MQA logits and DeepSelect", allow_module_level=True
    )

HEADS, HEAD_DIM = 32, 128
FP8 = torch.float8_e4m3fn


def _quant_fp8_k(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token power-of-two FP8 quantization, as the FP8 indexer cache.

    Returns (values [N, D] e4m3, scales [N] int32 as four equal UE8M0 bytes,
    dequantized [N, D] fp32)."""
    amax = k.abs().amax(dim=-1).clamp(min=1e-4)
    exp = torch.ceil(torch.log2(amax / 448.0)).to(torch.int32)
    scale = torch.exp2(exp.float())
    values = (k / scale[:, None]).to(FP8)
    ue8m0 = (exp + 127).to(torch.int64) * 0x01010101
    return values, ue8m0.to(torch.int32), values.float() * scale[:, None]


def _inputs(rows: int, keys: int, seed: int):
    torch.manual_seed(seed)
    q = torch.randn(rows, HEADS, HEAD_DIM, device="cuda").to(FP8)
    k_values, k_scale, k_deq = _quant_fp8_k(
        torch.randn(keys, HEAD_DIM, device="cuda") * 3
    )
    weights = torch.randn(rows, HEADS, device="cuda") / HEADS  # fp32, as FP8 Q
    return q, k_values, k_scale, k_deq, weights


def _candidates(row_lens, num_candidates, block_size, seed):
    """Unique random request-local candidate blocks per row, -1 padded."""
    g = torch.Generator().manual_seed(seed)
    out = torch.full((len(row_lens), num_candidates), -1, dtype=torch.int32)
    for r, n in enumerate(row_lens):
        blocks = (int(n) + block_size - 1) // block_size
        pick = torch.randperm(blocks, generator=g)[:num_candidates]
        out[r, : pick.numel()] = pick.to(torch.int32)
    return out.cuda()


def _check(topk_idx, q, k_deq, weights_bf16, ks, ke, cand, block_size, topk):
    """Selected keys (request-local) vs the exact top-k over the candidates."""
    for r in range(q.shape[0]):
        n = int(ke[r] - ks[r])
        blocks = cand[r][cand[r] >= 0].long()
        pos = blocks[:, None] * block_size + torch.arange(block_size, device="cuda")
        pos = pos.flatten()
        pos = pos[pos < n].sort().values  # searchsorted below needs order
        sel = topk_idx[r][topk_idx[r] >= 0].long()
        assert sel.numel() == min(topk, pos.numel()), (r, sel.numel(), pos.numel())
        assert torch.isin(sel, pos).all(), r
        assert sel.unique().numel() == sel.numel(), r
        kk = k_deq[int(ks[r]) + pos]
        scores = (
            torch.relu(q[r].float() @ kk.T) * weights_bf16[r].float()[:, None]
        ).sum(0)
        exact = scores.topk(min(topk, pos.numel())).values
        chosen = scores[torch.searchsorted(pos, sel)].sort(descending=True).values
        # The kernels emit bf16 logits: their error is relative to the row's
        # score magnitude, so only near-ties at the k-th boundary may swap.
        atol = 1.5e-2 * float(scores.abs().max())
        torch.testing.assert_close(chosen, exact, rtol=0, atol=atol)


@pytest.mark.parametrize("candidate_block_size", [8, 16])
@pytest.mark.parametrize("topk", [512, 1024])
def test_prefill_chunk_fp8(candidate_block_size, topk):
    rows, keys, num_candidates = 96, 24576, 256
    sbk = pick_sparse_block_kv(candidate_block_size)
    num_sparse = num_candidates * (candidate_block_size // sbk)
    check_deep_select_layout(num_sparse * sbk, topk)
    q, k_values, k_scale, k_deq, weights = _inputs(rows, keys, seed=11)
    # One request, causal rows near the end of the context.
    ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ke = torch.arange(keys - rows + 1, keys + 1, dtype=torch.int32, device="cuda")
    cand = _candidates(ke.tolist(), num_candidates, candidate_block_size, seed=3)
    q_scale, w = mqa_logits_mx_inputs(q, None, weights)
    assert q_scale.dtype == torch.int32 and w.dtype == torch.bfloat16
    out = torch.full((rows, topk), -7, dtype=torch.int32, device="cuda")
    int32 = dict(dtype=torch.int32, device="cuda")
    sparse_mqa_logits_prefill_chunk(
        q,
        q_scale,
        k_values,
        k_scale,
        w,
        ks,
        ke,
        cand,
        candidate_block_size,
        sbk,
        topk,
        out,
        sparse_indices=torch.empty(rows, num_sparse, **int32),
        end=torch.empty(rows, **int32),
        col_indices=torch.empty(rows, topk, **int32),
    )
    _check(out, q, k_deq, w, ks, ke, cand, candidate_block_size, topk)


@pytest.mark.parametrize("candidate_block_size", [8, 16])
def test_paged_decode_fp8(candidate_block_size):
    topk, page_kv, num_candidates = 512, 64, 256
    batch, max_ctx = 12, 20000
    sbk = pick_sparse_block_kv(candidate_block_size)
    num_sparse = num_candidates * (candidate_block_size // sbk)
    check_deep_select_layout(num_sparse * sbk, topk)
    torch.manual_seed(5)
    ctx = torch.randint(4096, max_ctx, (batch,), dtype=torch.int32)
    pages_per_row = (int(ctx.max()) + page_kv - 1) // page_kv
    num_pages = batch * pages_per_row + 3
    q, k_values, k_scale, k_deq, weights = _inputs(batch, num_pages * page_kv, seed=13)
    # FP8 indexer cache page: page_kv x 128 value bytes, then page_kv uint32
    # scales (the DeepGEMM paged layout); 64 x 132 B = 33 x 256 B per page.
    head_bytes = HEAD_DIM + 4
    pages = torch.empty(
        num_pages, page_kv * head_bytes, dtype=torch.uint8, device="cuda"
    )
    vals = k_values.view(torch.uint8).view(num_pages, page_kv * HEAD_DIM)
    pages[:, : page_kv * HEAD_DIM] = vals
    pages[:, page_kv * HEAD_DIM :] = k_scale.view(num_pages, page_kv).view(torch.uint8)
    kv_cache = kv_cache_as_quant_view(
        pages.view(num_pages, page_kv, head_bytes), HEAD_DIM, use_fp4_cache=False
    )
    # Shuffled, disjoint pages per row.
    perm = torch.randperm(num_pages)[: batch * pages_per_row].view(batch, -1)
    block_table = perm.to(torch.int32).cuda()
    cand = _candidates(ctx.tolist(), num_candidates, candidate_block_size, seed=9)
    q_scale, w = mqa_logits_mx_inputs(q, None, weights)
    out = torch.full((batch, topk), -7, dtype=torch.int32, device="cuda")
    int32 = dict(dtype=torch.int32, device="cuda")
    sparse_mqa_logits_paged_decode(
        q.unsqueeze(1),
        q_scale.unsqueeze(1),
        kv_cache,
        w,
        ctx.cuda(),
        block_table,
        torch.arange(batch, **int32),
        cand,
        candidate_block_size,
        sbk,
        topk,
        out,
        row_ks=torch.zeros(batch, **int32),
        sparse_indices=torch.empty(batch, num_sparse, **int32),
        end=torch.empty(batch, **int32),
        col_indices=torch.empty(batch, topk, **int32),
    )
    # Reference over each row's logical K: gather its pages in order.
    tok = block_table.long()[:, :, None] * page_kv + torch.arange(
        page_kv, device="cuda"
    )
    for r in range(batch):
        rows_k = k_deq[tok[r].flatten()]
        _check(
            out[r : r + 1],
            q[r : r + 1],
            rows_k,
            w[r : r + 1],
            torch.zeros(1, **int32),
            ctx[r : r + 1].cuda(),
            cand[r : r + 1],
            candidate_block_size,
            topk,
        )
