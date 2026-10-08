# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""cand_logits.candidate_logits against AITER's dense
deepgemm_fp8_paged_mqa_logits, called the way vLLM's decode indexer calls it
on gfx942 (rocm_aiter_mla_sparse.rocm_fp8_paged_mqa_logits).

For one decode request of 6 rows, every compact logit and id must be
bit-identical to the dense logit gathered at the same candidate position,
with -inf and id -1 where a candidate block is -1 or a position is at or past
the row's end. The top 512 of the compact rows must then be the top 512 that
vLLM's path takes from the dense rows within the candidate blocks.
"""

import pytest
import torch

from vllm.platforms import current_platform


def _on_gfx942() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx942

    return on_gfx942()


pytestmark = pytest.mark.skipif(
    not _on_gfx942(), reason="the kernel reproduces AITER's gfx942 kernel"
)

ROWS, DIM, PAGE = 6, 128, 64
MAX_MODEL_LEN = 1 << 20
CAND_BLOCKS, CAND_BLOCK = 2048, 8


def _make_inputs(ctx: int, heads: int):
    """FP8 queries, head weights, row lengths, block tables and a paged
    indexer cache of FP8 keys and FP32 key scales. Row r is the request's
    token at position ctx - ROWS + r and sees the keys up to its own
    position."""
    from aiter import dtypes

    torch.manual_seed(0)
    q = (torch.randn(ROWS, 1, heads, DIM, device="cuda") * 0.5).to(dtypes.fp8)
    weights = torch.randn(ROWS, heads, device="cuda", dtype=torch.float32)
    context_lens = torch.arange(
        ctx - ROWS + 1, ctx + 1, device="cuda", dtype=torch.int32
    )
    used = (ctx + PAGE - 1) // PAGE
    num_pages = used + 64
    tables = torch.zeros(ROWS, MAX_MODEL_LEN // PAGE, device="cuda", dtype=torch.int32)
    tables[:, :used] = torch.randperm(num_pages, device="cuda")[:used].to(torch.int32)
    vals = (torch.randn(num_pages, PAGE * DIM, device="cuda") * 0.5).to(dtypes.fp8)
    scales = torch.rand(num_pages, PAGE, device="cuda") + 0.5
    kv = torch.cat([vals.view(torch.uint8), scales.view(torch.uint8)], dim=1)
    return q, weights, context_lens, tables, kv.view(num_pages, PAGE, 1, DIM + 4)


def _make_candidates(context_lens: torch.Tensor, gen) -> torch.Tensor:
    """Distinct candidate blocks of each row in random order, with the row's
    newest block among them and -1 padding where a row has fewer than 2048
    blocks. Row 1 also gets -1 in a few places in the middle, as layer 20
    writes for a block whose score is -inf."""
    cand = torch.full((ROWS, CAND_BLOCKS), -1, dtype=torch.int32)
    for r in range(ROWS):
        blocks = (int(context_lens[r]) + CAND_BLOCK - 1) // CAND_BLOCK
        if blocks <= CAND_BLOCKS:
            picks = torch.randperm(blocks, generator=gen)
        else:
            others = torch.randperm(blocks - 1, generator=gen)[: CAND_BLOCKS - 1]
            picks = torch.cat([others, torch.tensor([blocks - 1])])
            picks = picks[torch.randperm(CAND_BLOCKS, generator=gen)]
        cand[r, : picks.numel()] = picks.to(torch.int32)
    cand[1, 100:105] = -1
    return cand.cuda()


def _gather_reference(dense, context_lens, cand):
    """The compact rows that the candidate gather writes from dense logits."""
    i = torch.arange(CAND_BLOCKS * CAND_BLOCK, device=cand.device)
    block = cand[:, i // CAND_BLOCK].long()
    col = block * CAND_BLOCK + (i % CAND_BLOCK)[None, :]
    live = (block >= 0) & (col < context_lens[:, None].long())
    logits = torch.gather(dense[:ROWS], 1, torch.where(live, col, 0))
    logits = torch.where(live, logits, float("-inf"))
    ids = torch.where(live, col, -1).to(torch.int32)
    return logits, ids


@pytest.mark.parametrize("seq_lens_2d", [False, True])
@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("ctx", [8192, 131072])
def test_candidate_logits_bit_identical_to_dense(ctx, heads, seq_lens_2d):
    from vllm.model_executor.layers.dsv41_gfx942 import cand_logits, topk
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import paged_mqa_logits_module

    aiter_logits = paged_mqa_logits_module()
    if aiter_logits is None:
        pytest.skip("AITER's paged MQA logits module is not installed")
    q, weights, context_lens, tables, kv = _make_inputs(ctx, heads)
    cand = _make_candidates(context_lens.cpu(), torch.Generator().manual_seed(1))
    lens = context_lens.view(-1, 1) if seq_lens_2d else context_lens
    row_len = CAND_BLOCKS * CAND_BLOCK
    dense = torch.full((ROWS, MAX_MODEL_LEN), -1.0, device="cuda")
    out_logits = torch.empty(ROWS, row_len, device="cuda")
    out_ids = torch.empty(ROWS, row_len, device="cuda", dtype=torch.int32)
    aiter_logits.deepgemm_fp8_paged_mqa_logits(
        q,
        kv,
        weights,
        dense,
        context_lens,
        tables,
        MAX_MODEL_LEN,
        ChunkK=256,
        Preshuffle=True,
        KVBlockSize=PAGE,
        WavePerEU=2,
    )
    cand_logits.candidate_logits(
        q.view(ROWS, heads, DIM),
        kv,
        weights,
        lens,
        1,
        tables,
        cand,
        CAND_BLOCK,
        out_logits,
        out_ids,
    )
    torch.accelerator.synchronize()
    ref_logits, ref_ids = _gather_reference(dense, context_lens, cand)
    assert torch.equal(ref_ids, out_ids)
    assert torch.equal(ref_logits.view(torch.int32), out_logits.view(torch.int32))

    ext = topk._load()
    old = torch.empty(ROWS, 512, device="cuda", dtype=torch.int32)
    new = torch.empty_like(old)
    ext.candidate_top_k_512(dense, 1, lens, cand, CAND_BLOCK, old)
    ext.compact_top_k_512_regs(out_logits, out_ids, lens, 1, new)
    torch.accelerator.synchronize()
    assert torch.equal(old.sort(1).values, new.sort(1).values)
