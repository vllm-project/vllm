# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gfx942 indexer op of DeepSeek-V4.1 (gfx942_sparse_attn_indexer in
vllm/models/deepseek_v41/amd/ops/sparse_indexer.py) against vLLM's ROCm
indexer op, rocm_aiter_sparse_attn_indexer, on a decode step.

The gfx942 op runs its own copy of the decode path of vLLM's op, with the
gfx942 top-512 launches in front of vLLM's kernels, so this test fails when
the copy and vLLM's op stop agreeing. Both ops get the same FP8 queries, head
weights and paged indexer cache of one request with its 5 DSpark drafts, as
the 6 flattened rows that the server's decode indexer uses. The layers are one
without candidate blocks, layer 20, which writes the candidate blocks of
layers 24 to 36, and one of those layers, which reads them. The ops must write
the same candidate blocks and choose the same top-512 columns. Equal values at
the cut may be chosen differently, and then the chosen values must be the same
multiset.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform


def _on_gfx942() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx942

    return on_gfx942()


pytestmark = pytest.mark.skipif(
    not _on_gfx942(), reason="the op runs the gfx942 launches"
)

ROWS, HEADS, DIM, PAGE = 6, 64, 128, 64
TOPK = 512
MAX_MODEL_LEN = 1 << 20
CAND_BLOCKS, CAND_BLOCK = 2048, 8
PREFIX = "model.layers.24.attn.indexer.k_cache"
# The index layers of V4.1 have compress ratio 1 or 2. On gfx942 both ratios
# take AITER's paged logits kernel and the same top-k, so one is enough.
RATIO = 1


@pytest.fixture
def gfx942_step(monkeypatch):
    """Turns on AITER and the gfx942 decode launches for one test, and makes
    the workspace that vLLM's logits call takes its output from. The model
    records --monokernel when it is created, and this test has no model, so it
    sets the recorded flag itself."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.deepseek_v41.amd import ops
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import paged_mqa_logits_module
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        is_workspace_manager_initialized,
    )

    if paged_mqa_logits_module() is None:
        pytest.skip("AITER's paged MQA logits module is not installed")
    monkeypatch.setattr(rocm_aiter_ops, "_AITER_ENABLED", True)
    monkeypatch.setattr(ops, "_specialized_decode", True)
    ops.enabled.cache_clear()
    if not is_workspace_manager_initialized():
        init_workspace_manager(torch.device("cuda"))
    yield
    ops.enabled.cache_clear()


def _make_step(ctx: int):
    """FP8 queries, head weights, row lengths, block tables and a paged
    indexer cache of FP8 keys and FP32 key scales. Row r is the request's
    token at position ctx - ROWS + r and sees the keys up to its own
    position."""
    from aiter import dtypes

    torch.manual_seed(ctx)
    q = (torch.randn(ROWS, HEADS, DIM, device="cuda") * 0.5).to(dtypes.fp8)
    weights = torch.randn(ROWS, HEADS, device="cuda", dtype=torch.float32)
    lens = torch.arange(ctx - ROWS + 1, ctx + 1, device="cuda", dtype=torch.int32)
    used = (ctx + PAGE - 1) // PAGE
    num_pages = used + 64
    tables = torch.zeros(ROWS, MAX_MODEL_LEN // PAGE, device="cuda", dtype=torch.int32)
    tables[:, :used] = torch.randperm(num_pages, device="cuda")[:used].to(torch.int32)
    vals = (torch.randn(num_pages, PAGE * DIM, device="cuda") * 0.5).to(dtypes.fp8)
    scales = torch.rand(num_pages, PAGE, device="cuda") + 0.5
    kv = torch.cat([vals.view(torch.uint8), scales.view(torch.uint8)], dim=1)
    return q, weights, lens, tables, kv.view(num_pages, PAGE, DIM + 4)


def _use_metadata(monkeypatch, lens: torch.Tensor, tables: torch.Tensor) -> None:
    """Gives both ops the metadata of a step of only decode rows, which the
    model runner passes through the forward context. Each of the 6 rows is
    its own decode row of length 1, as in the server's flattened layout."""
    from vllm.config import CUDAGraphMode
    from vllm.models.deepseek_v41.amd.ops import sparse_indexer
    from vllm.v1.attention.backends.mla.indexer import (
        DeepSeekV32IndexerDecodeMetadata,
        DeepseekV32IndexerMetadata,
    )
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse

    decode = DeepSeekV32IndexerDecodeMetadata(
        block_table=tables,
        seq_lens=lens,
        decode_lens=torch.ones(ROWS, dtype=torch.int32, device="cuda"),
        requires_padding=False,
        schedule_metadata=torch.empty(0, dtype=torch.int32, device="cuda"),
    )
    meta = DeepseekV32IndexerMetadata(
        seq_lens=lens,
        max_seq_len=int(lens.max()),
        slot_mapping=torch.empty(0, dtype=torch.int64, device="cuda"),
        num_decodes=ROWS,
        num_decode_tokens=ROWS,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=decode,
    )
    context = SimpleNamespace(
        attn_metadata={PREFIX: meta}, cudagraph_runtime_mode=CUDAGraphMode.NONE
    )
    for module in (sparse_indexer, rocm_aiter_mla_sparse):
        monkeypatch.setattr(module, "get_forward_context", lambda: context)


def _run(op, step, cand=None, write: bool = False) -> torch.Tensor:
    """One call of an indexer op on the step, whose model already wrote the
    K cache. The -7 marks an entry that the op did not write. vLLM's op
    first fills the rows with -1, and the gfx942 op skips that fill because
    its top-k writes every entry."""
    q, weights, lens, tables, kv = step
    out = torch.full((ROWS, TOPK), -7, dtype=torch.int32, device="cuda")
    op(
        torch.empty(ROWS, 1, device="cuda"),
        PREFIX,
        kv,
        q,
        None,
        weights,
        128,
        "ue8m0",
        TOPK,
        DIM,
        MAX_MODEL_LEN,
        int(lens.sum()),
        out,
        skip_k_cache_insert=True,
        compress_ratio=RATIO,
        candidate_blocks=cand,
        candidate_block_size=CAND_BLOCK if cand is not None else 0,
        candidate_write=write,
    )
    return out


def _assert_same_top512(dense, ref, got, ends, cand=None):
    """Got must hold vLLM's columns, or other columns with the same logits
    where logits at the cut are equal. With candidate blocks, every column
    must also lie in one of the row's candidate blocks."""
    for r, e in enumerate(ends):
        g, w = got[r], ref[r]
        assert ((g >= -1) & (g < e)).all(), f"row {r}: an entry outside -1 to {e - 1}"
        gl, wl = g[g >= 0].long(), w[w >= 0].long()
        assert gl.unique().numel() == gl.numel(), f"row {r}: a column twice"
        assert gl.numel() == wl.numel(), (
            f"row {r}: {gl.numel()} columns, vLLM's op {wl.numel()}"
        )
        if cand is not None:
            blocks = cand[r][cand[r] >= 0].long()
            assert torch.isin(gl // CAND_BLOCK, blocks).all(), (
                f"row {r}: a column outside the candidate blocks"
            )
        if set(gl.tolist()) != set(wl.tolist()):
            vg = sorted(dense[r, gl].tolist())
            vw = sorted(dense[r, wl].tolist())
            assert vg == vw, f"row {r}: different top-512 logits"


def _block_scores(row: torch.Tensor, end: int) -> torch.Tensor:
    """VLLM's block scores of one row before its end: the largest logit of a
    block, and +inf for the newest block."""
    nblocks = math.ceil(end / CAND_BLOCK)
    padded = torch.full((nblocks * CAND_BLOCK,), -math.inf, device=row.device)
    padded[:end] = row[:end]
    scores = padded.view(nblocks, CAND_BLOCK).amax(dim=1)
    scores[-1] = math.inf
    return scores.cpu()


def _assert_same_candidates(dense, ref, got, ends):
    """Got must hold vLLM's candidate blocks in any order, or other blocks
    with the same scores where scores at the cut are equal."""
    for r, e in enumerate(ends):
        w, g = ref[r].tolist(), got[r].tolist()
        ws, gs = sorted(v for v in w if v >= 0), sorted(v for v in g if v >= 0)
        assert min(g) >= -1, f"row {r}: an entry below -1"
        assert len(gs) == len(set(gs)), f"row {r}: a block id twice"
        assert w.count(-1) == g.count(-1), f"row {r}: different -1 counts"
        if ws != gs:
            scores = _block_scores(dense[r], e)
            assert sorted(scores[ws].tolist()) == sorted(scores[gs].tolist()), (
                f"row {r}: different chosen block scores"
            )


@pytest.mark.parametrize("ctx", [8192, 131072])
def test_decode_step_matches_vllm_op(gfx942_step, monkeypatch, ctx):
    """At 8192 tokens a row has 1024 blocks of 8, so all of them are
    candidates. At 131072 tokens a row has 16384 blocks, and 2048 of them
    are candidates."""
    from vllm.models.deepseek_v41.amd.ops import sparse_indexer, topk
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        rocm_aiter_sparse_attn_indexer,
        rocm_fp8_paged_mqa_logits,
    )

    step = _make_step(ctx)
    q, weights, lens, tables, kv = step
    _use_metadata(monkeypatch, lens, tables)
    # Record that the gfx942 op ran its decode path and that the gfx942
    # launches of layer 20 and of layers 24 to 36 took their calls, so that
    # the test cannot pass on vLLM's kernels alone.
    taken = []

    def recorder(real, name):
        def record(*args, **kwargs):
            out = real(*args, **kwargs)
            if out is not False:
                taken.append(name)
            return out

        return record

    for module, name in (
        (sparse_indexer, "_decode_step"),
        (topk, "select_candidates"),
        (topk, "candidate_logits_top_k"),
    ):
        monkeypatch.setattr(module, name, recorder(getattr(module, name), name))

    dense = rocm_fp8_paged_mqa_logits(
        q.view(ROWS, 1, HEADS, DIM),
        kv.unsqueeze(-2),
        weights,
        lens,
        tables,
        torch.empty(0, dtype=torch.int32, device="cuda"),
        max_model_len=MAX_MODEL_LEN,
        compress_ratio=RATIO,
    ).clone()
    ends = lens.tolist()
    vllm_op = rocm_aiter_sparse_attn_indexer
    gfx942_op = sparse_indexer.gfx942_sparse_attn_indexer

    # A layer without candidate blocks.
    ref = _run(vllm_op, step)
    got = _run(gfx942_op, step)
    _assert_same_top512(dense, ref, got, ends)

    # Layer 20 writes the candidate blocks and takes its top 512 from all of
    # its logits.
    cand_ref = torch.full((ROWS, CAND_BLOCKS), -7, dtype=torch.int32, device="cuda")
    cand_got = torch.full_like(cand_ref, -7)
    ref = _run(vllm_op, step, cand_ref, write=True)
    got = _run(gfx942_op, step, cand_got, write=True)
    _assert_same_candidates(dense, cand_ref, cand_got, ends)
    _assert_same_top512(dense, ref, got, ends)

    # A layer of 24 to 36 takes its top 512 from the candidate blocks only.
    ref = _run(vllm_op, step, cand_ref.clone())
    got = _run(gfx942_op, step, cand_ref.clone())
    _assert_same_top512(dense, ref, got, ends, cand_ref)

    assert taken == [
        "_decode_step",
        "select_candidates",
        "_decode_step",
        "candidate_logits_top_k",
        "_decode_step",
    ]
