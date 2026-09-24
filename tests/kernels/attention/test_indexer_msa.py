# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for MiniMax M3 MSA indexer."""

import pytest

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("the MiniMax-M3 MSA indexer is AMD-only", allow_module_level=True)

import torch  # noqa: E402

from vllm.models.minimax_m3.amd.indexer_msa import (  # noqa: E402
    MAX_SUPPORTED_BLOCKS,
    MSA_INDEX_HEAD_DIM,
    MSA_SCORE_TYPE,
    MSA_SPARSE_BLOCK_SIZE,
    MSA_TOPK_BLOCKS,
    SLOTS_MAX,
    WAVE_SIZE,
    _score_buffer,
    candidate_keys,
    merge_and_emit,
    shard_scores,
)
from vllm.models.minimax_m3.amd.ops.sparse_pa import ASM_PAGE_SIZE  # noqa: E402
from vllm.platforms.rocm import on_gfx950  # noqa: E402
from vllm.utils.torch_utils import set_random_seed  # noqa: E402

HEAD_DIM = MSA_INDEX_HEAD_DIM
BLOCK = MSA_SPARSE_BLOCK_SIZE
TOPK = MSA_TOPK_BLOCKS
PAGES_PER_BLOCK = BLOCK // ASM_PAGE_SIZE
FP8 = current_platform.fp8_dtype()

requires_gfx950 = pytest.mark.skipif(
    not on_gfx950(), reason="the MSA indexer needs gfx950 for the fp8 MFMA"
)


_ADMITTED = dict(
    topk_blocks=MSA_TOPK_BLOCKS,
    sparse_block_size=MSA_SPARSE_BLOCK_SIZE,
    num_index_heads=1,
    index_head_dim=MSA_INDEX_HEAD_DIM,
    indexer_kv_dtype="fp8",
    max_model_len=8192,
    score_type=MSA_SCORE_TYPE,
)

# The longest context the top-k has register slots for.
_MAX_LEN = MAX_SUPPORTED_BLOCKS * MSA_SPARSE_BLOCK_SIZE


@pytest.fixture
def aiter_msa_indexer_gate(monkeypatch):
    import vllm.models.minimax_m3.amd.indexer_msa as indexer_msa_mod
    import vllm.platforms.rocm as rocm_mod

    def probe(rocm=True, gfx950=True, aiter_attend=True, **overrides):
        monkeypatch.setattr(indexer_msa_mod.current_platform, "is_rocm", lambda: rocm)
        monkeypatch.setattr(
            indexer_msa_mod,
            "_minimax_m3_aiter_sparse_pa_requested",
            lambda: aiter_attend,
        )
        monkeypatch.setattr(rocm_mod, "on_gfx950", lambda: gfx950)
        return indexer_msa_mod.msa_indexer_unsupported_reason(
            **{**_ADMITTED, **overrides}
        )

    return probe


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({}, None),
        ({"rocm": False}, "ROCm"),
        ({"gfx950": False}, "gfx950"),
        ({"aiter_attend": False}, "AITER sparse PA attend"),
        ({"indexer_kv_dtype": "bf16"}, "fp8 e4m3 index cache"),
        ({"score_type": "sum"}, "score_type"),
        ({"topk_blocks": MSA_TOPK_BLOCKS * 2}, "topk_blocks"),
        ({"sparse_block_size": MSA_SPARSE_BLOCK_SIZE // 2}, "sparse_block_size"),
        ({"index_head_dim": MSA_INDEX_HEAD_DIM // 2}, "index_head_dim"),
        ({"max_model_len": _MAX_LEN}, None),
        ({"max_model_len": _MAX_LEN + 1}, "the top-k is compiled for"),
    ],
)
def test_msa_indexer_gate_rejects_unsupported_configs(
    aiter_msa_indexer_gate, override, expected
):
    reason = aiter_msa_indexer_gate(**override)
    if expected is None:
        assert reason is None, f"{override} should be admitted, got {reason!r}"
    else:
        assert reason is not None, f"{override} should be refused"
        assert expected in reason, f"{override} refused for the wrong reason: {reason}"


def test_msa_slot_cap_matches_what_aiter_compiled():
    msa_block_select = pytest.importorskip("aiter.ops.msa_block_select")

    assert SLOTS_MAX == msa_block_select.TOPK_MAX_SLOTS
    assert WAVE_SIZE == msa_block_select.WAVE_SIZE

    def aiter_accepts(blocks: int) -> bool:
        strips = -(-blocks // WAVE_SIZE)
        return 1 << (strips - 1).bit_length() <= SLOTS_MAX

    assert aiter_accepts(MAX_SUPPORTED_BLOCKS)
    assert not aiter_accepts(MAX_SUPPORTED_BLOCKS + 1)


def _seq_len_patterns(batch: int, max_block: int, query_len: int) -> list[torch.Tensor]:
    uniform = torch.full((batch,), max_block * BLOCK, device="cuda", dtype=torch.int32)
    one_active = torch.full((batch,), query_len, device="cuda", dtype=torch.int32)
    one_active[0] = max(1, max_block // 2) * BLOCK
    # Partial tail blocks, which is the case the emitted sparse_ctx counts
    # differently from every full block ahead of it.
    non_aligned = torch.tensor(
        [
            max(query_len, (max_block - 1) * BLOCK + request % 3 + 1)
            for request in range(batch)
        ],
        device="cuda",
        dtype=torch.int32,
    )
    return [uniform, one_active, non_aligned]


@requires_gfx950
@pytest.mark.parametrize("batch", [4, 8, 16, 32])
@pytest.mark.parametrize("ctx", [65536, 131072])
def test_msa_indexer_matches_triton(batch: int, ctx: int):
    from aiter.ops.msa_attention import (
        pa_sparse_block_score_decode,
        pa_sparse_block_topk,
    )

    set_random_seed(0)
    # Plain decode. The speculative shape is covered by the CP test below, so
    # pinning it here keeps the batch x context sweep to eight cases.
    query_len = 1
    init_blocks, local_blocks = 1, 2
    max_block = ctx // BLOCK
    tokens = batch * query_len
    npages = batch * max_block

    block_table = torch.arange(npages, device="cuda", dtype=torch.int32).reshape(
        batch, max_block
    )
    idx_q = torch.randn(tokens, 1, HEAD_DIM, device="cuda").to(FP8)
    index_kv_cache = torch.randn(npages, BLOCK, HEAD_DIM, device="cuda").to(FP8)
    seq_lens = torch.empty(batch, device="cuda", dtype=torch.int32)
    # The attend's table, not the indexer's: one rebased page id per block,
    # numbered disjointly so a path that resolved the selection through the
    # wrong table cannot match by accident.
    page16_table = torch.arange(
        10_000, 10_000 + npages, device="cuda", dtype=torch.int32
    ).reshape(batch, max_block)

    aiter_topk, triton_topk = (
        torch.full((1, tokens, TOPK), -7, device="cuda", dtype=torch.int32)
        for _ in range(2)
    )
    aiter_bt, triton_bt = (
        torch.full(
            (tokens, TOPK * PAGES_PER_BLOCK), -7, device="cuda", dtype=torch.int32
        )
        for _ in range(2)
    )
    aiter_ctx, triton_ctx = (
        torch.full((tokens,), -7, device="cuda", dtype=torch.int32) for _ in range(2)
    )
    score = _score_buffer(1, tokens, ctx, BLOCK, idx_q.device)

    def launch_aiter() -> None:
        pa_sparse_block_score_decode(
            idx_q,
            index_kv_cache,
            score,
            block_table,
            seq_lens,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            query_len=query_len,
            max_seq_len=ctx,
        )
        pa_sparse_block_topk(
            score,
            aiter_topk,
            page16_table,
            seq_lens,
            sparse_bt=aiter_bt,
            sparse_ctx=aiter_ctx,
            max_seq_len=ctx,
            block_size=BLOCK,
            query_len=query_len,
            num_kv_heads=1,
            pages_per_block=PAGES_PER_BLOCK,
        )

    def launch_triton() -> None:
        keys = candidate_keys(
            shard_scores(
                idx_q,
                index_kv_cache,
                block_table,
                seq_lens,
                global_blocks=max_block,
                rank=0,
                world=1,
                query_len=query_len,
                scale=HEAD_DIM**-0.5,
            ),
            seq_lens,
            topk=TOPK,
            rank=0,
            world=1,
            query_len=query_len,
            global_blocks=max_block,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
        )
        merge_and_emit(
            keys.unsqueeze(0),
            triton_topk,
            seq_lens,
            page16_table,
            triton_bt,
            triton_ctx,
            query_len=query_len,
            num_kv_heads=1,
            pages_per_block=PAGES_PER_BLOCK,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
        )

    def assert_same_selection() -> None:
        aiter_rows = aiter_topk.reshape(-1, TOPK).tolist()
        triton_rows = triton_topk.reshape(-1, TOPK).tolist()
        for row, (want, got) in enumerate(zip(aiter_rows, triton_rows)):
            want_set = {block for block in want if block >= 0}
            got_set = {block for block in got if block >= 0}
            assert got_set == want_set, (
                f"row {row}: Triton added {sorted(got_set - want_set)} and "
                f"dropped {sorted(want_set - got_set)}"
            )
        # The page table is what the attend actually reads, so a matching
        # selection with a differing context bound would still be a bug.
        assert torch.equal(triton_ctx, aiter_ctx)

    patterns = _seq_len_patterns(batch, max_block, query_len)
    seq_lens.copy_(patterns[0])
    launch_aiter()
    launch_triton()
    assert_same_selection()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch_triton()
    addresses = (
        seq_lens.data_ptr(),
        triton_topk.data_ptr(),
        triton_bt.data_ptr(),
        triton_ctx.data_ptr(),
    )
    replayed = []
    for pattern in patterns:
        seq_lens.copy_(pattern)
        launch_aiter()
        graph.replay()
        assert_same_selection()
        replayed.append(triton_topk.clone())
        assert addresses == (
            seq_lens.data_ptr(),
            triton_topk.data_ptr(),
            triton_bt.data_ptr(),
            triton_ctx.data_ptr(),
        )
    # A graph that ignored seq_lens outright would agree with AITER on the
    # pattern it was captured on and never be asked again, so the replays have
    # to be shown to differ from each other for the loop above to mean
    # anything. The short pattern is the one that moves the selection; the
    # non-aligned one only moves the tail, which sparse_ctx carries.
    assert not torch.equal(replayed[0], replayed[1])


@requires_gfx950
@pytest.mark.parametrize("batch", [4, 8, 16, 32])
@pytest.mark.parametrize("ctx", [65536, 131072])
def test_cp_selection_is_independent_of_the_block_bound(batch: int, ctx: int):
    set_random_seed(0)
    # Speculative decode, which is the shape that gives the rows of one
    # request different causal bounds -- the thing a bound taken per request
    # rather than per row would get wrong once the block count changed.
    query_len = 4
    world, heads = 4, 4
    init_blocks, local_blocks = 1, 2
    max_block = ctx // BLOCK
    tokens = batch * query_len
    npages = batch * max_block
    # Four times the batch's own blocks, which is the shape of the gap when a
    # graph captured at max_model_len replays on a quarter-length batch.
    loose_block = max_block * 4

    block_table = torch.arange(npages, device="cuda", dtype=torch.int32).reshape(
        batch, max_block
    )
    idx_q = torch.randn(tokens, heads, HEAD_DIM, device="cuda").to(FP8)
    index_kv_cache = torch.randn(npages, BLOCK, HEAD_DIM, device="cuda").to(FP8)
    seq_lens = torch.empty(batch, device="cuda", dtype=torch.int32)
    page16_table = torch.arange(
        10_000, 10_000 + npages, device="cuda", dtype=torch.int32
    ).reshape(batch, max_block)

    owned = heads // world

    def run(global_blocks: int):
        # torch.stack stands in for exchange_candidates: the gather hands each
        # rank every shard's candidates for the heads it owns, which is this
        # stack sliced on the head axis. The collective cannot run here, since
        # these tests are one process.
        gathered = torch.stack(
            [
                candidate_keys(
                    shard_scores(
                        idx_q,
                        index_kv_cache,
                        block_table,
                        seq_lens,
                        global_blocks=global_blocks,
                        rank=rank,
                        world=world,
                        query_len=query_len,
                        scale=HEAD_DIM**-0.5,
                    ),
                    seq_lens,
                    topk=TOPK,
                    rank=rank,
                    world=world,
                    query_len=query_len,
                    global_blocks=global_blocks,
                    init_blocks=init_blocks,
                    local_blocks=local_blocks,
                )
                for rank in range(world)
            ]
        )
        out = []
        for rank in range(world):
            topk_idx = torch.full(
                (owned, tokens, TOPK), -7, device="cuda", dtype=torch.int32
            )
            sparse_bt = torch.full(
                (tokens * owned, TOPK * PAGES_PER_BLOCK),
                -7,
                device="cuda",
                dtype=torch.int32,
            )
            sparse_ctx = torch.full(
                (tokens * owned,), -7, device="cuda", dtype=torch.int32
            )
            lo = rank * owned
            merge_and_emit(
                gathered[:, lo : lo + owned].contiguous(),
                topk_idx,
                seq_lens,
                page16_table,
                sparse_bt,
                sparse_ctx,
                query_len=query_len,
                num_kv_heads=owned,
                pages_per_block=PAGES_PER_BLOCK,
                init_blocks=init_blocks,
                local_blocks=local_blocks,
            )
            out.append((topk_idx, sparse_bt, sparse_ctx))
        return out

    for pattern in _seq_len_patterns(batch, max_block, query_len):
        seq_lens.copy_(pattern)
        tight = run(max_block)
        loose = run(loose_block)
        for rank, (want, got) in enumerate(zip(tight, loose)):
            for name, a, b in zip(("topk", "sparse_bt", "sparse_ctx"), want, got):
                assert torch.equal(a, b), f"rank {rank} {name} moved with the bound"
