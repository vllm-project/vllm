# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 CP indexer top-k equals the non-CP top-k per global head.

Runs the production CP decode (all-gather index queries, score every head on
this rank's block shard, exchange per-head candidates, merge the owned heads)
across real TP ranks and checks each rank's result against the non-CP decode
kernel on the full context, per global index head.

Contexts are tens of thousands of tokens so the top-16 is a real selection
over hundreds of blocks; short prompts (<= 16 blocks) select every block and
would pass even if the CP merge mixed heads.
"""

import os

import pytest
import ray
import torch

from vllm.distributed.parallel_state import get_tp_group
from vllm.platforms import current_platform

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

# MiniMax-M3's sparse_attention_config.
TOTAL_HEADS = 4
HEAD_DIM = 128
BLOCK = 128
TOPK = 16
INIT_BLOCKS = 0
LOCAL_BLOCKS = 1
MAX_MODEL_LEN = 131072
SEQ_LENS = [49_229, 65_523, 32_773, 81_920, 40_001, 70_000]
_QUERY_LEN_ENV = "VLLM_TEST_M3_CP_QUERY_LEN"


def _global_inputs(query_len: int, device: torch.device):
    """Identical on every rank: generated on CPU from a fixed seed."""
    gen = torch.Generator().manual_seed(0)
    batch = len(SEQ_LENS)
    blocks = [(s + BLOCK - 1) // BLOCK for s in SEQ_LENS]
    pages = sum(blocks)
    cache = torch.randn(pages, BLOCK, HEAD_DIM, generator=gen, dtype=torch.bfloat16)
    perm = torch.randperm(pages, generator=gen).to(torch.int32)
    table = torch.zeros(batch, MAX_MODEL_LEN // BLOCK, dtype=torch.int32)
    start = 0
    for req, n in enumerate(blocks):
        table[req, :n] = perm[start : start + n]
        start += n
    q = torch.randn(
        batch * query_len, TOTAL_HEADS, HEAD_DIM, generator=gen, dtype=torch.bfloat16
    )
    seq_lens = torch.tensor(SEQ_LENS, dtype=torch.int32)
    return (
        q.to(device),
        cache.to(device),
        table.to(device),
        seq_lens.to(device),
    )


def _block_scores(q, cache, table, seq_lens, row, head, query_len):
    """fp32 per-block max score for one query row and head (causal)."""
    req, token = divmod(row, query_len)
    cutoff = int(seq_lens[req]) - query_len + token + 1
    n = (cutoff + BLOCK - 1) // BLOCK
    k = cache[table[req, :n].long()].float()  # [n, 128, D]
    dots = k @ q[row, head].float()  # [n, 128]
    pos = torch.arange(n * BLOCK, device=q.device).view(n, BLOCK)
    return dots.masked_fill(pos >= cutoff, float("-inf")).amax(dim=1)


def _assert_same_topk(got, ref, q, cache, table, seq_lens, heads, query_len):
    """Equal per-row block sets, except swaps across an fp32 near-tie."""
    swaps = 0
    for h_local, head in enumerate(heads):
        for row in range(got.shape[1]):
            a = set(got[h_local, row].tolist()) - {-1}
            b = set(ref[h_local, row].tolist()) - {-1}
            if a == b:
                continue
            s = _block_scores(q, cache, table, seq_lens, row, head, query_len)
            kth = s.topk(TOPK).values[-1]
            for blk in a ^ b:
                assert abs(float(s[blk] - kth)) <= 1e-3 * max(1.0, abs(float(kth))), (
                    f"head {head} row {row}: block {blk} differs and is not a tie"
                )
            swaps += 1
    # Near-ties are rare with random data; a broken merge mismatches everywhere.
    assert swaps <= max(1, got.shape[1] * len(heads) // 100), swaps


@ray.remote(num_gpus=1, max_calls=1)
def _run_cp_topk_matches_non_cp(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    query_len = int(os.environ[_QUERY_LEN_ENV])
    use_aiter = os.environ.get("VLLM_ROCM_USE_AITER") == "1"
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)

        from vllm._aiter_ops import rocm_aiter_ops
        from vllm.models.minimax_m3.amd.indexer_context_parallel import (
            IndexerCPDecode,
        )
        from vllm.models.minimax_m3.amd.ops.index_topk import (
            minimax_m3_index_decode,
        )

        if use_aiter:
            assert rocm_aiter_ops.get_aiter_allreduce() is not None

        q, cache, table, seq_lens = _global_inputs(query_len, device)
        dec = IndexerCPDecode(
            total_heads=TOTAL_HEADS,
            rank=rank,
            world_size=tp_size,
            max_tokens=q.shape[0],
            max_seq_len=MAX_MODEL_LEN,
            topk=TOPK,
            init_blocks=INIT_BLOCKS,
            local_blocks=LOCAL_BLOCKS,
            scale=HEAD_DIM**-0.5,
            group=get_tp_group(),
            device=device,
        )
        own = list(range(dec.head_offset, dec.head_offset + dec.local_heads))
        got = torch.full(
            (dec.local_heads, q.shape[0], TOPK), -2, dtype=torch.int32, device=device
        )
        with torch.inference_mode():
            dec(q[:, own].contiguous(), cache, table, seq_lens, query_len, got)
            ref = minimax_m3_index_decode(
                q,
                cache,
                table,
                seq_lens,
                max(SEQ_LENS),
                TOPK,
                INIT_BLOCKS,
                LOCAL_BLOCKS,
                TOTAL_HEADS,
                query_len,
                query_len,
            )
        torch.accelerator.synchronize()

        # The check must be able to tell heads apart, or mixing would pass.
        distinct = sum(
            set(ref[0, row].tolist()) != set(ref[1, row].tolist())
            for row in range(ref.shape[1])
        )
        assert distinct >= ref.shape[1] * 9 // 10

        _assert_same_topk(got, ref[own], q, cache, table, seq_lens, own, query_len)


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="MiniMax-M3 CP indexer is ROCm-only."
)
@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("query_len", [1, 4])
@pytest.mark.parametrize("use_aiter", [False, True])
def test_cp_topk_matches_non_cp_per_head(
    monkeypatch: pytest.MonkeyPatch, tp_size: int, query_len: int, use_aiter: bool
):
    if torch.accelerator.device_count() < tp_size:
        pytest.skip(f"Need at least {tp_size} GPUs to run the test.")
    if use_aiter:
        pytest.importorskip("aiter")
        # Read when vLLM is imported, so it must be set before ray starts.
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    else:
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "0")
    # Ray workers inherit the driver environment; multi_process_parallel
    # passes no per-test arguments.
    monkeypatch.setenv(_QUERY_LEN_ENV, str(query_len))
    multi_process_parallel(monkeypatch, tp_size, 1, _run_cp_topk_matches_non_cp)
