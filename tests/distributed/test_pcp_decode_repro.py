# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-process repro for the decode-side PCP path under TP=1.

Runs entirely on CPU via the gloo backend — no GPUs required. This is
the regression check for the bug that hung the Kimi-K2.5 TP=1+PCP=4
agentx bench (NCCL AllGather watchdog timeout after ~10 min of
sustained decode load).

The bug was that under TP=1+PCP=N, the decode-side CP logic was
AllGathering Q across the PCP group (a holdover from the legacy DCP
case where Q had H/TP heads and AllGather collected them). With TP=1,
Q already has all H heads, so the AllGather just 4x-replicated them,
poisoning the subsequent merge.

The fix: under TP=1+PCP=N, skip the Q-allgather and use AllReduce-
merge (cp_lse_ag_out_ar semantics) instead of reduce-scatter. This
test asserts that the fixed path produces output equal to single-rank
attention over the full K cache.

Run:
    .venv/bin/python -m pytest tests/distributed/test_pcp_decode_repro.py \
        -v --tb=short --confcutdir=tests/distributed
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os

import pytest
import torch
import torch.distributed as dist


def _compressed_attention(
    q: torch.Tensor,                # [B, H, D]
    kv_c_k_pe: torch.Tensor,        # [Sk, D]   (D == kv_lora_rank + qk_rope)
    kv_lora_rank: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compressed-space attention (CPU port of the TritonMLA decode kernel).

    The W_UK up-projection is assumed pre-absorbed into Q. Output is in
    compressed latent space: [B, H, kv_lora_rank]. LSE: [B, H].
    """
    scores = torch.einsum("bhd,kd->bhk", q.float(), kv_c_k_pe.float()) * softmax_scale
    lse = torch.logsumexp(scores, dim=-1)
    weights = torch.exp(scores - lse.unsqueeze(-1))
    v = kv_c_k_pe[:, :kv_lora_rank].float()
    out = torch.einsum("bhk,kd->bhd", weights, v)
    return out, lse


def _pcp_decode_under_tp1(
    rank: int,
    world_size: int,
    q_full: torch.Tensor,
    kv_c_k_pe_full: torch.Tensor,
    kv_lora_rank: int,
    softmax_scale: float,
    cp_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Mirror the post-fix decode CP path from mla_attention.py forward_impl.

    Under TP=1+PCP=N:
      1. Q already has all H heads on every rank — NO Q-allgather.
      2. Each rank computes attention against its 1/N K shard.
      3. AllReduce-merge using LSE-weighted online softmax, so every
         rank ends up with the global full-K attention output for all
         H heads (then fed to TP=1's unsharded v_up_proj).
    """
    # KV cache is sequence-sharded across CP ranks via slot mapping
    # interleave (block_table.py: total_cp_rank = pcp_rank * 1 + 0).
    kv_local = kv_c_k_pe_full[rank::world_size]

    # Partial attention against this rank's K shard.
    out_local, lse_local = _compressed_attention(
        q_full, kv_local, kv_lora_rank, softmax_scale
    )

    # AllReduce-merge (cp_lse_ag_out_ar equivalent): gather all ranks'
    # LSE to compute the global normalization, weight local out by
    # exp(lse_local - global_lse), then AllReduce-sum.
    lse_gathered = [torch.empty_like(lse_local) for _ in range(world_size)]
    dist.all_gather(lse_gathered, lse_local, group=cp_group)
    lse_stacked = torch.stack(lse_gathered, dim=0)         # [W, B, H]
    global_lse = torch.logsumexp(lse_stacked, dim=0)       # [B, H]
    weights = torch.exp(lse_local - global_lse).unsqueeze(-1)
    out_weighted = out_local * weights
    dist.all_reduce(out_weighted, group=cp_group)
    return out_weighted


def _worker(env: dict[str, str]) -> None:
    os.environ.update(env)
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    cp_group = dist.new_group(list(range(world_size)))

    torch.manual_seed(0)
    B, H = 4, 16
    qk_nope, qk_rope = 128, 64
    kv_lora_rank = qk_nope
    qk_head_dim = qk_nope + qk_rope
    Sk = world_size * 32
    softmax_scale = 1.0 / math.sqrt(qk_head_dim)

    q_full = torch.randn(B, H, qk_head_dim)
    kv_c_k_pe_full = torch.randn(Sk, kv_lora_rank + qk_rope)

    # Reference: single-rank attention over the full K.
    ref_out, _ = _compressed_attention(
        q_full, kv_c_k_pe_full, kv_lora_rank, softmax_scale
    )

    # Post-fix path under TP=1+PCP=N.
    cp_out = _pcp_decode_under_tp1(
        rank, world_size,
        q_full=q_full, kv_c_k_pe_full=kv_c_k_pe_full,
        kv_lora_rank=kv_lora_rank, softmax_scale=softmax_scale,
        cp_group=cp_group,
    )

    assert cp_out.shape == ref_out.shape, (
        f"[rank {rank}] cp_out shape {cp_out.shape} != reference "
        f"{ref_out.shape} — TP=1+PCP path should preserve heads dim"
    )
    diff = (cp_out - ref_out).abs().max().item()
    assert diff < 1e-4, (
        f"[rank {rank}] PCP merge diverges from single-rank reference: "
        f"max diff = {diff}"
    )

    dist.destroy_process_group()


def _spawn(world_size: int) -> None:
    mp.set_start_method("spawn", force=True)
    port = os.environ.get("TEST_DIST_PORT", "29501")
    procs: list[mp.Process] = []
    for rank in range(world_size):
        env = {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": port,
        }
        p = mp.Process(target=_worker, args=(env,))
        procs.append(p)
        p.start()

    for p in procs:
        p.join(timeout=120)
    failures: list[str] = []
    for i, p in enumerate(procs):
        if p.is_alive():
            p.kill()
            p.join()
            failures.append(f"worker {i} timed out")
        elif p.exitcode != 0:
            failures.append(f"worker {i} exited {p.exitcode}")
    assert not failures, "; ".join(failures)


@pytest.mark.parametrize("world_size", [2, 4])
def test_pcp_decode_under_tp1_matches_single_rank(world_size: int):
    """Verifies the post-fix decode CP path under TP=1+PCP=N.

    Spawns `world_size` processes on gloo (CPU), shards the K cache
    across them via interleaved slicing (mirrors block_table slot
    mapping), runs partial compressed-space attention per rank, then
    AllReduce-merges via LSE-weighted online softmax. Result must
    equal single-rank attention over the full K within FP32 tolerance.
    """
    _spawn(world_size)
