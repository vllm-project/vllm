# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-process repro for the Q-sharded prefill PCP path.

Runs entirely on CPU via the gloo backend. Mirrors `test_pcp_decode_repro.py`
but exercises the **new** Q-sharded PCP design (Phase 3 of `pcp-real`):
each PCP rank owns 1/N of the prefill Q tokens via DualChunkSwap and
all-gathers K/V across the PCP group so the per-rank attention sees the
full K/V cache.

The bug class this guards against: anything that breaks the round-trip
`partition_inputs -> per-rank local KV -> pcp_kv_allgather_and_restore`
identity. Specifically:

  - If `pcp_kv_allgather_and_restore` gathers the wrong slice size (e.g.,
    using `num_actual_tokens` that includes cudagraph padding rather than
    `padded_total // pcp_world_size`), each rank's contribution gets
    interleaved with cudagraph slack and the restore_idx permutation
    silently produces garbage at real positions.
  - If `partition_inputs` builds `pcp_allgather_restore_idx` such that
    the union of per-rank positions does not cover `[0, padded_total)`
    exactly once, the restored K/V will be missing positions or have
    duplicates — and downstream cache writes via `pad_slot_mapping`
    will land wrong K/V at real slots.
  - If `pad_slot_mapping` places real slots at the wrong positions in
    the padded layout (e.g., off-by-one against the unpad mask), then
    `do_kv_cache_update` will write K/V into mismatched slots and the
    next decode iteration silently attends to wrong K/V.

The test verifies (1) and (2) directly via tensor equality and (3) by
construction (we read the unpad mask + padded slot map and check that
real slots line up with real K/V).

Run::

    .venv/bin/python -m pytest tests/distributed/test_pcp_prefill_qshard.py \
        -v --tb=short --confcutdir=tests/distributed
"""

from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np
import pytest
import torch
import torch.distributed as dist


def _build_partition_state(
    pcp_world_size: int,
    pcp_rank: int,
    num_scheduled: np.ndarray,
    max_padded_num_tokens: int,
):
    """Construct a PCPManager and run partition_inputs.

    Returns the manager plus the partition outputs."""
    from vllm.v1.worker.cp_utils import PCPManager

    pm = PCPManager(
        pcp_world_size=pcp_world_size,
        pcp_rank=pcp_rank,
        max_num_reqs=len(num_scheduled),
        max_padded_num_tokens=max_padded_num_tokens,
        device=torch.device("cpu"),
    )
    num_computed = np.zeros_like(num_scheduled)
    arange = np.arange(max_padded_num_tokens, dtype=np.int64)
    total_real = int(num_scheduled.sum())
    positions_np = np.zeros(total_real, dtype=np.int32)
    req_indices_np = np.zeros(total_real, dtype=np.int64)
    offset = 0
    for r, n in enumerate(num_scheduled):
        positions_np[offset : offset + n] = np.arange(n, dtype=np.int32)
        req_indices_np[offset : offset + n] = r
        offset += n
    local_total, local_pos, local_req, gathered = pm.partition_inputs(
        positions_np,
        req_indices_np,
        num_scheduled.astype(np.int32),
        num_computed,
        arange,
        reorder_batch_threshold=1,
    )
    return pm, local_total, local_pos, local_req, gathered


def _local_kv_for_rank(
    pm,
    pcp_world_size: int,
    pcp_rank: int,
    num_scheduled: np.ndarray,
    global_kv: torch.Tensor,
) -> torch.Tensor:
    """Mirror what a rank would produce as its local K (or V) tensor.

    Each padded layout position p that this rank owns corresponds to a
    global token index; we look it up in `global_kv`. For padding rows
    (position >= num_scheduled[r] for that request), `partition_inputs`
    clamps the gather index to the request start, so we use the same
    clamped fallback — those rows are no-ops in the cache write later.
    """
    ws = pcp_world_size

    # Recompute the rank's padded positions inside each request the same
    # way partition_inputs does, so the test is self-contained and not
    # circular against PCPManager's internal arrays.
    cu_padded_per_req: list[int] = []
    chunks: list[int] = []
    starts: list[int] = []
    offset = 0
    for n in num_scheduled:
        padded = int(np.ceil(n / (2 * ws))) * (2 * ws)
        cu_padded_per_req.append(padded)
        chunks.append(max(padded // (2 * ws), 1))
        starts.append(offset)
        offset += padded

    pcp_total = sum(p // ws for p in cu_padded_per_req)
    assert pm.local_total == pcp_total, (
        f"local_total {pm.local_total} != recomputed pcp_total {pcp_total}"
    )

    local_kv = torch.empty(
        pcp_total, global_kv.shape[-1], dtype=global_kv.dtype
    )
    write_off = 0
    for r, (n, padded) in enumerate(zip(num_scheduled, cu_padded_per_req)):
        ch = chunks[r]
        # Each rank gets two chunks per request: head and tail.
        head_start_padded = starts[r] + pcp_rank * ch
        tail_start_padded = starts[r] + (2 * ws - pcp_rank - 1) * ch
        for chunk_start_padded in (head_start_padded, tail_start_padded):
            for j in range(ch):
                padded_pos = chunk_start_padded + j
                # Local position within request:
                local_pos_in_req = padded_pos - starts[r]
                if local_pos_in_req < n:
                    global_idx = (
                        int(num_scheduled[:r].sum()) + local_pos_in_req
                    )
                else:
                    # Padding: clamp to request start (matches partition_inputs).
                    global_idx = int(num_scheduled[:r].sum())
                local_kv[write_off] = global_kv[global_idx]
                write_off += 1
    assert write_off == pcp_total
    return local_kv


def _worker(env: dict[str, str]) -> None:
    os.environ.update(env)
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    cp_group = dist.new_group(list(range(world_size)))

    # Shim the PCPManager's `get_pcp_group()` call inside
    # `restore_hidden_states` / `pad_slot_mapping`. The function under
    # test (`pcp_kv_allgather_and_restore`) takes the group as an arg,
    # so the GroupCoordinator API only needs `.world_size` and
    # `.all_gather`.
    class _Group:
        def __init__(self, pg):
            self.pg = pg
            self.world_size = dist.get_world_size(pg)
            self.rank_in_group = dist.get_rank(pg)

        def all_gather(self, t: torch.Tensor, dim: int = 0) -> torch.Tensor:
            assert dim == 0
            chunks = [torch.empty_like(t) for _ in range(self.world_size)]
            dist.all_gather(chunks, t.contiguous(), group=self.pg)
            return torch.cat(chunks, dim=0)

    pcp_group = _Group(cp_group)

    torch.manual_seed(0)
    np.random.seed(0)

    # Test config: one request of length divisible by 2*ws (no padding),
    # one request that requires padding (length not divisible by 2*ws).
    num_scheduled = np.array([8 * world_size, 8 * world_size + 3], dtype=np.int32)
    total_real = int(num_scheduled.sum())
    head_dim = 16
    global_K = torch.arange(total_real * head_dim, dtype=torch.float32).reshape(
        total_real, head_dim
    )
    global_V = global_K * 0.5 - 1.0  # different shape from K to spot swaps

    pm, local_total, _, _, _ = _build_partition_state(
        pcp_world_size=world_size,
        pcp_rank=rank,
        num_scheduled=num_scheduled,
        max_padded_num_tokens=max(64, total_real * world_size),
    )

    local_K = _local_kv_for_rank(pm, world_size, rank, num_scheduled, global_K)
    local_V = _local_kv_for_rank(pm, world_size, rank, num_scheduled, global_V)
    assert local_K.shape[0] == local_total

    from vllm.v1.attention.backends.utils import pcp_kv_allgather_and_restore

    restore_idx_np = pm.pcp_allgather_restore_idx.np[: pm.padded_total]
    restore_idx = torch.from_numpy(restore_idx_np.copy()).long()

    # Simulate cudagraph padding by allocating a larger buffer and only
    # writing the real K/V into the prefix.
    cg_padded = local_total + 5
    local_K_cg = torch.zeros(cg_padded, head_dim, dtype=local_K.dtype)
    local_V_cg = torch.zeros(cg_padded, head_dim, dtype=local_V.dtype)
    local_K_cg[:local_total] = local_K
    local_V_cg[:local_total] = local_V
    # Poison the slack with values that, if accidentally gathered, would
    # produce a visible mismatch against `global_K`.
    local_K_cg[local_total:] = -9999.0
    local_V_cg[local_total:] = -9999.0

    K_restored, V_restored = pcp_kv_allgather_and_restore(
        local_K_cg,
        local_V_cg,
        num_actual_tokens=cg_padded,
        pcp_allgather_restore_idx=restore_idx,
        pcp_group=pcp_group,
    )

    assert K_restored.shape == (pm.padded_total, head_dim), (
        f"[rank {rank}] K_restored shape {K_restored.shape} != "
        f"({pm.padded_total}, {head_dim})"
    )

    # 1) Real positions must hold the exact global K/V.
    mask_real = torch.from_numpy(pm.pcp_unpad_mask[: pm.padded_total].copy())
    assert int(mask_real.sum()) == total_real, (
        f"[rank {rank}] unpad mask covers {int(mask_real.sum())} positions "
        f"but expected {total_real}"
    )

    K_real_restored = K_restored[mask_real]
    V_real_restored = V_restored[mask_real]
    # The unpadded positions in the padded layout, in order, correspond
    # exactly to global token indices [0, total_real).
    assert torch.allclose(K_real_restored, global_K), (
        f"[rank {rank}] K mismatch at real positions: "
        f"max abs diff = {(K_real_restored - global_K).abs().max().item()}"
    )
    assert torch.allclose(V_real_restored, global_V), (
        f"[rank {rank}] V mismatch at real positions"
    )

    # 2) pad_slot_mapping must place a sequential global slot map at the
    # unpadded positions and -1 elsewhere — this is exactly what
    # do_kv_cache_update needs so cache writes hit the right slots and
    # padding writes are no-ops.
    global_slot_map = torch.arange(total_real, dtype=torch.int64) + 1000
    padded_slot = pm.pad_slot_mapping(global_slot_map)
    assert padded_slot.shape[0] == pm.padded_total
    assert torch.equal(padded_slot[mask_real], global_slot_map), (
        f"[rank {rank}] pad_slot_mapping misplaced real slots"
    )
    assert (padded_slot[~mask_real] == -1).all(), (
        f"[rank {rank}] pad_slot_mapping leaked non-(-1) slots into padding"
    )

    # 3) End-to-end: simulate the cache write. Allocate a global "cache"
    # of size total_real, write K_restored into padded_slot positions,
    # then verify the cache equals global_K. This is the property that
    # decode-after-prefill correctness relies on.
    cache_K = torch.zeros_like(global_K)
    cache_V = torch.zeros_like(global_V)
    for p in range(pm.padded_total):
        s = padded_slot[p].item() - 1000  # global_slot_map offset
        if s < 0:
            continue
        cache_K[s] = K_restored[p]
        cache_V[s] = V_restored[p]
    assert torch.allclose(cache_K, global_K), (
        f"[rank {rank}] post-write cache does not match global K"
    )
    assert torch.allclose(cache_V, global_V), (
        f"[rank {rank}] post-write cache does not match global V"
    )

    dist.destroy_process_group()


def _spawn(world_size: int) -> None:
    mp.set_start_method("spawn", force=True)
    port = os.environ.get("TEST_DIST_PORT", "29503")
    procs: list[mp.Process] = []
    for r in range(world_size):
        env = {
            "RANK": str(r),
            "LOCAL_RANK": str(r),
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
def test_pcp_prefill_qshard_kv_roundtrip(world_size: int):
    """Q-sharded prefill: partition + per-rank local K/V + all-gather +
    restore must reproduce the global K/V at every real position, and
    pad_slot_mapping must direct cache writes to those positions and
    nowhere else.

    This is the property that makes decode-after-prefill correct under
    PCP: every prior token's K/V lands in the cache at the slot that
    decode-side block_table lookups will read from.
    """
    _spawn(world_size)
