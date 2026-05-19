"""Minimal repro: gloo cross-PG + DeviceMesh + distribute_tensor on all-Replicate.

Mimics the etha planning loop: 8 ranks in a gloo PG, with 4 (trainer) + 4
(inference) split into 2D meshes that don't span all ranks. Iterate over
several pairs, each calling get_m2m_map equivalent. The 4th pair uses
all-Replicate placements.

Run with:
  for r in 0 1 2 3 4 5 6 7; do RANK=$r WORLD=8 MASTER_ADDR=127.0.0.1 \\
    MASTER_PORT=29500 .venv/bin/python tmp_repro_etha_hang.py & done
"""

import contextlib
import datetime
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.distributed import DeviceMesh
from torch.distributed.distributed_c10d import (
    _get_default_group,
    _new_process_group_helper,
    _update_default_pg,
)
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard


@contextlib.contextmanager
def _temporary_default_pg(pg):
    saved = _get_default_group()
    _update_default_pg(pg)
    try:
        yield
    finally:
        _update_default_pg(saved)


def log(rank, msg):
    print(f"[rank={rank}] {msg}", flush=True)


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD"])

    # Need a default PG before we can build subgroups via the helper —
    # this mirrors what the trainer/vllm workers do in real runs. The
    # ETHA_BACKEND env knob lets us toggle the parent PG backend on each
    # rank so we can hunt for the real-run hang.
    backend = os.environ.get("ETHA_BACKEND", "gloo")
    if "nccl" in backend:
        # Pin this rank to one CUDA device for NCCL init.
        torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    # Bootstrap a SECOND PG via a fresh TCPStore + gloo so we can mirror the
    # `_build_cross_pg(store, ...)` shape (parent PG was just init'd above).
    store = dist.TCPStore(
        host_name=os.environ["MASTER_ADDR"],
        port=int(os.environ["MASTER_PORT"]) + 1,
        world_size=world_size,
        is_master=(rank == 0),
        timeout=datetime.timedelta(seconds=60),
    )
    cross_pg, _ = _new_process_group_helper(
        group_size=world_size,
        group_rank=rank,
        global_ranks_in_group=list(range(world_size)),
        backend="gloo",
        store=store,
        group_name="repro_planner",
        timeout=datetime.timedelta(seconds=60),
    )

    pairs = [
        # (name, src_mesh_tensor, src_pl, tgt_mesh_tensor, tgt_pl)
        (
            "embed_tokens",
            torch.tensor([[0, 1], [2, 3]]),
            (Replicate(), Shard(0)),
            torch.tensor([[4, 5], [6, 7]]),
            (Replicate(), Shard(0)),
        ),
        (
            "experts_down",
            torch.arange(4).view(1, 2, 2),  # ranks 0..3 reshaped as (1,2,2)
            (Replicate(), Shard(0), Shard(0)),
            torch.arange(4, 8).view(4),  # ranks 4..7 flat
            (Shard(0),),
        ),
        (
            "experts_gate_up",
            torch.arange(4).view(1, 2, 2),
            (Replicate(), Shard(0), Shard(0)),
            torch.arange(4, 8).view(4),
            (Shard(0),),
        ),
        (
            "layernorm",  # ← the hanging one
            torch.tensor([[0, 1], [2, 3]]),
            (Replicate(), Replicate()),
            torch.tensor([[4, 5], [6, 7]]),
            (Replicate(), Replicate()),
        ),
    ]

    with _temporary_default_pg(cross_pg):
        for handler, src_mesh_t, src_pl, tgt_mesh_t, tgt_pl in pairs:
            log(rank, f"=== pair {handler}: START ===")

            src_mesh = DeviceMesh("cpu", src_mesh_t)
            tgt_mesh = DeviceMesh("cpu", tgt_mesh_t)
            log(rank, f"=== pair {handler}: meshes built ===")

            src_ranks = src_mesh_t.flatten().tolist()
            tgt_ranks = tgt_mesh_t.flatten().tolist()
            # Tiny middle tensor: shape (1,) for all-Replicate
            shape = (1,) if all(isinstance(p, Replicate) for p in src_pl) else (2,)

            t0 = time.monotonic()
            reqs = []
            full_tensor_restored = None
            if rank in src_ranks:
                middle = torch.zeros(shape, device="cpu")
                log(rank, f"  {handler}: pre distribute_tensor (src)")
                dt = distribute_tensor(middle, src_mesh, src_pl)
                log(rank, f"  {handler}: post distribute_tensor (src)")
                local = dt.to_local()
                local.fill_(float(rank))
                log(rank, f"  {handler}: pre full_tensor")
                full_tensor_restored = dt.full_tensor()
                log(rank, f"  {handler}: post full_tensor")

                src_idx = src_ranks.index(rank)
                targets = [
                    tgt_ranks[ti]
                    for ti in range(src_idx, len(tgt_ranks), len(src_ranks))
                ]
                log(rank, f"  {handler}: isend to {targets}")
                for tr in targets:
                    reqs.append(dist.isend(full_tensor_restored, dst=tr))
            elif rank in tgt_ranks:
                full_tensor_restored = torch.empty(shape, device="cpu")
                tidx = tgt_ranks.index(rank)
                source_rank = src_ranks[tidx % len(src_ranks)]
                log(rank, f"  {handler}: irecv from {source_rank}")
                reqs.append(dist.irecv(full_tensor_restored, src=source_rank))

            log(rank, f"  {handler}: waiting on {len(reqs)} reqs")
            for r in reqs:
                r.wait()
            log(rank, f"  {handler}: reqs done")

            if rank in tgt_ranks:
                log(rank, f"  {handler}: pre distribute_tensor (tgt)")
                dt_tgt = distribute_tensor(
                    full_tensor_restored, tgt_mesh, tgt_pl, src_data_rank=None
                )
                log(rank, f"  {handler}: post distribute_tensor (tgt)")
                _ = dt_tgt.to_local()

            # Mirror the all_gather_object at the end of get_m2m_map
            log(rank, f"  {handler}: pre all_gather_object")
            m2m_local = {}
            holder = [None] * world_size
            dist.all_gather_object(holder, m2m_local, group=cross_pg)
            log(rank, f"  {handler}: post all_gather_object")

            log(rank, f"=== pair {handler}: DONE in {time.monotonic()-t0:.3f}s ===")

    log(rank, "all pairs done")
    dist.destroy_process_group(cross_pg)


if __name__ == "__main__":
    main()
