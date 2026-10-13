# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph replay and fault injection for the GLM-5 MonoKernel (four gfx950 GPUs).

* Graphs captured at each width replay 300 times with bit-identical output.
* A rank stalled past the poll limit (``VLLM_MONO_FAULT_INJECT``) does not hang
  the others: their waits expire, the expired stages are reported on every
  waiting rank and the stalled rank's peers flag it on rank 0.

Run under ``timeout``: ``timeout -s KILL 900 python3 test_glm_mono_safety.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from aiter.jit.utils.chip_info import get_gfx_runtime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_glm_mono_moe import FP8, TOPK, make_layer, step_inputs  # noqa: E402

from vllm.models.deepseek_v32.amd.ops.mono import (  # noqa: E402
    AttentionWeight,
    Glm5MonoKernel,
    KvCacheLayout,
    glm5_mono_launch_rows,
    glm5_tp_config,
    prepare_glm5_weights,
)

REPLAYS = 300
POLL_LIMIT = 20_000_000
FAULT_POLL_LIMIT = 1_000_000


def _setup(rank, tp, port, layers, ctx):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", rank=rank, world_size=tp)
    group = dist.new_group(list(range(tp)), backend="gloo")
    cfg = glm5_tp_config(tp)
    torch.manual_seed(1234 + rank)
    weights = [make_layer(cfg, rank, tp, dev) for _ in range(layers)]
    prepared = [prepare_glm5_weights(w, AttentionWeight.FP8_BLOCK128) for w in weights]
    kv = (torch.randn(8 * ctx, cfg.kv_lora + cfg.pe_dim, device=dev) * 0.1).to(FP8)
    angles = torch.rand(ctx, cfg.pe_dim // 2, device=dev) * 6.28
    return dict(
        dev=dev,
        group=group,
        cfg=cfg,
        weights=weights,
        prepared=prepared,
        kv=kv,
        cos=angles.cos().to(torch.bfloat16),
        sin=angles.sin().to(torch.bfloat16),
        cur_pos=torch.zeros(1, dtype=torch.int32, device=dev),
    )


def _op(env, rank, tp, chunk, layers, poll_limit):
    return Glm5MonoKernel(
        env["weights"][0],
        chunk,
        rank=rank,
        npes=tp,
        group=env["group"],
        topk=TOPK,
        launches_per_step=layers,
        attention_weight=AttentionWeight.FP8_BLOCK128,
        kv_cache_layout=KvCacheLayout.ATOM,
        kv_cache_dtype="fp8",
        prepared_weights=env["prepared"][0],
        native_fp4_mfma=True,
        poll_limit=poll_limit,
    )


def _stepper(env, op, rows, ctx, layers):
    padded, _ = glm5_mono_launch_rows(rows)
    gen = torch.Generator().manual_seed(42)
    positions, slots, indptr, indices = step_inputs(rows, padded, ctx, env["dev"], gen)
    h = (torch.randn(padded, env["cfg"].hidden, device=env["dev"]) * 0.5).to(
        torch.bfloat16
    )
    bufs = [h.clone(), torch.empty_like(h)]

    def step():
        for i, (w, p) in enumerate(zip(env["weights"], env["prepared"])):
            op.W, op.packed = w, p
            op.forward(
                bufs[i % 2],
                env["cur_pos"],
                env["kv"],
                env["kv"],
                indices,
                env["cos"],
                env["sin"],
                x_out=bufs[(i + 1) % 2],
                layer=i,
                advance=False,
                positions=positions,
                slot_mapping=slots,
                sparse_kv_indptr=indptr,
            )
        op.advance_step()

    return h, bufs, bufs[layers % 2], step


def replay_worker(rank, args, port, failures):
    env = _setup(rank, args.tp, port, args.layers, args.ctx)
    for rows in (2, 4, 8):
        _, chunk = glm5_mono_launch_rows(rows)
        op = _op(env, rank, args.tp, chunk, args.layers, POLL_LIMIT)
        h, bufs, last, step = _stepper(env, op, rows, args.ctx, args.layers)
        step()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        bufs[0].copy_(h)
        graph.replay()
        first = last[:rows].clone()
        same = True
        for _ in range(REPLAYS):
            bufs[0].copy_(h)
            graph.replay()
            same &= torch.equal(last[:rows], first)
        torch.cuda.synchronize()
        ok = bool(same) and not op.poll_errors()
        if rank == 0:
            print(
                f"[{'PASS' if ok else 'FAIL'}] {REPLAYS} replays rows={rows}",
                flush=True,
            )
        if not ok:
            failures.append(rows)
        del graph
        op.close()
    dist.destroy_process_group()
    if failures:
        raise SystemExit(f"rank {rank}: replay mismatch at rows {failures}")


def fault_worker(rank, args, port, _):
    # read at kernel build time; a private cache keeps the stalling build apart
    os.environ["VLLM_MONO_FAULT_INJECT"] = "rank=1,step=2,us=2000000"
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = tempfile.mkdtemp(prefix="mono_fault_")
    env = _setup(rank, args.tp, port, args.layers, args.ctx)
    _, chunk = glm5_mono_launch_rows(2)
    op = _op(env, rank, args.tp, chunk, args.layers, FAULT_POLL_LIMIT)
    h, bufs, _, step = _stepper(env, op, 2, args.ctx, args.layers)
    for _ in range(5):
        bufs[0].copy_(h)
        step()
    torch.cuda.synchronize()
    dist.barrier(env["group"])
    report = [None] * args.tp
    dist.all_gather_object(
        report, (op.poll_errors(), op.poll_xrank.tolist()), group=env["group"]
    )
    op.close()
    dist.destroy_process_group()
    if rank == 0:
        waiting = [r for r in range(args.tp) if r != 1]
        expired = all(report[r][0] for r in waiting)
        flagged = all(report[0][1][r] for r in waiting)
        ok = expired and flagged
        print(
            f"[{'PASS' if ok else 'FAIL'}] fault: expired stages per rank "
            f"{[list(e) for e, _ in report]}, flags on rank 0 {report[0][1]}",
            flush=True,
        )
        if not ok:
            raise SystemExit("rank 0: a stalled peer was not reported")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--port", type=int, default=29761)
    parser.add_argument("--case", choices=("replay", "fault"), default="replay")
    args = parser.parse_args(argv)
    if get_gfx_runtime() != "gfx950" or torch.cuda.device_count() < args.tp:
        print("skip: needs gfx950 and enough GPUs")
        return 0
    worker = replay_worker if args.case == "replay" else fault_worker
    mp.spawn(worker, args=(args, args.port, []), nprocs=args.tp, join=True)
    return 0


def _skip_reason():
    if get_gfx_runtime() != "gfx950" or torch.cuda.device_count() < 4:
        return "needs 4 gfx950 GPUs"
    return None


def test_graph_replay_is_bit_identical():
    if _skip_reason():
        pytest.skip(_skip_reason())
    assert main(["--tp", "4", "--case", "replay"]) == 0


def test_stalled_rank_is_reported_without_a_hang():
    if _skip_reason():
        pytest.skip(_skip_reason())
    assert main(["--tp", "4", "--case", "fault", "--port", "29767"]) == 0


if __name__ == "__main__":
    sys.exit(main())
