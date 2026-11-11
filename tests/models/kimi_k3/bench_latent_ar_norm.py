# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark: K3 latent all-reduce + routed-expert RMSNorm (+ up-proj), TP8.

old_tail = AR + RMSNorm + addmm; new_tail = AR + k3_latent_norm_up_proj;
fused1/fused2 = AITER fused AR+RMSNorm (1-/2-stage) for comparison.

    python tests/models/kimi_k3/bench_latent_ar_norm.py [variants]

Each variant is captured ITERS times into one CUDA graph (with an in-graph
producer so the AITER all-reduce sees a graph-registered input, as in serving)
and replayed; the producer-only graph is subtracted. Reports the max over ranks.
"""

import os
import sys

import torch
import torch.distributed as dist
from torch.multiprocessing import spawn

from tests.utils import ensure_current_vllm_config, init_test_distributed_environment
from vllm.utils.network_utils import get_open_port

LATENT = 3584
UP_N = 7168 // 8
EPS = 1e-5
ITERS = int(os.environ.get("ITERS", "100"))
REPLAYS = int(os.environ.get("REPLAYS", "20"))
TOKENS = [int(t) for t in os.environ.get("TOKENS", "4,16,64,192").split(",")]


def _time_graph(body, src) -> float:
    from vllm.distributed import get_tp_group
    from vllm.distributed.parallel_state import graph_capture

    device = src.device
    for _ in range(3):
        body(src * 1.0)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with graph_capture(device) as ctx, torch.cuda.graph(graph, stream=ctx.stream):
        for _ in range(ITERS):
            body(src * 1.0)
    torch.accelerator.synchronize()
    for _ in range(3):
        graph.replay()
    torch.accelerator.synchronize()
    dist.barrier(group=get_tp_group().cpu_group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPLAYS):
        graph.replay()
    end.record()
    torch.accelerator.synchronize()
    us = torch.tensor([start.elapsed_time(end) * 1e3 / (REPLAYS * ITERS)])
    dist.all_reduce(us, op=dist.ReduceOp.MAX, group=get_tp_group().cpu_group)
    del graph
    return us.item()


def _worker(local_rank: int, world_size: int, port: str, variants: list[str]):
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    with ensure_current_vllm_config():
        init_test_distributed_environment(
            world_size, 1, local_rank, port, local_rank=local_rank
        )
        import vllm.kernels  # noqa: F401
        from vllm._aiter_ops import rocm_aiter_ops
        from vllm.distributed import tensor_model_parallel_all_reduce

        torch.manual_seed(0)
        norm_w = (1 + 0.1 * torch.randn(LATENT, device=device)).to(torch.bfloat16)
        up_w = torch.randn(UP_N, LATENT, device=device, dtype=torch.bfloat16) * 0.02

        def old_ar_norm(x):
            return torch.ops.vllm_aiter.rms_norm(
                tensor_model_parallel_all_reduce(x), norm_w, EPS
            )

        aiter_ar = rocm_aiter_ops.get_aiter_allreduce()
        assert aiter_ar is not None
        zeros = torch.zeros(max(TOKENS), LATENT, device=device, dtype=torch.bfloat16)

        def fused(x, one_stage):
            return aiter_ar.aiter_ca.custom_fused_ar_rms(
                x, zeros[: x.shape[0]], norm_w, EPS, use_1stage=one_stage
            )[0]

        from vllm.models.kimi_k3.amd.ops import latent_norm_up_proj  # noqa: F401

        bodies = {
            "prod": lambda x: x,
            "ar": tensor_model_parallel_all_reduce,
            "norm": lambda x: torch.ops.vllm_aiter.rms_norm(x, norm_w, EPS),
            "old": old_ar_norm,
            "fused1": lambda x: fused(x, True),
            "fused2": lambda x: fused(x, False),
        }
        rows = []
        for m in TOKENS:
            torch.manual_seed(local_rank + 1)
            src = torch.randn(m, LATENT, device=device, dtype=torch.bfloat16) * 0.1
            shared = torch.randn(m, UP_N, device=device, dtype=torch.bfloat16)

            def old_tail(x, shared=shared):
                return shared.clone().addmm_(old_ar_norm(x), up_w.t())

            def new_tail(x, shared=shared):
                out = shared.clone()
                torch.ops.vllm.k3_latent_norm_up_proj(
                    tensor_model_parallel_all_reduce(x), norm_w, EPS, up_w, out, 0
                )
                return out

            bodies["old_tail"] = old_tail
            bodies["new_tail"] = new_tail
            base = _time_graph(bodies["prod"], src)
            one_ok = aiter_ar.use_1stage_fused_ar_rms(src)
            res = {
                v: _time_graph(bodies[v], src) - base
                if v != "fused1" or one_ok
                else float("nan")
                for v in variants
            }
            rows.append((m, res))
        if local_rank == 0:
            print("tokens " + " ".join(f"{v:>9}" for v in variants), flush=True)
            for m, res in rows:
                print(f"{m:6d} " + " ".join(f"{res[v]:9.2f}" for v in variants))


if __name__ == "__main__":
    variants = (
        sys.argv[1].split(",")
        if len(sys.argv) > 1
        else ["ar", "norm", "old", "old_tail", "new_tail"]
    )
    spawn(_worker, args=(8, str(get_open_port()), variants), nprocs=8, join=True)
