#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark MoonEP dispatch/combine against DeepEP-HT across router imbalance.

MoonEP plans dynamically redundant experts so every EP rank receives exactly
S x K tokens regardless of router skew; DeepEP's latency follows the hottest
rank. This benchmark sweeps router imbalance (MaxVio) on identical routing and
times, per backend:

- ``moonep``     raw ``moonep.Buffer`` dispatch (planning included), weight
                 prefetch, and combine
- ``moonep_pf``  the vLLM ``MoonEPPrepareAndFinalize`` prepare+finalize round
                 trip (padding, dispatch, prefetch, combine — what serving
                 pays at the modular-kernel boundary)
- ``deepep_ht``  raw ``deep_ep.Buffer`` HT dispatch (layout included) and
                 combine, the call sequence vLLM's DeepEP-HT backend issues

MaxVio = max_e(T_e) / mean_e(T_e) - 1 over global per-expert token counts;
the lognormal routing sigma is bisected to hit each target.

Usage:
    torchrun --nproc_per_node=<EP> \
        benchmarks/kernels/benchmark_moonep_dispatch_combine.py \
        [--maxvios 0.2,1,4] [--num-tokens 1024] [--hidden 2048] \
        [--topk 8] [--experts 128] [--intermediate 768]
"""

import argparse
import os

import torch
import torch.distributed as dist

from vllm.utils.import_utils import has_deep_ep, has_moonep


def generate_routing(
    S: int, K: int, E: int, sigma: float, device, shared_seed: int, rank: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lognormal expert popularity (shared across ranks) + per-rank draws.

    Returns (topk_ids [S,K] int32, topk_weights [S,K] fp32,
    tokens_per_expert [E] int32).
    """
    g = torch.Generator(device=device)
    g.manual_seed(shared_seed)
    popularity = torch.exp(
        sigma * torch.randn(E, generator=g, device=device, dtype=torch.float32)
    )
    g.manual_seed(shared_seed + 1 + rank)
    topk_ids = torch.multinomial(
        popularity.expand(S, E), K, replacement=False, generator=g
    ).to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(S, K, generator=g, device=device, dtype=torch.float32), dim=-1
    )
    tokens_per_expert = torch.bincount(
        topk_ids.reshape(-1).to(torch.int64), minlength=E
    ).to(torch.int32)
    return topk_ids, topk_weights, tokens_per_expert


def realized_maxvio(tokens_per_expert: torch.Tensor, group) -> float:
    total = tokens_per_expert.to(torch.int64).clone()
    dist.all_reduce(total, group=group)
    return (total.max() / total.float().mean()).item() - 1.0


def solve_sigma(target: float, S, K, E, device, group, rank) -> tuple[float, float]:
    """Log-scale bisection for the sigma whose realized MaxVio matches target."""

    def mv(sigma: float) -> float:
        _, _, counts = generate_routing(S, K, E, sigma, device, 1234, rank)
        return realized_maxvio(counts, group)

    lo, hi = 1e-4, 10.0
    if mv(lo) >= target:
        return lo, mv(lo)
    best_s, best_mv = lo, mv(lo)
    for _ in range(40):
        mid = (lo * hi) ** 0.5
        m = mv(mid)
        if abs(m - target) < abs(best_mv - target):
            best_s, best_mv = mid, m
        if abs(m - target) <= max(0.02 * target, 0.005):
            return mid, m
        if m < target:
            lo = mid
        else:
            hi = mid
    return best_s, best_mv


def time_op(fn, group, warmup: int, iters: int) -> float:
    """CUDA-event timing, cross-rank mean microseconds."""
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    dist.barrier(group=group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    local_us = start.elapsed_time(end) / iters * 1e3
    t = torch.tensor([local_us], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, group=group)
    return t.item() / dist.get_world_size(group)


class MoonEPRunner:
    """Raw moonep.Buffer ops plus the vLLM P/F round trip."""

    def __init__(self, group, S, K, E, H, inter, num_prefetch_slots):
        from vllm.model_executor.layers.fused_moe.prepare_finalize.moonep import (
            MoonEPBufferPool,
            MoonEPExpertWeightLayout,
            MoonEPPrepareAndFinalize,
        )

        self.pool = MoonEPBufferPool(
            dict(
                H=H,
                K=K,
                E=E,
                num_ep_ranks=dist.get_world_size(group),
                B=num_prefetch_slots,
                group=group,
                explicitly_destroy=True,
            ),
            max_tokens_per_rank=S,
        )
        # Raw-op timings use the full-capacity buffer, the same one the
        # pool serves for S-token steps.
        _, self.buffer = self.pool.get(S)
        rows = E + num_prefetch_slots
        mk = lambda *shape: torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
        self.layout = MoonEPExpertWeightLayout(
            full_gate_weight=mk(rows, inter, H),
            full_up_weight=mk(rows, inter, H),
            full_down_weight=mk(rows, H, inter),
            num_prefetch_slots=num_prefetch_slots,
        )
        self.pf = MoonEPPrepareAndFinalize(
            buffer_pool=self.pool,
            max_tokens_per_rank=S,
            num_dispatchers=dist.get_world_size(group),
            num_global_experts=E,
            weight_layout=self.layout,
        )
        self.S, self.H = S, H

    def prepare(self, topk_ids, topk_weights, tokens_per_expert, hidden):
        self.topk_ids = topk_ids
        self.topk_weights = topk_weights
        self.tokens_per_expert = tokens_per_expert
        self.hidden = hidden
        self.nvsh, self.rw_nvs, _, self.plan = self.buffer.dispatch(
            hidden, topk_weights, topk_ids, tokens_per_expert
        )

    def dispatch(self):
        self.buffer.dispatch(
            self.hidden, self.topk_weights, self.topk_ids, self.tokens_per_expert
        )

    def prefetch(self):
        self.buffer.prefetch_weight(
            plan=self.plan,
            full_gate_weight=self.layout.full_gate_weight,
            full_up_weight=self.layout.full_up_weight,
            full_down_weight=self.layout.full_down_weight,
        )

    def combine(self):
        self.buffer.combine(plan=self.plan, hidden_nvsh=self.nvsh)

    def pf_round_trip(self):
        from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
        from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
            TopKWeightAndReduceNoOP,
        )

        quant_config = FusedMoEQuantConfig.make(quant_dtype=None)
        out = torch.empty_like(self.hidden)
        noop = TopKWeightAndReduceNoOP()

        def run():
            nvsh, _, _, _, _ = self.pf.prepare(
                self.hidden,
                self.topk_weights,
                self.topk_ids,
                num_experts=self.pf.num_global_experts,
                expert_map=None,
                apply_router_weight_on_input=False,
                quant_config=quant_config,
            )
            self.pf.finalize(out, nvsh, self.topk_weights, self.topk_ids, False, noop)

        return run

    def destroy(self):
        self.pool.destroy()


class DeepEPHTRunner:
    """Raw deep_ep.Buffer HT ops, mirroring vLLM's DeepEP-HT call sequence."""

    def __init__(self, group, S, K, E, H):
        import deep_ep

        self.buffer = deep_ep.Buffer(
            group=group,
            num_nvl_bytes=1024 * 1024 * 1024,
            num_rdma_bytes=0,
            low_latency_mode=False,
            num_qps_per_rank=1,
        )
        self.E = E

    def prepare(self, topk_ids, topk_weights, tokens_per_expert, hidden):
        self.topk_ids = topk_ids.to(torch.int64)
        self.topk_weights = topk_weights
        self.hidden = hidden
        recv_x, _, _, _, self.handle, _ = self._dispatch_once()
        self.recv_x = recv_x

    def _dispatch_once(self):
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            expert_num_tokens,
            is_token_in_rank,
            _,
        ) = self.buffer.get_dispatch_layout(topk_idx=self.topk_ids, num_experts=self.E)
        return self.buffer.dispatch(
            x=self.hidden,
            handle=None,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=expert_num_tokens,
            topk_idx=self.topk_ids,
            topk_weights=self.topk_weights,
            expert_alignment=1,
        )

    def dispatch(self):
        self._dispatch_once()

    def combine(self):
        self.buffer.combine(x=self.recv_x, handle=self.handle)

    def destroy(self):
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=1024, help="S per rank")
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument(
        "--intermediate", type=int, default=768, help="expert FFN size for prefetch"
    )
    parser.add_argument("--num-prefetch-slots", type=int, default=4)
    parser.add_argument("--maxvios", default="0.2,1,4")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    assert has_moonep(), "moonep is required for this benchmark"

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.accelerator.set_device_index(local_rank)
    device = torch.device("cuda", local_rank)
    group = dist.group.WORLD

    S, K, E, H = args.num_tokens, args.topk, args.experts, args.hidden
    hidden = torch.randn(S, H, dtype=torch.bfloat16, device=device) / 10

    runners: dict = {
        "moonep": MoonEPRunner(
            group, S, K, E, H, args.intermediate, args.num_prefetch_slots
        )
    }
    if has_deep_ep():
        # Probe with one dispatch: a deep_ep build without this GPU's arch
        # fails at kernel launch (cudaErrorSymbolNotFound), not at import.
        try:
            probe = DeepEPHTRunner(group, S, K, E, H)
            ids, weights, counts = generate_routing(S, K, E, 0.1, device, 7, rank)
            probe.prepare(ids, weights, counts, hidden)
            runners["deepep_ht"] = probe
        except RuntimeError as e:
            if rank == 0:
                print(f"deep_ep unavailable on this device, skipping: {e}")
    elif rank == 0:
        print("deep_ep not installed; benchmarking MoonEP only")

    comm_bytes = S * K * H * 2  # bf16 payload each rank sends on dispatch

    def gbps(us: float) -> float:
        return comm_bytes / (us * 1e-6) / 1e9

    rows = []
    for target in [float(x) for x in args.maxvios.split(",")]:
        sigma, mv = solve_sigma(target, S, K, E, device, group, rank)
        topk_ids, topk_weights, counts = generate_routing(
            S, K, E, sigma, device, 1234, rank
        )
        for name, r in runners.items():
            r.prepare(topk_ids, topk_weights, counts, hidden)
            row = {"maxvio": mv, "backend": name}
            row["dispatch_us"] = time_op(r.dispatch, group, args.warmup, args.iters)
            row["combine_us"] = time_op(r.combine, group, args.warmup, args.iters)
            if name == "moonep":
                row["prefetch_us"] = time_op(r.prefetch, group, args.warmup, args.iters)
                row["pf_round_trip_us"] = time_op(
                    r.pf_round_trip(), group, args.warmup, args.iters
                )
            rows.append(row)

    if rank == 0:
        hdr = (
            f"{'maxvio':>8} {'backend':>10} {'dispatch_us':>12} {'(GB/s)':>8} "
            f"{'combine_us':>11} {'(GB/s)':>8} {'prefetch_us':>12} "
            f"{'pf_round_trip_us':>17}"
        )
        print(f"\nEP={dist.get_world_size(group)} S={S} K={K} E={E} H={H}")
        print(hdr)
        print("-" * len(hdr))
        for row in rows:
            pre = row.get("prefetch_us")
            rt = row.get("pf_round_trip_us")
            pre_s = "-" if pre is None else f"{pre:.1f}"
            rt_s = "-" if rt is None else f"{rt:.1f}"
            print(
                f"{row['maxvio']:>8.2f} {row['backend']:>10} "
                f"{row['dispatch_us']:>12.1f} {gbps(row['dispatch_us']):>8.1f} "
                f"{row['combine_us']:>11.1f} {gbps(row['combine_us']):>8.1f} "
                f"{pre_s:>12} {rt_s:>17}"
            )

    for r in runners.values():
        r.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
