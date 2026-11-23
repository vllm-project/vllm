# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The release DSpark draft's `DAR` route (`akr_draft._dar`): its all-reduce +
RMSNorm pairs on `AkrArNorm` (`allreduce_attnres_rmsnorm` without bank) against the
stock backbone. The draft is the release's uncompiled `DFlashQwen3Model` (5
`DFlashQwen3DecoderLayer`s, hidden 7168, TP shards of the real o_proj 768 x 7168 and
MLP 7168 x 1792; attention reduced to its o_proj, it is not part of the pairs), with
the stock `Qwen2MLP` and with the amd_kernels MLP (`AkrDenseMlp`).

8 ranks: every pair takes the kernel up to `AR_MAX_ROWS` rows (eager and in a CUDA
graph captured under `graph_capture`), more rows fall back. Numerics: one pair within
one bf16 ulp of the stock all-reduce + RMSNorm (the residual bit for bit); every pair
of the backbone, on its own inputs: the residual the stock pair's, the output within
one bf16 ulp of the fp64 norm (the stock pair is itself up to one ulp off); the backbone
no further from the stock one than the stock one is from itself run to run or on vLLM's
native norms (1.4-2.2% relative on these random weights). A compiled backbone is
declined. 2 ranks: no kernel for the communicator, the route declines and the backbone
stays stock.
`--bench`: graph replay of the backbone (stock vs route) per draft step.

Run (spawns <world> ranks, needs that many GPUs):
    python tests/models/kimi_k3/akr/draft_ar_norm_ranks.py 8
    python tests/models/kimi_k3/akr/draft_ar_norm_ranks.py 2
    python tests/models/kimi_k3/akr/draft_ar_norm_ranks.py 8 --bench
"""

import os
import sys
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp
from torch import nn

HIDDEN, ATTN_SHARD, INTER_SHARD, LAYERS, EPS = 7168, 768, 1792, 5, 1e-5
ROWS = (1, 7, 20, 64, 128, 152, 153, 192, 300)
GRAPH_ROWS = (20, 128)
BENCH_ROWS = (7, 20, 64, 96, 128, 152, 156, 160, 192)


class _OProj(nn.Module):
    """`DFlashQwen3Attention` reduced to its o_proj on `ATTN_SHARD` columns."""

    def __init__(self, world: int):
        super().__init__()
        from vllm.model_executor.layers.linear import RowParallelLinear

        self.o_proj = RowParallelLinear(ATTN_SHARD * world, HIDDEN, bias=False)

    def forward(self, positions, hidden_states):
        out, _ = self.o_proj(hidden_states[:, :ATTN_SHARD].contiguous())
        return out


def _draft(world: int):
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.models.qwen2 import Qwen2MLP
    from vllm.model_executor.models.qwen3_dflash import (
        DFlashQwen3DecoderLayer,
        DFlashQwen3Model,
    )

    m = DFlashQwen3Model.__new__(DFlashQwen3Model)
    nn.Module.__init__(m)
    m.do_not_compile = True  # what the decorator sets under compile mode NONE
    layers = []
    for _ in range(LAYERS):
        layer = DFlashQwen3DecoderLayer.__new__(DFlashQwen3DecoderLayer)
        nn.Module.__init__(layer)
        layer.self_attn = _OProj(world)
        layer.mlp = Qwen2MLP(HIDDEN, INTER_SHARD * world, "silu")
        layer.input_layernorm = RMSNorm(HIDDEN, eps=EPS)
        layer.post_attention_layernorm = RMSNorm(HIDDEN, eps=EPS)
        layers.append(layer)
    m.layers = nn.ModuleList(layers)
    m.norm = RMSNorm(HIDDEN, eps=EPS)
    m = m.to(torch.bfloat16)
    for p in m.parameters():
        p.data.normal_(0, 0.02)
    for n in m.modules():
        if isinstance(n, RMSNorm):
            n.weight.data.uniform_(0.5, 1.5)
    return SimpleNamespace(model=m)


def _spec(cfg):
    return SimpleNamespace(
        device=torch.device("cuda", torch.accelerator.current_device_index()),
        vllm_config=cfg,
        max_num_reqs=16,
        num_speculative_steps=4,
    )


def _inputs(rows: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)  # same on every rank
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    return torch.arange(rows, device="cuda"), x


def _fwd(model, rows, seed):
    positions, x = _inputs(rows, seed)
    return model.model.forward(None, positions, input_embeds=x)


def _counting(comm):
    calls = [0]
    real = comm.allreduce_attnres_rmsnorm

    def f(*a, **k):
        calls[0] += 1
        return real(*a, **k)

    comm.allreduce_attnres_rmsnorm = f
    return calls


def _rel(a, b):
    a, b = a.float(), b.float()
    cos = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
    return float((a - b).norm() / b.norm()), cos


def _pair_check(norm) -> str:
    """One `AkrArNorm` call against the stock all-reduce + RMSNorm: the new residual
    bit for bit up to the order of the eight-way sum, the output within one bf16 ulp."""
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.distributed.parallel_state import get_tp_group

    rank = get_tp_group().rank_in_group
    worst = [0.0, 0.0]
    for m in ROWS:
        torch.manual_seed(900 + 8 * m + rank)  # a partial per rank
        x = torch.randn(m, HIDDEN).to(torch.bfloat16) * 0.05
        torch.manual_seed(901 + m)
        res = torch.randn(m, HIDDEN).to(torch.bfloat16)
        ref_out, ref_res = norm.stock(tensor_model_parallel_all_reduce(x), res.clone())
        out, new_res = norm(x, res.clone())
        ulp = ref_out.float().abs() * 2.0**-7 + 1e-5
        d_out = float(((out.float() - ref_out.float()).abs() / ulp).max())
        d_res = float((new_res.float() - ref_res.float()).abs().max())
        tol = float(ref_res.float().abs().max()) * 2.0**-7  # sum order
        assert d_out <= 1.0 and d_res <= tol, (rank, m, d_out, d_res)
        worst = [max(worst[0], d_out), max(worst[1], d_res)]
    return f"pair: output <= {worst[0]:.2f} bf16 ulp, residual max |d| {worst[1]:.3g}"


def _in_situ(akr_draft, worst: list[float]):
    """`AkrArNorm.forward` checking every kernel call of the backbone on its own inputs:
    the new residual the stock pair's up to the order of the eight-way sum, the output
    within one bf16 ulp of the fp64 RMSNorm of that residual. Returns the original
    forward."""
    from vllm.distributed import tensor_model_parallel_all_reduce

    real = akr_draft.AkrArNorm.forward

    def checked(self, x, residual):
        ref_res = tensor_model_parallel_all_reduce(x.clone()) + residual
        out, new_res = real(self, x, residual)
        if x.shape[0] <= akr_draft.AR_MAX_ROWS:
            u = new_res.double()
            y = u * torch.rsqrt(u.square().mean(-1, keepdim=True) + self.eps)
            y = y * self.stock.weight.double()
            d_out = float(((out.double() - y).abs() / (y.abs() * 2.0**-7 + 1e-5)).max())
            d_res = float((new_res.float() - ref_res.float()).abs().max())
            tol = float(ref_res.float().abs().max()) * 2.0**-7  # sum order
            assert d_out <= 1.0 and d_res <= tol, (x.shape[0], d_out, d_res)
            worst[0] = max(worst[0], d_out)
        return out, new_res

    akr_draft.AkrArNorm.forward = checked
    return real


def _check(rank, world, cfg, akr_mlp: bool) -> str:
    from vllm import ir
    from vllm.distributed.parallel_state import graph_capture
    from vllm.models.kimi_k3.amd import akr_draft, akr_runtime

    torch.manual_seed(5)  # replicated parts equal on every rank
    model = _draft(world)
    spec = _spec(cfg)
    # the dense MLP kernel takes up to MLP_MAX_ROWS rows (its route's set-up gate)
    rows = tuple(m for m in ROWS if not akr_mlp or m <= akr_draft.MLP_MAX_ROWS)
    if akr_mlp:
        assert akr_draft._mlp_reason(spec, model) is None
        assert akr_draft._install_mlp(spec, model) == LAYERS
    ref = {m: _fwd(model, m, m) for m in rows}
    ref_graph = {m: _fwd(model, m, m + 1000) for m in GRAPH_ROWS}
    # the bf16 noise floor of these random weights: the stock backbone run to run (its
    # GEMMs and all-reduce are not bit-reproducible) and on vLLM's native norms
    floor = {m: _rel(_fwd(model, m, m), ref[m])[0] for m in rows}
    with (
        ir.ops.fused_add_rms_norm.set_priority(["native"]),
        ir.ops.rms_norm.set_priority(["native"]),
    ):
        for m in rows:
            floor[m] = max(floor[m], _rel(_fwd(model, m, m), ref[m])[0])
    assert akr_draft._ar_reason(spec, model) is None
    model.model.do_not_compile = False
    assert "compiled" in (akr_draft._ar_reason(spec, model) or "")
    model.model.do_not_compile = True
    akr_draft._dar(spec, model)
    m_ = model.model
    pairs = sum(isinstance(n, akr_draft.AkrArNorm) for n in m_.modules())
    if world != 8:  # no communicator: nothing installed, every projection reduces
        assert pairs == 0 and akr_runtime.attnres_comm is None
        assert all(lyr.self_attn.o_proj.reduce_results for lyr in m_.layers)
        assert all(akr_draft._mlp_reduces(lyr.mlp) for lyr in m_.layers)
        rel, _ = _rel(_fwd(model, 20, 20), ref[20])
        assert rel <= 1.25 * floor[20] + 1e-3, (rel, floor[20])
        return f"{world} ranks, akr MLP {akr_mlp}: declined (no kernel), stock backbone"
    assert pairs == 2 * LAYERS, pairs
    assert not any(lyr.self_attn.o_proj.reduce_results for lyr in m_.layers)
    assert not any(akr_draft._mlp_reduces(lyr.mlp) for lyr in m_.layers)
    msg = _pair_check(m_.norm) if not akr_mlp else ""
    comm = m_.norm.comm
    calls = _counting(comm)
    worst = 0.0
    for m in rows:
        calls[0] = 0
        rel, cos = _rel(_fwd(model, m, m), ref[m])
        assert calls[0] == (2 * LAYERS if m <= akr_draft.AR_MAX_ROWS else 0), m
        assert rel <= 1.25 * floor[m] + 1e-3 and cos > 0.9995, (rank, m, rel, floor[m])
        worst = max(worst, rel)
    in_situ = [0.0]
    real = _in_situ(akr_draft, in_situ)
    try:
        for m in rows:
            _fwd(model, m, m + 2000)
    finally:
        akr_draft.AkrArNorm.forward = real  # type: ignore[method-assign]
    for m in GRAPH_ROWS:
        positions, x = _inputs(m, m)
        with graph_capture(spec.device) as ctx:
            m_.forward(None, positions, input_embeds=x)  # capture-stream warm-up
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=ctx.stream):
                out = m_.forward(None, positions, input_embeds=x)
        x.copy_(_inputs(m, m + 1000)[1])
        graph.replay()
        torch.accelerator.synchronize()
        rel, cos = _rel(out, ref_graph[m])
        assert rel <= 1.25 * max(floor.values()) + 1e-3 and cos > 0.9995, (m, rel)
        worst = max(worst, rel)
    del comm.allreduce_attnres_rmsnorm  # the counter
    return (
        f"8 ranks, akr MLP {akr_mlp}: {pairs} pairs fused, rows {rows} eager + "
        f"{GRAPH_ROWS} graph, worst rel {worst:.2e} vs stock (stock floor "
        f"{min(floor.values()):.2e}..{max(floor.values()):.2e}), every pair <= "
        f"{in_situ[0]:.2f} ulp of fp64 {msg}"
    )


def _bench(rank, world, cfg) -> str:
    """Microseconds per backbone forward (5 layers: o_proj, 2 pairs, MLP), stock vs
    route: graph of 10 forwards on distinct inputs, min of 20 replays, max of ranks."""
    import torch.distributed as dist

    from vllm.distributed.parallel_state import get_tp_group, graph_capture
    from vllm.models.kimi_k3.amd import akr_draft

    reps, rounds = 10, 20
    torch.manual_seed(5)
    model = _draft(world)
    spec = _spec(cfg)
    cpu = get_tp_group().cpu_group

    def timed(m):
        xs = [_inputs(m, 50 + i) for i in range(reps)]
        with graph_capture(spec.device) as ctx:
            model.model.forward(None, *xs[0][:1], input_embeds=xs[0][1])
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=ctx.stream):
                for p, x in xs:
                    model.model.forward(None, p, input_embeds=x)
        best = float("inf")
        for _ in range(rounds):
            dist.barrier(group=cpu)
            torch.accelerator.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(True)
            a.record()
            graph.replay()
            b.record()
            b.synchronize()
            best = min(best, a.elapsed_time(b) * 1000 / reps)
        t = torch.tensor([best], dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=cpu)
        return float(t)

    stock = {m: timed(m) for m in BENCH_ROWS}
    akr_draft._dar(spec, model)
    route = {m: timed(m) for m in BENCH_ROWS}
    lines = ["| rows | stock us/step | DAR us/step | saving us | per pair us |"]
    lines.append("|---:|---:|---:|---:|---:|")
    for m in BENCH_ROWS:
        d = stock[m] - route[m]
        lines.append(
            f"| {m} | {stock[m]:.1f} | {route[m]:.1f} | {d:.1f} | "
            f"{d / (2 * LAYERS):.2f} |"
        )
    return "\n".join(lines)


def worker(rank, world, port, bench):
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
        MASTER_ADDR="localhost",
        MASTER_PORT=str(port),
        VLLM_ROCM_USE_AITER="1",
    )
    torch.accelerator.set_device_index(rank)
    torch.set_default_device(f"cuda:{rank}")
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.config import (
        CompilationConfig,
        CompilationMode,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    rocm_aiter_ops.refresh_env_variables()
    # the served draft: compile mode NONE (breakable CUDA graphs), AITER norms
    cfg = VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.NONE))
    with set_current_vllm_config(cfg):
        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=world)
        cfg.kernel_config.ir_op_priority.set_default()
        if bench:
            msgs = [_bench(rank, world, cfg)]
        else:
            msgs = [_check(rank, world, cfg, a) for a in (False, True)]
    if rank == 0:
        print("\n".join(msgs), flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    world = int(args[0]) if args else 8
    port = 29400 + os.getpid() % 300
    mp.spawn(worker, args=(world, port, "--bench" in sys.argv), nprocs=world)
    print("PASS", world)
