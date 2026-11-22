# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integrated check of one real `KimiMoE` layer at TP8 (random weights, real shapes,
real vLLM modules): stock `KimiMoE.forward` vs the amd_kernels chain (router+aux ->
down+quant -> own experts -> latent AR tail [+ 16-slot fold] with the deferred final
all-reduce), and the stock-experts latent tail alone.

Spawns 8 ranks (needs 8 GPUs):

    python tests/models/kimi_k3/akr/test_akr_moe_layer_ranks8.py [bench]
"""

import os
import sys

import torch
import torch.distributed as dist
from torch.multiprocessing import spawn

os.environ.setdefault("AITER_SITUV2_A4W4", "1")
os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
os.environ.setdefault("VLLM_ROCM_USE_AITER_MOE_SITUV2", "1")

from tests.utils import (  # noqa: E402
    ensure_current_vllm_config,
    init_test_distributed_environment,
)
from vllm.utils.network_utils import get_open_port  # noqa: E402

H = 7168
M_LIST = [
    int(t) for t in os.environ.get("TOKENS", "1,12,24,40,65,80,128,192").split(",")
]


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def _fill_random(module, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    for name, p in module.named_parameters():
        if p.dtype == torch.uint8 or p.dtype == torch.int8:
            lo, hi = (118, 124) if "scale" in name else (0, 256)
            p.data.copy_(
                torch.randint(
                    lo, hi, p.shape, dtype=torch.uint8, device=p.device, generator=g
                ).view(p.dtype)
            )
        elif p.dtype in (torch.bfloat16, torch.float16, torch.float32):
            sc = 0.1 if "bias" in name else (1.0 if "norm" in name else 0.02)
            base = 1.0 if "norm" in name and "weight" in name else 0.0
            p.data.copy_(
                (
                    torch.randn(p.shape, device=p.device, generator=g)
                    * (0.05 if base else sc)
                    + base
                ).to(p.dtype)
            )
        else:
            raise RuntimeError(f"{name}: {p.dtype}")


def _worker(rank, world, port, mode):
    dev = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(dev)
    from vllm.config import (
        CacheConfig,
        ModelConfig,
        ParallelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.forward_context import set_forward_context

    with ensure_current_vllm_config():
        init_test_distributed_environment(world, 1, rank, port, local_rank=rank)
    # Hugging Face id or local path (only the config is read: random weights).
    path = os.environ.get("KIMI_K3_MODEL", "moonshotai/Kimi-K3.1-Pro")
    mc = ModelConfig(
        model=path,
        skip_tokenizer_init=True,
        trust_remote_code=True,
        dtype="bfloat16",
        hf_overrides={"text_config": {"num_hidden_layers": 14}},
        max_model_len=4096,
    )
    vc = VllmConfig(
        model_config=mc,
        parallel_config=ParallelConfig(tensor_parallel_size=8),
        cache_config=CacheConfig(),
    )
    text = mc.hf_text_config
    from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
    from vllm.models.kimi_k3.amd import akr_moe, akr_runtime, akr_switch  # noqa: F401
    from vllm.models.kimi_k3.amd.linear import KimiMoE

    cpu = get_tp_group().cpu_group
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(dev)
    with set_current_vllm_config(vc), torch.device(dev):
        torch.set_default_dtype(torch.bfloat16)
        moe = KimiMoE(
            config=text,
            quant_config=vc.quant_config,
            prefix="model.layers.1.block_sparse_moe",
            layer_idx=1,
        )
        _fill_random(
            moe,
            1000,
            # every rank: same gate/up weights shard (random per rank is fine: TP
            # partials)
        )
        for name, p in moe.named_parameters():
            if "e_score_correction_bias" in name:
                p.data.copy_(
                    torch.randn(
                        p.shape,
                        device=dev,
                        generator=torch.Generator(device=dev).manual_seed(5),
                    )
                    * 0.1
                )
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        ck = None
        if os.environ.get("DEBUG_ROUTE"):
            rx = moe.experts.routed_experts
            g13 = getattr(rx, "w13_weight", None)
            g13 = g13 if g13 is not None else rx.w13_weight_packed
            g2 = getattr(rx, "w2_weight", None)
            g2 = g2 if g2 is not None else rx.w2_weight_packed
            ck = tuple(
                t.detach().clone().view(torch.uint8)
                for t in (g13, rx.w13_weight_scale, g2, rx.w2_weight_scale)
            )
            if rank == 0:
                print(
                    "DEBUG checkpoint-layout shapes",
                    [tuple(t.shape) for t in ck],
                    flush=True,
                )

        process_weights_after_loading(moe, mc, dev)
        runner = moe.experts
        if rank == 0:
            print(
                "akr static setup:",
                moe._akr_moe,
                "tail_shardable",
                runner._tail_shardable,
                "preshard",
                runner._up_proj_preshard,
                flush=True,
            )
        assert moe._akr_moe, "akr_moe.setup() declined this layer"

        def run(x):
            with set_forward_context(None, vc):
                return moe(x)

        # stock reference (before the akr state is prepared)
        outs = {}
        torch.manual_seed(77)
        xs = {m: (torch.randn(m, H, device=dev) * 0.5).bfloat16() for m in M_LIST}
        # the stock reference runs the shared experts inline: with the aux stream the
        # stock path races on the modular kernel's shared workspace manager in this
        # harness (one lane), which is not what serving does
        sx = runner._shared_experts
        assert sx is not None
        stream_saved, sx._stream = sx._stream, None
        for m in M_LIST:
            outs[m] = run(xs[m]).clone()
        torch.accelerator.synchronize()
        sx._stream = stream_saved
        # akr: communicators, layer state, deferral
        mem0 = torch.accelerator.memory_allocated()
        akr_runtime.init_comms(
            attnres=True,
            latent=akr_switch.on("LATENT_AR"),
            hidden=H,
            up_rows=H // world,
            latent_dim=moe.moe_hidden_size,
        )
        torch.accelerator.synchronize()
        mem1 = torch.accelerator.memory_allocated()
        ok = akr_moe.prepare(moe)
        assert ok, "prepare() declined"
        akr_moe.warm_tail(moe)
        akr_moe.prewarm_capture_stream(moe)
        torch.accelerator.synchronize()
        mem2 = torch.accelerator.memory_allocated()
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            # the graph-capture stream's own workspaces
            akr_moe.prewarm_capture_stream(moe)
        torch.accelerator.synchronize()
        mem3 = torch.accelerator.memory_allocated()
        if rank == 0:
            print(
                f"extra memory per rank: comms {(mem1 - mem0) / 2**20:.1f} MiB "
                "(AttnRes + latent tail), "
                f"main-stream workspaces + aux per M {(mem2 - mem1) / 2**20:.1f} MiB, "
                f"capture-stream workspaces {(mem3 - mem2) / 2**20:.1f} MiB",
                flush=True,
            )
        st = moe._akr_state
        assert st is not None
        if rank == 0:
            print(
                f"akr state: router_ok={st.router_ok} down_ok={st.down_ok} "
                f"tail_ok={st.tail_ok}",
                flush=True,
            )
        if os.environ.get("DEBUG_ROUTE"):
            dp = runner.routed_input_transform
            assert dp is not None
            for m in (1, 12, 24, 65):
                y_stock = dp(xs[m])
                y_stock = y_stock[0] if isinstance(y_stock, tuple) else y_stock
                y_mine = (xs[m].float() @ dp.weight.data.float().t()).bfloat16()
                if rank == 0:
                    print(
                        f"DEBUG down-proj M={m}: stock y vs x@W^T rel "
                        f"{_rel(y_stock, y_mine):.2e} "
                        f"weight {tuple(dp.weight.shape)} {dp.weight.dtype} "
                        f"quant_method {type(dp.quant_method).__name__} "
                        f"is_shuffled {getattr(dp.weight, 'is_shuffled', None)} bias "
                        f"{getattr(dp, 'bias', None)}",
                        flush=True,
                    )
            from amd_kernels.moe.latent_moe_common import router_sigmoid_bias_topk

            for m in (12, 24):
                x = xs[m]
                logits, _ = moe.gate(x)
                ws_, is_ = runner.router.select_experts(
                    hidden_states=x,
                    router_logits=logits,
                    topk_indices_dtype=runner._quant_method.topk_indices_dtype,
                )
                wa, ia = router_sigmoid_bias_topk(
                    x,
                    moe.gate.weight.data,
                    moe.gate.e_score_correction_bias.data,
                    1.0,
                    topk=st.topk,
                    shuffled=False,
                    aux=akr_moe._aux(moe, m, dev, st.topk),
                )
                same_ids = torch.equal(
                    is_.sort(1).values.int(), ia.sort(1).values.int()
                )
                ref = torch.mm(x, moe.gate.weight.data.t(), out_dtype=torch.float32)
                if rank == 0:
                    w_diff = (ws_.sort(1).values - wa.sort(1).values).abs().max().item()
                    print(
                        f"DEBUG M={m} ids equal {same_ids} logits maxdiff "
                        f"{(logits - ref).abs().max().item():.2e} "
                        f"w maxdiff {w_diff:.2e} "
                        f"gate dtype {moe.gate.weight.dtype} stock ids dtype "
                        f"{is_.dtype}",
                        flush=True,
                    )
        if os.environ.get("DEBUG_ROUTE"):
            for name, env in (
                ("default", {}),
                ("SLOT_FOLD=0", {"VLLM_KIMI_AKR_SLOT_FOLD": "0"}),
                ("SHARED_AUX=0", {"VLLM_KIMI_AKR_SHARED_AUX": "0"}),
                ("SHARED_AT=start", {"VLLM_KIMI_AKR_SHARED_AT": "start"}),
            ):
                os.environ.update(env)
                runner._akr_defer_final = False
                got = run(xs[12])
                torch.accelerator.synchronize()
                got2 = run(xs[12])
                for k in env:
                    del os.environ[k]
                if rank == 0:
                    print(
                        f"DEBUG M=12 {name}: akr vs stock {_rel(got, outs[12]):.2e}; "
                        f"akr run-to-run {_rel(got2, got):.2e}",
                        flush=True,
                    )
        # (a) akr chain + latent tail kernel (+ slot fold) + final AR, deferred or not,
        #     vs the SAME akr routed experts through the stock tail (isolates the
        #     tail / deferral / AR); (b) the routed experts vs the fp64 reference;
        #     (c) info: vs the stock KimiMoE path.
        for m in M_LIST:
            runner._akr_defer_final = False
            st.tail_ok = False
            tail_saved, runner._akr_tail = runner._akr_tail, None
            # akr experts + stock tail + stock final all-reduce
            base = run(xs[m]).clone()
            st.tail_ok = True
            runner._akr_tail = tail_saved
            for defer in (False, True):
                runner._akr_defer_final = defer
                got = run(xs[m])
                if defer:
                    got = tensor_model_parallel_all_reduce(got)
                torch.accelerator.synchronize()
                e, e_stock = _rel(got, base), _rel(got, outs[m])
                if rank == 0:
                    print(
                        f"KimiMoE M={m:3d} defer={defer!s:5s} akr(tail kernel) vs "
                        f"akr(stock tail) rel L2 {e:.2e}"
                        f" | akr vs stock path {e_stock:.2e}",
                        flush=True,
                    )
                assert e < 2.5e-2, (
                    m,
                    defer,
                    e,
                )  # shared-expert split-K noise ~8e-3 + bf16
        # stock experts + latent tail kernel only
        st_saved, moe._akr_state = moe._akr_state, None
        runner._akr_defer_final = False
        for m in [t for t in (12, 65) if t in xs]:
            got = run(xs[m])
            e = _rel(got, outs[m])
            if rank == 0:
                print(
                    f"stock experts + latent tail kernel M={m} vs stock rel L2 {e:.2e}",
                    flush=True,
                )
            assert e < 2.5e-2
        moe._akr_state = st_saved
        dist.barrier(group=cpu)
        if mode == "bench":
            from vllm.distributed.parallel_state import graph_capture

            def two_graph(body):
                def build(n):
                    g = torch.cuda.CUDAGraph()
                    with (
                        graph_capture(dev) as ctx,
                        torch.cuda.graph(g, stream=ctx.stream),
                    ):
                        for _ in range(n):
                            body()
                    return g

                for _ in range(2):
                    body()
                torch.accelerator.synchronize()
                g1, g2 = build(10), build(20)
                res = []
                for g in (g1, g2):
                    for _ in range(3):
                        g.replay()
                    torch.accelerator.synchronize()
                    best = 1e9
                    for _ in range(15):
                        dist.barrier(group=cpu)
                        a, b = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        a.record()
                        g.replay()
                        b.record()
                        torch.accelerator.synchronize()
                        best = min(best, a.elapsed_time(b) * 1e3)
                    res.append(best)
                t = torch.tensor([(res[1] - res[0]) / 10])
                dist.all_reduce(t, op=dist.ReduceOp.MAX, group=cpu)
                return t.item()

            if rank == 0:
                print(
                    "| M | stock KimiMoE us | akr chain us (final AR deferred, not "
                    "included) | akr + AR us | akr + AR, shared inline us | speedup "
                    "(akr+AR) |"
                )
            for m in M_LIST:
                x = xs[m]
                moe._akr_state = None
                runner._akr_defer_final = False
                with set_forward_context(None, vc):
                    ts = two_graph(lambda x=x: moe(x))
                moe._akr_state = st_saved
                runner._akr_defer_final = True
                with set_forward_context(None, vc):
                    td = two_graph(lambda x=x: moe(x))
                    tw = two_graph(lambda x=x: tensor_model_parallel_all_reduce(moe(x)))
                    os.environ["VLLM_KIMI_AKR_SHARED_AUX"] = "0"
                    ti = two_graph(lambda x=x: tensor_model_parallel_all_reduce(moe(x)))
                    del os.environ["VLLM_KIMI_AKR_SHARED_AUX"]
                if rank == 0:
                    print(
                        f"| {m} | {ts:.1f} | {td:.1f} | {tw:.1f} | {ti:.1f} | "
                        f"{ts / tw:.2f}x |",
                        flush=True,
                    )
        dist.barrier(group=cpu)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    spawn(_worker, args=(8, str(get_open_port()), mode), nprocs=8, join=True)
