# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
from functools import partial

import torch

from vllm.model_executor.layers.fused_moe.sonic_moe import (
    is_sonic_moe_supported,
    permute_weights_for_sonic,
)

SHAPES = [
    # SonicMoE cutedsl kernels require hidden_dim (k) >= 512 and divisible by 64.
    (256, 512, 2048, 16, 2),
    (512, 1024, 4096, 16, 4),
]


def _bench_us(fn, warmup: int, iters: int, graph_calls: int) -> tuple[float, str]:
    # Prefer CUDA graphs for stable timings, but fall back to eager if capture fails.
    divisor = 1
    mode = "eager"
    if graph_calls > 1:
        try:
            divisor = graph_calls
            stream = torch.cuda.Stream()
            graph = torch.cuda.CUDAGraph()
            fn()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(divisor):
                    fn()
            fn = graph.replay
            mode = "cudagraph"
        except Exception:
            mode = "eager"

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()

    us = start.elapsed_time(end) * 1000.0 / iters / divisor
    return us, mode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark SonicMoeExperts vs TritonExperts on Hopper (H100/H200)."
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--graph-calls", type=int, default=1)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "both"], default="both")
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; skipping.")
        return 0
    if not is_sonic_moe_supported():
        print("Sonic MoE not supported (needs SonicMoE + Hopper). Skipping.")
        return 0

    # FusedMoEModularKernel allocates from v1 WorkspaceManager.
    # When running as a standalone script (outside pytest), we must init it.
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        is_workspace_manager_initialized,
    )

    if not is_workspace_manager_initialized():
        init_workspace_manager(torch.device("cuda"))

    from vllm.model_executor.layers.fused_moe.config import (
        FUSED_MOE_UNQUANTIZED_CONFIG,
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEModularKernel,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize import (
        MoEPrepareAndFinalizeNoEP,
    )
    from vllm.model_executor.layers.fused_moe.sonic_moe import SonicMoeExperts

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    dtypes = (
        [torch.bfloat16, torch.float16]
        if args.dtype == "both"
        else [torch.bfloat16 if args.dtype == "bf16" else torch.float16]
    )

    rows: list[dict] = []
    for m, k, n, e, topk in SHAPES:
        for dtype in dtypes:
            hidden_states = torch.randn((m, k), device="cuda", dtype=dtype) / 10
            w1 = torch.randn((e, n, k), device="cuda", dtype=dtype) / 10
            w2 = torch.randn((e, k, n // 2), device="cuda", dtype=dtype) / 10

            router_logits = torch.randn((m, e), device="cuda", dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1)
            topk_ids = topk_ids.to(torch.int32)
            topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1).to(dtype)

            moe_config = FusedMoEConfig(
                num_experts=e,
                experts_per_token=topk,
                hidden_dim=k,
                intermediate_size_per_partition=n // 2,
                num_local_experts=e,
                num_logical_experts=e,
                moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
                activation="silu",
                in_dtype=dtype,
                device="cuda",
                routing_method=RoutingMethodType.TopK,
            )

            triton_kernel = FusedMoEModularKernel(
                MoEPrepareAndFinalizeNoEP(),
                TritonExperts(moe_config, FUSED_MOE_UNQUANTIZED_CONFIG),
                inplace=False,
            )

            w1_sonic = permute_weights_for_sonic(w1)
            sonic_kernel = FusedMoEModularKernel(
                MoEPrepareAndFinalizeNoEP(),
                SonicMoeExperts(
                    moe_config,
                    FUSED_MOE_UNQUANTIZED_CONFIG,
                    weights_prepermuted=True,
                ),
                inplace=False,
            )

            run_triton = partial(
                triton_kernel,
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
            )
            run_sonic = partial(
                sonic_kernel,
                hidden_states=hidden_states,
                w1=w1_sonic,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
            )

            with torch.inference_mode():
                out_triton = run_triton()
                out_sonic = run_sonic()
                torch.cuda.synchronize()

                denom = out_triton.abs().max().clamp_min(1e-6)
                rel_err = ((out_sonic - out_triton).abs().max() / denom).item()
                if rel_err > 2e-2:
                    raise RuntimeError(f"rel_err={rel_err:.4f} > 2e-2")

                triton_us, triton_mode = _bench_us(
                    run_triton, args.warmup, args.iters, args.graph_calls
                )
                sonic_us, sonic_mode = _bench_us(
                    run_sonic, args.warmup, args.iters, args.graph_calls
                )

            rows.append(
                {
                    "dtype": "bf16" if dtype == torch.bfloat16 else "fp16",
                    "m": m,
                    "k": k,
                    "n": n,
                    "e": e,
                    "topk": topk,
                    "triton_us": triton_us,
                    "sonic_us": sonic_us,
                    "speedup": triton_us / sonic_us if sonic_us > 0 else float("nan"),
                    "mode": f"{triton_mode}/{sonic_mode}",
                    "rel_err": rel_err,
                }
            )

    print("dtype,e,topk,m,k,n,triton_us,sonic_us,speedup,mode,rel_err")
    for r in rows:
        print(
            f"{r['dtype']},{r['e']},{r['topk']},{r['m']},{r['k']},{r['n']},"
            f"{r['triton_us']:.2f},{r['sonic_us']:.2f},{r['speedup']:.3f},"
            f"{r['mode']},{r['rel_err']:.4f}"
        )

    payload = {"device": torch.cuda.get_device_name(0), "results": rows}
    print(json.dumps(payload, indent=2))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
