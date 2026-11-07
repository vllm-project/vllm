# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the fused MoE-LoRA fast path (one-shot) vs two-kernel baseline.

The "one_shot" provider goes through `vllm.lora.ops.triton_ops.fused_moe_lora`
which dispatches to the single-kernel one-shot implementation when
fully_sharded=False (the prefill default).

The "two_kernel" provider drives `fused_moe_lora_shrink` + `fused_moe_lora_expand`
directly, bypassing the dispatch and matching the legacy two-kernel path's
work distribution. This isolates the win from kernel fusion.

With --projections, compare scalar LoRA and routed composition through the
native expert wrappers. Timing includes alignment, scratch initialization and
weighted accumulation, but excludes base GEMMs and activation. Down projection
uses fixed post-activation inputs. JSON artifacts record raw latency samples,
software versions and source hashes; this is not a full-model benchmark.

Run:
    .venv/bin/python -m benchmarks.kernels.benchmark_fused_moe_lora_one_shot
    .venv/bin/python -m benchmarks.kernels.benchmark_fused_moe_lora_one_shot \\
        --model qwen3moe
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
from pathlib import Path

import torch

from vllm import _custom_ops as ops
from vllm.lora.ops.triton_ops import (
    fused_moe_lora,
    fused_moe_lora_expand,
    fused_moe_lora_shrink,
)
from vllm.triton_utils import triton

DTYPE = torch.bfloat16
DEVICE = "cuda"


# ----- input fabrication -----------------------------------------------------


def _round_up(x: int, base: int) -> int:
    return ((x + base - 1) // base) * base


def _ceildiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def _assign_loras(num_tokens: int, num_sequences: int, max_loras: int) -> torch.Tensor:
    tokens_per_seq = num_tokens // num_sequences
    rem = num_tokens % num_sequences
    out = torch.empty(num_tokens, dtype=torch.int32)
    start = 0
    for i in range(num_sequences):
        end = start + tokens_per_seq + (1 if i < rem else 0)
        out[start:end] = random.randint(0, max_loras - 1)
        start = end
    return out


def _assign_experts(num_tokens: int, num_experts: int, top_k: int):
    expert_indices = torch.empty((num_tokens, top_k), dtype=torch.int32)
    for i in range(num_tokens):
        expert_indices[i] = torch.randperm(num_experts)[:top_k]
    weights = torch.rand((num_tokens, top_k), dtype=torch.float32)
    weights = weights / weights.sum(dim=1, keepdim=True)
    return expert_indices, weights


def _make_inputs(
    M: int,
    K: int,
    N_per_slice: int,
    rank: int,
    num_experts: int,
    top_k: int,
    max_loras: int,
    num_slices: int,
    block_size_m: int,
):
    """Mirrors the production caller's tensor layout."""
    torch.manual_seed(0)
    random.seed(0)

    num_sequences = max(1, min(M, 8))
    topk_ids_cpu, topk_weights_cpu = _assign_experts(M, num_experts, top_k)
    token_lora_cpu = _assign_loras(M, num_sequences, max_loras)
    lora_ids_cpu = torch.full((max_loras + 1,), -1, dtype=torch.int32)
    uniq = torch.unique(token_lora_cpu, sorted=True)
    lora_ids_cpu[: uniq.size(0)].copy_(uniq)

    topk_ids = topk_ids_cpu.to(DEVICE)
    topk_weights = topk_weights_cpu.to(device=DEVICE, dtype=DTYPE)
    token_lora_mapping = token_lora_cpu.to(DEVICE)
    lora_ids = lora_ids_cpu.to(DEVICE)
    adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int32, device=DEVICE)

    lora_a = [
        torch.randn((max_loras, num_experts, rank, K), dtype=DTYPE, device=DEVICE)
        / max(K, 1) ** 0.5
        for _ in range(num_slices)
    ]
    lora_b = [
        torch.randn(
            (max_loras, num_experts, N_per_slice, rank),
            dtype=DTYPE,
            device=DEVICE,
        )
        / max(rank, 1) ** 0.5
        for _ in range(num_slices)
    ]
    hidden = torch.randn((M, K), dtype=DTYPE, device=DEVICE)
    out_template = torch.zeros(
        (M, top_k, num_slices * N_per_slice), dtype=DTYPE, device=DEVICE
    )

    # Sorted-path metadata (the prefill default).
    max_pad = topk_ids.numel() + num_experts * (block_size_m - 1)
    max_pad = _round_up(max_pad, block_size_m)
    max_blocks = _ceildiv(max_pad, block_size_m)
    sorted_token_ids = torch.empty(
        (max_loras * max_pad,), dtype=torch.int32, device=DEVICE
    )
    expert_ids = torch.empty(
        (max_loras * max_blocks,), dtype=torch.int32, device=DEVICE
    )
    num_post = torch.empty((max_loras,), dtype=torch.int32, device=DEVICE)
    ops.moe_lora_align_block_size(
        topk_ids,
        token_lora_mapping,
        num_experts,
        block_size_m,
        max_loras,
        max_pad,
        max_blocks,
        sorted_token_ids,
        expert_ids,
        num_post,
        adapter_enabled,
        lora_ids,
    )
    expert_ids = expert_ids.view(max_loras, -1).contiguous()
    sorted_token_ids = sorted_token_ids.view(max_loras, -1).contiguous()
    num_active = torch.tensor([max_loras + 1], dtype=torch.int32, device="cpu")

    return dict(
        hidden=hidden,
        lora_a=lora_a,
        lora_b=lora_b,
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_post=num_post,
        token_lora_mapping=token_lora_mapping,
        lora_ids=lora_ids,
        num_active=num_active,
        adapter_enabled=adapter_enabled,
        out_template=out_template,
        # bookkeeping
        M=M,
        K=K,
        N_per_slice=N_per_slice,
        rank=rank,
        num_experts=num_experts,
        top_k=top_k,
        max_loras=max_loras,
        num_slices=num_slices,
        block_size_m=block_size_m,
    )


# ----- providers -------------------------------------------------------------


def _run_one_shot(inp: dict):
    """Drive `fused_moe_lora` with fully_sharded=False -> one-shot fast path."""
    out = inp["out_template"].clone()
    fused_moe_lora(
        out,
        inp["hidden"],
        inp["lora_a"],
        inp["lora_b"],
        inp["topk_weights"],
        inp["sorted_token_ids"],
        inp["expert_ids"],
        inp["num_post"],
        inp["token_lora_mapping"],
        inp["rank"],
        inp["top_k"],
        inp["lora_ids"],
        inp["num_active"],
        inp["adapter_enabled"],
        inp["block_size_m"],
        64,
        32,
        8,
        4,
        3,
        1,
        inp["block_size_m"],
        64,
        32,
        8,
        4,
        3,
        1,
        False,
        False,
        0,
    )
    return out


def _run_two_kernel(inp: dict):
    """Drive `fused_moe_lora_shrink` + `fused_moe_lora_expand` directly,
    bypassing the dispatch. Matches the legacy two-kernel work distribution.
    """
    M = inp["M"]
    top_k = inp["top_k"]
    rank = inp["rank"]
    num_slices = inp["num_slices"]
    N_per_slice = inp["N_per_slice"]
    K = inp["K"]
    num_experts = inp["num_experts"]
    block_m = inp["block_size_m"]

    intermediate = torch.zeros((num_slices, M, top_k, rank), dtype=DTYPE, device=DEVICE)
    out = inp["out_template"].clone()
    EM = inp["sorted_token_ids"].shape[1]
    num_tokens = M * top_k

    fused_moe_lora_shrink(
        intermediate,
        inp["hidden"],
        inp["lora_a"],
        inp["topk_weights"],
        inp["sorted_token_ids"],
        inp["expert_ids"],
        inp["num_post"],
        inp["token_lora_mapping"],
        top_k,
        inp["lora_ids"],
        inp["adapter_enabled"],
        torch.device(DEVICE),
        rank,
        M,
        EM,
        K,
        num_tokens,
        num_experts,
        num_slices,
        block_m,
        64,
        32,
        8,
        4,
        3,
        1,
        inp["num_active"],
        False,
    )
    fused_moe_lora_expand(
        out,
        intermediate,
        inp["lora_b"],
        inp["topk_weights"],
        inp["sorted_token_ids"],
        inp["expert_ids"],
        inp["num_post"],
        inp["token_lora_mapping"],
        top_k,
        inp["lora_ids"],
        inp["adapter_enabled"],
        torch.device(DEVICE),
        rank,
        M,
        EM,
        K,
        num_tokens,
        num_experts,
        num_slices,
        rank,
        N_per_slice,
        block_m,
        64,
        32,
        8,
        4,
        3,
        1,
        inp["num_active"],
        False,
        0,
    )
    return out


PROVIDER_FNS = {
    "one_shot": _run_one_shot,
    "two_kernel": _run_two_kernel,
}


# ----- model presets ---------------------------------------------------------


MODEL_PRESETS: dict[str, dict] = {
    # Mixtral-8x7B style: E=8, top_k=2, hidden=4096, intermediate=14336
    "mixtral": dict(
        K=4096,
        N_per_slice=7168,
        num_experts=8,
        top_k=2,
        max_loras=4,
        num_slices=2,
        block_size_m=64,
    ),
    # Qwen3-MoE / DeepSeek-V2 style: E=64, top_k=8, hidden=2048, inter=1408
    "qwen3moe": dict(
        K=2048,
        N_per_slice=1408,
        num_experts=64,
        top_k=8,
        max_loras=4,
        num_slices=2,
        block_size_m=64,
    ),
    # GLM-5.1 (zai-org/GLM-5.1-FP8): E=256, top_k=8, hidden=6144,
    # moe_intermediate=2048
    "glm5_1": dict(
        K=6144,
        N_per_slice=2048,
        num_experts=256,
        top_k=8,
        max_loras=4,
        num_slices=2,
        block_size_m=64,
    ),
}


M_RANGE = [16, 64, 256, 1024, 4096, 16384]
RANK_RANGE = [8, 16, 32, 64]


def _projection_case(preset, tokens, rank, routes, distribution, provider):
    """Time the shared expert wrapper, including its alignment and temporaries.

    Down uses fixed post-activation inputs. The combined scope measures both
    adapter projections, not the base MoE GEMMs or activation.
    """
    from vllm.config import LoRAConfig
    from vllm.lora.layers import LoRAMapping, LoRARouteMapping
    from vllm.lora.punica_wrapper import get_punica_wrapper
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        FusedMoEQuantConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.experts.lora_context import MoELoRAContext
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.platforms import current_platform

    torch.manual_seed(0)
    device = current_platform.device_type
    hidden, intermediate, experts, topk = (
        preset[k] for k in ("K", "N_per_slice", "num_experts", "top_k")
    )
    adapters = preset["max_loras"]
    ids = (
        torch.arange(tokens * topk, device=device, dtype=torch.int32).view(tokens, topk)
        % experts
    )
    if distribution == "concentrated":
        ids = (
            torch.arange(topk, device=device, dtype=torch.int32)
            .expand(tokens, -1)
            .contiguous()
        )
    expert_weights = torch.full((tokens, topk), 1 / topk, device=device)
    mapping_ids = tuple(
        tuple((t + r) % adapters + 1 for r in range(routes)) for t in range(tokens)
    )
    if distribution == "concentrated":
        mapping_ids = (tuple(range(1, routes + 1)),) * tokens
    mapping_weights = ((1 / routes,) * routes,) * tokens
    wrapper = get_punica_wrapper(
        tokens,
        tokens,
        device,
        lora_config=LoRAConfig(
            max_loras=adapters,
            max_lora_rank=rank,
            max_loras_per_request=max(2, routes),
        ),
    )
    mapping = (
        LoRAMapping(tuple(row[0] for row in mapping_ids), (), False)
        if provider == "scalar"
        else LoRARouteMapping(mapping_ids, mapping_weights, (), False)
    )
    wrapper.update_metadata(mapping, list(range(1, adapters + 1)), adapters + 1, 0)

    def weight(*shape):
        return torch.randn(*shape, device=device, dtype=DTYPE) * 0.01

    ctx = MoELoRAContext(
        w13_lora_a_stacked=tuple(
            weight(adapters, experts, rank, hidden) for _ in range(2)
        ),
        w13_lora_b_stacked=tuple(
            weight(adapters, experts, intermediate, rank) for _ in range(2)
        ),
        w2_lora_a_stacked=(weight(adapters, experts, rank, intermediate),),
        w2_lora_b_stacked=(weight(adapters, experts, hidden, rank),),
        adapter_enabled=torch.ones(adapters + 1, dtype=torch.int32, device=device),
        max_loras=adapters,
        top_k=topk,
        w13_num_slices=2,
        fully_sharded=False,
        tp_rank=0,
        tp_size=1,
        local_num_experts=experts,
        punica_wrapper=wrapper,
        use_tuned_config=False,
    )
    layer = TritonExperts(
        FusedMoEConfig(
            num_experts=experts,
            experts_per_token=topk,
            hidden_dim=hidden,
            intermediate_size=intermediate,
            num_local_experts=experts,
            num_logical_experts=experts,
            activation=MoEActivation.SILU,
            device=device,
            routing_method=RoutingMethodType.Unspecified,
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=DTYPE,
        ),
        FusedMoEQuantConfig.make(),
    )
    x, activated = weight(tokens, hidden), weight(tokens, topk, intermediate)
    up = torch.empty(tokens, topk, 2 * intermediate, dtype=DTYPE, device=device)
    down = torch.empty(tokens, topk, hidden, dtype=DTYPE, device=device)
    w1 = torch.empty(experts, 2 * intermediate, hidden, dtype=DTYPE, device="meta")
    w2 = torch.empty(experts, hidden, intermediate, dtype=DTYPE, device="meta")

    def gate_up():
        up.zero_()
        return layer.apply_w13_lora(
            ctx,
            y=up,
            x=x,
            topk_ids=ids,
            topk_weights=expert_weights,
            expert_map=None,
            w1=w1,
            w2=w2,
            num_tokens=tokens,
            top_k_num=topk,
        )

    state = gate_up()
    # Keep the same harness runnable on the pre-composition scalar baseline.
    metadata_names = (
        (
            "sorted_token_ids_lora",
            "expert_ids_lora",
            "num_tokens_post_padded_lora",
            "token_lora_mapping",
        )
        if isinstance(state, tuple)
        else ("lora_metadata",)
    )

    def down_projection(metadata=state):
        down.zero_()
        values = metadata if isinstance(metadata, tuple) else (metadata,)
        layer.apply_w2_lora(
            ctx,
            y=down,
            x=activated.view(-1, intermediate),
            topk_weights=expert_weights,
            **dict(zip(metadata_names, values)),
            num_tokens=tokens,
            w1=w1,
            w2=w2,
            top_k_num=topk,
        )

    def combined():
        down_projection(gate_up())

    # Independent FP32 matmuls, grouped only to keep the oracle's memory bounded.
    ref_up, ref_down = (
        torch.zeros_like(up, dtype=torch.float32),
        torch.zeros_like(down, dtype=torch.float32),
    )
    for route in range(routes):
        slots = torch.tensor([row[route] - 1 for row in mapping_ids], device=device)
        for slot in range(adapters):
            for expert in range(experts):
                token, choice = torch.where((ids == expert) & (slots[:, None] == slot))
                if token.numel() == 0:
                    continue
                for part in range(2):
                    delta = (
                        x[token].float()
                        @ ctx.w13_lora_a_stacked[part][slot, expert].float().T
                        @ ctx.w13_lora_b_stacked[part][slot, expert].float().T
                    )
                    ref_up[
                        token, choice, part * intermediate : (part + 1) * intermediate
                    ] += delta / routes
                delta = (
                    activated[token, choice].float()
                    @ ctx.w2_lora_a_stacked[0][slot, expert].float().T
                    @ ctx.w2_lora_b_stacked[0][slot, expert].float().T
                )
                ref_down[token, choice] += delta / (routes * topk)
    combined()
    torch.testing.assert_close(up.float(), ref_up, atol=1e-5, rtol=1e-2)
    torch.testing.assert_close(down.float(), ref_down, atol=1e-5, rtol=1e-2)
    return {"gate_up": gate_up, "down": down_projection, "combined": combined}


def benchmark_projections(args):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.platforms import current_platform

    output = Path(args.save_path)
    output.mkdir(parents=True, exist_ok=True)
    sources = [
        p
        for root in ("vllm/lora", "vllm/model_executor/layers/fused_moe")
        for p in Path(root).rglob("*.py")
    ]
    manifest = {
        "args": vars(args),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "dtype": str(DTYPE),
        "device": current_platform.get_device_name(),
        "commit": os.getenv("VLLM_BENCHMARK_REVISION")
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if current_platform.is_xpu() and "graph" in args.modes:
        raise ValueError("XPU projection benchmarks currently support eager only")

    def measure(fn, mode):
        if current_platform.is_cuda():
            from flashinfer.testing import bench_gpu_time_with_cupti

            return [
                1000 * t
                for t in bench_gpu_time_with_cupti(
                    fn,
                    dry_run_iters=3,
                    repeat_iters=20,
                    use_cuda_graph=mode == "graph",
                    cold_l2_cache=args.cache == "cold",
                )
            ]
        if args.cache != "hot":
            raise ValueError(
                "XPU eager timing supports only the explicitly hot cache mode"
            )
        return [1000 * triton.testing.do_bench(fn) for _ in range(3)]

    preset = dict(MODEL_PRESETS[args.model])
    if args.max_loras is not None:
        preset["max_loras"] = args.max_loras
    if max(args.routes) > preset["max_loras"]:
        raise ValueError("routes must not exceed max_loras")
    with (
        set_current_vllm_config(VllmConfig()),
        (output / "measurements.jsonl").open("w") as log,
    ):
        for tokens in args.tokens:
            for rank in args.ranks:
                for distribution in args.distributions:
                    cases = {}
                    if "scalar" in args.providers:
                        cases["scalar"] = _projection_case(
                            preset, tokens, rank, 1, distribution, "scalar"
                        )
                    if "routed" in args.providers:
                        for routes in args.routes:
                            cases[f"routed_{routes}"] = _projection_case(
                                preset, tokens, rank, routes, distribution, "routed"
                            )
                    if args.check_only:
                        continue
                    for scope in ("gate_up", "down", "combined"):
                        for mode in args.modes:
                            for trial in range(args.trials):
                                order = (
                                    list(cases)
                                    if trial % 2 == 0
                                    else list(reversed(cases))
                                )
                                for provider in order:
                                    fn = cases[provider][scope]
                                    for _ in range(3):
                                        fn()
                                    torch.accelerator.synchronize()
                                    samples = measure(fn, mode)
                                    row = dict(
                                        tokens=tokens,
                                        rank=rank,
                                        distribution=distribution,
                                        provider=provider,
                                        scope=scope,
                                        mode=mode,
                                        trial=trial,
                                        median_us=statistics.median(samples),
                                        samples_us=samples,
                                    )
                                    log.write(json.dumps(row) + "\n")
                                    log.flush()
                                    print(
                                        json.dumps(
                                            {
                                                k: v
                                                for k, v in row.items()
                                                if k != "samples_us"
                                            }
                                        ),
                                        flush=True,
                                    )


def get_benchmark(model: str, max_loras: int | None = None):
    preset = dict(MODEL_PRESETS[model])
    if max_loras is not None:
        preset["max_loras"] = max_loras

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M", "rank"],
            x_vals=[(M, R) for M in M_RANGE for R in RANK_RANGE],
            line_arg="provider",
            line_vals=list(PROVIDER_FNS.keys()),
            line_names=["one_shot (fused)", "two_kernel (legacy)"],
            styles=[("red", "-"), ("blue", "-")],
            ylabel="ms",
            plot_name=f"fused_moe_lora-{model}-loras{preset['max_loras']}",
            args={"preset": preset},
        )
    )
    def benchmark(M, rank, provider, preset):
        inp = _make_inputs(
            M=M,
            K=preset["K"],
            N_per_slice=preset["N_per_slice"],
            rank=rank,
            num_experts=preset["num_experts"],
            top_k=preset["top_k"],
            max_loras=preset["max_loras"],
            num_slices=preset["num_slices"],
            block_size_m=preset["block_size_m"],
        )
        fn = PROVIDER_FNS[provider]
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fn(inp), quantiles=quantiles
        )
        return ms, max_ms, min_ms

    return benchmark


# ----- correctness sanity ---------------------------------------------------


def calculate_diff(model: str, M: int, rank: int, max_loras: int | None = None):
    preset = dict(MODEL_PRESETS[model])
    if max_loras is not None:
        preset["max_loras"] = max_loras
    inp = _make_inputs(
        M=M,
        K=preset["K"],
        N_per_slice=preset["N_per_slice"],
        rank=rank,
        num_experts=preset["num_experts"],
        top_k=preset["top_k"],
        max_loras=preset["max_loras"],
        num_slices=preset["num_slices"],
        block_size_m=preset["block_size_m"],
    )
    out_one = _run_one_shot(inp)
    out_two = _run_two_kernel(inp)
    max_abs = (out_one.float() - out_two.float()).abs().max().item()
    print(
        f"  model={model:<9} M={M:<6} rank={rank:<3}  "
        f"max|one_shot - two_kernel|={max_abs:.4g}  "
        f"ref|max|={out_two.float().abs().max().item():.3g}"
    )
    if max_abs <= 5e-2:
        print("  ✅ outputs match within bf16 tolerance")
    else:
        print("  ❌ outputs differ beyond expected bf16 noise")


# ----- main ------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="mixtral",
        choices=list(MODEL_PRESETS.keys()),
        help="Model preset to sweep",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/fused_moe_lora_one_shot/",
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run correctness sanity check only, no perf sweep",
    )
    parser.add_argument(
        "--max-loras",
        type=int,
        default=None,
        help="Override max_loras in the model preset (number of LoRA adapters "
        "active in the batch). Defaults to the preset's value.",
    )
    parser.add_argument(
        "--projections",
        action="store_true",
        help="Benchmark native expert wrappers after checking correctness",
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128, 1024])
    parser.add_argument("--ranks", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--routes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=["scalar", "routed"],
        default=["scalar", "routed"],
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=["balanced", "concentrated"],
        default=["balanced", "concentrated"],
    )
    parser.add_argument(
        "--modes", nargs="+", choices=["eager", "graph"], default=["eager", "graph"]
    )
    parser.add_argument("--cache", choices=["cold", "hot"], default="cold")
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if args.projections:
        if min(args.tokens + args.ranks + args.routes + [args.trials]) < 1:
            parser.error("tokens, ranks, routes and trials must be positive")
        benchmark_projections(args)
        raise SystemExit(0)

    print(f"Correctness check ({args.model}):")
    calculate_diff(args.model, M=256, rank=32, max_loras=args.max_loras)
    if args.check_only:
        raise SystemExit(0)

    effective_max_loras = (
        args.max_loras
        if args.max_loras is not None
        else MODEL_PRESETS[args.model]["max_loras"]
    )
    print(f"\nGPU: {torch.cuda.get_device_name()}")
    print(f"Model preset: {args.model}  max_loras={effective_max_loras}\n")
    benchmark = get_benchmark(args.model, max_loras=args.max_loras)
    os.makedirs(args.save_path, exist_ok=True)
    benchmark.run(print_data=True, save_path=args.save_path)
