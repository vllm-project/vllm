# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline tuner for vLLM's BF16-by-NVFP4 FlyDSL MoE kernels.

The generated JSON is consumed by ``FlydslNvfp4Experts``.  Tune power-of-two
token counts; runtime rounds a request up to the next available entry.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from aiter.fused_moe import moe_sorting

try:
    from benchmark_utils import _profiler_bench_us
except ModuleNotFoundError:
    from benchmarks.kernels.benchmark_utils import _profiler_bench_us

from vllm.kernels.flydsl.nvfp4_moe_2stages import (
    nvfp4_moe_stage1,
    nvfp4_moe_stage2,
)
from vllm.model_executor.layers.fused_moe.experts.flydsl_nvfp4_moe import (
    FlydslNvfp4Experts,
)
from vllm.utils.platform_utils import get_device_name_as_file_name

TILE_MS = (16, 32, 64, 128)
TILE_NS = (64, 128)
TILE_KS = (64, 128, 256)
K_BATCHES = (1, 2, 4, 7, 14)
DEFAULT_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
NUM_WARMUP = 10
NUM_ITERS = 100
JIT_SUSPECTED_WALL_TIME_S = 5.0
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "vllm/model_executor/layers/fused_moe/configs"


def get_flydsl_stage1_kernels_nvfp4_bf16(
    model_dim: int,
    inter_dim: int,
    *,
    use_g1u1: bool = True,
    block_ms: tuple[int, ...] = TILE_MS,
    out_dtype: str = "bf16",
) -> dict[str, dict[str, int]]:
    """Return valid AITER BF16-by-NVFP4 stage-one candidates."""
    registry = {
        f"flydsl_moe1_abf16_wnvfp4_{out_dtype}_t{tm}x{tn}x{tk}"
        + (f"_kb{kb}" if kb != 1 else ""): {
            "tile_m": tm,
            "tile_n": tn,
            "tile_k": tk,
            "k_batch": kb,
        }
        for tm in TILE_MS
        for tn in TILE_NS
        for tk in TILE_KS
        for kb in K_BATCHES
    }
    candidates = {}
    for name, params in registry.items():
        tm, tk, kb = (params[key] for key in ("tile_m", "tile_k", "k_batch"))
        if not use_g1u1 or tm not in block_ms or model_dim % tk:
            continue
        if kb > 1:
            if model_dim % kb:
                continue
            k_per_batch = model_dim // kb
            k_tiles = k_per_batch // tk
            if k_per_batch % tk or k_tiles < 4 or k_tiles % 2:
                continue
        else:
            k_tiles = model_dim // tk
            if k_tiles < 2 or k_tiles % 2:
                continue
        if inter_dim % params["tile_n"]:
            continue
        if (tm * tk * 2 // 256) % 16:
            continue
        candidates[name] = params
    return candidates


def get_flydsl_stage2_kernels_nvfp4_bf16(
    model_dim: int,
    inter_dim: int,
    *,
    use_g1u1: bool = True,
    block_ms: tuple[int, ...] = TILE_MS,
    out_dtype: str = "bf16",
) -> dict[str, dict[str, int | str]]:
    """Return valid AITER BF16-by-NVFP4 atomic stage-two candidates."""
    registry = {
        f"flydsl_moe2_abf16_wnvfp4_{out_dtype}_t{tm}x{tn}x{tk}_atomic": {
            "tile_m": tm,
            "tile_n": tn,
            "tile_k": tk,
            "mode": "atomic",
        }
        for tm in TILE_MS
        for tn in TILE_NS
        for tk in TILE_KS
    }
    return {
        name: params
        for name, params in registry.items()
        if use_g1u1
        and params["tile_m"] in block_ms
        and inter_dim % params["tile_k"] == 0
        and model_dim % params["tile_n"] == 0
        and params["mode"] == "atomic"
    }


def _route(topk_ids: torch.Tensor, topk_weights: torch.Tensor, experts: int, m: int):
    ids, weights, expert_ids, valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, experts, 1, torch.bfloat16, m
    )
    valid_ids = valid_ids[:1].contiguous()
    blocks = (int(valid_ids.item()) + m - 1) // m
    return ids[: blocks * m], weights[: blocks * m], expert_ids[:blocks], valid_ids


def tune(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    result: dict[str, dict[str, int]] = {}
    s1_registry = get_flydsl_stage1_kernels_nvfp4_bf16(
        args.hidden_size, args.inter_dim
    )
    s2_registry = get_flydsl_stage2_kernels_nvfp4_bf16(
        args.hidden_size, args.inter_dim
    )

    for tokens in args.tokens:
        token_start = time.perf_counter()
        print(f"[nvfp4-tune] M={tokens}: preparing candidates", flush=True)
        hidden = torch.randn(
            tokens, args.hidden_size, device=device, dtype=torch.bfloat16
        )
        scores = torch.randn(tokens, args.experts, device=device)
        topk_values, topk_ids = torch.topk(scores, args.topk, dim=1)
        topk_weights = torch.softmax(topk_values, dim=1).to(torch.float32)
        topk_ids = topk_ids.to(torch.int32)
        w1 = torch.randint(
            0, 256, (args.experts, 2 * args.inter_dim, args.hidden_size // 2),
            device=device, dtype=torch.uint8,
        )
        w2 = torch.randint(
            0, 256, (args.experts, args.hidden_size, args.inter_dim // 2),
            device=device, dtype=torch.uint8,
        )
        w1 = FlydslNvfp4Experts.shuffle_nvfp4_weight_for_flydsl(w1)
        w2 = FlydslNvfp4Experts.shuffle_nvfp4_weight_for_flydsl(w2)
        w1_scale = torch.ones(
            args.experts, args.hidden_size // 16, 2 * args.inter_dim,
            device=device, dtype=torch.uint8,
        )
        w2_scale = torch.ones(
            args.experts, args.inter_dim // 16, args.hidden_size,
            device=device, dtype=torch.uint8,
        )
        global_scale = torch.ones(args.experts, device=device, dtype=torch.float32)
        routing: dict[int, tuple[Any, ...]] = {}
        intermediate = torch.randn(
            tokens, args.topk, args.inter_dim, device=device, dtype=torch.bfloat16
        )
        output = torch.zeros(
            tokens, args.hidden_size, device=device, dtype=torch.bfloat16
        )
        best_stage1: dict[int, tuple[float, dict[str, int]]] = {}
        best_stage2: dict[int, tuple[float, dict[str, int | str]]] = {}

        def route_for(tile_m: int) -> tuple[Any, ...]:
            if tile_m not in routing:
                routing[tile_m] = _route(
                    topk_ids, topk_weights, args.experts, tile_m
                )
            return routing[tile_m]

        for index, params in enumerate(s1_registry.values(), start=1):
            tile_m = params["tile_m"]
            ids, _, expert_ids, valid_ids = route_for(tile_m)
            case_start = time.perf_counter()
            print(
                f"[nvfp4-tune] M={tokens} stage1 "
                f"case {index}/{len(s1_registry)} {params}",
                flush=True,
            )
            try:
                us = _profiler_bench_us(
                    lambda: nvfp4_moe_stage1(
                        hidden, w1, w1_scale, global_scale, ids, expert_ids,
                        valid_ids, topk=args.topk, inter_dim=args.inter_dim,
                        output=intermediate, **params,
                    ),
                    (),
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                )
            except Exception:
                print(
                    f"[nvfp4-tune] M={tokens} stage1 failed "
                    f"after {time.perf_counter() - case_start:.2f}s",
                    flush=True,
                )
                continue
            wall_time = time.perf_counter() - case_start
            print(
                f"[nvfp4-tune] M={tokens} stage1 {us:.2f}us "
                f"({wall_time:.2f}s wall)",
                flush=True,
            )
            if wall_time >= JIT_SUSPECTED_WALL_TIME_S:
                print(
                    "[nvfp4-tune] stage1 wall time suggests a first-use FlyDSL "
                    "JIT compilation",
                    flush=True,
                )
            if tile_m not in best_stage1 or us < best_stage1[tile_m][0]:
                best_stage1[tile_m] = (us, params)

        for index, params in enumerate(s2_registry.values(), start=1):
            tile_m = params["tile_m"]
            ids, weights, expert_ids, valid_ids = route_for(tile_m)
            case_start = time.perf_counter()
            print(
                f"[nvfp4-tune] M={tokens} stage2 "
                f"case {index}/{len(s2_registry)} {params}",
                flush=True,
            )
            try:
                us = _profiler_bench_us(
                    lambda: (
                        output.zero_(),
                        nvfp4_moe_stage2(
                            intermediate, w2, w2_scale, global_scale, ids,
                            expert_ids, valid_ids, topk=args.topk,
                            model_dim=args.hidden_size, output=output,
                            sorted_weights=weights, **params,
                        ),
                    ),
                    (),
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                )
            except Exception:
                print(
                    f"[nvfp4-tune] M={tokens} stage2 failed "
                    f"after {time.perf_counter() - case_start:.2f}s",
                    flush=True,
                )
                continue
            wall_time = time.perf_counter() - case_start
            print(
                f"[nvfp4-tune] M={tokens} stage2 {us:.2f}us "
                f"({wall_time:.2f}s wall)",
                flush=True,
            )
            if wall_time >= JIT_SUSPECTED_WALL_TIME_S:
                print(
                    "[nvfp4-tune] stage2 wall time suggests a first-use FlyDSL "
                    "JIT compilation",
                    flush=True,
                )
            if tile_m not in best_stage2 or us < best_stage2[tile_m][0]:
                best_stage2[tile_m] = (us, params)

        best: tuple[float, dict[str, int]] | None = None
        for tile_m, (stage1_us, stage1_params) in best_stage1.items():
            stage2 = best_stage2.get(tile_m)
            if stage2 is None:
                continue
            stage2_us, stage2_params = stage2
            config = {
                **stage1_params,
                "tile_n2": int(stage2_params["tile_n"]),
                "tile_k2": int(stage2_params["tile_k"]),
            }
            if best is None or stage1_us + stage2_us < best[0]:
                best = (stage1_us + stage2_us, config)
        if best is None:
            raise RuntimeError(
                "No valid matching FlyDSL NVFP4 stage-one/stage-two candidate "
                f"completed for token count {tokens}."
            )
        result[str(tokens)] = best[1]
        print(
            f"[nvfp4-tune] M={tokens} selected {best[1]} "
            f"in {time.perf_counter() - token_start:.2f}s",
            flush=True,
        )

    name = (
        f"E={args.experts},N={args.inter_dim},"
        f"device_name={get_device_name_as_file_name()},"
        "dtype=nvfp4_bf16,backend=flydsl.json"
    )
    output = Path(args.output_dir) / name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--inter-dim", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--tokens", type=int, nargs="+", default=DEFAULT_TOKENS)
    parser.add_argument("--iters", type=int, default=NUM_ITERS)
    parser.add_argument("--warmup", type=int, default=NUM_WARMUP)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR)
    )
    print(tune(parser.parse_args()))


if __name__ == "__main__":
    main()
