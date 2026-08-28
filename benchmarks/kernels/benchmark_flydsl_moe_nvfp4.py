# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline tuner for vLLM's BF16-by-NVFP4 FlyDSL MoE kernels.

The generated JSON is consumed by ``FlydslNvfp4Experts``.  Tune power-of-two
token counts; runtime rounds a request up to the next available entry.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import multiprocessing as mp
import os
import sys
import threading
import time
from pathlib import Path

import torch
from aiter.fused_moe import moe_sorting
from flydsl.runtime.device import get_rocm_arch

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
LDS_CAP_BYTES = {"gfx942": 65536, "gfx950": 163840}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "vllm/model_executor/layers/fused_moe/configs"


def get_flydsl_stage1_kernels_nvfp4_bf16(
    model_dim: int,
    inter_dim: int,
    *,
    use_g1u1: bool = True,
    block_ms: tuple[int, ...] = TILE_MS,
    out_dtype: str = "bf16",
    lds_cap_bytes: int | None = None,
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
        if lds_cap_bytes is not None and 2 * tm * tk * 2 > lds_cap_bytes:
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
    lds_cap_bytes: int | None = None,
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
        and (params["tile_m"] * params["tile_k"] * 2 // 256) % 16 == 0
        and (
            lds_cap_bytes is None
            or 2 * params["tile_m"] * params["tile_k"] * 2 <= lds_cap_bytes
        )
        and params["mode"] == "atomic"
    }


def _route(topk_ids: torch.Tensor, topk_weights: torch.Tensor, experts: int, m: int):
    ids, weights, expert_ids, valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, experts, 1, torch.bfloat16, m
    )
    valid_ids = valid_ids[:1].contiguous()
    blocks = (int(valid_ids.item()) + m - 1) // m
    return ids[: blocks * m], weights[: blocks * m], expert_ids[:blocks], valid_ids


def _generate_stage1_data(
    tokens: int,
    hidden_size: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    hidden = torch.randn(tokens, hidden_size, device=device, dtype=torch.bfloat16)
    scores = torch.randn(tokens, experts, device=device)
    topk_values, topk_ids = torch.topk(scores, topk, dim=1)
    topk_weights = torch.softmax(topk_values, dim=1).to(torch.float32)
    ids, _, expert_ids, valid_ids = _route(
        topk_ids.to(torch.int32), topk_weights, experts, tile_m
    )
    w1 = torch.randint(
        0,
        256,
        (experts, 2 * inter_dim, hidden_size // 2),
        device=device,
        dtype=torch.uint8,
    )
    w1 = FlydslNvfp4Experts.shuffle_nvfp4_weight_for_flydsl(w1)
    return {
        "hidden": hidden,
        "w1": w1,
        "w1_scale": torch.ones(
            experts,
            hidden_size // 16,
            2 * inter_dim,
            device=device,
            dtype=torch.uint8,
        ),
        "global_scale": torch.ones(experts, device=device, dtype=torch.float32),
        "ids": ids,
        "expert_ids": expert_ids,
        "valid_ids": valid_ids,
        "output": torch.empty(
            tokens, topk, inter_dim, device=device, dtype=torch.bfloat16
        ),
    }


def _generate_stage2_data(
    tokens: int,
    hidden_size: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    scores = torch.randn(tokens, experts, device=device)
    topk_values, topk_ids = torch.topk(scores, topk, dim=1)
    topk_weights = torch.softmax(topk_values, dim=1).to(torch.float32)
    ids, weights, expert_ids, valid_ids = _route(
        topk_ids.to(torch.int32), topk_weights, experts, tile_m
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, inter_dim // 2),
        device=device,
        dtype=torch.uint8,
    )
    w2 = FlydslNvfp4Experts.shuffle_nvfp4_weight_for_flydsl(w2)
    return {
        "intermediate": torch.randn(
            tokens, topk, inter_dim, device=device, dtype=torch.bfloat16
        ),
        "w2": w2,
        "w2_scale": torch.ones(
            experts,
            inter_dim // 16,
            hidden_size,
            device=device,
            dtype=torch.uint8,
        ),
        "global_scale": torch.ones(experts, device=device, dtype=torch.float32),
        "ids": ids,
        "weights": weights,
        "expert_ids": expert_ids,
        "valid_ids": valid_ids,
        "output": torch.empty(tokens, hidden_size, device=device, dtype=torch.bfloat16),
    }


def _run_stage1_candidate(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    global_scale: torch.Tensor,
    ids: torch.Tensor,
    expert_ids: torch.Tensor,
    valid_ids: torch.Tensor,
    output: torch.Tensor,
    topk: int,
    inter_dim: int,
    params: dict[str, int],
) -> torch.Tensor:
    return nvfp4_moe_stage1(
        hidden,
        w1,
        w1_scale,
        global_scale,
        ids,
        expert_ids,
        valid_ids,
        topk=topk,
        inter_dim=inter_dim,
        output=output,
        **params,
    )


def _run_stage2_candidate(
    intermediate: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    global_scale: torch.Tensor,
    ids: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    valid_ids: torch.Tensor,
    output: torch.Tensor,
    topk: int,
    model_dim: int,
    params: dict[str, int | str],
) -> torch.Tensor:
    # Atomic stage two requires a fresh zero destination on every invocation.
    # Keep this inside the timed callable to measure the complete operation.
    output.zero_()
    return nvfp4_moe_stage2(
        intermediate,
        w2,
        w2_scale,
        global_scale,
        ids,
        expert_ids,
        valid_ids,
        topk=topk,
        model_dim=model_dim,
        output=output,
        sorted_weights=weights,
        **params,
    )


def _make_task(
    args: argparse.Namespace,
    tokens: int,
    stage: str,
    name: str,
    params: dict[str, int | str],
) -> tuple:
    return (
        tokens,
        stage,
        name,
        params,
        args.hidden_size,
        args.inter_dim,
        args.experts,
        args.topk,
        args.warmup,
        args.iters,
    )


def _params_key(params: dict[str, int | str]) -> tuple[tuple[str, int | str], ...]:
    return tuple(sorted(params.items()))


def _init_worker(gpu_count: int) -> None:
    worker_index = mp.current_process()._identity[-1] - 1
    torch.accelerator.set_device_index(worker_index % gpu_count)


@contextlib.contextmanager
def _filter_native_stderr():
    """Hide Kineto lifecycle messages while preserving all other stderr."""
    sys.stderr.flush()
    saved_stderr = os.dup(2)
    read_fd, write_fd = os.pipe()

    def relay() -> None:
        with (
            os.fdopen(read_fd, "rb") as source,
            os.fdopen(os.dup(saved_stderr), "wb") as destination,
        ):
            for line in iter(source.readline, b""):
                if b"ActivityProfilerController.cpp" not in line:
                    destination.write(line)
                    destination.flush()

    relay_thread = threading.Thread(target=relay, daemon=True)
    relay_thread.start()
    os.dup2(write_fd, 2)
    os.close(write_fd)
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        relay_thread.join()


def _benchmark_candidate(task: tuple) -> tuple:
    (
        tokens,
        stage,
        name,
        params,
        hidden_size,
        inter_dim,
        experts,
        topk,
        warmup,
        iters,
    ) = task
    device = torch.device("cuda", torch.accelerator.current_device_index())
    try:
        if stage == "stage1":
            data = _generate_stage1_data(
                tokens,
                hidden_size,
                inter_dim,
                experts,
                topk,
                int(params["tile_m"]),
                device=device,
            )
            function = lambda: _run_stage1_candidate(
                data["hidden"],
                data["w1"],
                data["w1_scale"],
                data["global_scale"],
                data["ids"],
                data["expert_ids"],
                data["valid_ids"],
                data["output"],
                topk,
                inter_dim,
                params,
            )
        else:
            data = _generate_stage2_data(
                tokens,
                hidden_size,
                inter_dim,
                experts,
                topk,
                int(params["tile_m"]),
                device=device,
            )
            function = lambda: _run_stage2_candidate(
                data["intermediate"],
                data["w2"],
                data["w2_scale"],
                data["global_scale"],
                data["ids"],
                data["expert_ids"],
                data["weights"],
                data["valid_ids"],
                data["output"],
                topk,
                hidden_size,
                params,
            )
        torch.accelerator.synchronize()
        with _filter_native_stderr():
            us = _profiler_bench_us(function, (), warmup, iters)
        return (tokens, stage, name, params), us, None
    except Exception as exc:
        return (tokens, stage, name, params), -1.0, repr(exc)


def tune(args: argparse.Namespace) -> Path:
    if args.iters <= 1:
        raise ValueError("iters must be greater than one")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    arch = get_rocm_arch()
    try:
        lds_cap_bytes = LDS_CAP_BYTES[arch]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported GPU architecture: {arch}") from exc
    s1_registry = get_flydsl_stage1_kernels_nvfp4_bf16(
        args.hidden_size, args.inter_dim, lds_cap_bytes=lds_cap_bytes
    )
    s2_registry = get_flydsl_stage2_kernels_nvfp4_bf16(
        args.hidden_size, args.inter_dim, lds_cap_bytes=lds_cap_bytes
    )
    visible_gpu_count = torch.accelerator.device_count()
    if visible_gpu_count < 1:
        raise RuntimeError("No visible GPUs available for tuning")
    gpu_count = visible_gpu_count

    result: dict[str, dict[str, int]] = {}
    summary = []
    total_failed = 0
    start = time.perf_counter()
    context = mp.get_context("spawn")
    with context.Pool(
        processes=gpu_count,
        initializer=_init_worker,
        initargs=(gpu_count,),
    ) as pool:
        for tokens in args.tokens:
            tasks = [
                _make_task(args, tokens, "stage1", name, params)
                for name, params in s1_registry.items()
            ]
            tasks.extend(
                _make_task(args, tokens, "stage2", name, params)
                for name, params in s2_registry.items()
            )
            print(
                f"[nvfp4-tune] M={tokens}: dispatching {len(tasks)} candidates "
                f"across {gpu_count} visible GPU(s)",
                flush=True,
            )
            token_results = []
            candidates = pool.imap_unordered(_benchmark_candidate, tasks)
            for completed in range(1, len(tasks) + 1):
                try:
                    candidate_result = candidates.next(timeout=args.timeout)
                except mp.TimeoutError as exc:
                    raise RuntimeError(
                        f"M={tokens} made no tuning progress for {args.timeout}s"
                    ) from exc
                token_results.append(candidate_result)
                if args.verbose:
                    info, us, error = candidate_result
                    print(
                        f"[nvfp4-tune] completed {info[1]} {info[2]}: "
                        f"{error or f'{us:.2f}us'}",
                        flush=True,
                    )
                if completed % 20 == 0 or completed == len(tasks):
                    print(
                        f"[nvfp4-tune] M={tokens}: progress {completed}/{len(tasks)}",
                        flush=True,
                    )

            best_stage1: dict[int, tuple[float, dict[str, int]]] = {}
            best_stage2: dict[int, tuple[float, dict[str, int | str]]] = {}
            latencies: dict[str, dict[tuple[tuple[str, int | str], ...], float]] = {
                "stage1": {},
                "stage2": {},
            }
            failed = 0
            for info, us, error in token_results:
                _, stage, name, params = info
                if error is not None or not math.isfinite(us) or us <= 0:
                    failed += 1
                    print(
                        f"[nvfp4-tune] M={tokens} {stage} failed: "
                        f"{name}: {error or f'invalid time {us}'}",
                        flush=True,
                    )
                    continue
                latencies[stage][_params_key(params)] = us
                tile_m = int(params["tile_m"])
                stage_best = best_stage1 if stage == "stage1" else best_stage2
                if tile_m not in stage_best or us < stage_best[tile_m][0]:
                    stage_best[tile_m] = (us, params)

            total_failed += failed
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
                    "No valid matching FlyDSL NVFP4 stage-one/stage-two "
                    f"candidate completed for token count {tokens}."
                )

            default = FlydslNvfp4Experts._get_default_bf16_nvfp4_fused_moe_params(
                tokens,
                args.topk,
                args.experts,
                args.hidden_size,
                args.inter_dim,
            )
            default_stage1 = {
                key: default[key] for key in ("tile_m", "tile_n", "tile_k", "k_batch")
            }
            default_stage2 = {
                "tile_m": default["tile_m"],
                "tile_n": default["tile_n2"],
                "tile_k": default["tile_k2"],
                "mode": "atomic",
            }
            default_s1_us = latencies["stage1"].get(_params_key(default_stage1))
            default_s2_us = latencies["stage2"].get(_params_key(default_stage2))
            default_us = (
                default_s1_us + default_s2_us
                if default_s1_us is not None and default_s2_us is not None
                else None
            )
            best_us, best_config = best
            result[str(tokens)] = best[1]
            default_text = "failed" if default_us is None else f"{default_us:.2f}us"
            speedup_text = (
                "n/a" if default_us is None else f"{default_us / best_us:.3f}x"
            )
            print(
                f"[nvfp4-tune] M={tokens}: default={default_text}, "
                f"best={best_us:.2f}us, speedup={speedup_text}, "
                f"selected={best_config}; "
                f"{failed}/{len(token_results)} candidates failed",
                flush=True,
            )
            summary.append((tokens, default_us, best_us, speedup_text, best_config))

    print(
        f"[nvfp4-tune] completed in {time.perf_counter() - start:.2f}s; "
        f"{total_failed} candidates failed",
        flush=True,
    )
    print("[nvfp4-tune] summary", flush=True)
    print("M\tdefault_us\tbest_us\tspeedup\tbest_config", flush=True)
    for tokens, default_us, best_us, speedup, best_config in summary:
        default_text = "failed" if default_us is None else f"{default_us:.2f}"
        print(
            f"{tokens}\t{default_text}\t{best_us:.2f}\t{speedup}\t{best_config}",
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
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for each candidate",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    print(tune(parser.parse_args()))


if __name__ == "__main__":
    main()
