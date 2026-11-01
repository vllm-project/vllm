# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tune native per-token FP8 GEMM with transposed weights and BF16 output.

Example:
    uv run --no-project --python .venv/bin/python .venv/bin/python \
        benchmarks/kernels/benchmark_w8a8_per_token_fp8.py \
        --model MODEL_ID --tp-size 1 --gpus 0 1 \
        --save-path ./fp8-configs

The default search sweeps powers-of-two M from 1 through 32768, refines the
best tiles, and confirms winners against the production RDNA heuristic in
seven paired rounds. Model IDs select matrix workloads from weight_shapes.py;
model configuration and weights are not loaded. Expert kernels are separate.
Omit --save-path to print results without exporting tuning files.

"""

import gc
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from weight_shapes import TP_WEIGHT_SHAPES, WEIGHT_SHAPES

from vllm.model_executor.layers.quantization.utils import (
    fp8_per_token_heuristic as heuristic,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    get_w8a8_per_token_fp8_config_filename,
    get_w8a8_per_token_fp8_configs,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.platform_utils import get_device_name_as_file_name, num_compute_units

BATCH_SIZES = tuple(2**i for i in range(16))


def get_weight_shapes(model, tp_size):
    """Read (N, K) GEMM dimensions from the shared shape catalog."""
    if tp_size < 1:
        raise ValueError("TP size must be positive")
    if model in TP_WEIGHT_SHAPES:
        if tp_size not in TP_WEIGHT_SHAPES[model]:
            raise ValueError(f"No matrix shapes recorded for {model} at TP={tp_size}")
        return sorted({(n, k) for k, n in TP_WEIGHT_SHAPES[model][tp_size]})
    if model not in WEIGHT_SHAPES:
        raise ValueError(f"Add {model}'s matrix shapes to weight_shapes.py first")
    shapes = set()
    for dimensions, split_dim in WEIGHT_SHAPES[model]:
        dimensions = list(dimensions)
        if dimensions[split_dim] % tp_size:
            raise ValueError("TP size must divide the matrix's partitioned dimension")
        dimensions[split_dim] //= tp_size
        k, n = dimensions
        shapes.add((n, k))
    return sorted(shapes)


def launch_config(config):
    return {
        "engine": "shared",
        "block_size_m": config["BLOCK_SIZE_M"],
        "block_size_n": config["BLOCK_SIZE_N"],
        "block_size_k": config["BLOCK_SIZE_K"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
        "group_m": config.get("GROUP_M", 1),
        "n_stripe": config.get("N_STRIPE", 0),
        "unmasked_loads": bool(config.get("UNMASKED_LOADS", False)),
    }


def staged_candidates(heuristic, previous):
    keys = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps", "num_stages")
    choices = [
        launch_config(dict(zip(keys, values)))
        for values in itertools.product(
            (16, 32, 64, 128), (32, 64, 128), (64, 128, 256), (4, 8), (1, 2, 3)
        )
    ]
    for bm, bn, bk in [(128, 256, 64), (128, 256, 128), (256, 64, 64), (256, 128, 64)]:
        for warps, stages in [(4, 1), (4, 2), (8, 1), (8, 2)]:
            choices.append(
                launch_config(
                    dict(
                        BLOCK_SIZE_M=bm,
                        BLOCK_SIZE_N=bn,
                        BLOCK_SIZE_K=bk,
                        num_warps=warps,
                        num_stages=stages,
                    )
                )
            )
    if previous is not None:
        choices.append(previous)
    if heuristic is not None:
        choices.append(
            launch_config(
                dict(
                    BLOCK_SIZE_M=heuristic["block_size_m"],
                    BLOCK_SIZE_N=heuristic["block_size_n"],
                    BLOCK_SIZE_K=heuristic["block_size_k"],
                    num_warps=heuristic.get("num_warps", 4),
                    num_stages=heuristic.get("num_stages", 2),
                )
            )
        )
    return choices


def staged_refinements(ranked, m, n, k):
    choices = []
    for row in ranked[:4]:
        config = row["config"]
        for group, stripe in [(2, 0), (4, 0), (8, 0), (1, 2), (1, 4), (1, 8), (1, 16)]:
            choices.append(dict(config, group_m=group, n_stripe=stripe))
        if (
            m % config["block_size_m"]
            == n % config["block_size_n"]
            == k % config["block_size_k"]
            == 0
        ):
            choices.append(dict(config, unmasked_loads=True))
    return choices


def select_staged_config(record):
    timings, medians = record["samples_us"], record["median_us"]
    winner = min((name for name in medians if name != "heuristic"), key=medians.get)
    config = (
        record["old_tuned"]
        if winner == "old_tuned"
        else record["ranked"][int(winner.removeprefix("candidate_"))]["config"]
    )
    baseline, best = medians["heuristic"], medians[winner]
    wins = sum(a < b for a, b in zip(timings[winner], timings["heuristic"]))
    accepted = best < baseline * 0.98 and baseline - best > 0.15 and wins >= 5
    exported = {"USE_HEURISTIC": True}
    if accepted:
        exported = dict(
            BLOCK_SIZE_M=config["block_size_m"],
            BLOCK_SIZE_N=config["block_size_n"],
            BLOCK_SIZE_K=config["block_size_k"],
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
            EXACT_M=True,
        )
        if config.get("group_m", 1) != 1:
            exported["GROUP_M"] = config["group_m"]
        if config.get("n_stripe", 0):
            exported["N_STRIPE"] = config["n_stripe"]
        if config.get("unmasked_loads", False):
            exported["UNMASKED_LOADS"] = True
    return dict(
        config=exported,
        winner=winner,
        paired_wins=wins,
        accepted=accepted,
        heuristic_us=baseline,
        tuned_us=best if accepted else baseline,
        speedup=baseline / best if accepted else 1.0,
    )


def staged_search(a, b, sa, sb, previous, num_processors, cache_size_bytes, out_dtype):
    m, k = a.shape
    n = b.size(1)
    inputs = a, b, sa, sb
    use_heuristic = heuristic.can_use_fp8_per_token_heuristic(
        a, b, sa, sb, out_dtype, None
    )
    choice = (
        heuristic.get_heuristic_config(m, n, k, num_processors, cache_size_bytes)
        if use_heuristic
        else None
    )
    previous = (
        launch_config(previous)
        if previous and not previous.get("USE_HEURISTIC")
        else None
    )

    def run(config, operands=inputs):
        if use_heuristic:
            return heuristic.launch_fp8_per_token(*operands, out_dtype, None, config)
        if config is None:
            return heuristic.shared_mm.triton_scaled_mm(*operands, out_dtype)
        return heuristic.shared_mm.triton_scaled_mm(
            *operands,
            out_dtype,
            use_heuristic=False,
            block_size_m=config["block_size_m"],
            block_size_n=config["block_size_n"],
            block_size_k=config["block_size_k"],
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
            group_m=config.get("group_m", 1) if m > 1 and n > 1 else 1,
            n_stripe=config.get("n_stripe", 0),
            unmasked_loads=config.get("unmasked_loads", False),
        )

    def check_output(value, operands, full=False):
        lhs, rhs, scale_a, scale_b = operands
        rows = torch.linspace(0, m - 1, min(m, 16), device=a.device).long().unique()
        cols = torch.linspace(0, n - 1, min(n, 32), device=a.device).long().unique()
        ref = (
            (lhs[rows].float() @ rhs[:, cols].float())
            * scale_a[rows]
            * scale_b.flatten()[cols]
        )
        torch.testing.assert_close(
            value[rows[:, None], cols[None, :]], ref.to(out_dtype), rtol=1e-2, atol=1e-2
        )
        if full:
            for chunk in value.reshape(-1).split(16 * 1024 * 1024):
                if not torch.isfinite(chunk).all().item():
                    raise ValueError("Non-finite output")
            if out_dtype == torch.float16:
                rhs_fp32 = rhs.float()
                for start in range(0, m, 128):
                    ref = (
                        (lhs[start : start + 128].float() @ rhs_fp32)
                        * scale_a[start : start + 128]
                        * scale_b.t()
                    )
                    torch.testing.assert_close(
                        value[start : start + 128],
                        ref.to(out_dtype),
                        rtol=1e-2,
                        atol=1e-2,
                    )
                return
            ref_a, ref_b, ref_scale_b = lhs, rhs, scale_b
            if n % 16 or k % 16:
                padded_n, padded_k = triton.cdiv(n, 16) * 16, triton.cdiv(k, 16) * 16
                ref_a = lhs.new_zeros((m, padded_k))
                weights = rhs.new_zeros((padded_n, padded_k))
                ref_scale_b = scale_b.new_ones((padded_n, 1))
                ref_a[:, :k].copy_(lhs)
                weights[:n, :k].copy_(rhs.t())
                ref_scale_b[:n].copy_(scale_b)
                ref_b = weights.t()
            ref = torch._scaled_mm(
                ref_a,
                ref_b,
                scale_a=scale_a,
                scale_b=ref_scale_b.t(),
                out_dtype=out_dtype,
            )
            if isinstance(ref, tuple):
                ref = ref[0]
            ref = ref[:, :n]
            for start in range(0, m, 128):
                torch.testing.assert_close(
                    value[start : start + 128],
                    ref[start : start + 128],
                    rtol=1e-2,
                    atol=1e-2,
                )

    def graph_for(config):
        for _ in range(2):
            value = run(config)
        torch.accelerator.synchronize()
        calls = 8 if m * n * k < 1_000_000_000 else 1
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(calls):
                value = run(config)
        graph.replay()
        return graph, value, calls

    def graph_us(item, repeats=3):
        graph, _, calls = item
        samples = []
        for _ in range(repeats):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / calls)
        return statistics.median(samples)

    ranked, skipped, seen = [], [], set()

    def test(config):
        key = json.dumps(config, sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        try:
            check_output(run(config), inputs)
            latency = graph_us(graph_for(config))
            ranked.append(dict(config=config, graph_us=latency))
        except (triton.CompilationError, triton.runtime.errors.OutOfResources) as exc:
            skipped.append(dict(config=config, error=str(exc)[-700:]))

    choices = staged_candidates(choice, previous)
    random.Random(291 + m + n + k).shuffle(choices)
    for config in choices:
        test(config)
    ranked.sort(key=lambda row: row["graph_us"])
    if not ranked:
        raise RuntimeError(f"No valid config for {(m, n, k)}")
    for config in staged_refinements(ranked, m, n, k):
        test(config)
    ranked.sort(key=lambda row: row["graph_us"])
    finalists = {"heuristic": choice}
    if previous is not None:
        finalists["old_tuned"] = previous
    finalists.update(
        {f"candidate_{i}": row["config"] for i, row in enumerate(ranked[:8])}
    )
    graphs = {name: graph_for(config) for name, config in finalists.items()}
    samples = {name: [] for name in graphs}
    for rep in range(7):
        order = list(graphs)
        random.Random(849 + m + n + k + rep).shuffle(order)
        for name in order:
            samples[name].append(graph_us(graphs[name], 5))
    record = dict(
        M=m,
        N=n,
        K=k,
        heuristic=choice,
        uses_rdna_heuristic=use_heuristic,
        old_tuned=previous,
        median_us={name: statistics.median(t) for name, t in samples.items()},
        samples_us=samples,
        ranked=ranked,
        skipped=skipped,
    )
    record.update(select_staged_config(record))
    selected = launch_config(record["config"]) if record["accepted"] else choice
    check_output(run(selected), inputs, full=True)
    torch.manual_seed(712927 + m + n + k)
    holdout = (
        (0.2 * torch.randn(m, k, device=a.device)).to(a.dtype),
        (0.2 * torch.randn(n, k, device=a.device)).to(a.dtype).t(),
        torch.rand(m, 1, device=a.device) + 0.5,
        torch.rand(n, 1, device=a.device) + 0.5,
    )
    check_output(run(selected, holdout), holdout, full=True)
    record["correctness"] = (
        "full Torch + sampled FP32, search and independent input seeds"
    )
    return record


def tune_on_gpu(task):
    gpu, shapes, args, run_id = task
    torch.accelerator.set_device_index(gpu)
    from vllm.platforms.rocm import on_rdna

    if not current_platform.supports_fp8() or not on_rdna():
        raise RuntimeError("Tuning requires an FP8-capable RDNA GPU")
    torch.set_num_threads(1)
    out_dtype = torch.bfloat16
    dtype = current_platform.fp8_dtype()
    device = current_platform.device_type
    device_name = get_device_name_as_file_name(gpu)
    processors = num_compute_units(gpu)
    cache = heuristic.get_device_cache_size(gpu)
    references = {
        f"{n},{k}": get_w8a8_per_token_fp8_configs(n, k, device_name, dtype, out_dtype)
        or {}
        for n, k in shapes
    }
    folder = (
        args.save_path / "measurements" / f"{run_id}-gpu{gpu}"
        if args.save_path
        else None
    )
    if folder:
        folder.mkdir(parents=True)
        sources = [
            Path(__file__),
            Path(__file__).with_name("weight_shapes.py"),
            Path(heuristic.__file__),
            Path(get_w8a8_per_token_fp8_configs.__wrapped__.__code__.co_filename),
            Path(heuristic.shared_mm.__file__),
        ]
        metadata = dict(
            model=args.model,
            tp_size=args.tp_size,
            gpu=gpu,
            device=str(getattr(torch, device).get_device_properties(gpu)),
            torch=torch.__version__,
            hip=torch.version.hip,
            triton=triton.__version__,
            output_dtype="bfloat16",
            num_processors=processors,
            cache_size_bytes=cache,
            batch_sizes=BATCH_SIZES,
            shapes=shapes,
            rounds=7,
            finalists=8,
            reference_configs=references,
            tensor_descriptor_override=os.environ.get("VLLM_TRITON_USE_TD"),
            source_sha256={
                str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
            },
        )
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    counts = dict(tuned=0, heuristic=0)
    for n, k in shapes:
        configs = {}
        table = references[f"{n},{k}"]
        for m in BATCH_SIZES:
            torch.manual_seed(21994 + m + n + k)
            a = (0.2 * torch.randn(m, k, device=device)).to(dtype)
            b = (0.2 * torch.randn(n, k, device=device)).to(dtype).t()
            sa = torch.rand(m, 1, device=device) + 0.5
            sb = torch.rand(n, 1, device=device) + 0.5
            previous = (
                table[min(table, key=lambda value: abs(value - m))] if table else None
            )
            print(f"[GPU {gpu}] Searching M={m}, N={n}, K={k}", flush=True)
            record = staged_search(a, b, sa, sb, previous, processors, cache, out_dtype)
            configs[m] = record["config"]
            counts["tuned" if record["accepted"] else "heuristic"] += 1
            if folder:
                with (folder / "results.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
            print(
                f"[GPU {gpu}] M={m}, N={n}, K={k}: "
                f"{record['speedup']:.3f}x, {record['config']}",
                flush=True,
            )
            del a, b, sa, sb
            gc.collect()
            torch.accelerator.empty_cache()
        if args.save_path:
            name = get_w8a8_per_token_fp8_config_filename(
                n, k, device_name, dtype, out_dtype
            )
            target = args.save_path / name
            temporary = target.with_suffix(f".gpu{gpu}.tmp")
            temporary.write_text(json.dumps(configs, indent=4) + "\n")
            temporary.replace(target)
            print(f"Saved {target}", flush=True)
    return counts


def main(args):
    if not current_platform.is_rocm():
        raise ValueError("This tuner requires ROCm")
    available = torch.accelerator.device_count()
    gpus = args.gpus if args.gpus is not None else list(range(available))
    if (
        not gpus
        or len(set(gpus)) != len(gpus)
        or any(g < 0 or g >= available for g in gpus)
    ):
        raise ValueError("Choose distinct available GPU indices")
    if len({get_device_name_as_file_name(g) for g in gpus}) != 1:
        raise ValueError("Choose GPUs of the same model for parallel tuning")
    shapes = get_weight_shapes(args.model, args.tp_size)
    count = min(len(gpus), len(shapes))
    run_id = time.time_ns()
    tasks = [
        (gpu, shapes[i::count], args, run_id) for i, gpu in enumerate(gpus[:count])
    ]
    print(
        f"Model: {args.model}, TP={args.tp_size}, shapes={shapes}, GPUs={gpus[:count]}",
        flush=True,
    )
    with ProcessPoolExecutor(
        max_workers=count, mp_context=mp.get_context("spawn")
    ) as pool:
        results = list(pool.map(tune_on_gpu, tasks))
    print(
        f"Completed: {sum(r['tuned'] for r in results)} tuned, "
        f"{sum(r['heuristic'] for r in results)} heuristic fallbacks",
        flush=True,
    )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model ID from weight_shapes.py")
    parser.add_argument(
        "--tp-size",
        "-tp",
        type=int,
        default=1,
        help="Model tensor-parallel size (default: 1)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        help="Tuning GPU indices (default: all visible GPUs)",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        help="Export configs and measurements here; omit to print results only",
    )
    main(parser.parse_args())
