# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure MM launch candidates against a frozen segmented selector.

The manifest supplies absolute, inclusive image ranges for each sequence.
Numerical checks use sampled dense FP32 attention with the actual quantized KV.
Warm/cold graph timings are interleaved; output is resumable JSONL.
"""

import argparse
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import statistics
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import torch

from vllm.model_executor.warmup import rocm_segmented_attn_autotune_warmup as tuning
from vllm.v1.attention.backends.utils import compute_mm_prefix_range_tensor
from vllm.v1.attention.ops import segmented_attention as ops
from vllm.v1.attention.ops import segmented_attention_config as selection


def load_selector(path, name="vllm.v1.attention.ops._segmented_mm_baseline_config"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tuned_config(module, case, args, label):
    """Tune separate synthetic inputs, then exercise the real runtime lookup."""
    mode = "clamped" if case.get("clamp", True) else "unclamped"
    span = max(
        (last - first + 1 for rr in case["ranges"] for first, last in rr), default=0
    )
    ranges = max(map(len, case["ranges"]), default=0)
    if not span or not ranges:
        mode = "none"
    image_span, image_ranges = selection.segmented_mm_prefix_bucket(span, ranges)
    window = case.get("window", 1024) - 1
    attention_span = selection.segmented_attention_span(
        case["seq_len"], case["query_len"], window, mode, span
    )
    max_tokens = max(8192, case["batch"] * case["query_len"])
    if args.tune == "exact":
        workload = (case["batch"], case["query_len"], attention_span)
    else:
        max_len = (
            32768
            if window < 0
            else min(
                32768, 8192 + window + (max(0, image_span - 1) if mode != "none" else 0)
            )
        )
        eligible = [
            work
            for work in module._workloads(
                max_tokens,
                max_len,
                max(8, case["batch"]),
                mm_prefix_mode=mode,
                mm_prefix_max_span=image_span,
                mm_prefix_max_ranges=image_ranges,
                sliding_window=window,
            )
            if work[0] >= case["batch"]
            and work[1] >= case["query_len"]
            and work[2] >= attention_span
        ]
        if not eligible:
            raise ValueError(f"No startup workload covers {case}")
        workload = min(eligible, key=lambda w: (w[1], w[0], w[2]))
    dtype = torch.float8_e4m3fn if case["fp8"] else torch.bfloat16
    payload = dict(
        sources=args.source_identity,
        label=label,
        workload=list(workload),
        heads=case["heads"],
        kv_heads=case["kv_heads"],
        dim=case["dim"],
        page=case["page"],
        fp8=case["fp8"],
        scale=case.get("scale", 1.0),
        window=window,
        causal=case.get("causal", True),
        max_tokens=max_tokens,
        table_max_len=32768,
        mm_prefix_mode=mode,
        mm_prefix_max_span=image_span,
        mm_prefix_max_ranges=image_ranges,
    )
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    folder = args.output.parent / (args.output.stem + "-tuning") / label
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{digest}.json"
    reused = path.exists()
    if reused:
        saved = json.loads(path.read_text())
        assert saved["payload"] == payload
    else:
        started = time.monotonic()
        with module._temporary_tuning_kernels(torch.device("cuda:0")):
            record = module._tune_workload(
                "cuda:0",
                torch.bfloat16,
                dtype,
                case["heads"],
                case["kv_heads"],
                case["dim"],
                case["page"],
                case.get("scale", 1.0),
                max_tokens,
                workload,
                sliding_window=window,
                causal=case.get("causal", True),
                table_max_len=32768,
                mm_prefix_mode=mode,
                mm_prefix_max_span=image_span,
                mm_prefix_max_ranges=image_ranges,
            )
        saved = dict(
            payload=payload, record=record, elapsed_s=time.monotonic() - started
        )
        path.write_text(json.dumps(saved, indent=2) + "\n")
    key = module._key(
        "cuda:0",
        torch.bfloat16,
        dtype,
        case["heads"],
        case["kv_heads"],
        case["dim"],
        case["page"],
        case.get("scale", 1.0),
        window,
        case.get("causal", True),
        False,
        mode,
        image_span,
        image_ranges,
    )
    module._TABLES[key] = dict(identity=payload, records=[saved["record"]])
    config = module.get_segmented_config(
        "cuda:0",
        torch.bfloat16,
        dtype,
        case["heads"],
        case["kv_heads"],
        case["dim"],
        case["page"],
        case.get("scale", 1.0),
        case["batch"],
        case["query_len"],
        attention_span,
        window,
        case.get("causal", True),
        mm_prefix_mode=mode,
        mm_prefix_max_span=span,
        mm_prefix_max_ranges=ranges,
        num_query_tokens=sum(
            case.get("query_lengths", [case["query_len"]] * case["batch"])
        ),
    )
    tuned_hit = config is not None
    if config is None:
        config = module._normalized_config(
            select(selection, case, updated=True),
            case["batch"],
            case["query_len"],
            case["heads"],
            case["dim"],
        )
    return config, dict(
        workload=list(workload),
        profile_span=image_span,
        profile_ranges=image_ranges,
        elapsed_s=saved["elapsed_s"],
        reused=reused,
        tuned_hit=tuned_hit,
        path=str(path),
        search=saved["record"]["search"],
    )


def select(module, case, *, updated=False):
    window = case.get("window", 1024) - 1
    options = dict(sliding_window=window)
    if updated:
        # Filled by the production MM selector once its workload API is present.
        import inspect

        if (
            "mm_prefix_mode"
            in inspect.signature(module.select_segmented_config).parameters
        ):
            options.update(
                mm_prefix_mode="clamped" if case.get("clamp", True) else "unclamped",
                mm_prefix_max_span=max(
                    (
                        end - start + 1
                        for ranges in case["ranges"]
                        for start, end in ranges
                    ),
                    default=0,
                ),
                mm_prefix_max_ranges=max(map(len, case["ranges"]), default=0),
            )
    seq_len = case["seq_len"]
    if not any(case["ranges"]):
        seq_len = selection.segmented_attention_span(seq_len, case["query_len"], window)
        if updated:
            options.update(
                mm_prefix_mode="none", mm_prefix_max_span=0, mm_prefix_max_ranges=0
            )
    return module.select_segmented_config(
        case["batch"],
        case["query_len"],
        seq_len,
        case["heads"],
        case["kv_heads"],
        case["dim"],
        case["fp8"],
        **options,
    )


def broad_candidates(default, case):
    batch, query, dim = case["batch"], case["query_len"], case["dim"]
    native = case["fp8"]
    configs = tuning._candidate_configs(
        default,
        batch,
        query,
        case["heads"],
        dim,
        seq_len=case["seq_len"],
        native_fp8=native,
    )
    seed = tuning._normalized_config(default, batch, query, case["heads"], dim)
    tiles = (
        (16, 32, 64, 4, 1),
        (16, 64, 64, 4, 1),
        (32, 32, 64, 4, 1),
        (32, 64, 64, 4, 1),
        (32, 128, 64, 4, 1),
        (32, 64, dim, 4, 1),
        (32, 64, dim, 8, 1),
        (32, 128, 64, 8, 1),
        (64, 32, 64, 4, 1),
        (64, 64, 64, 4, 1),
        (64, 64, 64, 8, 1),
        (64, 128, 64, 8, 1),
        (64, 32, dim, 4, 1),
        (64, 64, dim, 8, 1),
        (64, 64, 64, 8, 2),
        (64, 128, 64, 8, 3),
        (128, 32, 64, 4, 1),
        (128, 64, 64, 8, 1),
        (128, 32, dim, 8, 1),
        (128, 64, dim, 8, 1),
    )
    split_choices = sorted({1, min(4, seed["splits"]), seed["splits"]})
    for bm, bn, bk, warps, stages in tiles:
        if bk > dim or (query <= 8 and bm > 32):
            continue
        for splits in split_choices:
            for mode in selection.KV_UPCAST_MODES if native else ("normal",):
                configs.append(
                    {
                        **seed,
                        "bm": bm,
                        "bn": bn,
                        "bk": bk,
                        "warps": warps,
                        "stages": stages,
                        "splits": splits,
                        "waves_per_eu": 2 if bm <= 32 else 6,
                        "pv_split": False,
                        "prefix_fast": False,
                        "qk_pipeline": None,
                        "qk_unroll": None,
                        "kv_upcast": mode,
                    }
                )
    return list({tuning._config_key(c): c for c in configs}.values())


@torch.inference_mode()
def make_inputs(case, seed):
    batch, query, length = case["batch"], case["query_len"], case["seq_len"]
    heads, kv_heads, dim, page = (case[k] for k in ("heads", "kv_heads", "dim", "page"))
    queries = case.get("query_lengths", [query] * batch)
    lengths = case.get("seq_lengths", [length] * batch)
    generator = torch.Generator(device="cuda:0").manual_seed(seed)
    q = (
        torch.randn(
            sum(queries),
            heads,
            dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.25
    )
    blocks = math.ceil(max(lengths) / page)
    cache_shape = (batch * blocks, 2, page, kv_heads, dim)
    dtype = torch.float8_e4m3fn if case["fp8"] else torch.bfloat16
    scale = torch.tensor(0.125 if case["fp8"] else 1.0, device="cuda")
    cache = (
        torch.randn(cache_shape, device="cuda", generator=generator) * 0.25 / scale
    ).to(dtype)
    kc, vc = cache.unbind(1)
    table = (
        torch.randperm(batch * blocks, device="cuda", generator=generator)
        .to(torch.int32)
        .view(batch, blocks)
    )
    starts_cpu = [0, *itertools.accumulate(queries)]
    starts = torch.tensor(starts_cpu, device="cuda", dtype=torch.int32)
    lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    ranges = {i: [tuple(r) for r in rr] for i, rr in enumerate(case["ranges"])}
    mm = compute_mm_prefix_range_tensor(ranges, batch, q.device)
    samples, expected = [], []
    for seq, (nq, seq_len) in enumerate(zip(queries, lengths, strict=True)):
        if not nq:
            continue
        key = kc[table[seq].long()].flatten(0, 1)[:seq_len].float() * scale
        value = vc[table[seq].long()].flatten(0, 1)[:seq_len].float() * scale
        key = key.repeat_interleave(heads // kv_heads, dim=1)
        value = value.repeat_interleave(heads // kv_heads, dim=1)
        positions = {0, nq // 4, nq // 2, nq - 1}
        prefix = seq_len - nq
        for start, end in ranges[seq]:
            positions.update(
                p - prefix for p in (start, start + 1, end) if prefix <= p < seq_len
            )
        keys = torch.arange(seq_len, device="cuda")
        window = case.get("window", 1024)
        for position in sorted(positions):
            absolute = prefix + position
            visible = (
                keys <= absolute
                if case.get("causal", True)
                else torch.ones_like(keys, dtype=torch.bool)
            )
            if window:
                visible &= keys > absolute - window
                if not case.get("causal", True):
                    visible &= keys < absolute + window
            for start, end in ranges[seq]:
                if start < end and start <= absolute <= end:
                    image = (keys >= start) & (keys <= end)
                    if window and case.get("clamp", True):
                        image &= keys > absolute - window
                    visible |= image
            row = starts_cpu[seq] + position
            scores = torch.einsum("hd,khd->hk", q[row].float(), key)
            scores *= case.get("scale", 1.0)
            scores.masked_fill_(~visible[None], -float("inf"))
            expected.append(torch.einsum("hk,khd->hd", scores.softmax(-1), value))
            samples.append(row)
    return dict(
        q=q,
        kc=kc,
        vc=vc,
        table=table,
        starts=starts,
        lens=lens,
        scale=scale,
        mm=mm,
        samples=samples,
        expected=torch.stack(expected),
    )


@torch.inference_mode()
def benchmark(case, args, baseline):
    began = time.monotonic()
    inputs = make_inputs(case, case.get("seed", 419) + args.seed_offset)
    default = tuning._normalized_config(
        select(baseline, case),
        case["batch"],
        case["query_len"],
        case["heads"],
        case["dim"],
    )
    current = tuning._normalized_config(
        select(selection, case, updated=True),
        case["batch"],
        case["query_len"],
        case["heads"],
        case["dim"],
    )
    variants = {"baseline": default, "static": current}
    tune_results = {}
    if args.configs:
        frozen = json.loads(args.configs.read_text())
        supplied = frozen[str(case["index"])]
        if isinstance(supplied, dict):
            variants.update(supplied)
            configs = list(supplied.values())
        else:
            configs = supplied
    elif args.search == "broad":
        configs = broad_candidates(default, case)
    elif args.search == "compact":
        configs = tuning._candidate_configs(
            current,
            case["batch"],
            case["query_len"],
            case["heads"],
            case["dim"],
            seq_len=case["seq_len"],
            native_fp8=case["fp8"],
            mm_prefix_mode="clamped" if case.get("clamp", True) else "unclamped",
            sliding_window=case.get("window", 1024) - 1,
        )
    else:
        configs = [default, current]
    if args.tune != "none":
        modules = [("compact_autotune", tuning)]
        if args.full_tuner is not None:
            modules.append(("full_autotune", args.full_tuner_module))
        if (case["index"] // 2) % 2:
            modules.reverse()
        for label, module in modules:
            config, result = tuned_config(module, case, args, label)
            variants[label] = config
            tune_results[label] = result
            configs.append(config)
    if any(case["ranges"]):

        def canonical(config):
            return tuning._canonical_mm_config(
                config, case["dim"], case.get("window", 1024) - 1
            )

        default, current = canonical(default), canonical(current)
        configs = [canonical(c) for c in configs]
        variants = {name: canonical(c) for name, c in variants.items()}
    if args.baseline_kernel:
        default = dict(default, _benchmark_kernel="baseline")
        current = dict(current, _benchmark_kernel="current")
        configs = [dict(c, _benchmark_kernel="current") for c in configs]
        variants = {
            name: dict(
                c, _benchmark_kernel="baseline" if name == "baseline" else "current"
            )
            for name, c in variants.items()
        }
        variants["baseline_config_new_kernel"] = dict(
            default, _benchmark_kernel="current"
        )
        configs.extend(variants.values())
    configs = list(
        {tuning._config_key(c): c for c in [default, current, *configs]}.values()
    )
    configs = tuning._rotated(configs, args.config_order_offset % len(configs))
    if args.reverse_config_order:
        configs.reverse()
    output = torch.empty_like(inputs["q"])
    shapes = ops.segmented_workspace_shapes(
        case["batch"],
        selection.segmented_query_capacity(case["query_len"]),
        case["heads"],
        case["kv_heads"],
        case["dim"],
        max(c["splits"] for c in configs),
    )
    workspace = (
        None
        if shapes is None
        else tuple(torch.empty(s, device="cuda", dtype=torch.float32) for s in shapes)
    )
    options = dict(
        sliding_window=case.get("window", 1024) - 1,
        causal=case.get("causal", True),
        mm_prefix_range=inputs["mm"],
        mm_prefix_clamp_sliding_window=case.get("clamp", True),
    )

    def run(config):
        module = (
            args.baseline_kernel_module
            if config.get("_benchmark_kernel") == "baseline"
            else ops
        )
        module.run_segmented_attention(
            inputs["q"],
            output,
            inputs["kc"],
            inputs["vc"],
            inputs["table"],
            inputs["starts"],
            inputs["lens"],
            case["query_len"],
            case["seq_len"],
            inputs["scale"],
            inputs["scale"],
            case.get("scale", 1.0),
            config=config,
            workspace=workspace,
            **options,
        )

    baseline_context = (
        tuning._temporary_tuning_kernels(
            torch.device("cuda:0"),
            kernels=(
                args.baseline_kernel_module._segmented_attention_stage,
                args.baseline_kernel_module._segmented_attention_reduce,
            ),
        )
        if args.baseline_kernel
        else nullcontext()
    )
    with tuning._temporary_tuning_kernels(torch.device("cuda:0")), baseline_context:
        from concurrent.futures import ThreadPoolExecutor

        from vllm.triton_utils import triton

        with (
            ThreadPoolExecutor(max_workers=2) as executor,
            triton.AsyncCompileMode(executor, ignore_errors=True),
        ):
            for config in configs:
                module = (
                    args.baseline_kernel_module
                    if config.get("_benchmark_kernel") == "baseline"
                    else ops
                )
                module.compile_segmented_attention(
                    inputs["q"],
                    output,
                    inputs["kc"],
                    inputs["vc"],
                    inputs["table"],
                    inputs["starts"],
                    inputs["lens"],
                    case["query_len"],
                    inputs["scale"],
                    inputs["scale"],
                    case.get("scale", 1.0),
                    config,
                    workspace,
                    **options,
                )
        compile_s = time.monotonic() - began
        print(
            json.dumps(
                dict(
                    index=case["index"],
                    phase="compiled",
                    candidates=len(configs),
                    elapsed_s=compile_s,
                )
            ),
            flush=True,
        )
        valid, rejected = [], []
        for config in configs:
            try:
                output.fill_(float("nan"))
                run(config)
                torch.accelerator.synchronize()
                assert bool(torch.isfinite(output).all()), "Nonfinite output"
                relative = (
                    output[inputs["samples"]].float() - inputs["expected"]
                ).norm(dim=-1)
                relative /= inputs["expected"].norm(dim=-1).clamp_min(1e-6)
                assert relative.max().item() < 0.01, (
                    f"Dense error {relative.max().item()}"
                )
                valid.append(config)
            except (
                RuntimeError,
                AssertionError,
                triton.OutOfResources,
                triton.CompilationError,
            ) as error:
                rejected.append(dict(config=config, error=str(error)[:1200]))
        assert default in valid and current in valid, (
            "Selector failed numerical validation"
        )
        checked_s = time.monotonic() - began
        eviction = torch.empty(256 * 1024**2, device="cuda", dtype=torch.uint8)
        tuning._condition_configs(valid, run, torch.device("cuda:0"))
        cold = tuning._measure_configs(valid, run, "cuda:0", eviction, args.rounds, 1)
        estimate = statistics.median(cold[tuning._config_key(default)])
        calls = min(128, max(2, math.ceil(1000 / estimate)))
        warm = tuning._measure_configs(
            valid, run, "cuda:0", eviction, args.rounds, calls
        )
        winner, comparisons = tuning._select_tuned_winner(
            default, valid, cold, warm_samples=warm
        )
        rows = []
        for comparison in comparisons:
            key = tuning._config_key(comparison["config"])
            rows.append(
                dict(
                    comparison,
                    cold_us=cold[key],
                    warm_us=warm[key],
                    warm_median_us=statistics.median(warm[key]),
                )
            )
    return dict(
        case=case,
        status="ok",
        default=default,
        current=current,
        winner=winner["config"],
        measurements=rows,
        rejected=rejected,
        available_candidates=len(configs),
        variants=variants,
        tuning=tune_results,
        compile_s=compile_s,
        checked_s=checked_s,
        elapsed_s=time.monotonic() - began,
        warm_graph_calls=calls,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-selector", type=Path, required=True)
    parser.add_argument("--baseline-kernel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--search", choices=("broad", "compact", "static"), default="broad"
    )
    parser.add_argument("--configs", type=Path)
    parser.add_argument("--tune", choices=("none", "exact", "ceiling"), default="none")
    parser.add_argument("--full-tuner", type=Path)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--config-order-offset",
        type=int,
        default=0,
        help="Rotate kernel loading/capture positions in a fresh process.",
    )
    parser.add_argument(
        "--reverse-config-order",
        action="store_true",
        help="Repeat in a fresh process with reversed kernel loading/capture order.",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert torch.accelerator.device_count() == 1, "Expose physical GPU0 only"
    baseline = load_selector(args.baseline_selector)
    if args.baseline_kernel:
        args.baseline_kernel_module = load_selector(
            args.baseline_kernel, "vllm.v1.attention.ops._segmented_mm_baseline_kernel"
        )
    if args.full_tuner:
        args.full_tuner_module = load_selector(
            args.full_tuner, "vllm.model_executor.warmup._segmented_mm_full_tuner"
        )
    source_paths = [
        __file__,
        args.manifest,
        args.baseline_selector,
        ops.__file__,
        selection.__file__,
        tuning.__file__,
    ]
    if args.baseline_kernel:
        source_paths.append(args.baseline_kernel)
    snapshots = args.output.parent / "sources"
    snapshots.mkdir(parents=True, exist_ok=True)
    identity = {}
    for path in source_paths:
        source = Path(path).read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        identity[str(path)] = digest
        (snapshots / f"{digest}-{Path(path).name}").write_bytes(source)
    identity.update(
        gpu=torch.cuda.get_device_name(0),
        rounds=args.rounds,
        seed_offset=args.seed_offset,
        search=args.search,
        tune=args.tune,
        reverse_config_order=args.reverse_config_order,
        config_order_offset=args.config_order_offset,
    )
    if args.full_tuner:
        source = args.full_tuner.read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        identity[str(args.full_tuner)] = digest
        (snapshots / f"{digest}-{args.full_tuner.name}").write_bytes(source)
    if args.configs:
        identity[str(args.configs)] = hashlib.sha256(
            args.configs.read_bytes()
        ).hexdigest()
    metadata = args.output.with_suffix(".metadata.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if metadata.exists():
        assert json.loads(metadata.read_text()) == identity, "Resume source mismatch"
    else:
        metadata.write_text(json.dumps(identity, indent=2) + "\n")
    args.source_identity = identity
    done = (
        {
            json.loads(line)["case"]["index"]
            for line in args.output.read_text().splitlines()
        }
        if args.output.exists()
        else set()
    )
    with args.output.open("a", buffering=1) as output:
        for case in json.loads(args.manifest.read_text())["cases"]:
            if case["index"] in done:
                continue
            try:
                result = benchmark(case, args, baseline)
            except Exception as error:
                result = dict(
                    case=case,
                    status="failed",
                    error=repr(error),
                    traceback=traceback.format_exc(),
                )
            output.write(json.dumps(result) + "\n")
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in ("case", "status", "elapsed_s", "error")
                        if k in result
                    }
                ),
                flush=True,
            )
            gc.collect()
            torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
