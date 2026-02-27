#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Universal vLLM Attention Benchmark

Benchmark any attention backend with the extended grammar.
Supports standard attention (Flash/Triton/FlashInfer) and MLA backends.

Examples:
    # Standard attention
    python benchmark.py --backends flash flashinfer --batch-specs "q2k" "8q1s1k"

    # MLA backends
    python benchmark.py --backends cutlass_mla flashinfer_mla --batch-specs "64q1s1k"

    # Parameter sweep (CLI)
    python benchmark.py --backend cutlass_mla \
                        --batch-specs "64q1s1k" \
                        --sweep-param num_kv_splits \
                        --sweep-values 1 4 8 16

    # Parameter sweep (YAML config - recommended)
    python benchmark.py --config configs/cutlass_numsplits.yaml
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import yaml
from rich.console import Console
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from batch_spec import parse_batch_spec
from common import (
    BenchmarkConfig,
    BenchmarkResult,
    ModelParameterSweep,
    ParameterSweep,
    ResultsFormatter,
    batch_spec_sort_key,
    is_mla_backend,
)


def run_standard_attention_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run standard attention benchmark (Flash/Triton/FlashInfer)."""
    from runner import run_attention_benchmark

    return run_attention_benchmark(config)


def run_mla_benchmark(config: BenchmarkConfig, **kwargs) -> BenchmarkResult:
    """Run MLA benchmark with appropriate backend."""
    from mla_runner import run_mla_benchmark as run_mla

    return run_mla(config.backend, config, **kwargs)


def run_benchmark(config: BenchmarkConfig, **kwargs) -> BenchmarkResult:
    """
    Run a single benchmark with proper backend selection.

    Args:
        config: BenchmarkConfig with backend, batch_spec, and model params
        **kwargs: Additional arguments passed to MLA benchmarks

    Returns:
        BenchmarkResult (may have error field set on failure)
    """
    try:
        if is_mla_backend(config.backend):
            return run_mla_benchmark(config, **kwargs)
        else:
            return run_standard_attention_benchmark(config)
    except Exception as e:
        return BenchmarkResult(
            config=config,
            mean_time=float("inf"),
            std_time=0,
            min_time=float("inf"),
            max_time=float("inf"),
            error=str(e),
        )


def run_model_parameter_sweep(
    backends: list[str],
    batch_specs: list[str],
    base_config_args: dict,
    sweep: ModelParameterSweep,
    console: Console,
) -> list[BenchmarkResult]:
    """
    Run model parameter sweep for given backends and batch specs.

    Args:
        backends: List of backend names
        batch_specs: List of batch specifications
        base_config_args: Base configuration arguments (num_layers, head_dim, etc.)
        sweep: ModelParameterSweep configuration
        console: Rich console for output

    Returns:
        List of BenchmarkResult objects
    """
    all_results = []

    console.print(
        f"[yellow]Model sweep mode: testing {sweep.param_name} = {sweep.values}[/]"
    )

    total = len(backends) * len(batch_specs) * len(sweep.values)

    with tqdm(total=total, desc="Benchmarking") as pbar:
        for backend in backends:
            for spec in batch_specs:
                for value in sweep.values:
                    # Create config with modified model parameter
                    config_args = base_config_args.copy()
                    config_args[sweep.param_name] = value

                    # Create config with original backend for running
                    clean_config = BenchmarkConfig(
                        backend=backend, batch_spec=spec, **config_args
                    )

                    # Run benchmark
                    result = run_benchmark(clean_config)

                    # Replace backend with labeled version for display
                    backend_label = sweep.get_label(backend, value)
                    labeled_config = replace(result.config, backend=backend_label)
                    result = replace(result, config=labeled_config)
                    all_results.append(result)

                    if not result.success:
                        console.print(
                            f"[red]Error {backend} {spec} {sweep.param_name}="
                            f"{value}: {result.error}[/]"
                        )

                    pbar.update(1)

    # Display sweep results - create separate table for each parameter value
    console.print("\n[bold green]Model Parameter Sweep Results:[/]")
    formatter = ResultsFormatter(console)

    # Group results by parameter value and extract backend mapping
    by_param_value = {}
    backend_mapping = {}  # Maps labeled backend -> original backend

    for r in all_results:
        # Extract original backend and param value from labeled backend
        # The label format is: {backend}_{param_name}_{value}
        # We need to reverse engineer this
        labeled_backend = r.config.backend

        # Try each backend to find which one this result belongs to
        for backend in backends:
            for value in sweep.values:
                expected_label = sweep.get_label(backend, value)
                if labeled_backend == expected_label:
                    backend_mapping[labeled_backend] = backend
                    param_value = str(value)

                    if param_value not in by_param_value:
                        by_param_value[param_value] = []
                    by_param_value[param_value].append(r)
                    break

    # Create a table for each parameter value
    sorted_param_values = sorted(
        by_param_value.keys(), key=lambda x: int(x) if x.isdigit() else x
    )

    for param_value in sorted_param_values:
        console.print(f"\n[bold cyan]{sweep.param_name} = {param_value}[/]")
        param_results = by_param_value[param_value]

        # Create modified results with original backend names
        modified_results = []
        for r in param_results:
            # Get the original backend name from our mapping
            original_backend = backend_mapping[r.config.backend]
            modified_config = replace(r.config, backend=original_backend)
            modified_result = replace(r, config=modified_config)
            modified_results.append(modified_result)

        # Print table with original backend names
        formatter.print_table(modified_results, backends, compare_to_fastest=True)

    # Show optimal backend for each (param_value, batch_spec) combination
    console.print(
        f"\n[bold cyan]Optimal backend for each ({sweep.param_name}, batch_spec):[/]"
    )

    # Group by (param_value, batch_spec)
    by_param_and_spec = {}
    for r in all_results:
        if r.success:
            # Find which (backend, value) this result corresponds to
            labeled_backend = r.config.backend
            for backend in backends:
                for value in sweep.values:
                    expected_label = sweep.get_label(backend, value)
                    if labeled_backend == expected_label:
                        param_value = str(value)
                        spec = r.config.batch_spec
                        key = (param_value, spec)

                        if key not in by_param_and_spec:
                            by_param_and_spec[key] = []
                        by_param_and_spec[key].append(r)
                        break

    # Sort by param value then spec (batch_size, q_len, kv_len)
    sorted_keys = sorted(
        by_param_and_spec.keys(),
        key=lambda x: (
            int(x[0]) if x[0].isdigit() else x[0],
            batch_spec_sort_key(x[1]),
        ),
    )

    current_param_value = None
    for param_value, spec in sorted_keys:
        # Print header when param value changes
        if param_value != current_param_value:
            console.print(f"\n  [bold]{sweep.param_name}={param_value}:[/]")
            current_param_value = param_value

        results = by_param_and_spec[(param_value, spec)]
        best = min(results, key=lambda r: r.mean_time)

        # Extract original backend name using the mapping
        backend_name = backend_mapping[best.config.backend]

        # Show all backends' times for comparison
        times_str = " | ".join(
            [
                f"{backend_mapping[r.config.backend]}: {r.mean_time:.6f}s"
                for r in sorted(results, key=lambda r: r.mean_time)
            ]
        )

        console.print(
            f"    {spec:12s} -> [bold green]{backend_name:15s}[/] ({times_str})"
        )

    return all_results


def run_parameter_sweep(
    backends: list[str],
    batch_specs: list[str],
    base_config_args: dict,
    sweep: ParameterSweep,
    console: Console,
) -> list[BenchmarkResult]:
    """
    Run parameter sweep for given backends and batch specs.

    Args:
        backends: List of backend names
        batch_specs: List of batch specifications
        base_config_args: Base configuration arguments (num_layers, head_dim, etc.)
        sweep: ParameterSweep configuration
        console: Rich console for output

    Returns:
        List of BenchmarkResult objects
    """
    all_results = []

    # Build list of values to sweep (including auto if requested)
    sweep_values = list(sweep.values)
    if sweep.include_auto:
        sweep_values.append("auto")

    console.print(f"[yellow]Sweep mode: testing {sweep.param_name} = {sweep_values}[/]")

    total = len(backends) * len(batch_specs) * len(sweep_values)

    with tqdm(total=total, desc="Benchmarking") as pbar:
        for backend in backends:
            for spec in batch_specs:
                for value in sweep_values:
                    # Create config with original backend for running
                    config = BenchmarkConfig(
                        backend=backend, batch_spec=spec, **base_config_args
                    )

                    # Prepare kwargs for benchmark runner
                    kwargs = {}
                    if value != "auto":
                        kwargs[sweep.param_name] = value

                    # Run benchmark
                    result = run_benchmark(config, **kwargs)

                    # Replace backend with labeled version for display
                    backend_label = sweep.get_label(backend, value)
                    labeled_config = replace(result.config, backend=backend_label)
                    result = replace(result, config=labeled_config)
                    all_results.append(result)

                    if not result.success:
                        console.print(
                            f"[red]Error {backend} {spec} {sweep.param_name}="
                            f"{value}: {result.error}[/]"
                        )

                    pbar.update(1)

    # Display sweep results
    console.print("\n[bold green]Sweep Results:[/]")
    backend_labels = [sweep.get_label(b, v) for b in backends for v in sweep_values]
    formatter = ResultsFormatter(console)
    formatter.print_table(all_results, backend_labels)

    # Show optimal values
    console.print(f"\n[bold cyan]Optimal {sweep.param_name} per batch spec:[/]")
    by_spec = {}
    for r in all_results:
        if r.success:
            spec = r.config.batch_spec
            if spec not in by_spec:
                by_spec[spec] = []
            by_spec[spec].append(r)

    for spec in sorted(by_spec.keys(), key=batch_spec_sort_key):
        results = by_spec[spec]
        best = min(results, key=lambda r: r.mean_time)
        console.print(
            f"  {spec}: [bold green]{best.config.backend}[/] ({best.mean_time:.6f}s)"
        )

    return all_results


def load_config_from_yaml(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_batch_specs_from_ranges(ranges: list[dict]) -> list[str]:
    """
    Generate batch specs from range specifications.

    Args:
        ranges: List of range specifications, each containing:
            - template: Batch spec template (e.g., "q{q_len}kv1k")
            - q_len: Dict with start, stop, step, end_inclusive (optional)
            - Other parameters can also be ranges

    Returns:
        List of generated batch spec strings

    Example:
        ranges = [
            {
                "template": "q{q_len}kv1k",
                "q_len": {
                    "start": 1,
                    "stop": 16,
                    "step": 1,
                    "end_inclusive": true  # Optional, defaults to true
                }
            }
        ]
        Returns: ["q1kv1k", "q2kv1k", ..., "q16kv1k"]
    """
    all_specs = []

    for range_spec in ranges:
        template = range_spec.get("template")
        if not template:
            raise ValueError("Range specification must include 'template'")

        # Extract all range parameters from the spec
        range_params = {}
        for key, value in range_spec.items():
            if key == "template":
                continue
            if isinstance(value, dict) and "start" in value:
                # This is a range specification
                start = value["start"]
                stop = value["stop"]
                step = value.get("step", 1)
                # Check if end should be inclusive (default: True)
                end_inclusive = value.get("end_inclusive", True)

                # Adjust stop based on end_inclusive
                if end_inclusive:
                    range_params[key] = list(range(start, stop + 1, step))
                else:
                    range_params[key] = list(range(start, stop, step))
            else:
                # This is a fixed value
                range_params[key] = [value]

        # Generate all combinations (Cartesian product)
        if range_params:
            import itertools

            param_names = list(range_params.keys())
            param_values = [range_params[name] for name in param_names]

            for values in itertools.product(*param_values):
                params = dict(zip(param_names, values))
                spec = template.format(**params)
                all_specs.append(spec)
        else:
            # No parameters, just use template as-is
            all_specs.append(template)

    return all_specs


def main():
    parser = argparse.ArgumentParser(
        description="Universal vLLM attention benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config file
    parser.add_argument(
        "--config",
        help="Path to YAML config file (overrides other args)",
    )

    # Backend selection
    parser.add_argument(
        "--backends",
        nargs="+",
        help="Backends to benchmark (flash, triton, flashinfer, cutlass_mla, "
        "flashinfer_mla, flashattn_mla, flashmla)",
    )
    parser.add_argument(
        "--backend",
        help="Single backend (alternative to --backends)",
    )

    # Batch specifications
    parser.add_argument(
        "--batch-specs",
        nargs="+",
        default=["q2k", "8q1s1k"],
        help="Batch specifications using extended grammar",
    )

    # Model config
    parser.add_argument("--num-layers", type=int, default=10, help="Number of layers")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--num-q-heads", type=int, default=32, help="Query heads")
    parser.add_argument("--num-kv-heads", type=int, default=8, help="KV heads")
    parser.add_argument("--block-size", type=int, default=16, help="Block size")

    # Benchmark settings
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--repeats", type=int, default=1, help="Repetitions")
    parser.add_argument("--warmup-iters", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--profile-memory", action="store_true", help="Profile memory")

    # Parameter sweep (use YAML config for advanced sweeps)
    parser.add_argument(
        "--sweep-param",
        help="Parameter name to sweep (e.g., num_kv_splits, reorder_batch_threshold)",
    )
    parser.add_argument(
        "--sweep-values",
        type=int,
        nargs="+",
        help="Values to sweep for the parameter",
    )

    # Output
    parser.add_argument("--output-csv", help="Save to CSV")
    parser.add_argument("--output-json", help="Save to JSON")

    args = parser.parse_args()

    console = Console()
    console.print("[bold cyan]vLLM Attention Benchmark[/]")

    # Load config from YAML if provided
    if args.config:
        console.print(f"[yellow]Loading config from: {args.config}[/]")
        yaml_config = load_config_from_yaml(args.config)

        # Show description if available
        if "description" in yaml_config:
            console.print(f"[dim]{yaml_config['description']}[/]")

        # Override args with YAML values, but CLI args take precedence
        # Check if CLI provided backends (they would be non-None and not default)
        cli_backends_provided = args.backends is not None or args.backend is not None

        # Backend(s) - only use YAML if CLI didn't specify
        if not cli_backends_provided:
            if "backend" in yaml_config:
                args.backend = yaml_config["backend"]
                args.backends = None
            elif "backends" in yaml_config:
                args.backends = yaml_config["backends"]
                args.backend = None

        # Check for special modes
        if "mode" in yaml_config:
            args.mode = yaml_config["mode"]
        else:
            args.mode = None

        # Batch specs and sizes
        # Support both explicit batch_specs and generated batch_spec_ranges
        if "batch_spec_ranges" in yaml_config:
            # Generate batch specs from ranges
            generated_specs = generate_batch_specs_from_ranges(
                yaml_config["batch_spec_ranges"]
            )
            # Combine with any explicit batch_specs
            if "batch_specs" in yaml_config:
                args.batch_specs = yaml_config["batch_specs"] + generated_specs
            else:
                args.batch_specs = generated_specs
            console.print(
                f"[dim]Generated {len(generated_specs)} batch specs from ranges[/]"
            )
        elif "batch_specs" in yaml_config:
            args.batch_specs = yaml_config["batch_specs"]

        if "batch_sizes" in yaml_config:
            args.batch_sizes = yaml_config["batch_sizes"]
        else:
            args.batch_sizes = None

        # Model config
        if "model" in yaml_config:
            model = yaml_config["model"]
            args.num_layers = model.get("num_layers", args.num_layers)
            args.head_dim = model.get("head_dim", args.head_dim)
            args.num_q_heads = model.get("num_q_heads", args.num_q_heads)
            args.num_kv_heads = model.get("num_kv_heads", args.num_kv_heads)
            args.block_size = model.get("block_size", args.block_size)

        # Benchmark settings (top-level keys)
        if "device" in yaml_config:
            args.device = yaml_config["device"]
        if "repeats" in yaml_config:
            args.repeats = yaml_config["repeats"]
        if "warmup_iters" in yaml_config:
            args.warmup_iters = yaml_config["warmup_iters"]
        if "profile_memory" in yaml_config:
            args.profile_memory = yaml_config["profile_memory"]

        # Parameter sweep configuration
        if "parameter_sweep" in yaml_config:
            sweep_config = yaml_config["parameter_sweep"]
            args.parameter_sweep = ParameterSweep(
                param_name=sweep_config["param_name"],
                values=sweep_config["values"],
                include_auto=sweep_config.get("include_auto", False),
                label_format=sweep_config.get(
                    "label_format", "{backend}_{param_name}_{value}"
                ),
            )
        else:
            args.parameter_sweep = None

        # Model parameter sweep configuration
        if "model_parameter_sweep" in yaml_config:
            sweep_config = yaml_config["model_parameter_sweep"]
            args.model_parameter_sweep = ModelParameterSweep(
                param_name=sweep_config["param_name"],
                values=sweep_config["values"],
                label_format=sweep_config.get(
                    "label_format", "{backend}_{param_name}_{value}"
                ),
            )
        else:
            args.model_parameter_sweep = None

        # Output
        if "output" in yaml_config:
            output = yaml_config["output"]
            if "csv" in output and not args.output_csv:
                args.output_csv = output["csv"]
            if "json" in output and not args.output_json:
                args.output_json = output["json"]

        console.print()

    # Handle CLI-based parameter sweep (if not from YAML)
    if (
        (not hasattr(args, "parameter_sweep") or args.parameter_sweep is None)
        and args.sweep_param
        and args.sweep_values
    ):
        args.parameter_sweep = ParameterSweep(
            param_name=args.sweep_param,
            values=args.sweep_values,
            include_auto=False,
            label_format="{backend}_{param_name}_{value}",
        )

    # Determine backends
    backends = args.backends or ([args.backend] if args.backend else ["flash"])
    console.print(f"Backends: {', '.join(backends)}")
    console.print(f"Batch specs: {', '.join(args.batch_specs)}")
    console.print()

    # Run benchmarks
    all_results = []

    # Handle special mode: decode_vs_prefill comparison
    if hasattr(args, "mode") and args.mode == "decode_vs_prefill":
        console.print("[yellow]Mode: Decode vs Prefill pipeline comparison[/]")
        console.print(
            "[dim]For each query length, testing both decode and prefill pipelines[/]"
        )
        console.print("[dim]Using batched execution for optimal performance[/]")

        # Extract batch sizes from config
        batch_sizes = getattr(args, "batch_sizes", [1])
        backend = backends[0]  # Use first backend (should only be one)

        # Calculate total benchmarks
        total = len(batch_sizes)

        with tqdm(total=total, desc="Benchmarking") as pbar:
            for batch_size in batch_sizes:
                # Prepare all configs for this batch size
                configs_with_thresholds = []

                for spec in args.batch_specs:
                    # Parse the batch spec to get query length
                    requests = parse_batch_spec(spec)
                    if not requests:
                        console.print(
                            f"[red]Error: Could not parse batch spec '{spec}'[/]"
                        )
                        continue

                    # Get query length from first request
                    query_length = requests[0].q_len

                    # Create batch spec for this batch size
                    # For batch_size > 1, we need to prepend the count
                    batch_spec = f"{batch_size}{spec}" if batch_size > 1 else spec

                    # Create base config (without backend name)
                    base_config = BenchmarkConfig(
                        backend=backend,  # Will be overridden later
                        batch_spec=batch_spec,
                        num_layers=args.num_layers,
                        head_dim=args.head_dim,
                        num_q_heads=args.num_q_heads,
                        num_kv_heads=args.num_kv_heads,
                        block_size=args.block_size,
                        device=args.device,
                        repeats=args.repeats,
                        warmup_iters=args.warmup_iters,
                        profile_memory=args.profile_memory,
                    )

                    # Add decode pipeline config
                    decode_threshold = query_length
                    config_decode = replace(
                        base_config,
                        backend=f"{backend}_decode_qlen{query_length}_bs{batch_size}",
                    )
                    configs_with_thresholds.append((config_decode, decode_threshold))

                    # Add prefill pipeline config if query_length > 1
                    if query_length > 1:
                        prefill_threshold = query_length - 1
                        config_prefill = replace(
                            base_config,
                            backend=f"{backend}_prefill_qlen{query_length}"
                            f"_bs{batch_size}",
                        )
                        configs_with_thresholds.append(
                            (config_prefill, prefill_threshold)
                        )

                # Run all benchmarks for this batch size in one go (batched mode)
                try:
                    from mla_runner import run_mla_benchmark as run_mla

                    # Use batched API: pass list of (config, threshold) tuples
                    timing_results = run_mla(backend, configs_with_thresholds)

                    # Create BenchmarkResult objects from timing results
                    for (config, _), timing in zip(
                        configs_with_thresholds, timing_results
                    ):
                        result = BenchmarkResult(
                            config=config,
                            mean_time=timing["mean"],
                            std_time=timing["std"],
                            min_time=timing["min"],
                            max_time=timing["max"],
                            throughput_tokens_per_sec=timing.get("throughput", None),
                        )
                        all_results.append(result)

                except Exception as e:
                    import traceback

                    console.print(
                        f"[red]Error running batched benchmarks for "
                        f"batch_size={batch_size}: {e}[/]"
                    )
                    console.print("[red]Traceback:[/]")
                    traceback.print_exc()
                    # Add error results for all configs
                    for config, _ in configs_with_thresholds:
                        result = BenchmarkResult(
                            config=config,
                            mean_time=float("inf"),
                            std_time=0,
                            min_time=float("inf"),
                            max_time=float("inf"),
                            error=str(e),
                        )
                        all_results.append(result)

                pbar.update(1)

        # Display decode vs prefill results
        console.print("\n[bold green]Decode vs Prefill Results:[/]")

        # Group by batch size
        by_batch_size = {}
        for r in all_results:
            if r.success:
                # Extract batch size from backend name
                parts = r.config.backend.split("_")
                bs_part = [p for p in parts if p.startswith("bs")]
                if bs_part:
                    bs = int(bs_part[0][2:])
                    if bs not in by_batch_size:
                        by_batch_size[bs] = []
                    by_batch_size[bs].append(r)

        # For each batch size, analyze crossover point
        for bs in sorted(by_batch_size.keys()):
            console.print(f"\n[bold cyan]Batch size: {bs}[/]")
            results = by_batch_size[bs]

            # Group by query length
            by_qlen = {}
            for r in results:
                parts = r.config.backend.split("_")
                qlen_part = [p for p in parts if p.startswith("qlen")]
                if qlen_part:
                    qlen = int(qlen_part[0][4:])
                    if qlen not in by_qlen:
                        by_qlen[qlen] = {}

                    pipeline = "decode" if "decode" in r.config.backend else "prefill"
                    by_qlen[qlen][pipeline] = r

            # Find crossover point
            last_decode_faster = None
            for qlen in sorted(by_qlen.keys()):
                pipelines = by_qlen[qlen]
                if "decode" in pipelines and "prefill" in pipelines:
                    decode_time = pipelines["decode"].mean_time
                    prefill_time = pipelines["prefill"].mean_time
                    faster = "decode" if decode_time < prefill_time else "prefill"

                    speedup = (
                        prefill_time / decode_time
                        if decode_time < prefill_time
                        else decode_time / prefill_time
                    )

                    console.print(
                        f"  qlen={qlen:3d}: decode={decode_time:.6f}s, "
                        f"prefill={prefill_time:.6f}s -> "
                        f"[bold]{faster}[/] ({speedup:.2f}x)"
                    )

                    if faster == "decode":
                        last_decode_faster = qlen

            if last_decode_faster is not None:
                optimal_threshold = last_decode_faster
                console.print(
                    f"\n  [bold green]Optimal threshold for batch_size={bs}: "
                    f"{optimal_threshold}[/]"
                )
                console.print(
                    f"  [dim](Use decode pipeline for query_length <= "
                    f"{optimal_threshold})[/]"
                )
            else:
                console.print(
                    f"\n  [yellow]Prefill always faster for batch_size={bs}[/]"
                )

    # Handle model parameter sweep mode
    elif hasattr(args, "model_parameter_sweep") and args.model_parameter_sweep:
        # Model parameter sweep
        base_config_args = {
            "num_layers": args.num_layers,
            "head_dim": args.head_dim,
            "num_q_heads": args.num_q_heads,
            "num_kv_heads": args.num_kv_heads,
            "block_size": args.block_size,
            "device": args.device,
            "repeats": args.repeats,
            "warmup_iters": args.warmup_iters,
            "profile_memory": args.profile_memory,
        }
        all_results = run_model_parameter_sweep(
            backends,
            args.batch_specs,
            base_config_args,
            args.model_parameter_sweep,
            console,
        )

    # Handle parameter sweep mode (unified)
    elif hasattr(args, "parameter_sweep") and args.parameter_sweep:
        # Unified parameter sweep
        base_config_args = {
            "num_layers": args.num_layers,
            "head_dim": args.head_dim,
            "num_q_heads": args.num_q_heads,
            "num_kv_heads": args.num_kv_heads,
            "block_size": args.block_size,
            "device": args.device,
            "repeats": args.repeats,
            "warmup_iters": args.warmup_iters,
            "profile_memory": args.profile_memory,
        }
        all_results = run_parameter_sweep(
            backends, args.batch_specs, base_config_args, args.parameter_sweep, console
        )

    else:
        # Normal mode: compare backends
        total = len(backends) * len(args.batch_specs)

        with tqdm(total=total, desc="Benchmarking") as pbar:
            for spec in args.batch_specs:
                for backend in backends:
                    config = BenchmarkConfig(
                        backend=backend,
                        batch_spec=spec,
                        num_layers=args.num_layers,
                        head_dim=args.head_dim,
                        num_q_heads=args.num_q_heads,
                        num_kv_heads=args.num_kv_heads,
                        block_size=args.block_size,
                        device=args.device,
                        repeats=args.repeats,
                        warmup_iters=args.warmup_iters,
                        profile_memory=args.profile_memory,
                    )

                    result = run_benchmark(config)
                    all_results.append(result)

                    if not result.success:
                        console.print(f"[red]Error {backend} {spec}: {result.error}[/]")

                    pbar.update(1)

        # Display results
        console.print("\n[bold green]Results:[/]")
        formatter = ResultsFormatter(console)
        formatter.print_table(all_results, backends)

    # Save results
    if all_results:
        formatter = ResultsFormatter(console)
        if args.output_csv:
            formatter.save_csv(all_results, args.output_csv)
        if args.output_json:
            formatter.save_json(all_results, args.output_json)


if __name__ == "__main__":
    main()
