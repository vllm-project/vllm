# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Regression tests for attention benchmarks.

This module uses the attention benchmark infrastructure in
<vllm>/benchmarks/attention/ to test for performance regressions in
the attention kernels/backends by running benchmark.py with YAML
configs and validating results against golden references.

Directory Structure:
    <vllm_root>/
    ├── benchmarks/attention_benchmarks/
    │   ├── benchmark.py                    (benchmark runner script)
    │   └── configs/pytest_regression/
    │       └── *.yaml                      (test configs)
    └── tests/kernels/attention/benchmark/
        ├── test_benchmark_attention.py     (this file)
        ├── goldens/<gfx_target>/
        │   └── *.json                      (golden reference JSONs)
        └── output/<gfx_target>/
            └── *.json                      (test outputs - gitignored)

Golden Reference Workflow:
    1. First run: No golden exists
       → Run benchmark; saves results as golden, skips validation
       → Sanity-check contents before committing
    2. Subsequent runs: Golden exists
       → Run benchmark; compares against golden, passes/fails
    3. Regenerating existing golden(s)
       → Run benchmark, sanity check output vs golden
       → Replace old golden with new output and commit

Skip/Intermittent Cases:
    - Skip: Cases that should not run (e.g., known bugs, unsupported combos)
      → Defined in SKIP_CASES dict at module level
      → Passed to benchmark.py via --skip CLI arg
      → Appear in JSON with "skip": "reason", timing fields null
      → Validation skipped

    - Intermittent: Cases that run but may be flaky
      → Defined in INTERMITTENT_CASES dict at module level
      → Passed to benchmark.py via --intermittent CLI arg
      → Appear in JSON with "intermittent": "reason"
      → Performance validation skipped by default
      → Always run/collect data to avoid accidental invalid goldens
      → Use --intermittent pytest flag to also enable validation

Adding Skip/Intermittent Cases:
    Edit SKIP_CASES or INTERMITTENT_CASES dicts:

    SKIP_CASES = {
        ("gfx1151", "gemma_2b_awq", "q1s4k", "TRITON_ATTN"): "Bug #123",
        ("*", "llama_7b", "*", "ROCM_AITER_UNIFIED_ATTN"): "Skip FLASH_ATTN backend",
    }

    Pattern matching:
        - platform/config/backend: "*" = wildcard (any value),
          otherwise exact match
        - batch_spec: "*" = wildcard, or substring match
          (e.g., "*128*" matches "q128s1k")

Adding New Test Configs:
    1. Create YAML: benchmarks/attention_benchmarks/configs/pytest_regression/<config_name>.yaml
    2. Add <config_name> to @pytest.mark.parametrize list with target platforms
    3. Run test once to generate new golden references
    4. Review golden JSON and commit to git
"""  # noqa: E501

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from vllm.platforms import current_platform

# Performance validation thresholds
VARIANCE_WARNING_THRESHOLD = 0.01  # 1% - warn if SEM/mean > this
REGRESSION_ERROR_THRESHOLD = 0.10  # 10% - fail if |delta|/golden > this

# Subprocess timeout (seconds)
BENCHMARK_TIMEOUT = 900

# Skip cases:
# (platform, config_name, batch_spec_pattern, backend) -> reason
# Use "*" as wildcard, substring matching for batch_specs
SKIP_CASES: dict[tuple[str, str, str, str], str] = {
    # Example:
    # ("gfx1151", "gemma_2b_awq", "q1s4k", "TRITON_ATTN"):
    #     "Triton kernel bug #123",
}

# Intermittent cases:
# (platform, config_name, batch_spec_pattern, backend) -> reason
INTERMITTENT_CASES: dict[tuple[str, str, str, str], str] = {
    # Example:
    # ("gfx1151", "llama_7b", "q128s8k", "TRITON_ATTN"):
    #     "Flaky timing on large contexts",
}


def get_gfx_target() -> str:
    """
    Get current gcn architecture name.
    Likely works on non-rocm platforms, but untested
    """
    if current_platform.is_rocm():
        return current_platform.get_gcn_arch()
    return "unknown"


def get_attn_benchmarks_dir() -> Path:
    """
    Get path to attention benchmark directory
    This directory contains the benchmark script and YAML configs
    """
    test_dir = Path(__file__).parent
    # Navigate: <vllm_root>/tests/kernels/attention/benchmark -> <vllm_root>
    repo_root = test_dir.parent.parent.parent.parent
    return repo_root / "benchmarks" / "attention_benchmarks"


def get_benchmark_script_path() -> Path:
    """
    Get path to attention benchmark.py script
    """
    return get_attn_benchmarks_dir() / "benchmark.py"


def get_config_path(config_name: str) -> Path:
    """
    Get path to benchmark config YAML.
    config_name: config name without .yaml extension
    """
    configs_dir = get_attn_benchmarks_dir() / "configs" / "pytest_regression"
    return configs_dir / f"{config_name}.yaml"


def _build_filters(cases_dict: dict, platform: str, config_name: str) -> list[dict]:
    """
    Build filter list from skip/intermittent cases dictionary.

    Args:
        cases_dict: Dictionary of (platform, config, batch_pattern, backend) -> reason
        platform: Platform to match (e.g., "gfx1151")
        config_name: Config name to match (e.g., "gemma_2b_awq")

    Returns:
        List of filter dicts suitable for --skip or --intermittent CLI arg.
        Groups by reason for efficiency.
    """
    filters: list[dict[str, Any]] = []
    for (plat, cfg, batch_pattern, backend), reason in cases_dict.items():
        # Match platform
        if plat not in (platform, "*"):
            continue
        # Match config
        if cfg not in (config_name, "*"):
            continue

        # Add to filters (group by reason for efficiency)
        existing = next((f for f in filters if f["reason"] == reason), None)
        if existing:
            if batch_pattern not in existing["batch_specs"]:
                existing["batch_specs"].append(batch_pattern)
            if backend not in existing["backends"]:
                existing["backends"].append(backend)
        else:
            filters.append(
                {
                    "batch_specs": [batch_pattern],
                    "backends": [backend],
                    "reason": reason,
                }
            )

    return filters


def get_skip_cases(platform: str, config_name: str) -> list[dict]:
    """Get skip filters for this platform + config."""
    return _build_filters(SKIP_CASES, platform, config_name)


def get_intermittent_cases(platform: str, config_name: str) -> list[dict]:
    """Get intermittent filters for this platform + config."""
    return _build_filters(INTERMITTENT_CASES, platform, config_name)


def run_benchmark_subprocess(
    config_name: str,
    gfx_target: str,
    skip_filters: list[dict] | None = None,
    intermittent_filters: list[dict] | None = None,
) -> tuple[bool, str, str]:
    """
    Run benchmark.py as subprocess and capture output.

    Creates output directory if needed, runs benchmark,
    saves stdout/stderr to log file.

    Args:
        config_name: Name of config YAML (without .yaml)
        gfx_target: GCN architecture string (e.g., "gfx1151")
        skip_filters: Optional list of skip filter dicts
        intermittent_filters: Optional list of intermittent filter dicts

    Returns:
        Tuple of (success, json_path, log_path)
        - success: True if subprocess exited 0 and JSON exists
        - json_path: path to output JSON
        - log_path: path to output log

    Raises:
        subprocess.TimeoutExpired: If benchmark exceeds BENCHMARK_TIMEOUT
    """
    skip_filters = skip_filters or []
    intermittent_filters = intermittent_filters or []
    test_dir = Path(__file__).parent
    output_dir = test_dir / "output" / gfx_target
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{config_name}.json"
    log_path = output_dir / f"{config_name}.log"

    config_path = get_config_path(config_name)
    benchmark_script = get_benchmark_script_path()

    cmd = [
        sys.executable,
        str(benchmark_script),
        "--config",
        str(config_path),
        "--output-json",
        str(json_path),
        "--inter-batch-cooldown",
        "20",  # 20 second cooldown for thermal management
    ]

    # Add skip filters
    for f in skip_filters:
        cmd.extend(["--skip", json.dumps(f)])

    # Add intermittent filters
    for f in intermittent_filters:
        cmd.extend(["--intermittent", json.dumps(f)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=BENCHMARK_TIMEOUT,
        )
    except subprocess.TimeoutExpired as e:
        # Log timeout error
        with open(log_path, "w") as log_file:
            log_file.write(f"Command: {' '.join(cmd)}\n")
            log_file.write(f"ERROR: Benchmark timed out after {BENCHMARK_TIMEOUT}s\n")
            if e.stdout:
                log_file.write("\n=== STDOUT (partial) ===\n")
                stdout_str = (
                    e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout
                )
                log_file.write(stdout_str)
            if e.stderr:
                log_file.write("\n=== STDERR (partial) ===\n")
                stderr_str = (
                    e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr
                )
                log_file.write(stderr_str)
        raise

    # Save stdout + stderr to log file
    with open(log_path, "w") as log_file:
        log_file.write(f"Command: {' '.join(cmd)}\n")
        log_file.write(f"Return code: {result.returncode}\n")
        log_file.write("\n=== STDOUT ===\n")
        log_file.write(result.stdout)
        log_file.write("\n=== STDERR ===\n")
        log_file.write(result.stderr)

    success = result.returncode == 0 and json_path.exists()
    return success, str(json_path), str(log_path)


def load_benchmark_results(json_path: str) -> list[dict]:
    """
    Load and parse benchmark results from JSON.

    Args:
        json_path: Path to JSON results file

    Returns:
        List of result dictionaries (from BenchmarkResult.to_dict())

    Raises:
        FileNotFoundError: If JSON file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    with open(json_path) as f:
        return json.load(f)


def make_config_key(entry: dict) -> tuple:
    """
    Create hashable key from benchmark result entry for matching.

    Each test config YAML has a single model configuration, so only
    backend and batch_spec vary within a test run.

    Args:
        entry: Benchmark result dict with "config" key

    Returns:
        Tuple of (backend, batch_spec)

    Example:
        >>> entry = {
        ...     "config": {
        ...         "backend": "TRITON_ATTN",
        ...         "batch_spec": "q1ks128",
        ...         ...
        ...     }
        ... }
        >>> make_config_key(entry)
        ('TRITON_ATTN', 'q1ks128')
    """
    config = entry["config"]
    return (config["backend"], config["batch_spec"])


def calculate_sem(std_time: float, num_samples: int) -> float:
    """
    Calculate standard error of the mean.

    SEM = σ / √n
    where σ is standard deviation, n is number of samples

    Args:
        std_time: Standard deviation of timing samples
        num_samples: Number of timing samples (repeats)

    Returns:
        Standard error of mean

    Example:
        >>> calculate_sem(0.0001, 10)
        3.162277660168379e-05
    """
    return std_time / math.sqrt(num_samples)


def format_config_details(entry: dict) -> str:
    """
    Format config entry as human-readable string for error messages.

    Args:
        entry: Benchmark result dict with "config" key

    Returns:
        Formatted string with key config parameters

    Example:
        >>> format_config_details(entry)
        'backend=TRITON_ATTN, batch_spec=q1ks128, ...'
    """
    config = entry["config"]
    return (
        f"backend={config['backend']}, "
        f"batch_spec={config['batch_spec']}, "
        f"layers={config['num_layers']}, "
        f"q_heads={config['num_q_heads']}, "
        f"kv_heads={config['num_kv_heads']}, "
        f"head_dim={config['head_dim']}"
    )


def print_last_n_lines(filepath: str, n: int = 20) -> None:
    """
    Print last N lines of a file.

    Useful for showing relevant error context without flooding output.

    Args:
        filepath: Path to file
        n: Number of lines to print (default: 20)
    """
    with open(filepath) as f:
        lines = f.readlines()

    print(f"\nLast {n} lines of {filepath}:")
    print("=" * 70)
    print("".join(lines[-n:]))
    print("=" * 70)


# Parameterized on config file (minus .yaml) + supported GPUs
@pytest.mark.parametrize(
    "config_name,target_platforms",
    [
        ("gemma_2b_awq", ["gfx1151"]),
        ("llama_2_7b", ["gfx1151"]),
        ("qwen2.5_1.5b_awq", ["gfx1151"]),
        ("qwen2.5_1.5b_fp16", ["gfx1151"]),
        ("qwen2.5_3b_awq", ["gfx1151"]),
        ("qwen2.5_7b_awq", ["gfx1151"]),
        ("qwen3_1.7b", ["gfx1151"]),
        ("qwen3_4b_awq", ["gfx1151"]),
        ("qwen3_8b_awq", ["gfx1151"]),
        ("qwen3_8b_w4a16", ["gfx1151"]),
        ("qwen3_omni_30b_a3b", ["gfx1151"]),
        ("qwen3.5_35b_a3b_w4a16", ["gfx1151"]),
        ("smollm2_1.7b", ["gfx1151"]),
    ],
)
def test_attn_bench_regression(
    config_name: str,
    target_platforms: list[str],
    request: pytest.FixtureRequest,
):
    """
    Regression test for attention benchmark configurations.

    This test runs the attention benchmark with a specified YAML config,
    validates the output structure against a golden reference, and checks
    for performance regressions and speedups.

    Each config specifies which GPU platforms it should run on. The test
    is skipped if the current platform is not in the target list.

    Test Phases:
        1. Run benchmark.py as subprocess (with skip/intermittent filters)
        2. Load and validate JSON results
        3. Load or create golden reference
        4. Validate structure (entry count, config matching)
        5. Validate performance (variance, regression, speedup)
        6. Print summary and assert

    Validation Rules:
        - Entry count must match golden
        - All configs in golden must exist in actual (order-independent)
        - No benchmark errors allowed
        - Skipped entries have timing fields set to null, validation skipped
        - Intermittent entries validated only if --intermittent flag set
        - Warning if SEM > 1% of mean_time (high variance)
        - Error if mean_time differs from golden by >10% (regression or speedup)
        - Both regressions AND speedups fail to ensure golden is updated

    Args:
        - config_name: Name of YAML config (without .yaml extension)
        - target_platforms: List of GCN arch strings this config runs on
        - request: context for requesting test function (used to retrieve --intermittent flag)

    Examples:
        pytest test_benchmark_attention.py::test_benchmark_regression[gemma_2b_awq-gfx1151]
    """  # noqa: E501
    gfx_target = get_gfx_target()

    # Skip if current platform not in target list
    if gfx_target not in target_platforms:
        pytest.skip(
            f"Config '{config_name}' not configured for platform '{gfx_target}'. "
            f"Target platforms: {target_platforms}"
        )

    test_dir = Path(__file__).parent

    # Run benchmark
    skip_filters = get_skip_cases(gfx_target, config_name)
    intermittent_filters = get_intermittent_cases(gfx_target, config_name)
    success, json_path, log_path = run_benchmark_subprocess(
        config_name,
        gfx_target,
        skip_filters=skip_filters,
        intermittent_filters=intermittent_filters,
    )

    if not success:
        print("\nBenchmark subprocess failed!")
        print(f"Log file: {log_path}")
        print_last_n_lines(log_path, 30)
        pytest.fail(f"Benchmark subprocess failed. Check log: {log_path}")

    actual_results = load_benchmark_results(json_path)

    golden_dir = test_dir / "goldens" / gfx_target
    golden_path = golden_dir / f"{config_name}.json"
    if not golden_path.exists():
        print(f"\nNo golden reference found at {golden_path}")
        print("Creating new golden reference from current benchmark run...")
        golden_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(json_path, golden_path)
        print(f"Golden reference created: {golden_path}")
        print("\nNext steps:")
        print("  1. Review the golden reference JSON")
        print("  2. If correct, commit it to git")
        print("  3. Future runs will validate against this reference")
        pytest.skip(f"First run - created golden reference at {golden_path}. ")

    golden_results = load_benchmark_results(str(golden_path))

    # Build lookup tables by config key
    actual_by_key = {make_config_key(e): e for e in actual_results}
    golden_by_key = {make_config_key(e): e for e in golden_results}

    # Get repeats from first actual result (same for all cases in config)
    repeats = actual_results[0]["config"]["repeats"] if actual_results else 10

    # Validate entry counts match
    has_config_errors = False
    num_actual = len(actual_results)
    num_golden = len(golden_results)
    if num_actual != num_golden:
        has_config_errors = True
        print(f"\nEntry count mismatch: {num_actual} actual vs {num_golden} golden")
        if num_actual > num_golden:
            extra_keys = set(actual_by_key.keys()) - set(golden_by_key.keys())
            print(f"Extra entries in actual (not in golden): {extra_keys}")
        else:
            missing_keys = set(golden_by_key.keys()) - set(actual_by_key.keys())
            print(f"Missing entries in actual (present in golden): {missing_keys}")

    # Validate results
    num_passed = 0
    num_warnings = 0
    num_regressions = 0
    num_speedups = 0
    num_skipped_intermittent = 0
    intermittent_enabled = request.config.getoption("--intermittent")
    for key in sorted(actual_by_key.keys()):
        actual = actual_by_key[key]

        # Check if config exists in golden
        if key not in golden_by_key:
            has_config_errors = True
            print(f"Config missing from golden: {format_config_details(actual)}")
            continue

        golden = golden_by_key[key]

        if actual.get("skip"):
            continue

        # Skip performance validation for intermittent unless flag set
        is_intermittent = actual.get("intermittent")
        if is_intermittent and not intermittent_enabled:
            num_skipped_intermittent += 1
            continue

        if actual.get("error") is not None:
            print(f"Benchmark error: {format_config_details(actual)}")
            print(f"Error: {actual['error']}")
            has_config_errors = True
            continue

        # Variance check (warning only - test still passes)
        actual_mean = actual["mean_time"]
        actual_std = actual["std_time"]
        sem = calculate_sem(actual_std, repeats)
        rel_uncertainty = sem / actual_mean if actual_mean > 0 else 0
        if rel_uncertainty > VARIANCE_WARNING_THRESHOLD:
            num_warnings += 1
            print(f"High variance: {format_config_details(actual)}")
            print(f"Mean time: {actual_mean:.6f}s")
            print(f"Std dev:   {actual_std:.6f}s")
            print(f"SEM:       {sem:.6f}s ({rel_uncertainty * 100:.2f}% of mean)")
            print("Recommendation: Increase repeats or investigate variance source\n")

        # Performance change detection
        golden_mean = golden["mean_time"]
        if golden_mean > 0:
            # Proportional change in mean
            prop_change = (actual_mean - golden_mean) / golden_mean

            if prop_change > REGRESSION_ERROR_THRESHOLD:
                # Regression (slower)
                num_regressions += 1
                print(f"PERFORMANCE REGRESSION: {format_config_details(actual)}")
                print(f"Expected: {golden_mean:.6f}s")
                print(f"Actual:   {actual_mean:.6f}s")
                print(f"Change:   {prop_change * 100:+.2f}% (slower)\n")
            elif prop_change < -REGRESSION_ERROR_THRESHOLD:
                # Speedup (faster)
                num_speedups += 1
                print(f"PERFORMANCE SPEEDUP: {format_config_details(actual)}")
                print(f"Expected: {golden_mean:.6f}s")
                print(f"Actual:   {actual_mean:.6f}s")
                print(f"Change:   {prop_change * 100:+.2f}% (faster)\n")
            else:
                # Within threshold
                num_passed += 1

    # Summary and assertions
    print(
        f"\nValidation Summary: {num_passed} passed, "
        f"{num_regressions} regressions, {num_speedups} speedups, "
        f"{num_warnings} warnings"
    )
    if num_skipped_intermittent > 0:
        print(
            f"({num_skipped_intermittent} intermittent cases skipped - "
            f"use --intermittent to validate)"
        )

    assert not has_config_errors, (
        f"Configuration validation failed. Check benchmark output at {log_path}"
    )

    if num_regressions > 0 and num_speedups > 0:
        pytest.fail(
            f"Performance changes detected: {num_regressions} regression(s) "
            f"and {num_speedups} speedup(s). Review results:\n"
            f"  Actual: {json_path}\n"
            f"  Golden: {golden_path}\n"
            f"For speedups: Update golden by running:\n"
            f"  cp {json_path} {golden_path}"
        )
    elif num_regressions > 0:
        pytest.fail(
            f"Performance regression detected "
            f"(>{REGRESSION_ERROR_THRESHOLD * 100:.0f}% slower). "
            f"Actual results: {json_path}, Golden: {golden_path}"
        )
    elif num_speedups > 0:
        pytest.fail(
            f"Performance speedup detected "
            f"(>{REGRESSION_ERROR_THRESHOLD * 100:.0f}% faster). "
            f"Golden reference needs updating. Review results, then run:\n"
            f"  cp {json_path} {golden_path}\n"
            f"Then commit the updated golden."
        )
