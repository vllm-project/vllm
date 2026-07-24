# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate and analyze FusedMoE per-GPU tuning config files.

Checks schema correctness, filename format, and coverage across GPUs/dtypes.
Designed to run without a GPU (no torch/CUDA imports).

Usage:
    python benchmarks/kernels/validate_moe_configs.py
    python benchmarks/kernels/validate_moe_configs.py --verbose --strict
    python benchmarks/kernels/validate_moe_configs.py --json-output results.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_DEFAULT_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "vllm",
    "model_executor",
    "layers",
    "fused_moe",
    "configs",
)

REQUIRED_KEYS: set[str] = {
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
}

OPTIONAL_KEYS: set[str] = {
    "waves_per_eu",
    "matrix_instr_nonkdim",
    "kpack",
    "SPLIT_K",
    "num_ctas",
}

ALL_KNOWN_KEYS: set[str] = REQUIRED_KEYS | OPTIONAL_KEYS

POWER_OF_TWO_KEYS: set[str] = {
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "num_warps",
}

FILENAME_PATTERN = re.compile(
    r"^E=(?P<E>\d+)"
    r",N=(?P<N>\d+)"
    r",device_name=(?P<device>[A-Za-z0-9_\-]+)"
    r"(?:,dtype=(?P<dtype>[A-Za-z0-9_]+))?"
    r"(?:,block_shape=\[(?P<block_shape>[\d,]+)\])?"
    r"\.json$"
)


def _is_power_of_two(n: int) -> bool:
    """Return True if *n* is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0


def parse_filename(filename: str) -> dict[str, Any] | None:
    """Parse a MoE config filename into its component parts.

    Args:
        filename: Basename of the config file (e.g.
            ``E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json``).

    Returns:
        Dictionary with keys ``E``, ``N``, ``device``, ``dtype``, and
        ``block_shape``, or ``None`` if the filename does not match the
        expected pattern.
    """
    m = FILENAME_PATTERN.match(filename)
    if m is None:
        return None
    result: dict[str, Any] = {
        "E": int(m.group("E")),
        "N": int(m.group("N")),
        "device": m.group("device"),
        "dtype": m.group("dtype"),
    }
    bs = m.group("block_shape")
    result["block_shape"] = [int(x) for x in bs.split(",")] if bs else None
    return result


def validate_config_entry(
    entry: dict[str, Any],
    batch_key: str,
    filename: str,
) -> list[str]:
    """Validate a single config entry (one batch-size mapping).

    Args:
        entry: The config dict for one batch size.
        batch_key: The top-level key (batch size as string).
        filename: Source filename (for error messages).

    Returns:
        List of human-readable error strings (empty if valid).
    """
    errors: list[str] = []
    prefix = f"{filename}[{batch_key}]"

    if not isinstance(entry, dict):
        errors.append(f"{prefix}: entry is not a dict (got {type(entry).__name__})")
        return errors

    present_keys = set(entry.keys())
    missing = REQUIRED_KEYS - present_keys
    if missing:
        errors.append(f"{prefix}: missing required keys: {sorted(missing)}")

    unexpected = present_keys - ALL_KNOWN_KEYS
    if unexpected:
        errors.append(f"{prefix}: unexpected keys: {sorted(unexpected)}")

    for key in present_keys & ALL_KNOWN_KEYS:
        val = entry[key]
        if not isinstance(val, int):
            errors.append(f"{prefix}: {key} must be int, got {type(val).__name__}")
            continue

        if key in POWER_OF_TWO_KEYS and not _is_power_of_two(val):
            errors.append(f"{prefix}: {key}={val} is not a power of 2")

        if key == "num_warps" and not (1 <= val <= 32):
            errors.append(f"{prefix}: num_warps={val} outside [1, 32]")

        if key == "num_stages" and not (1 <= val <= 8):
            errors.append(f"{prefix}: num_stages={val} outside [1, 8]")

        if key == "GROUP_SIZE_M" and val < 0:
            errors.append(f"{prefix}: GROUP_SIZE_M={val} is negative")

        if key == "SPLIT_K" and val < 1:
            errors.append(f"{prefix}: SPLIT_K={val} must be >= 1")

        if key == "waves_per_eu" and val < 0:
            errors.append(f"{prefix}: waves_per_eu={val} is negative")

    return errors


def validate_file(filepath: str | Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Load and validate one JSON config file.

    Args:
        filepath: Full path to the JSON file.

    Returns:
        Tuple of (parsed data or None, list of error strings).
    """
    filepath = Path(filepath)
    errors: list[str] = []
    filename = filepath.name

    if filepath.stat().st_size == 0:
        errors.append(f"{filename}: file is empty")
        return None, errors

    try:
        with open(filepath) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        errors.append(f"{filename}: malformed JSON: {exc}")
        return None, errors
    except OSError as exc:
        errors.append(f"{filename}: cannot read file: {exc}")
        return None, errors

    if not isinstance(data, dict):
        errors.append(
            f"{filename}: top-level value is not a dict (got {type(data).__name__})"
        )
        return None, errors

    for key, value in data.items():
        if key == "triton_version":
            if not isinstance(value, str):
                errors.append(
                    f"{filename}: triton_version should be str, "
                    f"got {type(value).__name__}"
                )
            continue

        if not key.isdigit():
            errors.append(f"{filename}: non-numeric top-level key: {key!r}")
            continue

        entry_errors = validate_config_entry(value, key, filename)
        errors.extend(entry_errors)

    return data, errors


def build_coverage_matrix(
    parsed_files: dict[str, dict[str, Any]],
) -> dict[str, dict[str, set[str | None]]]:
    """Build a coverage matrix of device x (E, N) x dtype.

    Args:
        parsed_files: Mapping of filename -> parsed filename components.

    Returns:
        Nested dict: ``device -> "(E={E},N={N})" -> set of dtypes``.
    """
    matrix: dict[str, dict[str, set[str | None]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for _filename, parts in parsed_files.items():
        device = parts["device"]
        expert_key = f"E={parts['E']},N={parts['N']}"
        dtype = parts.get("dtype")
        matrix[device][expert_key].add(dtype)
    return dict(matrix)


def format_coverage_table(
    matrix: dict[str, dict[str, set[str | None]]],
) -> str:
    """Format the coverage matrix as an ASCII table.

    Args:
        matrix: Output of :func:`build_coverage_matrix`.

    Returns:
        Multi-line string with the formatted table.
    """
    all_expert_keys = sorted(
        {ek for device_data in matrix.values() for ek in device_data},
        key=lambda k: (
            int(k.split(",")[0].split("=")[1]),
            int(k.split(",")[1].split("=")[1]),
        ),
    )
    devices = sorted(matrix.keys())

    if not devices or not all_expert_keys:
        return "  (no data)\n"

    dev_width = max(len(d) for d in devices)
    col_width = max(max(len(ek) for ek in all_expert_keys), 10)

    header = f"{'Device':<{dev_width}}  " + "  ".join(
        f"{ek:^{col_width}}" for ek in all_expert_keys
    )
    separator = "-" * len(header)

    lines = [header, separator]
    for device in devices:
        cells = []
        for ek in all_expert_keys:
            dtypes = matrix.get(device, {}).get(ek, set())
            if dtypes:
                count = len(dtypes)
                cells.append(f"{count:^{col_width}}")
            else:
                cells.append(f"{'---':^{col_width}}")
        lines.append(f"{device:<{dev_width}}  " + "  ".join(cells))

    return "\n".join(lines) + "\n"


def run_validation(
    config_dir: str,
    verbose: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Run full validation and coverage analysis on a config directory.

    Args:
        config_dir: Path to the directory containing JSON config files.
        verbose: If True, include per-file details in the result.
        strict: If True, treat warnings as errors.

    Returns:
        Dictionary with validation results.
    """
    config_path = Path(config_dir)
    results: dict[str, Any] = {
        "config_dir": str(config_path),
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "unrecognized_filenames": [],
        "errors": [],
        "warnings": [],
        "coverage": {},
        "devices_found": [],
        "file_details": {},
    }

    if not config_path.is_dir():
        results["errors"].append(f"Config directory not found: {config_dir}")
        return results

    json_files = sorted(f for f in config_path.iterdir() if f.suffix == ".json")
    results["total_files"] = len(json_files)

    parsed_filenames: dict[str, dict[str, Any]] = {}

    for filepath in json_files:
        filename = filepath.name
        file_info: dict[str, Any] = {"path": str(filepath), "errors": []}

        # Validate filename
        parsed = parse_filename(filename)
        if parsed is None:
            results["unrecognized_filenames"].append(filename)
            warning = f"Filename does not match expected pattern: {filename}"
            results["warnings"].append(warning)
        else:
            parsed_filenames[filename] = parsed

        # Validate file contents
        _data, errors = validate_file(filepath)
        file_info["errors"] = errors

        if errors:
            results["invalid_files"] += 1
            results["errors"].extend(errors)
        else:
            results["valid_files"] += 1

        if verbose:
            results["file_details"][filename] = file_info

    # Coverage analysis
    matrix = build_coverage_matrix(parsed_filenames)
    results["coverage"] = {
        device: {ek: sorted(str(d) for d in dtypes) for ek, dtypes in ek_map.items()}
        for device, ek_map in matrix.items()
    }
    results["devices_found"] = sorted(matrix.keys())

    # Check for common missing GPUs
    known_gpus = {
        "NVIDIA_A100-SXM4-80GB",
        "NVIDIA_H100_80GB_HBM3",
        "NVIDIA_H200",
        "NVIDIA_B200",
        "AMD_Instinct_MI300X",
    }
    found_devices = set(matrix.keys())
    missing_common = sorted(known_gpus - found_devices)
    if missing_common:
        for gpu in missing_common:
            results["warnings"].append(f"Common GPU missing from configs: {gpu}")

    if strict:
        results["errors"].extend(results["warnings"])

    return results


def print_results(results: dict[str, Any], verbose: bool = False) -> None:
    """Print validation results to stdout.

    Args:
        results: Output of :func:`run_validation`.
        verbose: If True, print per-file details.
    """
    print("=" * 60)
    print("FusedMoE Config Validation Report")
    print("=" * 60)
    print(f"Config directory: {results['config_dir']}")
    print(f"Total JSON files: {results['total_files']}")
    print(f"Valid files:      {results['valid_files']}")
    print(f"Invalid files:    {results['invalid_files']}")
    print()

    if results["unrecognized_filenames"]:
        print("Unrecognized filenames:")
        for fn in results["unrecognized_filenames"]:
            print(f"  - {fn}")
        print()

    if results["errors"]:
        print(f"Errors ({len(results['errors'])}):")
        for err in results["errors"]:
            print(f"  ERROR: {err}")
        print()

    if results["warnings"]:
        print(f"Warnings ({len(results['warnings'])}):")
        for warn in results["warnings"]:
            print(f"  WARN:  {warn}")
        print()

    if verbose and results.get("file_details"):
        print("Per-file details:")
        for filename, info in sorted(results["file_details"].items()):
            status = "PASS" if not info["errors"] else "FAIL"
            print(f"  [{status}] {filename}")
            for err in info["errors"]:
                print(f"         {err}")
        print()

    # Coverage summary
    print("Coverage Summary")
    print("-" * 60)
    devices = results.get("devices_found", [])
    print(f"Devices with configs: {len(devices)}")
    for device in devices:
        coverage = results.get("coverage", {}).get(device, {})
        print(f"  {device}: {len(coverage)} expert configs")
    print()


def main() -> int:
    """Entry point for the validation script.

    Returns:
        Exit code: 0 if all valid, 1 if errors found.
    """
    parser = argparse.ArgumentParser(
        description="Validate FusedMoE per-GPU tuning config files.",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=_DEFAULT_CONFIG_DIR,
        help="Path to the configs directory "
        "(default: vllm/model_executor/layers/fused_moe/configs)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-file validation results.",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Write results as JSON to the given path (for CI integration).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors.",
    )
    args = parser.parse_args()

    results = run_validation(
        config_dir=args.config_dir,
        verbose=args.verbose,
        strict=args.strict,
    )

    print_results(results, verbose=args.verbose)

    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print(f"JSON results written to {args.json_output}")

    has_errors = bool(results["errors"])
    if has_errors:
        print("RESULT: FAIL")
    else:
        print("RESULT: PASS")

    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
