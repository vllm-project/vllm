#!/usr/bin/env python3
"""Aggregate per-step coverage JSON files into a test-selection mapping.

Downloads all coverage_*.json artifacts from the current Buildkite build,
then produces two output files:

1. coverage_map.json — inverted index: {source_file: [step_keys]}
   Used by the pipeline generator to determine which steps to trigger.

2. step_coverage.json — forward index: {step_key: [source_files]}
   Useful for debugging and understanding test coverage.

Usage:
    # Run as a Buildkite step at the end of nightly CI
    python3 .buildkite/scripts/coverage/aggregate-coverage.py

    # Or locally with downloaded artifacts
    python3 .buildkite/scripts/coverage/aggregate-coverage.py --local-dir ./artifacts/
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


def download_artifacts(dest_dir: str) -> list[str]:
    """Download all coverage_*.json artifacts from the current build."""
    try:
        subprocess.run(
            ["buildkite-agent", "artifact", "download", "coverage_*.json", dest_dir],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("buildkite-agent not found, skipping download", file=sys.stderr)
        return []
    except subprocess.CalledProcessError as e:
        print(f"Artifact download failed: {e.stderr}", file=sys.stderr)
        return []

    return list(Path(dest_dir).glob("coverage_*.json"))


def load_coverage_files(files: list[Path]) -> dict[str, list[str]]:
    """Load coverage JSON files and extract source files per step.

    Returns: {step_key: [source_files]}
    """
    step_coverage = {}

    for filepath in files:
        filename = filepath.name
        # coverage_<step_key>.json -> step_key
        step_key = filename.removeprefix("coverage_").removesuffix(".json")

        try:
            with open(filepath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skipping {filename}: {e}", file=sys.stderr)
            continue

        source_files = []
        for fpath, fdata in data.get("files", {}).items():
            # Skip files with zero executed lines — coverage.py reports
            # all files in the source tree, not just those actually run.
            if fdata.get("summary", {}).get("covered_lines", 0) == 0:
                continue
            # Normalize paths to be relative to the vllm package root.
            # coverage.py may report absolute paths or paths relative to
            # the installed package location. We only care about files
            # under the vllm/ directory.
            normalized = _normalize_path(fpath)
            if normalized:
                source_files.append(normalized)

        if source_files:
            step_coverage[step_key] = sorted(set(source_files))
            print(f"  {step_key}: {len(source_files)} source files")

    return step_coverage


def _normalize_path(path: str) -> str | None:
    """Normalize a coverage path to a vllm-relative path.

    Returns None for paths outside the vllm package (tests, third-party, etc).
    """
    # Strip common prefixes from installed package paths
    markers = ["/site-packages/", "/dist-packages/", "/vllm-workspace/src/"]
    for marker in markers:
        idx = path.find(marker)
        if idx != -1:
            path = path[idx + len(marker):]
            break

    # Also handle paths that are already relative
    if path.startswith("vllm/"):
        return path

    # Handle absolute paths that contain /vllm/
    idx = path.find("/vllm/")
    if idx != -1:
        return path[idx + 1:]

    return None


def build_inverted_index(
    step_coverage: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Build {source_file: [step_keys]} from {step_key: [source_files]}."""
    inverted = defaultdict(list)
    for step_key, source_files in step_coverage.items():
        for src_file in source_files:
            inverted[src_file].append(step_key)

    # Sort step lists for deterministic output
    return {k: sorted(v) for k, v in sorted(inverted.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-dir",
        help="Directory containing coverage_*.json files (skip artifact download)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write output files (default: cwd)",
    )
    args = parser.parse_args()

    if args.local_dir:
        artifact_dir = args.local_dir
        files = list(Path(artifact_dir).glob("coverage_*.json"))
    else:
        artifact_dir = tempfile.mkdtemp(prefix="coverage_artifacts_")
        files = download_artifacts(artifact_dir)

    if not files:
        print("No coverage files found. Nothing to aggregate.")
        sys.exit(0)

    print(f"Found {len(files)} coverage files:")

    # Build the forward index: step -> source files
    step_coverage = load_coverage_files(files)

    if not step_coverage:
        print("No valid coverage data found.")
        sys.exit(0)

    # Build the inverted index: source file -> steps
    coverage_map = build_inverted_index(step_coverage)

    # Write outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step_coverage_path = output_dir / "step_coverage.json"
    with open(step_coverage_path, "w") as f:
        json.dump(step_coverage, f, indent=2)
    print(f"\nWrote {step_coverage_path} ({len(step_coverage)} steps)")

    coverage_map_path = output_dir / "coverage_map.json"
    with open(coverage_map_path, "w") as f:
        json.dump(coverage_map, f, indent=2)
    print(f"Wrote {coverage_map_path} ({len(coverage_map)} source files)")

    # Summary stats
    total_files = len(coverage_map)
    total_mappings = sum(len(v) for v in coverage_map.values())
    print(f"\nSummary: {total_files} source files mapped to "
          f"{len(step_coverage)} steps ({total_mappings} total mappings)")

    # Upload aggregated files as artifacts
    for output_file in [step_coverage_path, coverage_map_path]:
        try:
            subprocess.run(
                ["buildkite-agent", "artifact", "upload", str(output_file)],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"Uploaded {output_file}")
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass  # Not in Buildkite or upload failed — that's fine for local runs


if __name__ == "__main__":
    main()
