#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Generate docker/versions.json from Dockerfile ARG defaults.

This script parses the Dockerfile and extracts ARG defaults to create
a bake-native versions.json file that can be used directly with:
    docker buildx bake -f docker/docker-bake.hcl -f docker/versions.json

Usage:
    python tools/generate_versions_json.py [--check]

Options:
    --check    Verify versions.json matches Dockerfile (for CI validation)

Requirements:
    pip install dockerfile-parse
"""

import json
import sys
from pathlib import Path

from dockerfile_parse import DockerfileParser

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
VERSIONS_JSON = REPO_ROOT / "docker" / "versions.json"

# Map Dockerfile ARG names (lowercase) to bake variable names (uppercase)
# This matches docker-bake.hcl variable naming convention
BAKE_VAR_NAMES = {
    "torch_cuda_arch_list": "TORCH_CUDA_ARCH_LIST",
    "max_jobs": "MAX_JOBS",
    "nvcc_threads": "NVCC_THREADS",
}


def parse_dockerfile_args(dockerfile_path: Path) -> dict[str, str]:
    """Extract all ARG defaults from Dockerfile using dockerfile-parse."""
    parser = DockerfileParser(path=str(dockerfile_path))

    # Extract ARGs from structure (more reliable for multi-stage Dockerfiles)
    args = {}
    for item in parser.structure:
        if item["instruction"] != "ARG":
            continue

        value = item["value"]
        if "=" not in value:
            continue

        # Parse ARG NAME=value (handle quotes)
        name, _, default = value.partition("=")
        name = name.strip()

        if name in args:
            # Keep first occurrence
            continue

        # Strip surrounding quotes if present
        default = default.strip()
        if (default.startswith('"') and default.endswith('"')) or (
            default.startswith("'") and default.endswith("'")
        ):
            default = default[1:-1]

        if default:
            args[name] = default

    # Resolve variable interpolation (e.g., ${CUDA_VERSION} -> 12.9.1)
    resolved = {}
    for name, value in args.items():
        if "${" in value:
            # Substitute ${VAR} references with their values
            for ref_name, ref_value in args.items():
                value = value.replace(f"${{{ref_name}}}", ref_value)
        # Skip if still has unresolved references (no default available)
        if "${" not in value:
            resolved[name] = value

    return resolved


def generate_bake_native_json(args: dict[str, str]) -> dict:
    """Generate bake-native JSON structure."""
    variables = {}
    for name, value in args.items():
        # Use uppercase bake variable name if mapped, otherwise keep as-is
        bake_name = BAKE_VAR_NAMES.get(name, name)
        variables[bake_name] = {"default": value}

    return {
        "_comment": (
            "Auto-generated from Dockerfile ARGs. "
            "Do not edit manually. Run: python tools/generate_versions_json.py"
        ),
        "variable": variables,
    }


def main():
    check_mode = "--check" in sys.argv

    # Parse Dockerfile
    args = parse_dockerfile_args(DOCKERFILE)

    # Generate bake-native JSON
    data = generate_bake_native_json(args)
    new_content = json.dumps(data, indent=2) + "\n"

    if check_mode:
        # Verify existing file matches
        if not VERSIONS_JSON.exists():
            print(f"ERROR: {VERSIONS_JSON} does not exist")
            sys.exit(1)

        existing_content = VERSIONS_JSON.read_text()
        if existing_content != new_content:
            print("ERROR: docker/versions.json is out of sync with Dockerfile")
            print("Run: python tools/generate_versions_json.py")
            sys.exit(1)

        print("✅ docker/versions.json is in sync with Dockerfile")
        sys.exit(0)

    # Write versions.json
    VERSIONS_JSON.write_text(new_content)
    print(f"✅ Generated {VERSIONS_JSON}")

    # Print summary
    print("\nExtracted versions:")
    for name, value in args.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
