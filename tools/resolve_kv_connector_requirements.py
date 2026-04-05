#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Resolve CUDA-specific kv connector dependencies for Docker builds."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _normalize_name(name: str) -> str:
    return name.lower().replace("_", "-")


def _package_name(line: str) -> str | None:
    stripped = line.lstrip()
    if not stripped or stripped.startswith("#"):
        return None
    match = _REQ_NAME_RE.match(line)
    if match is None:
        return None
    return _normalize_name(match.group(1))


def _replace_package_name(line: str, replacement: str) -> str:
    return _REQ_NAME_RE.sub(replacement, line, count=1)


def resolve_requirements(
    content: str,
    cuda_major: int,
    skip_packages: set[str] | None = None,
) -> str:
    skip_packages = skip_packages or set()
    output_lines: list[str] = []
    nixl_runtime = "nixl-cu13" if cuda_major >= 13 else "nixl-cu12"

    for line in content.splitlines():
        package_name = _package_name(line)
        if package_name is None:
            output_lines.append(line)
            continue

        if package_name in skip_packages:
            continue

        if package_name == "nixl":
            line = _replace_package_name(line, nixl_runtime)

        output_lines.append(line)

    return "\n".join(output_lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite kv connector requirements to use CUDA-runtime-specific "
            "packages during Docker builds."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cuda-major", required=True, type=int)
    parser.add_argument(
        "--skip-package",
        action="append",
        default=[],
        help="Normalized package name to omit from the output.",
    )
    args = parser.parse_args()

    content = args.input.read_text()
    resolved = resolve_requirements(
        content=content,
        cuda_major=args.cuda_major,
        skip_packages={_normalize_name(name) for name in args.skip_package},
    )
    args.output.write_text(resolved)


if __name__ == "__main__":
    main()
