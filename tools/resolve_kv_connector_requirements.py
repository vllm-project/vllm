#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Resolve CUDA-specific kv connector dependencies for Docker builds."""

from __future__ import annotations

import argparse
from pathlib import Path

_LMCACHE_CUDA13_REQ = "lmcache >= 0.4.5"
_LMCACHE_CUDA12_REQ = "lmcache >= 0.3.9, < 0.4.5"
_NAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


def _normalize_name(name: str) -> str:
    return name.lower().replace("_", "-")


def _split_comment(suffix: str) -> tuple[str, str]:
    if "#" not in suffix:
        return suffix.rstrip(), ""

    requirement, comment = suffix.split("#", 1)
    return requirement.rstrip(), f" #{comment}"


def _split_requirement(line: str) -> tuple[str, str, str] | None:
    stripped = line.lstrip()
    if not stripped or stripped.startswith("#"):
        return None

    name_start = len(line) - len(stripped)
    if stripped[0] not in _NAME_CHARS or not stripped[0].isalnum():
        return None

    name_end = name_start
    while name_end < len(line) and line[name_end] in _NAME_CHARS:
        name_end += 1

    return line[:name_start], line[name_start:name_end], line[name_end:]


def _replace_requirement(line: str, requirement: str) -> str:
    split = _split_requirement(line)
    if split is None:
        return line

    indent, _, suffix = split
    _, comment = _split_comment(suffix)
    return f"{indent}{requirement}{comment}"


def _replace_package_name(line: str, replacement: str) -> str:
    split = _split_requirement(line)
    if split is None:
        return line

    indent, _, suffix = split
    return f"{indent}{replacement}{suffix}"


def _package_name(line: str) -> str | None:
    split = _split_requirement(line)
    if split is None:
        return None

    _, name, _ = split
    return _normalize_name(name)


def resolve_requirements(content: str, cuda_major: int) -> str:
    output_lines: list[str] = []
    nixl_runtime = f"nixl-cu{cuda_major}"
    lmcache_req = _LMCACHE_CUDA13_REQ if cuda_major >= 13 else _LMCACHE_CUDA12_REQ

    for line in content.splitlines():
        package_name = _package_name(line)
        if package_name is None:
            output_lines.append(line)
            continue

        if package_name == "lmcache":
            line = _replace_requirement(line, lmcache_req)
        elif package_name == "nixl":
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
    args = parser.parse_args()

    args.output.write_text(
        resolve_requirements(
            content=args.input.read_text(),
            cuda_major=args.cuda_major,
        )
    )


if __name__ == "__main__":
    main()
