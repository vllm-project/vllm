#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Export exact repository translation-unit -> CUDA-kernel edges.

The artifact-level kernel map answers which shared object defines a kernel,
but a shared object can contain hundreds of CUDA sources. This exporter scans
the final object files while the native build tree still exists, finds nvcc
device stubs in each object, and uses that object's Ninja dependency rule to
identify its one compiled repository source.

Output edge shape::

    file --compiles_kernel--> kernel

The CMake target and object path are retained as metadata. Kernel-bearing
objects whose source is generated or outside the repository are counted and
left explicit in the summary rather than guessed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
from collections import defaultdict

try:
    from .depfiles import load_rules, target_from_object_path
    from .extract_kernel_symbols import kernel_names
except ImportError:
    from depfiles import load_rules, target_from_object_path
    from extract_kernel_symbols import kernel_names


def _repo_relative_dependency(
    dependency: str, build_dir: pathlib.Path, source_root: pathlib.Path
) -> str | None:
    path = pathlib.Path(dependency)
    if not path.is_absolute():
        path = build_dir / path
    resolved = path.resolve()
    try:
        return resolved.relative_to(source_root).as_posix()
    except ValueError:
        return None


def export_object_kernel_edges(
    build_dir: pathlib.Path,
    source_root: pathlib.Path,
    rules: dict[str, list[str]],
    member_sources: set[tuple[str, str]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    build_dir = build_dir.resolve()
    source_root = source_root.resolve()
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str]] = set()
    stats: defaultdict[str, int] = defaultdict(int)

    for object_name, dependencies in sorted(rules.items()):
        target = target_from_object_path(object_name)
        if target is None:
            stats["objects_without_target"] += 1
            continue
        object_path = pathlib.Path(object_name)
        if not object_path.is_absolute():
            object_path = build_dir / object_path
        if not object_path.is_file():
            stats["objects_missing"] += 1
            continue
        stats["objects_scanned"] += 1
        kernels = kernel_names(str(object_path))
        if not kernels:
            continue
        stats["kernel_objects"] += 1
        stats["kernel_symbols"] += len(kernels)

        source_candidates = sorted(
            {
                relative
                for dependency in dependencies
                if (
                    relative := _repo_relative_dependency(
                        dependency, build_dir, source_root
                    )
                )
                is not None
                and (relative, target) in member_sources
            }
        )
        if len(source_candidates) != 1:
            stats[
                "kernel_objects_without_repo_source"
                if not source_candidates
                else "kernel_objects_with_ambiguous_repo_sources"
            ] += 1
            continue

        source = source_candidates[0]
        try:
            object_relative = object_path.resolve().relative_to(build_dir).as_posix()
        except ValueError:
            object_relative = str(object_path.resolve())
        for kernel in sorted(kernels):
            key = (source, kernel, target, object_relative)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "source_kind": "file",
                    "source": source,
                    "edge_kind": "compiles_kernel",
                    "destination_kind": "kernel",
                    "destination": kernel,
                    "target": target,
                    "object_path": object_relative,
                }
            )
            stats["translation_unit_kernel_edges"] += 1

    rows.sort(key=lambda row: json.dumps(row, sort_keys=True))
    stats["unique_translation_units"] = len({row["source"] for row in rows})
    stats["unique_mapped_kernels"] = len({row["destination"] for row in rows})
    return rows, dict(stats)


def _load_member_sources(path: pathlib.Path) -> set[tuple[str, str]]:
    members = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (
                row.get("source_kind") == "file"
                and row.get("edge_kind") == "member_of"
                and row.get("destination_kind") == "target"
            ):
                members.add((row["source"], row["destination"]))
    return members


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=pathlib.Path)
    parser.add_argument("--source-root", type=pathlib.Path, required=True)
    parser.add_argument("--ninja-deps", type=pathlib.Path, required=True)
    parser.add_argument("--build-graph", type=pathlib.Path, required=True)
    parser.add_argument("--out", default="-")
    args = parser.parse_args(argv)

    rules, source = load_rules(args.ninja_deps, args.build_dir)
    if source != "ninja_deps":
        raise SystemExit("object kernel provenance requires `ninja -t deps` input")
    rows, stats = export_object_kernel_edges(
        args.build_dir,
        args.source_root,
        rules,
        _load_member_sources(args.build_graph),
    )

    if args.out == "-":
        for row in rows:
            print(json.dumps(row, sort_keys=True))
    else:
        output = pathlib.Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=output.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = pathlib.Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        temporary.replace(output)
    print(json.dumps(stats, indent=2, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
