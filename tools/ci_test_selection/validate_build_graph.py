#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate one native file-to-kernel path in a combined build graph."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

Node = tuple[str, str]


def load_graph(path: Path) -> dict[Node, set[Node]]:
    graph: dict[Node, set[Node]] = defaultdict(set)
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row: dict[str, Any] = json.loads(line)
                source = (str(row["source_kind"]), str(row["source"]))
                destination = (
                    str(row["destination_kind"]),
                    str(row["destination"]),
                )
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(
                    f"invalid edge on line {line_number}: {error}"
                ) from error
            graph[source].add(destination)
    return graph


def find_path(graph: dict[Node, set[Node]], start: Node, target: Node) -> list[Node]:
    queue = deque([start])
    parent: dict[Node, Node | None] = {start: None}
    while queue:
        node = queue.popleft()
        if node == target:
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return list(reversed(path))
        for successor in sorted(graph.get(node, ())):
            if successor not in parent:
                parent[successor] = node
                queue.append(successor)
    return []


def validate(path: Path, required_file: str, required_target: str) -> list[Node]:
    graph = load_graph(path)
    start = ("file", required_file)
    target = ("target", required_target)
    prefix = find_path(graph, start, target)
    if not prefix:
        raise ValueError(f"no path from {start!r} to {target!r}")

    for artifact in sorted(graph.get(target, ())):
        if artifact[0] != "artifact":
            continue
        kernels = sorted(
            node for node in graph.get(artifact, ()) if node[0] == "kernel"
        )
        if kernels:
            return [*prefix, artifact, kernels[0]]
    raise ValueError(f"{target!r} has no structurally joined artifact-to-kernel path")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path)
    parser.add_argument("--required-file", required=True)
    parser.add_argument("--required-target", required=True)
    args = parser.parse_args(argv)
    path = validate(args.graph, args.required_file, args.required_target)
    print(json.dumps({"validated_path": path}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
