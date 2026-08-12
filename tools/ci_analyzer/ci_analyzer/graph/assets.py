# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data-asset wall parser: literal path strings in tests -> file edges.

A test that names an in-repo file in a string literal (chat templates, prompt
txt, fixture json, spec_from_file_location targets) depends on it. A .py target
is a graph node whose own import closure composes normally; data files are leaves
selected directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .imports import ImportGraph

CANDIDATE_ROOTS = ("", "tests/")


@dataclass
class AssetParse:
    edges_added: int = 0
    files_claimed: set[str] = field(default_factory=set)


def _looks_like_path(lit: str) -> bool:
    return (
        "/" in lit
        and " " not in lit
        and not lit.startswith(("http://", "https://", "s3://", "gs://", "/"))
        and "." in lit.rsplit("/", 1)[-1]
    )


def add_asset_edges(repo: Path, graph: ImportGraph) -> AssetParse:
    parse = AssetParse()
    for test_file, literals in graph.string_literals.items():
        own_dir = test_file.rsplit("/", 1)[0] + "/"
        for lit in literals:
            if not _looks_like_path(lit):
                continue
            for root in (*CANDIDATE_ROOTS, own_dir):
                candidate = f"{root}{lit}"
                path = repo / candidate
                if path.is_file():
                    if candidate != test_file:
                        graph.add_edge(test_file, candidate)
                        parse.edges_added += 1
                        parse.files_claimed.add(candidate)
                    break
    return parse
