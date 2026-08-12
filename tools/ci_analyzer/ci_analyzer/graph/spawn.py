# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spawn wall parser: literal ["vllm", ...] argv -> entrypoint closure.

Edge shape avoids the helper-amplifier problem: spawn sites live in helper
classes (RemoteOpenAIServer et al.), and edging the helper file would drag its
~180 importers into every engine closure. Instead, leaf test files that import a
spawner class, or contain their own spawn argv, get the edge directly.
[sys.executable, "-c", ...] sites are unresolvable by design and noted, not
parsed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..curated import CLI_ENTRYPOINT_MODULE, CONFTESTS_NOT_ENGINE_STARTING
from ..repo import ModuleIndex
from .imports import ImportGraph


@dataclass
class SpawnParse:
    spawner_classes: dict[str, set[str]] = field(default_factory=dict)
    edges_added: int = 0
    entrypoint_file: str | None = None


def _spawn_leaf(path: str) -> bool:
    """Leaf files that get a direct spawn edge: test files, plus conftests
    whose server fixtures boot engines for every test beneath them (minus
    the curated amplifier excludes)."""
    basename = path.rsplit("/", 1)[-1]
    if basename.startswith("test_"):
        return True
    return (
        basename == "conftest.py"
        and path.startswith("tests/")
        and path not in CONFTESTS_NOT_ENGINE_STARTING
    )


def add_spawn_edges(repo: Path, index: ModuleIndex, graph: ImportGraph) -> SpawnParse:
    parse = SpawnParse()
    entry = index.resolve(CLI_ENTRYPOINT_MODULE)
    parse.entrypoint_file = entry
    if entry is None:
        return parse
    # helper file -> class names containing a spawn argv literal
    for file, owners in graph.spawn_sites.items():
        classes = {c for c in owners if c}
        if _spawn_leaf(file):
            graph.add_edge(file, entry)
            parse.edges_added += 1
        elif classes:
            parse.spawner_classes[file] = classes
    for file, base_file, alias in graph.from_import_aliases:
        if not _spawn_leaf(file):
            continue
        if alias in parse.spawner_classes.get(base_file, ()):
            graph.add_edge(file, entry)
            parse.edges_added += 1
    return parse
