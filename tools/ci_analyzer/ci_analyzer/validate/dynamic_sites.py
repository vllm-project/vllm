# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic-import site audit: prove the graph has no unmodeled dynamic edges.

Each unresolved dynamic call (importlib/__import__/resolve_obj_by_qualname)
is classified, in order:

1. pragma'd - a `# ci: external` comment on the call line.
2. external/table-handled - matched by the audited AUDITED_DYNAMIC_FILES census
   (external by nature, or already structurally handled by a wall parser).
3. UNCLASSIFIED - an unmodeled dynamic dispatch. A non-empty list exits 1.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import regex as re

from ..curated import (
    AUDITED_DYNAMIC_FILES,
    DYNAMIC_IMPORT_FILES,
    WALL_PARSER_FILES,
)
from ..graph.build import build_full_graph
from ..graph.imports import DynamicSite

PRAGMA_RE = re.compile(r"#\s*ci:\s*external\b")


def census_rot(repo: Path, sites: list[DynamicSite]) -> tuple[list[str], list[str]]:
    """The reverse gate on the census: (dead dynamic entries, missing wall
    entries). A DYNAMIC_IMPORT_FILES file with zero live sites blesses a future
    site it should force UNCLASSIFIED; a WALL_PARSER file that moved is a dead
    blessing too. Forward-only checks (0 unclassified) never catch either."""
    live_files = {s.file for s in sites}
    dead = [f for f in DYNAMIC_IMPORT_FILES if f not in live_files]
    missing = [f for f in WALL_PARSER_FILES if not (repo / f).is_file()]
    return dead, missing


@dataclass
class ClassifiedSites:
    external: list[DynamicSite]
    pragma: list[DynamicSite]
    unclassified: list[DynamicSite]


def classify_dynamic_sites(repo: Path, sites: list[DynamicSite]) -> ClassifiedSites:
    out = ClassifiedSites([], [], [])
    line_cache: dict[str, list[str]] = {}
    for site in sites:
        if site.file not in line_cache:
            try:
                line_cache[site.file] = (repo / site.file).read_text().splitlines()
            except (UnicodeDecodeError, OSError):
                line_cache[site.file] = []
        lines = line_cache[site.file]
        line = lines[site.lineno - 1] if site.lineno <= len(lines) else ""
        if PRAGMA_RE.search(line):
            out.pragma.append(site)
        elif site.file.startswith(AUDITED_DYNAMIC_FILES):
            out.external.append(site)
        else:
            out.unclassified.append(site)
    return out


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, required=True)


def run(args) -> int:
    repo = args.repo.resolve()
    fg = build_full_graph(repo)
    classified = classify_dynamic_sites(repo, fg.graph.dynamic_sites)
    print(
        f"dynamic sites: {len(classified.external)} known "
        f"(external/table-handled), {len(classified.pragma)} pragma'd, "
        f"{len(classified.unclassified)} UNCLASSIFIED"
    )
    for site in classified.unclassified:
        print(f"  UNCLASSIFIED {site.file}:{site.lineno} ({site.func})")
    dead, missing = census_rot(repo, fg.graph.dynamic_sites)
    for f in dead:
        print(f"  DEAD CENSUS ENTRY {f} (no live dynamic site; remove it)")
    for f in missing:
        print(f"  MISSING CENSUS FILE {f} (wall-parser entry moved/deleted)")
    if fg.graph.ambiguities:
        print(f"bare-import ambiguities ({len(fg.graph.ambiguities)}):")
        for file, name, sibling, other in fg.graph.ambiguities:
            print(f"  {file}: `{name}` -> {sibling} (sibling wins) vs {other}")
    # Detection floor: an empty site list means the walker stopped recording,
    # so both the forward check and the reverse gate pass vacuously.
    if not fg.graph.dynamic_sites:
        print("  DETECTOR COLLAPSE: walker produced zero dynamic sites")
        return 1
    return 1 if (classified.unclassified or dead or missing) else 0
