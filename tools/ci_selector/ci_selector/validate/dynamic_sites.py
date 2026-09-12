# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic imports the graph cannot follow.

Each unresolved `import_module` / `__import__` / `resolve_obj_by_qualname` call
is either KNOWN (a parser read this file's table, or the file is hand-listed as
importing from outside the repo) or UNCLASSIFIED.

Read at selection time by `codemap/guards.py`, and asserted clean at HEAD by the
drift-marked tests in `tests/test_dynamic_sites.py`, which are the only enforcer.

The parser half is derived, so one that stops matching drops its file and the
warning returns by itself. The hand list stays, because whether an import leaves
the tree is not something a checkout can answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..codemap.graph.imports import DynamicSite
from ..handwritten import DYNAMIC_IMPORT_FILES


def unused_external_entries(sites: list[DynamicSite]) -> list[str]:
    """Hand-list entries with no dynamic import left in them. A stale entry
    pre-approves whatever import lands in that file next, which checking only
    for unclassified sites never catches. The derived half needs no equivalent:
    it drops files by itself."""
    live = {s.file for s in sites}
    return [f for f in DYNAMIC_IMPORT_FILES if f not in live]


@dataclass
class ClassifiedSites:
    external: list[DynamicSite]
    unclassified: list[DynamicSite]


def classify_dynamic_sites(
    sites: list[DynamicSite], table_files: set[str]
) -> ClassifiedSites:
    out = ClassifiedSites([], [])
    known = table_files | set(DYNAMIC_IMPORT_FILES)
    for site in sites:
        (out.external if site.file in known else out.unclassified).append(site)
    return out
