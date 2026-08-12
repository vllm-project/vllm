# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Docs cross-reference audit: flag doc->code references that no longer resolve.

Every `[text][vllm.X]` autoref, `--8<--` snippet, and `](...py)` link in docs/
is resolved against the current tree. A reference to a missing symbol, module,
or file fails the Read the Docs build (mkdocs `fail_on_warning`), so a nonzero
count exits 1. Only high-confidence breaks are reported; re-export and
dynamic-attribute cases stay silent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..docs import (
    extract_refs,
    is_literal_snippet,
    resolve_autoref,
    resolve_snippet_file,
    symbol_status,
)
from ..graph.build import build_full_graph


def _broken_pylink(ref, repo: Path) -> str | None:
    """url_schemes.py rewrites only links pointing OUTSIDE docs/, and only while
    the target exists; a missing outside-docs target then warns and fails the
    build. Docs-internal links are resolved by mkdocs itself, so leave them
    alone (avoids directory / index.md / anchor false positives)."""
    try:
        resolved = ((repo / ref.md_file).parent / ref.target).resolve()
        rel = resolved.relative_to(repo.resolve()).as_posix()
    except (OSError, ValueError):
        return None
    if rel.startswith("docs/") or resolved.exists():
        return None
    return f"link target missing: {ref.target}"


def _broken_reason(ref, index, repo: Path) -> str | None:
    if ref.kind == "autoref":
        if not ref.target.startswith("vllm."):
            return None  # torch/stdlib inventory or a heading anchor
        module, symbol = resolve_autoref(ref.target, index)
        if module is None or symbol is None:
            return None
        if symbol_status(module, symbol, repo) == "BROKEN":
            return f"symbol {ref.target} absent from {module}"
        return None
    if ref.kind == "snippet":
        if not is_literal_snippet(ref.target):
            return None
        if resolve_snippet_file(ref.target, ref.md_file, repo) is None:
            return f"snippet target missing: {ref.target}"
        return None
    return _broken_pylink(ref, repo)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, required=True)


def run(args) -> int:
    repo = args.repo.resolve()
    index = build_full_graph(repo).index
    refs = list(extract_refs(repo))
    broken = []
    for ref in refs:
        reason = _broken_reason(ref, index, repo)
        if reason is not None:
            broken.append((ref, reason))
    print(f"docs refs: {len(refs)} extracted, {len(broken)} broken")
    for ref, reason in broken:
        print(f"  BROKEN {ref.md_file}:{ref.line} {reason}")
    # Detection floor: extracting nothing looks identical to finding nothing
    # broken. A docs/ move or a syntax change upstream would pass silently.
    if not refs:
        print("  COLLAPSE: zero docs references extracted")
        return 1
    return 1 if broken else 0
