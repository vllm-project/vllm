# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-dispatch wall parser (files under vllm/platforms/ only).

For every get_* method across the platform classes, union every string literal
returned in the body (no dataflow) and edge each `X.get_<method>()` call site to
all candidate modules. A method with any non-literal return falls back to edges
onto all platform modules (small, bounded). C-extension imports (the CMake seam)
are recorded, not edged.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import regex as re

from ..curated import PLATFORMS_DIR
from ..repo import ModuleIndex
from .factories import _record_parse_error
from .imports import ImportGraph


@dataclass
class PlatformParse:
    # method name -> candidate module files (resolved literals)
    candidates: dict[str, set[str]] = field(default_factory=dict)
    # methods with non-literal returns -> fallback to all platform modules
    non_literal: set[str] = field(default_factory=set)
    unresolved: dict[str, set[str]] = field(default_factory=dict)
    consumers: dict[str, set[str]] = field(default_factory=dict)  # method->files
    edges_added: int = 0
    # platform files that would not parse. Merged into graph.parse_errors by
    # the caller so a changed one fails open: an unparsed platform file loses
    # table entries, and losing entries under-selects.
    parse_errors: list[str] = field(default_factory=list)


def _method_return_literals(func: ast.FunctionDef) -> tuple[set[str], bool]:
    literals: set[str] = set()
    all_literal = True
    for node in ast.walk(func):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            literals.add(value.value)
        elif isinstance(value, ast.Constant):
            continue  # None/bool returns are not dispatch targets
        else:
            all_literal = False
    return literals, all_literal


def parse_platform_tables(repo: Path, index: ModuleIndex) -> PlatformParse:
    parse = PlatformParse()
    base = repo / PLATFORMS_DIR
    for path in sorted(base.glob("*.py")):
        rel = path.relative_to(repo).as_posix()
        try:
            tree = ast.parse(path.read_text(), filename=rel)
        except SyntaxError:
            parse.parse_errors.append(rel)
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if not isinstance(
                    item, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) or not item.name.startswith("get_"):
                    continue
                literals, all_literal = _method_return_literals(item)
                if not all_literal:
                    parse.non_literal.add(item.name)
                for lit in literals:
                    resolved = _resolve_qualname(lit, index)
                    if resolved:
                        parse.candidates.setdefault(item.name, set()).add(resolved)
                    elif "." in lit:
                        parse.unresolved.setdefault(item.name, set()).add(lit)
    return parse


def _resolve_qualname(qualname: str, index: ModuleIndex) -> str | None:
    resolved = index.resolve(qualname)
    if resolved is None and "." in qualname:
        resolved = index.resolve(qualname.rsplit(".", 1)[0])
    return resolved


def add_platform_qualname_edges(
    repo: Path,
    index: ModuleIndex,
    graph: ImportGraph,
    claimed: set[str],
    parse: PlatformParse,
) -> None:
    """Platform files wire engine internals by literal qualname ASSIGNMENT,
    not only by get_* returns: e.g. cuda.py sets parallel_config.worker_cls =
    "vllm.v1.worker.gpu_worker.Worker", resolved later by worker_base. Every
    in-repo-resolvable "vllm.*" string literal in a platform file becomes an
    edge platform_file -> target, and platforms/__init__.py gets edges to the
    platform modules it dispatches among. Targets that a string-keyed parser
    claims (attention backends, models) are SKIPPED: their coverage routes
    through leaf test edges, and an import edge here would re-amplify them
    (found by the #49364 revert replay: test_logprobs -> engine ->
    worker_cls seam -> cudagraph_utils was unreachable)."""
    qualname_re = re.compile(r"^vllm\.[\w.]+$")
    base = repo / PLATFORMS_DIR
    init_file = f"{PLATFORMS_DIR}/__init__.py"
    for path in sorted(base.glob("*.py")):
        rel = path.relative_to(repo).as_posix()
        if rel != init_file:
            graph.add_edge(init_file, rel)
            parse.edges_added += 1
        try:
            tree = ast.parse(path.read_text(), filename=rel)
        except SyntaxError:
            _record_parse_error(graph, rel)
            continue
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and qualname_re.match(node.value)
            ):
                continue
            target = _resolve_qualname(node.value, index)
            if (
                target
                and target != rel
                and target not in claimed
                and not any(c.endswith("/") and target.startswith(c) for c in claimed)
            ):
                # Gated: worker_cls etc. resolve only when an engine boots, so
                # tests reached only via them are limited to engine-starting ones.
                graph.add_gated_edge(rel, target)
                parse.edges_added += 1


def add_platform_edges(
    repo: Path,
    index: ModuleIndex,
    graph: ImportGraph,
    claimed: set[str] = frozenset(),
) -> PlatformParse:
    """Consumers come from the main visitor pass (graph.method_calls)."""
    parse = parse_platform_tables(repo, index)
    add_platform_qualname_edges(repo, index, graph, set(claimed), parse)
    for rel in parse.parse_errors:
        _record_parse_error(graph, rel)
    all_platform_files = {
        f for f in index.file_to_module if f.startswith(PLATFORMS_DIR + "/")
    }
    for method in set(parse.candidates) | parse.non_literal:
        files = graph.method_calls.get(method, set())
        if not files:
            continue
        parse.consumers[method] = files
        targets = set(parse.candidates.get(method, ()))
        if method in parse.non_literal:
            targets |= all_platform_files
        for consumer in files:
            for target in targets:
                if target != consumer:
                    graph.add_edge(consumer, target)
                    parse.edges_added += 1
    return parse
