# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Factory/lazy-table wall parsers beyond the model registry.

Literal registration calls (register_*("Key", "module", "Class") for KV/EC
connectors, weight transfer, kv_offload, sleep-mode), the reasoning/tool-parser
lazy tables, attention-backend enum qualnames, vllm/__init__ MODULE_ATTRS lazy
imports, and pkgutil package enumerators all become string-keyed leaf edges keyed
by the registered name; their targets become claims.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import regex as re

from ..curated import (
    ATTN_REGISTRY,
    REASONING_INIT,
    TOOL_PARSER_INIT,
    TRANSFORMERS_CONFIGS_INIT,
    TRANSFORMERS_PROCESSORS_INIT,
    VLLM_INIT,
)
from ..repo import ModuleIndex, is_test_basename
from .imports import ImportGraph


@dataclass
class FactoryParse:
    register_entries: dict[str, str] = field(default_factory=dict)  # key->file
    parser_entries: dict[str, str] = field(default_factory=dict)
    # Entries parsed per lazy table (anchor -> count). Parser names collide
    # across the four tables (deepseek_v3 is both a reasoning and a tool
    # parser), so parser_entries above is last-wins: 93 parsed entries land as
    # 56 keys, and a dead tokenizers anchor leaves its size unchanged. Preflight
    # therefore guards each table on its own count, not on the merged dict.
    parser_table_counts: dict[str, int] = field(default_factory=dict)
    enum_entries: dict[str, str] = field(default_factory=dict)
    module_attrs: dict[str, str] = field(default_factory=dict)
    pkgutil_dirs: list[str] = field(default_factory=list)
    # class name -> module file, from _CLASS_TO_MODULE tables
    class_table_entries: dict[str, str] = field(default_factory=dict)
    # unified parser-engine module stem -> vllm/parser/<stem>.py
    parser_engine_entries: dict[str, str] = field(default_factory=dict)
    claims: set[str] = field(default_factory=set)
    edges_added: int = 0
    module_attr_resolved: int = 0


def _record_parse_error(graph: ImportGraph, file: str) -> None:
    if file not in graph.parse_errors:
        graph.parse_errors.append(file)


def _leaf_consumer(file: str) -> bool:
    """Files that EXERCISE a keyed target rather than register it: tests plus
    example/benchmark scripts (steps run those directly; a script selecting a
    key in its argv/choices depends on the keyed module exactly like a test)."""
    return is_test_basename(file) or file.startswith(("examples/", "benchmarks/"))


def _leaf_edges(graph: ImportGraph, keys: set[str], target: str) -> int:
    added = 0
    for leaf_file, literals in graph.string_literals.items():
        if _leaf_consumer(leaf_file) and not keys.isdisjoint(literals):
            graph.add_edge(leaf_file, target)
            added += 1
    return added


def _path_leaf_edges(graph: ImportGraph, tokens: set[str], target: str) -> int:
    """Like _leaf_edges, but keys off the leaf file's PATH rather than its
    string literals: a runtime-only integration test (tests/tool_use/<name>/...)
    can exercise a parser with neither a static import nor an exact literal."""
    added = 0
    for leaf_file in graph.string_literals:
        low = leaf_file.lower()
        if _leaf_consumer(leaf_file) and any(t in low for t in tokens):
            graph.add_edge(leaf_file, target)
            added += 1
    return added


def _claim(parse: FactoryParse, target: str) -> None:
    parse.claims.add(target)
    if target.endswith("/__init__.py"):
        parse.claims.add(target[: -len("__init__.py")])


MODULE_PATH_RE = re.compile(r"^[a-zA-Z_][\w.]*\.[\w.]+$")


def add_register_call_edges(
    repo: Path, index: ModuleIndex, graph: ImportGraph, parse: FactoryParse
) -> None:
    for file in list(index.file_to_module):
        if not file.startswith("vllm/"):
            continue
        text_path = repo / file
        try:
            text = text_path.read_text()
        except (UnicodeDecodeError, OSError):
            _record_parse_error(graph, file)
            continue
        if "register_" not in text:
            continue
        try:
            tree = ast.parse(text, filename=file)
        except SyntaxError:
            _record_parse_error(graph, file)
            continue
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, (ast.Attribute, ast.Name))
            ):
                continue
            name = (
                node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            )
            if not name.startswith("register_"):
                continue
            consts = [
                a.value
                for a in node.args
                if isinstance(a, ast.Constant) and isinstance(a.value, str)
            ]
            if len(consts) < 2:
                continue
            key = consts[0]
            module = next(
                (c for c in consts[1:] if MODULE_PATH_RE.match(c) and index.resolve(c)),
                None,
            )
            if module is None:
                continue
            target = index.resolve(module)
            parse.register_entries[key] = target
            _claim(parse, target)
            parse.edges_added += _leaf_edges(graph, {key}, target)


def add_lazy_parser_table_edges(
    repo: Path, index: ModuleIndex, graph: ImportGraph, parse: FactoryParse
) -> None:
    for init_file, prefix in (
        (REASONING_INIT, "vllm.reasoning"),
        (TOOL_PARSER_INIT, "vllm.tool_parsers"),
        # Same key -> (module_stem, Class) shape as the parser tables;
        # the dynamic-sites audit relies on these being parsed.
        ("vllm/renderers/registry.py", "vllm.renderers"),
        ("vllm/tokenizers/registry.py", "vllm.tokenizers"),
    ):
        # Zeroed before the anchor is opened: a moved or unparsable table must
        # read as empty rather than as a missing row.
        parse.parser_table_counts[init_file] = 0
        path = repo / init_file
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(), filename=init_file)
        except (SyntaxError, UnicodeDecodeError, OSError):
            _record_parse_error(graph, init_file)
            continue
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and (
                    node.targets[0].id.endswith("_TO_REGISTER")
                    or node.targets[0].id in ("_VLLM_RENDERERS", "_VLLM_TOKENIZERS")
                )
            ):
                continue
            if isinstance(node.value, ast.Dict):
                entries = [
                    ([k.value], v)
                    for k, v in zip(node.value.keys, node.value.values)
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                ]
            elif isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
                entries = [([], elt) for elt in node.value.elts]
            else:
                continue
            for keys, elt in entries:
                consts = keys + [
                    c.value
                    for c in ast.walk(elt)
                    if isinstance(c, ast.Constant) and isinstance(c.value, str)
                ]
                if not consts:
                    continue
                key = consts[0]
                target = None
                for c in consts:
                    target = index.resolve(f"{prefix}.{c}")
                    if target:
                        break
                if target is None:
                    continue
                parse.parser_entries[key] = target
                parse.parser_table_counts[init_file] += 1
                _claim(parse, target)
                parse.edges_added += _leaf_edges(graph, set(consts), target)


def add_attention_enum_edges(
    repo: Path, index: ModuleIndex, graph: ImportGraph, parse: FactoryParse
) -> None:
    path = repo / ATTN_REGISTRY
    if not path.is_file():
        return
    try:
        tree = ast.parse(path.read_text(), filename=ATTN_REGISTRY)
    except (SyntaxError, UnicodeDecodeError, OSError):
        _record_parse_error(graph, ATTN_REGISTRY)
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or "Enum" not in ast.dump(node):
            continue
        for item in node.body:
            if not (
                isinstance(item, ast.Assign)
                and len(item.targets) == 1
                and isinstance(item.targets[0], ast.Name)
                and isinstance(item.value, ast.Constant)
                and isinstance(item.value.value, str)
            ):
                continue
            member, qualname = item.targets[0].id, item.value.value
            target = index.resolve(qualname) or (
                index.resolve(qualname.rsplit(".", 1)[0]) if "." in qualname else None
            )
            if target is None:
                continue
            parse.enum_entries[member] = target
            _claim(parse, target)
            parse.edges_added += _leaf_edges(graph, {member}, target)


def add_module_attr_edges(
    repo: Path, index: ModuleIndex, graph: ImportGraph, parse: FactoryParse
) -> None:
    path = repo / VLLM_INIT
    try:
        tree = ast.parse(path.read_text(), filename=VLLM_INIT)
    except (SyntaxError, UnicodeDecodeError, OSError):
        _record_parse_error(graph, VLLM_INIT)
        return
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "MODULE_ATTRS"
            and isinstance(node.value, ast.Dict)
        ):
            continue
        for k, v in zip(node.value.keys, node.value.values):
            if (
                isinstance(k, ast.Constant)
                and isinstance(k.value, str)
                and isinstance(v, ast.Constant)
                and isinstance(v.value, str)
            ):
                # Values are "module[:attr]" (vllm/__init__.py splits on ":"
                # before importing); keep the module part only.
                parse.module_attrs[k.value] = v.value.split(":", 1)[0].lstrip(".")
    if not parse.module_attrs:
        return
    for file, base_file, alias in graph.from_import_aliases:
        if base_file != VLLM_INIT:
            continue
        module = parse.module_attrs.get(alias)
        if not module:
            continue
        target = index.resolve(f"vllm.{module}") or index.resolve(module)
        if target:
            graph.add_edge(file, target)
            parse.edges_added += 1
            parse.module_attr_resolved += 1


def add_class_module_table_edges(
    repo: Path, index: ModuleIndex, graph: ImportGraph, parse: FactoryParse
) -> None:
    """_CLASS_TO_MODULE tables (transformers_utils configs/processors):
    class name -> FULL module qualname, spelled as AnnAssign at HEAD. The
    modules become claims (lazy-import deferral keeps working) with leaf
    edges by class-name literal (zero today; self-materializing the day a
    test names one) plus real edges for from-import alias consumers."""
    for init_file in (TRANSFORMERS_CONFIGS_INIT, TRANSFORMERS_PROCESSORS_INIT):
        path = repo / init_file
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(), filename=init_file)
        except (SyntaxError, UnicodeDecodeError, OSError):
            _record_parse_error(graph, init_file)
            continue
        table: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                target, value = node.target, node.value
            else:
                continue
            if not (
                isinstance(target, ast.Name)
                and target.id == "_CLASS_TO_MODULE"
                and isinstance(value, ast.Dict)
            ):
                continue
            for k, v in zip(value.keys, value.values):
                if (
                    isinstance(k, ast.Constant)
                    and isinstance(k.value, str)
                    and isinstance(v, ast.Constant)
                    and isinstance(v.value, str)
                ):
                    table[k.value] = v.value
        for cls_name, qualname in table.items():
            target_file = index.resolve(qualname)
            if target_file is None:
                continue  # external value ("transformers")
            parse.class_table_entries[cls_name] = target_file
            _claim(parse, target_file)
            parse.edges_added += _leaf_edges(graph, {cls_name}, target_file)
        for file, base_file, alias in graph.from_import_aliases:
            if base_file != init_file:
                continue
            qualname = table.get(alias)
            target_file = index.resolve(qualname) if qualname else None
            if target_file:
                graph.add_edge(file, target_file)
                parse.edges_added += 1


def add_pkgutil_edges(
    repo: Path, index: ModuleIndex, graph: ImportGraph, parse: FactoryParse
) -> None:
    for file in list(index.file_to_module):
        if not file.endswith("/__init__.py") or not file.startswith("vllm/"):
            continue
        try:
            text = (repo / file).read_text()
        except (UnicodeDecodeError, OSError):
            _record_parse_error(graph, file)
            continue
        if "iter_modules" not in text and "walk_packages" not in text:
            continue
        pkg_dir = file[: -len("__init__.py")]
        parse.pkgutil_dirs.append(pkg_dir)
        for sibling in index.file_to_module:
            if sibling.startswith(pkg_dir) and sibling != file:
                graph.add_edge(file, sibling)
                parse.edges_added += 1


# The unified parser engine is dispatched from these two hubs; both are
# graph-node ids we read edges from, not files we re-parse.
_REGISTERED_ADAPTERS = "vllm/parser/engine/registered_adapters.py"
_PARSER_MANAGER = "vllm/parser/parser_manager.py"


def _is_parser_leaf(dst: str) -> bool:
    return (
        dst.startswith("vllm/parser/")
        and dst.count("/") == 2
        and not dst.endswith("/__init__.py")
    )


def _parser_engine_modules(graph: ImportGraph) -> set[str]:
    """The unified parser engine leaves (vllm/parser/<name>.py): the concrete
    parsers registered_adapters imports at load time, plus the parser_manager
    model dispatch reached only by a function-local (lazy) import. Base/shared
    modules the concrete engines themselves import (abstract_parser, utils) are
    infra, not dispatch targets -- excluded so they are never claimed.

    NOT idempotent after build_full_graph: reads graph.lazy_imports, which
    finalize_lazy_edges clears. Consumers past build read parser_engine_entries.
    """
    primary = {
        dst
        for dst in graph.imports.get(_REGISTERED_ADAPTERS, ())
        if _is_parser_leaf(dst)
    }
    # A base/shared parser module (abstract_parser) is imported BY another
    # parser leaf; a concrete engine is imported by nothing (or only __init__).
    # Concrete engines that reuse each other (nemotron<-qwen3) still register
    # via `primary`, so this only filters the parser_manager candidates.
    shared = {
        d
        for src, dsts in graph.imports.items()
        if _is_parser_leaf(src)
        for d in dsts
        if _is_parser_leaf(d) and d != src
    }
    engines = set(primary)
    for src, dst in graph.lazy_imports:
        if src == _PARSER_MANAGER and _is_parser_leaf(dst) and dst not in shared:
            engines.add(dst)
    return engines


def add_parser_engine_edges(
    repo: Path, index: ModuleIndex, graph: ImportGraph, parse: FactoryParse
) -> None:
    """The fifth registry (after register-calls, parser tables, attention enum,
    class tables): the unified parser engine. Each vllm/parser/<name>.py is
    CLAIMED and keyed by its module stem, so finalize_lazy_edges drops the
    conservative lazy edge that otherwise routes it through the api_server hub
    into a near-run-all closure. A parser test attaches by stem via its string
    literals (_leaf_edges) AND its path (_path_leaf_edges) -- integration tests
    exercise a parser at runtime with neither a static import nor a literal."""
    for engine in _parser_engine_modules(graph):
        stem = Path(engine).stem
        parse.parser_engine_entries[stem] = engine
        _claim(parse, engine)
        parse.edges_added += _leaf_edges(graph, {stem}, engine)
        parse.edges_added += _path_leaf_edges(graph, {stem}, engine)


def add_factory_edges(
    repo: Path, index: ModuleIndex, graph: ImportGraph
) -> FactoryParse:
    parse = FactoryParse()
    add_register_call_edges(repo, index, graph, parse)
    add_lazy_parser_table_edges(repo, index, graph, parse)
    add_parser_engine_edges(repo, index, graph, parse)
    add_attention_enum_edges(repo, index, graph, parse)
    add_module_attr_edges(repo, index, graph, parse)
    add_class_module_table_edges(repo, index, graph, parse)
    add_pkgutil_edges(repo, index, graph, parse)
    return parse
