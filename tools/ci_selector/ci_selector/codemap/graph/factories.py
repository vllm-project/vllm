# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Edge parsers for the tables that register things by name.

Registration calls, the lazy parser tables, backend enums, vllm/__init__'s lazy
imports and pkgutil package scans all end up the same way: an edge from any test
naming the key, and a claim on the target so its lazy import route can be cut.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import regex as re

from ..repo import ModuleIndex, is_test_basename
from .imports import ImportGraph

REASONING_INIT = "vllm/reasoning/__init__.py"
TOOL_PARSER_INIT = "vllm/tool_parsers/__init__.py"
VLLM_INIT = "vllm/__init__.py"
RENDERERS_REGISTRY = "vllm/renderers/registry.py"
TOKENIZERS_REGISTRY = "vllm/tokenizers/registry.py"
PARSER_ADAPTERS_HUB = "vllm/parser/engine/registered_adapters.py"
PARSER_MANAGER_HUB = "vllm/parser/parser_manager.py"
PARSER_ENGINE_DIR = "vllm/parser/"
LAZY_ALIAS_DICT = "MODULE_ATTRS"  # vllm/__init__.py
LAZY_TABLE_SUFFIX = "_TO_REGISTER"
LAZY_TABLE_NAMES = ("_VLLM_RENDERERS", "_VLLM_TOKENIZERS")
PKGUTIL_ENUMERATORS = ("iter_modules", "walk_packages")
REGISTER_CALL_PREFIX = "register_"

RENDERERS_MODULE_PREFIX = "vllm.renderers"
TOKENIZERS_MODULE_PREFIX = "vllm.tokenizers"


@dataclass
class FactoryParse:
    register_entries: dict[str, str] = field(default_factory=dict)  # key->file
    parser_entries: dict[str, str] = field(default_factory=dict)
    # Every file a colliding name registers in (kimi_k3 is a tool parser AND a
    # tokenizer mode). parser_entries keeps one winner; key routing needs all.
    parser_entry_files: dict[str, set[str]] = field(default_factory=dict)
    # Per table, because names collide across them (deepseek_v3 is both a
    # reasoning and a tool parser) and the merged dict is last-wins. A dead
    # table would leave that size unchanged, so preflight guards these counts.
    parser_table_counts: dict[str, int] = field(default_factory=dict)
    enum_entries: dict[str, str] = field(default_factory=dict)
    # qualname-enum file -> entries parsed from it (see enum_entries above)
    enum_table_counts: dict[str, int] = field(default_factory=dict)
    module_attrs: dict[str, str] = field(default_factory=dict)
    pkgutil_dirs: list[str] = field(default_factory=list)
    # lazy-export key -> module file (see add_lazy_export_table_edges)
    class_table_entries: dict[str, str] = field(default_factory=dict)
    # lazy-table file -> entries parsed out of it
    lazy_table_counts: dict[str, int] = field(default_factory=dict)
    # package -> names read out of its Literal
    backend_literal_counts: dict[str, int] = field(default_factory=dict)
    # unified parser-engine module stem -> vllm/parser/<stem>.py
    parser_engine_entries: dict[str, str] = field(default_factory=dict)
    # target file -> the table files that name it, so a member with no coverage
    # of its own can inherit its registry's. Parser-engine entries excluded.
    table_of: dict[str, set[str]] = field(default_factory=dict)
    claims: set[str] = field(default_factory=set)
    edges_added: int = 0
    module_attr_resolved: int = 0


def _record_parse_error(graph: ImportGraph, file: str) -> None:
    if file not in graph.parse_errors:
        graph.parse_errors.append(file)


def _leaf_consumer(file: str) -> bool:
    """Files that USE a keyed target rather than register it. Tests, plus the
    example and benchmark scripts steps run directly."""
    return is_test_basename(file) or file.startswith(("examples/", "benchmarks/"))


def _leaf_edges(graph: ImportGraph, keys: set[str], target: str) -> int:
    added = 0
    for leaf_file, literals in graph.string_literals.items():
        if _leaf_consumer(leaf_file) and not keys.isdisjoint(literals):
            graph.add_edge(leaf_file, target)
            added += 1
    return added


def _path_leaf_edges(graph: ImportGraph, tokens: set[str], target: str) -> int:
    """Like _leaf_edges but matches the file's PATH, not its literals. An
    integration test can exercise a parser with neither an import nor the exact
    name in its source."""
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
        if REGISTER_CALL_PREFIX not in text:
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
            if not name.startswith(REGISTER_CALL_PREFIX):
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
            graph.table_files.add(file)
            parse.register_entries[key] = target
            if target != file:
                parse.table_of.setdefault(target, set()).add(file)
            _claim(parse, target)
            parse.edges_added += _leaf_edges(graph, {key}, target)


def add_lazy_parser_table_edges(
    repo: Path, index: ModuleIndex, graph: ImportGraph, parse: FactoryParse
) -> None:
    for init_file, prefix in (
        (REASONING_INIT, "vllm.reasoning"),
        (TOOL_PARSER_INIT, "vllm.tool_parsers"),
        # Same shape as the parser tables, and the dynamic-import audit
        # relies on them being parsed.
        (RENDERERS_REGISTRY, RENDERERS_MODULE_PREFIX),
        (TOKENIZERS_REGISTRY, TOKENIZERS_MODULE_PREFIX),
    ):
        # Zeroed before opening, so a moved table reads as empty rather than
        # as a missing row.
        parse.parser_table_counts[init_file] = 0
        path = repo / init_file
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(), filename=init_file)
        except (SyntaxError, UnicodeDecodeError, OSError):
            _record_parse_error(graph, init_file)
            continue
        graph.table_files.add(init_file)
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and (
                    node.targets[0].id.endswith(LAZY_TABLE_SUFFIX)
                    or node.targets[0].id in LAZY_TABLE_NAMES
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
                parse.parser_entry_files.setdefault(key, set()).add(target)
                parse.parser_table_counts[init_file] += 1
                if target != init_file:
                    parse.table_of.setdefault(target, set()).add(init_file)
                _claim(parse, target)
                parse.edges_added += _leaf_edges(graph, set(consts), target)


def _enum_bases(node: ast.ClassDef) -> bool:
    for base in node.bases:
        name = (
            base.id
            if isinstance(base, ast.Name)
            else base.attr
            if isinstance(base, ast.Attribute)
            else ""
        )
        if name.endswith("Enum"):
            return True
    return False


def add_qualname_enum_edges(
    repo: Path, index: ModuleIndex, graph: ImportGraph, parse: FactoryParse
) -> None:
    """Backend registries written as an Enum whose members are class qualnames.

    Found by shape rather than by path, so a new registry is read the day it
    lands instead of going unnoticed. Sorted walk because member names collide
    across registries and enum_entries is last-wins: the edges below are made
    per member either way, but which file the flat map reports must not depend
    on directory order.
    """
    for file in sorted(index.file_to_module):
        if not file.startswith("vllm/"):
            continue
        try:
            text = (repo / file).read_text()
        except (UnicodeDecodeError, OSError):
            _record_parse_error(graph, file)
            continue
        if "Enum" not in text:
            continue
        try:
            tree = ast.parse(text, filename=file)
        except SyntaxError:
            _record_parse_error(graph, file)
            continue
        found = 0
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ClassDef) and _enum_bases(node)):
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
                    index.resolve(qualname.rsplit(".", 1)[0])
                    if "." in qualname
                    else None
                )
                if target is None:
                    continue
                found += 1
                parse.enum_entries[member] = target
                if target != file:
                    parse.table_of.setdefault(target, set()).add(file)
                _claim(parse, target)
                parse.edges_added += _leaf_edges(graph, {member}, target)
        # Only a file that yielded entries counts as read, and per file, since
        # a dead second registry leaves the merged size unchanged.
        if found:
            graph.table_files.add(file)
            parse.enum_table_counts[file] = found


def add_module_attr_edges(
    repo: Path, index: ModuleIndex, graph: ImportGraph, parse: FactoryParse
) -> None:
    path = repo / VLLM_INIT
    try:
        tree = ast.parse(path.read_text(), filename=VLLM_INIT)
    except (SyntaxError, UnicodeDecodeError, OSError):
        _record_parse_error(graph, VLLM_INIT)
        return
    graph.table_files.add(VLLM_INIT)
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == LAZY_ALIAS_DICT
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
                # Values are "module[:attr]"; keep the module part.
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


def _lazy_export_table(tree: ast.Module) -> dict[str, str]:
    """The lazy-export shape: a module-level `__getattr__` plus a dict of
    strings. Empty when the file is not one of these."""
    if not any(
        isinstance(n, ast.FunctionDef) and n.name == "__getattr__" for n in tree.body
    ):
        return {}
    table: dict[str, str] = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1:
            target, value = n.targets[0], n.value
        elif isinstance(n, ast.AnnAssign) and n.value is not None:
            target, value = n.target, n.value
        else:
            continue
        if not (isinstance(target, ast.Name) and isinstance(value, ast.Dict)):
            continue
        pairs = {
            k.value: v.value
            for k, v in zip(value.keys, value.values)
            if isinstance(k, ast.Constant)
            and isinstance(k.value, str)
            and isinstance(v, ast.Constant)
            and isinstance(v.value, str)
        }
        # All strings, or it is some other table that happens to share the
        # file with a __getattr__.
        if pairs and len(pairs) == len(value.keys):
            table.update(pairs)
    return table


def _imports_relative_to_self(tree: ast.Module) -> bool:
    """True when the file imports the relative form
    `import_module(f".{x}", __name__)`, which can only reach a sibling. That is
    what makes resolving a bare stem inside its own package safe."""
    return any(
        isinstance(node, ast.Call)
        and len(node.args) == 2
        and isinstance(node.args[1], ast.Name)
        and node.args[1].id == "__name__"
        for node in ast.walk(tree)
    )


def _lazy_target(index: ModuleIndex, value: str, package: str) -> str | None:
    """A table value to the file it imports. `pkg.mod` and `pkg.mod:attr` name
    the module outright; a bare stem is relative and resolves inside `package`,
    which is empty when the file has no relative import."""
    module = value.split(":", 1)[0]
    if "." in module:
        return index.resolve(module)
    return index.resolve(f"{package}.{module}") if package else None


def add_lazy_export_table_edges(
    repo: Path, index: ModuleIndex, graph: ImportGraph, parse: FactoryParse
) -> None:
    """Lazy-export tables: `from X import Name` fires X's `__getattr__`, which
    imports a module the static graph never sees.

    Found by shape rather than by path. Deliberately NOT vllm/__init__.py: its
    `__getattr__` sits inside a conditional so this scan misses it, and
    add_module_attr_edges keeps it. Its keys are names like `LLM`, and
    leaf-edging every test whose literals contain that is not something to start
    doing by accident.

    Who picks a key is often an HF checkpoint we cannot see, so entries get
    claims and broad leaf edges rather than scoped routing. The overshoot is
    deliberate.
    """
    tables: dict[str, dict[str, str]] = {}
    for file in sorted(index.file_to_module):
        if not file.startswith("vllm/"):
            continue
        try:
            text = (repo / file).read_text()
        except (UnicodeDecodeError, OSError):
            _record_parse_error(graph, file)
            continue
        if "__getattr__" not in text:
            continue
        try:
            tree = ast.parse(text, filename=file)
        except SyntaxError:
            _record_parse_error(graph, file)
            continue
        table = _lazy_export_table(tree)
        if not table:
            continue
        # Reading the table counts as handled even when nothing resolves.
        # "Every value leaves the repo" is an answer, not a failure.
        graph.table_files.add(file)
        parse.lazy_table_counts[file] = len(table)
        module = index.file_to_module.get(file, "")
        package = module if file.endswith("/__init__.py") else module.rpartition(".")[0]
        if not _imports_relative_to_self(tree):
            package = ""
        resolved: dict[str, str] = {}
        for key, value in table.items():
            target = _lazy_target(index, value, package)
            if target is None:
                continue
            resolved[key] = target
            parse.class_table_entries[key] = target
            if target != file:
                parse.table_of.setdefault(target, set()).add(file)
            _claim(parse, target)
            parse.edges_added += _leaf_edges(graph, {key}, target)
        if resolved:
            tables[file] = resolved
    for file, base_file, alias in graph.from_import_aliases:
        target = tables.get(base_file, {}).get(alias)
        if target:
            graph.add_edge(file, target)
            parse.edges_added += 1


def _backend_literal_names(tree: ast.Module) -> list[str]:
    """String members of a module-level `Literal` alias. All-string only: a
    member we cannot read is a sibling we would miss."""
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            or isinstance(node, ast.AnnAssign)
            and node.value is not None
        ):
            value = node.value
        else:
            continue
        if not isinstance(value, ast.Subscript):
            continue
        base = value.value
        name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
        if name != "Literal":
            continue
        members = (
            value.slice.elts if isinstance(value.slice, ast.Tuple) else [value.slice]
        )
        if not members or not all(
            isinstance(m, ast.Constant) and isinstance(m.value, str) for m in members
        ):
            continue
        return [m.value for m in members]
    return []


def add_relative_literal_edges(
    repo: Path, index: ModuleIndex, graph: ImportGraph, parse: FactoryParse
) -> None:
    """A package that dispatches `import_module(f".{x}", __name__)` over a
    `Literal` naming its own submodules. The Literal is the table, so the set
    is derived rather than listed here."""
    for file in sorted(index.file_to_module):
        if not file.endswith("/__init__.py") or not file.startswith("vllm/"):
            continue
        try:
            text = (repo / file).read_text()
        except (UnicodeDecodeError, OSError):
            _record_parse_error(graph, file)
            continue
        if "Literal[" not in text or "import_module" not in text:
            continue
        try:
            tree = ast.parse(text, filename=file)
        except SyntaxError:
            _record_parse_error(graph, file)
            continue
        if not _imports_relative_to_self(tree):
            continue
        names = _backend_literal_names(tree)
        if not names:
            continue
        package = index.file_to_module.get(file, "")
        targets = [t for t in (_lazy_target(index, n, package) for n in names) if t]
        # Every member must land, or we bless a file we only half-read.
        if len(targets) != len(names):
            continue
        graph.table_files.add(file)
        parse.backend_literal_counts[file] = len(names)
        for target in targets:
            if target != file:
                parse.table_of.setdefault(target, set()).add(file)
                graph.add_edge(file, target)
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
        if not any(name in text for name in PKGUTIL_ENUMERATORS):
            continue
        pkg_dir = file[: -len("__init__.py")]
        parse.pkgutil_dirs.append(pkg_dir)
        graph.table_files.add(file)
        for sibling in index.file_to_module:
            if sibling.startswith(pkg_dir) and sibling != file:
                graph.add_edge(file, sibling)
                parse.edges_added += 1


# Both hubs are graph nodes we read edges from, not files we re-parse.


def _is_parser_leaf(dst: str) -> bool:
    return (
        dst.startswith(PARSER_ENGINE_DIR)
        and dst.count("/") == 2
        and not dst.endswith("/__init__.py")
    )


def _parser_engine_modules(graph: ImportGraph) -> set[str]:
    """The concrete parser engines: what registered_adapters imports at load
    time, plus what parser_manager reaches through a lazy import. The shared
    base modules those engines import are infra, not dispatch targets, so they
    are excluded and never claimed.

    Call this before build_full_graph finishes. It reads graph.lazy_imports,
    which finalize_lazy_edges clears; afterwards read parser_engine_entries.
    """
    primary = {
        dst
        for dst in graph.imports.get(PARSER_ADAPTERS_HUB, ())
        if _is_parser_leaf(dst)
    }
    # A shared base module is imported by another parser; a concrete engine is
    # imported by nothing, or only by __init__. Engines that reuse each other
    # still register through `primary`, so this only filters the candidates.
    shared = {
        d
        for src, dsts in graph.imports.items()
        if _is_parser_leaf(src)
        for d in dsts
        if _is_parser_leaf(d) and d != src
    }
    engines = set(primary)
    for src, dst in graph.lazy_imports:
        if src == PARSER_MANAGER_HUB and _is_parser_leaf(dst) and dst not in shared:
            engines.add(dst)
    return engines


def add_parser_engine_edges(
    repo: Path, index: ModuleIndex, graph: ImportGraph, parse: FactoryParse
) -> None:
    """The parser engine, keyed by module stem. Claiming each engine lets the
    lazy edge be dropped, which otherwise routes it through the api_server hub
    and pulls in nearly everything. Tests attach both by literal and by path,
    because an integration test may use a parser without naming it."""
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
    add_qualname_enum_edges(repo, index, graph, parse)
    add_module_attr_edges(repo, index, graph, parse)
    add_lazy_export_table_edges(repo, index, graph, parse)
    add_relative_literal_edges(repo, index, graph, parse)
    add_pkgutil_edges(repo, index, graph, parse)
    return parse
