# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-registry wall parser.

The registry maps arch names to (module, Class) pairs; edges are string-keyed,
test-side, and leaf-only. A test depends on a model module when it literally
mentions the arch name or an HF example id; the parametrize-over-all-archs tests
depend on every model module. Edges never attach to non-test files: two measured
amplifiers otherwise blow every model's closure up to the full suite. Accepted
gap: a test reaching a model only through a helper or conftest default is not
edged (fixture granularity, deferred; nightly is the backstop).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from ..curated import (
    MODEL_MODULE_PREFIX as MODULE_PREFIX,
)
from ..curated import (
    QUANT_INIT_FILE,
    REGISTRY_FILE,
    TEST_REGISTRY_FILE,
)
from ..repo import ModuleIndex, is_test_basename
from .imports import ImportGraph


@dataclass
class RegistryParse:
    entries: dict[str, tuple[str, str]] = field(default_factory=dict)
    hf_ids: dict[str, set[str]] = field(default_factory=dict)  # arch -> ids
    unresolved: list[str] = field(default_factory=list)  # modules not in index
    edges_added: int = 0


def parse_model_registry(repo: Path) -> RegistryParse:
    result = RegistryParse()
    path = repo / REGISTRY_FILE
    tree = ast.parse(path.read_text(), filename=REGISTRY_FILE)
    dicts: dict[str, dict[str, tuple[str, str]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(node.value, ast.Dict):
            continue
        entries = _literal_entries(node.value, dicts)
        if entries:
            dicts[target.id] = entries
    merged = dicts.get("_VLLM_MODELS")
    if not merged:
        # _VLLM_MODELS is built from spreads of the named dicts; if the exact
        # name vanished, fall back to the union of all model-shaped dicts.
        merged = {}
        for name, entries in dicts.items():
            if name.startswith("_") and name.endswith("_MODELS"):
                merged.update(entries)
    result.entries = merged
    return result


def _literal_entries(
    node: ast.Dict, known: dict[str, dict[str, tuple[str, str]]]
) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for key, value in zip(node.keys, node.values):
        if key is None:
            # `**_OTHER_DICT` spread
            if isinstance(value, ast.Name) and value.id in known:
                out.update(known[value.id])
            continue
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            continue
        if (
            isinstance(value, ast.Tuple)
            and len(value.elts) == 2
            and all(
                isinstance(e, ast.Constant) and isinstance(e.value, str)
                for e in value.elts
            )
        ):
            out[key.value] = (value.elts[0].value, value.elts[1].value)
    return out


def resolve_module_name(mod: str) -> str:
    if mod.startswith("vllm."):
        return mod
    return f"{MODULE_PREFIX}.{mod}"


def parse_hf_example_ids(repo: Path) -> dict[str, set[str]]:
    """arch -> HF model id strings, from tests/models/registry.py's
    _HfExamplesInfo("org/id", ...) calls (first arg + literal extras)."""
    path = repo / TEST_REGISTRY_FILE
    tree = ast.parse(path.read_text(), filename=TEST_REGISTRY_FILE)
    out: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and isinstance(value, ast.Call)
                and getattr(value.func, "id", "") == "_HfExamplesInfo"
            ):
                continue
            ids = out.setdefault(key.value, set())
            for sub in ast.walk(value):
                if (
                    isinstance(sub, ast.Constant)
                    and isinstance(sub.value, str)
                    and "/" in sub.value
                    # A leading / is a local filesystem path, not an HF id:
                    # no CI text can name it, and it pollutes key displays.
                    and not sub.value.startswith("/")
                ):
                    ids.add(sub.value)
    return out


def string_keyed_claims(parse: RegistryParse, index: ModuleIndex) -> set[str]:
    """Files (and package dirs, as 'prefix/') whose coverage is routed by
    string-keyed leaf edges, so lazy import edges into them may be dropped."""
    claims: set[str] = set()
    for _arch, (mod, _cls) in parse.entries.items():
        resolved = index.resolve(resolve_module_name(mod))
        if resolved is None:
            continue
        claims.add(resolved)
        if resolved.endswith("/__init__.py"):
            claims.add(resolved[: -len("__init__.py")])
    return claims


@dataclass
class QuantParse:
    methods: dict[str, str] = field(default_factory=dict)  # method -> file
    edges_added: int = 0
    claims: set[str] = field(default_factory=set)


def add_quant_method_edges(
    repo: Path, index: ModuleIndex, graph: ImportGraph
) -> QuantParse:
    """String-key the quantization method table (method_to_config).

    Maps method names to Config classes lazily imported inside the accessor;
    modeled like the model registry: leaf test files mentioning the method
    name or config class name get the edge, and the lazily imported targets
    become claims so finalize_lazy_edges drops the near-run-all import route
    through quantization/__init__.py.
    """
    result = QuantParse()
    path = repo / QUANT_INIT_FILE
    try:
        tree = ast.parse(path.read_text(), filename=QUANT_INIT_FILE)
    except (SyntaxError, UnicodeDecodeError, OSError):
        graph.parse_errors.append(QUANT_INIT_FILE)
        return result
    # __init__.py: relative imports resolve against the package itself.
    package = QUANT_INIT_FILE[: -len("/__init__.py")].replace("/", ".")
    class_to_module: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - (node.level - 1)])
                module = f"{base}.{node.module}"
            else:
                module = node.module
            for alias in node.names:
                class_to_module[alias.asname or alias.name] = module
    method_to_class: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        pairs = {
            k.value: v.id
            for k, v in zip(node.keys, node.values)
            if isinstance(k, ast.Constant)
            and isinstance(k.value, str)
            and isinstance(v, ast.Name)
        }
        if len(pairs) > len(method_to_class):
            method_to_class = pairs
    for method, cls in method_to_class.items():
        module = class_to_module.get(cls)
        resolved = index.resolve(module) if module else None
        if resolved is None:
            target = index.resolve(f"{cls}")  # pragma: no cover - defensive
            if target is None:
                continue
            resolved = target
        result.methods[method] = resolved
        result.claims.add(resolved)
        if resolved.endswith("/__init__.py"):
            result.claims.add(resolved[: -len("__init__.py")])
        keys = {method, cls}
        for test_file, literals in graph.string_literals.items():
            if is_test_basename(test_file) and not keys.isdisjoint(literals):
                graph.add_edge(test_file, resolved)
                result.edges_added += 1
    return result


def add_registry_edges(
    repo: Path, index: ModuleIndex, graph: ImportGraph
) -> RegistryParse:
    try:
        parse = parse_model_registry(repo)
    except (SyntaxError, UnicodeDecodeError, OSError):
        graph.parse_errors.append(REGISTRY_FILE)
        parse = RegistryParse()
    try:
        parse.hf_ids = parse_hf_example_ids(repo)
    except (SyntaxError, UnicodeDecodeError, OSError):
        graph.parse_errors.append(TEST_REGISTRY_FILE)
    all_arch_tests = {
        f for f in graph.reverse.get(TEST_REGISTRY_FILE, ()) if is_test_basename(f)
    }
    test_literals = {
        f: lits for f, lits in graph.string_literals.items() if is_test_basename(f)
    }
    for arch, (mod, _cls) in parse.entries.items():
        module = resolve_module_name(mod)
        resolved = index.resolve(module)
        if resolved is None:
            parse.unresolved.append(f"{arch} -> {module}")
            continue
        for test_file in all_arch_tests:
            graph.add_edge(test_file, resolved)
            parse.edges_added += 1
        keys = {arch} | parse.hf_ids.get(arch, set())
        for test_file, literals in test_literals.items():
            if not keys.isdisjoint(literals):
                graph.add_edge(test_file, resolved)
                parse.edges_added += 1
    return parse
