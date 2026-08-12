# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Table-aware diffing: an entries-only edit to a parsed registry file scopes to
the changed entries instead of the file's engine-wide closure.

Soundness rests on two rules. Content comes from `git show <ref>:<path>` for both
sides, never the working tree (the CLI checkout sits at head, so reading the tree
as "base" yields an empty diff and a silently wrong tiny claim). And only dicts
the strict parser consumed entirely are stripped before comparing the file
remainders: anything unparsable stays in the remainder, forcing ast-level
inequality and a fallback to file-level treatment.

Accepted residual: a test reaching a changed entry only through a helper or
conftest DEFAULT model (no literal mention) is dropped relative to file-level;
helper accessors are literal-parameterized today (an audit test pins that), and
the nightly is the backstop.
"""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .curated import REGISTRY_FILE, TEST_REGISTRY_FILE

TABLE_FILES = (REGISTRY_FILE, TEST_REGISTRY_FILE)

GIT_SHOW_TIMEOUT_S = 15


@dataclass
class TableParse:
    # kind -> {entry key -> canonical value string}
    kinds: dict[str, dict[str, str]] = field(default_factory=dict)
    # arch -> module string (vllm registry) for module resolution
    modules: dict[str, str] = field(default_factory=dict)
    # arch -> HF id strings (tests registry) for string matching
    ids: dict[str, set[str]] = field(default_factory=dict)
    remainder_dump: str = ""


@dataclass
class EntryChange:
    kind: str
    key: str
    change: str  # added | removed | modified


@dataclass
class TableDiff:
    file: str
    texts_differ: bool
    changes: list[EntryChange] = field(default_factory=list)
    base: TableParse | None = None
    head: TableParse | None = None


def git_show(repo: Path, ref: str, path: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "show", f"{ref}:{path}"],
            capture_output=True,
            timeout=GIT_SHOW_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    try:
        return proc.stdout.decode()
    except UnicodeDecodeError:
        return None


def _module_level_dicts(tree: ast.Module) -> dict[str, ast.Dict]:
    return {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Dict)
    }


def _consume_str_tuple_dict(
    node: ast.Dict,
    dicts: dict[str, ast.Dict],
    consumed: dict[str, tuple[dict[str, tuple[str, str]], set[str]]],
    visiting: set[str],
) -> tuple[dict[str, tuple[str, str]], set[str]] | None:
    """Fully consume a str -> (str, str) dict, following Name spreads into
    other module-level dicts recursively. None = not fully consumable. Returns
    (entries, names); names is the set of dicts spread in transitively. The
    caller marks strippable ONLY names reachable from a spread that SUCCEEDED,
    so a name consumed under a sibling that later fails stays in the remainder
    (ast-guarded, an edit forces fallback) instead of being lost."""
    out: dict[str, tuple[str, str]] = {}
    names: set[str] = set()
    for key, value in zip(node.keys, node.values):
        if key is None:
            if not isinstance(value, ast.Name):
                return None
            name = value.id
            if name in visiting:
                return None
            cached = consumed.get(name)
            if cached is None:
                if name not in dicts:
                    return None
                cached = _consume_str_tuple_dict(
                    dicts[name], dicts, consumed, visiting | {name}
                )
                if cached is None:
                    return None
                consumed[name] = cached
            sub, subnames = cached
            out.update(sub)
            names.add(name)
            names |= subnames
            continue
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            return None
        if not (
            isinstance(value, ast.Tuple)
            and len(value.elts) == 2
            and all(
                isinstance(e, ast.Constant) and isinstance(e.value, str)
                for e in value.elts
            )
        ):
            return None
        out[key.value] = (value.elts[0].value, value.elts[1].value)
    return out, names


def _consume_str_str_dict(node: ast.Dict) -> dict[str, str] | None:
    out: dict[str, str] = {}
    for key, value in zip(node.keys, node.values):
        if not (
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            return None
        out[key.value] = value.value
    return out


def _remainder_dump(tree: ast.Module, stripped_names: set[str]) -> str:
    kept = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in stripped_names
        )
    ]
    return "\n".join(ast.dump(node) for node in kept)


def parse_vllm_registry_strict(text: str) -> TableParse | None:
    """Strict parse of the vllm model registry for diffing. Exact shape or
    None; the graph parser's lenient heuristics are deliberately absent."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    dicts = _module_level_dicts(tree)
    root = dicts.get("_VLLM_MODELS")
    if root is None:
        return None
    consumed: dict[str, tuple[dict[str, tuple[str, str]], set[str]]] = {}
    # Spreads into non-consumable dicts (e.g. _EMBEDDING_MODELS's DictComp)
    # are tolerated at the ROOT only if the name is a module-level dict Assign:
    # entries stay out of the merge, the Assign stays in the remainder (ast-guarded).
    merged: dict[str, tuple[str, str]] = {}
    strippable: set[str] = set()
    for key, value in zip(root.keys, root.values):
        if key is None:
            if not isinstance(value, ast.Name) or value.id not in dicts:
                return None
            name = value.id
            res = _consume_str_tuple_dict(dicts[name], dicts, consumed, {name})
            if res is not None:
                sub, subnames = res
                merged.update(sub)
                # Only a SUCCEEDING root spread makes its dicts strippable;
                # subnames are exactly the dicts whose entries reached `merged`.
                strippable.add(name)
                strippable |= subnames
            # else: left in remainder, entries not diffable (guarded)
        else:
            return None  # direct entries in _VLLM_MODELS: unexpected shape
    strippable.add("_VLLM_MODELS")

    parse = TableParse()
    parse.kinds["models"] = {k: f"{m}:{c}" for k, (m, c) in merged.items()}
    parse.modules = {k: m for k, (m, c) in merged.items()}
    for extra in ("_PREVIOUSLY_SUPPORTED_MODELS", "_OOT_SUPPORTED_MODELS"):
        node = dicts.get(extra)
        if node is None:
            continue
        entries = _consume_str_str_dict(node)
        if entries is None:
            continue  # unparsable shape: stays in remainder, guarded
        parse.kinds[extra.strip("_").lower()] = entries
        strippable.add(extra)
    parse.remainder_dump = _remainder_dump(tree, strippable)
    return parse


def parse_tests_registry_strict(text: str) -> TableParse | None:
    """Strict parse of tests/models/registry.py: every module-level dict
    whose values are all _HfExamplesInfo(...) calls becomes diffable, with
    the entry value being the ast.dump of the WHOLE call so kwarg-only
    changes (version pins, flags) register as modifications."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    dicts = _module_level_dicts(tree)
    parse = TableParse()
    entries: dict[str, str] = {}
    strippable: set[str] = set()
    for name, node in dicts.items():
        rows: dict[str, str] = {}
        ok = bool(node.keys)
        for key, value in zip(node.keys, node.values):
            if key is None and isinstance(value, ast.Name) and value.id in dicts:
                continue  # merge-of-sub-dicts spread; sub-dicts parsed on their own
            if not (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and isinstance(value, ast.Call)
                and getattr(value.func, "id", "") == "_HfExamplesInfo"
            ):
                ok = False
                break
            rows[key.value] = ast.dump(value)
            ids = {
                sub.value
                for sub in ast.walk(value)
                if isinstance(sub, ast.Constant)
                and isinstance(sub.value, str)
                and "/" in sub.value
            }
            parse.ids.setdefault(key.value, set()).update(ids)
        if ok and rows:
            # Accumulate, never overwrite: an arch can appear in two dicts
            # (LlavaNext/Phi3V/Qwen2VL are in both the embedding and the
            # multimodal table). Overwriting made an edit to the shadowed copy
            # diff as texts_differ with an empty change list, which routes to
            # registry importers only and drops every literal-only test.
            for key, dump in rows.items():
                entries[key] = f"{entries.get(key, '')}{name}:{dump}\n"
            strippable.add(name)
    if not entries:
        return None
    parse.kinds["hf_examples"] = entries
    parse.remainder_dump = _remainder_dump(tree, strippable)
    return parse


_PARSERS = {
    REGISTRY_FILE: parse_vllm_registry_strict,
    TEST_REGISTRY_FILE: parse_tests_registry_strict,
}


def diff_table(path: str, base_text: str, head_text: str) -> TableDiff | None:
    """None = fall back to file-level. A TableDiff with texts_differ and no
    entry changes is still meaningful (sub-dict reshuffles: the all-arch
    tests must run)."""
    parser = _PARSERS.get(path)
    if parser is None:
        return None
    base = parser(base_text)
    head = parser(head_text)
    if base is None or head is None:
        return None
    if base.remainder_dump != head.remainder_dump:
        return None  # non-entry code changed: file-level is the honest claim
    diff = TableDiff(
        file=path, texts_differ=base_text != head_text, base=base, head=head
    )
    kinds = set(base.kinds) | set(head.kinds)
    for kind in kinds:
        b = base.kinds.get(kind, {})
        h = head.kinds.get(kind, {})
        for key in b.keys() - h.keys():
            diff.changes.append(EntryChange(kind, key, "removed"))
        for key in h.keys() - b.keys():
            diff.changes.append(EntryChange(kind, key, "added"))
        for key in b.keys() & h.keys():
            if b[key] != h[key]:
                diff.changes.append(EntryChange(kind, key, "modified"))
    return diff
