# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Derive the docs build's file-dependency set from the docs sources.

The docs build runs on Read the Docs, not Buildkite, so it declares no
source_file_dependencies. We derive the set instead: a conservative floor
(api-autonav re-renders the whole non-excluded vllm/ tree under fail_on_warning,
plus docs/, examples/, and the build config) extended with the out-of-tree files
docs pages reference by snippet or link. The reference extractor is shared by the
selection signal and the docs-refs audit, and follows the docs autoref hook's
symbol-visibility rules.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import regex as re
import yaml

_INFRA_FILES = frozenset(
    {
        "mkdocs.yaml",
        ".readthedocs.yaml",
        "requirements/docs.in",
        "requirements/docs.txt",
        # generate_argparse.py renders the CLI reference from this file.
        "requirements/test/cuda.txt",
    }
)
_FLOOR_PREFIXES = ("docs/", "examples/")


# --- mkdocs.yaml (custom-tag tolerant) -------------------------------------


class _MkdocsLoader(yaml.SafeLoader):
    """SafeLoader that neutralizes mkdocs' custom tags instead of raising.

    Safety: the base is SafeLoader (never constructs arbitrary objects), and
    both multi-constructors *return None* without touching the node, so a
    `!!python/object/apply:...` tag resolves to None and is never executed.
    `safe_load` alone raises ConstructorError on these tags; we only read plain
    string lists (gen-files scripts, api-autonav excludes), so dropping the
    tagged values (`!ENV`, `!!python/name:` in theme/markdown_extensions) is
    lossless.
    """


_MkdocsLoader.add_multi_constructor("!", lambda *_: None)
_MkdocsLoader.add_multi_constructor("tag:yaml.org,2002:python/", lambda *_: None)


def load_mkdocs(repo: Path) -> dict | None:
    """Parsed mkdocs.yaml, or None on any read/parse failure (caller fails
    open to the coarse floor)."""
    try:
        data = yaml.load((repo / "mkdocs.yaml").read_text(), Loader=_MkdocsLoader)
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


def _plugin_config(data: dict, name: str) -> dict:
    for plugin in data.get("plugins") or []:
        if isinstance(plugin, dict) and name in plugin:
            return plugin[name] or {}
    return {}


def api_autonav_excludes(data: dict) -> list[str]:
    return [
        e
        for e in (_plugin_config(data, "api-autonav").get("exclude") or [])
        if isinstance(e, str)
    ]


def gen_files_scripts(data: dict) -> list[str]:
    return [
        s
        for s in (_plugin_config(data, "gen-files").get("scripts") or [])
        if isinstance(s, str)
    ]


def hooks(data: dict) -> list[str]:
    return [h for h in (data.get("hooks") or []) if isinstance(h, str)]


# --- reference extraction ---------------------------------------------------

# [text][target] reference-style link; target has no brackets.
_AUTOREF_RE = re.compile(r"\]\[([^\]\[]+)\]")
# [identifier][] collapsed reference: the identifier is in the FIRST bracket and
# the second is empty. autorefs resolves it exactly like [text][identifier].
_AUTOREF_COLLAPSED_RE = re.compile(r"\[([^\]\[]+)\]\[\]")
# --8<-- "path[:section]" snippet include (pymdownx.snippets).
_SNIPPET_RE = re.compile(r'--8<--\s*"([^"]+)"')
# ](target) link to a repo file; relative only (url_schemes.py rewrites every
# non-URL, non-anchor link, any file type not just .py). _rel_if_file then
# keeps only real files, so the broad match is filtered downstream.
_PYLINK_RE = re.compile(r"\]\((?!(?:https?|ftp)://|#)([^)\s#]+)")
# ::: module.path mkdocstrings directive (none today; guarded by a tripwire).
MKDOCSTRINGS_RE = re.compile(r"^:::\s")


@dataclass(frozen=True)
class Ref:
    kind: str  # "autoref" | "snippet" | "pylink"
    target: str
    md_file: str
    line: int


def _fence(stripped: str) -> str | None:
    for mark in ("```", "~~~"):
        if stripped.startswith(mark):
            return mark
    return None


def extract_refs(repo: Path) -> list[Ref]:
    """Every autoref/snippet/pylink reference in docs/**/*.md.

    Autoref and pylink matches inside fenced code blocks are skipped (inert
    markdown there); `--8<--` snippets are expanded even inside fences by
    pymdownx, so they are collected regardless.
    """
    refs: list[Ref] = []
    for md in sorted((repo / "docs").rglob("*.md")):
        rel = md.relative_to(repo).as_posix()
        try:
            lines = md.read_text().splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        in_fence: str | None = None
        for i, line in enumerate(lines, 1):
            mark = _fence(line.lstrip())
            for m in _SNIPPET_RE.finditer(line):
                refs.append(Ref("snippet", m.group(1), rel, i))
            if in_fence is not None:
                if mark is not None and line.lstrip().startswith(in_fence):
                    in_fence = None
                continue
            if mark is not None:
                in_fence = mark
                continue
            for m in _AUTOREF_RE.finditer(line):
                refs.append(Ref("autoref", m.group(1), rel, i))
            for m in _AUTOREF_COLLAPSED_RE.finditer(line):
                refs.append(Ref("autoref", m.group(1), rel, i))
            for m in _PYLINK_RE.finditer(line):
                refs.append(Ref("pylink", m.group(1), rel, i))
    return refs


# --- reference -> file resolution -------------------------------------------


def resolve_autoref(target: str, index) -> tuple[str | None, str | None]:
    """Longest importable prefix of a `vllm.*` dotted target -> (module file,
    first trailing symbol segment). The trailing segment is None when the
    target is itself a module. `vllm` always resolves, so the file is only
    None for a non-vllm target."""
    parts = target.split(".")
    for depth in range(len(parts), 0, -1):
        file = index.resolve(".".join(parts[:depth]))
        if file:
            trailing = parts[depth:]
            return file, (trailing[0] if trailing else None)
    return None, None


def is_literal_snippet(target: str) -> bool:
    """A `--8<--` target that names a real file (not a `gen:` marker or a
    `{placeholder}` from a template)."""
    return not target.startswith("gen:") and "{" not in target


def resolve_snippet_file(target: str, md_file: str, repo: Path) -> str | None:
    if not is_literal_snippet(target):
        return None
    path = target.split(":", 1)[0].strip()  # drop :section / :start:end
    if not path:
        return None
    if (repo / path).is_file():  # snippets are repo-root relative
        return path
    return _rel_if_file((repo / md_file).parent / path, repo)


def _rel_if_file(path: Path, repo: Path) -> str | None:
    try:
        resolved = path.resolve()
        rel = resolved.relative_to(repo.resolve())
    except (OSError, ValueError):
        return None
    return rel.as_posix() if resolved.is_file() else None


# --- symbol presence (Layer B) ----------------------------------------------


def _module_stmts(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Module-level statements, descending into module-level if/try (so
    TYPE_CHECKING and try/except-guarded imports count) but never into
    function or class bodies (those are inner scopes)."""
    for node in body:
        yield node
        if isinstance(node, ast.If):
            yield from _module_stmts(node.body)
            yield from _module_stmts(node.orelse)
        elif isinstance(node, ast.Try):
            yield from _module_stmts(node.body)
            for handler in node.handlers:
                yield from _module_stmts(handler.body)
            yield from _module_stmts(node.orelse)
            yield from _module_stmts(node.finalbody)


def _str_elements(value: ast.expr) -> set[str]:
    if isinstance(value, ast.List | ast.Tuple):
        return {
            e.value
            for e in value.elts
            if isinstance(e, ast.Constant) and isinstance(e.value, str)
        }
    return set()


def symbol_status(module_file: str, symbol: str, repo: Path) -> str:
    """PRESENT / BROKEN / UNCERTAIN for a top-level `symbol` in `module_file`.

    BROKEN only with confidence: the module parses, has no `__getattr__` or
    `import *`, and the name is defined, assigned, imported/aliased, or listed
    in `__all__` nowhere at module level. Anything ambiguous stays UNCERTAIN
    (silent) to keep the audit free of false alarms.
    """
    try:
        tree = ast.parse((repo / module_file).read_text())
    except (OSError, SyntaxError, UnicodeDecodeError):
        return "UNCERTAIN"
    names: set[str] = set()
    dunder_all: set[str] = set()
    has_getattr = has_star = False
    for node in _module_stmts(tree.body):
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
            has_getattr = has_getattr or node.name == "__getattr__"
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    names.add(tgt.id)
                    if tgt.id == "__all__":
                        dunder_all |= _str_elements(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    has_star = True
                else:
                    names.add(alias.asname or alias.name.split(".")[0])
    if symbol in names or symbol in dunder_all:
        return "PRESENT"
    if has_getattr or has_star:
        return "UNCERTAIN"
    return "BROKEN"


# --- api-autonav exclude matching -------------------------------------------


def _module_of(path: str) -> str:
    parts = path.split("/")
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    elif parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)


def _parse_excludes(
    data: dict | None,
) -> tuple[tuple[str, ...], tuple[re.Pattern, ...]]:
    """api-autonav excludes as (path prefixes, module-name regexes). Empty on
    a missing/degraded parse, so the vllm floor covers everything (fail-open).
    """
    if data is None:
        return (), ()
    prefixes: list[str] = []
    regexes: list[re.Pattern] = []
    for entry in api_autonav_excludes(data):
        if entry.startswith("re:"):
            try:
                regexes.append(re.compile(entry[3:]))
            except re.error:
                continue
        else:
            prefixes.append(entry.replace(".", "/"))
    return tuple(prefixes), tuple(regexes)


# --- the dependency set -----------------------------------------------------


@dataclass
class DocsDeps:
    infra_files: frozenset[str] = frozenset()
    floor_prefixes: tuple[str, ...] = ()
    vllm_exclude_prefixes: tuple[str, ...] = ()
    vllm_exclude_regexes: tuple[re.Pattern, ...] = ()
    # out-of-floor file -> a docs page that references it
    extension: dict[str, str] = field(default_factory=dict)
    refs: tuple[Ref, ...] = ()
    degraded: bool = False

    def _vllm_excluded(self, path: str) -> bool:
        for pre in self.vllm_exclude_prefixes:
            if path == pre or path.startswith(pre + "/"):
                return True
        if path.endswith(".py"):
            module = _module_of(path)
            return any(rx.match(module) for rx in self.vllm_exclude_regexes)
        return False

    def _reason(self, path: str) -> str | None:
        if path in self.infra_files:
            return f"floor: {path} (docs build config)"
        for pre in self.floor_prefixes:
            if path.startswith(pre):
                return f"floor: {path} ({pre.rstrip('/')} source)"
        if path.startswith("vllm/") and not self._vllm_excluded(path):
            return f"floor: {path} (api-autonav re-renders the vllm API reference)"
        md = self.extension.get(path)
        if md is not None:
            return f"precise: {path} (referenced by {md})"
        return None

    def docs_affected(self, paths: list[str]) -> tuple[bool, list[str]]:
        reasons = [r for r in (self._reason(p) for p in dict.fromkeys(paths)) if r]
        if self.degraded:
            reasons.append(
                "note: mkdocs.yaml did not parse; vllm floor covers "
                "all of vllm/ (fail-open)"
            )
        return bool(reasons), reasons


def build_docs_deps(repo: Path) -> DocsDeps:
    data = load_mkdocs(repo)
    excl_prefixes, excl_regexes = _parse_excludes(data)
    deps = DocsDeps(
        infra_files=_INFRA_FILES,
        floor_prefixes=_FLOOR_PREFIXES,
        vllm_exclude_prefixes=excl_prefixes,
        vllm_exclude_regexes=excl_regexes,
        refs=tuple(extract_refs(repo)),
        degraded=data is None,
    )
    for ref in deps.refs:
        if ref.kind == "snippet":
            file = resolve_snippet_file(ref.target, ref.md_file, repo)
            if file is not None and deps._reason(file) is None:
                deps.extension.setdefault(file, ref.md_file)
        elif ref.kind == "pylink":
            _add_pylink_dep(deps, ref, repo)
        # autoref targets are vllm/ -> already floored
    return deps


def _add_pylink_dep(deps: DocsDeps, ref: Ref, repo: Path) -> None:
    """A relative link to a specific out-of-floor FILE makes that file a precise
    dep: renaming/removing it breaks the build (url_schemes.py rewrites the link
    to a GitHub URL only while the target exists). Directory links break only on
    a rename of the directory itself, which flooring the whole subtree would
    massively over-cover, so they are left unmodeled (a documented boundary)."""
    try:
        resolved = ((repo / ref.md_file).parent / ref.target).resolve()
        rel = resolved.relative_to(repo.resolve()).as_posix()
    except (OSError, ValueError):
        return
    if resolved.is_file() and deps._reason(rel) is None:
        deps.extension.setdefault(rel, ref.md_file)
