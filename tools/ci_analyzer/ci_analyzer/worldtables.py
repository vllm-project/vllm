# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Table-granular world-ness for structured config files.

For a changed pyproject.toml, decide world-ness from WHICH top-level TOML
tables changed. Build-defining tables ([build-system]/[project]) and tool
tables a Buildkite step consumes stay world; a change confined to config-only
tool tables (ruff/mypy -- tools no modeled step runs) defers to CI. Additive
only: the analyzer-policy world is dropped, never CI's own run_all match, so
selection never drops below CI. Every ambiguity fails safe to world.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import regex as re
import tomllib

from .curated import EXTRA_WORLD_FILES
from .policy import matches_run_all
from .tablediff import git_show

if TYPE_CHECKING:
    from .select import AnalyzerState

# PEP 517/621 tables that define the wheel/metadata. Spec constants, not vLLM
# facts: a change here can affect any build or test, so it stays world.
_BUILD_TABLES = frozenset({"build-system", "project"})


def _normalize(name: str) -> str:
    """PEP 503 name normalization (setuptools_scm == setuptools-scm)."""
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def changed_sections(old_text: str, new_text: str) -> set[str] | None:
    """Top-level TOML tables that differ, drilling one level under [tool] so
    tool.<name> tables are distinguished. None if either side is unparsable."""
    try:
        old = tomllib.loads(old_text)
        new = tomllib.loads(new_text)
    except tomllib.TOMLDecodeError:
        return None
    changed: set[str] = set()
    for key in set(old) | set(new):
        if key == "tool":
            old_tools, new_tools = old.get("tool", {}), new.get("tool", {})
            changed |= {
                f"tool.{name}"
                for name in set(old_tools) | set(new_tools)
                if old_tools.get(name) != new_tools.get(name)
            }
        elif old.get(key) != new.get(key):
            changed.add(key)
    return changed


def _build_tool_pkgs(new_text: str) -> set[str]:
    """Normalized package names used at build time (backend + requires): a
    tool.<pkg> table for one of these configures the build."""
    try:
        bs = tomllib.loads(new_text).get("build-system", {})
    except tomllib.TOMLDecodeError:
        return set()
    pkgs = {
        _normalize(re.split(r"[<>=!~,;\s\[]", req, maxsplit=1)[0])
        for req in bs.get("requires", [])
    }
    backend = bs.get("build-backend", "")
    if backend:
        pkgs.add(_normalize(backend.split(".")[0]))
    return pkgs - {""}


def _referenced_tools(state: AnalyzerState) -> set[str]:
    """Normalized tokens across every step's command+script text. A tool a step
    invokes (pytest) reads its own config table, so that table stays world; a
    tool absent from every step (ruff, run in GitHub Actions) does not. A stray
    textual match only over-selects.

    A dotted token also contributes its pieces. _normalize folds package names
    per PEP 503, so on a filename it turns `tools/mypy.sh` into `mypy-sh` and a
    tool named only that way would read as unreferenced -- the under-selection
    direction. Splitting keeps `mypy` in the set. Script bodies are already in
    the haystack, so this only matters when a wrapper never names its tool."""
    corpus = "\n".join(t.haystack for p in state.pipelines for t in p.targets.values())
    return {
        _normalize(part)
        for tok in re.findall(r"[A-Za-z0-9_.-]+", corpus)
        for part in (tok, *tok.split("."))
        if part
    }


def _config_only(
    sections: set[str], build_pkgs: set[str], referenced: set[str]
) -> bool:
    """True iff every changed section is a tool table that no Buildkite step
    consumes and that does not configure the build -- safe to narrow."""
    for sec in sections:
        if sec in _BUILD_TABLES or not sec.startswith("tool."):
            return False
        name = _normalize(sec[len("tool.") :])
        if name in build_pkgs or name in referenced:
            return False
    return True


def _pyproject_config_only(
    state: AnalyzerState, base: str, head: str, path: str
) -> bool:
    old = git_show(state.repo, base, path)
    new = git_show(state.repo, head, path)
    if old is None or new is None:
        return False
    sections = changed_sections(old, new)
    if not sections:
        return False
    return _config_only(sections, _build_tool_pkgs(new), _referenced_tools(state))


def narrowable_world_paths(
    state: AnalyzerState, base: str, head: str, changed: set[str]
) -> set[str]:
    """EXTRA_WORLD_FILES paths whose change is confined to config-only tables,
    so classify_world may drop the analyzer-policy world and defer to CI. Only
    when the file still has a real run_all hit to fall back to -- otherwise
    dropping it would fall through to the terminal fail-open (run-all)."""
    configs = [p.config for p in state.pipelines]
    return {
        path
        for path in EXTRA_WORLD_FILES
        if path == "pyproject.toml"
        and path in changed
        and any(matches_run_all(c, path) for c in configs)
        and _pyproject_config_only(state, base, head, path)
    }
