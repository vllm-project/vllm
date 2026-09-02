# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Steps a path reaches by a route the import graph cannot see.

Three of them: a step naming the file outright, a step declaring it as a source
dependency, and the hardware naming convention. Shared, because both the graph
rule and the co-location rule need all three and neither owns them.
"""

from __future__ import annotations

import os

from . import hardware
from .claim import step_declares
from .state import RepoState

ENV_VAR = "CI_SELECTOR_DECLARED_DEPS"
MODES = ("on", "off")


def mode() -> str:
    """Whether the hand-written `source_file_dependencies` lists pick steps.
    Unset means "off": everything is derived from the code. "on" adds the
    declared steps back on top. An unrecognized value raises rather than
    defaulting, since a swallowed typo looks exactly like the switch doing
    nothing."""
    raw = os.environ.get(ENV_VAR)
    if raw is None or raw == "":
        return "off"
    if raw in MODES:
        return raw
    raise ValueError(f"{ENV_VAR}={raw!r}, expected one of: {', '.join(MODES)}")


def _source_dep_steps_ungated(
    state: RepoState, path: str, specific_only: bool = False
) -> set[str]:
    """Steps that declare `path` in their source_file_dependencies. For a file
    the import graph cannot reach, the declaration is the ground truth, since
    it is the generator's own mechanism.

    With specific_only, a step counts only when a dep more specific than a
    catch-all prefix matches. On a file the graph knows, the graph is the
    better answer and a blanket `vllm/` adds only the CI config's
    over-declaration, which would cap the saving at zero. Graph-blind files
    always take the full union, since the declaration is all they have.

    Ignores the switch, for three callers: the requirements rule, which picks
    steps from declarations by design, and the two places where a declaration
    only ever says "this file is still tested". Silencing those would make the
    switch invent empty answers. Everything that picks steps by declaration
    goes through `_source_dep_steps` instead."""
    return {
        s.step_id
        for p in state.pipelines
        for s in p.steps
        if step_declares(s.source_file_dependencies, path, specific_only)
    }


def _source_dep_steps(
    state: RepoState, path: str, specific_only: bool = False
) -> set[str]:
    """The one place the switch acts: every read that picks steps from the
    declared lists comes through here."""
    if mode() == "off":
        return set()
    return _source_dep_steps_ungated(state, path, specific_only)


def _direct_step_refs(state: RepoState, path: str) -> set[str]:
    """Steps naming the file itself: as a target, a scanned script, or a data
    file."""
    return {
        sid
        for p in state.pipelines
        for sid, st in p.targets.items()
        if path in st.data_files
        or path in st.scripts_seen
        or any(t.path == path for t in st.targets)
    }


def _hardware_family_steps(state: RepoState, path: str) -> tuple[str | None, set[str]]:
    """Steps a source file reaches by hardware naming convention.

    For a source file whose compiled kernels reach a family's jobs invisibly.
    A test, benchmark or example has no such reach, since nothing under vllm/
    imports them, so their steps are exactly their own coverage.
    """
    family = hardware.family_of_path(path)
    if not family or path.startswith(("tests/", "benchmarks/", "examples/")):
        return None, set()
    return family, state.family_steps(family)
