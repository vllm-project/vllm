# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Policy predicates and the Claim type. Rule ORDER lives in classify.py's
module docstring, where the chain actually runs.

The generator's `run_all_patterns` is deliberately not read here. Every breadth
this module produces is the analyzer's own. The one replica of the hand-written
predicate lives in `validate/generator_replica.py`, which models what CI does
today; importing it back would make that comparison measure nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..handwritten import (
    CATCH_ALL_DEP_PREFIXES,
    DOCS_ONLY_EXACT,
    DOCS_ONLY_PREFIXES,
    DOCS_ONLY_SUFFIXES,
)
from .pipeline.step import PipelineConfig

# Files carrying no code, so no test can be affected. Not the same table as
# DOCS_ONLY_*: that one is ci-infra's, this one is ours, and merging them would
# quietly adopt their policy as our own.
NO_CODE_PREFIXES = ("docs/", ".github/", "LICENSE")
NO_CODE_SUFFIXES = (".md",)
NO_CODE_EXACT = ("mkdocs.yaml",)
EXTRA_WORLD_FILES = ("pyproject.toml",)

# Every rule name a Claim may carry. A set and not a comment because the record
# routes on these, so a rename that quietly stopped matching would change what
# it drops.
RULES = frozenset(
    {
        "world",
        "image-input",
        "buildkite",
        "legacy-ci",
        "inert-ci",
        "release-ci",
        "no-hardware",
        "graph",
        "table-diff",
        "no-code",
        "added-conftest",
        "added-in-claimed-package",
        "added-trivial-init",
        "added-test",
        "added-benchmark",
        "added-head-closure",
        "renamed",
        "requirements",
        "target-coverage",
        "package-data",
        "declared-deps",
        "fail-open",
    }
)

# Emitted alongside selected steps, never as a Claim.rule. "coverage" is written
# by the CLI for steps the record added, which skip `_record`.
SYNTHETIC_RULES = frozenset({"preflight", "run-all", "always-run", "coverage"})

OUTPUT_RULES = RULES | SYNTHETIC_RULES


@dataclass
class Claim:
    rule: str
    detail: str
    # pipeline names -> run everything there
    run_all: set[str] = field(default_factory=set)
    step_ids: set[str] = field(default_factory=set)
    # test files implicated, mapped to steps by the caller
    test_files: set[str] = field(default_factory=set)
    # device prefix a device-named data file targets. Scopes the test_files
    # routing, since no other known device can read the file.
    device_scope: str | None = None
    # Which routings the record may overturn with function evidence. Per
    # mechanism, not per rule: one rule reaches a step several ways and they do
    # not all say the step runs the path. Both default to not droppable.
    droppable_step_ids: set[str] = field(default_factory=set)
    droppable_test_files: bool = False

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"unpinned claim rule {self.rule!r}; add it to RULES")
        stray = self.droppable_step_ids - self.step_ids
        if stray:
            raise ValueError(
                f"{self.rule}: droppable_step_ids must be a subset of step_ids; "
                f"stray {sorted(stray)[:3]}"
            )


def docs_only(paths: list[str]) -> bool:
    """A frozen copy of the generator's is_docs_only_change, with nothing tying
    it back to the original. A whole-diff zero-job answer rides on it, and the
    replica shares the same predicate, so only the docs_only_but_ran flag
    catches drift."""
    return bool(paths) and all(
        p.startswith(DOCS_ONLY_PREFIXES)
        or p.endswith(DOCS_ONLY_SUFFIXES)
        or p in DOCS_ONLY_EXACT
        for p in paths
    )


def matches_source_dependency(dep: str, diff_file: str) -> bool:
    """The generator's source_file_dependencies match: exact path or a
    directory prefix at a `/` boundary (no globbing)."""
    normalized = dep.rstrip("/")
    if not normalized:
        return False
    return diff_file == normalized or diff_file.startswith(f"{normalized}/")


def is_catch_all_dep(dep: str) -> bool:
    """A declaration so broad it says nothing about this file: the step named a
    whole package root rather than what it uses. One home, because the selector
    and crosscheck both read it and must not disagree."""
    return dep.rstrip("/") in {p.rstrip("/") for p in CATCH_ALL_DEP_PREFIXES}


def split_deps(deps: list[str] | None) -> tuple[list[str], list[str]]:
    """(positive, negated). A `!` entry carves a subtree out of a broader
    positive one, so the decision is per STEP and never per entry: testing
    entries one by one reads a negation as a dep that never matches."""
    positive, negated = [], []
    for dep in deps or ():
        (negated if dep.startswith("!") else positive).append(dep.lstrip("!"))
    return positive, negated


def step_declares(
    deps: list[str] | None, path: str, specific_only: bool = False
) -> bool:
    """True when a step's source_file_dependencies fire for `path`."""
    positive, negated = split_deps(deps)
    if not positive:
        return False
    if any(matches_source_dependency(d, path) for d in negated):
        return False
    if specific_only:
        positive = [d for d in positive if not is_catch_all_dep(d)]
    return any(matches_source_dependency(d, path) for d in positive)


def deps_match(deps: list[str] | None, paths: list[str]) -> bool:
    return any(step_declares(deps, path) for path in paths)


def is_no_code(path: str) -> bool:
    return (
        path.startswith(NO_CODE_PREFIXES)
        or path.endswith(NO_CODE_SUFFIXES)
        or path in NO_CODE_EXACT
    )


def classify_world(path: str, configs: list[PipelineConfig]) -> Claim | None:
    """Run everything, when the file is one of ours to escalate on.

    Analyzer policy and not CI's: a file listed here escalates every pipeline
    because we say so. That is why the escalation is always the full config
    set, with no per-pipeline pattern left to make it partial.

    A no-op at HEAD, checked rather than assumed. Kept for its ORDER, so a
    member some narrower rule would claim still escalates here.
    """
    if path not in EXTRA_WORLD_FILES:
        return None
    return Claim(
        "world",
        f"{path} is an analyzer-policy world file; running everything",
        run_all={c.name for c in configs},
    )
