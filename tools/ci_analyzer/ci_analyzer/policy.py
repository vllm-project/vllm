# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Policy predicates and the Claim type: world files, the docs-only /
no-code predicates, run_all pattern matching. Rule ORDER lives in
select.py's module docstring (the single home; the chain physically runs
there)."""

from __future__ import annotations

from dataclasses import dataclass, field

import regex as re

from .curated import CATCH_ALL_DEP_PREFIXES, EXTRA_WORLD_FILES
from .jobs.model import PipelineConfig

# Every rule name a Claim may carry. Pinned as a set rather than described in a
# comment because pass 2 routes on these: it may only narrow a step whose every
# claim is function-attributable, so a rename that silently stopped matching
# would change what gets dropped.
RULES = frozenset(
    {
        "world",
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

# Emitted alongside selected steps, never as a Claim.rule.
SYNTHETIC_RULES = frozenset({"preflight", "run-all", "always-run"})

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
    # device prefix a device-named data file targets (h200, mi, intel_gpu): the
    # test_files routing is scoped to it, since the file is unreadable on any
    # other known device (package-data claims only).
    device_scope: str | None = None
    # pipelines inside run_all that the repo's own config would NOT have run:
    # breadth we chose, not breadth CI asked for. A subset of run_all, and a
    # separate field rather than a rule name because one claim can match on
    # some pipelines and diverge on others.
    divergent: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"unpinned claim rule {self.rule!r}; add it to RULES")


def docs_only(paths: list[str]) -> bool:
    """The ci-infra generator's is_docs_only_change, verbatim.

    A frozen copy with no runtime tether, and a whole-diff zero-job answer rides
    on it: update when ci-infra changes is_docs_only_change. The crosscheck
    replica shares this predicate, so analyzer-vs-replica comparison is blind to
    its drift; the docs_only_but_ran red flag is the only detector."""
    return bool(paths) and all(
        p.startswith("docs/") or p.endswith(".md") or p == "mkdocs.yaml" for p in paths
    )


def matches_run_all(config: PipelineConfig, path: str) -> bool:
    for pattern in config.run_all_patterns:
        if re.match(pattern, path) and not any(
            re.match(e, path) for e in config.run_all_exclude_patterns
        ):
            return True
    return False


def matches_source_dependency(dep: str, diff_file: str) -> bool:
    """The generator's source_file_dependencies match: exact path or a
    directory prefix at a `/` boundary (no globbing)."""
    normalized = dep.rstrip("/")
    if not normalized:
        return False
    return diff_file == normalized or diff_file.startswith(f"{normalized}/")


def is_catch_all_dep(dep: str) -> bool:
    """A declaration so broad it carries no information about this file: the
    step named a whole package root rather than what it uses.

    Single home. The selector subtracts these on graph-known files and
    crosscheck buckets its gaps by them, and a gate that suspends one copy
    while the other kept deciding would be worse than having no gate."""
    return dep.rstrip("/") in {p.rstrip("/") for p in CATCH_ALL_DEP_PREFIXES}


def split_deps(deps: list[str] | None) -> tuple[list[str], list[str]]:
    """(positive, negated). A `!`-prefixed entry carves a subtree out of a
    broader positive one (`vllm/` plus `!vllm/distributed/kv_transfer/`), so
    the decision is per STEP, never per entry: testing entries individually
    reads a negation as a dep that simply never matches."""
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
        path.startswith("docs/")
        or path.endswith(".md")
        or path == "mkdocs.yaml"
        or path.startswith(".github/")
        or path.startswith("LICENSE")
    )


def classify_world(
    path: str, configs: list[PipelineConfig], *, policy_world: bool = True
) -> Claim | None:
    """World if the file matches a run_all pattern or is an EXTRA_WORLD_FILES
    policy file. policy_world=False drops the policy half (the diff was confined
    to config-only tables, per worldtables); the real run_all match still holds."""
    hit = {c.name for c in configs if matches_run_all(c, path)}
    extra = (
        {c.name for c in configs}
        if (path in EXTRA_WORLD_FILES and policy_world)
        else set()
    )
    world = hit | extra
    if not world:
        return None
    divergent = extra - hit
    if divergent:
        # One detail string is stamped on every pipeline in `world`, so it must
        # name the match/divergence sets, not blanket-claim a run_all match that
        # is false for the divergent pipelines.
        detail = (
            f"{path} is a world file: run_all match on {sorted(hit)}, "
            f"analyzer-policy divergence on {sorted(divergent)}"
        )
    else:
        detail = f"{path} matches run_all_patterns"
    return Claim("world", detail, run_all=world, divergent=divergent)
