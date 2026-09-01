# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing a source file by the tests that sit beside it.

Inside the big import cycle (`graph/cycles.py`) the import graph is no help:
every file in it reaches every other, so they all pick the same steps. We use
the tests next to the file instead, plus the tests that import it directly. The
same swap happens outside the cycle for a file that reaches nearly as much
(`_colocated_hub`). `classify.py` decides when to call this; this file decides
the answer.

It only works if vLLM keeps its tests laid out like its source, so
`vllm/lora/...` is tested by `tests/lora/...`. Nothing makes that true. About
half the tree follows it, and the rest is grouped by topic instead.

So the walk is careful. It climbs up the path until it finds a directory that
really holds tests, and it never falls back to plain `tests/`. Finding nothing
is fine, because the caller then uses the graph rule. Finding the WRONG
directory is the bad case: the file routes to tests that do not cover it, and
nobody notices until a test we skipped would have caught a bug. Adding the
direct importers is what makes that case rare.
"""

from __future__ import annotations

import os

from ..handwritten import PR_PIPELINE
from .claim import Claim
from .selection import _targets_cover
from .state import RepoState
from .step_refs import (
    _direct_step_refs,
    _hardware_family_steps,
    _source_dep_steps,
)

ENV_VAR = "CI_SELECTOR_COLOCATION"
SOURCE_ROOT = "vllm/"
TESTS_ROOT = "tests/"

# A test directory is named after its source subtree except where that would
# collide, and the collisions all end in one of these (`tests/utils_`,
# `tests/tokenizers_`). Derived rather than listed, so the next rename of the
# same kind needs no edit. Checked for false positives at every depth.
ALIAS_SUFFIXES = ("_", "_tests")

# The one collision no suffix rule produces.
TEST_DIR_ALIASES = {"compilation": "compile"}

# How broad the graph rule's own answer has to be before the hub arm takes
# over, counted in PR-pipeline auto steps. Below it the graph gives the
# narrower answer, so the graph keeps the file. Measured, but a wrong value
# only costs us some saving: the clamp in `_colocated_hub` is what stops the
# swap ever widening, and the plain-static floor is what keeps it safe.
MIN_GRAPH_STEPS = 90


MODES = ("on", "off", "cycle-only")


def mode() -> str:
    """Which arms of the rule are live. Unset means "on", the full rule.

    "cycle-only" keeps cycle members and disables the hub arm: the measurement
    arm and the escape hatch. An unrecognized value raises, since reading a typo
    as "off" would look exactly like the rule doing nothing.
    """
    raw = os.environ.get(ENV_VAR)
    if raw is None or raw == "":
        return "on"
    if raw in MODES:
        return raw
    raise ValueError(f"{ENV_VAR}={raw!r}, expected one of: {', '.join(MODES)}")


def colocated_tests(state: RepoState, path: str) -> tuple[frozenset[str], str | None]:
    """The tests sitting beside `path`, and the directory they came from.

    `vllm/A/B/c.py` takes `test_c.py` from anywhere under `tests/`, then the
    deepest of `tests/A/B/` and `tests/A/` that holds tests. Never `tests/`
    itself, which is a run-all wearing a directory's clothes.
    """
    if not path.startswith(SOURCE_ROOT) or not path.endswith(".py"):
        return frozenset(), None
    index = state.test_index()
    parts = path[len(SOURCE_ROOT) :].split("/")
    by_name = index.by_name.get(f"test_{parts[-1][: -len('.py')]}.py", frozenset())
    for depth in range(len(parts) - 1, 0, -1):
        for head in _spellings(parts[:depth]):
            directory = TESTS_ROOT + "/".join(head) + "/"
            if directory in index.dirs:
                beneath = {f for f in index.files if f.startswith(directory)}
                return frozenset(by_name | beneath), directory
    return by_name, None


def _spellings(head: list[str]) -> list[list[str]]:
    """The candidate `tests/` spellings of one source subtree, literal first.

    Only the first segment is aliased: collisions happen at the top level, where
    a `tests/` directory shares a namespace with `tests/*.py` and with installed
    packages. Deeper segments have no such pressure.
    """
    first, rest = head[0], head[1:]
    names = [first, TEST_DIR_ALIASES.get(first)]
    names += [first + suffix for suffix in ALIAS_SUFFIXES]
    return [[name, *rest] for name in names if name]


def implicated_tests(state: RepoState, path: str) -> tuple[frozenset[str], str | None]:
    """Co-located tests widened by the tests that import `path` directly.

    The `if colocated` guard is a safety property, not an optimization: a file
    with no co-located tests must return empty so the caller falls back to the
    graph claim. With it, this can only ADD test files, never remove one.
    """
    colocated, directory = colocated_tests(state, path)
    if not colocated:
        return frozenset(), None
    importers = state.direct_test_importers().get(path, frozenset())
    return colocated | importers, directory


def _colocated_claim(
    state: RepoState,
    path: str,
    tests: frozenset[str],
    directory: str | None,
    lead: str,
    own_key_steps: frozenset[str] = frozenset(),
) -> Claim | None:
    """The colocated claim both arms share; only the trigger and `lead` differ.

    None covers both ways co-location can decline: no co-located tests exist,
    or the ones that do reach no auto-run step. The second is the graph rule's
    own question, so it is left to the graph rule.
    """
    # Everything the graph rule reaches WITHOUT the closure is kept, since the
    # cycle never collapsed it; only the closure-derived half is replaced.
    # Declarers matter most: `source_file_dependencies` is vLLM's hand-written
    # source-to-test map, and it names coverage co-location cannot see.
    direct_steps = _direct_step_refs(state, path)
    dep_steps = _source_dep_steps(state, path, specific_only=True)
    family, hw_steps = _hardware_family_steps(state, path)
    inferred_steps = direct_steps | dep_steps | own_key_steps
    if (
        not (tests & state.invoked)
        and not ((inferred_steps | hw_steps) & state.auto_step_ids)
        and not hw_steps
    ):
        return None
    detail = (
        f"{lead}; routed by {len(tests)} tests co-located at "
        f"{directory or 'a matching test file name'}"
    )
    if own_key_steps:
        detail += f"; its own registered key(s) name it in {len(own_key_steps)} steps"
    if dep_steps:
        detail += f"; {len(dep_steps)} steps declare it as a source dep"
    if family:
        detail += f"; {family} hardware-convention tagging adds {len(hw_steps)} steps"
    return Claim(
        "colocated-tests",
        detail,
        test_files=set(tests),
        step_ids=inferred_steps | hw_steps,
        # Subtracted, not merely left out: hardware steps stand for compiled
        # reach nothing records, so a step one holds stays held.
        droppable_step_ids=inferred_steps - hw_steps,
        droppable_test_files=True,
    )


def _pr_auto_selected(state: RepoState, claim: Claim) -> set[str]:
    """Auto-run PR-pipeline steps this claim alone would select.

    The same question selection answers, scoped to PR-pipeline auto steps on
    BOTH legs so two claims compare on one axis.
    """
    out: set[str] = set()
    for pipeline in state.pipelines:
        if pipeline.config.name != PR_PIPELINE:
            continue
        for step in pipeline.steps:
            step_id = step.step_id
            if step_id not in state.auto_step_ids:
                continue
            if step_id in claim.step_ids:
                out.add(step_id)
                continue
            targets = pipeline.targets.get(step_id)
            if (
                targets is not None
                and claim.test_files
                and _targets_cover(targets, claim.test_files)
            ):
                out.add(step_id)
    return out


def _classify_colocated_tests(state: RepoState, path: str) -> Claim | None:
    """Route a file inside the import cycle by its tests instead of its reach.

    None means "no answer here" and the graph rule runs unchanged, which is
    the only fallback mechanism.
    """
    if mode() == "off":
        return None
    if path not in state.full.import_cycle().reach_blind:
        return None
    tests, directory = implicated_tests(state, path)
    if not tests:
        return None
    # Registered-key routing is deliberately NOT carried over. It unions keys
    # across the whole reverse closure, so inside the cycle it collects the
    # entire registry, which is the amplification this rule exists to remove.
    return _colocated_claim(
        state,
        path,
        tests,
        directory,
        f"{path} is inside the import cycle, where reach selects everything",
    )


def _plain_static_tests(state: RepoState, path: str) -> frozenset[str]:
    """Test files reaching `path` transitively via module-level import edges.

    The hub arm's safety floor. Outside the cycle a file can reach everything
    two ways: lazy-edge amplification, which colocation legitimately drops, and
    plain module-level fan-in, which is REAL reach that a known breakage
    propagated through. Only what lazy machinery amplified may be dropped.
    """
    reverse = state.full.plain_reverse
    seen, stack = {path}, [path]
    while stack:
        node = stack.pop()
        for src in reverse.get(node, ()):
            if src not in seen:
                seen.add(src)
                stack.append(src)
    return frozenset(f for f in seen if f.startswith("tests/"))


def _colocated_hub(state: RepoState, path: str, graph_claim: Claim) -> Claim | None:
    """The hub arm: a non-cycle file whose closure has gone hub-like is routed
    by its tests plus its plain-static reach, dropping only lazy-edge breadth.

    Outside the cycle the closure often still tells you something, so the swap
    is gated twice. The size gate keeps the graph's answer wherever it is narrow
    enough to be information. The strict `<` clamp makes "never widen"
    structural rather than a side effect of the threshold: a file whose plain
    fan-in IS its whole closure declines here and keeps the graph answer.
    """
    if mode() != "on":
        return None
    tests, directory = implicated_tests(state, path)
    if not tests:
        return None
    before = _pr_auto_selected(state, graph_claim)
    if len(before) < MIN_GRAPH_STEPS:
        return None
    tests = tests | _plain_static_tests(state, path)
    # Unlike the cycle arm, a file's OWN registered keys are kept: by-name
    # coverage no mirror or importer can see. Only closure-derived keys go.
    own_keys = frozenset(state.keys.steps_naming(state.keys.for_file(path)))
    hub = _colocated_claim(
        state,
        path,
        tests,
        directory,
        f"{path} is outside the import cycle but reaches {len(before)} auto "
        "steps, hub-like",
        own_key_steps=own_keys,
    )
    if hub is None or len(_pr_auto_selected(state, hub)) >= len(before):
        return None
    return hub
