# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check 5.4's contract: absence of evidence is never freshness.

The gate only ever adds keeps, so its failure mode is not a wrong drop directly
-- it is a step that should have been disqualified and was not. Every test here
pins a case where the safe answer is "stale".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from ci_selector.coverage.freshness import Freshness, Surface, build


@dataclass
class FakeConfig:
    name: str = "vllm_ci"
    config_file: str = ".buildkite/ci_config.yaml"


@dataclass
class FakeStep:
    step_id: str
    source_file: str = ".buildkite/test_areas/x.yaml"


@dataclass
class FakeTargets:
    scripts_seen: list[str] = field(default_factory=list)
    targets: list = field(default_factory=list)


@dataclass
class FakeGraph:
    imports: dict = field(default_factory=dict)

    def forward_closure(self, files, include_boot=True):
        seen = set(files)
        stack = list(files)
        while stack:
            for dst in self.imports.get(stack.pop(), ()):
                if dst not in seen:
                    seen.add(dst)
                    stack.append(dst)
        return seen


@dataclass
class FakePipeline:
    config: FakeConfig
    steps: list
    targets: dict


@dataclass
class FakeFull:
    graph: FakeGraph


@dataclass
class FakeState:
    pipelines: list
    full: FakeFull
    catalog: list = field(default_factory=list)


def state_with(step_targets, imports=None, catalog=None):
    steps = [FakeStep(sid) for sid in step_targets]
    return FakeState(
        pipelines=[FakePipeline(FakeConfig(), steps, dict(step_targets))],
        full=FakeFull(FakeGraph(imports or {})),
        catalog=catalog or [],
    )


class TestUnknownIsStale:
    def test_a_step_with_no_surface_is_always_stale(self):
        f = Freshness(commit="r", surfaces={})
        assert f.stale("vllm_ci:anything", frozenset())

    def test_a_step_without_targets_gets_no_surface(self):
        state = state_with({})
        state.pipelines[0].steps = [FakeStep("vllm_ci:a")]
        got = build(state, "R")
        assert "vllm_ci:a" in got.unknown
        assert got.stale("vllm_ci:a", frozenset())

    def test_an_empty_diff_leaves_a_known_step_fresh(self):
        f = Freshness(commit="r", surfaces={"s": Surface(frozenset({"tests/a.py"}))})
        assert not f.stale("s", frozenset())


class TestTestSide:
    def test_a_changed_target_file_makes_it_stale(self):
        f = Freshness(commit="r", surfaces={"s": Surface(frozenset({"tests/a.py"}))})
        assert f.stale("s", frozenset({"tests/a.py"}))

    def test_the_conftest_chain_arrives_through_the_forward_closure(self):
        # No conftest special-casing exists in freshness.py on purpose: the
        # import graph already carries test -> conftest edges, so a forward walk
        # picks them up. This pins that we rely on it.
        state = state_with(
            {"vllm_ci:a": FakeTargets()},
            imports={"tests/a.py": {"tests/conftest.py"}},
            catalog=["tests/a.py"],
        )
        state.pipelines[0].targets["vllm_ci:a"].targets = [
            type("T", (), {"path": "tests/"})()
        ]
        got = build(state, "R")
        assert "tests/conftest.py" in got.surfaces["vllm_ci:a"].tests
        assert got.stale("vllm_ci:a", frozenset({"tests/conftest.py"}))

    def test_pytest_config_is_in_every_surface(self):
        state = state_with({"vllm_ci:a": FakeTargets()})
        got = build(state, "R")
        assert got.stale("vllm_ci:a", frozenset({"pyproject.toml"}))

    def test_a_non_surface_file_leaves_it_fresh(self):
        f = Freshness(commit="r", surfaces={"s": Surface(frozenset({"tests/a.py"}))})
        assert not f.stale("s", frozenset({"vllm/model_executor/models/opt.py"}))


class TestStepSide:
    def test_the_defining_yaml_makes_it_stale(self):
        state = state_with({"vllm_ci:a": FakeTargets()})
        got = build(state, "R")
        assert got.stale("vllm_ci:a", frozenset({".buildkite/test_areas/x.yaml"}))

    def test_a_recursed_script_makes_it_stale(self):
        state = state_with(
            {"vllm_ci:a": FakeTargets(scripts_seen=[".buildkite/scripts/run.sh"])}
        )
        got = build(state, "R")
        assert got.stale("vllm_ci:a", frozenset({".buildkite/scripts/run.sh"}))

    def test_the_pipeline_config_does_NOT_make_it_stale(self):
        # `ci_config*.yaml` says which steps exist and when everything runs,
        # never what one step does. Including it disqualified every step on
        # every benchmark PR (40 of 40 ranges touch it) for no soundness gain.
        state = state_with({"vllm_ci:a": FakeTargets()})
        got = build(state, "R")
        assert not got.stale("vllm_ci:a", frozenset({".buildkite/ci_config.yaml"}))

    def test_another_steps_yaml_does_not(self):
        state = state_with({"vllm_ci:a": FakeTargets()})
        got = build(state, "R")
        assert not got.stale(
            "vllm_ci:a", frozenset({".buildkite/test_areas/other.yaml"})
        )


def test_stale_steps_filters_the_selection():
    f = Freshness(
        commit="r",
        surfaces={
            "a": Surface(frozenset({"tests/a.py"})),
            "b": Surface(frozenset({"tests/b.py"})),
        },
    )
    got = f.stale_steps(["a", "b", "never-recorded"], frozenset({"tests/a.py"}))
    assert got == {"a", "never-recorded"}


@pytest.mark.parametrize(
    "changed", [frozenset(), frozenset({"README.md"}), frozenset({"vllm/x.py"})]
)
def test_unrelated_changes_never_disqualify(changed):
    f = Freshness(commit="r", surfaces={"s": Surface(frozenset({"tests/a.py"}))})
    assert not f.stale("s", changed)
