# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic-state units for the zero-auto-coverage branch matrix: polarity
dispatch is pinned off-HEAD so live-repo drift cannot mask a broken branch."""

from pathlib import Path

from ci_analyzer.graph.assets import AssetParse
from ci_analyzer.graph.build import FullGraph
from ci_analyzer.graph.dispatch import DispatchParse
from ci_analyzer.graph.factories import FactoryParse
from ci_analyzer.graph.imports import ImportGraph
from ci_analyzer.graph.platform import PlatformParse
from ci_analyzer.graph.registry import QuantParse, RegistryParse
from ci_analyzer.graph.spawn import SpawnParse
from ci_analyzer.jobs.model import LoadReport, PipelineConfig, Step
from ci_analyzer.jobs.testmap import StepTargets
from ci_analyzer.keys import KeyIndex
from ci_analyzer.repo import ModuleIndex
from ci_analyzer.select import AnalyzerState, PipelineData, _classify_graph

FILES = (
    "vllm/foo.py",
    "tests/orphan/test_x.py",
    "tests/orphan/helper.py",
    "tests/legacy/test_l.py",
    "tests/legacy/helper.py",
    "tests/other/test_y.py",
    "tests/plugins/myplug/myplug/mod.py",
)


def _mini_state():
    index = ModuleIndex()
    for f in FILES:
        index.add(f[:-3].replace("/", "."), f)
    index.installable_roots["tests/plugins/myplug/myplug/mod.py"] = (
        "tests/plugins/myplug"
    )
    graph = ImportGraph()
    graph.add_edge("tests/orphan/test_x.py", "vllm/foo.py")
    graph.add_edge("tests/orphan/test_x.py", "tests/orphan/helper.py")
    graph.add_edge("tests/orphan/test_x.py", "tests/orphan/data.yaml")
    graph.add_edge("tests/legacy/test_l.py", "tests/legacy/helper.py")
    full = FullGraph(
        index=index,
        graph=graph,
        registry=RegistryParse(),
        quant=QuantParse(),
        factories=FactoryParse(),
        dispatch=DispatchParse(),
        platform=PlatformParse(),
        spawn=SpawnParse(),
        assets=AssetParse(),
    )
    step = Step(
        pipeline="p",
        source_file="x.yaml",
        label="auto",
        key="auto",
        group=None,
        commands=["pytest other/"],
        source_file_dependencies=None,
    )
    st = StepTargets(step_id=step.step_id)
    st.add_target("tests/other", "pytest")
    pdata = PipelineData(
        PipelineConfig("p", ".buildkite/x.yaml", [], [], []),
        [step],
        {step.step_id: st},
    )
    state = AnalyzerState(
        repo=Path("."),
        pipelines=[pdata],
        full=full,
        catalog=[f for f in FILES if "/test_" in f],
        load_report=LoadReport(),
    )
    state.invoked = {"tests/other/test_y.py"}
    state.auto_step_ids = {step.step_id}
    state.legacy_invoked = {"tests/legacy/test_l.py"}
    state.keys = KeyIndex(
        searchable={step.step_id: "uv pip install -e ./plugins/myplug"}
    )
    return state, step.step_id


def test_vllm_zero_coverage_fails_open():
    state, _sid = _mini_state()
    claim = _classify_graph(state, "vllm/foo.py")
    assert claim.rule == "fail-open" and claim.run_all == {"p"}
    assert "zero-closure polarity" in claim.detail


def test_orphan_test_selects_nothing():
    state, _sid = _mini_state()
    claim = _classify_graph(state, "tests/orphan/test_x.py")
    assert claim.rule == "graph" and not claim.run_all
    assert not claim.step_ids and "orphan" in claim.detail


def test_orphan_helper_fails_open():
    state, _sid = _mini_state()
    claim = _classify_graph(state, "tests/orphan/helper.py")
    assert claim.rule == "fail-open" and claim.run_all == {"p"}


def test_legacy_only_helper_selects_nothing_with_note():
    state, _sid = _mini_state()
    claim = _classify_graph(state, "tests/legacy/helper.py")
    assert claim.rule == "graph" and not claim.run_all
    assert "legacy" in claim.detail


def test_nonpy_helper_with_asset_test_closure_keeps_fail_open():
    """The `not test_files` guard: a non-.py file reaching a test (asset edge)
    that auto-runs nowhere must keep fail-open, not route by target coverage."""
    state, _sid = _mini_state()
    claim = _classify_graph(state, "tests/orphan/data.yaml")
    assert claim.rule == "fail-open" and claim.run_all == {"p"}


def test_installable_package_routes_by_haystack():
    state, sid = _mini_state()
    claim = _classify_graph(state, "tests/plugins/myplug/myplug/mod.py")
    assert claim.rule == "graph" and not claim.run_all
    assert sid in claim.step_ids


def test_installable_package_falls_open_when_unreferenced():
    state, sid = _mini_state()
    state.keys.searchable[sid] = "pytest other/"
    claim = _classify_graph(state, "tests/plugins/myplug/myplug/mod.py")
    assert claim.rule == "fail-open" and claim.run_all == {"p"}
