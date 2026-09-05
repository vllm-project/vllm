# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic-state units for the zero-auto-coverage branch matrix: direction
dispatch is pinned off-HEAD so live-vllm_repo drift cannot mask a broken branch."""

from pathlib import Path

from ci_selector.codemap.classify import _classify_graph
from ci_selector.codemap.graph.assets import AssetParse
from ci_selector.codemap.graph.build import FullGraph
from ci_selector.codemap.graph.demote import DispatchParse
from ci_selector.codemap.graph.factories import FactoryParse
from ci_selector.codemap.graph.imports import ImportGraph
from ci_selector.codemap.graph.model_registry import QuantParse, RegistryParse
from ci_selector.codemap.graph.platform import PlatformParse
from ci_selector.codemap.graph.spawn import SpawnParse
from ci_selector.codemap.pipeline.step import LoadReport, PipelineConfig, Step
from ci_selector.codemap.pipeline.targets import StepTargets
from ci_selector.codemap.registered_names import KeyIndex
from ci_selector.codemap.repo import ModuleIndex
from ci_selector.codemap.state import PipelineData, RepoState

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
    state = RepoState(
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


def test_vllm_zero_coverage_unreferenced_selects_nothing():
    """A vllm file with an empty closure that nothing names -- no step text,
    key, specific declarer, or invoked-test literal -- cannot be run by any
    job, so it selects the floor."""
    state, _sid = _mini_state()
    claim = _classify_graph(state, "vllm/foo.py")
    assert claim.rule == "graph" and not claim.run_all
    assert not claim.step_ids and "nothing can run it" in claim.detail


def test_vllm_zero_coverage_named_in_commands_keeps_fail_open():
    """The same empty-closure file keeps the run-all once any step's command
    text names its module -- the fail-open is for loaders the graph cannot
    see."""
    from ci_selector.codemap.registered_names import KeyIndex

    state, sid = _mini_state()
    state.keys = KeyIndex(searchable={sid: "python -m vllm.foo --check"})
    claim = _classify_graph(state, "vllm/foo.py")
    assert claim.rule == "fail-open" and claim.run_all == {"p"}
    assert "empty-closure direction" in claim.detail


def test_orphan_test_selects_nothing():
    state, _sid = _mini_state()
    claim = _classify_graph(state, "tests/orphan/test_x.py")
    assert claim.rule == "graph" and not claim.run_all
    assert not claim.step_ids and "orphan" in claim.detail


def test_orphan_helper_selects_the_floor():
    """A tests/ helper with zero auto-run coverage has no live job that loads
    it; its coverage rides along as manual hits instead of escalating."""
    state, _sid = _mini_state()
    claim = _classify_graph(state, "tests/orphan/helper.py")
    assert claim.rule == "graph" and not claim.run_all
    assert "nothing auto-runs it" in claim.detail


def test_legacy_only_helper_selects_nothing_with_note():
    state, _sid = _mini_state()
    claim = _classify_graph(state, "tests/legacy/helper.py")
    assert claim.rule == "graph" and not claim.run_all
    assert "legacy" in claim.detail


def test_nonpy_helper_with_asset_test_closure_carries_it():
    """The `not test_files` guard still skips target-coverage routing for a
    non-.py file whose asset edge reaches a test, but the zero-auto answer is
    the floor carrying that closure, not run-all."""
    state, _sid = _mini_state()
    claim = _classify_graph(state, "tests/orphan/data.yaml")
    assert claim.rule == "graph" and not claim.run_all
    assert claim.test_files


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
