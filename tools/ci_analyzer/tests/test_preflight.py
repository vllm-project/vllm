# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preflight units (synthetic), the clean-at-HEAD check, and select() wiring."""

import dataclasses
import os
from pathlib import Path

from ci_analyzer.curated import DYNAMIC_IMPORT_FILES, ENGINE_ENTRY_MODULES
from ci_analyzer.graph.assets import AssetParse
from ci_analyzer.graph.build import FullGraph
from ci_analyzer.graph.dispatch import DispatchParse
from ci_analyzer.graph.factories import FactoryParse
from ci_analyzer.graph.imports import DynamicSite, ImportGraph
from ci_analyzer.graph.platform import PlatformParse
from ci_analyzer.graph.registry import QuantParse, RegistryParse
from ci_analyzer.graph.spawn import SpawnParse
from ci_analyzer.jobs.model import LoadReport, PipelineConfig, Step
from ci_analyzer.jobs.testmap import StepTargets
from ci_analyzer.preflight import PreflightReport, run_preflight
from ci_analyzer.repo import ModuleIndex
from ci_analyzer.select import PipelineData, detect_duplicate_ids, select

# The synthetic units still need a checkout root: classifying a dynamic site
# reads its source line to look for the pragma.
REPO = Path(os.environ.get("VLLM_REPO", Path(__file__).resolve().parents[3]))


def _healthy_graph():
    """A healthy vLLM graph always has dynamic sites, and every one is either
    census-blessed or pragma'd. An empty list is the walker having stopped
    recording, which preflight treats as blindness, so the fixture has to carry
    one or it models a broken build rather than a healthy one."""
    graph = ImportGraph()
    graph.dynamic_sites.append(
        DynamicSite(sorted(DYNAMIC_IMPORT_FILES)[0], 1, "import_module")
    )
    return graph


def _healthy_full(**overrides):
    index = ModuleIndex()
    for m in ENGINE_ENTRY_MODULES:
        index.add(m, m.replace(".", "/") + ".py")
    parts = dict(
        index=index,
        graph=_healthy_graph(),
        registry=RegistryParse(
            entries={"Arch": ("mod", "Cls")}, hf_ids={"Arch": {"org/x"}}
        ),
        quant=QuantParse(methods={"fp8": "f.py"}),
        factories=FactoryParse(
            register_entries={"K": "k.py"},
            parser_entries={"p": "p.py"},
            enum_entries={"E": "e.py"},
            parser_engine_entries={"pe": "pe.py"},
            class_table_entries={"C": "c.py"},
            module_attrs={"LLM": "entrypoints.llm"},
            module_attr_resolved=1,
        ),
        dispatch=DispatchParse(),
        platform=PlatformParse(),
        spawn=SpawnParse(entrypoint_file="vllm/entrypoints/cli/main.py"),
        assets=AssetParse(),
    )
    parts.update(overrides)
    return FullGraph(**parts)


def _step(key="s1"):
    return Step(
        pipeline="p",
        source_file="x.yaml",
        label=key,
        key=key,
        group=None,
        commands=["pytest x/"],
        source_file_dependencies=None,
    )


def _pipe(steps, targets):
    config = PipelineConfig("p", ".buildkite/x.yaml", [], [], [])
    return PipelineData(config, steps, targets)


def _covered_targets(step):
    st = StepTargets(step_id=step.step_id)
    st.add_target("tests/x", "pytest")
    return {step.step_id: st}


def test_clean_on_healthy_inputs():
    step = _step()
    pf = run_preflight(
        REPO, [_pipe([step], _covered_targets(step))], _healthy_full(), LoadReport()
    )
    assert pf.clean


def test_unparsable_step_force_selected():
    step = _step()
    st = StepTargets(step_id=step.step_id)
    st.unparsable.append("docker exec ci pytest x")
    pf = run_preflight(
        REPO, [_pipe([step], {step.step_id: st})], _healthy_full(), LoadReport()
    )
    assert step.step_id in pf.force_select
    assert "unparsable" in pf.force_select[step.step_id]


def test_unknown_field_force_selected():
    step = _step()
    report = LoadReport()
    report.record_unknown({"commands_gpu": ["x"]}, step.step_id)
    pf = run_preflight(
        REPO, [_pipe([step], _covered_targets(step))], _healthy_full(), report
    )
    assert step.step_id in pf.force_select


def test_duplicate_id_force_selected():
    a, b = _step("dup"), _step("dup")
    report = LoadReport()
    detect_duplicate_ids([a, b], report)
    assert report.duplicate_ids == ["p:dup"]
    pf = run_preflight(
        REPO, [_pipe([a, b], _covered_targets(a))], _healthy_full(), report
    )
    assert "p:dup" in pf.force_select


def test_zero_target_step_warns_only():
    step = _step()
    st = StepTargets(step_id=step.step_id)
    pf = run_preflight(
        REPO, [_pipe([step], {step.step_id: st})], _healthy_full(), LoadReport()
    )
    assert not pf.force_select and not pf.run_all_reasons
    assert any("no derivable targets" in w for w in pf.warnings)


def test_always_run_step_exempt_from_zero_target_warning():
    """always_runs steps run regardless, so the zero-target guard must skip them."""
    step = _step(key="image-build")
    st = StepTargets(step_id=step.step_id)
    pf = run_preflight(
        REPO, [_pipe([step], {step.step_id: st})], _healthy_full(), LoadReport()
    )
    assert pf.clean


def test_empty_registry_escalates_run_all():
    step = _step()
    pf = run_preflight(
        REPO,
        [_pipe([step], _covered_targets(step))],
        _healthy_full(registry=RegistryParse()),
        LoadReport(),
    )
    assert any("model registry" in r for r in pf.run_all_reasons)


def test_dead_module_attrs_escalates():
    factories = FactoryParse(
        register_entries={"K": "k.py"},
        parser_entries={"p": "p.py"},
        enum_entries={"E": "e.py"},
        parser_engine_entries={"pe": "pe.py"},
        class_table_entries={"C": "c.py"},
        module_attrs={"LLM": "x:LLM"},
        module_attr_resolved=0,
    )
    step = _step()
    pf = run_preflight(
        REPO,
        [_pipe([step], _covered_targets(step))],
        _healthy_full(factories=factories),
        LoadReport(),
    )
    assert any("MODULE_ATTRS" in r for r in pf.run_all_reasons)


def test_empty_parser_engine_table_escalates():
    """A parser subsystem restructure that empties parser_engine_entries leaves
    the analyzer blind to that coverage channel -> run-all, not silent."""
    factories = FactoryParse(
        register_entries={"K": "k.py"},
        parser_entries={"p": "p.py"},
        enum_entries={"E": "e.py"},
        class_table_entries={"C": "c.py"},
        module_attrs={"LLM": "entrypoints.llm"},
        module_attr_resolved=1,
    )
    pf = run_preflight(
        REPO,
        [_pipe([_step()], _covered_targets(_step()))],
        _healthy_full(factories=factories),
        LoadReport(),
    )
    assert any("parser engine" in r for r in pf.run_all_reasons)


def test_one_dead_parser_table_escalates():
    """The four lazy tables merge into ONE parser_entries dict and their parser
    names collide (deepseek_v3 is both a reasoning and a tool parser), so a dead
    anchor barely dents its size -- killing the tokenizers table changes it by
    zero at HEAD. Each table is guarded on its own count, and the reason must
    name the dead one. parser_entries stays non-empty here so only the per-table
    row can produce the reason."""
    factories = FactoryParse(
        register_entries={"K": "k.py"},
        parser_entries={"p": "p.py"},
        parser_table_counts={
            "vllm/reasoning/__init__.py": 0,
            "vllm/tool_parsers/__init__.py": 46,
        },
        enum_entries={"E": "e.py"},
        parser_engine_entries={"pe": "pe.py"},
        class_table_entries={"C": "c.py"},
        module_attrs={"LLM": "entrypoints.llm"},
        module_attr_resolved=1,
    )
    pf = run_preflight(
        REPO,
        [_pipe([_step()], _covered_targets(_step()))],
        _healthy_full(factories=factories),
        LoadReport(),
    )
    assert any("vllm/reasoning/__init__.py" in r for r in pf.run_all_reasons)
    assert not any("vllm/tool_parsers" in r for r in pf.run_all_reasons)


def test_dangling_only_target_force_selects():
    """A pytest target resolving to no file (rename, zero-glob) with no other
    coverage force-selects. A non-pytest command body only warns."""
    step = _step()
    st = StepTargets(step_id=step.step_id)
    st.dangling.append("kernels/test_renamed_away_*.py")
    pf = run_preflight(
        REPO, [_pipe([step], {step.step_id: st})], _healthy_full(), LoadReport()
    )
    assert step.step_id in pf.force_select
    assert "dangling" in pf.force_select[step.step_id]


def test_dangling_beside_other_coverage_still_force_selects():
    """The escalation used to sit behind an other-coverage skip, so a step
    holding one live target (or merely a scanned script) beside a stale one
    could never reach it -- which is how the NPU steps went silent."""
    step = _step()
    st = StepTargets(step_id=step.step_id)
    st.add_target("tests/basic_correctness/test_basic.py", "pytest")
    st.dangling.append("tests/e2e/vllm_interface/")
    pf = run_preflight(
        REPO, [_pipe([step], {step.step_id: st})], _healthy_full(), LoadReport()
    )
    assert step.step_id in pf.force_select

    script_only = _step()
    st2 = StepTargets(step_id=script_only.step_id)
    st2.scripts_seen.append(".buildkite/scripts/hardware_ci/run-npu-test.sh")
    st2.dangling.append("tests/e2e/vllm_interface/")
    pf = run_preflight(
        REPO,
        [_pipe([script_only], {script_only.step_id: st2})],
        _healthy_full(),
        LoadReport(),
    )
    assert script_only.step_id in pf.force_select


def test_missing_spawn_entrypoint_escalates():
    step = _step()
    pf = run_preflight(
        REPO,
        [_pipe([step], _covered_targets(step))],
        _healthy_full(spawn=SpawnParse()),
        LoadReport(),
    )
    assert any("entrypoint" in r for r in pf.run_all_reasons)


def test_unresolved_engine_entry_disables_seam_gate():
    index = ModuleIndex()
    for m in ENGINE_ENTRY_MODULES[1:]:
        index.add(m, m.replace(".", "/") + ".py")
    step = _step()
    pf = run_preflight(
        REPO,
        [_pipe([step], _covered_targets(step))],
        _healthy_full(index=index),
        LoadReport(),
    )
    assert not pf.seam_gate_ok
    assert not pf.run_all_reasons


def test_parse_errors_recorded_per_file():
    graph = ImportGraph()
    graph.parse_errors.append("vllm/broken.py")
    step = _step()
    pf = run_preflight(
        REPO,
        [_pipe([step], _covered_targets(step))],
        _healthy_full(graph=graph),
        LoadReport(),
    )
    assert "vllm/broken.py" in pf.parse_error_paths


def test_preflight_clean_at_head(state):
    """Every guard is quiet against the live checkout. A failure here is
    upstream drift the analyzer would previously have swallowed."""
    assert state.preflight.clean, (
        state.preflight.run_all_reasons,
        state.preflight.force_select,
        state.preflight.warnings,
    )


def test_soft_fail_step_is_not_escalated():
    """The only dangling steps at HEAD are the NPU ones, which are soft_fail:
    their failure never gates the merge, so forcing them on every PR buys no
    recall. A hard-fail step in the same shape must still escalate."""
    hard = _step()
    st = StepTargets(step_id=hard.step_id)
    st.dangling.append("tests/e2e/vllm_interface/")
    pf = run_preflight(
        REPO, [_pipe([hard], {hard.step_id: st})], _healthy_full(), LoadReport()
    )
    assert hard.step_id in pf.force_select

    soft = _step()
    soft.soft_fail = True
    st2 = StepTargets(step_id=soft.step_id)
    st2.dangling.append("tests/e2e/vllm_interface/")
    pf = run_preflight(
        REPO, [_pipe([soft], {soft.step_id: st2})], _healthy_full(), LoadReport()
    )
    assert soft.step_id not in pf.force_select


def test_force_select_applies_to_any_code_diff(state):
    step_id = next(
        s.step_id
        for p in state.pipelines
        for s in p.steps
        if not s.manual_only and not s.always_runs
    )
    pf = PreflightReport(force_select={step_id: "preflight: test escalation"})
    st2 = dataclasses.replace(state, preflight=pf)
    sel = select(st2, ["vllm/logger.py"])
    assert step_id in sel.selected
    assert "preflight: test escalation" in sel.selected[step_id]


def test_run_all_reason_escalates_every_pipeline(state):
    pf = PreflightReport(run_all_reasons=["preflight: core table empty"])
    st2 = dataclasses.replace(state, preflight=pf)
    sel = select(st2, ["vllm/logger.py"])
    assert set(sel.run_all) == {p.config.name for p in state.pipelines}


def test_docs_only_immune_to_preflight_escalation(state):
    pf = PreflightReport(run_all_reasons=["preflight: core table empty"])
    st2 = dataclasses.replace(state, preflight=pf)
    sel = select(st2, ["docs/serving/index.md"])
    assert sel.docs_only and not sel.run_all and not sel.selected


def test_parse_error_path_fails_open(state):
    pf = PreflightReport(parse_error_paths=frozenset({"vllm/logger.py"}))
    st2 = dataclasses.replace(state, preflight=pf)
    sel = select(st2, ["vllm/logger.py"])
    assert "vllm_ci" in sel.run_all
    assert "failed to parse" in sel.run_all["vllm_ci"]


def test_unmapped_device_recorded_and_warned():
    step = _step()
    odd = dataclasses.replace(step, key="tpu-new", device="v6e-8")
    pf = run_preflight(
        REPO,
        [_pipe([step, odd], _covered_targets(step))],
        _healthy_full(),
        LoadReport(),
    )
    assert "v6e-8" in pf.unmapped_devices
    assert any("taxonomy" in w for w in pf.warnings)


def test_zero_dynamic_sites_escalates_run_all():
    """The detection floor: an empty site list means the walker stopped
    recording, so both the forward audit and the census gate pass vacuously."""
    step = _step()
    pf = run_preflight(
        REPO,
        [_pipe([step], _covered_targets(step))],
        _healthy_full(graph=ImportGraph()),
        LoadReport(),
    )
    assert any("zero sites" in r for r in pf.run_all_reasons)


def test_unclassified_site_recorded_and_warned():
    graph = _healthy_graph()
    graph.dynamic_sites.append(
        DynamicSite("vllm/nobody/vouched/for_this.py", 7, "import_module")
    )
    pf = run_preflight(
        REPO,
        [_pipe([_step()], _covered_targets(_step()))],
        _healthy_full(graph=graph),
        LoadReport(),
    )
    assert pf.unclassified_sites == frozenset({"vllm/nobody/vouched/for_this.py"})
    assert not pf.run_all_reasons, "an unmodeled site is bounded, not total"
    assert any("unmodeled dynamic-import" in w for w in pf.warnings)
    assert not pf.clean


def test_census_rot_is_not_a_preflight_escalation():
    """Selection analyzes the checkout at a PR's merge base, which can predate
    the census this package ships. Dead/missing entries are the expected reading
    against an older tree, so escalating on them would run everything for every
    PR based before the newest entry. The dynamic-sites check owns that, at
    head, where the comparison is fair."""
    graph = _healthy_graph()
    graph.dynamic_sites[:] = [DynamicSite("vllm/some/live_site.py", 1, "import_module")]
    pf = run_preflight(
        REPO,
        [_pipe([_step()], _covered_targets(_step()))],
        _healthy_full(graph=graph),
        LoadReport(),
    )
    assert not pf.run_all_reasons, pf.run_all_reasons
