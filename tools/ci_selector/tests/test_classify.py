# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end selection cases on the real checkout + generator-replica units."""

from pathlib import Path

import pytest
import regex as re
from ci_selector.codemap.claim import matches_source_dependency
from ci_selector.codemap.classify import select
from ci_selector.gitdiff import DiffFile, changed_paths
from ci_selector.validate.generator_replica import today_select
from helpers import HW, drift_message


def _selected(sel, pipeline="vllm_ci"):
    return {s for s in sel.selected if s.startswith(f"{pipeline}:")}


def _non_always(sel, state, pipeline="vllm_ci"):
    """Selected steps minus the always-run floor (image-build/AMD-base steps
    select() injects), for 'selects nothing but the floor' cases."""
    always = {s.step_id for p in state.pipelines for s in p.steps if s.always_runs}
    return {s for s in _selected(sel, pipeline) if s not in always}


def test_docs_only_short_circuit(state):
    sel = select(state, ["docs/serving/index.md", "README.md"])
    assert sel.docs_only and not sel.selected


def test_shared_build_inputs_reach_every_platform(state, declared_deps_on):
    """The guard for the two pipelines nothing else watches.

    The shared build inputs reach the AMD and Intel steps only through the
    image DAG, and that DAG is thin there: one Dockerfile copies its whole
    context, and the CPU suites depend on nothing because they build their own
    image in-step. With either mechanism missing, `csrc/` collapsed to a
    handful of steps on each.

    The crosscheck scores `vllm_ci` only, so no amount of PR replay catches
    that. This test is the floor.

    Since the build map shipped: an unmapped shared input must still reach
    every platform, while a mapped cuda+amd source keeps its own families wide
    and loses the intel pipeline down to CI's own declarers. The mapped
    fixture is asserted to exist and be mapped, because the file this test
    used before had been deleted and so pinned nothing.
    """
    names = {p.config.name for p in state.pipelines}
    auto_per = {
        n: len(
            [
                s
                for p in state.pipelines
                if p.config.name == n
                for s in p.steps
                if s.step_id in state.auto_step_ids
            ]
        )
        for n in names
    }
    # Control: an input only one platform's image copies must stay narrow, so
    # a trivially-everything selector cannot pass the assertion above it.
    narrow = select(state, ["requirements/test/rocm.txt"])

    for path in ("cmake/hipify.py", "requirements/common.txt"):
        assert path not in state.build_map.families
        sel = select(state, [path])
        assert not sel.run_all, f"{path} escalated instead of routing"
        for name in names:
            got = len([s for s in sel.selected if s.startswith(f"{name}:")])
            # Half is a collapse detector, not a tuned figure: the failure
            # being caught takes these pipelines to single digits.
            assert got >= auto_per[name] // 2, (
                f"{path} reaches {got} of {auto_per[name]} {name} steps; the "
                "image DAG stopped covering a platform that builds it"
            )
        for name in ("vllm_intel_ci", "vllm_rocm_ci"):
            wide = len([s for s in sel.selected if s.startswith(f"{name}:")])
            thin = len([s for s in narrow.selected if s.startswith(f"{name}:")])
            assert wide > thin, (
                f"{path} is a shared build input but reaches no more of "
                f"{name} than a rocm-only requirements file does"
            )

    from ci_selector.codemap import classify
    from ci_selector.codemap.step_refs import _source_dep_steps

    shared_tu = "csrc/libtorch_stable/cache_kernels.cu"
    assert (state.repo / shared_tu).is_file(), "the mapped fixture went ghost"
    assert state.build_map.families.get(shared_tu) == frozenset({"cuda", "amd"})
    sel = select(state, [shared_tu])
    assert not sel.run_all
    per, _union, nonfamily = state.family_partition()
    picked = set(sel.selected)
    # Its own families stay wide.
    assert len(picked & nonfamily) >= len(nonfamily) // 2
    assert len(picked & per["amd"]) >= len(per["amd"]) // 2
    # And the narrowing is live. Without this the test cannot tell if the
    # wider answer came back.
    declared = _source_dep_steps(state, shared_tu)
    native = classify._classify_native_tests(state, shared_tu)
    derived_core = native.step_ids if native else set()
    assert picked & per["xpu"] <= declared | derived_core, (
        f"{shared_tu} reaches intel steps beyond its declarers and its own "
        "op-test joints; the family scoping stopped applying"
    )


def test_csrc_cpu_routes_to_declarers_plus_cpu_family(state, declared_deps_on):
    """csrc/cpu/ is cpu-exclusive and unclaimed: route via the blanket csrc/
    declarers plus the cpu family, not the bare complement (GPU jobs can't run it)."""
    sel = select(state, ["csrc/cpu/cpu_attn.cpp"])
    assert "vllm_ci" not in sel.run_all
    assert any(c.rule == "declared-deps" for c in sel.claims)
    assert "vllm_ci:torch-stable-abi-audit" in sel.selected
    assert "vllm_ci:cpu-kernel-tests" in sel.selected
    # device-less GPU suites and rust cargo steps the bare complement kept are gone
    assert "vllm_ci:rust-frontend-cargo-tests" not in sel.selected
    assert "vllm_ci:distributed-comm-ops" not in sel.selected
    devices = {
        s.device
        for p in state.pipelines
        for s in p.steps
        if s.step_id in sel.selected and s.device
    }
    assert not any(d.startswith(("h100", "h200", "b200", "mi")) for d in devices), (
        devices
    )


def test_decorator_registered_member_routes_broad(state):
    """mm_preprocess feeds minimax's module-level @register_processor, so it
    really does reach the multimodal tests: route by coverage, not run-all.

    Either reachability rule may answer it. What matters is that the multimodal
    steps survive whichever one does."""
    sel = select(state, ["vllm/models/minimax_m3/common/mm_preprocess.py"])
    assert "vllm_ci" not in sel.run_all
    assert any(c.rule in ("graph", "colocated-tests") for c in sel.claims)
    labels = {
        s.label for p in state.pipelines for s in p.steps if s.step_id in sel.selected
    }
    assert any("Multi-Modal" in lbl for lbl in labels)


def test_package_data_device_scoping_trims_cross_device(state, vllm_repo, monkeypatch):
    """A device-named tuning config is unreadable on other devices, so its routing
    and family floor are device-scoped; disabling scope selects strictly more."""
    import ci_selector.codemap.hardware as hw

    cfgs = sorted(
        (vllm_repo / "vllm/model_executor/layers/fused_moe/configs").glob(
            "*device_name=NVIDIA_H*.json"
        )
    )
    assert cfgs, "no H-series tuning config at HEAD to exercise scoping"
    j = str(cfgs[0].relative_to(vllm_repo))
    scoped = select(state, [j])
    assert any(c.device_scope for c in scoped.claims)
    monkeypatch.setattr(hw, "device_scoped_out", lambda step, prefix: False)
    unscoped = select(state, [j])
    assert len(scoped.selected) < len(unscoped.selected)


def test_hardware_named_test_not_family_tagged(state):
    """A leaf-consumer file (a rocm-named test) has no invisible kernel reach,
    so it must NOT pull the whole AMD family, only its real coverage."""
    sel = select(state, ["tests/v1/attention/test_rocm_aiter_mla_mtp_split.py"])
    assert "vllm_ci" not in sel.run_all
    amd_steps = sum(
        1
        for p in state.pipelines
        for s in p.steps
        if s.step_id in sel.selected and s.mirror_hw == "amd"
    )
    assert amd_steps < 10, amd_steps  # the whole AMD family (~45) is not pulled


def test_no_vllm_file_imports_tests(state):
    """Soundness basis for tests-side scoping: nothing under vllm/ imports tests/,
    so a tests-side file's executing steps are exactly its own target coverage."""
    imports = state.full.graph.imports
    leaks = [
        (s, d)
        for s, dsts in imports.items()
        if s.startswith("vllm/")
        for d in dsts
        if d.startswith("tests/")
    ]
    assert not leaks, leaks[:5]


def test_engine_gate_on_worker_file_keeps_catching_job(state, declared_deps_on):
    """#49364: the engine-starting gate trims non-engine tests reached through a
    boot edge, but the catching job (V1 Sample + Logits) must survive.

    DECLARED-DEPS ONLY: rides a declaration the derived default gives up, so
    by default this file alone does not select the catching job. A derived
    route is queued in todo."""
    sel = select(state, ["vllm/v1/worker/gpu/cudagraph_utils.py"])
    assert "vllm_ci" not in sel.run_all
    labels = {
        s.label for p in state.pipelines for s in p.steps if s.step_id in sel.selected
    }
    assert any("Sample" in lbl and "Logits" in lbl for lbl in labels), (
        "the #49364 catching job must survive the engine gate"
    )


def test_gpu_worker_namespace_reaches_cpu_jobs(state, declared_deps_on):
    """cpu_worker.py subclasses gpu_worker.Worker, so gpu-namespace changes must reach
    intel_cpu jobs (the old subtractive rule under-selected).

    DECLARED-DEPS ONLY: rides a declaration the derived default gives up. CPU
    steps cannot record, so the add side cannot restore them either. A derived
    route is queued in todo."""
    sel = select(state, ["vllm/v1/worker/gpu_worker.py"])
    cpu_selected = {
        s.label
        for p in state.pipelines
        for s in p.steps
        if s.step_id in sel.selected and s.device in ("intel_cpu", "arm_cpu")
    }
    assert cpu_selected, "gpu_worker.py must reach at least one CPU job"


def test_var_prefixed_python_driver_in_script(state):
    """nixl scripts launch toy_proxy_server.py via a `python3 ${GIT_ROOT}/...` path,
    so a change must select a disaggregated step, not zero jobs."""
    sel = select(state, ["tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"])
    assert any(
        "nixl" in s.lower() or "disagg" in s.lower()
        for s in (*sel.selected, *sel.manual_hits)
    ), sorted(sel.selected)[:8]


def test_csrc_rocm_selects_amd_mirrors(state):
    """csrc/rocm is AMD-exclusive and routes through its declarers plus the
    amd family; it must also select vllm_ci's AMD mirror jobs, which run on
    the image built from it."""
    sel = select(state, ["csrc/rocm/attention.cu"])
    amd_mirrors = [
        s.step_id
        for p in state.pipelines
        for s in p.steps
        if s.step_id in sel.selected and s.mirror_hw == "amd"
    ]
    assert len(amd_mirrors) > 20, len(amd_mirrors)


def test_renderer_table_parsed(state):
    """_VLLM_RENDERERS/_VLLM_TOKENIZERS are parsed, matching the dynamic-import audit
    that classifies them as table-handled."""
    parsed = state.full.factories.parser_entries
    assert any(t.startswith("vllm/renderers/") for t in parsed.values()), (
        "renderer registry entries missing from the lazy-table parser"
    )


def test_lora_worked_example(state):
    sel = select(state, ["vllm/lora/punica_wrapper/punica_gpu.py"])
    assert "vllm_ci:lora" in sel.selected
    assert "vllm_ci:lora-amd:amd" in sel.selected  # mirror comes along
    assert "vllm_ci" not in sel.run_all


def test_single_model_file_scoped(state):
    sel = select(state, ["vllm/model_executor/models/mllama4.py"])
    assert "vllm_ci" not in sel.run_all, "single-model change must not run-all"
    picked = _selected(sel)
    assert any("models" in s for s in picked)
    # scoped: well under half the pipeline
    non_build = {s for s in picked if "image-build" not in s}
    total = sum(
        1
        for p in state.pipelines
        if p.config.name == "vllm_ci"
        for s in p.steps
        if not s.manual_only
    )
    assert len(non_build) < total / 2


def test_core_config_near_run_all_is_honest(state):
    sel = select(state, ["vllm/config/__init__.py"])
    assert "vllm_ci" not in sel.run_all  # via graph, not via world
    assert len(_selected(sel)) > 50  # the honest wide closure


def test_determinism_helper_selects_its_step(state):
    sel = select(state, ["tests/v1/determinism/utils.py"])
    labels = {
        s.label for p in state.pipelines for s in p.steps if s.step_id in sel.selected
    }
    assert any("nvariance" in lbl or "eterminism" in lbl for lbl in labels), labels


def test_evals_config_txt_selects_lm_eval_step(state):
    sel = select(state, ["tests/evals/gsm8k/configs/models-small.txt"])
    assert any("lm_eval" in s or "lm-eval" in s for s in sel.selected), sorted(
        sel.selected
    )


def test_job_yaml_selects_its_steps(state):
    sel = select(state, [".buildkite/test_areas/lora.yaml"])
    assert "vllm_ci:lora" in sel.selected
    assert "vllm_ci" not in sel.run_all


def test_brand_new_file_fails_open(state):
    sel = select(state, ["vllm/some_totally_new_subsystem.py"])
    assert "vllm_ci" in sel.run_all


def test_changed_test_file_selects_owning_job(state):
    sel = select(state, ["tests/lora/test_llama_tp.py"])
    assert any("lora" in s for s in sel.selected)


def test_nixl_connector_key_routing(state):
    """PR #50326: nixl connector sources must select the PD/accuracy e2e jobs naming
    "NixlConnector" in commands, scripts (incl. var-assigned nested), or test files."""
    sel = select(
        state, ["vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py"]
    )
    assert "vllm_ci" not in sel.run_all
    picked = _selected(sel)
    assert any("nixlconnector-pd-accuracy" in s for s in picked), sorted(picked)
    assert any(s.endswith(":amd") for s in picked), "AMD mirrors must come along"


def test_hardware_convention_tagging(state):
    """PR #46952: a rocm_* attention backend must select AMD-family jobs (hardware
    dispatch picks it by platform, invisible to imports/keys)."""
    sel = select(state, ["vllm/v1/attention/backends/mla/rocm_aiter_mla.py"])
    assert "vllm_ci" not in sel.run_all
    picked = _selected(sel)
    assert any(":amd" in s or "amd" in s for s in picked), sorted(picked)[:8]
    amd_sample = [
        s for s in picked if "v1-sample" in s.lower() or "sample" in s.lower()
    ]
    assert any(s.endswith(":amd") or "-amd" in s for s in amd_sample), (
        "the amd v1-sample mirror (the #46952 failing job) must be selected"
    )

    sel_cpu = select(state, ["vllm/v1/worker/cpu_worker.py"])
    cpu_labels = {
        s.label
        for p in state.pipelines
        for s in p.steps
        if s.step_id in sel_cpu.selected and s.device in ("intel_cpu", "arm_cpu")
    }
    assert any("Distributed" in lbl for lbl in cpu_labels), cpu_labels


def test_model_named_in_step_env_var(state):
    """misc.yaml pins VLLM_TEST_MODEL=deepseek-ai/... in a step's command env, so the
    model module must select that step even though no test file names it."""
    sel = select(state, ["vllm/model_executor/models/deepseek_v2.py"])
    labels = {
        s.label for p in state.pipelines for s in p.steps if s.step_id in sel.selected
    }
    assert any("Invariance" in lbl for lbl in labels), sorted(labels)[:10]


def test_lm_eval_routes_by_declared_deps(state, declared_deps_on):
    """lm-eval steps route by source_file_dependencies, not the import graph, so a
    quant file and a base-model file select different lm-eval jobs."""
    quant = set(
        select(state, ["vllm/model_executor/layers/quantization/fp8.py"]).selected
    )
    assert "vllm_ci:lm-eval-small-models" in quant
    model = set(select(state, ["vllm/model_executor/models/llama.py"]).selected)
    # declares quantization, not models -> a base-model change must NOT select it
    assert "vllm_ci:lm-eval-small-models" not in model
    # the amd mirror declares vllm/model_executor/models/ -> it must
    assert "vllm_ci:lm-eval-small-models-amd:amd" in model


def test_no_classifier_drops_declared_source_deps(state, declared_deps_on):
    """A step declaring a path in source_file_dependencies must survive into that path's
    claim; dropping it is under-selection. Probes a deep new-subpackage file under every
    declared dir plus every declared file. Exempts run-all and authoritative-nothing
    claims; on a graph-known file catch-all (bare `vllm/`) declarers are omitted
    (specific-only)."""
    from ci_selector.codemap.classify import (
        _classify,
        _graph_known,
        _source_dep_steps,
    )
    from ci_selector.codemap.unions import _DEP_UNION_EXEMPT

    probes: set[str] = set()
    for p in state.pipelines:
        for s in p.steps:
            for dep in s.source_file_dependencies or []:
                d = dep.rstrip("/")
                if (state.repo / d).is_dir():
                    probes.add(f"{d}/__ci_probe_pkg__/__init__.py")
                else:
                    probes.add(d)
    checked = 0
    for probe in sorted(probes):
        declarers = _source_dep_steps(
            state, probe, specific_only=_graph_known(state, probe)
        )
        if not declarers:
            continue
        claim = _classify(state, probe, None)
        if claim.run_all or claim.rule in _DEP_UNION_EXEMPT:
            continue
        missing = declarers - claim.step_ids
        assert not missing, f"{probe} ({claim.rule}) drops {sorted(missing)[:5]}"
        checked += 1
    assert checked >= 5, f"only {checked} probes exercised the invariant"


def _modelled_dep(dep: str) -> bool:
    """A repo-relative path or directory prefix, with at most one leading `!`.

    A whitelist, not a blacklist of known-bad shapes: anything else silently
    never matches matches_source_dependency, and the whole point is to catch
    the syntax nobody has thought of yet."""
    body = dep.removeprefix("!")
    if not body or body.startswith("/"):
        return False
    segments = body.rstrip("/").split("/")
    return all(re.fullmatch(r"[\w.-]+", s) and s not in (".", "..") for s in segments)


def test_every_declared_dep_uses_a_syntax_we_model(state):
    """The dep model understands a plain path, a directory prefix and a `!`
    negation. Anything else never matches, quietly, which is how a batch of
    kv_transfer exclusions sat unmodelled.

    `!!x` matters as much as a glob and cuts the other way: we strip every `!`
    and read a full negation, skipping the step, while a generator stripping
    one gets a dep that matches nothing and runs it."""
    deps = {
        dep
        for p in state.pipelines
        for s in p.steps
        for dep in s.source_file_dependencies or []
    }
    assert len(deps) > 100, f"only {len(deps)} deps parsed; the assert below is thin"
    unmodelled = sorted(d for d in deps if not _modelled_dep(d))
    assert not unmodelled, unmodelled


def test_negated_dep_carves_out_of_a_broader_positive(state):
    """A `!` entry is per-step, not per-entry: testing entries individually
    reads it as a dep that simply never matches, so the step fires anyway."""
    from ci_selector.codemap.claim import step_declares

    deps = ["vllm/", "!vllm/distributed/kv_transfer/"]
    assert step_declares(deps, "vllm/config/model.py")
    assert not step_declares(deps, "vllm/distributed/kv_transfer/connector.py")
    assert not step_declares(["!vllm/"], "vllm/config/model.py")

    live = [
        s
        for p in state.pipelines
        for s in p.steps
        if any(d.startswith("!") for d in s.source_file_dependencies or [])
    ]
    assert live, "no negated deps in the live config any more: specimen drifted"


def test_catch_all_declarers_omitted_on_graph_known_leaf(state, declared_deps_on):
    """On a graph-known leaf the graph is authoritative, so a step declaring only
    bare `vllm/` is omitted (graph closure plus SPECIFIC declarers, not CI blanket)."""
    from ci_selector.codemap.classify import _source_dep_steps
    from ci_selector.codemap.state import _graph_known

    leaf = "vllm/model_executor/layers/quantization/experts_int8.py"
    assert _graph_known(state, leaf), "specimen drifted (no longer graph-known)"
    full = _source_dep_steps(state, leaf)
    specific = _source_dep_steps(state, leaf, specific_only=True)
    assert specific < full, "specimen has no catch-all declarers to omit"
    sel = select(state, [leaf])
    assert "vllm_ci" not in sel.run_all
    assert any("catch-all-only declarers omitted" in c.detail for c in sel.claims)
    assert (full - specific) - set(sel.selected), (
        "narrowing dropped nothing -- every catch-all declarer also rides the graph"
    )


def test_legacy_test_amd_yaml_selects_nothing(state):
    """test-amd.yaml is in no ci_config job_dirs (retired external AMD
    pipeline); an edit to it must not trigger a conservative run-all."""
    sel = select(state, [".buildkite/test-amd.yaml"])
    assert not sel.run_all, sel.run_all
    non_always = {
        s.step_id
        for p in state.pipelines
        for s in p.steps
        if s.step_id in sel.selected and not s.always_runs
    }
    assert not non_always, sorted(non_always)[:5]


def test_added_conftest_in_new_dir_scoped_but_existing_dir_fails_open(state):
    """The added-conftest rule protects PRE-EXISTING descendant tests: a conftest in a
    brand-new dir has none, so it must not run-all (an existing tree target-covers)."""
    from ci_selector.codemap.classify import _classify
    from ci_selector.codemap.state import DiffContext

    ctx = DiffContext(
        base="x",
        head="y",
        status={
            "tests/brand_new_area/conftest.py": "A",
            "tests/models/language/conftest.py": "A",
        },
    )
    new_dir = _classify(state, "tests/brand_new_area/conftest.py", ctx)
    assert new_dir.rule == "added-conftest", new_dir.rule
    # existing-tree conftest routes by subtree target-coverage, not fail-open
    existing = _classify(state, "tests/models/language/conftest.py", ctx)
    assert existing.rule == "target-coverage", existing.rule
    assert not existing.run_all and existing.step_ids


def test_added_conftest_routes_where_steps_name_only_py_targets(state):
    """The subtree leg has to run BEFORE the target-kind split. Nested under
    the directory-target branch it never fired for a directory whose steps name
    .py files, and an added conftest there got a zero-job claim rather than a
    fail-open."""
    from ci_selector.codemap.classify import _classify
    from ci_selector.codemap.state import DiffContext

    under_evals = {
        t.path
        for p in state.pipelines
        for st in p.targets.values()
        for t in st.targets
        if t.path.startswith("tests/evals/")
    }
    assert under_evals and all(t.endswith(".py") for t in under_evals), (
        "tests/evals gained a directory target, so it no longer proves the leg "
        f"runs before the split: {sorted(under_evals)[:3]}"
    )

    path = "tests/evals/conftest.py"
    ctx = DiffContext(base="x", head="y", status={path: "A"})
    claim = _classify(state, path, ctx)
    assert claim.rule == "target-coverage", claim.rule
    assert claim.step_ids & state.auto_step_ids, claim.detail


def test_added_file_in_claimed_package_routed_by_keys(state):
    """An added file inside a registered model package must inherit the package's
    string-key routing, not fail open."""
    from ci_selector.codemap.classify import _classify
    from ci_selector.codemap.state import DiffContext

    ctx = DiffContext(
        base="x",
        head="y",
        status={"vllm/models/minimax_m3/common/new_kernel_helper.py": "A"},
    )
    claim = _classify(state, "vllm/models/minimax_m3/common/new_kernel_helper.py", ctx)
    assert claim.rule == "added-in-claimed-package", claim.rule
    assert not claim.run_all
    # Coverage may legitimately be empty (nothing at HEAD names the keys);
    # the contract is: routed by keys, never fail-open.
    assert "MiniMaxM3" in claim.detail


def test_added_benchmark_file_selects_nothing(state):
    """A brand-new standalone benchmark no step invokes has nothing to run, so it
    must not fail open the whole pipeline."""
    from ci_selector.codemap.classify import _classify
    from ci_selector.codemap.state import DiffContext

    ctx = DiffContext(
        base="x",
        head="y",
        status={"benchmarks/kernels/bench_totally_new.py": "A"},
    )
    claim = _classify(state, "benchmarks/kernels/bench_totally_new.py", ctx)
    assert claim.rule == "added-benchmark", claim.rule
    assert not claim.run_all and not claim.step_ids


def test_world_policy_members_are_load_bearing_or_provably_not(state):
    """The world rule is analyzer policy with no CI floor under it, so every
    member has to earn its breadth. Today the one member would get the same
    breadth from the terminal fail-open anyway. That is checked rather than
    assumed, because the day it stops being true the entry starts costing a
    run-all nothing else asked for."""
    from ci_selector.codemap.claim import EXTRA_WORLD_FILES, classify_world

    assert EXTRA_WORLD_FILES, "no world-policy members left; the mechanism is dead"
    names = {p.config.name for p in state.pipelines}
    for path in EXTRA_WORLD_FILES:
        sel = select(state, [path])
        assert set(sel.run_all) == names
        claim = next(c for c in sel.claims if c.rule == "world")
        assert claim.run_all == names, "policy world is all-or-nothing"

    with_rule = {p: set(select(state, [p]).selected) for p in EXTRA_WORLD_FILES}
    monkeyed = classify_world.__globals__
    saved = monkeyed["EXTRA_WORLD_FILES"]
    monkeyed["EXTRA_WORLD_FILES"] = ()
    try:
        without = {p: set(select(state, [p]).selected) for p in EXTRA_WORLD_FILES}
    finally:
        monkeyed["EXTRA_WORLD_FILES"] = saved
    redundant = [p for p in EXTRA_WORLD_FILES if with_rule[p] == without[p]]
    assert set(redundant) == set(EXTRA_WORLD_FILES), (
        "a world-policy member now selects more than the rules below it would: "
        f"{sorted(set(EXTRA_WORLD_FILES) - set(redundant))}. Justify the "
        "breadth or drop the entry."
    )


# Files the graph rule routes by import closure, where no earlier rule may
# narrow the answer. Entries must pass `_closure_hub_guards` and sit below the
# hub gate, since a mirror-holding file above it is colocation's to narrow.
# Kept out on purpose: `vllm/__init__.py`, a docker image input whose steps
# arrive whatever claim wins; and `tests/vllm_test_utils/setup.py`, which
# reaches `rule="graph"` through _nothing_auto_runs's command-text grep.
CLOSURE_HUBS = ("vllm/v1/attention/ops/paged_attn.py", "vllm/cute_utils/cvt.py")

# Files where `colocated-tests` answers narrower than the closure ON PURPOSE:
# cycle members and, since the hub arm landed, non-cycle files past the size
# gate. They cannot carry the invariant above, so they are pinned separately
# with the rule switched off on both ends.
COLOCATED_HUBS = (
    "vllm/envs.py",
    "vllm/lora/layers/base.py",
    "vllm/compilation/decorators.py",
    "vllm/model_executor/layers/mla.py",
)


def _closure_hub_guards(state, path):
    """Why no earlier rule can narrow `path`, asserted rather than trusted.

    Each guard rules out one narrowing rule, so a fixture that stops qualifying
    fails here with the reason instead of quietly weakening the test.
    """
    from ci_selector.codemap import hardware
    from ci_selector.codemap.classify import _classify_graph

    assert path not in state.full.import_cycle().reach_blind, (
        f"{path} joined the import cycle, so `colocated-tests` now answers it "
        "and narrowing is deliberate. Move it to COLOCATED_HUBS."
    )
    assert hardware.exclusive_family_of_path(path) is None, (
        f"{path} became hardware-exclusive, so `no-hardware` may legitimately "
        "answer it with nothing. Drop it; see test_tpu_platform_no_hardware_rule."
    )
    claim = _classify_graph(state, path)
    assert claim is not None and claim.rule == "graph", (
        f"{path}: expected the graph rule, got {claim.rule if claim else None!r}. "
        "If it is 'colocated-tests' the file crossed the hub gate; move it to "
        "COLOCATED_HUBS."
    )
    # The closure branch specifically. _nothing_auto_runs also returns
    # rule="graph" from a command-text grep, which would make the assertion
    # below a lie about closures.
    assert claim.test_files & state.invoked, (
        f"{path} no longer routes through the test closure; its claim came "
        "from _nothing_auto_runs. Pick another hub."
    )
    return claim


def test_no_rule_above_graph_swallows_a_closure_routed_hub(state):
    """A file the graph rule routes by closure must never be answered more
    narrowly by a rule that runs earlier in the chain.

    What the hand list got wrong and why it went. No rule above graph fires on
    these fixtures today, so this is a canary against a future one.

    The floor is a residual, not a magnitude. The test went partly vacuous when
    `colocated-tests` shipped and began answering a fixture narrowly, and the
    old `assert checked` could not see it because it counted claims rather than
    what they proved.
    """
    from ci_selector.codemap.classify import _source_dep_steps

    always_run = {s.step_id for p in state.pipelines for s in p.steps if s.always_runs}
    for path in CLOSURE_HUBS:
        claim = _closure_hub_guards(state, path)
        sel = select(state, [path])
        assert not sel.run_all, f"{path} escalated instead of routing"
        want = claim.step_ids & state.auto_step_ids
        # Steps selected regardless of which claim won, so containing them
        # proves nothing about rule ordering.
        free = (
            state.artifacts.steps_for_input(path)
            | _source_dep_steps(state, path, specific_only=True)
            | always_run
        ) & state.auto_step_ids
        residual = want - free
        assert residual, (
            f"{path} proves nothing about rule ordering: all {len(want)} of its "
            "closure steps are supplied by the image union, its declarers or the "
            "always-run floor. Replace the fixture."
        )
        missing = residual - set(sel.selected)
        assert not missing, (
            f"{path}: {len(missing)} steps the import closure reaches are not "
            f"selected, e.g. {sorted(missing)[:3]}"
        )


def test_colocation_narrows_a_cycle_member_and_only_colocation_does(state):
    """The other half of the invariant above, for files it cannot cover.

    A cycle member is narrowed on purpose, so it gets two weaker pins:
    colocation is what narrows it, and with the rule off the closure invariant
    holds again. If colocation stops firing on one, the first assertion says so
    rather than the path quietly dropping out of every check.
    """
    from ci_selector.codemap.classify import _classify_graph

    for path in COLOCATED_HUBS:
        claim = _classify_graph(state, path)
        assert claim is not None and claim.rule == "colocated-tests", (
            f"{path} is no longer answered by colocation (got "
            f"{claim.rule if claim else None!r}). Move it to CLOSURE_HUBS."
        )
        with_reach = _select_without_colocation(state, path)
        assert not with_reach.run_all, f"{path} escalated with colocation off"
        reach_claim = _classify_graph_without_colocation(state, path)
        want = reach_claim.step_ids & state.auto_step_ids
        missing = want - set(with_reach.selected)
        assert not missing, (
            f"{path}: with colocation off, {len(missing)} closure steps are "
            f"unselected, e.g. {sorted(missing)[:3]}"
        )


def test_only_the_oracle_may_read_run_all_patterns():
    """The deletion, pinned structurally as well as behaviourally.

    `run_all_patterns` is vLLM's hand-maintained "run everything" list, and the
    point of the analyzer is to derive that rather than inherit it. Only the
    field, the parser that fills it, `validate/`, and one sanctioned read may
    touch it: the escalation branch in `_classify_buildkite`, which uses the
    pattern only to escalate a `.buildkite` file to its pipeline's run-all,
    never to route or drop. Any other read makes the comparison measure
    nothing.

    Scanned as attribute access through the AST and not as text, so the
    docstrings explaining the deletion do not trip it.
    """
    import ast

    root = Path(__file__).resolve().parents[1] / "ci_selector"
    banned = {"run_all_patterns", "run_all_exclude_patterns"}
    allowed = {"codemap/pipeline/step.py", "codemap/pipeline/buildkite.py"}
    offenders = []
    classify_reads = 0
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel in allowed or rel.startswith("validate/"):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            name = (
                getattr(node, "attr", None)
                or getattr(node, "id", None)
                # keyword= and def f(arg=...) spellings too: a re-introduction
                # written either way would otherwise slip past a guard whose
                # docstring promises the deletion is pinned structurally.
                or getattr(node, "arg", None)
            )
            if name in banned:
                if rel == "codemap/classify.py":
                    classify_reads += 1
                    continue
                offenders.append(f"{rel}:{node.lineno}")
    assert classify_reads == 2, (
        "classify.py's sanctioned pattern read is exactly the escalation "
        f"branch's pair (run_all + exclude); found {classify_reads} reads. "
        "A new read is routing off the hand list -- the one thing this "
        "guard exists to prevent."
    )
    assert not offenders, (
        "the selector is reading the hand-maintained run-all list again: "
        f"{offenders}. Removing it measured BETTER (32 missed failures across "
        "296 PRs against 37 with it); re-run the crosscheck before trusting "
        "the intuition that put it back."
    )
    # Detection floor: the oracle must still read it, or this test is scanning
    # for something nothing in the tree could contain.
    oracle = ast.parse((root / "validate/generator_replica.py").read_text())
    assert any(getattr(n, "attr", None) in banned for n in ast.walk(oracle)), (
        "the baseline oracle stopped reading it; the comparison is now circular"
    )


def test_world_ignores_the_run_all_patterns_in_the_config():
    """The deletion, pinned behaviourally.

    Built off synthetic configs so an upstream edit cannot rot it: `beta`
    lists the path in its `run_all_patterns` and `gamma` does not, and the two
    must be treated identically. A reader who re-introduces the read to "fix"
    a missed escalation fails here, which is the moment to re-run the
    measurement rather than trust the intuition.
    """
    from ci_selector.codemap.claim import EXTRA_WORLD_FILES, classify_world
    from ci_selector.codemap.pipeline.step import PipelineConfig

    assert EXTRA_WORLD_FILES, "no world-policy members left; the mechanism is dead"
    path = EXTRA_WORLD_FILES[0]
    beta = PipelineConfig("beta", "beta.yaml", [], [path], [])
    gamma = PipelineConfig("gamma", "gamma.yaml", [], [], [])

    assert classify_world(path, [beta, gamma]).run_all == {"beta", "gamma"}
    assert classify_world(path, [gamma]).run_all == {"gamma"}

    unlisted = "vllm/config/__init__.py"
    matching = PipelineConfig("delta", "delta.yaml", [], [unlisted], [])
    assert classify_world(unlisted, [matching]) is None, (
        "a run_all_patterns entry made a file world again"
    )


def test_rename_contributes_both_sides():
    files = [DiffFile("R", "vllm/new_name.py", old_path="vllm/old_name.py")]
    assert changed_paths(files) == ["vllm/new_name.py", "vllm/old_name.py"]


def test_optional_steps_never_auto_selected(state):
    sel = select(state, ["vllm/config/__init__.py"])
    optional_ids = {
        s.step_id for p in state.pipelines for s in p.steps if s.manual_only
    }
    assert not optional_ids & set(sel.selected)


# --- requirements-file routing (the #50522 fix) ---


def test_requirements_tpu_is_scoped_not_run_all(state):
    """#50522: requirements/tpu.txt failed open to run-all; it must now scope to the ray
    dep-check step (real TPU jobs are an external, unmodeled pipeline)."""
    sel = select(state, ["requirements/tpu.txt"])
    assert not sel.run_all
    assert any(c.rule == "requirements" for c in sel.claims)
    assert any("ray-dependency" in s for s in sel.selected)


def test_requirements_cpu_unions_device_family(state):
    """A device-specific requirements file unions the declaring step(s) with its device
    family's jobs; the declaring step alone drops the CPU suites (under-selection)."""
    sel = select(state, ["requirements/cpu.txt"])
    assert not sel.run_all
    cpu_family = state.family_steps("cpu")
    assert cpu_family and cpu_family <= set(sel.selected)
    assert any("ray-dependency" in s for s in sel.selected)


def test_requirements_test_xpu_unions_xpu_family(state):
    """The cpu case above with manual-only steps in the family, which land in
    manual_hits rather than selected."""
    sel = select(state, ["requirements/test/xpu.txt"])
    assert not sel.run_all
    xpu_family = state.family_steps("xpu")
    # some XPU steps are manual_only -> manual_hits, not selected
    assert xpu_family and xpu_family <= set(sel.selected) | set(sel.manual_hits)


def test_requirements_nightly_torch_hits_declaring_step(state):
    """test/nightly-torch.txt is named in a step's source_file_dependencies, a file
    the import graph can't reach."""
    sel = select(state, ["requirements/test/nightly-torch.txt"])
    assert not sel.run_all
    assert any("nightly-dependency" in s for s in sel.selected)


def test_requirements_unmapped_is_ray_only_not_run_all(state):
    """lint.txt has no device token and no declaring step except ray's requirements/
    blanket, so just the ray dep-check, never run-all."""
    sel = select(state, ["requirements/lint.txt"])
    assert not sel.run_all
    assert any("ray-dependency" in s for s in sel.selected)


def test_requirements_common_is_broad_without_escalating(state):
    """The broad requirements files used to be intercepted by the run_all
    match before the requirements rule ever saw them. Now they route: broad
    because every image installs them, but as named steps rather than a
    pipeline sweep, so the record can still speak about each one."""
    sel = select(state, ["requirements/common.txt"])
    assert not sel.run_all
    narrow = select(state, ["requirements/test/rocm.txt"])
    assert len(sel.selected) > len(narrow.selected)


def test_ray_compat_blanket_still_declares_requirements(state):
    """The narrow-requirements no-run-all property rides on SOME auto-run step declaring
    a requirements/ blanket (ray_compat today); fail loudly if that disappears."""
    from ci_selector.codemap.claim import deps_match

    declarers = [
        s.step_id
        for p in state.pipelines
        for s in p.steps
        if not s.manual_only
        and deps_match(s.source_file_dependencies, ["requirements/_probe.txt"])
    ]
    assert declarers, (
        "no auto-run step declares a requirements/ dep; narrow "
        "requirements files would fail open to run-all"
    )


# --- generator-replica units (semantics mirror the ci-infra generator) ---


def test_dep_matching_is_directory_boundary():
    assert matches_source_dependency("vllm/lora", "vllm/lora/x.py")
    assert matches_source_dependency("vllm/lora/", "vllm/lora")
    assert not matches_source_dependency("vllm/lora", "vllm/lora_extra/x.py")
    assert matches_source_dependency("setup.py", "setup.py")
    assert not matches_source_dependency("setup.py", "setup.py.bak/x")


def test_today_yaml_self_dep_triggers_steps(state):
    pipelines = [(p.config, p.steps) for p in state.pipelines]
    sel = today_select(pipelines, [".buildkite/test_areas/lora.yaml"])
    assert "vllm_ci:lora" in sel.selected["vllm_ci"]


def test_today_optional_blocked_even_under_run_all(state):
    pipelines = [(p.config, p.steps) for p in state.pipelines]
    sel = today_select(pipelines, ["setup.py"])
    assert sel.run_all["vllm_ci"]
    optional_ids = {
        s.step_id
        for p in state.pipelines
        for s in p.steps
        if s.optional and not s.always_runs
    }
    assert not optional_ids & sel.selected["vllm_ci"]


def test_today_docs_only(state):
    pipelines = [(p.config, p.steps) for p in state.pipelines]
    sel = today_select(pipelines, ["docs/index.md", "mkdocs.yaml"])
    assert sel.docs_only and not sel.selected


def test_llm_entrypoint_selects_cudagraph_job(state):
    """MODULE_ATTRS colon fix: `from vllm import LLM` edges make an LLM
    entrypoint diff reach the cudagraph suites that boot LLM()."""
    sel = select(state, ["vllm/entrypoints/llm.py"])
    assert "vllm_ci" not in sel.run_all
    assert any("cudagraph" in s for s in _selected(sel))


def test_plugin_package_file_selects_plugin_step(state, vllm_repo):
    """Entry-point-loaded plugin packages route to the steps that
    pip-install them instead of silently selecting nothing."""
    yaml_text = (vllm_repo / ".buildkite/test_areas/plugins.yaml").read_text()
    assert "plugins/vllm_add_dummy_model" in yaml_text, (
        "plugins.yaml no longer references the package: update specimen"
    )
    sel = select(
        state,
        ["tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_llava.py"],
    )
    assert not sel.run_all
    assert any("plugin" in s for s in _selected(sel))


def test_orphan_test_file_selects_nothing(state):
    """An orphan test file no step declares runs nowhere: zero jobs with a claim, not
    run-all. A declared orphan differs (see
    test_no_classifier_drops_declared_source_deps); specimen derived from HEAD."""
    from ci_selector.codemap.classify import _source_dep_steps

    orphans = [
        f
        for f in sorted(set(state.catalog) - state.invoked - state.legacy_invoked)
        if not _source_dep_steps(state, f)
    ]
    assert orphans, "no undeclared orphan test files at HEAD to exercise the rule"
    path = orphans[0]
    sel = select(state, [path])
    assert not sel.run_all
    assert "orphan" in sel.claims[0].detail
    always = {s.step_id for p in state.pipelines for s in p.steps if s.always_runs}
    assert set(sel.selected) <= always


def test_joined_fixture_path_routes_to_consumers(state):
    """os.path.join-built fixture paths (example_prompts = join(_TEST_DIR, "prompts",
    "example.txt")) carry no slash-bearing literal, so the asset parser missed them;
    join-literal synthesis restores the edge to consuming tests."""
    fixture = "tests/prompts/example.txt"
    assert fixture in state.full.graph.reverse, "fixture drift: no asset edge"
    sel = select(state, [fixture])
    assert not sel.run_all
    assert "vllm_ci:basic-correctness" in sel.selected


def test_uncovered_member_inherits_its_registry(state):
    """A registered member whose own closure auto-runs nothing inherits its
    registry's coverage instead of running everything, and the steps that name
    it only by key still ride along as manual hits."""
    specimen = "vllm/model_executor/layers/quantization/fbgemm_fp8.py"
    closure = state.full.graph.reverse_closure({specimen})
    assert not [f for f in closure if f in state.invoked], (
        "the specimen grew auto-run coverage: pick a new one"
    )
    tables = state.full.table_of().get(specimen)
    assert tables, "specimen no longer registry-named: pick a new one"
    sel = select(state, [specimen])
    assert not sel.run_all
    assert "inheriting the coverage of registry" in sel.claims[0].detail
    assert any("kernels-fp8-moe" in s for s in sel.manual_hits)


def test_package_init_routes_to_package_steps(state):
    """Ancestor __init__.py auto-load edges: a test-package __init__ edit
    runs the package's own suites, neither run-all nor nothing."""
    sel = select(state, ["tests/basic_correctness/__init__.py"])
    assert not sel.run_all
    assert any("basic-correctness" in s for s in _selected(sel))


def test_boot_edge_reaches_conftest_server_suites(state, declared_deps_on):
    """A boot-edge diff must select suites whose engine boot happens in a
    conftest server fixture, not just direct entrypoint importers.

    DECLARED-DEPS ONLY: rides a declaration the derived default gives up. A
    derived route is queued in todo."""
    sel = select(state, ["vllm/v1/worker/gpu_model_runner.py"])
    assert "vllm_ci" not in sel.run_all
    assert any("metrics-tracing" in s for s in _selected(sel))


def test_tpu_platform_no_hardware_rule(state):
    """tpu has zero live CI steps and tpu.py is import-isolated: nothing to run."""
    sel = select(state, ["vllm/platforms/tpu.py"])
    assert sel.claims[0].rule == "no-hardware"
    assert not sel.run_all
    always = {s.step_id for p in state.pipelines for s in p.steps if s.always_runs}
    assert set(sel.selected) <= always


def test_no_hardware_rule_yields_to_live_family_steps(state):
    """Fallback: the moment a tpu-family device exists, the rule stops
    firing and family routing selects the new step."""
    import dataclasses

    p0 = state.pipelines[0]
    synth = dataclasses.replace(
        p0.steps[0],
        key="synthetic-tpu",
        label="TPU synth",
        device="tpu_v6",
        optional=False,
        mirror_hw=None,
    )
    from ci_selector.codemap.state import PipelineData

    p0b = PipelineData(p0.config, p0.steps + [synth], p0.targets)
    st2 = dataclasses.replace(state, pipelines=[p0b] + state.pipelines[1:])
    sel = select(st2, ["vllm/platforms/tpu.py"])
    assert sel.claims[0].rule != "no-hardware"
    assert synth.step_id in sel.selected


def test_xpu_scoping_floor_and_ceiling(state):
    """xpu exclusion: no cuda-device step may run (ceiling); the xpu steps covering its
    closure must run (floor, the bamba lesson)."""
    sel = select(state, ["vllm/platforms/xpu.py"])
    assert not sel.run_all
    devices = {
        s.device
        for p in state.pipelines
        for s in p.steps
        if s.step_id in sel.selected and s.device
    }
    assert not any(
        d.startswith(("h100", "h200", "b200", "a100", "gh200")) for d in devices
    ), devices
    assert "intel_gpu" in devices, devices


def test_exclusivity_violation_disables_scoping(state):
    """Fallback: a cross-family importer disables the exclusion and cuda
    steps come back (fail-open in production, not just in pytest)."""
    import dataclasses

    st2 = dataclasses.replace(
        state,
        exclusive_disabled=state.exclusive_disabled | {"vllm/platforms/xpu.py"},
    )
    sel = select(st2, ["vllm/platforms/xpu.py"])
    devices = {
        s.device
        for p in state.pipelines
        for s in p.steps
        if s.step_id in sel.selected and s.device
    }
    assert any(d.startswith(("h100", "b200")) for d in devices), devices


def test_live_violation_member_reaches_cuda_jobs(state):
    """platforms/cpu.py has real cuda-run module-level importers
    (test_attention_selector et al.); its exclusion must be disabled."""
    assert "vllm/platforms/cpu.py" in state.exclusive_disabled
    sel = select(state, ["vllm/platforms/cpu.py"])
    devices = {
        s.device
        for p in state.pipelines
        for s in p.steps
        if s.step_id in sel.selected and s.device
    }
    assert any(d.startswith("h100") for d in devices), devices


def test_inert_ci_trees_select_nothing(state):
    """Trees no live pipeline consumes (external nightly perf, deprecated
    stub, native SLURM) select zero jobs with a note instead of run-all x3."""
    for path in (
        ".buildkite/performance-benchmarks/tests/serving-tests.json",
        ".buildkite/test-pipeline.yaml",
        ".buildkite/amd-disagg/pipeline-disagg.yaml",
    ):
        sel = select(state, [path])
        assert sel.claims[0].rule == "inert-ci", path
        assert not sel.run_all


def test_inert_readme_does_not_escalate_mixed_diff(state):
    sel = select(
        state,
        [".buildkite/performance-benchmarks/README.md", "vllm/logger.py"],
    )
    assert not sel.run_all


def test_inert_prefixes_untouched_by_live_steps(state):
    """Soundness premise for the inert route: no live step's source, scripts, or data
    files reach the trees. A hit means: drop that INERT_CI_PREFIXES entry."""
    from ci_selector.handwritten import INERT_CI_PREFIXES

    touched = set()
    for p in state.pipelines:
        for s in p.steps:
            if s.source_file.startswith(INERT_CI_PREFIXES):
                touched.add(s.source_file)
        for st in p.targets.values():
            for ref in (*st.scripts_seen, *st.data_files):
                if ref.startswith(INERT_CI_PREFIXES):
                    touched.add(ref)
    assert not touched, touched


def test_legacy_file_that_rejoins_claims_its_steps(state):
    """Fallback: if test-amd.yaml ever rejoins job_dirs, its steps load and
    the live source_file rule claims it BEFORE the legacy zero-claim."""
    import dataclasses

    from ci_selector.codemap.classify import _classify_buildkite
    from ci_selector.codemap.state import PipelineData

    p0 = state.pipelines[0]
    synth = dataclasses.replace(
        p0.steps[0],
        key="legacy-back",
        label="Legacy back",
        source_file=".buildkite/test-amd.yaml",
        mirror_hw=None,
    )
    p0b = PipelineData(p0.config, p0.steps + [synth], p0.targets)
    st2 = dataclasses.replace(state, pipelines=[p0b] + state.pipelines[1:])
    claim = _classify_buildkite(
        st2, ".buildkite/test-amd.yaml", [p.config for p in st2.pipelines]
    )
    assert claim.rule == "buildkite"
    assert synth.step_id in claim.step_ids


def test_inert_tree_referenced_by_live_step_claims_steps(state):
    """Fallback: a live step reaching into an inert tree wins over the inert zero-claim
    (rule order), independent of the soundness-premise test."""
    import dataclasses

    from ci_selector.codemap.classify import _classify_buildkite
    from ci_selector.codemap.pipeline.targets import StepTargets
    from ci_selector.codemap.state import PipelineData

    p0 = state.pipelines[0]
    sid = p0.steps[0].step_id
    st = StepTargets(step_id=sid)
    st.scripts_seen.append(".buildkite/performance-benchmarks/run.sh")
    targets2 = dict(p0.targets)
    targets2[sid] = st
    p0b = PipelineData(p0.config, p0.steps, targets2)
    st2 = dataclasses.replace(state, pipelines=[p0b] + state.pipelines[1:])
    claim = _classify_buildkite(
        st2,
        ".buildkite/performance-benchmarks/run.sh",
        [p.config for p in st2.pipelines],
    )
    assert claim.rule == "buildkite"
    assert sid in claim.step_ids


def test_class_table_module_inherits_the_table_coverage(state):
    """medusa.py is claimed by the class-table parser but its consumers are HF
    checkpoints, so no test names MedusaConfig and its own closure is empty.
    It inherits the coverage of configs/__init__.py, whose consumers are what
    can load it."""
    assert "MedusaConfig" in state.full.factories.class_table_entries
    specimen = "vllm/transformers_utils/configs/medusa.py"
    table = "vllm/transformers_utils/configs/__init__.py"
    assert state.full.table_of().get(specimen) == frozenset({table})
    sel = select(state, [specimen])
    assert not sel.run_all
    assert sel.claims[0].rule == "graph"
    assert table in sel.claims[0].detail
    table_sel = select(state, [table])
    always = {s.step_id for p in state.pipelines for s in p.steps if s.always_runs}
    assert set(sel.selected) <= set(table_sel.selected) | always | set(sel.manual_hits)


@pytest.mark.drift
def test_the_registry_table_map_holds_its_anchors(state):
    """table_of is the one hop from a registered member back to its table.
    Anchors pin one entry per parser family; the size floor catches the map
    silently emptying, which would re-open the run-all hole it closed."""
    tof = state.full.table_of()
    anchors = {
        "vllm/transformers_utils/configs/afmoe.py": (
            "vllm/transformers_utils/configs/__init__.py"
        ),
        "vllm/v1/attention/backends/mla/cpu_mla.py": (
            "vllm/v1/attention/backends/registry.py"
        ),
        "vllm/model_executor/layers/quantization/fbgemm_fp8.py": (
            "vllm/model_executor/layers/quantization/__init__.py"
        ),
    }
    missing = [
        f"{target} -> {table}"
        for target, table in anchors.items()
        if table not in tof.get(target, frozenset())
    ]
    assert not missing, drift_message(
        f"registry table map lost anchors: {missing}",
        "their empty-closure members fall back to run-everything",
        "the parser sites populating table_of in codemap/graph/factories.py",
        "FullGraph.table_of() in codemap/graph/build.py",
    )
    assert len(tof) >= 300, drift_message(
        f"registry table map collapsed to {len(tof)} entries",
        "empty-closure registered members fall back to run-everything",
        "the parser sites populating table_of in codemap/graph/factories.py",
    )


def test_zero_closure_specimen_fails_open(state):
    """Empty-closure direction: a vllm/ file whose coverage auto-runs nowhere must
    run-all, not silently select zero. Derived from HEAD (import-free source)."""
    rev = state.full.graph.reverse
    for file in state.full.index.file_to_module:
        if not (file.startswith("vllm/") and file.endswith(".py")):
            continue
        if rev.get(file):
            continue
        sel = select(state, [file])
        if "empty-closure direction" in sel.run_all.get("vllm_ci", ""):
            return
    raise AssertionError("no import-free empty-closure specimen at HEAD")


def test_config_file_edit_scopes_to_one_pipeline(state):
    sel = select(state, [".buildkite/ci_config_intel.yaml"])
    assert set(sel.run_all) == {"vllm_intel_ci"}


def test_referenced_ci_script_selects_its_steps(state):
    sel = select(state, [".buildkite/scripts/hardware_ci/run-cpu-test.sh"])
    assert not sel.run_all
    always = {s.step_id for p in state.pipelines for s in p.steps if s.always_runs}
    assert set(sel.selected) - always


def test_lm_eval_harness_routes_to_its_steps_not_run_all(state, vllm_repo):
    """lm-eval-harness files are not unrecognized CI infra:
    `test_lm_eval_correctness.py` is a pytest TARGET of the lm_eval steps,
    reached through their `working_dir`, and the config yamls sit beside it.
    The referencing leg read only `scripts_seen` and `data_files`, so all of it
    fell to the terminal run-all.
    """
    files = sorted(
        p.relative_to(vllm_repo).as_posix()
        for p in (vllm_repo / ".buildkite/lm-eval-harness").rglob("*")
        if p.is_file()
    )
    assert len(files) > 20, "fixture drift: lm-eval-harness shrank"
    for path in files:
        sel = select(state, [path])
        assert not sel.run_all, f"{path} still escalates"
        assert len(sel.selected) < 60, f"{path} selects {len(sel.selected)}"


def test_a_buildkite_script_beside_the_pipeline_yaml_is_not_owned_by_it(state):
    """The floor for the leg above, and why `.buildkite/` is in
    `_ROOT_PREFIXES`. Scripts live directly in `.buildkite/`, so without that
    entry co-location would make every pipeline yaml a dependency of whatever
    step runs one, and the legacy and inert zero-claims below would never be
    reached."""
    for path in (".buildkite/test-amd.yaml", ".buildkite/test-pipeline.yaml"):
        sel = select(state, [path])
        non_always = {
            s.step_id
            for p in state.pipelines
            for s in p.steps
            if s.step_id in sel.selected and not s.always_runs
        }
        assert not non_always, f"{path}: {sorted(non_always)[:5]}"


def test_a_docker_file_the_build_dag_knows_does_not_run_everything(state):
    """The terminal fail-open used to set run_all, and the widening rules bail
    on run_all, so it preempted an image-input answer that existed."""
    for path in (
        "docker/versions.json",
        "docker/Dockerfile.ppc64le",
        "docker/Dockerfile.s390x",
        "docker/Dockerfile.tpu",
        "docker/entrypoints/vllm-nonroot-entrypoint.sh",
    ):
        sel = select(state, [path])
        assert not sel.run_all, f"{path} still escalates"
        assert sel.selected, f"{path} now selects nothing, which is the unsafe way"


def test_an_unclaimed_unreferenced_file_is_inert(state):
    """A file no derived surface reaches -- no step target, key, docker COPY,
    specific declarer, or command text names it -- selects nothing beyond the
    floor, because no job can execute it. Files under package roots keep the
    fail-open (see test_brand_new_file_fails_open)."""
    from ci_selector.codemap.classify import _classify

    path = "some_unclaimed_top_level_thing.conf"
    claim = _classify(state, path, None)
    assert claim.rule == "inert"
    assert not claim.run_all and not claim.step_ids and not claim.test_files


def test_unrecognized_ci_file_runs_all_pipelines(state):
    """Root-level only: a file beside the generator configs may be generator
    input the way .pipeline_gen_v2 is, so the catch-all stands there. Subdir
    files rest at the floor (see the inert tests)."""
    sel = select(state, [".buildkite/some_new_infra_thing.xyz"])
    assert set(sel.run_all) == {p.config.name for p in state.pipelines}


def test_an_added_step_yaml_escalates_its_own_pipeline(state):
    """A step file the base did not load escalates the pipeline that will load
    it, so a new-suite PR still runs its new steps."""
    sel = select(state, [".buildkite/test_areas/some_brand_new_area.yaml"])
    assert set(sel.run_all) == {"vllm_ci"}


def test_an_added_step_yaml_escalates_every_pipeline_sharing_its_dir(state):
    """hardware_tests/ is a job_dir of two pipelines, so an added suite there
    must escalate both -- escalating only the first would leave the ROCm steps
    it also creates unrun."""
    shared = {
        c.config.name
        for c in state.pipelines
        if ".buildkite/hardware_tests" in c.config.job_dirs
    }
    assert len(shared) >= 2, "hardware_tests no longer shared; pick a new dir"
    sel = select(state, [".buildkite/hardware_tests/some_brand_new_suite.yaml"])
    assert set(sel.run_all) == shared


def test_a_generator_pattern_escalates_only_its_pipeline(state):
    """run-amd-test.sh matches ci_config_rocm.yaml's run_all_patterns, so it
    escalates the ROCm pipeline and nothing else, matching CI."""
    sel = select(state, [".buildkite/scripts/hardware_ci/run-amd-test.sh"])
    assert set(sel.run_all) == {"vllm_rocm_ci"}


def test_a_dockerfile_executed_ci_script_rests_at_the_floor(state):
    """check-wheel-size.py runs inside the image build, and the builds are
    always-run, so the floor is its complete test."""
    sel = select(state, [".buildkite/check-wheel-size.py"])
    claim = sel.claims[0]
    assert claim.rule == "inert" and not sel.run_all
    assert not _non_always(sel, state)


def test_a_consumerless_ci_script_rests_at_the_floor(state):
    """A .buildkite subdir script no surface reaches selects the floor only."""
    sel = select(state, [".buildkite/scripts/rerun-test.sh"])
    claim = sel.claims[0]
    assert claim.rule == "inert" and not sel.run_all
    assert not _non_always(sel, state)


def test_the_abi_audit_script_routes_to_its_step(state):
    """check-torch-abi.py is the torch-stable-abi-audit step's own script, so
    it routes there instead of running everything."""
    sel = select(state, [".buildkite/check-torch-abi.py"])
    assert not sel.run_all
    assert any("torch-stable-abi-audit" in s for s in sel.selected)


def test_added_test_without_owning_target_selects_the_floor(state):
    """An added test whose directory no step sweeps selects the floor: no job
    can run it. A step yaml added with it escalates the pipeline through the
    job_dirs guard."""
    from ci_selector.codemap.classify import _classify
    from ci_selector.codemap.state import DiffContext

    path = "tests/brand_new_area/test_x.py"
    ctx = DiffContext(base="b", head="h", status={path: "A"})
    claim = _classify(state, path, ctx)
    assert claim.rule == "added-test"
    assert not claim.run_all
    assert not claim.step_ids and not claim.test_files
    assert "no step sweeps" in claim.detail


def test_added_test_under_a_swept_directory_routes_to_its_steps(state):
    """An added test under a directory a live step's command targets runs in
    exactly those steps, existing or not -- the sweep is string-based."""
    from ci_selector.codemap.classify import _classify
    from ci_selector.codemap.state import DiffContext

    path = "tests/basic_correctness/test_shiny_new_thing.py"
    ctx = DiffContext(base="b", head="h", status={path: "A"})
    claim = _classify(state, path, ctx)
    assert claim.rule == "added-test"
    assert claim.step_ids and not claim.run_all


def test_added_vllm_file_with_no_head_coverage_fails_open(state):
    """An added vllm/ .py with no registered keys and no head-side importer
    reaching live coverage has unknown reach and fails open. Inheriting the
    sibling directory's closure over-selects (its files sit in the import
    cycle); diff-side import parsing is the queued sound route."""
    from ci_selector.codemap.classify import _classify
    from ci_selector.codemap.state import DiffContext

    path = "vllm/v1/worker/gpu/sample/brand_new_helper.py"
    ctx = DiffContext(base="b", head="h", status={path: "A"})
    claim = _classify(state, path, ctx)
    assert claim.rule == "fail-open"
    assert claim.run_all == {p.config.name for p in state.pipelines}


def test_today_amd_native_runtime_branch(state, vllm_repo):
    """The replica's AMD branch: mi-device non-dind steps run on
    run-amd-test.sh diffs; dind or non-amd steps do not."""
    import dataclasses

    from ci_selector.validate.generator_replica import step_should_run

    assert (vllm_repo / ".buildkite/scripts/hardware_ci/run-amd-test.sh").is_file(), (
        "AMD native runtime script moved: update curated dep"
    )
    base = state.pipelines[0].steps[0]
    amd = dataclasses.replace(
        base,
        key="amd-native",
        label="a",
        device="mi300_4",
        dind=False,
        optional=False,
        mirror_hw=None,
    )
    paths = [".buildkite/scripts/hardware_ci/run-amd-test.sh"]
    assert step_should_run(amd, paths, run_all=False)
    assert not step_should_run(
        dataclasses.replace(amd, dind=True), paths, run_all=False
    )
    assert not step_should_run(
        dataclasses.replace(amd, device="h100"), paths, run_all=False
    )
    assert not step_should_run(amd, ["vllm/logger.py"], run_all=False)


def test_unmapped_device_disables_no_hardware_rule(state):
    """The no-hardware rule must never fail closed: an unmappable device means
    family_steps() is incomplete, so the rule disables and tpu.py falls to graph."""
    import dataclasses

    pf = dataclasses.replace(state.preflight, unmapped_devices=frozenset({"v6e-8"}))
    st2 = dataclasses.replace(state, preflight=pf)
    sel = select(st2, ["vllm/platforms/tpu.py"])
    assert sel.claims[0].rule != "no-hardware"


# ---- examples-step recall (demoted members must keep their example steps) --


def test_eagle_proposer_selects_examples_steps(state):
    """The steps run `spec_decode_offline.py --method eagle`: demotion must
    not drop them (the examples-step under-selection regression)."""
    sel = select(state, ["vllm/v1/spec_decode/eagle.py"])
    assert "vllm_ci:examples" in sel.selected


def test_pooling_runner_selects_examples_steps(state):
    """Same regression, different demotion shape: a runner rather than a
    proposer."""
    sel = select(state, ["vllm/v1/worker/gpu/pool/pooling_runner.py"])
    assert "vllm_ci:examples" in sel.selected


def test_eagle_config_selects_examples_steps(state):
    """And again from the config side, which reaches the step by a third
    route."""
    sel = select(state, ["vllm/transformers_utils/configs/eagle.py"])
    assert "vllm_ci:examples" in sel.selected


def test_examples_route_by_workdir_affinity_not_zero_claims(state):
    """Every examples file reaches its tree's steps, never a zero claim.

    The Examples step invokes each example by name, so it declares file
    targets and no directory target and the directory leg cannot fire. Its
    working_dir carries the tree instead. The floor below checks that leg is
    alive, since without it the tree falls back to the declarations.
    """
    from ci_selector.codemap.classify import (
        _classify_testside,
        _steps_targeting,
        _workdir_affinity_steps,
    )

    examples = sorted(
        f.relative_to(state.repo).as_posix()
        for f in (state.repo / "examples").rglob("*")
        if f.is_file()
    )
    assert len(examples) > 100, "examples tree collapsed; refresh the fixture"
    affinity = _workdir_affinity_steps(state, examples[0])
    assert affinity & state.auto_step_ids, (
        "no auto step declares an examples working_dir; the tree would fall "
        "back to the declarations"
    )
    for path in examples[:200]:
        covering = _steps_targeting(state, path)
        assert covering & state.auto_step_ids, f"{path} lost its tree's steps"
        claim = _classify_testside(state, path)
        assert claim is not None and claim.step_ids & state.auto_step_ids


def test_no_examples_file_reaches_the_terminal_fail_open(state):
    """The acceptance bar for widening the testside gate: routing examples/
    must never push one to run-all. A naive PACKAGE_ROOTS change did exactly
    that to 75 of them, which is why the gate moved instead."""
    from ci_selector.codemap.classify import _classify

    examples = sorted(
        f.relative_to(state.repo).as_posix()
        for f in (state.repo / "examples").rglob("*")
        if f.is_file()
    )
    assert len(examples) > 100, "examples tree collapsed; refresh the fixture"
    fail_open = [f for f in examples if _classify(state, f, None).run_all]
    assert not fail_open, (
        f"{len(fail_open)} examples files run everything: {fail_open[:3]}"
    )


# ---- declared-deps routing -------------------------------------------------

RUST_DECLARERS = {
    "vllm_ci::nvidia: (H200 MIG 18GB) Rust Frontend Core Correctness",
    "vllm_ci::nvidia: (L4) Rust Frontend Distributed",
    "vllm_ci::nvidia: (H200 MIG 18GB) Rust Frontend OpenAI Coverage",
    "vllm_ci::nvidia: (H200 MIG 18GB) Rust Frontend Serve/Admin Coverage",
    "vllm_ci::nvidia: (H200 MIG 35GB) Rust Frontend Tool Use",
    "vllm_ci:rust-frontend-cargo-style-clippy",
    "vllm_ci:rust-frontend-cargo-tests",
}


def test_rust_file_routes_to_declared_steps(state):
    """rust/ used to be a rocm-only run_all match, so the whole AMD pipeline
    swept while vllm_ci stayed narrow. Derived it routes on both sides: the
    declaring cargo steps, plus every step running on an image `rust/` is
    built into."""
    sel = select(state, ["rust/frontend/src/lib.rs"])
    assert not sel.run_all
    assert set(sel.selected) >= RUST_DECLARERS


def test_rust_toolchain_routes_to_cargo_steps(state):
    sel = select(state, ["rust-toolchain.toml"])
    assert not sel.run_all
    assert "vllm_ci:rust-frontend-cargo-tests" in sel.selected


def test_cmake_cpu_extension_routes_to_declarers_plus_cpu_family(
    state, declared_deps_on
):
    """Real CI's route (the declaring steps) plus the CPU family whose suites compile
    the extension in-step without declaring it (previously excluded, then run-all)."""
    sel = select(state, ["cmake/cpu_extension.cmake"])
    assert not sel.run_all
    for sid in (
        "vllm_ci:torch-stable-abi-audit",
        "vllm_ci:cpu-kernel-tests",
        "vllm_rocm_ci:cpu-kernel-tests",
    ):
        assert sid in sel.selected, sid


def test_rocm_named_rust_file_selects_rust_steps(state):
    """The basename hardware-token heuristic is a Python/shell convention; a
    rocm-named rust file must keep its declaring h200 steps (.rs is not subject
    to amd-exclusive subtraction)."""
    sel = select(state, ["rust/src/rocm_support.rs"])
    assert not sel.run_all
    assert set(sel.selected) >= RUST_DECLARERS


def test_undeclared_oddball_still_fails_open(state):
    sel = select(state, ["tools/install_gdrcopy.sh"])
    assert set(sel.run_all) == {"vllm_ci", "vllm_intel_ci", "vllm_rocm_ci"}


def test_all_manual_declarers_fall_open(state, declared_deps_on):
    """Direction guard: with no auto-run declarer the rule must not silently
    select nothing. cmake/cpu_extension.cmake is graph-blind and declared only
    by steps this test strips from `auto_step_ids`, so it falls through — and
    since the build map scopes it, the fail-open runs its device family
    instead of everything."""
    import dataclasses

    from ci_selector.codemap.classify import _classify, _source_dep_steps

    path = "cmake/cpu_extension.cmake"
    declarers = _source_dep_steps(state, path)
    assert declarers, "fixture drift: no step declares cmake/cpu_extension.cmake"
    st2 = dataclasses.replace(state, auto_step_ids=state.auto_step_ids - declarers)
    claim = _classify(st2, path, None)
    assert claim.rule == "fail-open" and not claim.run_all
    assert "build-map scoped" in claim.detail
    assert claim.step_ids


def test_a_mapped_build_file_fails_open_scoped_not_run_all(state):
    """The terminal fail-open consults the build map: a {cpu}-mapped cmake
    file runs its device family (~24 steps, CI's own 21 on PR 50219) instead
    of everything, and the step_ids claim leaves the coverage stage armed on
    the rest of the diff, where a run_all claim vetoed every drop PR-wide.

    The no-device steps are deliberately shed — torch-stable-abi-audit
    included: the audited wheel is CUDA-built and the lane-2 containment
    measurement endorsed exactly this class (build-map-endorsed misses)."""
    from ci_selector.codemap.classify import _classify

    audit = "vllm_ci:torch-stable-abi-audit"
    assert audit in state.auto_step_ids, "fixture drift: the audit step moved"
    claim = _classify(state, "cmake/cpu_extension.cmake", None)
    assert claim.rule == "fail-open" and not claim.run_all
    assert "build-map scoped to ['cpu']" in claim.detail
    kept = set(claim.step_ids) & state.auto_step_ids
    assert 0 < len(kept) < 60
    assert audit not in claim.step_ids
    assert not claim.droppable_step_ids


def test_the_scoped_terminal_stands_down_with_the_build_map_off(state, monkeypatch):
    """The knob disarms the scoping, never the selection: off means back to
    running everything, the wider answer."""
    from ci_selector.codemap import build_map
    from ci_selector.codemap.classify import _classify

    monkeypatch.setenv(build_map.ENV_VAR, "off")
    claim = _classify(state, "cmake/cpu_extension.cmake", None)
    assert claim.rule == "fail-open" and claim.run_all


def test_an_empty_scoped_complement_falls_back_to_run_all(state, monkeypatch):
    """Nothing structurally forbids a family with zero live steps; the guard
    must refuse to convert run-everything into run-nothing."""
    from ci_selector.codemap import classify as cl

    monkeypatch.setattr(cl, "_build_map_allowed", lambda s, f: set())
    claim = cl._classify(state, "cmake/cpu_extension.cmake", None)
    assert claim.rule == "fail-open" and claim.run_all


def test_requirements_all_manual_declarers_fall_open(state, declared_deps_on):
    import dataclasses

    from ci_selector.codemap.classify import _classify, _source_dep_steps

    path = "requirements/nightly_torch_test.txt"
    declarers = _source_dep_steps(state, path)
    assert declarers, "fixture drift: no step declares the nightly torch file"
    st2 = dataclasses.replace(state, auto_step_ids=state.auto_step_ids - declarers)
    claim = _classify(st2, path, None)
    assert claim.rule == "fail-open" and claim.run_all


def test_hipify_also_selects_declaring_abi_step(state, declared_deps_on):
    """`cmake/hipify.py` was a rocm-only run_all match, but vllm_ci's
    torch-abi audit declares cmake/: real CI runs it, so must we. Derived, the
    declarer route is what carries it now that the sweep is gone."""
    sel = select(state, ["cmake/hipify.py"])
    assert not sel.run_all
    assert "vllm_ci:torch-stable-abi-audit" in sel.selected


def test_family_exclusive_no_declarers_keeps_complement(state):
    """A family-exclusive path with no auto declarer keeps the device-family
    complement (the declarers-consult must not silence it)."""
    from ci_selector.codemap.classify import _classify

    claim = _classify(state, "tools/rocm_env_check.sh", None)
    assert claim.rule == "fail-open" and not claim.run_all and claim.step_ids


def test_family_exclusive_inside_roots_keeps_complement(state, declared_deps_on):
    """The roots gate holds inside the scoped consult: a cpu-exclusive vllm asset with
    blanket vllm/ declarers keeps the complement, not the (narrowing) declarers."""
    from ci_selector.codemap.classify import _classify, _source_dep_steps

    path = "vllm/v1/worker/cpu_tuning_table.json"
    assert _source_dep_steps(state, path) & state.auto_step_ids, (
        "fixture drift: expected blanket vllm/ auto declarers"
    )
    claim = _classify(state, path, None)
    assert claim.rule == "fail-open" and not claim.run_all


def test_manual_only_declarers_keep_complement(state):
    """Family-exclusive path whose declarers are all manual-only falls back to
    the complement rather than selecting nothing."""
    import dataclasses

    from ci_selector.codemap.classify import _classify, _source_dep_steps

    path = "csrc/cpu/cpu_attn.cpp"
    declarers = _source_dep_steps(state, path)
    st2 = dataclasses.replace(state, auto_step_ids=state.auto_step_ids - declarers)
    claim = _classify(st2, path, None)
    assert claim.rule == "fail-open" and not claim.run_all and claim.step_ids


# ---- R3: target-coverage (tests/benchmarks-side empty-closure files) --------


def test_tests_side_script_zero_coverage_selects_nothing(state):
    """#50110: an unimported .sh in a directory no step covers or declares routes to
    nothing. Synthetic path because every real tests/*.sh now sits under a declared
    tree, so this exercises the truly-unclaimed branch."""
    from ci_selector.codemap.classify import _classify, _source_dep_steps

    path = "tests/__unclaimed_helper_area__/run_nothing.sh"
    assert not _source_dep_steps(state, path), "synthetic path unexpectedly declared"
    claim = _classify(state, path, None)
    assert claim.rule == "target-coverage"
    assert not claim.run_all and not claim.step_ids


def test_tests_yaml_dir_target_coverage(state):
    """A tests-side yaml under a step's directory target routes to that step. Synthetic
    path because a real config yaml rides an asset edge into conftest, routes broad."""
    sel = select(state, ["tests/config/__target_coverage_probe__.yaml"])
    assert not sel.run_all and _non_always(sel, state)
    assert any(c.rule == "target-coverage" for c in sel.claims)


def test_eval_config_yaml_covered_via_file_target_parent(state):
    """#49881: an eval config yaml sits beside a step's .py file-target, so the
    parent-dir leg routes it to the lm-eval steps, bounded to a handful, not run-all."""
    sel = select(state, ["tests/evals/gsm8k/configs/DeepSeek-R1-DP.yaml"])
    assert not sel.run_all
    assert "vllm_ci:lm-eval-small-models" in sel.selected
    assert len(_non_always(sel, state)) <= 8  # over-selection ceiling, not run-all


def test_manual_only_script_ref_selects_nothing_with_manual_hits(state):
    """A tests .sh referenced only by manual-only steps auto-selects nothing
    but shows those steps as manual hits (the _nothing_auto_runs hook)."""
    path = "tests/weight_loading/run_model_weight_loading_test.sh"
    sel = select(state, [path])
    assert not sel.run_all
    assert not _non_always(sel, state)
    assert sel.manual_hits


def test_added_init_under_covered_tests_dir_routes(state, declared_deps_on):
    """#50330 shape: an added tests __init__ under a directory a step targets
    routes to that step (not run-all)."""
    from ci_selector.codemap.classify import _classify

    claim = _classify(state, "tests/v1/e2e/spec_decode/zz_new/__init__.py", None)
    assert claim.rule == "target-coverage"
    assert not claim.run_all and claim.step_ids & state.auto_step_ids


# ---- R4: added-trivial-init (added trivial __init__.py under vllm/) --------


def test_trivial_init_units():
    from ci_selector.codemap.classify import _is_trivial_init

    assert _is_trivial_init("# SPDX header only\n")
    assert _is_trivial_init('"""a docstring"""\n')
    assert _is_trivial_init("")
    assert not _is_trivial_init("__all__ = []\n")
    assert not _is_trivial_init("import x\n")
    assert not _is_trivial_init("def f(:\n")  # syntax error -> not trivial


def test_added_trivial_init_routes_to_package_closure(state, monkeypatch):
    """#50131 shape: a new SPDX-only __init__ routes to its package subtree's
    reverse closure, not run-all."""
    from ci_selector.codemap import registry_diff
    from ci_selector.codemap.classify import _classify_added_init
    from ci_selector.codemap.state import DiffContext

    monkeypatch.setattr(registry_diff, "git_show", lambda r, ref, p: "# SPDX\n")
    ctx = DiffContext(base="b", head="h", status={})
    claim = _classify_added_init(
        state, "vllm/model_executor/layers/mamba/ops/__init__.py", ctx
    )
    assert claim is not None and claim.rule == "added-trivial-init"
    assert claim.test_files and not claim.run_all


def test_added_nontrivial_or_new_package_init_falls_through(state, monkeypatch):
    from ci_selector.codemap import registry_diff
    from ci_selector.codemap.classify import _classify_added_init
    from ci_selector.codemap.state import DiffContext

    ctx = DiffContext(base="b", head="h", status={})
    monkeypatch.setattr(registry_diff, "git_show", lambda r, ref, p: "import x\n")
    assert (
        _classify_added_init(
            state, "vllm/model_executor/layers/mamba/ops/__init__.py", ctx
        )
        is None
    )
    monkeypatch.setattr(registry_diff, "git_show", lambda r, ref, p: "\n")
    assert _classify_added_init(state, "vllm/zz_brand_new_pkg/__init__.py", ctx) is None


def test_added_init_in_keyed_package_prefers_key_routing(state):
    """The added-in-claimed-package rule wins before added-trivial-init when
    the package is string-keyed (order guard)."""
    from ci_selector.codemap.classify import _classify
    from ci_selector.codemap.state import DiffContext

    path = "vllm/models/minimax_m3/zz_new/__init__.py"
    ctx = DiffContext(base="b", head="h", status={path: "A"})
    claim = _classify(state, path, ctx)
    assert claim.rule == "added-in-claimed-package"


# ---- R5: package-data (non-.py data files under vllm/) ---------------------


def test_mamba_tuning_json_routes_scoped(state):
    """#50006: a mamba tuning json routes to its owning package's tests plus the amd
    family parsed from device_name= in the filename, not run-all."""
    path = (
        "vllm/model_executor/layers/mamba/ops/configs/selective_state_update/"
        "headdim=64,dstate=128,device_name=AMD_Instinct_MI300X,cache_dtype=float32.json"
    )
    sel = select(state, [path])
    assert not sel.run_all
    assert any(c.rule == "package-data" for c in sel.claims)
    assert set(sel.selected) & state.family_steps("amd")


def test_helion_config_json_routes_to_consumers(state):
    """#50345 shape: a helion config json routes to the helion tests plus the
    cuda family from its nvidia_b200 filename."""
    path = "vllm/kernels/helion/configs/fused_qk_norm_rope/nvidia_b200.json"
    sel = select(state, [path])
    assert not sel.run_all
    assert any(c.rule == "package-data" for c in sel.claims)
    assert set(sel.selected) & state.family_steps("cuda")


def test_declared_deps_never_fires_inside_graph_roots(state):
    """An unmodelable vllm asset (tuning json) is owned by package-data now,
    not the blanket vllm/ declarers, and never runs-all via declared-deps."""
    from ci_selector.codemap.classify import _classify, _classify_declared_deps

    path = "vllm/model_executor/layers/fused_moe/configs/zzz_probe.json"
    assert _classify_declared_deps(state, path) is None
    claim = _classify(state, path, None)
    assert claim.rule == "package-data" and not claim.run_all


@pytest.mark.drift
def test_release_pipeline_files_still_exist(vllm_repo):
    """The release pipeline sits in no job_dir, so it is named by hand rather
    than derived. A rename leaves the name matching nothing and the file falls
    through to whatever rule catches it next, with no sign anything changed."""
    from ci_selector.handwritten import RELEASE_PIPELINE_FILES

    missing = [p for p in RELEASE_PIPELINE_FILES if not (vllm_repo / p).is_file()]
    assert not missing, drift_message(
        f"RELEASE_PIPELINE_FILES names files that do not exist: {missing}",
        "These route to the release pipeline, which Buildkite runs directly "
        "rather than through the configs we parse. An entry matching nothing "
        "sends release-only edits down the ordinary rules instead.",
        f"the pipeline file was renamed: update RELEASE_PIPELINE_FILES in {HW}",
        f"the release pipeline is gone: delete the entry from {HW}",
    )


def test_release_refs_follow_one_level_of_indirection(vllm_repo):
    """A file a release script references (not named directly in the release yaml)
    is release-only too; one-level recursion catches manylinux.sh, so no run-all."""
    from ci_selector.codemap.externals import _REPO_PATH_RE, release_pipeline_refs
    from ci_selector.handwritten import RELEASE_PIPELINE_FILES

    refs = release_pipeline_refs(vllm_repo)
    direct: set[str] = set()
    for rel in RELEASE_PIPELINE_FILES:
        try:
            text = (vllm_repo / rel).read_text()
        except OSError:
            continue
        direct |= {m for m in _REPO_PATH_RE.findall(text) if (vllm_repo / m).is_file()}
    assert refs - direct, "recursion found no transitively-referenced release file"


def test_package_data_zero_auto_coverage_falls_open(state):
    """Direction guard: a data file whose owning closure auto-runs nowhere and
    parses no family must keep the run-all fail-open, not select nothing."""
    import dataclasses

    from ci_selector.codemap.classify import _classify

    st2 = dataclasses.replace(
        state, invoked=set(), auto_run_files=set(), auto_prefixes=()
    )
    path = "vllm/model_executor/layers/fused_moe/configs/zzz_probe.json"
    claim = _classify(st2, path, None)
    assert claim.rule == "fail-open" and claim.run_all


# ---- R6: release-ci + docker-input relabel ---------------------------------


def test_release_file_with_auto_declarer_selects_it(state):
    """A release-pipeline file a live auto step also declares as a source dep must
    select it: Docker Build Metadata runs docker-build-metadata-args.sh in its test."""
    sel = select(state, [".buildkite/scripts/docker-build-metadata-args.sh"])
    assert not sel.run_all
    assert "vllm_ci::computer: (CPU) Docker Build Metadata" in sel.selected
    assert any(c.rule == "release-ci" for c in sel.claims)


def test_release_only_script_selects_nothing(state):
    path = ".buildkite/scripts/build-macos-wheel.sh"
    assert path in state.release_refs, "fixture drift: not a release ref"
    sel = select(state, [path])
    assert not sel.run_all and not _non_always(sel, state)
    assert any(c.rule == "release-ci" for c in sel.claims)


def test_rocm_release_script_zero_claims_before_family_fail_open(state):
    """A non-.buildkite release script with a rocm basename must be zeroed
    before the amd scoped fail-open would swallow it."""
    from ci_selector.codemap.classify import _classify

    path = "tools/vllm-rocm/generate-rocm-wheels-root-index.sh"
    claim = _classify(state, path, None)
    assert claim.rule == "release-ci" and not claim.run_all and not claim.step_ids


def test_release_refs_derived_and_disjoint_from_live_steps(state):
    """The derived release refs are non-empty, include the tools/ member, and reference
    nothing a modeled step consumes."""
    from ci_selector.handwritten import RELEASE_PIPELINE_FILES

    refs = state.release_refs
    assert len(refs) >= 10
    assert "tools/vllm-rocm/generate-rocm-wheels-root-index.sh" in refs
    live: set[str] = set()
    for p in state.pipelines:
        for st in p.targets.values():
            live |= set(st.scripts_seen) | set(st.data_files)
            live |= {t.path for t in st.targets}
    assert refs.isdisjoint(live), sorted(refs & live)
    for rel in RELEASE_PIPELINE_FILES:
        assert not any(rel == p.config.config_file for p in state.pipelines)


def test_unreferenced_rocm_ci_script_escalates_its_own_pipeline(state):
    """A `.buildkite/scripts/rocm/` script no live step references matches the
    ROCm config's run_all_patterns prefix, so it escalates that pipeline only,
    never every pipeline and never the floor."""
    sel = select(state, [".buildkite/scripts/rocm/ci-bake-rocm.sh"])
    assert set(sel.run_all) == {"vllm_rocm_ci"}


def test_docker_input_relabel_still_run_all(state):
    """A docker-image COPY source keeps its run-all but says why."""
    from ci_selector.codemap.classify import _classify

    path = "tools/ep_kernels/install_python_libraries.sh"
    assert path in state.docker_inputs, "fixture drift: not a docker input"
    claim = _classify(state, path, None)
    assert claim.rule == "fail-open"
    assert set(claim.run_all) == {"vllm_ci", "vllm_intel_ci", "vllm_rocm_ci"}
    assert "docker-image build input" in claim.detail


# ---- R7: rename pairing + added-file head-closure --------------------------


def test_diff_context_carries_rename_map(state, monkeypatch):
    """A rename source gets status D (it vanishes); a copy source keeps its
    real status (it still exists at head)."""
    import ci_selector.codemap.classify as sel_mod
    from ci_selector.codemap.classify import _diff_context

    files = [
        DiffFile("R", "new_r.py", old_path="old_r.py"),
        DiffFile("C", "new_c.py", old_path="old_c.py"),
    ]
    monkeypatch.setattr(sel_mod, "diff_files", lambda vllm_repo, base, head: files)
    ctx = _diff_context(state, "b", "h")
    assert ctx.renames == {"new_r.py": "old_r.py", "new_c.py": "old_c.py"}
    assert ctx.status["old_r.py"] == "D"
    assert "old_c.py" not in ctx.status


def test_renamed_path_claims_via_old_closure(state):
    """A renamed new path routes via the old path's base closure, not run-all."""
    from ci_selector.codemap.classify import _classify
    from ci_selector.codemap.state import DiffContext

    old = "vllm/v1/spec_decode/eagle.py"
    new = "vllm/v1/spec_decode/eagle_renamed.py"
    ctx = DiffContext(base="b", head="h", status={new: "R"}, renames={new: old})
    claim = _classify(state, new, ctx)
    assert claim.rule == "renamed"
    assert not claim.run_all and claim.step_ids
    assert f"renamed from {old}" in claim.detail


def test_renamed_path_with_unclaimed_old_still_fails_open(state):
    """The old path being unclaimed carries run-all forward (never a double
    fail-open, direction intact)."""
    from ci_selector.codemap.classify import _classify
    from ci_selector.codemap.state import DiffContext

    new, old = "vllm/ghost_new.py", "vllm/ghost_zzz_helper.py"
    ctx = DiffContext(base="b", head="h", status={new: "R"}, renames={new: old})
    claim = _classify(state, new, ctx)
    assert claim.rule == "renamed" and claim.run_all


def test_copied_path_routes_via_source(state):
    from ci_selector.codemap.classify import _classify
    from ci_selector.codemap.state import DiffContext

    old = "vllm/v1/spec_decode/eagle.py"
    new = "vllm/v1/spec_decode/eagle_copy.py"
    ctx = DiffContext(base="b", head="h", status={new: "C"}, renames={new: old})
    claim = _classify(state, new, ctx)
    assert claim.rule == "renamed" and "copied from" in claim.detail


def test_a_rename_keeps_the_droppability_it_routed_through(state):
    """`graph` is the only rule that grants droppability, so a rebuild that
    forgets to carry it makes the step permanently un-droppable, and nothing
    reports that it happened."""
    from ci_selector.codemap.classify import _classify, _classify_graph
    from ci_selector.codemap.state import DiffContext

    old = "vllm/v1/spec_decode/eagle.py"
    new = "vllm/v1/spec_decode/eagle_renamed.py"
    base = _classify_graph(state, old)
    assert base is not None and base.droppable_step_ids, "fixture no longer droppable"

    ctx = DiffContext(base="b", head="h", status={new: "R"}, renames={new: old})
    claim = _classify(state, new, ctx)
    assert claim.droppable_step_ids >= base.droppable_step_ids
    assert claim.droppable_test_files == base.droppable_test_files


def test_a_rename_keeps_its_device_scope(state, vllm_repo):
    """The unsafe half of the same omission. `device_scope` stops a
    device-named data file selecting steps no other device can read, so losing
    it on a rename widens rather than narrows."""
    from ci_selector.codemap.classify import _classify
    from ci_selector.codemap.state import DiffContext

    cfgs = sorted(
        (vllm_repo / "vllm/model_executor/layers/fused_moe/configs").glob(
            "*device_name=NVIDIA_H*.json"
        )
    )
    assert cfgs, "no H-series tuning config at HEAD to exercise scoping"
    old = str(cfgs[0].relative_to(vllm_repo))
    expected = _classify(state, old, None).device_scope
    assert expected, "fixture is no longer device-scoped"

    # The name carries the device, so the renamed path keeps the same scope.
    new = str(Path(old).with_name("renamed_" + Path(old).name))
    ctx = DiffContext(base="b", head="h", status={new: "R"}, renames={new: old})
    assert _classify(state, new, ctx).device_scope == expected


def _head_stub(closure):
    from types import SimpleNamespace

    return SimpleNamespace(
        graph=SimpleNamespace(reverse_closure=lambda files, include_boot=True: closure)
    )


def test_added_head_closure_maps_to_base_steps(state, monkeypatch):
    """An added vllm/ file whose HEAD closure reaches a test under a base
    step's directory target routes there, not run-all."""
    import ci_selector.codemap.classify as sel_mod
    from ci_selector.codemap.classify import _classify
    from ci_selector.codemap.state import DiffContext

    closure = {
        "vllm/newarea/brand_new.py",
        "tests/v1/e2e/spec_decode/eagle/test_head_new.py",
    }
    monkeypatch.setattr(sel_mod, "_head_graph", lambda st, ctx: _head_stub(closure))
    ctx = DiffContext(base="b", head="h", status={"vllm/newarea/brand_new.py": "A"})
    claim = _classify(state, "vllm/newarea/brand_new.py", ctx)
    assert claim.rule == "added-head-closure"
    assert not claim.run_all and claim.test_files


def test_added_head_closure_empty_falls_through(state, monkeypatch):
    """A closure with no test/script members falls through to fail-open (the
    file may be dynamically loaded)."""
    import ci_selector.codemap.classify as sel_mod
    from ci_selector.codemap.classify import _classify
    from ci_selector.codemap.state import DiffContext

    monkeypatch.setattr(
        sel_mod, "_head_graph", lambda st, ctx: _head_stub({"vllm/a.py", "vllm/b.py"})
    )
    ctx = DiffContext(base="b", head="h", status={"vllm/newarea/brand_new.py": "A"})
    claim = _classify(state, "vllm/newarea/brand_new.py", ctx)
    assert claim.rule == "fail-open" and claim.run_all


def test_added_head_closure_unmappable_falls_through(state, monkeypatch):
    """A closure test in a brand-new dir no base step covers -> fail-open (the
    zero-step-mapping guard, never a silent empty claim)."""
    import ci_selector.codemap.classify as sel_mod
    from ci_selector.codemap.classify import _classify
    from ci_selector.codemap.state import DiffContext

    monkeypatch.setattr(
        sel_mod,
        "_head_graph",
        lambda st, ctx: _head_stub({"tests/brand_new_zzz_dir/test_x.py"}),
    )
    ctx = DiffContext(base="b", head="h", status={"vllm/newarea/brand_new.py": "A"})
    claim = _classify(state, "vllm/newarea/brand_new.py", ctx)
    assert claim.rule == "fail-open" and claim.run_all


def test_head_graph_inert_without_ctx(state, monkeypatch):
    """ctx=None must never build a head graph (ref-less select stays as-is)."""
    import ci_selector.codemap.classify as sel_mod
    from ci_selector.codemap.classify import _classify

    def boom(*a, **k):
        raise AssertionError("head graph must not be built without a diff ctx")

    monkeypatch.setattr(sel_mod, "_head_graph", boom)
    claim = _classify(state, "vllm/whatever_totally_new.py", None)
    assert claim.rule == "fail-open"


# --- item 3: exclusivity subtracts inferred reach, never direct collection ---


def _targets(**kw):
    from ci_selector.codemap.pipeline.targets import StepTargets

    st = StepTargets(step_id="s")
    for path, kind in kw.pop("targets", ()):
        st.add_target(path, kind)
    for k, v in kw.items():
        setattr(st, k, list(v))
    return st


def test_directly_collects_dir_target_covers_member():
    from ci_selector.codemap.selection import _directly_collects

    st = _targets(targets=[("tests/v1/attention", "pytest")])
    assert _directly_collects(st, "tests/v1/attention/test_rocm_backends.py")
    assert not _directly_collects(st, "tests/v1/other/test_rocm_backends.py")
    assert not _directly_collects(None, "tests/v1/attention/test_rocm_backends.py")


def test_directly_collects_honours_ignore():
    """`pytest kernels/ --ignore=kernels/attention` never imports the ignored
    subtree, so it is not proof of collection and must not disarm the rule."""
    from ci_selector.codemap.selection import _directly_collects

    st = _targets(
        targets=[("tests/kernels", "pytest")], ignored=["tests/kernels/attention"]
    )
    assert _directly_collects(st, "tests/kernels/test_rocm_misc.py")
    assert not _directly_collects(st, "tests/kernels/attention/test_rocm_mla.py")


def test_directly_collects_scripts_and_data():
    from ci_selector.codemap.selection import _directly_collects

    assert _directly_collects(_targets(scripts_seen=["a/run.sh"]), "a/run.sh")
    assert _directly_collects(_targets(data_files=["a/cfg.json"]), "a/cfg.json")


def test_rocm_named_test_selects_the_cuda_jobs_that_collect_it(state):
    """A rocm-named test under a directory a CUDA job pytest-collects runs on
    that job: a bad import there fails it, so the AMD-exclusive rule must not
    subtract it. Regression for the verified under-selection."""
    path = "tests/v1/attention/test_rocm_attention_backends_selection.py"
    sel = select(state, [path])
    picked = _selected(sel)
    assert "vllm_ci:v1-attention-h100-mi300" in picked, sorted(picked)
    assert "vllm_ci:v1-attention-b200" in picked, sorted(picked)
    assert "vllm_ci:v1-attention-h100-mi300-amd:amd" in picked, sorted(picked)


def test_disarm_is_not_a_blanket_revert(state):
    """Every CUDA step an AMD-exclusive test selects must be one that actually
    collects it. Stated as the invariant rather than against a fixed example,
    since which jobs collect which directory changes constantly."""
    from ci_selector.codemap import hardware
    from ci_selector.codemap.selection import _directly_collects

    targets = {sid: st for p in state.pipelines for sid, st in p.targets.items()}
    by_id = {s.step_id: s for p in state.pipelines for s in p.steps}
    paths = [
        f
        for f in state.catalog
        if hardware.exclusive_family_of_path(f) == "amd"
        and f.startswith("tests/")
        # exclusive_disabled turns the whole rule off for these (a cross-family
        # module-level importer exists), so there is no subtraction to disarm
        and f not in state.exclusive_disabled
    ]
    assert len(paths) > 5, f"too few rocm-named tests to be a real check: {paths}"
    for path in paths:
        for sid in _non_always(select(state, [path]), state):
            step = by_id.get(sid)
            if step is None or step.mirror_hw:
                continue
            # The rule only ever subtracts a step on a KNOWN device of another
            # family; unmapped and device-less steps are kept by design.
            if hardware.family_of_device(step.device) in (None, "amd"):
                continue
            assert _directly_collects(targets.get(sid), path), (
                f"{sid} selected for {path} without collecting it"
            )


# --- item 5: rule names are emitted and pinned ---


def test_selected_rules_parallel_to_reasons(state):
    sel = select(state, ["vllm/v1/attention/selector.py"])
    assert set(sel.selected_rules) == set(sel.selected)
    assert set(sel.selected_paths) == set(sel.selected)
    for sid, reasons in sel.selected.items():
        assert len(sel.selected_rules[sid]) == len(reasons), sid
        # The record pairs these by index. An entry missing from one list silently
        # shifts every later reason onto the wrong rule and the wrong file,
        # which is how an always-run step reads as attributable and droppable.
        assert len(sel.selected_paths[sid]) == len(reasons), sid


def test_always_run_is_never_attributable(state):
    # The one _record bypass. It writes the dicts directly, so it is the site
    # where the three lists can fall out of step.
    sel = select(state, ["vllm/v1/attention/selector.py"])
    for sid, rules in sel.selected_rules.items():
        for rule, paths in zip(rules, sel.selected_paths[sid]):
            if rule in ("always-run", "run-all", "preflight"):
                assert paths is None, (sid, rule)


def test_precommit_hooks_do_not_run_the_pipeline(state):
    from ci_selector.codemap.classify import _classify, _lint_only_files

    hooks = _lint_only_files(state)
    # Detection floor: an empty parse would silently disable the rule and
    # every lint edit would go back to running everything.
    assert len(hooks) > 5, hooks

    for path in [".pre-commit-config.yaml", *sorted(hooks)]:
        claim = _classify(state, path, None)
        assert not claim.run_all, f"{path} still runs everything"


def test_a_hook_script_ci_also_runs_keeps_its_steps(state, monkeypatch):
    # The rule must not zero a file just because pre-commit names it. If a
    # Buildkite step references it too, that reference wins.
    from ci_selector.codemap import classify as sel
    from ci_selector.codemap.classify import _classify

    path = "tools/pre_commit/mypy.py"
    assert _classify(state, path, None).rule == "no-code"
    monkeypatch.setattr(sel, "_direct_step_refs", lambda _s, _p: {"vllm_ci:some-step"})
    assert _classify(state, path, None).rule != "no-code"


def test_a_tools_file_precommit_does_not_name_is_untouched(state):
    # Narrow by construction: living under tools/ earns nothing from the
    # lint-only rule; an unreferenced one lands on the inert floor instead.
    from ci_selector.codemap.classify import _classify

    claim = _classify(state, "tools/not_a_real_hook_script.py", None)
    assert claim.rule == "inert"


def test_a_cycle_file_is_routed_by_colocation_and_selects_far_less(state):
    """The rule's whole point. Inside the cycle every file reaches the same
    closure, so reach selects near-everything and co-location has to cut it."""
    path = "vllm/lora/layers/base.py"
    assert path in state.full.import_cycle().reach_blind
    on = select(state, [path])
    assert any(c.rule == "colocated-tests" for c in on.claims)
    assert not on.run_all
    with_reach = _select_without_colocation(state, path)
    assert len(on.selected) < len(with_reach.selected) / 2, (
        len(on.selected),
        len(with_reach.selected),
    )


def _without_colocation(call):
    import os

    from ci_selector.codemap.colocation import ENV_VAR

    previous = os.environ.get(ENV_VAR)
    os.environ[ENV_VAR] = "off"
    try:
        return call()
    finally:
        if previous is None:
            del os.environ[ENV_VAR]
        else:
            os.environ[ENV_VAR] = previous


def _select_without_colocation(state, path):
    return _without_colocation(lambda: select(state, [path]))


def _classify_graph_without_colocation(state, path):
    from ci_selector.codemap.classify import _classify_graph

    return _without_colocation(lambda: _classify_graph(state, path))


def test_a_cycle_file_with_no_mirror_falls_back_to_the_graph_rule(state):
    """The fallback is the safety property: declining has to hand the file to
    reach, not select nothing."""
    path = "vllm/device_allocator/cumem.py"
    assert path in state.full.import_cycle().reach_blind
    sel = select(state, [path])
    assert not any(c.rule == "colocated-tests" for c in sel.claims)
    assert sel.selected or sel.run_all


def test_a_below_gate_file_outside_the_cycle_is_untouched_by_colocation(state):
    """Outside the knot a narrow closure is information, so below the size
    gate the rule must not fire."""
    from ci_selector.codemap import colocation
    from ci_selector.codemap.colocation import _pr_auto_selected

    path = "vllm/parser/mistral.py"
    assert path not in state.full.import_cycle().reach_blind
    reach_claim = _classify_graph_without_colocation(state, path)
    assert len(_pr_auto_selected(state, reach_claim)) < colocation.MIN_GRAPH_STEPS, (
        f"{path} crossed the hub gate; this test needs a below-gate fixture"
    )
    assert not any(c.rule == "colocated-tests" for c in select(state, [path]).claims)
    assert set(select(state, [path]).selected) == set(
        _select_without_colocation(state, path).selected
    )


def test_an_above_gate_hub_outside_the_cycle_is_routed_by_colocation(state):
    """The extension's point: a non-cycle file whose closure has gone hub-like
    is routed by its tests like a cycle member, and the answer must narrow."""
    path = "vllm/compilation/decorators.py"
    assert path not in state.full.import_cycle().reach_blind
    sel = select(state, [path])
    assert any(c.rule == "colocated-tests" for c in sel.claims)
    assert not sel.run_all
    with_reach = _select_without_colocation(state, path)
    assert len(sel.selected) < len(with_reach.selected) / 2, (
        len(sel.selected),
        len(with_reach.selected),
    )


def test_cycle_only_mode_keeps_the_pre_extension_behavior(state, monkeypatch):
    """cycle-only is the measurement arm: cycle members keep colocation while
    non-cycle hubs keep the closure. The A/B harnesses ride on this split."""
    from ci_selector.codemap.colocation import ENV_VAR

    monkeypatch.setenv(ENV_VAR, "cycle-only")
    hub = select(state, ["vllm/compilation/decorators.py"])
    assert not any(c.rule == "colocated-tests" for c in hub.claims)
    member = select(state, ["vllm/lora/layers/base.py"])
    assert any(c.rule == "colocated-tests" for c in member.claims)


def test_an_above_gate_file_without_a_mirror_keeps_the_closure(state, monkeypatch):
    """The `if colocated` fallback guard holds outside the cycle too. No real
    file above the gate lacks a mirror at this base, so the guard is exercised
    by stubbing the mirror away from a real hub."""
    from ci_selector.codemap import colocation
    from ci_selector.codemap.classify import _classify_graph

    monkeypatch.setattr(
        colocation, "implicated_tests", lambda st, p: (frozenset(), None)
    )
    claim = _classify_graph(state, "vllm/compilation/decorators.py")
    assert claim is not None and claim.rule == "graph"


def test_a_would_widen_hub_keeps_the_closure(state, monkeypatch):
    """The strict clamp: a mirror answer no narrower than the graph answer must
    not fire. Unreachable on a real file at this base, since zero widened files
    at the gate is how MIN_GRAPH_STEPS was derived, so the widening is
    manufactured by handing the mirror every invoked test."""
    from ci_selector.codemap import colocation
    from ci_selector.codemap.classify import _classify_graph

    path = "vllm/compilation/decorators.py"
    everything = frozenset(f for f in state.invoked if f.startswith("tests/"))
    monkeypatch.setattr(
        colocation, "implicated_tests", lambda st, p: (everything, "tests/")
    )
    claim = _classify_graph(state, path)
    assert claim is not None and claim.rule == "graph"


def test_a_hub_claim_carries_the_plain_static_floor(state):
    """A hub claim must keep every test reaching the file through plain
    module-level imports. PR 38962's CPU-only breakage was caught by tests that
    import the file transitively, which no mirror or direct-importer union
    sees."""
    from ci_selector.codemap.classify import _classify_graph
    from ci_selector.codemap.colocation import _plain_static_tests

    path = "vllm/compilation/decorators.py"
    claim = _classify_graph(state, path)
    assert claim is not None and claim.rule == "colocated-tests", (
        "fixture went stale: expected hub routing, got "
        f"{claim.rule if claim else None!r}"
    )
    floor = _plain_static_tests(state, path)
    assert len(floor) > 20, "fixture went stale: decorators lost its plain fan-in"
    assert floor <= claim.test_files


def test_a_hub_routed_registered_file_keeps_its_own_key_steps(state):
    """Outside the cycle a file's OWN registered keys are kept: by-name e2e
    coverage no mirror or importer can see. Only closure-derived keys are
    dropped."""
    from ci_selector.codemap.classify import _classify_graph

    path = "vllm/model_executor/models/llama.py"
    own = state.keys.steps_naming(state.keys.for_file(path))
    assert len(own) > 10, "fixture went stale: llama.py no longer keys steps"
    claim = _classify_graph(state, path)
    assert claim is not None and claim.rule == "colocated-tests", (
        "fixture went stale: expected hub routing, got "
        f"{claim.rule if claim else None!r}"
    )
    assert own <= claim.step_ids


def test_a_cycle_file_that_fails_preflight_still_fails_open(state, monkeypatch):
    """Both preflight guards say the graph is known-incomplete here. Answering
    from co-location instead would be trusting it anyway."""
    from dataclasses import replace

    from ci_selector.codemap.classify import _classify

    path = "vllm/lora/layers/base.py"
    for field in ("parse_error_paths", "unclassified_sites"):
        broken = replace(state.preflight, **{field: frozenset({path})})
        monkeypatch.setattr(state, "preflight", broken)
        claim = _classify(state, path, None)
        assert claim.rule == "fail-open", (field, claim.rule)
        assert claim.run_all


def test_colocated_claims_keep_the_droppability_contract(state):
    """Same contract the graph rule owes: droppables are a subset, and hardware
    tagging never becomes droppable."""
    from ci_selector.codemap import hardware
    from ci_selector.codemap.classify import _classify

    cycle = state.full.import_cycle().reach_blind
    # Both arms owe the contract, so the sample carries hub-routed files too.
    # Their liveness floor is the COLOCATED_HUBS test above, which asserts
    # every member routes here.
    hubs = [p for p in COLOCATED_HUBS if p not in cycle]
    assert hubs, "every COLOCATED_HUBS member is in the cycle; add a hub fixture"
    sample = sorted(cycle)[:400] + hubs
    claims = [_classify(state, p, None) for p in sample]
    colocated = [c for c in claims if c is not None and c.rule == "colocated-tests"]
    assert colocated, "no sampled file routed by co-location; the sample is wrong"
    for claim in colocated:
        assert claim.droppable_step_ids <= claim.step_ids
        assert claim.droppable_test_files
    rocm = "vllm/platforms/rocm.py"
    family = hardware.family_of_path(rocm)
    claim = _classify(state, rocm, None)
    if family:
        assert not (claim.droppable_step_ids & state.family_steps(family))


def test_declared_deps_are_droppable_but_hardware_tagging_is_not(state):
    from ci_selector.codemap import hardware
    from ci_selector.codemap.classify import _classify, _source_dep_steps

    # Both reachability rules owe the same contract, so assert it on both
    # rather than on whichever one happens to claim this path today.
    reach_rules = ("graph", "colocated-tests")
    path = "vllm/v1/attention/selector.py"
    claim = _classify(state, path, None)
    assert claim.rule in reach_rules
    assert claim.droppable_step_ids <= claim.step_ids
    assert _source_dep_steps(state, path, specific_only=True) <= (
        claim.droppable_step_ids
    )
    # An empty droppable set would stop scoping silently, reading as a clean 0.
    assert claim.droppable_test_files

    rocm = "vllm/platforms/rocm.py"
    fam = hardware.family_of_path(rocm)
    hw_claim = _classify(state, rocm, None)
    if fam and hw_claim.rule in reach_rules:
        assert not (hw_claim.droppable_step_ids & state.family_steps(fam))


def test_droppable_steps_must_be_a_subset(state):
    import pytest
    from ci_selector.codemap.claim import Claim

    with pytest.raises(ValueError, match="subset of step_ids"):
        Claim("graph", "d", step_ids={"a"}, droppable_step_ids={"a", "b"})


def test_every_emitted_rule_is_pinned(state):
    from ci_selector.codemap.claim import OUTPUT_RULES

    sel = select(state, ["vllm/v1/attention/selector.py", "requirements/cuda.txt"])
    emitted = {r for rules in sel.selected_rules.values() for r in rules}
    assert emitted <= OUTPUT_RULES, emitted - OUTPUT_RULES
    assert "always-run" in emitted


def test_unpinned_rule_raises():
    import pytest
    from ci_selector.codemap.claim import Claim

    with pytest.raises(ValueError, match="unpinned claim rule"):
        Claim("not-a-rule", "detail")


def test_render_json_carries_rules(state):
    import json

    from ci_selector.report import render_json

    d = json.loads(render_json(select(state, ["vllm/v1/attention/selector.py"])))
    assert set(d["selected_rules"]) == set(d["selected"])


def test_rules_constant_matches_the_claims_actually_constructed():
    """Derived, not duplicated: scan for Claim(...) literals and compare. A new
    rule, a rename, or a removal all show up here without anyone maintaining a
    second list. RULES is a contract because the record routes on these names."""
    import ast
    from pathlib import Path

    from ci_selector.codemap.claim import RULES

    src_dir = Path(__file__).resolve().parents[1] / "ci_selector"
    found = set()
    for py in src_dir.rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text())):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "Claim"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                found.add(node.args[0].value)
    assert found, "scanner found no Claim(...) literals; it has gone blind"
    assert found == RULES, {"unpinned": found - RULES, "stale": RULES - found}


# --- item 7: the blanket-declarer divergence, and its gate ---


def test_unmodeled_dynamic_site_does_not_restore_the_yaml_declarers(state):
    """An unclassified site used to put every catch-all `vllm/`/`tests/`
    declarer back for every changed file, on every PR, until someone noticed.
    That leaned on the yaml this tool exists to delete, and it was never the
    net it looked like: catch-all declarers are a fraction of the pipeline, so
    they could not have covered an unknown edge anyway."""
    import dataclasses

    path = "vllm/model_executor/layers/quantization/fp8.py"
    assert (state.repo / path).exists(), f"{path} moved; pick another specimen"
    gated = dataclasses.replace(
        state,
        preflight=dataclasses.replace(
            state.preflight,
            unclassified_sites=frozenset({"vllm/utils/humming.py"}),
        ),
    )
    assert _selected(select(gated, [path])) == _selected(select(state, [path]))


def test_the_site_file_itself_still_fails_open(state):
    """What is left of the protection, and it must stay: changing the file that
    holds the unfollowable import runs everything, because what it loads is the
    part we cannot see."""
    import dataclasses

    site = "vllm/utils/humming.py"
    gated = dataclasses.replace(
        state,
        preflight=dataclasses.replace(
            state.preflight, unclassified_sites=frozenset({site})
        ),
    )
    sel = select(gated, [site])
    claim = next(c for c in sel.claims)
    assert claim.rule == "fail-open", claim.rule
    assert "unmodeled dynamic import" in claim.detail
    assert _selected(sel) > _selected(select(state, [site]))


def test_selected_by_file_covers_every_attributable_step(state):
    """The per-file inverse of `selected`, and its completeness contract.

    Deciding between the map and the coverage record has to happen per
    changed file: the record is competent for files inside its root and blind
    to the rest, so a whole-diff answer cannot be split. Nothing else in
    `Selection` carries that: every other field is step-keyed.

    Anything absent from the map is file-independent, always-on rather than
    not-needed, and a consumer has to read it that way.
    """
    paths = ["vllm/lora/lora_weights.py", "tests/lora/test_layers.py"]
    sel = select(state, paths)
    assert sel.selected_by_file, "no per-file attribution recorded"
    assert set(sel.selected_by_file) <= set(paths), "attributed to an unchanged file"

    attributed = {s for steps in sel.selected_by_file.values() for s in steps}
    assert attributed <= set(sel.selected), "attributed a step that was not selected"

    always = {s.step_id for p in state.pipelines for s in p.steps if s.always_runs}
    unattributed = set(sel.selected) - attributed
    assert unattributed <= always, (
        f"steps selected with no file and no always-run reason: "
        f"{sorted(unattributed - always)[:3]}"
    )


def test_run_all_is_attributed_to_the_file_that_caused_it(state):
    """A run-all sweep records every step in the pipeline, so without naming
    the file that escalated it the whole selection would look file-independent
    and the per-file split would collapse to 'everything, always'."""
    escalates = "tools/install_gdrcopy.sh"  # unclaimed by any rule: fail-open
    sel = select(state, [escalates, "vllm/lora/lora_weights.py"])
    assert sel.run_all, f"fixture drift: {escalates} no longer escalates"
    assert set(sel.run_all_paths) == set(sel.run_all)
    assert set(sel.run_all_paths.values()) == {escalates}, sel.run_all_paths
    assert len(sel.selected_by_file.get(escalates, [])) > 100


class _HostileOrderSet(set):
    """A set that iterates worst-first.

    A real set's order moves with the interpreter's hash seed, which is fixed
    for the life of a process and so cannot be varied from inside a test. This
    stands in for it: any code that trusts iteration order gets the answer it
    deserves, deterministically.
    """

    def __iter__(self):
        return iter(sorted(super().__iter__(), reverse=True))


def test_the_cited_test_file_does_not_depend_on_set_iteration_order():
    """Reasons must not change between two runs of identical code.

    `_targets_cover` picks the test file a reason quotes ("src -> THIS ->
    step"). It used to return the first match while iterating `test_files`,
    which is a set, and set order for strings moves with the interpreter's
    hash seed, so output fields differed run to run on an unchanged diff.
    Harmless to the step list, but every before/after check of a refactor then
    needs a normaliser, which taxes exactly the work that proves a change safe.
    """
    from ci_selector.codemap.pipeline.targets import StepTargets, Target
    from ci_selector.codemap.selection import _targets_cover

    st = StepTargets(
        step_id="p:s", targets=[Target(path="tests/models", kind="pytest")]
    )
    members = [f"tests/models/test_{n}.py" for n in ("zeta", "alpha", "mid", "beta")]
    hostile = _HostileOrderSet(members)

    # Detection floor: the stand-in must actually mislead. If it ever yields
    # the answer first, the assertion below passes without discriminating.
    assert next(iter(hostile)) == max(members), "the hostile set stopped being hostile"

    assert _targets_cover(st, hostile) == min(members), (
        "the cited file follows iteration order; two runs will disagree"
    )


# ---- rust two-root rule ----------------------------------------------------


def test_rust_binary_only_file_stays_off_the_image_union(state):
    """The whole point of the rust rule: a binary-only crate file keeps its
    declarers, gate-env steps and hardware-image consumers, and nothing else.
    Borrowed whole-context images used to balloon it. The ceiling is loose on
    purpose so pipeline churn does not break the test."""
    sel = select(state, ["rust/src/server/src/lib.rs"])
    assert not sel.run_all
    assert set(sel.selected) >= RUST_DECLARERS
    vllm_ci = {s for s in sel.selected if s.startswith("vllm_ci:")}
    assert len(vllm_ci) < 60, sorted(vllm_ci)


def test_rust_cdylib_file_carries_the_pyo3_bridge(state):
    """A parser-closure file affects vllm._rust_tool_parser, which production
    parsers import with no env gate, so it inherits the bridge file's whole
    claim on top of the binary route."""
    bridge = select(state, ["vllm/tool_parsers/rust_tool_parser.py"])
    sel = select(state, ["rust/src/parser/src/lib.rs"])
    assert not sel.run_all
    assert set(sel.selected) >= set(bridge.selected)


def test_rust_workspace_root_is_union_of_both_routes(state):
    """Cargo.lock can change both artifacts, so it takes at least everything
    either bucket takes."""
    binary = select(state, ["rust/src/server/src/lib.rs"])
    cdylib = select(state, ["rust/src/parser/src/lib.rs"])
    root = select(state, ["rust/Cargo.lock"])
    assert not root.run_all
    assert set(root.selected) >= set(binary.selected) | set(cdylib.selected)


def test_rust_file_keeps_env_keyed_steps(state):
    """Leg 1 in isolation: the gate-env search must find the e2e steps and the
    rule must carry them, so a step that opts in without declaring rust/ still
    runs on rust changes."""
    from ci_selector.handwritten import RUST_GATE_ENV_VARS

    gate_steps = state.keys.steps_naming_raw(set(RUST_GATE_ENV_VARS))
    assert gate_steps, "no step exports the rust gates; leg 1 is dead"
    sel = select(state, ["rust/src/server/src/lib.rs"])
    assert gate_steps & state.auto_step_ids <= set(sel.selected)


def test_image_union_exempt_membership():
    """Every entry is a decision that a rule's own answer beats the build
    graph's, so force a re-read on any change. "inert" is here so the image
    COPY does not borrow consumer steps back onto a file proved unreferenced,
    but stays out of _DEP_UNION_EXEMPT since a declarer disproves the veto."""
    from ci_selector.codemap.unions import _DEP_UNION_EXEMPT, _IMAGE_UNION_EXEMPT

    assert (
        frozenset(
            {
                "no-code",
                "no-hardware",
                "legacy-ci",
                "inert-ci",
                "inert",
                "release-ci",
                "rust",
            }
        )
        == _IMAGE_UNION_EXEMPT
    )
    assert "inert" not in _DEP_UNION_EXEMPT


def test_rust_reaches_hardware_image_consumers(state):
    """Leg 2 pinned both ways: the ROCm bake steps and CPU suites compile rust
    in-step so they stay; the intel test steps only consume a prebuilt image
    whose rust is inert without the gate, so they go. The derived replacement
    for rust/ in rocm's run_all_patterns."""
    sel = select(state, ["rust/src/server/src/lib.rs"])
    for sid in (
        "vllm_rocm_ci:cpu-kernel-tests",
        "vllm_rocm_ci:image-build-amd",
        "vllm_rocm_ci:ensure-ci-base-amd",
        "vllm_intel_ci:image-build-xpu",
    ):
        assert sid in sel.selected, sid
    intel_tests = {
        s
        for s in sel.selected
        if s.startswith("vllm_intel_ci:") and "image-build" not in s
    }
    assert not intel_tests, sorted(intel_tests)


# ---- requirements Phase A --------------------------------------------------


@pytest.mark.drift
def test_requirements_family_map_pins_the_tree(state):
    """Drift pin for every requirements/ file's derived family, including the
    cuda mapping PATH_TOKEN_FAMILIES cannot carry: a global cuda token would
    misfire on vllm/ paths and on the generic Dockerfile. An unknown-token new
    file maps to None and keeps the image widening, which fails safe.

    The paths are checked against the tree as well as the tokenizer. Without
    that this pinned a pure string function and a renamed requirements file
    moved nothing, despite the name.
    """
    from ci_selector.codemap.hardware import requirements_family_of_path

    expected = {
        "requirements/cuda.txt": "cuda",
        "requirements/build/cuda.txt": "cuda",
        "requirements/test/cuda.in": "cuda",
        "requirements/test/cuda.txt": "cuda",
        "requirements/rocm.txt": "amd",
        "requirements/kv_connectors_rocm.txt": "amd",
        "requirements/cpu.txt": "cpu",
        "requirements/xpu.txt": "xpu",
        "requirements/tpu.txt": "tpu",
        "requirements/common.txt": None,
        "requirements/lint.txt": None,
        "requirements/build/rust.txt": None,
        "requirements/test/nightly-torch.txt": None,
    }
    got = {f: requirements_family_of_path(f) for f in expected}
    assert got == expected
    gone = sorted(f for f in expected if not (state.repo / f).is_file())
    assert not gone, drift_message(
        f"These requirements files no longer exist: {gone}",
        "Each one is here because its name is what gives it a hardware "
        "family. A path that is gone pins a spelling nothing uses, and the "
        "file that replaced it is untested.",
        "the file moved or was renamed: update the path in this test",
        f"a family lost its requirements file: check REQUIREMENTS_EXTRA_TOKEN_"
        f"FAMILIES in {HW} still earns its place",
    )


def test_requirements_cuda_token_stays_inside_requirements():
    """The cuda token must not leak into the global tokenizer: vllm/ paths and
    the generic Dockerfile stay family-less, or the image rules quietly change
    which files they treat as shared."""
    from ci_selector.codemap.hardware import (
        family_of_path,
        requirements_family_of_path,
    )

    assert requirements_family_of_path("vllm/cuda_utils.py") is None
    assert family_of_path("requirements/cuda.txt") is None
    assert family_of_path("docker/Dockerfile") is None


def test_requirements_build_validated_files_select_the_floor(state):
    """lint.txt and dev.txt exist for tooling no test imports, so their honest
    reach is the declaring steps plus the always-run builds, not the full
    docker-image widening."""
    for path in ("requirements/lint.txt", "requirements/dev.txt"):
        sel = select(state, [path])
        assert not sel.run_all, path
        assert "vllm_ci:ray-dependency-compatibility-check" in sel.selected, path
        assert len(sel.selected) < 20, (path, len(sel.selected))


@pytest.mark.drift
def test_requirements_build_validated_members_exist(state):
    from ci_selector.handwritten import REQUIREMENTS_BUILD_VALIDATED

    missing = [
        p for p in REQUIREMENTS_BUILD_VALIDATED if not (state.repo / p).is_file()
    ]
    assert not missing, drift_message(
        f"REQUIREMENTS_BUILD_VALIDATED names files that no longer exist: {missing}",
        "Listed files skip the docker-image widening because no test imports "
        "them. An entry naming nothing exempts nothing, and the next "
        "tooling-only requirements file added is widened instead.",
        f"the file moved: update the path in REQUIREMENTS_BUILD_VALIDATED in {HW}",
        f"the file is gone for good: delete the entry from {HW}",
    )


def test_requirements_cuda_keeps_unlabeled_consumers(state):
    """The anti-under-selection finding: the image widening's answer for a cuda
    requirements file includes device-less steps that are real CUDA suites.
    Family-scoping would drop them, so the widening stays."""
    sel = select(state, ["requirements/cuda.txt"])
    assert not sel.run_all
    assert "vllm_ci:kernels-attention-test" in sel.selected


def test_build_validated_manual_only_declarers_fall_open(state, declared_deps_on):
    """Same guarantee as the plain requirements rule: if every declarer is
    manual-only the floor would select nothing real, so fall open."""
    import dataclasses

    from ci_selector.codemap.classify import _classify, _source_dep_steps

    path = "requirements/lint.txt"
    declarers = _source_dep_steps(state, path)
    assert declarers, "fixture drift: nothing declares lint.txt"
    st2 = dataclasses.replace(state, auto_step_ids=state.auto_step_ids - declarers)
    claim = _classify(st2, path, None)
    assert claim.rule == "fail-open" and claim.run_all
