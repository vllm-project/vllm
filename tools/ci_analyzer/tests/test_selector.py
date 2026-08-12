# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end selection cases on the real checkout + todaymatcher units."""

import regex as re
from ci_analyzer.gitdiff import DiffFile, changed_paths
from ci_analyzer.policy import matches_run_all, matches_source_dependency
from ci_analyzer.select import select
from ci_analyzer.validate.todaymatcher import today_select


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


def test_csrc_is_world(state):
    sel = select(state, ["csrc/attention/attention_kernels.cu"])
    assert "vllm_ci" in sel.run_all
    assert "vllm_intel_ci" in sel.run_all  # csrc/ in its run_all too


def test_csrc_cpu_routes_to_declarers_plus_cpu_family(state):
    """csrc/cpu/ is cpu-exclusive and unclaimed: route via the blanket csrc/
    declarers plus the cpu family, not the bare complement (GPU jobs can't run it)."""
    sel = select(state, ["csrc/cpu/cpu_attn.cpp"])
    assert "vllm_ci" not in sel.run_all
    assert any(c.rule == "declared-deps" for c in sel.claims)
    assert "vllm_ci:torch-stable-abi-audit" in sel.selected
    assert "vllm_ci:CPU-Kernel Tests" in sel.selected
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
    """mm_preprocess feeds minimax's module-level @register_processor, so it genuinely
    reaches the multimodal tests: route via graph (not demoted), not run-all."""
    sel = select(state, ["vllm/models/minimax_m3/common/mm_preprocess.py"])
    assert "vllm_ci" not in sel.run_all
    assert any(c.rule == "graph" for c in sel.claims)
    labels = {
        s.label for p in state.pipelines for s in p.steps if s.step_id in sel.selected
    }
    assert any("Multi-Modal" in lbl for lbl in labels)


def test_package_data_device_scoping_trims_cross_device(state, repo, monkeypatch):
    """A device-named tuning config is unreadable on other devices, so its routing
    and family floor are device-scoped; disabling scope selects strictly more."""
    import ci_analyzer.hardware as hw

    cfgs = sorted(
        (repo / "vllm/model_executor/layers/fused_moe/configs").glob(
            "*device_name=NVIDIA_H*.json"
        )
    )
    assert cfgs, "no H-series tuning config at HEAD to exercise scoping"
    j = str(cfgs[0].relative_to(repo))
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


def test_engine_gate_on_worker_file_keeps_catching_job(state):
    """#49364: the engine-starting gate trims non-engine tests reached through the
    worker seam, but the catching job (V1 Sample + Logits) must survive."""
    sel = select(state, ["vllm/v1/worker/gpu/cudagraph_utils.py"])
    assert "vllm_ci" not in sel.run_all
    labels = {
        s.label for p in state.pipelines for s in p.steps if s.step_id in sel.selected
    }
    assert any("Sample" in lbl and "Logits" in lbl for lbl in labels), (
        "the #49364 catching job must survive the engine gate"
    )


def test_gpu_worker_namespace_reaches_cpu_jobs(state):
    """cpu_worker.py subclasses gpu_worker.Worker, so gpu-namespace changes must reach
    intel_cpu jobs (the old subtractive rule under-selected)."""
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
    """csrc/rocm is world for the ROCm pipeline AND must
    select vllm_ci's AMD mirror jobs (they run on the image built from it)."""
    sel = select(state, ["csrc/rocm/attention.cu"])
    amd_mirrors = [
        s.step_id
        for p in state.pipelines
        for s in p.steps
        if s.step_id in sel.selected and s.mirror_hw == "amd"
    ]
    assert len(amd_mirrors) > 20, len(amd_mirrors)


def test_renderer_table_parsed(state):
    """_VLLM_RENDERERS/_VLLM_TOKENIZERS are parsed, matching the dynamic-sites audit
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


def test_lm_eval_routes_by_declared_deps(state):
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


def test_no_classifier_drops_declared_source_deps(state):
    """A step declaring a path in source_file_dependencies must survive into that path's
    claim; dropping it is under-selection. Probes a deep new-subpackage file under every
    declared dir plus every declared file. Exempts run-all and authoritative-nothing
    claims; on a graph-known file catch-all (bare `vllm/`) declarers are omitted
    (specific-only)."""
    from ci_analyzer.select import (
        _DEP_UNION_EXEMPT,
        _classify,
        _graph_known,
        _source_dep_steps,
    )

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
    """The dep model understands a plain path, a directory prefix, and a `!`
    negation. Anything else (a glob, a per-file include, whatever lands next)
    silently never matches, which is how 59 kv_transfer exclusions sat
    unmodelled. Fail here rather than quietly diverging from the generator.

    `!!x` matters as much as a glob and in the opposite direction: split_deps
    lstrips every `!`, so we would read it as a full negation of x and skip the
    step, while a generator stripping one `!` gets a dep matching nothing and
    runs it."""
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
    from ci_analyzer.policy import step_declares

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


def test_catch_all_declarers_omitted_on_graph_known_leaf(state):
    """On a graph-known leaf the graph is authoritative, so a step declaring only
    bare `vllm/` is omitted (graph closure plus SPECIFIC declarers, not CI blanket)."""
    from ci_analyzer.select import _graph_known, _source_dep_steps

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
    from ci_analyzer.select import DiffContext, _classify

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
    from ci_analyzer.select import DiffContext, _classify

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
    from ci_analyzer.select import DiffContext, _classify

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
    from ci_analyzer.select import DiffContext, _classify

    ctx = DiffContext(
        base="x",
        head="y",
        status={"benchmarks/kernels/bench_totally_new.py": "A"},
    )
    claim = _classify(state, "benchmarks/kernels/bench_totally_new.py", ctx)
    assert claim.rule == "added-benchmark", claim.rule
    assert not claim.run_all and not claim.step_ids


def test_pyproject_world_unions_match_and_policy(state):
    """Drift oracle: pyproject is a run_all match on some pipelines AND an
    analyzer-policy world file on all of them, so the two sources must union.
    Upstream adding it to one pipeline's run_all_patterns once short-circuited
    the others away."""
    sel = select(state, ["pyproject.toml"])
    assert set(sel.run_all) == {p.config.name for p in state.pipelines}
    claim = next(c for c in sel.claims if c.rule == "world")
    assert claim.divergent == {
        p.config.name
        for p in state.pipelines
        if not matches_run_all(p.config, "pyproject.toml")
    }


def test_world_divergence_is_structural_and_named():
    """The disclosure contract, off the live config so a reword or an upstream
    run_all edit cannot rot it: policy breadth is reported as a set, and the
    one detail string shared by every pipeline must name which pipelines
    actually matched rather than blanket-claiming they all did."""
    from ci_analyzer.curated import EXTRA_WORLD_FILES
    from ci_analyzer.jobs.model import PipelineConfig
    from ci_analyzer.policy import classify_world

    assert EXTRA_WORLD_FILES, "no world-policy members left; the mechanism is dead"
    path = EXTRA_WORLD_FILES[0]
    alpha = PipelineConfig("alpha", "alpha.yaml", [], [path], [])
    beta = PipelineConfig("beta", "beta.yaml", [], [], [])

    mixed = classify_world(path, [alpha, beta])
    assert mixed.run_all == {"alpha", "beta"}
    assert mixed.divergent == {"beta"}
    assert "alpha" in mixed.detail and "beta" in mixed.detail

    everywhere = classify_world(path, [alpha])
    assert everywhere.run_all == {"alpha"} and not everywhere.divergent

    assert classify_world("vllm/config/__init__.py", [beta]) is None


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


def test_requirements_common_stays_world_run_all(state):
    """Broad requirements files are world-intercepted before the requirements rule."""
    sel = select(state, ["requirements/common.txt"])
    assert "vllm_ci" in sel.run_all


def test_ray_compat_blanket_still_declares_requirements(state):
    """The narrow-requirements no-run-all property rides on SOME auto-run step declaring
    a requirements/ blanket (ray_compat today); fail loudly if that disappears."""
    from ci_analyzer.policy import deps_match

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


# --- todaymatcher units (semantics mirror the ci-infra generator) ---


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


def test_plugin_package_file_selects_plugin_step(state, repo):
    """Entry-point-loaded plugin packages route to the steps that
    pip-install them instead of silently selecting nothing."""
    yaml_text = (repo / ".buildkite/test_areas/plugins.yaml").read_text()
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
    from ci_analyzer.select import _source_dep_steps

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


def test_optional_only_coverage_fails_open(state):
    """A vllm/ file whose entire coverage auto-runs nowhere (optional steps
    only) must run-all, not silently select zero auto steps."""
    specimen = "vllm/distributed/weight_transfer/sparse_nccl_engine.py"
    covering = "tests/distributed/test_weight_transfer.py"
    closure = state.full.graph.reverse_closure({specimen})
    assert covering in closure, "specimen closure changed: update"
    assert covering not in state.invoked, (
        "the covering test is now auto-invoked: pick a new optional-only specimen"
    )
    sel = select(state, [specimen])
    assert "vllm_ci" in sel.run_all
    assert "zero-closure polarity" in sel.run_all["vllm_ci"]
    assert any("2xh100-2xmi300" in s for s in sel.manual_hits)


def test_package_init_routes_to_package_steps(state):
    """Ancestor __init__.py auto-load edges: a test-package __init__ edit
    runs the package's own suites, neither run-all nor nothing."""
    sel = select(state, ["tests/basic_correctness/__init__.py"])
    assert not sel.run_all
    assert any("basic-correctness" in s for s in _selected(sel))


def test_worker_seam_reaches_conftest_server_suites(state):
    """A worker-seam diff must select suites whose engine boot happens in a
    conftest server fixture, not just direct entrypoint importers."""
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
    from ci_analyzer.select import PipelineData

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
    from ci_analyzer.curated import INERT_CI_PREFIXES

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

    from ci_analyzer.select import PipelineData, _classify_buildkite

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

    from ci_analyzer.jobs.testmap import StepTargets
    from ci_analyzer.select import PipelineData, _classify_buildkite

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


def test_class_table_module_keeps_zero_closure_run_all(state):
    """medusa.py is claimed by the class-table parser but its consumers are HF
    checkpoints, so with no test naming MedusaConfig it must still fail open run-all."""
    assert "MedusaConfig" in state.full.factories.class_table_entries
    sel = select(state, ["vllm/transformers_utils/configs/medusa.py"])
    assert "vllm_ci" in sel.run_all
    assert "zero-closure polarity" in sel.run_all["vllm_ci"]


def test_zero_closure_specimen_fails_open(state):
    """Zero-closure polarity: a vllm/ file whose coverage auto-runs nowhere must
    run-all, not silently select zero. Derived from HEAD (import-free source)."""
    rev = state.full.graph.reverse
    for file in state.full.index.file_to_module:
        if not (file.startswith("vllm/") and file.endswith(".py")):
            continue
        if rev.get(file):
            continue
        sel = select(state, [file])
        if "zero-closure polarity" in sel.run_all.get("vllm_ci", ""):
            return
    raise AssertionError("no import-free zero-closure specimen at HEAD")


def test_config_file_edit_scopes_to_one_pipeline(state):
    sel = select(state, [".buildkite/ci_config_intel.yaml"])
    assert set(sel.run_all) == {"vllm_intel_ci"}


def test_referenced_ci_script_selects_its_steps(state):
    sel = select(state, [".buildkite/scripts/hardware_ci/run-cpu-test.sh"])
    assert not sel.run_all
    always = {s.step_id for p in state.pipelines for s in p.steps if s.always_runs}
    assert set(sel.selected) - always


def test_unrecognized_ci_file_runs_all_pipelines(state):
    sel = select(state, [".buildkite/some_new_infra_thing.xyz"])
    assert set(sel.run_all) == {p.config.name for p in state.pipelines}


def test_added_test_without_owning_target_fails_open(state):
    """An added test in a brand-new area is a new uninvoked test and must run-all,
    never 'nothing to run' (where added-file rules hand back to polarity)."""
    from ci_analyzer.select import DiffContext, _classify

    path = "tests/brand_new_area/test_x.py"
    ctx = DiffContext(base="b", head="h", status={path: "A"})
    claim = _classify(state, path, ctx)
    assert claim.rule == "fail-open"
    assert claim.run_all == {p.config.name for p in state.pipelines}


def test_today_amd_native_runtime_branch(state, repo):
    """The replica's AMD branch: mi-device non-dind steps run on
    run-amd-test.sh diffs; dind or non-amd steps do not."""
    import dataclasses

    from ci_analyzer.validate.todaymatcher import step_should_run

    assert (repo / ".buildkite/scripts/hardware_ci/run-amd-test.sh").is_file(), (
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
    assert "vllm_ci:model-runner-v2-examples" in sel.selected


def test_pooling_runner_selects_examples_steps(state):
    sel = select(state, ["vllm/v1/worker/gpu/pool/pooling_runner.py"])
    assert "vllm_ci:examples" in sel.selected


def test_eagle_config_selects_examples_steps(state):
    sel = select(state, ["vllm/transformers_utils/configs/eagle.py"])
    assert "vllm_ci:examples" in sel.selected


# ---- declared-deps routing -------------------------------------------------

RUST_DECLARERS = {
    "vllm_ci:Rust Frontend Core Correctness",
    "vllm_ci:Rust Frontend Distributed",
    "vllm_ci:Rust Frontend OpenAI Coverage",
    "vllm_ci:Rust Frontend Serve/Admin Coverage",
    "vllm_ci:Rust Frontend Tool Use",
    "vllm_ci:rust-frontend-cargo-style-clippy",
    "vllm_ci:rust-frontend-cargo-tests",
}


def test_rust_file_routes_to_declared_steps(state):
    """rust/ is in rocm's run_all_patterns (rocm-only), so rocm runs everything while
    vllm_ci stays narrow; the declarers still ride along via the world claim."""
    sel = select(state, ["rust/frontend/src/lib.rs"])
    assert set(sel.run_all) == {"vllm_rocm_ci"}
    assert "vllm_ci" not in sel.run_all
    assert set(sel.selected) >= RUST_DECLARERS
    assert any(c.rule == "world" for c in sel.claims)


def test_rust_toolchain_routes_to_cargo_steps(state):
    sel = select(state, ["rust-toolchain.toml"])
    assert set(sel.run_all) == {"vllm_rocm_ci"}
    assert "vllm_ci" not in sel.run_all
    assert "vllm_ci:rust-frontend-cargo-tests" in sel.selected


def test_cmake_cpu_extension_routes_to_declarers_plus_cpu_family(state):
    """Real CI's route (the declaring steps) plus the CPU family whose suites compile
    the extension in-step without declaring it (previously excluded, then run-all)."""
    sel = select(state, ["cmake/cpu_extension.cmake"])
    assert not sel.run_all
    for sid in (
        "vllm_ci:torch-stable-abi-audit",
        "vllm_ci:CPU-Kernel Tests",
        "vllm_rocm_ci:CPU-Kernel Tests",
    ):
        assert sid in sel.selected, sid


def test_rocm_named_rust_file_selects_rust_steps(state):
    """The basename hardware-token heuristic is a Python/shell convention; a
    rocm-named rust file must keep its declaring h200 steps (.rs is not subject
    to amd-exclusive subtraction), even though rocm runs everything on rust/."""
    sel = select(state, ["rust/src/rocm_support.rs"])
    assert set(sel.run_all) == {"vllm_rocm_ci"}
    assert "vllm_ci" not in sel.run_all
    assert set(sel.selected) >= RUST_DECLARERS


def test_undeclared_oddball_still_fails_open(state):
    sel = select(state, ["tools/install_gdrcopy.sh"])
    assert set(sel.run_all) == {"vllm_ci", "vllm_intel_ci", "vllm_rocm_ci"}


def test_all_manual_declarers_fall_open(state):
    """Polarity guard: with no auto-run declarer the rule must not silently
    select nothing. cmake/cpu_extension.cmake is graph-blind, declared, and
    excluded from every run_all_patterns, so it reaches terminal fail-open
    (rust files can no longer test this: rocm runs everything on rust/)."""
    import dataclasses

    from ci_analyzer.select import _classify, _source_dep_steps

    path = "cmake/cpu_extension.cmake"
    declarers = _source_dep_steps(state, path)
    assert declarers, "fixture drift: no step declares cmake/cpu_extension.cmake"
    st2 = dataclasses.replace(state, auto_step_ids=state.auto_step_ids - declarers)
    claim = _classify(st2, path, None)
    assert claim.rule == "fail-open" and claim.run_all


def test_requirements_all_manual_declarers_fall_open(state):
    import dataclasses

    from ci_analyzer.select import _classify, _source_dep_steps

    path = "requirements/nightly_torch_test.txt"
    declarers = _source_dep_steps(state, path)
    assert declarers, "fixture drift: no step declares the nightly torch file"
    st2 = dataclasses.replace(state, auto_step_ids=state.auto_step_ids - declarers)
    claim = _classify(st2, path, None)
    assert claim.rule == "fail-open" and claim.run_all


def test_hipify_world_also_selects_declaring_abi_step(state):
    """World for vllm_rocm_ci only, but vllm_ci's torch-abi audit declares
    cmake/: real CI runs it, so must we."""
    sel = select(state, ["cmake/hipify.py"])
    assert set(sel.run_all) == {"vllm_rocm_ci"}
    assert "vllm_ci:torch-stable-abi-audit" in sel.selected


def test_family_exclusive_no_declarers_keeps_complement(state):
    """A family-exclusive path with no auto declarer keeps the device-family
    complement (the declarers-consult must not silence it)."""
    from ci_analyzer.select import _classify

    claim = _classify(state, "tools/rocm_env_check.sh", None)
    assert claim.rule == "fail-open" and not claim.run_all and claim.step_ids


def test_family_exclusive_inside_roots_keeps_complement(state):
    """The roots gate holds inside the scoped consult: a cpu-exclusive vllm asset with
    blanket vllm/ declarers keeps the complement, not the (narrowing) declarers."""
    from ci_analyzer.select import _classify, _source_dep_steps

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

    from ci_analyzer.select import _classify, _source_dep_steps

    path = "csrc/cpu/cpu_attn.cpp"
    declarers = _source_dep_steps(state, path)
    st2 = dataclasses.replace(state, auto_step_ids=state.auto_step_ids - declarers)
    claim = _classify(st2, path, None)
    assert claim.rule == "fail-open" and not claim.run_all and claim.step_ids


# ---- R3: target-coverage (tests/benchmarks-side zero-closure files) --------


def test_tests_side_script_zero_coverage_selects_nothing(state):
    """#50110: an unimported .sh in a directory no step covers or declares routes to
    nothing. Synthetic path because every real tests/*.sh now sits under a declared
    tree, so this exercises the truly-unclaimed branch."""
    from ci_analyzer.select import _classify, _source_dep_steps

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
    but shows those steps as manual hits (the _zero_auto_coverage hook)."""
    path = "tests/weight_loading/run_model_weight_loading_test.sh"
    sel = select(state, [path])
    assert not sel.run_all
    assert not _non_always(sel, state)
    assert sel.manual_hits


def test_added_init_under_covered_tests_dir_routes(state):
    """#50330 shape: an added tests __init__ under a directory a step targets
    routes to that step (not run-all)."""
    from ci_analyzer.select import _classify

    claim = _classify(state, "tests/v1/e2e/spec_decode/zz_new/__init__.py", None)
    assert claim.rule == "target-coverage"
    assert not claim.run_all and claim.step_ids & state.auto_step_ids


# ---- R4: added-trivial-init (added trivial __init__.py under vllm/) --------


def test_trivial_init_units():
    from ci_analyzer.select import _is_trivial_init

    assert _is_trivial_init("# SPDX header only\n")
    assert _is_trivial_init('"""a docstring"""\n')
    assert _is_trivial_init("")
    assert not _is_trivial_init("__all__ = []\n")
    assert not _is_trivial_init("import x\n")
    assert not _is_trivial_init("def f(:\n")  # syntax error -> not trivial


def test_added_trivial_init_routes_to_package_closure(state, monkeypatch):
    """#50131 shape: a new SPDX-only __init__ routes to its package subtree's
    reverse closure, not run-all."""
    from ci_analyzer import tablediff
    from ci_analyzer.select import DiffContext, _classify_added_init

    monkeypatch.setattr(tablediff, "git_show", lambda r, ref, p: "# SPDX\n")
    ctx = DiffContext(base="b", head="h", status={})
    claim = _classify_added_init(
        state, "vllm/model_executor/layers/mamba/ops/__init__.py", ctx
    )
    assert claim is not None and claim.rule == "added-trivial-init"
    assert claim.test_files and not claim.run_all


def test_added_nontrivial_or_new_package_init_falls_through(state, monkeypatch):
    from ci_analyzer import tablediff
    from ci_analyzer.select import DiffContext, _classify_added_init

    ctx = DiffContext(base="b", head="h", status={})
    monkeypatch.setattr(tablediff, "git_show", lambda r, ref, p: "import x\n")
    assert (
        _classify_added_init(
            state, "vllm/model_executor/layers/mamba/ops/__init__.py", ctx
        )
        is None
    )
    monkeypatch.setattr(tablediff, "git_show", lambda r, ref, p: "\n")
    assert _classify_added_init(state, "vllm/zz_brand_new_pkg/__init__.py", ctx) is None


def test_added_init_in_keyed_package_prefers_key_routing(state):
    """The added-in-claimed-package rule wins before added-trivial-init when
    the package is string-keyed (order guard)."""
    from ci_analyzer.select import DiffContext, _classify

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
    from ci_analyzer.select import _classify, _classify_declared_deps

    path = "vllm/model_executor/layers/fused_moe/configs/zzz_probe.json"
    assert _classify_declared_deps(state, path) is None
    claim = _classify(state, path, None)
    assert claim.rule == "package-data" and not claim.run_all


def test_release_refs_follow_one_level_of_indirection(repo):
    """A file a release script references (not named directly in the release yaml)
    is release-only too; one-level recursion catches manylinux.sh, so no run-all."""
    from ci_analyzer.curated import RELEASE_PIPELINE_FILES
    from ci_analyzer.externals import _REPO_PATH_RE, release_pipeline_refs

    refs = release_pipeline_refs(repo)
    direct: set[str] = set()
    for rel in RELEASE_PIPELINE_FILES:
        try:
            text = (repo / rel).read_text()
        except OSError:
            continue
        direct |= {m for m in _REPO_PATH_RE.findall(text) if (repo / m).is_file()}
    assert refs - direct, "recursion found no transitively-referenced release file"


def test_package_data_zero_auto_coverage_falls_open(state):
    """Polarity guard: a data file whose owning closure auto-runs nowhere and
    parses no family must keep the run-all fail-open, not select nothing."""
    import dataclasses

    from ci_analyzer.select import _classify

    st2 = dataclasses.replace(
        state, invoked=set(), auto_covered_files=set(), auto_prefixes=()
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
    assert "vllm_ci:Docker Build Metadata" in sel.selected
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
    from ci_analyzer.select import _classify

    path = "tools/vllm-rocm/generate-rocm-wheels-root-index.sh"
    claim = _classify(state, path, None)
    assert claim.rule == "release-ci" and not claim.run_all and not claim.step_ids


def test_release_refs_derived_and_disjoint_from_live_steps(state):
    """The derived release refs are non-empty, include the tools/ member, and reference
    nothing a modeled step consumes."""
    from ci_analyzer.curated import RELEASE_PIPELINE_FILES

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


def test_rocm_scripts_dir_stays_world(state):
    """Control: classify_world is undisturbed; a scripts/rocm/ file still
    runs its whole pipeline before any release-ci check."""
    sel = select(state, [".buildkite/scripts/rocm/ci-bake-rocm.sh"])
    assert "vllm_rocm_ci" in sel.run_all


def test_docker_input_relabel_still_run_all(state):
    """A docker-image COPY source keeps its run-all but says why."""
    from ci_analyzer.select import _classify

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
    import ci_analyzer.select as sel_mod
    from ci_analyzer.select import _diff_context

    files = [
        DiffFile("R", "new_r.py", old_path="old_r.py"),
        DiffFile("C", "new_c.py", old_path="old_c.py"),
    ]
    monkeypatch.setattr(sel_mod, "diff_files", lambda repo, base, head: files)
    ctx = _diff_context(state, "b", "h")
    assert ctx.renames == {"new_r.py": "old_r.py", "new_c.py": "old_c.py"}
    assert ctx.status["old_r.py"] == "D"
    assert "old_c.py" not in ctx.status


def test_renamed_path_claims_via_old_closure(state):
    """A renamed new path routes via the old path's base closure, not run-all."""
    from ci_analyzer.select import DiffContext, _classify

    old = "vllm/v1/spec_decode/eagle.py"
    new = "vllm/v1/spec_decode/eagle_renamed.py"
    ctx = DiffContext(base="b", head="h", status={new: "R"}, renames={new: old})
    claim = _classify(state, new, ctx)
    assert claim.rule == "renamed"
    assert not claim.run_all and claim.step_ids
    assert f"renamed from {old}" in claim.detail


def test_renamed_path_with_unclaimed_old_still_fails_open(state):
    """The old path being unclaimed carries run-all forward (never a double
    fail-open, polarity intact)."""
    from ci_analyzer.select import DiffContext, _classify

    new, old = "vllm/ghost_new.py", "vllm/ghost_zzz_helper.py"
    ctx = DiffContext(base="b", head="h", status={new: "R"}, renames={new: old})
    claim = _classify(state, new, ctx)
    assert claim.rule == "renamed" and claim.run_all


def test_copied_path_routes_via_source(state):
    from ci_analyzer.select import DiffContext, _classify

    old = "vllm/v1/spec_decode/eagle.py"
    new = "vllm/v1/spec_decode/eagle_copy.py"
    ctx = DiffContext(base="b", head="h", status={new: "C"}, renames={new: old})
    claim = _classify(state, new, ctx)
    assert claim.rule == "renamed" and "copied from" in claim.detail


def _head_stub(closure):
    from types import SimpleNamespace

    return SimpleNamespace(
        graph=SimpleNamespace(reverse_closure=lambda files, include_gated=True: closure)
    )


def test_added_head_closure_maps_to_base_steps(state, monkeypatch):
    """An added vllm/ file whose HEAD closure reaches a test under a base
    step's directory target routes there, not run-all."""
    import ci_analyzer.select as sel_mod
    from ci_analyzer.select import DiffContext, _classify

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
    import ci_analyzer.select as sel_mod
    from ci_analyzer.select import DiffContext, _classify

    monkeypatch.setattr(
        sel_mod, "_head_graph", lambda st, ctx: _head_stub({"vllm/a.py", "vllm/b.py"})
    )
    ctx = DiffContext(base="b", head="h", status={"vllm/newarea/brand_new.py": "A"})
    claim = _classify(state, "vllm/newarea/brand_new.py", ctx)
    assert claim.rule == "fail-open" and claim.run_all


def test_added_head_closure_unmappable_falls_through(state, monkeypatch):
    """A closure test in a brand-new dir no base step covers -> fail-open (the
    zero-step-mapping guard, never a silent empty claim)."""
    import ci_analyzer.select as sel_mod
    from ci_analyzer.select import DiffContext, _classify

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
    import ci_analyzer.select as sel_mod
    from ci_analyzer.select import _classify

    def boom(*a, **k):
        raise AssertionError("head graph must not be built without a diff ctx")

    monkeypatch.setattr(sel_mod, "_head_graph", boom)
    claim = _classify(state, "vllm/whatever_totally_new.py", None)
    assert claim.rule == "fail-open"


# --- item 3: exclusivity subtracts inferred reach, never direct collection ---


def _targets(**kw):
    from ci_analyzer.jobs.testmap import StepTargets

    st = StepTargets(step_id="s")
    for path, kind in kw.pop("targets", ()):
        st.add_target(path, kind)
    for k, v in kw.items():
        setattr(st, k, list(v))
    return st


def test_directly_collects_dir_target_covers_member():
    from ci_analyzer.select import _directly_collects

    st = _targets(targets=[("tests/v1/attention", "pytest")])
    assert _directly_collects(st, "tests/v1/attention/test_rocm_backends.py")
    assert not _directly_collects(st, "tests/v1/other/test_rocm_backends.py")
    assert not _directly_collects(None, "tests/v1/attention/test_rocm_backends.py")


def test_directly_collects_honours_ignore():
    """`pytest kernels/ --ignore=kernels/attention` never imports the ignored
    subtree, so it is not proof of collection and must not disarm the rule."""
    from ci_analyzer.select import _directly_collects

    st = _targets(
        targets=[("tests/kernels", "pytest")], ignored=["tests/kernels/attention"]
    )
    assert _directly_collects(st, "tests/kernels/test_rocm_misc.py")
    assert not _directly_collects(st, "tests/kernels/attention/test_rocm_mla.py")


def test_directly_collects_scripts_and_data():
    from ci_analyzer.select import _directly_collects

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
    from ci_analyzer import hardware
    from ci_analyzer.select import _directly_collects

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
    for sid, reasons in sel.selected.items():
        assert len(sel.selected_rules[sid]) == len(reasons), sid


def test_every_emitted_rule_is_pinned(state):
    from ci_analyzer.policy import OUTPUT_RULES

    sel = select(state, ["vllm/v1/attention/selector.py", "requirements/cuda.txt"])
    emitted = {r for rules in sel.selected_rules.values() for r in rules}
    assert emitted <= OUTPUT_RULES, emitted - OUTPUT_RULES
    assert "always-run" in emitted


def test_unpinned_rule_raises():
    import pytest
    from ci_analyzer.policy import Claim

    with pytest.raises(ValueError, match="unpinned claim rule"):
        Claim("not-a-rule", "detail")


def test_render_json_carries_rules(state):
    import json

    from ci_analyzer.report import render_json

    d = json.loads(render_json(select(state, ["vllm/v1/attention/selector.py"])))
    assert set(d["selected_rules"]) == set(d["selected"])


def test_rules_constant_matches_the_claims_actually_constructed():
    """Derived, not duplicated: scan for Claim(...) literals and compare. A new
    rule, a rename, or a removal all show up here without anyone maintaining a
    second list. RULES is a contract because pass 2 routes on these names."""
    import ast
    from pathlib import Path

    from ci_analyzer.policy import RULES

    src_dir = Path(__file__).resolve().parents[1] / "ci_analyzer"
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


def test_catch_all_declarers_omitted_on_graph_known_file(state):
    """The divergence itself: on a file the graph knows, a step whose only
    matching declaration is a bare `vllm/` is dropped, and the claim says so."""
    sel = select(state, ["vllm/v1/attention/selector.py"])
    detail = next(c.detail for c in sel.claims)
    assert "catch-all-only declarers omitted" in detail, detail


def test_unmodeled_dynamic_site_restores_catch_all_declarers(state):
    """The gate. The omission is justified by 'the import graph is
    authoritative'; an unmodeled dynamic import is evidence against exactly
    that, so the blanket declarers come back until it is classified."""
    import dataclasses

    from ci_analyzer.preflight import PreflightReport

    # A file whose selection is narrow enough that the blanket declarers are
    # not already in it: on a hub the restored steps are redundant and the gate
    # is invisible, which is the whole shape of the hub problem.
    path = "vllm/model_executor/layers/quantization/fp8.py"
    assert (state.repo / path).exists(), (
        f"{path} moved; pick another narrow-selection file for this case"
    )
    gated = dataclasses.replace(
        state,
        preflight=dataclasses.replace(
            state.preflight,
            unclassified_sites=frozenset({"vllm/somewhere/loader.py"}),
        ),
    )
    assert isinstance(gated.preflight, PreflightReport)
    before, after = _selected(select(state, [path])), _selected(select(gated, [path]))
    assert after > before, f"gate added nothing: {len(before)} -> {len(after)}"
    detail = next(c.detail for c in select(gated, [path]).claims)
    assert "catch-all declarers restored" in detail, detail


def test_gate_only_ever_adds_steps(state):
    """The gate is a suspension of a narrowing rule, so it can only widen. A
    version that removed anything would be an under-selection bug wearing a
    safety label."""
    import dataclasses

    gated = dataclasses.replace(
        state,
        preflight=dataclasses.replace(
            state.preflight, unclassified_sites=frozenset({"vllm/x/loader.py"})
        ),
    )
    for path in (
        "vllm/model_executor/layers/quantization/fp8.py",
        "vllm/v1/attention/selector.py",
        "vllm/entrypoints/openai/serving_chat.py",
        "tests/v1/attention/test_rocm_attention_backends_selection.py",
    ):
        before = _selected(select(state, [path]))
        after = _selected(select(gated, [path]))
        assert before <= after, (path, sorted(before - after))


def test_gate_does_not_fire_when_the_site_is_classified(state):
    """No unclassified site means no suspension: the saving is the default."""
    assert not state.preflight.unclassified_sites
    detail = next(
        c.detail for c in select(state, ["vllm/v1/attention/selector.py"]).claims
    )
    assert "catch-all declarers restored" not in detail
