# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The declaration switch: the hand-written source_file_dependencies lists
pick steps only through `_source_dep_steps`, which CI_SELECTOR_DECLARED_DEPS=off
silences. Three reads ignore the switch by design, and the whitelist below
fails when a fourth appears or one of the three disappears."""

import ast
from pathlib import Path

import ci_selector
import pytest
from ci_selector.codemap import classify
from ci_selector.codemap.step_refs import (
    ENV_VAR,
    _source_dep_steps,
    _source_dep_steps_ungated,
    mode,
)

PKG = Path(ci_selector.__file__).parent

# A file the pipelines declare directly.
DECLARED_PROBE = "cmake/cpu_extension.cmake"


def test_mode_defaults_off_and_rejects_typos(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert mode() == "off", "the default is the fully-derived main path"
    monkeypatch.setenv(ENV_VAR, "on")
    assert mode() == "on"
    monkeypatch.setenv(ENV_VAR, "of")
    with pytest.raises(ValueError):
        mode()


def test_the_gate_is_silent_by_default_and_open_when_switched_on(state, monkeypatch):
    raw = _source_dep_steps_ungated(state, DECLARED_PROBE)
    assert raw, f"{DECLARED_PROBE} lost its declarers; pick a new probe"
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert _source_dep_steps(state, DECLARED_PROBE) == set()
    assert _source_dep_steps_ungated(state, DECLARED_PROBE) == raw
    monkeypatch.setenv(ENV_VAR, "on")
    assert _source_dep_steps(state, DECLARED_PROBE) == raw


def _calls_of(tree: ast.AST, name: str) -> int:
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Name) and f.id == name) or (
                isinstance(f, ast.Attribute) and f.attr == name
            ):
                n += 1
    return n


def test_raw_reads_are_whitelisted_in_both_directions():
    """A new caller is a declaration read the switch cannot silence; a
    vanished one means a guard died. Both fail here."""
    raw_calls: dict[str, int] = {}
    attr_reads: dict[str, int] = {}
    for py in sorted(PKG.rglob("*.py")):
        rel = py.relative_to(PKG).as_posix()
        tree = ast.parse(py.read_text())
        n = _calls_of(tree, "_source_dep_steps_ungated")
        if n:
            raw_calls[rel] = n
        a = sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr == "source_file_dependencies"
        )
        if a:
            attr_reads[rel] = a
    # step_refs: the gate's own call. classify: requirements twice, the
    # release-ci and lint-only guards, the csrc hold (which can only keep a
    # step), and the inert veto (which can only keep run-all: a declared file
    # must never be silenced to the floor).
    assert raw_calls == {
        "codemap/step_refs.py": 1,
        "codemap/classify.py": 6,
        "codemap/unions.py": 1,
    }, f"raw declaration reads moved: {raw_calls}"
    assert set(attr_reads) == {
        "codemap/step_refs.py",
        "codemap/pipeline/buildkite.py",
        "validate/crosscheck.py",
        "validate/generator_replica.py",
    }, f"Step.source_file_dependencies read outside the gate: {attr_reads}"


def test_switch_off_keeps_the_requirements_route(state, monkeypatch):
    monkeypatch.setenv(ENV_VAR, "off")
    claim = classify._classify_requirements(state, "requirements/lint.txt")
    assert claim is not None and claim.rule == "requirements"
    assert claim.step_ids & state.auto_step_ids


# --- lane 1: derived reference legs -----------------------------------------


def _cargo_suite(state) -> set[str]:
    return {
        sid
        for p in state.pipelines
        for sid, st in p.targets.items()
        if any(s.endswith("run-rust-frontend-cargo-ci.sh") for s in st.scripts_seen)
    } & state.auto_step_ids


def test_rust_files_keep_the_cargo_suite_without_declarations(state, monkeypatch):
    cargo = _cargo_suite(state)
    assert cargo, "no step invokes run-rust-frontend-cargo-ci.sh; probe died"
    monkeypatch.setenv(ENV_VAR, "off")
    claim = classify._classify(state, "rust/Cargo.toml", None)
    assert cargo <= claim.step_ids


def test_rust_reference_leg_over_match_floor(state):
    """The leg reaches the cargo suite plus steps that run anyway; anything
    else is an over-match to inspect."""
    from ci_selector.handwritten import RUST_GATE_ENV_VARS, RUST_TOOLCHAIN_FILES

    leg = (
        state.keys.steps_naming_raw({"rust/", *RUST_TOOLCHAIN_FILES})
        & state.auto_step_ids
    )
    cargo = _cargo_suite(state)
    assert cargo <= leg
    steps_by_id = {s.step_id: s for p in state.pipelines for s in p.steps}
    gate = state.keys.steps_naming_raw(set(RUST_GATE_ENV_VARS))
    # These declare docker/Dockerfile, which runs build_rust.sh, so a rust
    # change really does reach them. Named, so a different stray still shows.
    via_dockerfile = {
        "vllm_ci::computer: (CPU) Docker Build Metadata",
        "vllm_ci::computer: (CPU) Docker Build Metadata (amd):amd",
    }
    stray = {
        s for s in leg - cargo - gate - via_dockerfile if not steps_by_id[s].always_runs
    }
    assert not stray, f"rust reference leg over-matches: {sorted(stray)}"


def test_untargeted_example_asset_routes_by_workdir_affinity(state, monkeypatch):
    from ci_selector.codemap.step_refs import _direct_step_refs

    probe = None
    for p in sorted(state.repo.glob("examples/**/*.jinja")):
        rel = p.relative_to(state.repo).as_posix()
        if not _direct_step_refs(state, rel):
            probe = rel
            break
    if probe is None:
        pytest.skip("every examples jinja is directly referenced")
    monkeypatch.setenv(ENV_VAR, "off")
    claim = classify._classify(state, probe, None)
    assert not claim.run_all, f"{probe} escalated to run-all with the switch off"
    affinity = classify._workdir_affinity_steps(state, probe)
    assert affinity & state.auto_step_ids
    assert affinity <= claim.step_ids


def test_graph_known_untargeted_example_keeps_its_tree_step(state, monkeypatch):
    from ci_selector.codemap.state import _graph_known
    from ci_selector.codemap.step_refs import _direct_step_refs

    probe = None
    for p in sorted(state.repo.glob("examples/**/*.py")):
        rel = p.relative_to(state.repo).as_posix()
        if _graph_known(state, rel) and not _direct_step_refs(state, rel):
            probe = rel
            break
    if probe is None:
        pytest.skip("every graph-known example is directly referenced")
    monkeypatch.setenv(ENV_VAR, "off")
    claim = classify._classify(state, probe, None)
    affinity = classify._workdir_affinity_steps(state, probe)
    assert affinity & state.auto_step_ids
    assert affinity <= claim.step_ids
    assert not (affinity & claim.droppable_step_ids), (
        "affinity steps must not be droppable"
    )


# --- lane 2: the native-tests rule -----------------------------------------


def test_native_tests_routes_a_joined_tu_by_its_op_joints(state):
    fired = 0
    for p in sorted(state.native_ops.file_ops):
        if not state.native_ops.owns(p):
            continue
        claim = classify._classify_native_tests(state, p)
        if claim is None:
            continue
        fired += 1
        assert claim.rule == "native-tests"
        assert claim.step_ids & state.auto_step_ids
        assert not claim.droppable_step_ids and not claim.evidence_paths
    assert fired, "no owned csrc file fires native-tests; rule is dead"


def test_native_tests_stands_down_with_the_op_switch_off(state, monkeypatch):
    """Off means the op parse is not trusted, so this rule declines too
    rather than using evidence the switch just turned off."""
    monkeypatch.setenv("CI_SELECTOR_CSRC_OPS", "off")
    for p in sorted(state.native_ops.file_ops):
        if state.native_ops.owns(p):
            assert classify._classify_native_tests(state, p) is None
            return
    raise AssertionError("no owned csrc file to probe")


def test_native_tests_declines_outside_its_scope(state):
    assert classify._classify_native_tests(state, "csrc/cpu/cpu_attn.cpp") is None
    assert classify._classify_native_tests(state, "cmake/hipify.py") is None
    assert classify._classify_native_tests(state, "CMakeLists.txt") is None


def test_native_tests_declines_without_derived_evidence(state):
    """A csrc file with no ops and no references falls through, rather than
    firing on its family alone."""
    tracked = {p for p in state.native_ops.file_ops if state.native_ops.owns(p)}
    for p in sorted(state.repo.glob("csrc/**/*")):
        rel = p.relative_to(state.repo).as_posix()
        if not state.native_ops.owns(rel) or rel in tracked or not p.is_file():
            continue
        if state.native_ops.test_files_for(rel):
            continue
        from ci_selector.codemap.step_refs import _direct_step_refs

        if _direct_step_refs(state, rel):
            continue
        assert classify._classify_native_tests(state, rel) is None
        return
    pytest.skip("every owned csrc file currently derives evidence")


def test_switch_off_only_loses_declarers_the_build_map_rules_out(state, monkeypatch):
    """A cuda-only kernel keeps its xpu declarer with the switch on and loses
    it with the switch off, where the build map proves the file never compiles
    into that step."""
    if state.preflight.unmapped_devices:
        pytest.skip("the build map stood down, so every family is kept")
    # Test steps only: the xpu image builder carries no device name, so it
    # rides the remainder under cuda scoping.
    intel = {
        s
        for s in state.auto_step_ids
        if s.startswith("vllm_intel_ci:") and "image-build" not in s
    }
    probe = None
    for p in sorted(state.native_ops.file_ops):
        if not state.native_ops.owns(p):
            continue
        if state.build_map.families.get(p) != frozenset({"cuda"}):
            continue
        if not (_source_dep_steps_ungated(state, p) & intel):
            continue
        claim = classify._classify_native_tests(state, p)
        # Only promised where the intel steps ride the declaration alone. A
        # file whose op tests intel also runs keeps them, which is not a miss.
        if claim is None or claim.step_ids & intel:
            continue
        probe = p
        break
    assert probe, "no cuda-only file whose intel steps are declaration-only"
    monkeypatch.setenv(ENV_VAR, "on")
    on = classify._classify(state, probe, None)
    assert on.step_ids & intel, "switched on, the declared intel step is gone"
    monkeypatch.setenv(ENV_VAR, "off")
    off = classify._classify(state, probe, None)
    assert not (off.step_ids & intel), "the intel step should have gone"
    assert off.step_ids & state.auto_step_ids, "switched off, nothing is selected"
    assert off.step_ids <= on.step_ids, "the declared lists must only ever add"


def test_switch_off_still_never_selects_nothing_for_a_release_file(state, monkeypatch):
    """A release-referenced file with live declarers must not select nothing
    with the switch off, whether it re-routes or the guard adds the declarer
    steps back to the zero-claim."""
    live = [
        p
        for p in sorted(state.release_refs)
        if _source_dep_steps_ungated(state, p) & state.auto_step_ids
    ]
    assert live, "no release-referenced file with live declarers; probe died"
    monkeypatch.setenv(ENV_VAR, "off")
    for p in live[:2]:
        claim = classify._classify(state, p, None)
        want = _source_dep_steps_ungated(state, p) & state.auto_step_ids
        assert claim.run_all or want <= claim.step_ids
