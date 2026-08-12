# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Table-granular world-ness: pure classifiers, the real-repo tool oracle,
and the classify_world gate / select wiring."""

import regex as re
from ci_analyzer import select as select_mod
from ci_analyzer import worldtables
from ci_analyzer.curated import EXTRA_WORLD_FILES
from ci_analyzer.gitdiff import DiffFile
from ci_analyzer.jobs.model import PipelineConfig
from ci_analyzer.policy import classify_world
from ci_analyzer.select import select
from ci_analyzer.worldtables import (
    _build_tool_pkgs,
    _config_only,
    changed_sections,
    narrowable_world_paths,
)

BUILD_PKGS = {"setuptools", "setuptools-scm"}
REFERENCED = {"pytest", "uv"}


# ---- changed_sections ------------------------------------------------------


def test_changed_sections_build_system_only():
    old = '[build-system]\nrequires = ["setuptools"]\n[tool.ruff]\nx = 1\n'
    new = '[build-system]\nrequires = ["setuptools", "wheel"]\n[tool.ruff]\nx = 1\n'
    assert changed_sections(old, new) == {"build-system"}


def test_changed_sections_drills_under_tool():
    old = "[tool.ruff]\nx = 1\n[tool.mypy]\nstrict = true\n"
    new = "[tool.ruff]\nx = 2\n[tool.mypy]\nstrict = true\n"
    assert changed_sections(old, new) == {"tool.ruff"}


def test_changed_sections_mixed():
    old = '[build-system]\nrequires = ["a"]\n[tool.ruff]\nx = 1\n'
    new = '[build-system]\nrequires = ["b"]\n[tool.ruff]\nx = 2\n'
    assert changed_sections(old, new) == {"build-system", "tool.ruff"}


def test_changed_sections_added_table():
    old = "[tool.ruff]\nx = 1\n"
    new = "[tool.ruff]\nx = 1\n[tool.newtool]\ny = 2\n"
    assert changed_sections(old, new) == {"tool.newtool"}


def test_changed_sections_identical_is_empty():
    text = '[build-system]\nrequires = ["a"]\n'
    assert changed_sections(text, text) == set()


def test_changed_sections_unparsable_is_none():
    assert changed_sections("not = = valid", "[tool.ruff]\nx = 1\n") is None


# ---- _build_tool_pkgs ------------------------------------------------------


def test_build_tool_pkgs_from_backend_and_requires():
    text = (
        "[build-system]\n"
        'requires = ["setuptools>=77", "setuptools-scm>=8", "torch == 2.13"]\n'
        'build-backend = "setuptools.build_meta"\n'
    )
    assert _build_tool_pkgs(text) == {"setuptools", "setuptools-scm", "torch"}


# ---- _config_only ----------------------------------------------------------


def test_config_only_lint_tables_narrow():
    assert _config_only({"tool.ruff", "tool.mypy"}, BUILD_PKGS, REFERENCED) is True


def test_config_only_unknown_unreferenced_tool_narrows():
    assert _config_only({"tool.somethingnew"}, BUILD_PKGS, REFERENCED) is True


def test_config_only_build_tables_force_world():
    assert _config_only({"build-system"}, BUILD_PKGS, REFERENCED) is False
    assert _config_only({"project"}, BUILD_PKGS, REFERENCED) is False


def test_config_only_build_tool_forces_world():
    # tool.setuptools_scm normalizes to setuptools-scm, which is a build pkg.
    assert _config_only({"tool.setuptools_scm"}, BUILD_PKGS, REFERENCED) is False


def test_config_only_step_consumed_tool_forces_world():
    assert _config_only({"tool.pytest"}, BUILD_PKGS, REFERENCED) is False


def test_config_only_unknown_top_level_forces_world():
    assert _config_only({"dependency-groups"}, BUILD_PKGS, REFERENCED) is False


def test_config_only_any_world_section_forces_world():
    assert _config_only({"tool.ruff", "build-system"}, BUILD_PKGS, REFERENCED) is False


# ---- _referenced_tools: real-repo oracle -----------------------------------


def test_referenced_tools_has_pytest_not_lint(state):
    """Buildkite steps run pytest (28 test areas) but not ruff/mypy (those live
    in GitHub Actions), so tool.pytest stays world while tool.ruff can narrow."""
    ref = worldtables._referenced_tools(state)
    assert "pytest" in ref
    assert "ruff" not in ref and "mypy" not in ref


def test_referenced_tools_splits_dotted_tokens(state):
    """A tool named only as a filename must still count as referenced.
    _normalize folds `mypy.sh` to `mypy-sh`, which matches no table name, so
    without the split its table would narrow away from the step running it."""
    ref = worldtables._referenced_tools(state)
    corpus = "\n".join(t.haystack for p in state.pipelines for t in p.targets.values())
    stems = {
        stem
        for tok in re.findall(r"[A-Za-z0-9_.-]+", corpus)
        if "." in tok
        for stem in [worldtables._normalize(tok.split(".")[0])]
        if any(c.isalpha() for c in stem)  # a table name has letters
    }
    assert stems, "no dotted tokens in the step corpus: specimen gone"
    assert stems <= ref


# ---- classify_world gate ---------------------------------------------------


def test_classify_world_policy_gate():
    path = EXTRA_WORLD_FILES[0]
    match = PipelineConfig("rocm", "rocm.yaml", [], [path], [])
    silent = PipelineConfig("ci", "ci.yaml", [], [], [])

    on = classify_world(path, [match, silent], policy_world=True)
    assert on.run_all == {"rocm", "ci"} and on.divergent == {"ci"}

    off = classify_world(path, [match, silent], policy_world=False)
    assert off.run_all == {"rocm"} and not off.divergent


# ---- _pyproject_config_only / narrowable_world_paths (git_show stubbed) -----


def _stub_show(monkeypatch, old, new):
    sides = iter([old, new])
    monkeypatch.setattr(worldtables, "git_show", lambda *a, **k: next(sides))


def test_pyproject_config_only_narrows_tool_edit(state, monkeypatch):
    _stub_show(monkeypatch, "[tool.ruff]\nx = 1\n", "[tool.ruff]\nx = 2\n")
    assert worldtables._pyproject_config_only(state, "b", "h", "pyproject.toml")


def test_pyproject_config_only_keeps_world_on_build_edit(state, monkeypatch):
    _stub_show(
        monkeypatch,
        '[build-system]\nrequires = ["a"]\n',
        '[build-system]\nrequires = ["b"]\n',
    )
    assert not worldtables._pyproject_config_only(state, "b", "h", "pyproject.toml")


def test_pyproject_config_only_missing_side_keeps_world(state, monkeypatch):
    monkeypatch.setattr(worldtables, "git_show", lambda *a, **k: None)
    assert not worldtables._pyproject_config_only(state, "b", "h", "pyproject.toml")


def test_narrowable_world_paths_config_only(state, monkeypatch):
    _stub_show(monkeypatch, "[tool.ruff]\nx = 1\n", "[tool.ruff]\nx = 2\n")
    assert narrowable_world_paths(state, "b", "h", {"pyproject.toml"}) == {
        "pyproject.toml"
    }


def test_narrowable_world_paths_skips_unchanged(state):
    assert narrowable_world_paths(state, "b", "h", {"vllm/foo.py"}) == set()


# ---- end-to-end select wiring ----------------------------------------------


def _patch_diff(monkeypatch, narrowable):
    monkeypatch.setattr(
        select_mod, "diff_files", lambda *a, **k: [DiffFile("M", "pyproject.toml")]
    )
    monkeypatch.setattr(
        select_mod, "narrowable_world_paths", lambda *a, **k: narrowable
    )


def test_select_narrows_config_only_pyproject(state, monkeypatch):
    _patch_diff(monkeypatch, {"pyproject.toml"})
    sel = select(state, ["pyproject.toml"], base="b", head="h")
    assert "vllm_ci" not in sel.run_all
    assert "vllm_rocm_ci" in sel.run_all


def test_select_keeps_world_on_build_pyproject(state, monkeypatch):
    _patch_diff(monkeypatch, set())
    sel = select(state, ["pyproject.toml"], base="b", head="h")
    assert set(sel.run_all) == {p.config.name for p in state.pipelines}
