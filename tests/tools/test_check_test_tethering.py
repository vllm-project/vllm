# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the test-tethering pre-commit check.

The check answers one question - "does any Buildkite job (in
``.buildkite/test_areas/`` or the legacy ``.buildkite/test-amd.yaml``) collect
this test file?" - by parsing job ``commands`` as shell. The risk that
matters is a *false tether*: a misparse that invents coverage makes the gate
pass while the test still never runs, which is the exact failure the check
exists to catch. So the parser cases below pin both directions, and
:func:`test_no_selection_matches_everything` guards the catastrophic version of
it.
"""

import textwrap
from pathlib import Path

import pytest

import tools.pre_commit.check_test_tethering as checker
from tools.pre_commit.check_test_tethering import (
    FindSelection,
    PytestSelection,
    _parse_command,
    _to_repo_relative,
    all_test_modules,
    allowlist_entries_missing_reason,
    is_test_module,
    is_tethered,
    load_allowlist,
    load_selections,
    main,
    normalize_test_path,
    run_changed_files_check,
    run_full_scan,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def runs(command: str, test_file: str) -> bool:
    """True if `command` collects `test_file` (path relative to ``tests/``)."""
    return any(sel.runs(test_file) for sel in _parse_command(command))


# --------------------------------------------------------------------------- #
# pytest path arguments
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("command", "test_file", "expected"),
    [
        # Directory sweeps cover everything nested underneath.
        ("pytest -v -s kernels", "kernels/test_attention.py", True),
        ("pytest -v -s kernels/", "kernels/moe/test_moe.py", True),
        ("pytest -v -s kernels", "lora/test_lora.py", False),
        # An exact file argument covers that file and nothing beside it.
        ("pytest -v -s kernels/test_a.py", "kernels/test_a.py", True),
        ("pytest -v -s kernels/test_a.py", "kernels/test_b.py", False),
        # Commands run from the repo root and from tests/ alike.
        ("pytest -v -s tests/kernels/test_a.py", "kernels/test_a.py", True),
        ("pytest -v -s ./kernels/test_a.py", "kernels/test_a.py", True),
        (
            "pytest -v -s /vllm-workspace/tests/kernels/test_a.py",
            "kernels/test_a.py",
            True,
        ),
        # A ::nodeid selector still collects the file.
        ("pytest -v -s kernels/test_a.py::test_one", "kernels/test_a.py", True),
        ("pytest -v -s kernels/test_a.py::Cls::test_one", "kernels/test_a.py", True),
        # Several positional paths, including a bare root-level file.
        ("pytest -v -s test_envs.py test_outputs.py", "test_outputs.py", True),
        ("pytest -v -s test_envs.py test_outputs.py", "test_regression.py", False),
        # A bare word is a path only when it is really a directory under tests/.
        ("pytest -v -s samplers", "samplers/test_sampler.py", True),
        # Glob path args do not cross a directory separator, matching the shell.
        ("pytest -v -s kernels/test_*.py", "kernels/test_a.py", True),
        ("pytest -v -s kernels/test_*.py", "kernels/moe/test_a.py", False),
        ("pytest -v -s kernels/test_*.py", "kernels/helper.py", False),
    ],
)
def test_pytest_path_arguments(command, test_file, expected):
    assert runs(command, test_file) is expected


# --------------------------------------------------------------------------- #
# --ignore / --deselect
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("command", "test_file", "expected"),
    [
        # --ignore removes a file from an otherwise covering sweep, in both the
        # `=` and space-separated spellings.
        ("pytest -v -s kernels --ignore=kernels/test_a.py", "kernels/test_a.py", False),
        ("pytest -v -s kernels --ignore kernels/test_a.py", "kernels/test_a.py", False),
        # ...and only that file.
        ("pytest -v -s kernels --ignore=kernels/test_a.py", "kernels/test_b.py", True),
        # --ignore of a directory removes the whole subtree.
        ("pytest -v -s kernels --ignore=kernels/moe", "kernels/moe/test_moe.py", False),
        (
            "pytest -v -s lora --ignore-glob=lora/*_tp.py",
            "lora/test_llama_tp.py",
            False,
        ),
        # A node-id deselect leaves the file collected; a bare-path one does not.
        (
            "pytest -v -s kernels --deselect kernels/test_a.py::test_one",
            "kernels/test_a.py",
            True,
        ),
        (
            "pytest -v -s kernels --deselect kernels/test_a.py",
            "kernels/test_a.py",
            False,
        ),
    ],
)
def test_ignore_and_deselect(command, test_file, expected):
    assert runs(command, test_file) is expected


# --------------------------------------------------------------------------- #
# Options, env prefixes and other token noise
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("command", "test_file", "expected"),
    [
        # A -k / -m filter narrows which tests run but does not un-run the file.
        ("pytest -v -s kernels -k 'not slow'", "kernels/test_a.py", True),
        ("pytest -v -s kernels -m core_model", "kernels/test_a.py", True),
        # A value-taking option's value is never mistaken for a path.
        ("pytest -m core_model", "core_model", False),
        ("pytest -k kernels", "kernels/test_a.py", False),
        ("pytest -n 4 kernels/test_a.py", "kernels/test_a.py", True),
        # Buildkite shard flags carry a $$VAR and must not swallow the path.
        (
            "pytest -v -s kernels --shard-id=$$BUILDKITE_PARALLEL_JOB "
            "--num-shards=$$BUILDKITE_PARALLEL_JOB_COUNT",
            "kernels/test_a.py",
            True,
        ),
        # A leading env-var assignment is not a path.
        (
            "VLLM_TEST_FORCE_LOAD_FORMAT=auto pytest -v -s kernels",
            "kernels/test_a.py",
            True,
        ),
        # `python -m pytest` is the same invocation.
        ("python3 -m pytest -v -s kernels/test_a.py", "kernels/test_a.py", True),
        # Quoted paths, wrapper commands and redirects all survive tokenizing.
        ("pytest -v -s 'kernels/test_a.py'", "kernels/test_a.py", True),
        ("timeout 600 pytest kernels/test_a.py", "kernels/test_a.py", True),
        ("pytest kernels/test_a.py > out.log 2>&1", "kernels/test_a.py", True),
        # Each sub-command of a compound line is classified on its own.
        ("pytest kernels/test_a.py && pytest lora/test_b.py", "lora/test_b.py", True),
        ("pytest kernels/test_a.py ; pytest lora/test_b.py", "lora/test_b.py", True),
        # An unresolvable $VAR path yields no coverage rather than a guess.
        ("pytest -v -s $TEST_TARGET", "kernels/test_a.py", False),
    ],
)
def test_option_and_token_handling(command, test_file, expected):
    assert runs(command, test_file) is expected


# --------------------------------------------------------------------------- #
# find | xargs pytest pipelines
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("command", "test_file", "expected"),
    [
        ("find kernels -name 'test_*.py' | xargs pytest -v", "kernels/test_a.py", True),
        (
            "find kernels -name 'test_*.py' | xargs pytest -v",
            "kernels/moe/test_b.py",
            True,
        ),
        # A non-matching filename is not collected by the find expression.
        (
            "find kernels -name 'test_*.py' | xargs pytest -v",
            "kernels/helper.py",
            False,
        ),
        # -maxdepth 1 keeps direct children and drops nested ones.
        (
            "find kernels -maxdepth 1 -name 'test_*.py' | xargs pytest",
            "kernels/test_a.py",
            True,
        ),
        (
            "find kernels -maxdepth 1 -name 'test_*.py' | xargs pytest",
            "kernels/moe/test_b.py",
            False,
        ),
        # -not -name excludes its target and leaves siblings alone.
        (
            "find kernels -name 'test_*.py' -not -name 'test_skip.py' | xargs pytest",
            "kernels/test_skip.py",
            False,
        ),
        (
            "find kernels -name 'test_*.py' -not -name 'test_skip.py' | xargs pytest",
            "kernels/test_a.py",
            True,
        ),
        # The root confines the sweep.
        ("find kernels -name 'test_*.py' | xargs pytest", "lora/test_a.py", False),
        # -maxdepth is honored even when the find root is the tests/ root itself
        # ("." or "tests/"), not just a subdir.
        (
            "find . -maxdepth 1 -name 'test_*.py' | xargs pytest",
            "test_regression.py",
            True,
        ),
        (
            "find . -maxdepth 1 -name 'test_*.py' | xargs pytest",
            "v1/test_scheduler.py",
            False,
        ),
        (
            "find tests/ -maxdepth 1 -name 'test_*.py' | xargs pytest",
            "v1/test_scheduler.py",
            False,
        ),
    ],
)
def test_find_pipelines(command, test_file, expected):
    assert runs(command, test_file) is expected


def test_find_pipeline_parses_as_find_selection():
    (selection,) = [
        s
        for s in _parse_command("find kernels -name 'test_*.py' | xargs pytest -v")
        if isinstance(s, FindSelection)
    ]
    assert selection.root == "kernels"
    assert selection.name_globs == ["test_*.py"]


# --------------------------------------------------------------------------- #
# Direct file runners
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("command", "test_file", "expected"),
    [
        (
            "python3 standalone_tests/lazy_imports.py",
            "standalone_tests/lazy_imports.py",
            True,
        ),
        (
            "python standalone_tests/lazy_imports.py",
            "standalone_tests/lazy_imports.py",
            True,
        ),
        (
            "torchrun --nproc-per-node 2 distributed/test_a.py",
            "distributed/test_a.py",
            True,
        ),
        # The option's value is a count, not a path.
        ("torchrun --nproc-per-node 2 distributed/test_a.py", "2", False),
        (
            "VLLM_X=1 python3 standalone_tests/lazy_imports.py | grep -q ok",
            "standalone_tests/lazy_imports.py",
            True,
        ),
    ],
)
def test_direct_runners(command, test_file, expected):
    assert runs(command, test_file) is expected


# --------------------------------------------------------------------------- #
# Commands that must contribute no coverage
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "command",
    [
        "export VLLM_WORKER_MULTIPROC_METHOD=spawn",
        "pip install pytest-timeout pytest-forked",
        "uv pip install -r requirements/test/cuda.in",
        "bash tests/does_not_exist.sh",
        "echo 'pytest kernels/test_a.py'",
        'pytest -v -s "unbalanced',
        # A `find` whose output never reaches pytest is not a test selection.
        "find kernels -name 'test_*.py'",
        "find kernels -name 'test_*.py' | wc -l",
        "find kernels -name 'test_*.py' -delete",
    ],
)
def test_non_test_commands_contribute_nothing(command):
    assert _parse_command(command) == [] or not any(
        sel.runs("kernels/test_a.py") for sel in _parse_command(command)
    )


# --------------------------------------------------------------------------- #
# Job-level yaml parsing
# --------------------------------------------------------------------------- #


def _write_area(tmp_path, monkeypatch, body: str):
    mod = checker
    area_dir = tmp_path / "test_areas"
    area_dir.mkdir(exist_ok=True)
    (area_dir / "synthetic.yaml").write_text(textwrap.dedent(body))
    monkeypatch.setattr(mod, "TEST_AREAS_DIR", area_dir)
    # Keep the real test-amd.yaml out of these isolated parsing tests.
    monkeypatch.setattr(mod, "TEST_AMD_YAML", tmp_path / "no-such-test-amd.yaml")
    # The yaml-error path reports a repo-relative name, so REPO_ROOT has to
    # contain the synthetic area too.
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    return load_selections()


def test_plain_step_commands_are_collected(tmp_path, monkeypatch):
    selections = _write_area(
        tmp_path,
        monkeypatch,
        """
        steps:
        - label: ":nvidia: (H200) Kernels"
          commands:
          - pytest -v -s kernels
        """,
    )
    assert is_tethered("kernels/test_a.py", selections)


def test_singular_command_key_is_collected(tmp_path, monkeypatch):
    selections = _write_area(
        tmp_path,
        monkeypatch,
        """
        steps:
        - label: "Singular"
          command: pytest -v -s kernels/test_a.py
        """,
    )
    assert is_tethered("kernels/test_a.py", selections)


def test_test_amd_yaml_coverage_counts_as_tethered(tmp_path, monkeypatch):
    """A test wired only into the legacy test-amd.yaml is still run by CI."""
    mod = checker
    (tmp_path / "test_areas").mkdir()
    amd_yaml = tmp_path / "test-amd.yaml"
    amd_yaml.write_text(
        textwrap.dedent(
            """
            steps:
            - label: ":amd: (MI300) Quantization"
              commands:
              - VLLM_TEST_FORCE_LOAD_FORMAT=auto pytest -v -s quantization/
              - pytest -v -s rocm/test_moe_weight_replay.py
            """
        )
    )
    monkeypatch.setattr(mod, "TEST_AREAS_DIR", tmp_path / "test_areas")
    monkeypatch.setattr(mod, "TEST_AMD_YAML", amd_yaml)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    selections = load_selections()
    assert is_tethered("rocm/test_moe_weight_replay.py", selections)
    assert is_tethered("quantization/test_foo.py", selections)


def test_missing_test_amd_yaml_is_not_fatal(tmp_path, monkeypatch):
    """When the migration finally deletes test-amd.yaml, the checker still runs."""
    mod = checker
    (tmp_path / "test_areas").mkdir()
    monkeypatch.setattr(mod, "TEST_AREAS_DIR", tmp_path / "test_areas")
    monkeypatch.setattr(mod, "TEST_AMD_YAML", tmp_path / "gone.yaml")
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    assert load_selections() == []


def test_step_level_ignore_is_honored(tmp_path, monkeypatch):
    selections = _write_area(
        tmp_path,
        monkeypatch,
        """
        steps:
        - label: "Kernels"
          commands:
          - pytest -v -s kernels --ignore=kernels/test_a.py
        """,
    )
    assert not is_tethered("kernels/test_a.py", selections)
    assert is_tethered("kernels/test_b.py", selections)


def test_amd_mirror_only_command_is_collected(tmp_path, monkeypatch):
    """A mirror that overrides `commands` runs tests the base step never does."""
    selections = _write_area(
        tmp_path,
        monkeypatch,
        """
        steps:
        - label: ":nvidia: (H200) Base"
          commands:
          - pytest -v -s kernels/test_base.py
          mirror:
            amd:
              label: ":amd: (MI300) Base"
              commands:
              - pytest -v -s kernels/test_amd_only.py
        """,
    )
    assert is_tethered("kernels/test_base.py", selections)
    assert is_tethered("kernels/test_amd_only.py", selections)


def test_unparsable_yaml_is_fatal(tmp_path, monkeypatch):
    """Silently skipping a bad yaml would drop its coverage and report false
    untethered files, so it must fail loudly instead."""
    with pytest.raises(SystemExit):
        _write_area(tmp_path, monkeypatch, "steps: [oops\n")


def test_non_mapping_yaml_is_fatal(tmp_path, monkeypatch):
    """A yaml that parses but isn't a pipeline mapping (a bare list or scalar)
    must raise the actionable error, not crash on ``.get``."""
    with pytest.raises(SystemExit):
        _write_area(tmp_path, monkeypatch, "- just\n- a\n- list\n")


# --------------------------------------------------------------------------- #
# is_test_module / path normalization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tests/kernels/test_a.py", True),
        ("tests/kernels/a_test.py", True),
        ("tests/kernels/conftest.py", False),
        ("tests/kernels/__init__.py", False),
        ("tests/kernels/utils.py", False),
        ("tests/kernels/test_a.txt", False),
        ("vllm/kernels/test_a.py", False),
    ],
)
def test_is_test_module(path, expected):
    assert is_test_module(path) is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("tests/kernels/test_a.py", "kernels/test_a.py"),
        ("./kernels/test_a.py", "kernels/test_a.py"),
        ("/vllm-workspace/tests/kernels/test_a.py", "kernels/test_a.py"),
        ("'kernels/test_a.py'", "kernels/test_a.py"),
        ("kernels/test_a.py::test_one", "kernels/test_a.py"),
        ("kernels/", "kernels"),
    ],
)
def test_normalize_test_path(raw, expected):
    assert normalize_test_path(raw) == expected


# --------------------------------------------------------------------------- #
# Whole-tree invariants
# --------------------------------------------------------------------------- #


def test_real_tree_parses_into_selections():
    assert len(load_selections()) > 200
    assert len(all_test_modules()) > 1000


def test_no_selection_matches_everything():
    """A path arg that normalizes to '' or '.' would tether every test in the
    repo and silently disarm the gate - the worst possible misparse."""
    for selection in load_selections():
        if isinstance(selection, PytestSelection):
            offenders = [
                p
                for p in selection.included_paths
                if normalize_test_path(p) in ("", ".")
            ]
            assert not offenders, f"match-everything pytest arg: {offenders}"
        else:
            assert normalize_test_path(selection.root) not in ("", ".")


def test_tethering_is_deterministic():
    selections = load_selections()
    modules = all_test_modules()[:200]
    first = [is_tethered(m, selections) for m in modules]
    assert first == [is_tethered(m, load_selections()) for m in modules]


def test_allowlist_entries_are_well_formed():
    allowlist = load_allowlist()
    assert allowlist, "allowlist should not be empty while gaps remain"
    assert all(is_test_module(path) for path in allowlist)
    assert all(not path.startswith("/") for path in allowlist)


def test_real_allowlist_entries_all_have_a_reason():
    assert allowlist_entries_missing_reason() == []


def test_bare_allowlist_entry_is_flagged(tmp_path, monkeypatch):
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text(
        "tests/a/test_ok.py  # torchrun multi-GPU, no job\n"
        "tests/b/test_bare.py\n"
        "  # a standalone comment line is fine\n"
    )
    monkeypatch.setattr(checker, "ALLOWLIST_PATH", allowlist)
    assert allowlist_entries_missing_reason() == ["tests/b/test_bare.py"]


def test_real_tree_has_no_unallowlisted_gaps():
    """Every test module is either collected by a job or allowlisted.

    Deliberately non-strict: a stale allowlist entry is advisory by design, so
    asserting `strict=True` here would make unrelated allowlist cleanup block
    CI - the exact behaviour the two-mode design avoids.
    """
    assert run_full_scan(load_selections(), load_allowlist(), strict=False) == 0


# --------------------------------------------------------------------------- #
# CLI behaviour
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """Point the module at a throwaway tree so the changed-files path can be
    exercised without writing test files into the real repo."""
    mod = checker
    (tmp_path / "tests" / "kernels").mkdir(parents=True)
    (tmp_path / "tests" / "kernels" / "test_new.py").write_text("def test_x(): pass\n")
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "TESTS_DIR", tmp_path / "tests")
    monkeypatch.setattr(mod, "ALLOWLIST_PATH", tmp_path / "allowlist.txt")
    # The throwaway tree is not a git checkout, so the tracked-file inventory
    # has to be stubbed rather than shelled out.
    monkeypatch.setattr(mod, "all_test_modules", lambda: ["tests/kernels/test_new.py"])
    return tmp_path


def test_untethered_changed_file_fails(fake_repo, capsys):
    rc = run_changed_files_check(["tests/kernels/test_new.py"], [], set())
    assert rc == 1
    assert "is not run by any Buildkite job" in capsys.readouterr().out


def test_tethered_changed_file_passes(fake_repo):
    selections = _parse_command("pytest -v -s kernels")
    assert (
        run_changed_files_check(["tests/kernels/test_new.py"], selections, set()) == 0
    )


def test_allowlisted_changed_file_passes(fake_repo):
    allowlist = {"tests/kernels/test_new.py"}
    assert run_changed_files_check(["tests/kernels/test_new.py"], [], allowlist) == 0


def test_non_test_and_missing_paths_are_ignored(fake_repo):
    paths = ["tests/kernels/conftest.py", "vllm/config.py", "tests/kernels/gone.py"]
    assert run_changed_files_check(paths, [], set()) == 0


def test_stale_allowlist_entries_are_advisory_but_reported(fake_repo, capsys):
    """Dangling entries must not block an unrelated PR, but they have to be
    printed - the hook sets `verbose: true` so a passing run still shows them."""
    allowlist = {"tests/kernels/gone.py"}
    rc = run_full_scan(_parse_command("pytest -v -s kernels"), allowlist, strict=False)
    out = capsys.readouterr().out
    assert rc == 0
    assert "no longer exists" in out


def test_stale_allowlist_entries_are_fatal_under_all(fake_repo, capsys):
    allowlist = {"tests/kernels/gone.py"}
    rc = run_full_scan(_parse_command("pytest -v -s kernels"), allowlist, strict=True)
    assert rc == 1
    assert "no longer exists" in capsys.readouterr().out


def test_now_tethered_allowlist_entry_is_reported(fake_repo, capsys):
    allowlist = {"tests/kernels/test_new.py"}
    rc = run_full_scan(_parse_command("pytest -v -s kernels"), allowlist, strict=True)
    assert rc == 1
    assert "is now tethered" in capsys.readouterr().out


def test_main_passes_a_tethered_file(monkeypatch):
    """The CLI's normal pre-commit path: a tethered file exits 0.

    `--all` is not asserted here on purpose - it implies `strict=True`, so a
    stale allowlist entry would turn unrelated cleanup into a CI failure.
    """
    monkeypatch.setattr(
        "sys.argv",
        ["check_test_tethering.py", "tests/tools/test_check_test_tethering.py"],
    )
    assert main() == 0


def test_main_flags_an_untethered_file(monkeypatch):
    # Must be named like a test module, or the check correctly skips it.
    orphan = REPO_ROOT / "tests" / "test_tethering_probe_orphan.py"
    orphan.write_text("def test_x(): pass\n")
    monkeypatch.setattr("sys.argv", ["check_test_tethering.py", str(orphan)])
    try:
        assert main() == 1
    finally:
        orphan.unlink()


def test_to_repo_relative_accepts_absolute_paths():
    absolute = str(REPO_ROOT / "tests" / "kernels" / "test_a.py")
    assert _to_repo_relative(absolute) == "tests/kernels/test_a.py"
