# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gitdiff -z parsing (non-ASCII, renames, typechange) + ref resolution."""

import subprocess

import pytest
from ci_analyzer.gitdiff import (
    diff_files,
    merge_base,
    resolve_diff_ref,
)
from ci_analyzer.policy import docs_only


def _g(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _rev(repo, ref):
    return subprocess.run(
        ["git", "rev-parse", ref],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def tmp_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _g(repo, "init", "-q", "-b", "main")
    _g(repo, "config", "user.email", "t@t")
    _g(repo, "config", "user.name", "t")
    (repo / "docs").mkdir()
    (repo / "docs" / "index.md").write_text("hi")
    (repo / "b.txt").write_text("x")
    _g(repo, "add", ".")
    _g(repo, "commit", "-qm", "one")
    return repo


def test_non_ascii_path_unquoted_and_docs_only(tmp_repo):
    (tmp_repo / "docs" / "café.md").write_text("hola")
    _g(tmp_repo, "add", ".")
    _g(tmp_repo, "commit", "-qm", "two")
    files = diff_files(tmp_repo, "HEAD^", "HEAD")
    assert [f.path for f in files] == ["docs/café.md"]
    assert docs_only([f.path for f in files])


def test_rename_carries_both_sides(tmp_repo):
    _g(tmp_repo, "mv", "b.txt", "c.txt")
    _g(tmp_repo, "commit", "-qm", "mv")
    files = diff_files(tmp_repo, "HEAD^", "HEAD")
    assert files[0].status == "R"
    assert files[0].path == "c.txt" and files[0].old_path == "b.txt"


def test_typechange_flows_as_modification(tmp_repo):
    (tmp_repo / "b.txt").unlink()
    (tmp_repo / "b.txt").symlink_to("docs/index.md")
    _g(tmp_repo, "add", ".")
    _g(tmp_repo, "commit", "-qm", "link")
    files = diff_files(tmp_repo, "HEAD^", "HEAD")
    assert files[0].status == "T" and files[0].path == "b.txt"


def test_resolve_diff_ref_forms(tmp_repo):
    first = _rev(tmp_repo, "HEAD")
    (tmp_repo / "b.txt").write_text("y")
    _g(tmp_repo, "add", ".")
    _g(tmp_repo, "commit", "-qm", "two")
    _g(tmp_repo, "checkout", "-qb", "feat", "HEAD^")
    (tmp_repo / "f.txt").write_text("f")
    _g(tmp_repo, "add", ".")
    _g(tmp_repo, "commit", "-qm", "feat")

    assert merge_base(tmp_repo, "main", "feat") == first
    # triple-dot: CI semantics, base = merge-base
    assert resolve_diff_ref(tmp_repo, "main...feat") == (first, "feat")
    # two-dot: snapshot diff, refs verbatim
    assert resolve_diff_ref(tmp_repo, "main..feat") == ("main", "feat")
    assert resolve_diff_ref(tmp_repo, "main") == ("main", None)
