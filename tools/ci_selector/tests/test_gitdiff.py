# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gitdiff -z parsing (non-ASCII, renames, typechange) + ref resolution."""

import subprocess

import pytest
from ci_selector.codemap.claim import docs_only
from ci_selector.gitdiff import (
    diff_files,
    merge_base,
    resolve_diff_ref,
)


def _g(vllm_repo, *args):
    subprocess.run(["git", *args], cwd=vllm_repo, check=True, capture_output=True)


def _rev(vllm_repo, ref):
    return subprocess.run(
        ["git", "rev-parse", ref],
        cwd=vllm_repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def tmp_repo(tmp_path):
    vllm_repo = tmp_path / "vllm_repo"
    vllm_repo.mkdir()
    _g(vllm_repo, "init", "-q", "-b", "main")
    _g(vllm_repo, "config", "user.email", "t@t")
    _g(vllm_repo, "config", "user.name", "t")
    (vllm_repo / "docs").mkdir()
    (vllm_repo / "docs" / "index.md").write_text("hi")
    (vllm_repo / "b.txt").write_text("x")
    _g(vllm_repo, "add", ".")
    _g(vllm_repo, "commit", "-qm", "one")
    return vllm_repo


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
