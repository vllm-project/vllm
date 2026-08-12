# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Crosscheck metric units: slug matching, status partition, remote pick."""

import subprocess

import pytest
from ci_analyzer.validate.crosscheck import (
    FAILED_STATES,
    TRUNC_MIN,
    _upstream_remote,
    slug_matches,
)


def test_exact_never_absorbs_longer_job():
    """The failed_missed deflation bug: a selected step whose slug prefixes
    a longer, different job must not claim that job's status."""
    cands = ["distributed-tests"]
    assert slug_matches("distributed-tests", cands, exact=True)
    assert not slug_matches("distributed-tests-2-gpus", cands, exact=True)
    assert not slug_matches("distributed-tests-2-gpus", cands, exact=False)


def test_truncated_context_still_matches():
    full = "very-long-step-slug-" + "x" * 40
    ran = full[:49]
    assert len(ran) >= TRUNC_MIN
    assert slug_matches(ran, [full], exact=False)


def test_short_prefix_is_not_truncation():
    full = "engine-tests-2-gpus"
    assert not slug_matches("engine", [full], exact=False)


def test_pending_and_error_are_not_failures():
    assert {"FAILURE"} == FAILED_STATES


def test_upstream_remote_detection(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)
    with pytest.raises(RuntimeError):
        _upstream_remote(repo)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/vllm-project/vllm.git"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    assert _upstream_remote(repo) == "origin"
    assert _upstream_remote(repo, "custom") == "custom"


def test_numeric_shard_context_matches_parent_step():
    assert slug_matches("lora-1", ["lora"], exact=False)
    assert slug_matches("models-language-5", ["models-language"], exact=False)
    # NOT a shard: non-numeric suffix stays unabsorbed
    assert not slug_matches(
        "distributed-tests-2-gpus", ["distributed-tests"], exact=False
    )
