# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Crosscheck metric units: slug matching, status partition, remote pick."""

import subprocess
from dataclasses import dataclass

import pytest
from ci_selector.codemap.pipeline.match import TRUNC_MIN, match_jobs, slug_matches
from ci_selector.validate.crosscheck import GH_STATE_FAILED, _upstream_remote
from helpers import HW, drift_message


@dataclass
class FakeStep:
    step_id: str
    label: str
    key: str | None = None
    mirror_hw: str | None = None
    device: str | None = None


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
    assert {"FAILURE"} == GH_STATE_FAILED


def test_upstream_remote_detection(tmp_path):
    vllm_repo = tmp_path / "vllm_repo"
    vllm_repo.mkdir()
    subprocess.run(
        ["git", "init", "-q"], cwd=vllm_repo, check=True, capture_output=True
    )
    with pytest.raises(RuntimeError):
        _upstream_remote(vllm_repo)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/vllm-project/vllm.git"],
        cwd=vllm_repo,
        check=True,
        capture_output=True,
    )
    assert _upstream_remote(vllm_repo) == "origin"
    assert _upstream_remote(vllm_repo, "custom") == "custom"


def test_numeric_shard_context_matches_parent_step():
    assert slug_matches("lora-1", ["lora"], exact=False)
    assert slug_matches("models-language-5", ["models-language"], exact=False)
    # NOT a shard: non-numeric suffix stays unabsorbed
    assert not slug_matches(
        "distributed-tests-2-gpus", ["distributed-tests"], exact=False
    )


def test_match_jobs_partitions_ran_without_overlap():
    ran = {"lora": "SUCCESS", "engine": "FAILURE", "ghost": "SUCCESS"}
    matched, unmatched, by_step = match_jobs(
        ran, [FakeStep("vllm_ci:lora", "LoRA", "lora")]
    )
    assert matched == {"lora"}
    assert set(unmatched) == {"engine", "ghost"}
    assert by_step == {"vllm_ci:lora": ["lora"]}


def test_match_jobs_scores_each_side_independently():
    """The trade cell exists only because a job today covers and we do not is
    distinguishable. Sharing one unmatched pool would erase it."""
    ran = {"lora": "FAILURE", "engine": "FAILURE"}
    ours = [FakeStep("vllm_ci:lora", "LoRA", "lora")]
    today = [FakeStep("vllm_ci:engine", "Engine", "engine")]
    a_matched, a_unmatched, _ = match_jobs(ran, ours)
    t_matched, t_unmatched, _ = match_jobs(ran, today)
    # Caught only by us, lost only by us: one job each way, from the same run.
    assert [r for r in ran if r in a_matched and r in t_unmatched] == ["lora"]
    assert [r for r in ran if r in t_matched and r in a_unmatched] == ["engine"]


def test_match_jobs_exact_pass_precedes_truncation_across_steps():
    """Pinned when the loop was extracted: the exact pass must clear ALL steps
    before any step may claim a job by truncation, or a short-slugged step
    absorbs a longer job that another step owns exactly."""
    full = "very-long-step-slug-" + "x" * 40
    ran = {full[:49]: "FAILURE"}
    truncating = FakeStep("vllm_ci:a-truncating", full, full)
    exact_owner = FakeStep("vllm_ci:z-exact", full[:49], full[:49])
    # `a-` sorts first, so without the two-pass order it would win by truncation.
    _, _, by_step = match_jobs(ran, [truncating, exact_owner])
    assert by_step == {"vllm_ci:z-exact": [full[:49]]}


def test_the_replica_does_not_import_the_selection_it_is_a_baseline_for():
    """The oracle must not read our selection logic.

    It is allowed to share the parsed pipeline, the ci-infra transcriptions in
    `claim.py`, and curated facts -- its docstring says which and why. What it
    may never do is import from `classify.py` or `selection.py`, because then
    "what today's rules pick" and "what we pick" move together and the
    comparison reports agreement it did not earn. Written as a denylist rather
    than an allowlist so a new shared primitive in `claim.py` does not fail the
    build for no reason. `state.py` is allowed: a parsed pipeline is not a
    decision.
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "ci_selector" / "validate"
    tree = ast.parse((src / "generator_replica.py").read_text())
    banned = {"classify", "selection", "decide", "coverage"}
    hits = [
        f"line {node.lineno}: {node.module}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and banned & set(node.module.split("."))
    ]
    assert not hits, (
        "the replica imports the thing it is supposed to be an independent "
        f"baseline for: {hits}"
    )


def test_the_replica_is_not_used_to_launder_an_import():
    """It re-exported `is_catch_all_dep` purely so crosscheck could import it
    from here, which made the dependency graph lie about who owns what. Nothing
    should import a name from the replica that the replica did not define."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "ci_selector" / "validate"
    replica = ast.parse((src / "generator_replica.py").read_text())
    defined = {
        n.name for n in replica.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    } | {
        t.id
        for n in replica.body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    borrowed = []
    for path in sorted(src.glob("*.py")):
        if path.name == "generator_replica.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module == "generator_replica":
                borrowed += [
                    f"{path.name} imports {a.name}"
                    for a in node.names
                    if a.name not in defined
                ]
    assert not borrowed, f"laundered through the replica: {borrowed}"


@pytest.mark.drift
def test_amd_native_runtime_dependency_still_exists(vllm_repo):
    """The replica's AMD-native branch fires on this one script. If it is
    renamed the branch never fires, the baseline quietly shifts, and every
    recall number is then measured against a generator we no longer model."""
    from ci_selector.handwritten import AMD_NATIVE_RUNTIME_SOURCE_DEPENDENCIES

    missing = [
        p
        for p in AMD_NATIVE_RUNTIME_SOURCE_DEPENDENCIES
        if not (vllm_repo / p).is_file()
    ]
    assert not missing, drift_message(
        f"AMD_NATIVE_RUNTIME_SOURCE_DEPENDENCIES names files that do not "
        f"exist: {missing}",
        "ci-infra treats an edit to this script as touching every AMD-native "
        "step. The replica copies that. A path matching nothing makes the "
        "branch dead, so the oracle stops predicting jobs that really run.",
        f"the script moved in vLLM: update the path in {HW}",
        "ci-infra changed which files it watches: check amd.py upstream, then "
        f"update {HW}",
    )
