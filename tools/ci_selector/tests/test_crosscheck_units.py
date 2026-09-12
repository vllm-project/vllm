# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Crosscheck metric units: slug matching, status partition, remote pick."""

import subprocess
from dataclasses import dataclass

import pytest
from ci_selector.codemap.pipeline.buildkite import load_pipeline_configs, load_steps
from ci_selector.codemap.pipeline.match import (
    TRUNC_MIN,
    match_jobs,
    slug_matches,
    slug_matches_any,
    step_slug_candidates,
)
from ci_selector.codemap.pipeline.step import LoadReport
from ci_selector.validate.crosscheck import GH_STATE_FAILED, _totals, _upstream_remote
from helpers import HW, drift_message

MATCH = "ci_selector/codemap/pipeline/match.py"


@dataclass
class FakeStep:
    step_id: str
    label: str
    key: str | None = None
    mirror_hw: str | None = None
    device: str | None = None
    mirror_label: str | None = None


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


# Real labels from .buildkite/test_areas against real status contexts. A mirror
# has two spellings: its own yaml label, used verbatim, and the wrapped form
# "AMD: <parent> (<device>)" that older revisions carry.
MIRROR_CASES = [
    (
        "declared label",
        ":amd: (MI300) Basic Correctness",
        ":nvidia: (H200) Basic Correctness",
        "amd-mi300-basic-correctness",
    ),
    (
        "wrapped parent label",
        None,
        ":nvidia: (H200) Basic Correctness",
        "amd-nvidia-h200-basic-correctness-mi300-1",
    ),
    (
        # `_slug` strips a trailing `-n`, so assembling the slug from parts ate
        # the `%N` and these never matched.
        "sharded %N, wrapped label",
        None,
        ":nvidia: (H100) V1 Attention Shard %N",
        "amd-nvidia-h100-v1-attention-shard-n-mi300-1",
    ),
    (
        "sharded %N, declared",
        ":amd: (MI300) Attention Kernels Shard %N",
        ":nvidia: (H100) Attention Kernels Shard %N",
        "amd-mi300-attention-kernels-shard-3",
    ),
    (
        # Buildkite cut this at exactly 44 characters.
        "44-char truncation",
        ":amd: (MI300) Multimodal Models (Standard) 2: qwen3 + gemma",
        ":nvidia: (H200) Multimodal Models (Standard) 2",
        "amd-mi300-multimodal-models-standard-2-qwen3",
    ),
]


@pytest.mark.parametrize(
    "case,declared,parent,context", MIRROR_CASES, ids=[c[0] for c in MIRROR_CASES]
)
def test_a_mirror_matches_the_context_ci_actually_posted(
    case, declared, parent, context
):
    step = FakeStep(
        step_id="vllm_ci:x-amd:amd",
        label=f"{parent} (amd)",
        key="x-amd",
        mirror_hw="amd",
        device="mi300_1",
        mirror_label=declared,
    )
    assert slug_matches_any(context, step_slug_candidates(step)), (
        f"{case}: no candidate matches {context!r}\n"
        f"candidates: {step_slug_candidates(step)}"
    )


def test_the_44_char_case_is_really_a_truncation():
    """Pins TRUNC_MIN's value. If these contexts stop being 44 long the
    constant should go back up."""
    assert len("amd-mi300-multimodal-models-standard-2-qwen3") == 44
    assert TRUNC_MIN == 44


def _ambiguous_groups(steps, threshold):
    """Sets of steps a truncated context of this length cannot tell apart."""
    owners: dict[str, set[str]] = {}
    for step in steps:
        for cand in step_slug_candidates(step):
            if len(cand) > threshold:
                owners.setdefault(cand[:threshold], set()).add(step.step_id)
    return {frozenset(o) for o in owners.values() if len(o) > 1}


@pytest.mark.drift
def test_lowering_the_threshold_bought_no_new_ambiguity(vllm_repo):
    """What makes TRUNC_MIN safe. Not that nothing is ambiguous.

    Some steps already share a truncated prefix at any threshold in this range,
    which is structural and predates the constant. The narrower claim worth
    pinning: the current value leaves no further pair indistinguishable than the
    one above it does. Comparing two thresholds against each other needs no
    stored baseline, so it cannot be rubber-stamped.

    That comparison is asserted FIRST on purpose. It is the safety claim; the
    ceiling below is a stored number. With the ceiling first, a stale count
    short-circuits the real check and a regression sits unevaluated.
    """
    configs = load_pipeline_configs(vllm_repo)
    report = LoadReport()
    steps = [s for c in configs for s in load_steps(vllm_repo, c, report)]
    here = _ambiguous_groups(steps, TRUNC_MIN)
    stricter = _ambiguous_groups(steps, TRUNC_MIN + 1)
    assert here == stricter, drift_message(
        "Lowering TRUNC_MIN made these steps indistinguishable: "
        f"{[sorted(g) for g in here - stricter]}",
        "A job reported under a truncated name would be attributed to "
        "whichever of them was visited first, and the others would score as "
        "never run, understating what CI ran.",
        f"raise TRUNC_MIN in {MATCH} back until the two agree",
        "if the new context is real and the steps are genuinely distinct, give "
        "one of them a longer label in .buildkite/test_areas",
    )
    # A ceiling, not a pin: comparing two thresholds says nothing about how many
    # groups there are, so a twelfth could appear in silence. The groups come
    # from the device prefix eating most of the window, not from our matcher.
    assert len(here) <= 11, drift_message(
        f"Truncated step contexts are ambiguous in {len(here)} groups, above "
        f"the recorded 11: {[sorted(g) for g in here]}",
        "A job reported under a truncated name is attributed to whichever "
        "member was visited first; the others score as never run, which "
        "understates CI's own cost and can hide a real miss.",
        "two steps were given labels sharing a long prefix: give one a more "
        "distinctive label in .buildkite/test_areas",
        "the ambiguity is real and accepted: raise the ceiling here",
    )


def _scored(n, catchall=None):
    counts = {
        k: n
        for k in (
            "them_steps",
            "codemap_steps",
            "final_steps",
            "them_jobs",
            "codemap_jobs",
            "final_jobs",
        )
    }
    counts["missed_failures"] = []
    if catchall is not None:
        counts["catchall_only_missed"] = catchall
    return counts


def test_total_block_aggregates_catchall_only_missed():
    """The counter was emitted per PR and summed nowhere, so contamination
    could not be told apart from real losses. The total must be the sum, and a
    result missing the key must raise rather than read as zero."""
    scored = [
        _scored(1, catchall=["vllm_ci:a", "vllm_ci:b"]),
        _scored(2, catchall=["vllm_ci:c"]),
    ]
    total = _totals(scored)
    assert total["catchall_missed"] == 3
    assert total["them_jobs"] == 3
    with pytest.raises(KeyError):
        _totals([_scored(1)])
