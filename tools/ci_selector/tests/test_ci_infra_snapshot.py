# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every ci-infra check, run offline against what `--sync` last downloaded.

`sync` fetches and writes; it decides nothing. All the deciding is here, so the
checks run on an ordinary `pytest tests -q` with no network.

Three kinds of check, and none of them needs a human to sign anything off:

  values      our constants against the value the generator assigns
  behaviour   our functions against upstream's, both executed on the same
              generated inputs
  floors      that the two above are still looking at something

Behaviour rather than text is what makes this self-clearing. Our replica is
written in our own names and types, so comparing source could only ever say
"something moved" and would need a stored baseline plus a command to update it,
and that command could do nothing but rubber-stamp whatever was on disk.
Running both and comparing outputs answers the real question, so upstream
moving goes red and fixing our code goes green, with nothing in between.

Two anchored functions are not executable: `select_steps_and_dependencies`,
which we depend on rather than reproduce, and `read_steps_from_job_dir`, which
is here only for the working-dir default it assigns. The snapshot is committed,
so a `--sync` that changes either shows up as a git diff to read. That is a
weaker signal than a test and is the honest residual.
"""

import ci_infra
import pytest
from helpers import HW, drift_message

SIGN_OFF = "uv run python tests/ci_infra.py"
SYNC = "uv run pytest tests --sync -q"

# Ours -> the constant ci-infra assigns the same value to. Names differ where
# upstream chose a different one for the same fact.
SAME_VALUE = {
    "AMD_ALWAYS_RUN_STEP_KEYS": "AMD_ALWAYS_RUN_STEP_KEYS",
    "AMD_NATIVE_RUNTIME_SOURCE_DEPENDENCIES": "AMD_NATIVE_RUNTIME_SOURCE_DEPENDENCIES",
    "ONLY_STEP_KEYS_ENV": "ONLY_STEP_KEYS_ENV_VAR",
    "MIRROR_DEFAULT_DEPENDS_ON": "AMD_ARTIFACT_STEP",
    "CONTAINER_WORKSPACE": "AMD_NATIVE_WORKSPACE",
}

COST = (
    "Section 4 of handwritten.py is a hand copy of what ci-infra's generator "
    "does, and twelve of those constants sit on the main selection path. A "
    "value that no longer matches means we are modelling a generator that "
    "does not exist, which shows up as jobs we fail to select."
)


def ours(const):
    """Our constant as a sorted list of strings, so a lone string and a
    one-element tuple compare equal to each other."""
    from ci_selector import handwritten

    value = getattr(handwritten, const)
    return [value] if isinstance(value, str) else sorted(value)


def theirs(value):
    return [value] if isinstance(value, str) else sorted(value)


@pytest.mark.drift
@pytest.mark.parametrize("const", sorted(SAME_VALUE))
def test_our_constant_equals_the_value_ci_infra_assigns(const):
    """Value equality against the generator's own constant, not a substring
    search: a value that survives as a comment, or takes on a different role,
    no longer passes."""
    upstream = SAME_VALUE[const]
    try:
        recorded = ci_infra.constant(upstream)
    except LookupError:
        pytest.fail(
            drift_message(
                f"ci-infra no longer defines {upstream}, which is where "
                f"{const} came from.",
                COST,
                f"it was renamed: point {const} at the new name in "
                "tests/test_ci_infra_snapshot.py, then check the value",
                f"the behaviour is gone: remove {const} and whatever reads it",
            )
        )
    assert ours(const) == theirs(recorded), drift_message(
        f"{const} is {ours(const)}, but ci-infra's {upstream} is {theirs(recorded)}.",
        COST,
        f"update {const} in {HW} to match",
    )


@pytest.mark.drift
def test_the_image_build_prefix_matches_the_generators_own_test():
    """Upstream grants always-run by key prefix, written inline as a
    `startswith`. Read the literal out of that call rather than trusting the
    string to appear somewhere in the package."""
    try:
        upstream = ci_infra.method_arg("_step_should_run", "startswith")
    except LookupError as exc:
        pytest.fail(
            drift_message(
                f"Cannot find the always-run prefix test in ci-infra: {exc}",
                COST,
                "upstream restructured _step_should_run: re-read it and update "
                "the query in tests/ci_infra.py",
            )
        )
    assert ours("IMAGE_BUILD_KEY_PREFIX") == [upstream], drift_message(
        f"IMAGE_BUILD_KEY_PREFIX is {ours('IMAGE_BUILD_KEY_PREFIX')}, but "
        f"ci-infra grants always-run on {upstream!r}.",
        "These steps build the images every other step waits on. Getting the "
        "prefix wrong either runs them never or runs them always.",
        f"update IMAGE_BUILD_KEY_PREFIX in {HW} to {upstream!r}",
    )


@pytest.mark.drift
def test_the_default_working_dir_matches_the_generators_own():
    """Upstream sets it inline while reading a job dir, so there is no
    constant to compare; take the assignment."""
    try:
        upstream = ci_infra.attr_assignment("read_steps_from_job_dir", "working_dir")
    except LookupError as exc:
        pytest.fail(
            drift_message(
                f"Cannot find the working-dir default in ci-infra: {exc}",
                COST,
                "upstream restructured read_steps_from_job_dir: re-read it and "
                "update the query in tests/ci_infra.py",
            )
        )
    assert ours("DEFAULT_WORKING_DIR") == [upstream], drift_message(
        f"DEFAULT_WORKING_DIR is {ours('DEFAULT_WORKING_DIR')}, but ci-infra "
        f"defaults steps to {upstream!r}.",
        "Step working dirs are absolute paths under it, mapped back to "
        "repo-relative. A wrong root sends every relative test target astray.",
        f"update DEFAULT_WORKING_DIR in {HW} to {upstream!r}",
    )


@pytest.mark.drift
def test_the_docs_only_rule_uses_exactly_our_three_values():
    """All three of ours are the string literals of one upstream function, so
    the whole rule can be compared at once rather than value by value."""
    upstream = ci_infra.literals_in("is_docs_only_change")
    mine = set(
        ours("DOCS_ONLY_PREFIXES")
        + ours("DOCS_ONLY_SUFFIXES")
        + ours("DOCS_ONLY_EXACT")
    )
    assert mine == upstream, drift_message(
        f"The docs-only rule diverged. We use {sorted(mine)}; ci-infra's "
        f"is_docs_only_change uses {sorted(upstream)}.",
        "A whole-diff docs-only answer emits nothing at all, so this predicate "
        "is the one place we can take CI to zero steps. Reading it wrong in "
        "either direction is the most expensive mistake in the tool.",
        f"update DOCS_ONLY_PREFIXES, _SUFFIXES or _EXACT in {HW}",
    )


@pytest.mark.drift
def test_the_snapshot_still_holds_something_to_check():
    """Guard the guard. Every test above passes against an empty snapshot by
    having nothing to disagree with."""
    values = ci_infra.read_values()
    found = sum(len(v) for v in values.values())
    assert found >= 50, drift_message(
        f"The snapshot holds only {found} upstream constants, so the value "
        "checks above are running on almost nothing.",
        "An extraction that stopped working reads exactly like a generator "
        "that stopped changing.",
        "the source moved: check constants_in() in tests/ci_infra.py against "
        f"a fresh `{SYNC}`",
    )
    for name in sorted(ci_infra.ANCHORS):
        assert ci_infra.anchor_source(name), drift_message(
            f"No recorded source for ci-infra's {name}.",
            "Nothing can be compared against it, so its checks pass empty.",
            f"re-download with `{SYNC}`",
        )


# Inputs for the behaviour checks. Deliberately awkward: empty, bare slash,
# trailing slash, prefix-of-another, double slash, and one of each docs shape.
PATHS = (
    "",
    "/",
    "a",
    "a/",
    "a/b",
    "a/b.py",
    "ab",
    "a//b",
    "docs/x.md",
    "x.md",
    "mkdocs.yaml",
    "docs",
    "vllm",
    "vllm/",
    "vllm/x.py",
    "tests/t.py",
)
DEP_SETS = (
    None,
    [],
    ["a"],
    ["a", "b"],
    ["!a"],
    ["a", "!a/b"],
    ["vllm"],
    ["vllm", "!vllm/x.py"],
    ["/"],
    [""],
)


@pytest.fixture(scope="module")
def upstream():
    """ci-infra's own functions, executed out of the snapshot."""
    return ci_infra.upstream_callables()


@pytest.mark.drift
def test_the_upstream_functions_actually_loaded(upstream):
    """Guard the guard. Every behaviour check below compares against these, and
    an empty namespace would agree with anything."""
    missing = [n for n in ci_infra.SELF_CONTAINED if not callable(upstream.get(n))]
    assert not missing, drift_message(
        f"These ci-infra functions could not be executed from the snapshot: {missing}",
        "The behaviour checks compare our functions against these. Without "
        "them they compare against nothing and pass.",
        "upstream gave them a dependency they did not have: they may no longer "
        "be self-contained, so drop them from SELF_CONTAINED in "
        "tests/ci_infra.py and find another way to watch them",
        f"the snapshot is stale or partial: re-download with `{SYNC}`",
    )


@pytest.mark.drift
def test_our_dep_match_behaves_like_ci_infras(upstream):
    """One dep against one path. This decides what every declared-dep step
    selects, so a difference here is a difference in CI."""
    from ci_selector.codemap.claim import matches_source_dependency

    theirs = upstream["_matches_source_dependency"]
    for dep in PATHS:
        for path in PATHS:
            assert theirs(dep, path) == matches_source_dependency(dep, path), (
                drift_message(
                    f"matches_source_dependency disagrees with ci-infra on "
                    f"dep={dep!r}, path={path!r}: they say "
                    f"{theirs(dep, path)}, we say "
                    f"{matches_source_dependency(dep, path)}.",
                    "This is the whole of source_file_dependencies matching. "
                    "Disagreeing means we select a different set of steps than "
                    "CI runs, in whichever direction the difference goes.",
                    "read tests/ci_infra_snapshot/"
                    f"_matches_source_dependency{ci_infra.SUFFIX} and make "
                    "ci_selector/codemap/claim.py agree with it",
                )
            )


@pytest.mark.drift
def test_our_declaration_matching_behaves_like_ci_infras(upstream):
    """A whole declaration against a whole diff, so the include/exclude split
    and the any-file-matches rule are covered, not just one pair."""
    import itertools

    from ci_selector.codemap.claim import step_declares

    theirs = upstream["_source_file_dependencies_match"]
    for deps in DEP_SETS:
        for size in range(3):
            for diff in itertools.combinations(PATHS, size):
                mine = any(step_declares(deps, p) for p in diff)
                assert theirs(deps, list(diff)) == mine, drift_message(
                    f"Declaration matching disagrees with ci-infra on "
                    f"deps={deps}, diff={list(diff)}: they say "
                    f"{theirs(deps, list(diff))}, we say {mine}.",
                    "A `!` entry carves a subtree out of a broader positive "
                    "one, and the decision is per step rather than per entry. "
                    "Getting it wrong changes which steps a diff selects.",
                    "read tests/ci_infra_snapshot/"
                    f"_source_file_dependencies_match{ci_infra.SUFFIX} and "
                    "make step_declares in claim.py agree with it",
                )


@pytest.mark.drift
def test_our_docs_only_behaves_like_ci_infras_apart_from_empty_paths(upstream):
    """One deliberate difference, pinned so it cannot grow.

    ci-infra skips an empty path and can therefore call a diff of nothing but
    empty strings docs-only, which emits no steps at all. We call it not
    docs-only and run the ordinary rules. Ours is the conservative side and
    `git diff --name-only` never emits an empty path, so this is left as is;
    the assertions below fail if the difference ever spreads beyond that case
    or flips direction.
    """
    import itertools

    from ci_selector.codemap.claim import docs_only

    theirs = upstream["is_docs_only_change"]
    diverged = []
    for size in range(4):
        for diff in itertools.combinations(PATHS, size):
            paths = list(diff)
            if theirs(paths) != docs_only(paths):
                diverged.append(paths)
    assert all("" in d for d in diverged), drift_message(
        "docs_only now disagrees with ci-infra on a diff with no empty path: "
        f"{[d for d in diverged if '' not in d][:5]}",
        "A whole-diff docs-only answer emits nothing at all, so this predicate "
        "is the one place we can take CI to zero steps. It is the most "
        "expensive thing in the tool to get wrong.",
        "read tests/ci_infra_snapshot/"
        f"is_docs_only_change{ci_infra.SUFFIX} and make docs_only agree",
    )
    assert all(theirs(d) and not docs_only(d) for d in diverged), drift_message(
        "The empty-path difference with ci-infra has flipped direction: we now "
        "call something docs-only that they do not.",
        "That direction emits no steps where CI would run some, which is "
        "under-selection on the cheapest possible diff.",
        "make docs_only in claim.py at least as conservative as ci-infra's "
        "is_docs_only_change",
    )


@pytest.mark.drift
def test_our_replica_behaves_like_the_generators_own_rule():
    """The `_step_should_run` replica, against upstream's real code.

    Upstream needs a few things we have to supply: its global config, `os`, a
    step type, and its AMD-device test, which we substitute with our own
    `family_of_device`. So this checks the branch structure and its order, not
    those substituted parts, which have their own guards. The three branches
    we deliberately do not model (NOAUTO, only_step_keys, nightly) are pinned
    off here rather than left to chance.
    """
    import itertools
    import os
    from types import SimpleNamespace

    from ci_selector.codemap.hardware import family_of_device
    from ci_selector.handwritten import (
        AMD_ALWAYS_RUN_STEP_KEYS,
        AMD_NATIVE_RUNTIME_SOURCE_DEPENDENCIES,
        IMAGE_BUILD_KEY_PREFIX,
    )
    from ci_selector.validate.generator_replica import step_should_run

    config = {"only_step_keys": None, "nightly": "0", "run_all": False}
    namespace = ci_infra.upstream_callables()
    namespace.update(
        os=os,
        get_global_config=lambda: config,
        AMD_ALWAYS_RUN_STEP_KEYS=AMD_ALWAYS_RUN_STEP_KEYS,
        AMD_NATIVE_RUNTIME_SOURCE_DEPENDENCIES=list(
            AMD_NATIVE_RUNTIME_SOURCE_DEPENDENCIES
        ),
        is_amd_gpu_device=lambda d: family_of_device(d) == "amd",
        Step=SimpleNamespace,
    )
    exec(ci_infra.anchor_source("_step_should_run"), namespace)  # noqa: S102
    theirs = namespace["_step_should_run"]

    amd_script = AMD_NATIVE_RUNTIME_SOURCE_DEPENDENCIES[0]
    keys = [None, "lora", "image-build", "image-build-amd", *AMD_ALWAYS_RUN_STEP_KEYS]
    devices = [None, "h100", "mi300_1", "cpu-small"]
    diffs = [[], ["vllm/x.py"], ["tests/t.py"], [amd_script], ["docs/a.md"]]
    checked = 0
    for key, device, dind, optional, deps, diff, run_all in itertools.product(
        keys, devices, (False, True), (False, True), DEP_SETS, diffs, (False, True)
    ):
        config["run_all"] = run_all
        always = bool(key) and (
            key.startswith(IMAGE_BUILD_KEY_PREFIX) or key in AMD_ALWAYS_RUN_STEP_KEYS
        )
        common = dict(
            key=key,
            optional=optional,
            device=device,
            dind=dind,
            source_file_dependencies=deps,
        )
        mine = step_should_run(
            SimpleNamespace(always_runs=always, mirror_hw=None, **common), diff, run_all
        )
        checked += 1
        assert theirs(SimpleNamespace(**common), diff) == mine, drift_message(
            "Our _step_should_run replica disagrees with ci-infra's on "
            f"key={key!r}, device={device!r}, dind={dind}, optional={optional}, "
            f"deps={deps}, diff={diff}, run_all={run_all}.",
            "The replica is the baseline every recall and cost figure is "
            "measured against. If it stops describing the real generator, the "
            "numbers describe a CI that does not exist.",
            "read tests/ci_infra_snapshot/"
            f"_step_should_run{ci_infra.SUFFIX} and make step_should_run in "
            "ci_selector/validate/generator_replica.py agree with it",
        )
    assert checked > 1000, "the input space collapsed; this proves nothing"
