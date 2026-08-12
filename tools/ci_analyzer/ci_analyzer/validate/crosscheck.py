# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Three-way PR crosscheck: actually-ran vs generator replica vs analyzer.

Ground truth is public: every Buildkite job that ran on a PR reports a GitHub
commit status (`buildkite/ci/pr/<slug>`). Jobs that ran and FAILED are
near-proof the diff affects them, so "failed job not selected" is the
red-flag metric. Works for merged PRs (merge commit) and closed-unmerged PRs
(fetches pull/N/head, base = merge-base with upstream/main). Skipped/blocked
steps report no status, so "what was skipped" is inferred, not observed.

Usage:
  ci-validate crosscheck --repo ../.. --prs 50326 49364 [--json-out out.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import regex as re

from ..gitdiff import changed_paths, diff_files
from ..policy import matches_source_dependency, split_deps
from ..select import select
from ..worktree import git_out, state_for
from .todaymatcher import (
    is_catch_all_dep,
    today_select,
)

NON_STEP_CONTEXTS = {"bootstrap", "fastcheck"}
# Buildkite status contexts truncate around 49 chars; only a ran slug at
# least this long may prefix-match a longer candidate.
TRUNC_MIN = 45
# Statuses that count as real failures for the red-flag metric. A test that
# times out or asserts wrong reports FAILURE, capturing diff-caused
# regressions. PENDING (canceled at merge), ERROR (agent/infra), and EXPECTED
# are not diff-caused, so they land in `inconclusive` (visible for judgment,
# never in failed_missed; counting them would flag infra noise as
# under-selection). Pinned by test_pending_and_error_are_not_failures.
FAILED_STATES = {"FAILURE"}


def base_in_window(repo: Path, base: str) -> bool:
    """A base predating the Feb 2026 v2 restructure has no
    `.buildkite/.pipeline_gen_v2` marker; the replica has no semantics there."""
    probe = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "cat-file",
            "-e",
            f"{base}:.buildkite/.pipeline_gen_v2",
        ],
        capture_output=True,
    )
    return probe.returncode == 0


def _slug(label: str, plus_word: bool) -> str:
    s = label.lower().replace("+", " plus " if plus_word else " ")
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return re.sub(r"-n$", "", s)


def step_slug_candidates(step) -> list[str]:
    """Buildkite status-context spellings for a step: key, label slug with
    '+' as 'plus' or dropped, amd- prefix + device suffix for mirrors."""
    out = []
    if step.mirror_hw:
        base_label = step.label.rsplit(" (", 1)[0]
        for pw in (True, False):
            c = f"amd-{_slug(base_label, pw)}"
            out.append(c)
            if step.device:
                out.append(f"{c}-{step.device.replace('_', '-')}")
    else:
        if step.key:
            out.append(step.key)
        for pw in (True, False):
            out.append(_slug(step.label, pw))
    return out


def slug_matches(ran_slug: str, cands: list[str], *, exact: bool) -> bool:
    """exact: literal equality. Non-exact: the ran context is a TRUNCATED
    prefix of a candidate, or a NUMERIC SHARD of it ('lora-1' for the
    parallelism-expanded 'lora' step). General ran.startswith(c) is NOT a
    match: it silently absorbed longer jobs' statuses (a selected 'engine'
    claiming 'engine-2-gpus', deflating failed_missed); the shard rule
    requires a purely numeric suffix, which 'engine-2-gpus' fails."""
    if exact:
        return ran_slug in cands
    return any(
        (c.startswith(ran_slug) and len(ran_slug) >= TRUNC_MIN)
        or re.fullmatch(re.escape(c) + r"-\d+", ran_slug)
        for c in cands
    )


def slug_matches_any(ran_slug: str, cands: list[str]) -> bool:
    return slug_matches(ran_slug, cands, exact=True) or slug_matches(
        ran_slug, cands, exact=False
    )


def _gh_pr(pr: int) -> dict:
    out = subprocess.run(
        [
            "gh",
            "pr",
            "view",
            str(pr),
            "--repo",
            "vllm-project/vllm",
            "--json",
            "state,mergeCommit,headRefOid,labels,statusCheckRollup,title",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)


def _upstream_remote(repo: Path, override: str | None = None) -> str:
    if override:
        return override
    for line in git_out(repo, "remote", "-v").splitlines():
        parts = line.split()
        if len(parts) >= 2 and "vllm-project/vllm" in parts[1]:
            return parts[0]
    raise RuntimeError("no git remote points at vllm-project/vllm; pass --remote")


def _resolve_range(repo: Path, pr: int, data: dict, remote: str):
    if data["state"] == "MERGED" and data.get("mergeCommit"):
        merge = data["mergeCommit"]["oid"]
        return git_out(repo, "rev-parse", f"{merge}^"), merge
    head = data.get("headRefOid")
    if not head:
        return None, None
    probe = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", head], capture_output=True
    )
    if probe.returncode != 0:
        subprocess.run(
            ["git", "-C", str(repo), "fetch", "-q", remote, f"pull/{pr}/head"],
            check=True,
            capture_output=True,
        )
    return git_out(repo, "merge-base", head, f"{remote}/main"), head


def crosscheck_pr(repo: Path, pr: int, remote: str | None = None) -> dict:
    data = _gh_pr(pr)
    ran = {}
    for c in data["statusCheckRollup"]:
        ctx = c.get("context") or c.get("name") or ""
        if ctx.startswith("buildkite/ci/pr/"):
            s = ctx[len("buildkite/ci/pr/") :]
            if s not in NON_STEP_CONTEXTS:
                ran[s] = c.get("state") or c.get("conclusion")
    base, head = _resolve_range(repo, pr, data, _upstream_remote(repo, remote))
    if base is None:
        return {"pr": pr, "skip": f"state={data['state']}, no head"}
    if not base_in_window(repo, base):
        return {"pr": pr, "skip": "pre-restructure base"}
    paths = changed_paths(diff_files(repo, base, head))
    from ..policy import docs_only

    # The only channel that can observe upstream is_docs_only_change drift:
    # jobs that RAN on a diff our frozen predicate calls docs-only.
    docs_only_but_ran = sorted(ran) if docs_only(paths) and ran else []
    state = state_for(repo, base)
    sel = select(state, paths, base=base, head=head)
    today = today_select([(p.config, p.steps) for p in state.pipelines], paths)
    vllm_steps = {
        s.step_id: s
        for p in state.pipelines
        if p.config.name == "vllm_ci"
        for s in p.steps
    }
    a_ids = {s for s in sel.selected if s.startswith("vllm_ci:")}
    t_ids = today.selected.get("vllm_ci", set())

    unmatched = dict(ran)
    a_matched: set[str] = set()
    selected_steps = [vllm_steps[s] for s in a_ids if s in vllm_steps]
    # Exact pass over ALL selected steps first, truncation pass second, so a
    # short-slugged step cannot absorb a longer job's status.
    for exact in (True, False):
        for step in selected_steps:
            cands = step_slug_candidates(step)
            hit = {r for r in unmatched if slug_matches(r, cands, exact=exact)}
            a_matched |= hit
            for h in hit:
                unmatched.pop(h)

    miss_specific, miss_catchall, miss_unknown = [], [], []
    for r in unmatched:
        firing: set[str] = set()
        for sid in t_ids:
            step = vllm_steps.get(sid)
            if not step or not slug_matches_any(r, step_slug_candidates(step)):
                continue
            positive, negated = split_deps(step.source_file_dependencies)
            for dep in positive:
                if any(
                    matches_source_dependency(dep, p)
                    and not any(matches_source_dependency(n, p) for n in negated)
                    for p in paths
                ):
                    firing.add(dep)
        specific = [d for d in firing if not is_catch_all_dep(d)]
        if specific:
            miss_specific.append((r, sorted(specific)[:2]))
        elif firing:
            miss_catchall.append((r, sorted(firing)[:2]))
        else:
            miss_unknown.append((r, "run_all/always/no dep traced"))

    extra_selected = sorted(
        step.label
        for step in selected_steps
        if not any(slug_matches_any(r, step_slug_candidates(step)) for r in ran)
    )
    failed_ran = {r: s for r, s in ran.items() if s in FAILED_STATES}
    inconclusive = {
        r: s
        for r, s in ran.items()
        if s not in FAILED_STATES and s not in ("SUCCESS", None)
    }
    return {
        "pr": pr,
        "title": data["title"][:70],
        "state": data["state"],
        "files": len(set(paths)),
        "ran": len(ran),
        "analyzer": len(a_ids),
        "today_replica": len(t_ids),
        "a_run_all": "vllm_ci" in sel.run_all,
        "run_all_reason": sel.run_all.get("vllm_ci", "")[:100],
        "covered": len(a_matched),
        "extra_selected": extra_selected,
        "miss_specific": miss_specific,
        "miss_catchall": miss_catchall,
        "miss_catchall_n": len(miss_catchall),
        "miss_unknown_n": len(miss_unknown),
        "failed_ran": failed_ran,
        "inconclusive": inconclusive,
        "failed_missed": {r: s for r, s in failed_ran.items() if r in unmatched},
        "docs_only_but_ran": docs_only_but_ran,
    }


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--prs", type=int, nargs="+", required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--remote", help="remote for vllm-project/vllm (auto-detected)")


def run(args) -> int:
    repo = args.repo.resolve()
    results = []
    for pr in args.prs:
        try:
            r = crosscheck_pr(repo, pr, remote=args.remote)
        except subprocess.CalledProcessError as e:
            r = {"pr": pr, "skip": f"command failed: {e.stderr[:100]}"}
        results.append(r)
        if "skip" in r:
            print(f"PR #{pr}: SKIP ({r['skip']})", flush=True)
            continue
        print(
            f"PR #{pr}: files={r['files']} ran={r['ran']} "
            f"analyzer={r['analyzer']}{' RUN_ALL' if r['a_run_all'] else ''} "
            f"covered={r['covered']} missSpec={len(r['miss_specific'])} "
            f"failedMissed={len(r['failed_missed'])} "
            f"inconc={len(r['inconclusive'])}  {r['title'][:45]}",
            flush=True,
        )
        if r["docs_only_but_ran"]:
            print(
                f"  RED FLAG: docs-only by our predicate but "
                f"{len(r['docs_only_but_ran'])} jobs ran "
                "(is_docs_only_change drifted upstream?)",
                flush=True,
            )
    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=1))
    # Detection floor. `ran` is built by filtering status contexts on a literal
    # prefix, so a context rename upstream (or a `gh` field change) empties it
    # for every PR and the run still prints one clean line each. Comparing our
    # selection against nothing is the vacuous pass this check exists to avoid.
    # Deliberately NOT gated on failed_missed: judging those is a human call.
    if not any(r.get("ran") for r in results):
        print(
            "  COLLAPSE: no PR reported a single buildkite status context; "
            "the comparison ran against nothing"
        )
        return 1
    return 0
