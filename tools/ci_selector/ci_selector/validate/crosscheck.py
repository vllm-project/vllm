# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay a real PR: what ran, what today's rules would pick, what we pick.

Ground truth is public. Every Buildkite job that ran posts a GitHub commit
status, and a job that ran AND FAILED is near-proof the diff affects it, so
"failed job not selected" is the number that matters. Merged and
closed-unmerged PRs both work.

Skipped and blocked steps post nothing, so what CI skipped is inferred rather
than seen. The comparison is against what CI actually spent.

Usage:
  ci-validate crosscheck --repo ../.. --prs 50326 49364 [--json-out out.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from ..codemap.claim import (
    is_catch_all_dep,
    matches_source_dependency,
    split_deps,
)
from ..codemap.classify import select
from ..codemap.pipeline.match import (
    match_jobs,
    slug_matches_any,
    step_slug_candidates,
)
from ..codemap.worktree import git_out, state_for
from ..coverage.source import fetch_table
from ..decide import decide
from ..gitdiff import changed_paths, diff_files
from ..handwritten import PR_PIPELINE
from .generator_replica import today_select

GH_STATE_FAILED = frozenset({"FAILURE"})
GH_STATE_SUCCESS = "SUCCESS"
GH_STATE_MERGED = "MERGED"
GH_REPO = "vllm-project/vllm"
GH_DEFAULT_BRANCH = "main"
GH_PR_REFSPEC = "pull/{pr}/head"
BUILDKITE_CONTEXT_PREFIX = "buildkite/ci/pr/"
NON_STEP_CONTEXTS = frozenset({"bootstrap", "fastcheck"})
PIPELINE_GEN_V2_MARKER = ".buildkite/.pipeline_gen_v2"


def base_in_window(repo: Path, base: str) -> bool:
    """A base from before the pipeline restructure carries no marker file, and
    the replica has no meaning against one."""
    probe = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "cat-file",
            "-e",
            f"{base}:{PIPELINE_GEN_V2_MARKER}",
        ],
        capture_output=True,
    )
    return probe.returncode == 0


def _gh_pr(pr: int) -> dict:
    out = subprocess.run(
        [
            "gh",
            "pr",
            "view",
            str(pr),
            "--repo",
            GH_REPO,
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
        if len(parts) >= 2 and GH_REPO in parts[1]:
            return parts[0]
    raise RuntimeError(f"no git remote points at {GH_REPO}; pass --remote")


def _resolve_range(repo: Path, pr: int, data: dict, remote: str):
    if data["state"] == GH_STATE_MERGED and data.get("mergeCommit"):
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
            [
                "git",
                "-C",
                str(repo),
                "fetch",
                "-q",
                remote,
                GH_PR_REFSPEC.format(pr=pr),
            ],
            check=True,
            capture_output=True,
        )
    return git_out(repo, "merge-base", head, f"{remote}/{GH_DEFAULT_BRANCH}"), head


def _jobs(steps) -> int:
    """Jobs a step set expands to.

    `parallelism` is a literal in the job yaml, read when the pipeline is
    uploaded, so this is exact rather than estimated. Only WHICH tests land in a
    shard is decided at run time. A mirror inherits its parent's count and
    cannot override it, so an amd variant expands the same way.
    """
    return sum(step.parallelism or 1 for step in steps)


def _totals(scored: list[dict]) -> dict:
    """Sums for the TOTAL block. Indexes directly on purpose: a missing key is
    a harness bug and must raise, not read as zero."""
    total = {
        k: sum(r[k] for r in scored)
        for k in (
            "them_steps",
            "codemap_steps",
            "final_steps",
            "them_jobs",
            "codemap_jobs",
            "final_jobs",
        )
    }
    total["missed"] = sum(len(r["missed_failures"]) for r in scored)
    total["catchall_missed"] = sum(len(r["catchall_only_missed"]) for r in scored)
    return total


def crosscheck_pr(repo: Path, pr: int, remote: str | None = None, table=None) -> dict:
    data = _gh_pr(pr)
    ran = {}
    for c in data["statusCheckRollup"]:
        ctx = c.get("context") or c.get("name") or ""
        if ctx.startswith(BUILDKITE_CONTEXT_PREFIX):
            s = ctx[len(BUILDKITE_CONTEXT_PREFIX) :]
            if s not in NON_STEP_CONTEXTS:
                ran[s] = c.get("state") or c.get("conclusion")
    base, head = _resolve_range(repo, pr, data, _upstream_remote(repo, remote))
    if base is None:
        return {"pr": pr, "skip": f"state={data['state']}, no head"}
    if not base_in_window(repo, base):
        return {"pr": pr, "skip": "pre-restructure base"}
    paths = changed_paths(diff_files(repo, base, head))
    from ..codemap.claim import docs_only

    # The only channel that can see the upstream docs-only predicate drift:
    # jobs that RAN on a diff our copy of it calls docs-only.
    docs_only_but_ran = sorted(ran) if docs_only(paths) and ran else []
    state = state_for(repo, base)
    sel = select(state, paths, base=base, head=head)
    today = today_select([(p.config, p.steps) for p in state.pipelines], paths)
    vllm_steps = {
        s.step_id: s
        for p in state.pipelines
        if p.config.name == PR_PIPELINE
        for s in p.steps
    }
    a_ids = {s for s in sel.selected if s.startswith(f"{PR_PIPELINE}:")}
    t_ids = today.selected.get(PR_PIPELINE, set())

    # What the tool actually answers. `a_ids` is the code map alone, kept only
    # so the two halves stay distinguishable.
    decision = decide(state, sel, repo, base, head, table=table)
    f_ids = {s for s in decision.steps if s.startswith(f"{PR_PIPELINE}:")}

    # Steps today's rules reach ONLY through a blanket dependency, which we drop
    # when the graph already knows the file. Anything swept in by run-all has no
    # firing dep and falls out here by itself.
    catchall_only_missed = []
    for sid in sorted(t_ids - a_ids):
        step = vllm_steps.get(sid)
        if not step:
            continue
        positive, negated = split_deps(step.source_file_dependencies)
        firing = [
            dep
            for dep in positive
            if any(
                matches_source_dependency(dep, p)
                and not any(matches_source_dependency(n, p) for n in negated)
                for p in paths
            )
        ]
        if firing and all(is_catch_all_dep(dep) for dep in firing):
            catchall_only_missed.append(sid)

    selected_steps = [vllm_steps[s] for s in a_ids if s in vllm_steps]
    # Which jobs each selected step accounts for. Narrowing this further means
    # asking whether a step it drops actually failed.
    a_matched, unmatched, matched_by_step = match_jobs(ran, selected_steps)
    # The same question of today's rules, scored apart: a job they cover and we
    # do not is the case that matters, and a shared pool would hide it.
    today_steps = [vllm_steps[s] for s in t_ids if s in vllm_steps]
    t_matched, t_unmatched, _ = match_jobs(ran, today_steps)
    final_steps = [vllm_steps[s] for s in f_ids if s in vllm_steps]
    f_matched, f_unmatched, _ = match_jobs(ran, final_steps)

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
    failed_ran = {r: s for r, s in ran.items() if s in GH_STATE_FAILED}
    # The trade: of the jobs that really ran and really failed, which side
    # covers them. `trade_caught` is what we buy over today's rules, `trade_lost`
    # is what we would give up, and anything there stops work until judged.
    # `a_run_all` rides along, because a catch earned by running everything is
    # not a catch.
    trade_caught = sorted(r for r in failed_ran if r in a_matched and r in t_unmatched)
    trade_lost = sorted(r for r in failed_ran if r in t_matched and r in unmatched)
    # Jobs that ran although today's rules would not have picked them. This is
    # the only window onto a failure today would skip, since otherwise the job
    # never runs and leaves no trace. Listed in full, so a replica gap reads as
    # one name recurring rather than a pile of separate findings.
    today_window = sorted(t_unmatched)
    inconclusive = {
        r: s
        for r, s in ran.items()
        if s not in GH_STATE_FAILED and s not in (GH_STATE_SUCCESS, None)
    }
    # What CI ran, as STEPS: fold job slugs back onto the steps behind them, or
    # the two sides are counted in different units.
    _, unmappable, ran_by_step = match_jobs(ran, list(vllm_steps.values()))
    them_steps = [vllm_steps[sid] for sid in ran_by_step]

    # Both sides expanded the same way from the same model, so a difference is
    # real rather than an artifact of counting. NOT `len(ran)` for their side: a
    # step whose shards were skipped posts fewer statuses than it defines, which
    # would read as a win we did not earn.
    them_jobs = _jobs(them_steps)
    codemap_jobs = _jobs([vllm_steps[s] for s in a_ids if s in vllm_steps])
    final_jobs = _jobs(final_steps)
    return {
        "pr": pr,
        "title": data["title"][:70],
        "state": data["state"],
        # The refs and step ids behind the counts, so a reader can dig without
        # resolving the PR range again.
        "base": base,
        "head": head,
        "selected_ids": sorted(a_ids),
        "selected_rules": {s: sel.selected_rules.get(s, []) for s in sorted(a_ids)},
        # Parallel to selected_rules: per reason, the changed files the record
        # may weigh, or null where the reason cannot be argued with. A missing
        # key means treat every step as NOT droppable, never as droppable
        # against the whole diff, which would re-authorise drops on always-run
        # steps.
        "selected_paths": {s: sel.selected_paths.get(s, []) for s in sorted(a_ids)},
        # changed file -> the steps it selected. A step absent from every value
        # is selected regardless of the diff, not unnecessary.
        "selected_by_file": {
            f: sorted(s for s in steps if s in a_ids)
            for f, steps in sorted(sel.selected_by_file.items())
        },
        "matched_slugs": {s: sorted(v) for s, v in sorted(matched_by_step.items())},
        "catchall_only_missed": catchall_only_missed,
        "files": len(set(paths)),
        "ran": len(ran),
        "analyzer": len(a_ids),
        "today_replica": len(t_ids),
        "today_ids": sorted(t_ids),
        "a_run_all": PR_PIPELINE in sel.run_all,
        "t_run_all": bool(today.run_all.get(PR_PIPELINE)),
        "t_docs_only": today.docs_only,
        "run_all_reason": sel.run_all.get(PR_PIPELINE, "")[:100],
        "covered": len(a_matched),
        "today_covered": len(t_matched),
        # ---- the bottom line: what the tool would actually do --------------
        # `analyzer` above is the code map alone; these are both halves.
        "final": len(f_ids),
        "final_ids": sorted(f_ids),
        "final_covered": len(f_matched),
        "coverage_added": len(decision.added_by_coverage),
        "coverage_dropped": len(decision.dropped_by_coverage),
        "coverage_stale": decision.stale_steps,
        # The reason histogram the rule already built. Diagnostic, but the
        # only per-PR view of a mode-specific verdict, and recomputing it
        # costs a whole run.
        "reasons": decision.reasons,
        # Empty when the record was used. Anything here means code map only, so
        # `final` equals `analyzer` and this is not the real comparison.
        "coverage_note": decision.coverage_note,
        # From the jobs that ran, so this is what CI spent rather than what the
        # pipeline defines.
        "them_steps": len(them_steps),
        "codemap_steps": len(a_ids),
        "final_steps": len(f_ids),
        # Jobs, both sides, expanded by the same rule.
        "them_jobs": them_jobs,
        "codemap_jobs": codemap_jobs,
        "final_jobs": final_jobs,
        # Positive means we eliminated jobs; negative means we ran more.
        "win": them_jobs - final_jobs,
        # Job slugs no step of ours explains. They are missing from
        # `them_steps`, so the win is measured against a partial picture.
        "unmappable_jobs": sorted(unmappable),
        # What preflight refused to reason about, grouped by why. Forced steps
        # are non-droppable, so this is a cost floor no coverage evidence can
        # lift. Per PR, because it moves with the replayed revision.
        "forced_steps": state.preflight.forced_by_reason,
        # Ran, failed, and we did not pick it. Judge each: relevant to the diff
        # is a real slip, flaky or infra is a correct omission.
        "missed_failures": {r: st for r, st in failed_ran.items() if r in f_unmatched},
        "today_window": today_window,
        "today_failed_missed": {
            r: s for r, s in failed_ran.items() if r in t_unmatched
        },
        "trade_caught": trade_caught,
        "trade_lost": trade_lost,
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
    parser.add_argument(
        "--table", type=Path, help="coverage table (default: the configured one)"
    )


def run(args) -> int:
    repo = args.repo.resolve()
    # Once. Every PR reads the same table, and reloading per PR would dominate
    # a long replay.
    table = fetch_table(args.table)
    if not table.available:
        print(f"NOTE: {table.unavailable}\n  final == codemap for every PR below.")
    print("triples read: CI ran / codemap only / coverage + codemap\n")
    results = []
    # TODO: parallelise. Each PR builds a worktree and a graph at its own base,
    # and `worktree.py` shares caches and a reaper across them without locking,
    # so a pool has to be processes rather than threads.
    for pr in args.prs:
        try:
            r = crosscheck_pr(repo, pr, remote=args.remote, table=table)
        except subprocess.CalledProcessError as e:
            r = {"pr": pr, "skip": f"command failed: {e.stderr[:100]}"}
        results.append(r)
        if "skip" in r:
            print(f"PR #{pr}: SKIP ({r['skip']})", flush=True)
            continue
        steps = f"{r['them_steps']}/{r['codemap_steps']}/{r['final_steps']}"
        jobs = f"{r['them_jobs']}/{r['codemap_jobs']}/{r['final_jobs']}"
        print(
            f"PR #{pr}  steps {steps}  jobs {jobs}  win {r['win']:+d}  "
            f"missed {len(r['missed_failures'])}"
            # A win earned by bailing out to run-everything is not a win.
            f"{'  RUN_ALL' if r['a_run_all'] else ''}"
            f"{f'  catchall {n}' if (n := len(r['catchall_only_missed'])) else ''}"
            f"   {r['title'][:36]}",
            flush=True,
        )
        if r["unmappable_jobs"]:
            print(
                f"  ! {len(r['unmappable_jobs'])} ran jobs match no step we model, "
                "so CI's side is understated",
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
    # Floor. `ran` filters status contexts on a fixed prefix, so a rename
    # upstream empties it for every PR while the run still prints a clean line
    # each. NOT gated on missed failures: judging those is a human call.
    scored = [r for r in results if "skip" not in r]
    if scored:
        total = _totals(scored)
        ratio = (
            f"   ({total['final_jobs'] / total['them_jobs']:.3f}x of CI)"
            if total["them_jobs"]
            else ""
        )
        missed = total["missed"]
        print(
            f"\nTOTAL over {len(scored)} PRs"
            f"\n  Steps    CI ran {total['them_steps']:<8} "
            f"codemap {total['codemap_steps']:<8} ours {total['final_steps']}"
            f"\n  Jobs     CI ran {total['them_jobs']:<8} "
            f"codemap {total['codemap_jobs']:<8} ours {total['final_jobs']}{ratio}"
            f"\n  Win      {total['them_jobs'] - total['final_jobs']:+d} jobs"
            f"\n  Missed   {missed} failed job{'' if missed == 1 else 's'}, "
            "each still to be judged"
            f"\n  Catchall {total['catchall_missed']} missed steps reached only "
            "through a blanket dep (B4, not the arm under test)"
        )
    if not any(r.get("ran") for r in results):
        print(
            "  COLLAPSE: no PR reported a single buildkite status context; "
            "the comparison ran against nothing"
        )
        return 1
    return 0
