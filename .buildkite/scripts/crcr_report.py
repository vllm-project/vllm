#!/usr/bin/env python3
"""Post one CRCR nightly callback per Buildkite job.

Separated from crcr-report.sh because the payload needs real JSON encoding: job
labels routinely contain quotes, colons and emoji, and hand-building JSON in
shell corrupts them silently.

CRCR's nightly contract (aws/lambda/cross_repo_ci_relay/callback/callback_handler.py):

    {"delivery_id": str,
     "event_type": "nightly",
     "workflow": {"status": "completed",   # nightly rejects any other status
                  "conclusion": str,
                  "run_id": int,
                  "run_attempt": int,
                  "name": str,
                  "job_name": str}}
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


# Buildkite job states -> the conclusion vocabulary HUD already stores for
# GitHub Actions callbacks, so nightly rows aggregate with the rest.
_CONCLUSION = {
    "passed": "success",
    "failed": "failure",
    "broken": "failure",
    "canceled": "cancelled",
    "canceling": "cancelled",
    "timed_out": "timed_out",
    "timing_out": "timed_out",
    "skipped": "skipped",
    "not_run": "skipped",
}


def post(url: str, token: str, body: dict, timeout: int = 30) -> tuple[int, str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")[:200]
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:200]
    except Exception as exc:  # network, DNS, TLS
        return 0, f"{type(exc).__name__}: {exc}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-json", required=True)
    p.add_argument("--callback-url", required=True)
    p.add_argument("--oidc-token", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    with open(args.build_json) as f:
        build = json.load(f)
    build_number = build.get("number")
    commit = build.get("commit", "")

    # Only script jobs are real work. Buildkite also returns waiter/trigger/
    # manual pseudo-jobs, which would otherwise show up in HUD as phantom
    # zero-duration entries.
    jobs = [
        j
        for j in build.get("jobs") or []
        if j.get("type") == "script" and j.get("state") in _CONCLUSION
    ]
    if not jobs:
        print("no terminal script jobs in this build; nothing to report")
        return 0

    ok = failed = 0
    for job in jobs:
        name = job.get("name") or job.get("command") or "<unnamed>"
        body = {
            # Stable per (build, job) so a re-run of this reporting step is
            # idempotent from HUD's point of view rather than double-counting.
            "delivery_id": f"buildkite-{build_number}-{job['id']}",
            "event_type": "nightly",
            "pytorch_head_sha": commit,
            "workflow": {
                "status": "completed",
                "conclusion": _CONCLUSION[job["state"]],
                "run_id": int(build_number),
                "run_attempt": 1,
                "name": build.get("pipeline", {}).get("slug", "ci"),
                "job_name": name,
                "url": job.get("web_url", ""),
                "started_at": job.get("started_at"),
                "completed_at": job.get("finished_at"),
            },
        }
        if args.dry_run:
            print(f"[dry-run] {_CONCLUSION[job['state']]:<10} {name[:70]}")
            ok += 1
            continue
        status, detail = post(args.callback_url, args.oidc_token, body)
        if 200 <= status < 300:
            ok += 1
        else:
            failed += 1
            # Printed once per job on purpose: a 403 means the pipeline mapping
            # is missing, and a 200 {"status":"ignored"} means the repo is not
            # allowlisted -- two failures that otherwise look identical.
            print(f"  {status} for {name[:60]}: {detail}", file=sys.stderr)

    print(f"reported {ok} job(s), {failed} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
