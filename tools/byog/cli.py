# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CLI entrypoint (C1): `catalog list`, `validate`, `run`, `validate-bundle`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from tools.byog.bundle import (
    BundleValidationError,
    redact_env,
    redact_serve_config,
    validate_bundle,
)
from tools.byog.catalog import Catalog, CatalogError, instantiate_plugin
from tools.byog.executor.coordinator import Coordinator
from tools.byog.executor.nvidia import NvidiaServerLauncher
from tools.byog.interfaces import PlatformAdapter
from tools.byog.schemas.execution import ExecutionStatus, ResolvedJob
from tools.byog.spec import SpecResolutionError, load_spec_file, resolve_spec

EXIT_OK = 0
EXIT_EXEC_FAILED = 1
EXIT_VALIDATION_ERROR = 2
EXIT_TIMEOUT_OR_CANCELLED = 3


def _expand_points(job: ResolvedJob) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for w in job.workloads:
        defn = w.definition
        pts = instantiate_plugin(defn.entrypoint, defn.kind).points(w.params)
        if not pts:
            raise SpecResolutionError(f"Workload '{defn.name}' produced zero points.")
        out[defn.name] = [p.point_id for p in pts]
    return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="byog", description="vLLM BYOG harness.")
    p.add_argument("--catalog-dir", type=Path, default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    cat = sub.add_parser("catalog", help="Inspect workloads and suites.")
    cat_sub = cat.add_subparsers(dest="cat_cmd", required=True)
    cl = cat_sub.add_parser("list", help="List discovered workloads and suites.")
    cl.add_argument("--json", action="store_true", dest="as_json")

    val = sub.add_parser("validate", help="Validate and resolve `spec.yaml`.")
    val.add_argument("--spec", type=Path, required=True)
    val.add_argument("--allow-unpinned-revision", action="store_true")

    run = sub.add_parser("run", help="Run `spec.yaml` and seal `bundle.tar.gz`.")
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--out", type=Path, default=Path("./byog-out"))
    run.add_argument("--job", type=str, default=None, help="Run only this job_id.")
    run.add_argument("--allow-unpinned-revision", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--image", type=str, default=None)
    run.add_argument("--image-digest", type=str, default=None)
    run.add_argument("--vllm-commit", type=str, default=None)

    vb = sub.add_parser("validate-bundle", help="Validate `bundle.tar.gz`.")
    vb.add_argument("bundle", type=Path)
    return p


def main(
    argv: list[str] | None = None,
    *,
    launcher: PlatformAdapter | None = None,
) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.cmd == "validate-bundle":
            manifest, res = validate_bundle(args.bundle)
            print(
                f"Bundle OK: job={manifest.job_id} attempt={manifest.attempt_id} "
                f"status={res.execution_status.value}"
            )
            return EXIT_OK

        cat = Catalog.discover(args.catalog_dir)

        if args.cmd == "catalog":
            ws, ss = cat.list_workloads(), cat.list_suites()
            if args.as_json:
                print(
                    json.dumps(
                        {
                            "workloads": [w.model_dump(mode="json") for w in ws],
                            "suites": [s.model_dump(mode="json") for s in ss],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                for w in ws:
                    print(f"- {w.name} ({w.kind.value}, {w.execution_mode.value})")
                for s in ss:
                    print(f"- suite:{s.name}: {s.description}")
            return EXIT_OK

        spec = load_spec_file(args.spec)
        jobs = resolve_spec(
            spec, cat, allow_unpinned_revision=args.allow_unpinned_revision
        )
        if args.cmd == "run" and args.job is not None:
            matched = [j for j in jobs if j.job_id == args.job]
            if not matched:
                known = ", ".join(j.job_id for j in jobs)
                raise SpecResolutionError(
                    f"Unknown --job '{args.job}'. Available: {known}."
                )
            jobs = matched
        points = {j.job_id: _expand_points(j) for j in jobs}

        if args.cmd == "validate":
            total_pts = sum(len(v) for pmap in points.values() for v in pmap.values())
            print(f"Spec OK: jobs={len(jobs)} points={total_pts}")
            return EXIT_OK

        if args.dry_run:
            plan: list[dict[str, Any]] = [
                {
                    "job_id": j.job_id,
                    "serve_config": redact_serve_config(j.serve_config),
                    "serve_env": redact_env(j.serve_env, allowlist_only=False),
                    "timeout_s": j.timeout_s,
                    "workloads": [
                        {
                            "name": w.definition.name,
                            "points": points[j.job_id][w.definition.name],
                        }
                        for w in j.workloads
                    ],
                }
                for j in jobs
            ]
            print(json.dumps(plan, indent=2, sort_keys=True))
            return EXIT_OK
    except (CatalogError, SpecResolutionError, BundleValidationError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_VALIDATION_ERROR

    image = args.image or os.environ.get("BYOG_IMAGE")
    digest = args.image_digest or os.environ.get("BYOG_IMAGE_DIGEST")
    commit = (
        args.vllm_commit
        or os.environ.get("BYOG_VLLM_COMMIT")
        or os.environ.get("BUILDKITE_COMMIT")
    )
    try:
        coord = Coordinator(
            launcher=launcher or NvidiaServerLauncher(),
            image_ref=image,
            vllm_commit=commit,
            image_digest=digest,
        )
        rc = EXIT_OK
        for j in jobs:
            res, archive = coord.run(j, args.out, source_spec_path=args.spec)
            print(f"Done: status={res.execution_status.value} bundle={archive}")
            for err in res.errors:
                print(f"ERROR: {err}", file=sys.stderr)
            for m in res.measurements:
                if m.error_message:
                    print(
                        f"ERROR [{m.workload}:{m.point_id}]: {m.error_message}",
                        file=sys.stderr,
                    )
            if res.execution_status in (
                ExecutionStatus.TIMED_OUT,
                ExecutionStatus.CANCELLED,
            ):
                rc = max(rc, EXIT_TIMEOUT_OR_CANCELLED)
            elif res.execution_status != ExecutionStatus.COMPLETED:
                rc = max(rc, EXIT_EXEC_FAILED)
        return rc
    except Exception as exc:  # noqa: BLE001
        print(f"Execution error: {exc}", file=sys.stderr)
        return EXIT_EXEC_FAILED
