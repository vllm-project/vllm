# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C1: CLI entrypoints (`catalog`, `validate`, `run`, `validate-bundle`, `upload`)."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from tools.byog.bundle.validator import BundleValidationError, validate_bundle
from tools.byog.bundle.writer import redact_env, redact_serve_config
from tools.byog.catalog import Catalog, CatalogError, instantiate_plugin
from tools.byog.executor.coordinator import Coordinator
from tools.byog.executor.interfaces import (
    AssetProvider,
    FixtureAssetProvider,
    PlatformAdapter,
)
from tools.byog.executor.platform.nvidia import NvidiaServerLauncher
from tools.byog.schemas.catalog import WorkloadKind
from tools.byog.schemas.result import ExecutionStatus, QualityState
from tools.byog.schemas.spec import ResolvedJob
from tools.byog.spec.resolver import (
    SpecResolutionError,
    load_spec_file,
    resolve_single_job,
)

EXIT_OK = 0
EXIT_EXEC_FAILED = 1
EXIT_VALIDATION_ERROR = 2
EXIT_TIMEOUT_OR_CANCELLED = 3


def _default_asset_provider() -> AssetProvider:
    try:
        mod = importlib.import_module("tools.byog.catalog.assets")
        provider_cls = mod.CachingAssetProvider
        return provider_cls()
    except (ImportError, AttributeError):
        return FixtureAssetProvider(Path.cwd())


def _validate_plugins_and_points(job: ResolvedJob) -> dict[str, list[str]]:
    """Instantiate every plugin and expand points so errors fail at validate time."""
    points_by_workload: dict[str, list[str]] = {}
    for w in job.workloads:
        plugin = instantiate_plugin(w.entrypoint, expected_kind=WorkloadKind(w.kind))
        pts = plugin.points(w.params)
        if not pts:
            raise SpecResolutionError(
                f"Workload '{w.name}' produced zero execution points for params "
                f"{w.params}."
            )
        points_by_workload[w.name] = [p.point_id for p in pts]
    return points_by_workload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="byog",
        description="vLLM Bring Your Own GPU (BYOG) benchmark and eval harness.",
    )
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=None,
        help="Override catalog directory (defaults to tools/byog/catalog).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # byog catalog list
    cat_p = sub.add_parser("catalog", help="Inspect auto-discovered workloads/suites.")
    cat_sub = cat_p.add_subparsers(dest="catalog_cmd", required=True)
    cat_list = cat_sub.add_parser("list", help="List workloads and suites.")
    cat_list.add_argument("--json", action="store_true", dest="as_json")

    # byog validate
    val_p = sub.add_parser("validate", help="Validate and resolve a spec YAML.")
    val_p.add_argument("--spec", type=Path, required=True)
    val_p.add_argument("--allow-unpinned-revision", action="store_true")

    # byog run
    run_p = sub.add_parser("run", help="Execute a spec and produce bundle.tar.gz.")
    run_p.add_argument("--spec", type=Path, required=True)
    run_p.add_argument("--out", type=Path, default=Path("./byog-out"))
    run_p.add_argument("--allow-unpinned-revision", action="store_true")
    run_p.add_argument("--dry-run", action="store_true")
    run_p.add_argument("--upload", action="store_true")
    run_p.add_argument(
        "--image",
        type=str,
        default=None,
        help="Container image ref recorded in repro.sh (default: $BYOG_IMAGE).",
    )
    run_p.add_argument(
        "--image-digest",
        type=str,
        default=None,
        help="Container image digest recorded in result.json.",
    )
    run_p.add_argument(
        "--vllm-commit",
        type=str,
        default=None,
        help="vLLM git commit recorded in result.json.",
    )
    run_p.add_argument(
        "--source-tier",
        choices=["ci_nightly", "ci_pr", "community"],
        default="community",
    )

    # byog validate-bundle
    vb_p = sub.add_parser(
        "validate-bundle", help="Verify bundle.tar.gz schema, safety, and hashes."
    )
    vb_p.add_argument("bundle", type=Path)

    # byog upload (Phase 1 Results Service stub)
    up_p = sub.add_parser("upload", help="Upload a validated bundle (Phase 1).")
    up_p.add_argument("bundle", type=Path)

    return parser


def _cmd_catalog_list(catalog: Catalog, *, as_json: bool) -> int:
    workloads = catalog.list_workloads()
    suites = catalog.list_suites()
    if as_json:
        payload = {
            "workloads": [w.model_dump(mode="json") for w in workloads],
            "suites": [s.model_dump(mode="json") for s in suites],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return EXIT_OK

    print("Workloads:")
    for w in workloads:
        print(
            f"  - {w.name} ({w.kind.value}, mode={w.execution_mode.value}, "
            f"lifecycle={w.lifecycle.value})"
        )
    print("Suites:")
    for s in suites:
        print(f"  - {s.name}: {s.description}")
    return EXIT_OK


def main(
    argv: list[str] | None = None,
    *,
    launcher: PlatformAdapter | None = None,
    assets: AssetProvider | None = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "validate-bundle":
            manifest, result = validate_bundle(args.bundle)
            print(
                f"Bundle OK: job_id={manifest.job_id} "
                f"attempt_id={manifest.attempt_id} "
                f"status={result.execution_status.value} "
                f"quality={result.quality_state.value} "
                f"files={len(manifest.files)}"
            )
            return EXIT_OK

        if args.command == "upload":
            print(
                "Error: `byog upload` requires the Phase 1 Results Service "
                "(not yet available in Phase 0).",
                file=sys.stderr,
            )
            return EXIT_VALIDATION_ERROR

        # Reject `run --upload` upfront BEFORE doing any GPU or catalog work.
        if args.command == "run" and args.upload:
            print(
                "Error: `--upload` is not supported in Phase 0 (planned in Phase 1 "
                "Results Service). Re-run without `--upload` to produce a local "
                "bundle.tar.gz.",
                file=sys.stderr,
            )
            return EXIT_VALIDATION_ERROR

        catalog = Catalog.discover(args.catalog_dir)

        if args.command == "catalog":
            return _cmd_catalog_list(catalog, as_json=args.as_json)

        if args.command == "validate":
            spec = load_spec_file(args.spec)
            job = resolve_single_job(
                spec,
                catalog,
                allow_unpinned_revision=args.allow_unpinned_revision,
            )
            points_map = _validate_plugins_and_points(job)
            print(
                f"Spec OK: job_id={job.job_id} workloads={len(job.workloads)} "
                f"points={sum(len(v) for v in points_map.values())}"
            )
            return EXIT_OK

        if args.command == "run":
            spec = load_spec_file(args.spec)
            job = resolve_single_job(
                spec,
                catalog,
                allow_unpinned_revision=args.allow_unpinned_revision,
                source_tier=args.source_tier,
            )
            points_map = _validate_plugins_and_points(job)

            image_ref = args.image or os.environ.get("BYOG_IMAGE")
            image_digest = args.image_digest or os.environ.get("BYOG_IMAGE_DIGEST")
            vllm_commit = (
                args.vllm_commit
                or os.environ.get("BYOG_VLLM_COMMIT")
                or os.environ.get("BUILDKITE_COMMIT")
            )

            if args.dry_run:
                dry_payload: dict[str, Any] = {
                    "schema_version": job.schema_version,
                    "job_id": job.job_id,
                    "attempt_id": "<assigned-at-run>",
                    "spec_name": job.spec_name,
                    "serve_config": redact_serve_config(job.serve_config),
                    "serve_env": redact_env(job.serve_env, allowlist_only=False),
                    "hardware": job.hardware.model_dump(),
                    "timeout_s": job.timeout_s,
                    "image": image_ref,
                    "vllm_commit": vllm_commit,
                    "resolved_config_hash": job.resolved_config_hash,
                    "workloads": [
                        {
                            "name": w.name,
                            "kind": w.kind,
                            "execution_mode": w.execution_mode,
                            "lifecycle": w.lifecycle,
                            "points": points_map[w.name],
                        }
                        for w in job.workloads
                    ],
                }
                print(json.dumps(dry_payload, indent=2, sort_keys=True))
                return EXIT_OK

            coord = Coordinator(
                launcher=launcher or NvidiaServerLauncher(),
                assets=assets or _default_asset_provider(),
                image_ref=image_ref,
                vllm_commit=vllm_commit,
                image_digest=image_digest,
            )
            result, archive_path = coord.run(
                job,
                args.out,
                source_spec_path=args.spec,
            )
            print(
                f"Run finished: status={result.execution_status.value} "
                f"quality={result.quality_state.value} "
                f"bundle={archive_path}"
            )
            for err in result.errors:
                print(f"ERROR: {err}", file=sys.stderr)
            for m in result.measurements:
                if m.error_message:
                    print(
                        f"ERROR [{m.workload}:{m.point_id}]: {m.error_message}",
                        file=sys.stderr,
                    )

            if result.execution_status in (
                ExecutionStatus.TIMED_OUT,
                ExecutionStatus.CANCELLED,
            ):
                return EXIT_TIMEOUT_OR_CANCELLED
            if (
                result.execution_status != ExecutionStatus.COMPLETED
                or result.quality_state == QualityState.FAILED
            ):
                return EXIT_EXEC_FAILED
            return EXIT_OK

    except (
        CatalogError,
        SpecResolutionError,
        BundleValidationError,
        OSError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_VALIDATION_ERROR

    return EXIT_VALIDATION_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
