# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict reusable plugin contract assertion helper for contributor CPU tests."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any

from tests.byog.fake_launcher import FakeLauncher
from tools.byog.bundle.validator import validate_bundle
from tools.byog.catalog import Catalog
from tools.byog.executor.coordinator import Coordinator
from tools.byog.executor.interfaces import AssetProvider, FixtureAssetProvider
from tools.byog.schemas.catalog import ExecutionMode, WorkloadKind
from tools.byog.schemas.result import ExecutionStatus, RunResult
from tools.byog.schemas.spec import RunSpec
from tools.byog.spec.resolver import resolve_single_job

_CONTRACT_TEST_SHA = "0123456789abcdef0123456789abcdef01234567"


def assert_plugin_contract(
    *,
    workload_name: str,
    tmp_path: Path,
    catalog: Catalog | None = None,
    assets: AssetProvider | None = None,
    launcher: FakeLauncher | None = None,
    workload_overrides: dict[str, Any] | None = None,
    min_accuracy: float | None = None,
) -> tuple[RunResult, Path]:
    """Run a cataloged plugin end-to-end against `FakeLauncher` and assert C0-C5."""
    cat = catalog or Catalog.discover()
    defn = cat.get_workload(workload_name)
    overrides = dict(workload_overrides or {})

    spec_dict: dict[str, Any] = {
        "schema_version": 1,
        "name": f"contract-{workload_name}",
        "serve": {
            "config": {
                "model": "fake/contract-model",
                "revision": _CONTRACT_TEST_SHA,
            }
        },
        "hardware": {"gpu": ["h200"], "count": 1},
    }
    if defn.kind == WorkloadKind.BENCHMARK:
        spec_dict["benchmarks"] = [{"type": workload_name, **overrides}]
    else:
        eval_entry: dict[str, Any] = {"name": workload_name, **overrides}
        if min_accuracy is not None:
            eval_entry["min_accuracy"] = min_accuracy
        spec_dict["evals"] = [eval_entry]

    spec = RunSpec.model_validate(spec_dict)
    job = resolve_single_job(spec, cat, attempt_id=f"att-contract-{workload_name}")
    active_launcher = launcher or FakeLauncher()
    active_assets = assets or FixtureAssetProvider(tmp_path)

    coord = Coordinator(launcher=active_launcher, assets=active_assets)
    result, archive_path = coord.run(job, tmp_path / "out")

    assert result.execution_status == ExecutionStatus.COMPLETED, (
        f"Contract run for '{workload_name}' failed: {result.errors}"
    )
    assert result.measurements, f"'{workload_name}' produced zero measurements."
    assert active_launcher.stops == active_launcher.starts

    for m in result.measurements:
        assert m.status == ExecutionStatus.COMPLETED, (
            f"Measurement {m.point_id} failed: {m.error_message}"
        )
        assert m.metrics, f"Measurement {m.point_id} emitted empty metrics."
        assert m.attempted is not None and m.attempted >= 1, (
            f"Measurement {m.point_id} must report attempted >= 1."
        )
        assert m.completed is not None and m.completed >= 1, (
            f"Measurement {m.point_id} must report completed >= 1."
        )

    if defn.execution_mode == ExecutionMode.SERVER:
        assert active_launcher.total_requests >= 1, (
            f"Server plugin '{workload_name}' did not send any HTTP requests."
        )
    else:
        assert active_launcher.direct_calls >= 1, (
            f"Direct plugin '{workload_name}' did not execute via ctx.run_command."
        )

    manifest, validated = validate_bundle(archive_path)
    assert manifest.attempt_id == result.attempt_id
    assert validated.execution_status == ExecutionStatus.COMPLETED

    if defn.kind == WorkloadKind.EVAL:
        total_attempted = sum(m.attempted or 0 for m in result.measurements)
        for m in result.measurements:
            assert m.scoring_denominator is not None and m.scoring_denominator >= 1, (
                f"Eval measurement {m.point_id} must report scoring_denominator >= 1."
            )
        with tarfile.open(archive_path, "r:gz") as tar:
            names = {m.name for m in tar.getmembers() if m.isfile()}
            for fname in ("samples.jsonl", "inputs.jsonl", "responses.jsonl"):
                rel = f"evals/{workload_name}/{fname}"
                assert rel in names, f"Missing eval record file '{rel}' in bundle."
                blob = tar.extractfile(rel).read().decode("utf-8")  # type: ignore[union-attr]
                lines = [ln for ln in blob.splitlines() if ln.strip()]
                assert lines, f"Eval record file '{rel}' must be non-empty."
                for ln in lines:
                    json.loads(ln)
                if fname == "responses.jsonl":
                    assert len(lines) == total_attempted, (
                        f"responses.jsonl count ({len(lines)}) != attempted "
                        f"({total_attempted})."
                    )

    return result, archive_path
