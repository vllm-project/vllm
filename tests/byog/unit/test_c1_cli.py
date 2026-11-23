# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for C1 CLI and strict plugin contract helper."""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from tests.byog.contract_helpers import assert_plugin_contract
from tests.byog.fake_launcher import FakeLauncher
from tools.byog.catalog import Catalog
from tools.byog.cli import (
    EXIT_EXEC_FAILED,
    EXIT_OK,
    EXIT_VALIDATION_ERROR,
    main,
)
from tools.byog.executor.interfaces import (
    BenchmarkRunner,
    DirectRunContext,
    FixtureAssetProvider,
    PointResult,
    ServerRunContext,
    WorkloadPoint,
)

VALID_SHA = "0123456789abcdef0123456789abcdef01234567"


class WorkingHttpBench(BenchmarkRunner):
    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        return [WorkloadPoint("c1")]

    def run_point(
        self, point: WorkloadPoint, ctx: ServerRunContext | DirectRunContext
    ) -> PointResult:
        assert isinstance(ctx, ServerRunContext)
        body = json.dumps({"model": ctx.model, "prompt": "hi"}).encode("utf-8")
        req = urllib.request.Request(
            f"{ctx.base_url}/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
        return PointResult(metrics={"rps": 10.0}, attempted=1, completed=1)


class NoOpBrokenBench(BenchmarkRunner):
    """Intentionally broken plugin that never sends any HTTP requests."""

    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        return [WorkloadPoint("c1")]

    def run_point(
        self, point: WorkloadPoint, ctx: ServerRunContext | DirectRunContext
    ) -> PointResult:
        return PointResult(metrics={"rps": 0.0}, attempted=1, completed=1)


class FailingBench(BenchmarkRunner):
    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        return [WorkloadPoint("c1")]

    def run_point(
        self, point: WorkloadPoint, ctx: ServerRunContext | DirectRunContext
    ) -> PointResult:
        raise RuntimeError("boom-from-plugin")


def _setup_cli_catalog(tmp_path: Path) -> Path:
    cdir = tmp_path / "catalog"
    wdir = cdir / "workloads"
    sdir = cdir / "suites"
    wdir.mkdir(parents=True)
    sdir.mkdir(parents=True)
    (wdir / "ok_bench.yaml").write_text(
        "schema_version: 1\nname: ok_bench\nkind: benchmark\n"
        "execution_mode: server\nlifecycle: none\n"
        f"entrypoint: {__name__}:WorkingHttpBench\n",
        encoding="utf-8",
    )
    (wdir / "noop_bench.yaml").write_text(
        "schema_version: 1\nname: noop_bench\nkind: benchmark\n"
        "execution_mode: server\nlifecycle: none\n"
        f"entrypoint: {__name__}:NoOpBrokenBench\n",
        encoding="utf-8",
    )
    (wdir / "fail_bench.yaml").write_text(
        "schema_version: 1\nname: fail_bench\nkind: benchmark\n"
        "execution_mode: server\nlifecycle: none\n"
        f"entrypoint: {__name__}:FailingBench\n",
        encoding="utf-8",
    )
    (wdir / "bad_entry.yaml").write_text(
        "schema_version: 1\nname: bad_entry\nkind: benchmark\n"
        "execution_mode: server\nlifecycle: none\n"
        "entrypoint: nonexistent.module:MissingClass\n",
        encoding="utf-8",
    )
    return cdir


def test_cli_catalog_validate_dry_run_and_upload_guard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cdir = _setup_cli_catalog(tmp_path)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "schema_version: 1\n"
        "name: cli-test\n"
        "serve:\n"
        "  config:\n"
        "    model: org/m\n"
        f"    revision: {VALID_SHA}\n"
        "    api-key: secret-cli-key\n"
        "hardware:\n"
        "  gpu: [h200]\n"
        "benchmarks:\n"
        "  - type: ok_bench\n",
        encoding="utf-8",
    )

    # 1. catalog list --json
    rc = main(["--catalog-dir", str(cdir), "catalog", "list", "--json"])
    assert rc == EXIT_OK
    listed = json.loads(capsys.readouterr().out)
    assert any(w["name"] == "ok_bench" for w in listed["workloads"])

    # 2. validate succeeds and fails cleanly on bad entrypoint or malformed YAML.
    assert (
        main(["--catalog-dir", str(cdir), "validate", "--spec", str(spec_path)])
        == EXIT_OK
    )
    capsys.readouterr()

    bad_ep_spec = tmp_path / "bad_ep.yaml"
    bad_ep_spec.write_text(
        "schema_version: 1\nserve:\n  config:\n    model: m\n"
        f"    revision: {VALID_SHA}\nbenchmarks:\n  - type: bad_entry\n",
        encoding="utf-8",
    )
    assert (
        main(["--catalog-dir", str(cdir), "validate", "--spec", str(bad_ep_spec)])
        == EXIT_VALIDATION_ERROR
    )
    assert "nonexistent.module" in capsys.readouterr().err

    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("schema_version: [1", encoding="utf-8")
    assert (
        main(["--catalog-dir", str(cdir), "validate", "--spec", str(malformed)])
        == EXIT_VALIDATION_ERROR
    )
    capsys.readouterr()

    # 3. run --dry-run is deterministic (no random attempt_id) and redacts secrets.
    rc = main(
        [
            "--catalog-dir",
            str(cdir),
            "run",
            "--spec",
            str(spec_path),
            "--dry-run",
        ]
    )
    assert rc == EXIT_OK
    dry = json.loads(capsys.readouterr().out)
    assert dry["attempt_id"] == "<assigned-at-run>"
    assert dry["serve_config"]["api-key"] == "<REDACTED>"
    assert dry["timeout_s"] == 3600

    # 4. run --upload is rejected BEFORE launching any server.
    launcher = FakeLauncher()
    rc = main(
        [
            "--catalog-dir",
            str(cdir),
            "run",
            "--spec",
            str(spec_path),
            "--upload",
        ],
        launcher=launcher,
    )
    assert rc == EXIT_VALIDATION_ERROR
    assert launcher.starts == 0
    capsys.readouterr()


def test_cli_run_validate_bundle_without_catalog_and_failure_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cdir = _setup_cli_catalog(tmp_path)
    ok_spec = tmp_path / "ok.yaml"
    ok_spec.write_text(
        "schema_version: 1\nserve:\n  config:\n    model: org/m\n"
        f"    revision: {VALID_SHA}\nhardware:\n  gpu: [h200]\n"
        "benchmarks:\n  - type: ok_bench\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "runs"
    rc = main(
        [
            "--catalog-dir",
            str(cdir),
            "run",
            "--spec",
            str(ok_spec),
            "--out",
            str(out_dir),
        ],
        launcher=FakeLauncher(),
        assets=FixtureAssetProvider(tmp_path),
    )
    assert rc == EXIT_OK
    bundles = list(out_dir.rglob("bundle.tar.gz"))
    assert len(bundles) == 1
    capsys.readouterr()

    # validate-bundle does NOT load the catalog (works even with a broken catalog dir).
    broken_cat = tmp_path / "broken_cat"
    (broken_cat / "workloads").mkdir(parents=True)
    (broken_cat / "workloads" / "bad.yaml").write_text(": [broken", encoding="utf-8")
    assert (
        main(["--catalog-dir", str(broken_cat), "validate-bundle", str(bundles[0])])
        == EXIT_OK
    )
    capsys.readouterr()

    # Failing run prints error message to stderr and exits 1.
    fail_spec = tmp_path / "fail.yaml"
    fail_spec.write_text(
        "schema_version: 1\nserve:\n  config:\n    model: org/m\n"
        f"    revision: {VALID_SHA}\nhardware:\n  gpu: [h200]\n"
        "benchmarks:\n  - type: fail_bench\n",
        encoding="utf-8",
    )
    rc_fail = main(
        [
            "--catalog-dir",
            str(cdir),
            "run",
            "--spec",
            str(fail_spec),
            "--out",
            str(tmp_path / "fail_runs"),
        ],
        launcher=FakeLauncher(),
        assets=FixtureAssetProvider(tmp_path),
    )
    assert rc_fail == EXIT_EXEC_FAILED
    assert "boom-from-plugin" in capsys.readouterr().err


def test_strict_contract_helper_passes_real_and_rejects_noop(tmp_path: Path) -> None:
    cdir = _setup_cli_catalog(tmp_path)
    cat = Catalog.discover(cdir)
    assert_plugin_contract(
        workload_name="ok_bench",
        tmp_path=tmp_path / "ok",
        catalog=cat,
    )
    with pytest.raises(AssertionError, match="did not send any HTTP requests"):
        assert_plugin_contract(
            workload_name="noop_bench",
            tmp_path=tmp_path / "noop",
            catalog=cat,
        )


def test_module_entrypoint_help_subprocess() -> None:
    """Smoke-test `python -m tools.byog --help` in a fresh interpreter (no cycles)."""
    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [sys.executable, "-m", "tools.byog", "--help"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage: byog" in proc.stdout
