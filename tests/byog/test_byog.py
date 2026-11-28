# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 0 verification suite (schemas, resolver, bundle, executor, CLI, E2E)."""

from __future__ import annotations

import hashlib
import io
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tests.byog.fake_launcher import (
    TEST_CATALOG_DIR,
    FakeLauncher,
    assert_plugin_contract,
    make_offline_gsm8k_catalog,
)
from tools.byog.bundle import (
    BundleValidationError,
    DeduplicatedEvalRecordWriter,
    redact_value,
    seal_bundle,
    validate_bundle,
)
from tools.byog.catalog import Catalog
from tools.byog.cli import EXIT_EXEC_FAILED, EXIT_OK, EXIT_VALIDATION_ERROR, main
from tools.byog.executor.coordinator import Coordinator
from tools.byog.executor.nvidia import (
    _build_env,
    _normalize_gpu_id,
    build_vllm_serve_command,
)
from tools.byog.interfaces import WorkloadPoint
from tools.byog.schemas.execution import (
    EvalSelection,
    ExecutionMode,
    ExecutionStatus,
    LifecyclePolicy,
    RunSpec,
    SuiteDefinition,
    WorkloadKind,
)
from tools.byog.schemas.results import ArtifactEntry, MeasurementRecord
from tools.byog.spec import SpecResolutionError, load_spec_file, resolve_spec

FIXTURES = Path(__file__).resolve().parent / "fixtures"
VALID_SHA = "0123456789abcdef0123456789abcdef01234567"


def test_schema_invariants() -> None:
    with pytest.raises(ValueError):
        WorkloadPoint("../../escape")
    with pytest.raises(ValidationError):
        EvalSelection(name="gsm8k", concurrency=[0, -1])
    with pytest.raises(ValidationError):
        SuiteDefinition.model_validate(
            {"name": "bad", "evals": [{"name": "gsm8k", "concurrency": [0]}]}
        )
    with pytest.raises(ValidationError):
        ArtifactEntry(path="../escape", size_bytes=1, sha256="a" * 64)
    with pytest.raises(ValidationError):
        MeasurementRecord(
            workload="gsm8k",
            kind=WorkloadKind.EVAL,
            point_id="c1",
            execution_mode=ExecutionMode.SERVER,
            protocol_version="1.0",
            requested_lifecycle=LifecyclePolicy.NONE,
            actual_lifecycle=LifecyclePolicy.NONE,
            attempted=2,
            completed=3,
        )


@pytest.mark.parametrize(
    "bad_override",
    [
        {"recipe": {"model": "org/m"}},
        {"matrix": {"tp": [1, 2]}},
        {"compare_baseline": True},
        {"hardware": {"nodes": 2}},
    ],
)
def test_runspec_rejects_phase1_fields(bad_override: dict) -> None:
    base = {
        "schema_version": 1,
        "serve": {"config": {"model": "org/m", "revision": VALID_SHA}},
        "evals": [{"name": "gsm8k"}],
    }
    base.update(bad_override)
    with pytest.raises(ValidationError):
        RunSpec.model_validate(base)


@pytest.mark.parametrize(
    "bad_override",
    [
        {"serve": {"config": {"model": "m", "revision": "main"}}},
        {
            "serve": {
                "config": {"model": "m", "revision": VALID_SHA},
                "env": {"HF_TOKEN": "x"},
            }
        },
        {
            "serve": {
                "config": {"model": "m", "revision": VALID_SHA},
                "env": {"LD_AUDIT": "/evil.so"},
            }
        },
        {
            "serve": {
                "config": {"model": "m", "revision": VALID_SHA},
                "env": {"CUDA_VISIBLE_DEVICES": "0"},
            }
        },
        {
            "serve": {
                "config": {"model": "m", "revision": VALID_SHA},
                "env": {"HF_HOME": "/tmp/evil"},
            }
        },
        {
            "serve": {
                "config": {"model": "m", "revision": VALID_SHA},
                "env": {"VLLM_APIKEY": "secret"},
            }
        },
        # Specs must NOT override pinned catalog defaults (dataset URL / num_shots).
        {"evals": [{"name": "gsm8k", "test_source": "http://evil/x"}]},
        {"evals": [{"name": "gsm8k", "num_shots": 0}]},
        {"evals": [{"name": "gsm8k", "unknown_opt": 1}]},
    ],
)
def test_spec_resolver_security_and_guards(bad_override: dict) -> None:
    cat = Catalog.discover()
    base = {
        "schema_version": 1,
        "serve": {"config": {"model": "org/m", "revision": VALID_SHA}},
        "evals": [{"name": "gsm8k"}],
    }
    base.update(bad_override)
    with pytest.raises(SpecResolutionError):
        resolve_spec(RunSpec.model_validate(base), cat)


def test_launcher_security_flags_and_gpu_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_secret_123")
    monkeypatch.setenv("VLLM_APIKEY", "api_secret_456")
    env = _build_env({"VLLM_LOGGING_LEVEL": "INFO"}, enable_dev_mode=False)
    assert "HF_TOKEN" not in env
    assert "VLLM_APIKEY" not in env

    cmd = build_vllm_serve_command(
        {
            "model": "org/m",
            "enable-prefix-caching": False,
            "enforce-eager": True,
        }
    )
    assert "--no-enable-prefix-caching" in cmd
    assert "--enforce-eager" in cmd

    assert _normalize_gpu_id("NVIDIA GH200 480GB") == "gh200"
    assert _normalize_gpu_id("NVIDIA H200") == "h200"
    assert _normalize_gpu_id("NVIDIA L40") == "l40"
    assert _normalize_gpu_id("NVIDIA L40S") == "l40s"
    assert _normalize_gpu_id("NVIDIA L4") == "l4"


def test_suite_override_and_skip_precedence() -> None:
    cat = Catalog.discover(TEST_CATALOG_DIR)
    base = {
        "schema_version": 1,
        "serve": {"config": {"model": "org/m", "revision": VALID_SHA}},
        "benchmark_suite": "test_all",
    }
    spec = RunSpec.model_validate(
        {**base, "benchmarks": [{"type": "dummy_server_bench"}], "evals": []}
    )
    job = resolve_spec(spec, cat)[0]
    assert [w.definition.name for w in job.workloads] == ["dummy_server_bench"]


def test_bundle_redaction_tar_safety_and_dedup(tmp_path: Path) -> None:
    red = redact_value(
        {
            "args": ["--api-key", "secret1", "--token=secret2"],
            "url": "https://user:pw@example.com/a",
            "VLLM_APIKEY": "secret3",
        }
    )
    assert "secret" not in str(red) and "pw" not in str(red)

    w = DeduplicatedEvalRecordWriter(workload="ev", output_dir=tmp_path / "ev")
    for rep in (0, 1):
        w.write_response(
            dataset_id="ds",
            dataset_revision="r1",
            sample_id="s-0",
            question="2+2?",
            reference_answer="4",
            rendered_input="Q: 2+2?",
            point_id="c1",
            repeat=rep,
            concurrency=1,
            model_response="4",
            extracted_answer="4",
            score=1.0,
            correct=True,
        )
    paths = w.close()
    assert len(paths[0].read_text().splitlines()) == 1
    assert len(paths[1].read_text().splitlines()) == 1
    assert len(paths[2].read_text().splitlines()) == 2

    # Duplicate normalized shadow member (`a//b` vs `a/b`) is rejected.
    dup_tar = tmp_path / "dup.tar.gz"
    with tarfile.open(dup_tar, "w:gz") as tar:
        for name in ("a/b.txt", "a//b.txt"):
            info = tarfile.TarInfo(name)
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(BundleValidationError):
        validate_bundle(dup_tar)


def test_plugin_contracts_and_e2e_yaml_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "must-not-leak")
    # Test-only benchmark plugins discovered from real YAML files via --catalog-dir.
    test_cat = Catalog.discover(TEST_CATALOG_DIR)
    res_srv, _ = assert_plugin_contract(
        workload_name="dummy_server_bench",
        tmp_path=tmp_path / "server_bench",
        catalog=test_cat,
        overrides={"concurrency": [1, 2]},
    )
    # Point 0 started fresh (`none`), point 1 reset cache (`reset_cache`).
    assert [m.actual_lifecycle for m in res_srv.measurements] == [
        LifecyclePolicy.NONE,
        LifecyclePolicy.RESET_CACHE,
    ]
    res_dir, _ = assert_plugin_contract(
        workload_name="dummy_direct_bench",
        tmp_path=tmp_path / "direct_bench",
        catalog=test_cat,
    )
    assert res_dir.measurements[0].metrics["hf_token_leaked"] == 0

    # Production catalog ships only real design-doc workloads (`gsm8k` in Phase 0).
    prod_cat = Catalog.discover()
    assert [w.name for w in prod_cat.list_workloads()] == ["gsm8k"]

    # Offline GSM8K catalog verifies real SHA256 pin + end-to-end bundle & repro.sh.
    offline_cat_dir = make_offline_gsm8k_catalog(tmp_path)
    offline_cat = Catalog.discover(offline_cat_dir)
    raw = yaml.safe_load((FIXTURES / "smoke_gpu_spec.yaml").read_text())
    raw["evals"][0]["num_questions"] = 6
    local_spec = tmp_path / "local_smoke.yaml"
    local_spec.write_text(yaml.safe_dump(raw), encoding="utf-8")

    job = resolve_spec(load_spec_file(local_spec), offline_cat)[0]
    res, tar1 = Coordinator(
        launcher=FakeLauncher(),
        vllm_commit=VALID_SHA,
        image_ref="img:$(touch /tmp/pwned)",
    ).run(job, tmp_path / "e2e", source_spec_path=local_spec)
    assert res.execution_status == ExecutionStatus.COMPLETED
    assert res.measurements[0].metrics["accuracy"] == 1.0
    assert res.measurements[0].actual_lifecycle == LifecyclePolicy.NONE
    assert res.measurements[1].actual_lifecycle == LifecyclePolicy.RESET_CACHE
    assert res.observed_hardware.gpu_id == "h200"

    # repro.sh passes bash -n and quotes image_ref safely.
    repro_path = tar1.parent / "repro.sh"
    assert subprocess.run(["bash", "-n", str(repro_path)], check=False).returncode == 0
    assert "DEFAULT_IMAGE='img:$(touch /tmp/pwned)'" in repro_path.read_text()

    h1 = hashlib.sha256(tar1.read_bytes()).hexdigest()
    h2 = hashlib.sha256(
        seal_bundle(tar1.parent, res.job_id, res.attempt_id).read_bytes()
    ).hexdigest()
    assert h1 == h2
    validate_bundle(tar1)

    # Replayable spec.resolved.yaml parses cleanly against the catalog.
    resolved_spec = load_spec_file(tar1.parent / "spec.resolved.yaml")
    assert resolve_spec(resolved_spec, offline_cat)[0].job_id == "default"


def test_gsm8k_fails_on_infra_errors_and_coordinator_enforces_timeout(
    tmp_path: Path,
) -> None:
    # Partial infra errors (5xx on 2 of 6 questions) must fail the eval point.
    offline_cat = Catalog.discover(make_offline_gsm8k_catalog(tmp_path))
    spec = RunSpec.model_validate(
        {
            "schema_version": 1,
            "serve": {"config": {"model": "org/m", "revision": VALID_SHA}},
            "evals": [{"name": "gsm8k", "num_questions": 6}],
        }
    )
    job = resolve_spec(spec, offline_cat)[0]
    res, _ = Coordinator(launcher=FakeLauncher(fail_on_calls={2, 4})).run(
        job, tmp_path / "partial"
    )
    assert res.execution_status == ExecutionStatus.FAILED
    assert res.measurements[0].infra_errors == 2

    # 3s plugin under a 1s job timeout must time out promptly (not run to completion).
    test_cat = Catalog.discover(TEST_CATALOG_DIR)
    slow_spec = RunSpec.model_validate(
        {
            "schema_version": 1,
            "serve": {"config": {"model": "org/m", "revision": VALID_SHA}},
            "benchmarks": [{"type": "dummy_server_bench", "sleep_s": 3.0}],
            "timeout_s": 1,
        }
    )
    slow_job = resolve_spec(slow_spec, test_cat)[0]
    res_to, _ = Coordinator(launcher=FakeLauncher()).run(slow_job, tmp_path / "to")
    assert res_to.execution_status == ExecutionStatus.TIMED_OUT


def test_cli_flags_exit_codes_and_no_import_cycles(tmp_path: Path) -> None:
    # --job validation + unknown job exits 2; failed run exits 1.
    assert (
        main(["validate", "--spec", str(FIXTURES / "smoke_gpu_spec.yaml")]) == EXIT_OK
    )
    assert (
        main(
            [
                "run",
                "--spec",
                str(FIXTURES / "smoke_gpu_spec.yaml"),
                "--job",
                "nonexistent",
                "--dry-run",
            ]
        )
        == EXIT_VALIDATION_ERROR
    )
    assert (
        main(
            [
                "--catalog-dir",
                str(TEST_CATALOG_DIR),
                "run",
                "--spec",
                str(FIXTURES / "smoke_gpu_spec.yaml"),
                "--job",
                "default",
                "--dry-run",
            ]
        )
        == EXIT_VALIDATION_ERROR
    )
    # Runtime failure (reset_cache unsupported) exits 1, not 2.
    bad_spec = tmp_path / "fb.yaml"
    bad_spec.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "serve": {"config": {"model": "org/m", "revision": VALID_SHA}},
                "benchmarks": [{"type": "dummy_server_bench", "concurrency": [1, 2]}],
            }
        ),
        encoding="utf-8",
    )
    rc = main(
        [
            "--catalog-dir",
            str(TEST_CATALOG_DIR),
            "run",
            "--spec",
            str(bad_spec),
            "--out",
            str(tmp_path / "cli_out"),
            "--job",
            "default",
        ],
        launcher=FakeLauncher(allow_reset_cache=False),
    )
    assert rc == EXIT_EXEC_FAILED

    repo_root = Path(__file__).resolve().parents[2]
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
