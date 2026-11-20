# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C0 contract tests: schemas, invariants, fixtures, fake server, fake launcher."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.byog.fake_launcher import FakeLauncher
from tests.byog.fake_server import FakeServerState, FakeVLLMServer
from tools.byog.executor.interfaces import (
    CacheResetUnsupportedError,
    FixtureAssetProvider,
    ServerTransportError,
    WorkloadPoint,
)
from tools.byog.schemas import (
    BundleManifest,
    EvalRenderedInputRecord,
    EvalResponseRecord,
    EvalSampleRecord,
    EvidenceSource,
    ExecutionMode,
    ExecutionStatus,
    LifecyclePolicy,
    MeasurementRecord,
    PinnedAsset,
    PublicationPolicy,
    QualityState,
    QualityThreshold,
    ResolvedJob,
    ResolvedWorkload,
    RunResult,
    RunSpec,
    ThresholdCheckResult,
    UploadConflictResponse,
    UploadRequestMeta,
    UploadSuccessResponse,
    WorkloadDefinition,
    WorkloadKind,
    compute_compatibility_key,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_COMPONENTS = {
    "workload": "gsm8k",
    "point_id": "c1",
    "serve_config": {"model": "org/m"},
}
SAMPLE_KEY = compute_compatibility_key(SAMPLE_COMPONENTS)


def _http_json(
    url: str,
    payload: dict | bytes | None = None,
    method: str | None = None,
) -> tuple[int, dict]:
    data = None
    headers = {}
    if isinstance(payload, bytes):
        data = payload
        headers["Content-Type"] = "application/json"
    elif payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method or ("POST" if data is not None else "GET"),
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8"))


def _base_measurement(**overrides: object) -> dict:
    base: dict = {
        "workload": "gsm8k",
        "kind": "eval",
        "point_id": "c1",
        "execution_mode": "server",
        "protocol_version": "1.0",
        "requested_lifecycle": LifecyclePolicy.NONE,
        "protocol_variant": "none",
        "evidence_source": EvidenceSource.HARNESS,
        "compatibility_components": SAMPLE_COMPONENTS,
        "compatibility_key": SAMPLE_KEY,
    }
    base.update(overrides)
    return base


def test_catalog_and_resolved_workload_invariants() -> None:
    # Direct workload cannot request server lifecycle reset/restart.
    with pytest.raises(ValidationError):
        WorkloadDefinition(
            name="bad_direct",
            kind=WorkloadKind.BENCHMARK,
            execution_mode=ExecutionMode.DIRECT,
            lifecycle=LifecyclePolicy.RESET_CACHE,
            entrypoint="a.b:C",
        )
    # Eval + direct rejected in both WorkloadDefinition and ResolvedWorkload.
    with pytest.raises(ValidationError):
        WorkloadDefinition(
            name="bad_eval",
            kind=WorkloadKind.EVAL,
            execution_mode=ExecutionMode.DIRECT,
            lifecycle=LifecyclePolicy.NONE,
            entrypoint="a.b:C",
        )
    with pytest.raises(ValidationError):
        ResolvedWorkload(
            name="bad_eval",
            kind="eval",
            execution_mode="direct",
            lifecycle="none",
            entrypoint="a.b:C",
            protocol_version="1.0",
        )
    # required_for_quality=True without catalog quality_thresholds rejected.
    with pytest.raises(ValidationError):
        WorkloadDefinition(
            name="req_no_th",
            kind=WorkloadKind.EVAL,
            execution_mode=ExecutionMode.SERVER,
            lifecycle=LifecyclePolicy.NONE,
            entrypoint="a.b:C",
            required_for_quality=True,
        )
    with pytest.raises(ValidationError):
        ResolvedWorkload(
            name="req_spec_only",
            kind="eval",
            execution_mode="server",
            lifecycle="none",
            entrypoint="a.b:C",
            protocol_version="1.0",
            required_for_quality=True,
            quality_thresholds=[
                QualityThreshold(metric="accuracy", min_value=0.5, source="spec")
            ],
        )
    # PublicationPolicy is immutable (frozen).
    policy = PublicationPolicy(reviewed=False, allow_question_text=False)
    with pytest.raises(ValidationError):
        policy.reviewed = True  # type: ignore[misc]
    assert not policy.can_publish_question()

    # QualityThreshold requires at least one bound and min <= max.
    with pytest.raises(ValidationError):
        QualityThreshold(metric="accuracy")
    with pytest.raises(ValidationError):
        QualityThreshold(metric="accuracy", min_value=0.8, max_value=0.2)


def test_id_and_point_id_traversal_guards() -> None:
    with pytest.raises(ValidationError):
        ResolvedJob(job_id="../escape", attempt_id="att-1")
    with pytest.raises(ValidationError):
        ResolvedJob(job_id="default", attempt_id="a\nb")
    # Self-claimed `source_tier` rejected; must use advisory `claimed_source_tier`.
    with pytest.raises(ValidationError):
        ResolvedJob.model_validate(
            {"job_id": "default", "attempt_id": "att-1", "source_tier": "ci_nightly"}
        )
    with pytest.raises(ValidationError):
        RunSpec(schema_version=1)

    # Unconstrained point_id rejected in WorkloadPoint, EvalResponseRecord, Measurement.
    with pytest.raises(ValueError, match="SAFE_ID_PATTERN"):
        WorkloadPoint(point_id="../../x")
    with pytest.raises(ValidationError):
        MeasurementRecord.model_validate(_base_measurement(point_id="../../x"))
    with pytest.raises(ValidationError):
        EvalResponseRecord(
            workload="gsm8k",
            point_id="../../x",
            dataset_id="ds",
            dataset_revision="r1",
            sample_id="s-0",
            input_hash="12345678",
            repeat=0,
            concurrency=1,
        )


def test_threshold_check_derived_and_validated() -> None:
    # Auto-derives passed when omitted.
    ok_check = ThresholdCheckResult.model_validate(
        {"metric": "accuracy", "observed": 0.9, "min_value": 0.5}
    )
    assert ok_check.passed is True
    fail_check = ThresholdCheckResult.model_validate(
        {"metric": "accuracy", "observed": 0.2, "min_value": 0.5}
    )
    assert fail_check.passed is False

    # Contradictory passed=True rejected.
    with pytest.raises(ValidationError):
        ThresholdCheckResult(
            metric="accuracy", observed=0.0, min_value=0.9, passed=True
        )
    # Missing observed with passed=True rejected.
    with pytest.raises(ValidationError):
        ThresholdCheckResult(
            metric="accuracy", observed=None, min_value=0.5, passed=True
        )
    # Missing both bounds rejected.
    with pytest.raises(ValidationError):
        ThresholdCheckResult(metric="accuracy", observed=0.5, passed=True)


def test_measurement_and_run_result_invariants() -> None:
    now = datetime(2026, 11, 20, 18, 0, 0, tzinfo=UTC)

    # evidence_source is required (no silent HARNESS default).
    no_ev = _base_measurement()
    no_ev.pop("evidence_source")
    with pytest.raises(ValidationError):
        MeasurementRecord.model_validate(no_ev)

    # Eval + direct rejected.
    with pytest.raises(ValidationError):
        MeasurementRecord.model_validate(
            _base_measurement(
                kind="eval", execution_mode="direct", protocol_variant="direct"
            )
        )

    # compatibility_key must match hash of compatibility_components.
    with pytest.raises(ValidationError):
        MeasurementRecord.model_validate(_base_measurement(compatibility_key="a" * 64))
    with pytest.raises(ValidationError):
        MeasurementRecord.model_validate(
            _base_measurement(compatibility_components={}, compatibility_key="a" * 64)
        )

    # Counter cross-checks: completed, denominator, errors vs attempted.
    with pytest.raises(ValidationError):
        MeasurementRecord.model_validate(_base_measurement(attempted=3, completed=4))
    with pytest.raises(ValidationError):
        MeasurementRecord.model_validate(
            _base_measurement(attempted=3, scoring_denominator=4)
        )
    with pytest.raises(ValidationError):
        MeasurementRecord.model_validate(
            _base_measurement(attempted=3, model_errors=2, infra_errors=2)
        )
    with pytest.raises(ValidationError):
        MeasurementRecord.model_validate(
            _base_measurement(attempted=3, completed=2, infra_errors=2)
        )

    # Silent lifecycle fallback (differs without reason) rejected.
    with pytest.raises(ValidationError):
        MeasurementRecord.model_validate(
            _base_measurement(
                requested_lifecycle=LifecyclePolicy.RESET_CACHE,
                protocol_variant="restart_server",
            )
        )

    # quality_state=PASSED requires passing catalog threshold (spec-only rejected!).
    with pytest.raises(ValidationError):
        MeasurementRecord.model_validate(
            _base_measurement(quality_state=QualityState.PASSED)
        )
    with pytest.raises(ValidationError):
        MeasurementRecord.model_validate(
            _base_measurement(
                quality_state=QualityState.PASSED,
                threshold_checks=[
                    ThresholdCheckResult(
                        metric="accuracy",
                        observed=0.9,
                        min_value=0.0,
                        source="spec",
                        passed=True,
                    )
                ],
            )
        )

    passing_req = MeasurementRecord.model_validate(
        _base_measurement(
            required_for_quality=True,
            quality_state=QualityState.PASSED,
            threshold_checks=[
                ThresholdCheckResult(
                    metric="accuracy",
                    observed=0.9,
                    min_value=0.5,
                    source="catalog",
                    passed=True,
                )
            ],
        )
    )
    optional_failed_eval = MeasurementRecord.model_validate(
        _base_measurement(
            workload="opt_eval",
            required_for_quality=False,
            quality_state=QualityState.FAILED,
            threshold_checks=[
                ThresholdCheckResult(
                    metric="accuracy",
                    observed=0.1,
                    min_value=0.5,
                    source="catalog",
                    passed=False,
                )
            ],
        )
    )
    failed_bench = MeasurementRecord.model_validate(
        _base_measurement(
            workload="http_smoke",
            kind="benchmark",
            status=ExecutionStatus.FAILED,
            quality_state=QualityState.NOT_EVALUATED,
        )
    )

    # RunResult rejects naive timestamps and finished_at < started_at.
    with pytest.raises(ValidationError):
        RunResult(
            job_id="default",
            attempt_id="att-1",
            execution_status=ExecutionStatus.COMPLETED,
            started_at=datetime(2026, 11, 20, 18, 0, 0),
            finished_at=datetime(2026, 11, 20, 18, 1, 0),
            duration_s=60.0,
            harness_version="0.1.0",
        )
    with pytest.raises(ValidationError):
        RunResult(
            job_id="default",
            attempt_id="att-1",
            execution_status=ExecutionStatus.COMPLETED,
            started_at=now,
            finished_at=now - timedelta(seconds=1),
            duration_s=0.0,
            harness_version="0.1.0",
        )

    # RunResult cannot be PASSED when execution FAILED or a benchmark FAILED.
    with pytest.raises(ValidationError):
        RunResult(
            job_id="default",
            attempt_id="att-1",
            execution_status=ExecutionStatus.FAILED,
            quality_state=QualityState.PASSED,
            started_at=now,
            finished_at=now,
            duration_s=0.0,
            harness_version="0.1.0",
            measurements=[passing_req],
        )
    with pytest.raises(ValidationError):
        RunResult(
            job_id="default",
            attempt_id="att-1",
            execution_status=ExecutionStatus.COMPLETED,
            quality_state=QualityState.PASSED,
            started_at=now,
            finished_at=now,
            duration_s=0.0,
            harness_version="0.1.0",
            measurements=[passing_req, failed_bench],
        )

    # RunResult CAN be PASSED when all required_for_quality passed and all executed
    # measurements completed, even if a non-required eval failed its threshold.
    ok_run = RunResult(
        job_id="default",
        attempt_id="att-1",
        execution_status=ExecutionStatus.COMPLETED,
        quality_state=QualityState.PASSED,
        started_at=now,
        finished_at=now,
        duration_s=1.0,
        harness_version="0.1.0",
        measurements=[passing_req, optional_failed_eval],
    )
    assert ok_run.quality_state == QualityState.PASSED


def test_upload_api_and_golden_fixtures() -> None:
    upload = json.loads(
        (FIXTURES_DIR / "upload_api_fixtures.json").read_text(encoding="utf-8")
    )
    for key in ("upload_success", "upload_idempotent_retry"):
        meta = UploadRequestMeta.model_validate(upload[key]["request_meta"])
        resp = UploadSuccessResponse.model_validate(upload[key]["response"])
        assert resp.attempt_id == meta.attempt_id
        assert resp.bundle_sha256 == meta.bundle_sha256

    conflict = upload["upload_attempt_conflict"]
    UploadRequestMeta.model_validate(conflict["request_meta"])
    c_resp = UploadConflictResponse.model_validate(conflict["response"])
    assert c_resp.error == "attempt_conflict"

    # Client cannot self-claim source_tier on UploadRequestMeta.
    bad_meta = dict(upload["upload_success"]["request_meta"])
    bad_meta["source_tier"] = "ci_nightly"
    with pytest.raises(ValidationError):
        UploadRequestMeta.model_validate(bad_meta)

    golden = json.loads(
        (FIXTURES_DIR / "golden_records.json").read_text(encoding="utf-8")
    )
    ResolvedJob.model_validate(golden["resolved_job"])
    parsed_res = RunResult.model_validate(golden["run_result"])
    comps = parsed_res.measurements[0].compatibility_components
    assert "serve_config" in comps and "eval_settings" in comps
    BundleManifest.model_validate(golden["bundle_manifest"])
    EvalSampleRecord.model_validate(golden["eval_sample"])
    EvalRenderedInputRecord.model_validate(golden["eval_rendered_input"])
    EvalResponseRecord.model_validate(golden["eval_response"])


def test_fixture_asset_provider_sha256_and_override(tmp_path: Path) -> None:
    f = tmp_path / "sample.txt"
    f.write_bytes(b"hello-byog")
    digest = hashlib.sha256(b"hello-byog").hexdigest()

    asset = PinnedAsset(
        id="sample",
        source=str(f),
        revision="rev1",
        sha256=digest,
    )
    provider = FixtureAssetProvider(tmp_path)
    resolved = provider.resolve(asset)
    assert resolved.sha256 == digest
    assert not resolved.is_override

    bad_asset = PinnedAsset(
        id="sample", source=str(f), revision="rev1", sha256="0" * 64
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        provider.resolve(bad_asset)

    # Explicit override marks is_override=True.
    override_provider = FixtureAssetProvider(tmp_path, overrides={"sample": f})
    ov = override_provider.resolve(bad_asset)
    assert ov.is_override
    assert ov.resolved_revision == "fixture-override"


def test_fake_server_strict_default_stream_and_fault_injection() -> None:
    # strict_model defaults to True!
    state = FakeServerState(
        model_name="strict-m",
        allow_reset_cache=False,
        fail_on_calls={2},
        fail_after_n=3,
    )
    assert state.strict_model is True
    with FakeVLLMServer(state=state) as srv:
        # Query string on /health works.
        code, body = _http_json(f"{srv.base_url}/health?probe=1")
        assert code == 200 and body["status"] == "ok"

        # Reset cache disabled by default -> 404.
        code, _ = _http_json(f"{srv.base_url}/reset_prefix_cache", payload={})
        assert code == 404

        # Invalid JSON -> 400.
        code, _ = _http_json(f"{srv.base_url}/v1/completions", payload=b"{bad")
        assert code == 400

        # stream=True -> 400 (not silently ignored).
        code, _ = _http_json(
            f"{srv.base_url}/v1/completions",
            payload={"model": "strict-m", "prompt": "hi", "stream": True},
        )
        assert code == 400

        # Wrong model rejected by default -> 404.
        code, _ = _http_json(
            f"{srv.base_url}/v1/completions",
            payload={"model": "other", "prompt": "hi"},
        )
        assert code == 404

        # Missing prompt -> 400.
        code, _ = _http_json(
            f"{srv.base_url}/v1/completions", payload={"model": "strict-m"}
        )
        assert code == 400

        # Call 1 succeeds, call 2 fails (fail_on_calls), call 3 succeeds,
        # call 4 fails (fail_after_n=3).
        c1, _ = _http_json(
            f"{srv.base_url}/v1/completions",
            payload={"model": "strict-m", "prompt": "hi"},
        )
        c2, _ = _http_json(
            f"{srv.base_url}/v1/completions",
            payload={"model": "strict-m", "prompt": "hi"},
        )
        c3, _ = _http_json(
            f"{srv.base_url}/v1/completions",
            payload={"model": "strict-m", "prompt": "hi"},
        )
        c4, _ = _http_json(
            f"{srv.base_url}/v1/completions",
            payload={"model": "strict-m", "prompt": "hi"},
        )
        assert (c1, c2, c3, c4) == (200, 500, 200, 500)


def test_fake_launcher_reset_modes(tmp_path: Path) -> None:
    launcher = FakeLauncher(allow_reset_cache=True)
    h = launcher.start({"model": "m"}, {}, tmp_path, 10, enable_dev_mode=False)
    with pytest.raises(CacheResetUnsupportedError):
        launcher.reset_cache(h)
    launcher.stop(h)

    launcher_tx = FakeLauncher(fail_reset_transport=True)
    h2 = launcher_tx.start({"model": "m"}, {}, tmp_path, 10, enable_dev_mode=True)
    with pytest.raises(ServerTransportError):
        launcher_tx.reset_cache(h2)
    launcher_tx.stop(h2)
