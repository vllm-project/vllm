# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C0 contract tests: schemas, invariants, fixtures, fake server, fake launcher."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.byog.fake_launcher import FakeLauncher
from tests.byog.fake_server import FakeServerState, FakeVLLMServer
from tools.byog.executor.interfaces import (
    CacheResetUnsupportedError,
    FixtureAssetProvider,
    ServerTransportError,
)
from tools.byog.schemas import (
    BundleManifest,
    EvalRenderedInputRecord,
    EvalResponseRecord,
    EvalSampleRecord,
    ExecutionMode,
    ExecutionStatus,
    LifecyclePolicy,
    MeasurementRecord,
    PinnedAsset,
    PublicationPolicy,
    QualityState,
    QualityThreshold,
    ResolvedJob,
    RunResult,
    RunSpec,
    ThresholdCheckResult,
    UploadConflictResponse,
    UploadRequestMeta,
    UploadSuccessResponse,
    WorkloadDefinition,
    WorkloadKind,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
HASH_A = "a" * 64


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


def test_catalog_invariants() -> None:
    # Direct workload cannot request server lifecycle reset/restart.
    with pytest.raises(ValidationError):
        WorkloadDefinition(
            name="bad_direct",
            kind=WorkloadKind.BENCHMARK,
            execution_mode=ExecutionMode.DIRECT,
            lifecycle=LifecyclePolicy.RESET_CACHE,
            entrypoint="a.b:C",
        )
    # Eval workload cannot use direct execution mode.
    with pytest.raises(ValidationError):
        WorkloadDefinition(
            name="bad_eval",
            kind=WorkloadKind.EVAL,
            execution_mode=ExecutionMode.DIRECT,
            lifecycle=LifecyclePolicy.NONE,
            entrypoint="a.b:C",
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


def test_id_traversal_and_spec_invariants() -> None:
    with pytest.raises(ValidationError):
        ResolvedJob(job_id="../escape", attempt_id="att-1")
    with pytest.raises(ValidationError):
        ResolvedJob(job_id="default", attempt_id="a\nb")
    with pytest.raises(ValidationError):
        RunSpec(schema_version=1)  # neither recipe nor serve


def test_measurement_and_run_result_invariants() -> None:
    now = datetime(2026, 11, 20, 18, 0, 0, tzinfo=UTC)

    # completed > attempted rejected.
    with pytest.raises(ValidationError):
        MeasurementRecord(
            workload="gsm8k",
            kind="eval",
            point_id="c1",
            execution_mode="server",
            protocol_version="1.0",
            requested_lifecycle=LifecyclePolicy.RESET_CACHE,
            protocol_variant="reset_cache",  # type: ignore[arg-type]
            compatibility_key=HASH_A,
            attempted=3,
            completed=4,
        )

    # Silent lifecycle fallback (differs without reason) rejected.
    with pytest.raises(ValidationError):
        MeasurementRecord(
            workload="gsm8k",
            kind="eval",
            point_id="c1",
            execution_mode="server",
            protocol_version="1.0",
            requested_lifecycle=LifecyclePolicy.RESET_CACHE,
            protocol_variant="restart_server",  # type: ignore[arg-type]
            compatibility_key=HASH_A,
        )

    # quality_state=PASSED without passing threshold checks rejected.
    with pytest.raises(ValidationError):
        MeasurementRecord(
            workload="gsm8k",
            kind="eval",
            point_id="c1",
            execution_mode="server",
            protocol_version="1.0",
            requested_lifecycle=LifecyclePolicy.NONE,
            protocol_variant="none",  # type: ignore[arg-type]
            compatibility_key=HASH_A,
            quality_state=QualityState.PASSED,
        )

    # RunResult cannot be PASSED when execution FAILED.
    passing_m = MeasurementRecord(
        workload="gsm8k",
        kind="eval",
        point_id="c1",
        execution_mode="server",
        protocol_version="1.0",
        requested_lifecycle=LifecyclePolicy.NONE,
        protocol_variant="none",  # type: ignore[arg-type]
        compatibility_key=HASH_A,
        quality_state=QualityState.PASSED,
        threshold_checks=[
            ThresholdCheckResult(
                metric="accuracy", observed=0.9, min_value=0.5, passed=True
            )
        ],
    )
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
            measurements=[passing_m],
        )


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
    RunResult.model_validate(golden["run_result"])
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


def test_fake_server_hardened_and_fault_injection() -> None:
    state = FakeServerState(
        model_name="strict-m",
        strict_model=True,
        allow_reset_cache=False,
        fail_on_calls={2},
        fail_after_n=3,
    )
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

        # Wrong model with strict_model=True -> 404.
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
