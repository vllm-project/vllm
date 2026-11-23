# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for C5 bundle writer, eval record deduplication, and validator."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tools.byog.bundle import (
    BundleValidationError,
    DeduplicatedEvalRecordWriter,
    redact_env,
    redact_serve_config,
    seal_bundle,
    validate_bundle,
    write_bundle_metadata,
)
from tools.byog.schemas.catalog import (
    EvidenceSource,
    LifecyclePolicy,
    PublicationPolicy,
    QualityThreshold,
)
from tools.byog.schemas.result import (
    ErrorKind,
    ExecutionStatus,
    MeasurementRecord,
    ProtocolVariant,
    QualityState,
    RunResult,
    ThresholdCheckResult,
    compute_compatibility_key,
)
from tools.byog.schemas.spec import HardwareSpec, ResolvedJob, ResolvedWorkload
from tools.byog.spec import load_spec_file

VALID_SHA = "0123456789abcdef0123456789abcdef01234567"
HASH_A = "a" * 64


def _sample_job_and_result() -> tuple[ResolvedJob, RunResult]:
    now = datetime(2026, 11, 20, 18, 0, 0, tzinfo=UTC)
    job = ResolvedJob(
        job_id="default",
        attempt_id="att-bundle-01",
        spec_name="bundle-test",
        serve_config={
            "model": "org/model",
            "revision": VALID_SHA,
            "api-key": "super-secret-api-key",
            "hf-url": "https://user:pass@example.com/model",
        },
        serve_env={"VLLM_USE_V1": "1", "HF_TOKEN": "hf_secret_123"},
        hardware=HardwareSpec(gpu=["h200"], count=1),
        workloads=[
            ResolvedWorkload(
                name="gsm8k",
                kind="eval",
                execution_mode="server",
                lifecycle="reset_cache",
                entrypoint="tools.byog.executor.evals.gsm8k.plugin:GSM8KEvalRunner",
                protocol_version="1.0",
                quality_thresholds=[QualityThreshold(metric="accuracy", min_value=0.5)],
                params={"num_questions": 2, "repeats": 1, "concurrency": [1]},
            )
        ],
        timeout_s=600,
    )
    res = RunResult(
        job_id=job.job_id,
        attempt_id=job.attempt_id,
        spec_name=job.spec_name,
        execution_status=ExecutionStatus.COMPLETED,
        quality_state=QualityState.PASSED,
        started_at=now,
        finished_at=now,
        duration_s=1.5,
        harness_version="0.1.0",
        measurements=[
            MeasurementRecord(
                workload="gsm8k",
                kind="eval",
                point_id="c1",
                execution_mode="server",
                protocol_version="1.0",
                requested_lifecycle=LifecyclePolicy.RESET_CACHE,
                protocol_variant=ProtocolVariant.RESET_CACHE,
                evidence_source=EvidenceSource.HARNESS,
                required_for_quality=True,
                compatibility_components={"workload": "gsm8k", "point_id": "c1"},
                compatibility_key=compute_compatibility_key(
                    {"workload": "gsm8k", "point_id": "c1"}
                ),
                quality_state=QualityState.PASSED,
                threshold_checks=[
                    ThresholdCheckResult(
                        metric="accuracy",
                        observed=1.0,
                        min_value=0.5,
                        source="catalog",
                        passed=True,
                    )
                ],
                metrics={"accuracy": 1.0},
                attempted=2,
                completed=2,
                scoring_denominator=2,
            )
        ],
    )
    return job, res


def test_bundle_roundtrip_redaction_and_determinism(tmp_path: Path) -> None:
    job, res = _sample_job_and_result()
    src_spec = tmp_path / "input_spec.yaml"
    src_spec.write_text(
        "schema_version: 1\n"
        "serve:\n"
        "  config:\n"
        "    model: org/model\n"
        f"    revision: {VALID_SHA}\n"
        "    api-key: raw-spec-secret\n"
        "  env:\n"
        "    VLLM_USE_V1: '1'\n"
        "    HF_TOKEN: raw-token\n"
        "evals:\n"
        "  - name: gsm8k\n",
        encoding="utf-8",
    )

    bdir = tmp_path / "bundle"
    write_bundle_metadata(
        bdir,
        job=job,
        result=res,
        source_spec_path=src_spec,
        image_ref="vllm/byog@sha256:" + HASH_A,
        host_env={
            "VLLM_USE_V1": "1",
            "AWS_SECRET_ACCESS_KEY": "dont-leak",
            "RANDOM_HOST_VAR": "should-be-dropped",
            "VLLM_PROXY_URL": "https://alice:hunter2@proxy.internal/api",
        },
    )

    # Secrets are redacted across all serialized bundle metadata files.
    for fname in ("spec.yaml", "spec.resolved.yaml", "env.json", "resolved_job.json"):
        txt = (bdir / fname).read_text(encoding="utf-8")
        assert "raw-spec-secret" not in txt
        assert "raw-token" not in txt
        assert "super-secret-api-key" not in txt
        assert "hf_secret_123" not in txt
        assert "dont-leak" not in txt
        assert "hunter2" not in txt
        assert "RANDOM_HOST_VAR" not in txt

    # spec.resolved.yaml is a directly replayable RunSpec!
    replayed = load_spec_file(bdir / "spec.resolved.yaml")
    assert replayed.schema_version == 1
    assert replayed.serve is not None
    assert replayed.serve.config["model"] == "org/model"
    assert replayed.evals is not None and replayed.evals[0].name == "gsm8k"
    assert replayed.evals[0].min_accuracy == 0.5

    # Seal twice and confirm byte-for-byte identical archive SHA256.
    tar1 = seal_bundle(bdir, job.job_id, job.attempt_id)
    h1 = hashlib.sha256(tar1.read_bytes()).hexdigest()
    tar2 = seal_bundle(bdir, job.job_id, job.attempt_id)
    h2 = hashlib.sha256(tar2.read_bytes()).hexdigest()
    assert h1 == h2

    manifest, parsed_res = validate_bundle(tar1)
    assert manifest.attempt_id == job.attempt_id
    assert parsed_res.quality_state == QualityState.PASSED


def test_eval_record_dedup_and_withholding(tmp_path: Path) -> None:
    restricted = PublicationPolicy(
        reviewed=False,
        allow_question_text=True,
        allow_reference_answer=True,
        allow_model_response=True,
    )
    w = DeduplicatedEvalRecordWriter(
        workload="restricted_eval",
        output_dir=tmp_path / "evals" / "restricted_eval",
        policy=restricted,
    )
    # Write 2 repeats of the same sample; samples & inputs deduplicate to 1 line.
    for rep in (0, 1):
        w.write_response(
            dataset_id="ds1",
            dataset_revision="r1",
            sample_id="s-0",
            question="SECRET QUESTION",
            reference_answer="SECRET REF",
            rendered_input="SECRET PROMPT",
            point_id="c1",
            repeat=rep,
            concurrency=1,
            model_response="SECRET OUTPUT",
            extracted_answer="SECRET EXTRACTED",
            score=1.0,
            correct=True,
            error_kind=ErrorKind.MODEL,
            error_message="SECRET ERROR QUOTING MODEL OUTPUT",
        )
    paths = w.close()
    assert w.written_count == 2
    assert len(paths[0].read_text().splitlines()) == 1
    assert len(paths[1].read_text().splitlines()) == 1
    assert len(paths[2].read_text().splitlines()) == 2

    combined = "\n".join(p.read_text(encoding="utf-8") for p in paths)
    assert "SECRET" not in combined
    resp = json.loads(paths[2].read_text().splitlines()[0])
    assert resp["extracted_answer"] is None
    assert resp["error_message"] is None
    assert resp["error_kind"] == "model"


def test_validator_rejects_traversal_symlinks_and_tampering(tmp_path: Path) -> None:
    def _make_tar(name: str, entries: list[tuple[tarfile.TarInfo, bytes]]) -> Path:
        p = tmp_path / name
        with tarfile.open(p, "w:gz") as tar:
            for info, data in entries:
                if info.isfile():
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
                else:
                    tar.addfile(info)
        return p

    # 1. Path traversal `../escape.txt` (even behind `./`).
    for bad_name in ("../escape.txt", "./../escape.txt", "/etc/passwd"):
        info = tarfile.TarInfo(name=bad_name)
        tar_p = _make_tar("bad.tar.gz", [(info, b"x")])
        with pytest.raises(BundleValidationError):
            validate_bundle(tar_p)

    # 2. Symlink member inside tar rejected.
    sym = tarfile.TarInfo(name="link.txt")
    sym.type = tarfile.SYMTYPE
    sym.linkname = "/etc/passwd"
    sym_tar = _make_tar("sym.tar.gz", [(sym, b"")])
    with pytest.raises(BundleValidationError, match="unsupported non-file"):
        validate_bundle(sym_tar)

    # 3. Tampered file hash rejected.
    job, res = _sample_job_and_result()
    bdir = tmp_path / "good"
    write_bundle_metadata(bdir, job=job, result=res)
    good_tar = seal_bundle(bdir, job.job_id, job.attempt_id)

    tampered = tmp_path / "tampered.tar.gz"
    with (
        tarfile.open(good_tar, "r:gz") as src,
        tarfile.open(tampered, "w:gz") as dst,
    ):
        for m in src.getmembers():
            raw = src.extractfile(m).read()  # type: ignore[union-attr]
            if m.name == "repro.sh":
                raw = raw + b"# tampered\n"
            m.size = len(raw)
            dst.addfile(m, io.BytesIO(raw))
    with pytest.raises(BundleValidationError, match="Size mismatch"):
        validate_bundle(tampered)


def test_redact_helpers() -> None:
    env = redact_env(
        {"VLLM_USE_V1": "1", "VLLM_API_KEY": "abc", "SECRET_FOO": "xyz"},
        allowlist_only=True,
    )
    assert env == {"VLLM_USE_V1": "1", "VLLM_API_KEY": "<REDACTED>"}
    cfg = redact_serve_config({"model": "m", "hf_token": "tok"})
    assert cfg == {"model": "m", "hf_token": "<REDACTED>"}
