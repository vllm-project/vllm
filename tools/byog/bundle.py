# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Result packaging (C5): bundle writer, eval JSONL dedup, redaction, validation."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shlex
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml
from pydantic import ValidationError

from tools.byog.schemas.execution import ErrorKind, ResolvedJob, WorkloadKind
from tools.byog.schemas.results import (
    ArtifactEntry,
    BundleManifest,
    EvalRenderedInputRecord,
    EvalResponseRecord,
    EvalSampleRecord,
    RunResult,
)

_ALLOWED_ENV_PREFIXES = (
    "VLLM_",
    "CUDA_",
    "HIP_",
    "ROCR_",
    "NCCL_",
    "TORCH_",
    "OMP_",
    "HF_HUB_CACHE",
    "BYOG_",
)
_SECRET_SUBSTRS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AUTH",
    "APIKEY",
    "API_KEY",
    "API-KEY",
    "SSL_KEYFILE",
    "SSL-KEYFILE",
)
_SECRET_TOKENS = frozenset({"KEY", "PAT"})
MAX_TAR_MEMBERS = 10_000
MAX_TAR_BYTES = 512 * 1024 * 1024
REQUIRED_FILES = frozenset(
    {"manifest.json", "result.json", "spec.resolved.yaml", "repro.sh"}
)


class BundleValidationError(ValueError):
    """Raised when a bundle fails safety, schema, or integrity validation."""


def _is_secret_key(name: str) -> bool:
    up = name.upper().lstrip("-")
    return any(s in up for s in _SECRET_SUBSTRS) or bool(
        set(up.replace("-", "_").split("_")) & _SECRET_TOKENS
    )


def _strip_url_creds(val: str) -> str:
    if "://" not in val or "@" not in val:
        return val
    try:
        p = urlsplit(val)
    except ValueError:
        return val
    if p.username is None and p.password is None:
        return val
    host = p.hostname or ""
    netloc = f"***@{host}" + (f":{p.port}" if p.port is not None else "")
    return urlunsplit((p.scheme, netloc, p.path, p.query, p.fragment))


def redact_env(env: dict[str, str], *, allowlist_only: bool = True) -> dict[str, str]:
    out: dict[str, str] = {}
    for k in sorted(env):
        if allowlist_only and not k.upper().startswith(_ALLOWED_ENV_PREFIXES):
            continue
        out[k] = "<REDACTED>" if _is_secret_key(k) else _strip_url_creds(str(env[k]))
    return out


def redact_value(val: Any) -> Any:
    """Recursively redact secret keys, CLI flag values, and URL credentials."""
    if isinstance(val, dict):
        return {
            k: "<REDACTED>" if _is_secret_key(str(k)) else redact_value(v)
            for k, v in val.items()
        }
    if isinstance(val, list):
        out: list[Any] = []
        hide_next = False
        for item in val:
            if hide_next:
                out.append("<REDACTED>")
                hide_next = False
            elif isinstance(item, str) and item.startswith("-"):
                if "=" in item:
                    flag, _ = item.split("=", 1)
                    out.append(
                        f"{flag}=<REDACTED>"
                        if _is_secret_key(flag)
                        else _strip_url_creds(item)
                    )
                else:
                    out.append(item)
                    hide_next = _is_secret_key(item)
            else:
                out.append(redact_value(item))
        return out
    if isinstance(val, str):
        return _strip_url_creds(val)
    return val


def redact_serve_config(cfg: dict[str, Any]) -> dict[str, Any]:
    redacted = redact_value(cfg)
    assert isinstance(redacted, dict)
    return redacted


class DeduplicatedEvalRecordWriter:
    """Writes deduplicated `samples.jsonl`, `inputs.jsonl`, `responses.jsonl`."""

    def __init__(self, *, workload: str, output_dir: Path):
        self._workload = workload
        output_dir.mkdir(parents=True, exist_ok=True)
        self._paths = [
            output_dir / "samples.jsonl",
            output_dir / "inputs.jsonl",
            output_dir / "responses.jsonl",
        ]
        self._fhs = [p.open("w", encoding="utf-8") for p in self._paths]
        self._seen_samples: set[tuple[str, str, str]] = set()
        self._seen_inputs: set[str] = set()
        self._written_count = 0
        self._closed = False

    @property
    def written_count(self) -> int:
        return self._written_count

    def write_response(
        self,
        *,
        dataset_id: str,
        dataset_revision: str,
        sample_id: str,
        question: str | None,
        reference_answer: str | None,
        rendered_input: Any,
        point_id: str,
        repeat: int,
        concurrency: int,
        model_response: str | None,
        extracted_answer: Any,
        score: float | None,
        correct: bool | None,
        error_kind: ErrorKind = ErrorKind.NONE,
        error_message: str | None = None,
        latency_s: float | None = None,
    ) -> None:
        s_key = (dataset_id, dataset_revision, sample_id)
        if s_key not in self._seen_samples:
            self._seen_samples.add(s_key)
            rec = EvalSampleRecord(
                workload=self._workload,
                dataset_id=dataset_id,
                dataset_revision=dataset_revision,
                sample_id=sample_id,
                question=question,
                reference_answer=reference_answer,
            )
            self._fhs[0].write(rec.model_dump_json() + "\n")

        inp_raw = json.dumps(
            {
                "dataset_id": dataset_id,
                "dataset_revision": dataset_revision,
                "sample_id": sample_id,
                "rendered_input": rendered_input,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        inp_hash = hashlib.sha256(inp_raw.encode("utf-8")).hexdigest()[:32]
        if inp_hash not in self._seen_inputs:
            self._seen_inputs.add(inp_hash)
            inp_rec = EvalRenderedInputRecord(
                workload=self._workload,
                dataset_id=dataset_id,
                dataset_revision=dataset_revision,
                sample_id=sample_id,
                input_hash=inp_hash,
                rendered_input=rendered_input,
            )
            self._fhs[1].write(inp_rec.model_dump_json() + "\n")

        resp_rec = EvalResponseRecord(
            workload=self._workload,
            point_id=point_id,
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            sample_id=sample_id,
            input_hash=inp_hash,
            repeat=repeat,
            concurrency=concurrency,
            model_response=model_response,
            extracted_answer=extracted_answer,
            score=score,
            correct=correct,
            error_kind=error_kind,
            error_message=error_message,
            latency_s=latency_s,
        )
        self._fhs[2].write(resp_rec.model_dump_json() + "\n")
        self._written_count += 1

    def close(self) -> list[Path]:
        if not self._closed:
            for fh in self._fhs:
                fh.close()
            self._closed = True
        return list(self._paths)


def _allowed_params_only(params: dict[str, Any], allowed: frozenset[str]) -> dict:
    return {k: redact_value(v) for k, v in params.items() if k in allowed}


def write_bundle_metadata(
    bundle_dir: Path,
    *,
    job: ResolvedJob,
    result: RunResult,
    source_spec_path: Path | None = None,
    image_ref: str | None = None,
) -> None:
    """Write `spec.resolved.yaml`, `repro.sh`, `env.json`, and `result.json`."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    if source_spec_path and source_spec_path.is_file():
        raw = yaml.safe_load(source_spec_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = redact_value(raw)
        (bundle_dir / "spec.yaml").write_text(
            yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
        )

    benchmarks = [
        {
            "type": w.definition.name,
            **_allowed_params_only(w.params, w.definition.valid_param_keys()),
        }
        for w in job.workloads
        if w.definition.kind == WorkloadKind.BENCHMARK
    ]
    evals = [
        {
            "name": w.definition.name,
            **_allowed_params_only(
                w.params,
                w.definition.valid_param_keys() | {"repeats", "concurrency"},
            ),
        }
        for w in job.workloads
        if w.definition.kind == WorkloadKind.EVAL
    ]
    replayable: dict[str, Any] = {
        "schema_version": 1,
        "serve": {
            "config": redact_serve_config(job.serve_config),
            "env": redact_env(job.serve_env, allowlist_only=False),
        },
        "hardware": job.hardware.model_dump(),
        "timeout_s": job.timeout_s,
    }
    if job.spec_name:
        replayable["name"] = job.spec_name
    if benchmarks:
        replayable["benchmarks"] = benchmarks
    if evals:
        replayable["evals"] = evals
    (bundle_dir / "spec.resolved.yaml").write_text(
        yaml.safe_dump(replayable, sort_keys=False), encoding="utf-8"
    )

    (bundle_dir / "env.json").write_text(
        json.dumps(redact_env(dict(os.environ)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    repro = bundle_dir / "repro.sh"
    img_literal = shlex.quote(image_ref or "vllm/byog:UNSET")
    repro.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        f"DEFAULT_IMAGE={img_literal}\n"
        'BYOG_IMAGE="${BYOG_IMAGE:-${DEFAULT_IMAGE}}"\n'
        'OUT="${1:-./byog-repro-out}"\n'
        'if [[ "${BYOG_RUN_LOCAL:-0}" == "1" ]]; then\n'
        '  python3 -m tools.byog run --spec "${DIR}/spec.resolved.yaml" '
        '--out "${OUT}"\n'
        "else\n"
        '  docker run --rm --gpus all --ipc=host -v "${DIR}:/in:ro" '
        '-v "${OUT}:/out" "${BYOG_IMAGE}" '
        "byog run --spec /in/spec.resolved.yaml --out /out\n"
        "fi\n",
        encoding="utf-8",
    )
    os.chmod(repro, 0o755)

    (bundle_dir / "result.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def seal_bundle(bundle_dir: Path, job_id: str, attempt_id: str) -> Path:
    """Create `manifest.json` and a deterministic `bundle.tar.gz`."""
    root = bundle_dir.resolve()
    archive = root / "bundle.tar.gz"
    manifest_path = root / "manifest.json"

    blobs: list[tuple[str, bytes, int]] = []
    for cur, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        files.sort()
        cpath = Path(cur)
        for name in dirs + files:
            if (cpath / name).is_symlink():
                raise ValueError(f"Refusing to seal symlink '{name}' in bundle.")
        for fname in files:
            fpath = cpath / fname
            if fpath in (archive, manifest_path):
                continue
            rel = fpath.relative_to(root).as_posix()
            blobs.append(
                (rel, fpath.read_bytes(), 0o755 if rel == "repro.sh" else 0o644)
            )

    blobs.sort(key=lambda item: item[0])
    manifest = BundleManifest(
        job_id=job_id,
        attempt_id=attempt_id,
        files=[
            ArtifactEntry(
                path=rel,
                size_bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
            for rel, data, _ in blobs
        ],
    )
    manifest_bytes = (
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)

    all_blobs = sorted(
        [*blobs, ("manifest.json", manifest_bytes, 0o644)], key=lambda x: x[0]
    )
    with (
        archive.open("wb") as raw_fh,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_fh, mtime=0) as gz_fh,
        tarfile.open(fileobj=gz_fh, mode="w") as tar,
    ):
        for rel, data, mode in all_blobs:
            info = tarfile.TarInfo(name=rel)
            info.size = len(data)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = mode
            tar.addfile(info, io.BytesIO(data))
    return archive


def _safe_member_name(raw: str) -> str:
    if not raw or raw.startswith("/") or "\\" in raw:
        raise BundleValidationError(f"Unsafe archive member path: {raw!r}")
    stripped = raw[2:] if raw.startswith("./") else raw
    raw_parts = [p for p in stripped.split("/") if p != ""]
    if not raw_parts or any(p in (".", "..") for p in raw_parts):
        raise BundleValidationError(f"Unsafe archive member path: {raw!r}")
    return "/".join(raw_parts)


def validate_bundle(bundle_path: Path) -> tuple[BundleManifest, RunResult]:
    """Verify tar safety, SHA256 manifest, schemas, and eval JSONL integrity."""
    if not bundle_path.is_file():
        raise BundleValidationError(f"Bundle not found: '{bundle_path}'")

    files: dict[str, bytes] = {}
    total = 0
    try:
        with tarfile.open(bundle_path, mode="r:gz") as tar:
            members = tar.getmembers()
            if len(members) > MAX_TAR_MEMBERS:
                raise BundleValidationError("Archive exceeds member limit.")
            for m in members:
                if not (m.isfile() or m.isdir()):
                    raise BundleValidationError(f"Non-file member: {m.name!r}")
                if m.isdir():
                    if m.name not in (".", "./"):
                        _safe_member_name(m.name)
                    continue
                name = _safe_member_name(m.name)
                if name in files:
                    raise BundleValidationError(f"Duplicate member: {name!r}")
                fh = tar.extractfile(m)
                if fh is None:
                    raise BundleValidationError(f"Unreadable member: {name!r}")
                data = fh.read(MAX_TAR_BYTES - total + 1)
                total += len(data)
                if total > MAX_TAR_BYTES:
                    raise BundleValidationError("Archive exceeds size limit.")
                files[name] = data
    except BundleValidationError:
        raise
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise BundleValidationError(f"Corrupt tar.gz archive: {exc}") from exc

    missing = sorted(REQUIRED_FILES - set(files))
    if missing:
        raise BundleValidationError(f"Missing required files: {', '.join(missing)}")

    try:
        manifest = BundleManifest.model_validate_json(files["manifest.json"])
        result = RunResult.model_validate_json(files["result.json"])
    except (UnicodeDecodeError, ValidationError) as exc:
        raise BundleValidationError(f"Invalid manifest/result JSON: {exc}") from exc

    if (manifest.job_id, manifest.attempt_id) != (result.job_id, result.attempt_id):
        raise BundleValidationError("manifest.json IDs do not match result.json.")

    manifest_paths: set[str] = set()
    for entry in manifest.files:
        if entry.path not in files:
            raise BundleValidationError(f"Manifest file missing: {entry.path!r}")
        blob = files[entry.path]
        if len(blob) != entry.size_bytes:
            raise BundleValidationError(f"Size mismatch for {entry.path!r}.")
        if hashlib.sha256(blob).hexdigest() != entry.sha256:
            raise BundleValidationError(f"SHA256 mismatch for {entry.path!r}.")
        manifest_paths.add(entry.path)

    extras = sorted(set(files) - manifest_paths - {"manifest.json"})
    if extras:
        raise BundleValidationError(f"Unmanifested files: {', '.join(extras)}")

    eval_dirs = sorted(
        {
            str(PurePosixPath(p).parent)
            for p in files
            if p.startswith("evals/") and p.endswith(".jsonl")
        }
    )
    for d in eval_dirs:
        s_key, i_key, r_key = (
            f"{d}/samples.jsonl",
            f"{d}/inputs.jsonl",
            f"{d}/responses.jsonl",
        )
        if not all(k in files for k in (s_key, i_key, r_key)):
            raise BundleValidationError(f"Incomplete eval JSONL set in '{d}'.")
        try:
            samples = {
                (r.dataset_id, r.dataset_revision, r.sample_id)
                for ln in files[s_key].decode().splitlines()
                if ln.strip()
                for r in [EvalSampleRecord.model_validate_json(ln)]
            }
            inputs = {
                r.input_hash
                for ln in files[i_key].decode().splitlines()
                if ln.strip()
                for r in [EvalRenderedInputRecord.model_validate_json(ln)]
            }
            for ln in files[r_key].decode().splitlines():
                if not ln.strip():
                    continue
                rec = EvalResponseRecord.model_validate_json(ln)
                if (rec.dataset_id, rec.dataset_revision, rec.sample_id) not in samples:
                    raise BundleValidationError(f"Dangling sample_id: {rec.sample_id}")
                if rec.input_hash not in inputs:
                    raise BundleValidationError(
                        f"Dangling input_hash: {rec.input_hash}"
                    )
        except BundleValidationError:
            raise
        except (UnicodeDecodeError, ValidationError) as exc:
            raise BundleValidationError(f"Invalid eval JSONL in '{d}': {exc}") from exc

    return manifest, result
