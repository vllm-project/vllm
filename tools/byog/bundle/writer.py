# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C5: Bundle metadata writers, secret redaction, repro.sh, and deterministic tar."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shlex
import tarfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

from tools.byog.schemas.result import ArtifactEntry, BundleManifest, RunResult
from tools.byog.schemas.spec import ResolvedJob

_ALLOWED_ENV_PREFIXES: tuple[str, ...] = (
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

_SECRET_SUBSTRINGS: tuple[str, ...] = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AUTH",
    "API_KEY",
    "API-KEY",
    "SSL_KEYFILE",
    "SSL-KEYFILE",
)
_SECRET_TOKENS: frozenset[str] = frozenset({"KEY", "PAT"})


class BundleWriteError(RuntimeError):
    """Raised when bundle staging or sealing fails."""


def _is_secret_key(name: str) -> bool:
    upper = name.upper()
    if any(sub in upper for sub in _SECRET_SUBSTRINGS):
        return True
    normalized = upper.replace("-", "_")
    tokens = {tok for tok in normalized.split("_") if tok}
    return bool(tokens & _SECRET_TOKENS)


def _strip_url_credentials(value: str) -> str:
    if "://" not in value or "@" not in value:
        return value
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    if parts.username is None and parts.password is None:
        return value
    host = parts.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"***@{host}"
    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def redact_env(env: dict[str, str], *, allowlist_only: bool = True) -> dict[str, str]:
    """Allowlist-filter and redact environment variables before serialization."""
    out: dict[str, str] = {}
    for key in sorted(env):
        upper = key.upper()
        if allowlist_only and not any(
            upper.startswith(prefix) for prefix in _ALLOWED_ENV_PREFIXES
        ):
            continue
        if _is_secret_key(key):
            out[key] = "<REDACTED>"
        else:
            out[key] = _strip_url_credentials(str(env[key]))
    return out


def redact_serve_config(config: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact secret-bearing keys and URL credentials in serve_config."""
    redacted: dict[str, Any] = {}
    for key, val in config.items():
        if _is_secret_key(key):
            redacted[key] = "<REDACTED>"
        elif isinstance(val, str):
            redacted[key] = _strip_url_credentials(val)
        elif isinstance(val, dict):
            redacted[key] = redact_serve_config(val)
        elif isinstance(val, list):
            redacted[key] = [
                _strip_url_credentials(item)
                if isinstance(item, str)
                else (redact_serve_config(item) if isinstance(item, dict) else item)
                for item in val
            ]
        else:
            redacted[key] = val
    return redacted


def write_redacted_spec(source_spec_path: Path, out_path: Path) -> Path:
    """Copy a user spec into the bundle with secrets redacted."""
    raw = yaml.safe_load(source_spec_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        serve = raw.get("serve")
        if isinstance(serve, dict):
            if isinstance(serve.get("config"), dict):
                serve["config"] = redact_serve_config(serve["config"])
            if isinstance(serve.get("env"), dict):
                serve["env"] = {
                    k: "<REDACTED>" if _is_secret_key(k) else _strip_url_credentials(v)
                    for k, v in serve["env"].items()
                }
        out_path.write_text(
            yaml.safe_dump(raw, sort_keys=False),
            encoding="utf-8",
        )
    else:
        out_path.write_text("# empty spec\n", encoding="utf-8")
    return out_path


def write_resolved_spec(job: ResolvedJob, out_path: Path) -> Path:
    """Emit `spec.resolved.yaml` as a directly replayable `RunSpec`."""
    benchmarks: list[dict[str, Any]] = []
    evals: list[dict[str, Any]] = []
    for w in job.workloads:
        if w.kind == "benchmark":
            entry: dict[str, Any] = {"type": w.name}
            entry.update(w.params)
            benchmarks.append(entry)
        else:
            entry = {"name": w.name}
            entry.update(w.params)
            acc_thresholds = [t for t in w.quality_thresholds if t.metric == "accuracy"]
            if acc_thresholds and acc_thresholds[0].min_value is not None:
                entry["min_accuracy"] = acc_thresholds[0].min_value
            evals.append(entry)

    replayable: dict[str, Any] = {
        "schema_version": 1,
        "serve": {
            "config": redact_serve_config(job.serve_config),
            "env": redact_env(job.serve_env, allowlist_only=False),
        },
        "hardware": job.hardware.model_dump(),
        "timeout_s": job.timeout_s,
    }
    if job.spec_name is not None:
        replayable["name"] = job.spec_name
    if benchmarks:
        replayable["benchmarks"] = benchmarks
    if evals:
        replayable["evals"] = evals

    out_path.write_text(
        yaml.safe_dump(replayable, sort_keys=False),
        encoding="utf-8",
    )
    return out_path


def write_repro_sh(
    job: ResolvedJob,
    out_path: Path,
    *,
    image_ref: str | None = None,
    vllm_commit: str | None = None,
) -> Path:
    """Generate a self-contained reproduction script referencing spec.resolved.yaml."""
    resolved_image = image_ref or "vllm/byog:UNSET"
    commit_line = f"# vllm_commit: {vllm_commit}\n" if vllm_commit else ""
    script = (
        "#!/usr/bin/env bash\n"
        "# Auto-generated BYOG reproduction script.\n"
        f"# job_id: {job.job_id}\n"
        f"# attempt_id: {job.attempt_id}\n"
        f"{commit_line}"
        "set -euo pipefail\n\n"
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        f'BYOG_IMAGE="${{BYOG_IMAGE:-{shlex.quote(resolved_image)}}}"\n'
        'OUT_DIR="${1:-./byog-repro-out}"\n\n'
        'if [[ "${BYOG_RUN_LOCAL:-0}" == "1" ]]; then\n'
        '  python3 -m tools.byog run --spec "${SCRIPT_DIR}/spec.resolved.yaml" '
        '--out "${OUT_DIR}"\n'
        "else\n"
        "  docker run --rm --gpus all \\\n"
        '    -v "${SCRIPT_DIR}:/byog-in:ro" \\\n'
        '    -v "${OUT_DIR}:/byog-out" \\\n'
        '    "${BYOG_IMAGE}" \\\n'
        "    byog run --spec /byog-in/spec.resolved.yaml --out /byog-out\n"
        "fi\n"
    )
    out_path.write_text(script, encoding="utf-8")
    os.chmod(out_path, 0o755)
    return out_path


def write_bundle_metadata(
    bundle_dir: Path,
    *,
    job: ResolvedJob,
    result: RunResult,
    source_spec_path: Path | None = None,
    image_ref: str | None = None,
    host_env: dict[str, str] | None = None,
) -> None:
    """Stage all required top-level metadata files into `bundle_dir`."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    if source_spec_path is not None and source_spec_path.is_file():
        write_redacted_spec(source_spec_path, bundle_dir / "spec.yaml")
    write_resolved_spec(job, bundle_dir / "spec.resolved.yaml")

    redacted_job = job.model_copy(
        update={
            "serve_config": redact_serve_config(job.serve_config),
            "serve_env": redact_env(job.serve_env, allowlist_only=False),
        }
    )
    (bundle_dir / "resolved_job.json").write_text(
        json.dumps(
            redacted_job.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    env_snapshot = redact_env(
        host_env if host_env is not None else dict(os.environ),
        allowlist_only=True,
    )
    (bundle_dir / "env.json").write_text(
        json.dumps(env_snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    write_repro_sh(
        job,
        bundle_dir / "repro.sh",
        image_ref=image_ref,
        vllm_commit=result.vllm_commit,
    )
    (bundle_dir / "result.json").write_text(
        json.dumps(
            result.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def seal_bundle(bundle_dir: Path, job_id: str, attempt_id: str) -> Path:
    """Generate `manifest.json` and a deterministic `bundle.tar.gz`."""
    bundle_dir = bundle_dir.resolve()
    archive_path = bundle_dir / "bundle.tar.gz"
    manifest_path = bundle_dir / "manifest.json"

    file_blobs: list[tuple[str, bytes, int]] = []
    for root, dirs, files in os.walk(bundle_dir, followlinks=False):
        dirs.sort()
        files.sort()
        root_path = Path(root)
        for dname in dirs:
            if (root_path / dname).is_symlink():
                raise BundleWriteError(
                    f"Refusing to seal symlink directory inside bundle: {dname}"
                )
        for fname in files:
            fpath = root_path / fname
            if fpath.is_symlink():
                raise BundleWriteError(
                    f"Refusing to seal symlink inside bundle: {fpath.name}"
                )
            if fpath in (archive_path, manifest_path):
                continue
            rel = fpath.relative_to(bundle_dir).as_posix()
            blob = fpath.read_bytes()
            mode = 0o755 if rel == "repro.sh" else 0o644
            file_blobs.append((rel, blob, mode))

    file_blobs.sort(key=lambda item: item[0])
    entries = [
        ArtifactEntry(
            path=rel,
            size_bytes=len(blob),
            sha256=hashlib.sha256(blob).hexdigest(),
        )
        for rel, blob, _ in file_blobs
    ]
    manifest = BundleManifest(
        job_id=job_id,
        attempt_id=attempt_id,
        files=entries,
    )
    manifest_bytes = (
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)

    all_blobs = sorted(
        [*file_blobs, ("manifest.json", manifest_bytes, 0o644)],
        key=lambda item: item[0],
    )
    with (
        archive_path.open("wb") as raw_fh,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_fh, mtime=0) as gz_fh,
        tarfile.open(fileobj=gz_fh, mode="w") as tar,
    ):
        for rel, blob, mode in all_blobs:
            info = tarfile.TarInfo(name=rel)
            info.size = len(blob)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = mode
            tar.addfile(info, io.BytesIO(blob))

    return archive_path
