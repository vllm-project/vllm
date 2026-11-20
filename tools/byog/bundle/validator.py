# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C5: Hardened shared bundle validator for bundle.tar.gz."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from tools.byog.schemas.result import (
    BundleManifest,
    EvalRenderedInputRecord,
    EvalResponseRecord,
    EvalSampleRecord,
    RunResult,
)

MAX_MEMBERS = 10_000
MAX_TOTAL_BYTES = 512 * 1024 * 1024  # 512 MiB
REQUIRED_BUNDLE_FILES = frozenset(
    {"manifest.json", "result.json", "spec.resolved.yaml", "repro.sh"}
)


class BundleValidationError(ValueError):
    """Raised when a bundle fails security, schema, or integrity validation."""


def _validate_and_normalize_member_name(raw_name: str) -> str:
    """Reject traversal, absolute paths, and backslashes BEFORE normalization."""
    if not raw_name:
        raise BundleValidationError("Archive contains an empty member name.")
    if raw_name.startswith("/"):
        raise BundleValidationError(
            f"Archive contains absolute path member: {raw_name!r}"
        )
    if "\\" in raw_name:
        raise BundleValidationError(
            f"Archive contains backslash in member path: {raw_name!r}"
        )
    raw_parts = PurePosixPath(raw_name).parts
    if ".." in raw_parts:
        raise BundleValidationError(
            f"Archive contains path traversal member: {raw_name!r}"
        )
    normalized = raw_name[2:] if raw_name.startswith("./") else raw_name
    if (
        not normalized
        or normalized.startswith("/")
        or ".." in PurePosixPath(normalized).parts
    ):
        raise BundleValidationError(
            f"Archive contains invalid normalized member path: {raw_name!r}"
        )
    return normalized


def _validate_eval_dir(
    workload_dir: str,
    files: dict[str, bytes],
) -> None:
    samples_key = f"{workload_dir}/samples.jsonl"
    inputs_key = f"{workload_dir}/inputs.jsonl"
    responses_key = f"{workload_dir}/responses.jsonl"
    present = [k for k in (samples_key, inputs_key, responses_key) if k in files]
    if not present:
        return
    if len(present) != 3:
        raise BundleValidationError(
            f"Eval directory '{workload_dir}' must contain all three of "
            "samples.jsonl, inputs.jsonl, and responses.jsonl."
        )

    sample_keys: set[tuple[str, str, str]] = set()
    for line_no, line in enumerate(
        files[samples_key].decode("utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rec = EvalSampleRecord.model_validate_json(line)
        except ValidationError as exc:
            raise BundleValidationError(
                f"Invalid EvalSampleRecord at {samples_key}:{line_no}: {exc}"
            ) from exc
        sample_keys.add((rec.dataset_id, rec.dataset_revision, rec.sample_id))

    input_hashes: set[str] = set()
    for line_no, line in enumerate(
        files[inputs_key].decode("utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            inp_rec = EvalRenderedInputRecord.model_validate_json(line)
        except ValidationError as exc:
            raise BundleValidationError(
                f"Invalid EvalRenderedInputRecord at {inputs_key}:{line_no}: {exc}"
            ) from exc
        if (
            inp_rec.dataset_id,
            inp_rec.dataset_revision,
            inp_rec.sample_id,
        ) not in sample_keys:
            raise BundleValidationError(
                f"Dangling sample reference in {inputs_key}:{line_no}: "
                f"{inp_rec.sample_id}"
            )
        input_hashes.add(inp_rec.input_hash)

    for line_no, line in enumerate(
        files[responses_key].decode("utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            resp_rec = EvalResponseRecord.model_validate_json(line)
        except ValidationError as exc:
            raise BundleValidationError(
                f"Invalid EvalResponseRecord at {responses_key}:{line_no}: {exc}"
            ) from exc
        if (
            resp_rec.dataset_id,
            resp_rec.dataset_revision,
            resp_rec.sample_id,
        ) not in sample_keys:
            raise BundleValidationError(
                f"Dangling sample reference in {responses_key}:{line_no}: "
                f"{resp_rec.sample_id}"
            )
        if resp_rec.input_hash not in input_hashes:
            raise BundleValidationError(
                f"Dangling input_hash in {responses_key}:{line_no}: "
                f"{resp_rec.input_hash}"
            )


def validate_bundle(bundle_path: Path) -> tuple[BundleManifest, RunResult]:
    """Validate archive safety, SHA256 manifest, schemas, and eval integrity."""
    if not bundle_path.is_file():
        raise BundleValidationError(f"Bundle archive not found: '{bundle_path}'")

    files: dict[str, bytes] = {}
    total_bytes = 0

    try:
        with tarfile.open(bundle_path, mode="r:gz") as tar:
            members = tar.getmembers()
            if len(members) > MAX_MEMBERS:
                raise BundleValidationError(
                    f"Archive exceeds maximum member limit ({MAX_MEMBERS})."
                )
            for member in members:
                if not (member.isfile() or member.isdir()):
                    raise BundleValidationError(
                        f"Archive contains unsupported non-file entry "
                        f"{member.name!r} (type={member.type!r})."
                    )
                norm_name = _validate_and_normalize_member_name(member.name)
                if member.isdir():
                    continue
                if norm_name in files:
                    raise BundleValidationError(
                        f"Archive contains duplicate member path: {norm_name!r}"
                    )
                if member.size < 0 or total_bytes + member.size > MAX_TOTAL_BYTES:
                    raise BundleValidationError(
                        f"Archive exceeds maximum uncompressed size "
                        f"({MAX_TOTAL_BYTES} bytes)."
                    )
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise BundleValidationError(
                        f"Failed to read member {norm_name!r} from archive."
                    )
                data = extracted.read(MAX_TOTAL_BYTES - total_bytes + 1)
                total_bytes += len(data)
                if total_bytes > MAX_TOTAL_BYTES:
                    raise BundleValidationError(
                        f"Archive exceeds maximum uncompressed size "
                        f"({MAX_TOTAL_BYTES} bytes)."
                    )
                files[norm_name] = data
    except BundleValidationError:
        raise
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise BundleValidationError(
            f"Failed to read tar.gz archive '{bundle_path}': {exc}"
        ) from exc

    missing_required = sorted(REQUIRED_BUNDLE_FILES - set(files))
    if missing_required:
        raise BundleValidationError(
            f"Bundle is missing required file(s): {', '.join(missing_required)}"
        )

    try:
        manifest = BundleManifest.model_validate(
            json.loads(files["manifest.json"].decode("utf-8"))
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise BundleValidationError(f"Invalid manifest.json: {exc}") from exc

    try:
        result = RunResult.model_validate(
            json.loads(files["result.json"].decode("utf-8"))
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise BundleValidationError(f"Invalid result.json: {exc}") from exc

    if manifest.job_id != result.job_id or manifest.attempt_id != result.attempt_id:
        raise BundleValidationError(
            "manifest.json job_id/attempt_id does not match result.json."
        )

    manifest_paths: set[str] = set()
    for entry in manifest.files:
        if entry.path in manifest_paths:
            raise BundleValidationError(
                f"Duplicate path in manifest.json: {entry.path!r}"
            )
        manifest_paths.add(entry.path)
        if entry.path not in files:
            raise BundleValidationError(
                f"Manifest references missing file: {entry.path!r}"
            )
        blob = files[entry.path]
        if len(blob) != entry.size_bytes:
            raise BundleValidationError(
                f"Size mismatch for {entry.path!r}: manifest={entry.size_bytes}, "
                f"actual={len(blob)}"
            )
        digest = hashlib.sha256(blob).hexdigest()
        if digest != entry.sha256:
            raise BundleValidationError(
                f"SHA256 mismatch for {entry.path!r}: manifest={entry.sha256}, "
                f"actual={digest}"
            )

    unmanifested = sorted(set(files) - manifest_paths - {"manifest.json"})
    if unmanifested:
        raise BundleValidationError(
            f"Archive contains unmanifested file(s): {', '.join(unmanifested)}"
        )

    # Validate every eval record directory.
    eval_dirs = sorted(
        {
            str(PurePosixPath(p).parent)
            for p in files
            if p.startswith("evals/") and p.endswith(".jsonl")
        }
    )
    for eval_dir in eval_dirs:
        _validate_eval_dir(eval_dir, files)

    return manifest, result
