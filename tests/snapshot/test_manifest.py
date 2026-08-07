# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import os
import stat
from pathlib import Path

import pytest

from vllm_cli.snapshot.manifest import (
    SnapshotCompatibilityError,
    SnapshotManifest,
    SnapshotSecurityError,
    inspect_snapshot,
    read_manifest,
    validate_artifact_root,
    validate_identity,
    write_manifest_atomic,
)


def make_manifest(**changes: object) -> SnapshotManifest:
    manifest = SnapshotManifest(
        schema_version=1,
        boundary="post-engine-init-pre-http-bind",
        complete=True,
        created_at="2026-08-06T00:00:00Z",
        artifact_bytes=1234,
        source_revision="source-sha",
        binary_revision="binary-sha",
        python_version="3.12.3",
        torch_version="2.9.0",
        cuda_runtime="12.9",
        driver_version="575.57.08",
        criu_version="4.1",
        cuda_checkpoint_version="575.57.08",
        kernel_release="6.8.0",
        host_id="host-a",
        gpu_name="NVIDIA A10",
        gpu_uuid="GPU-abc",
        model="Qwen/Qwen3-0.6B",
        model_revision="model-sha",
        tokenizer_revision="tokenizer-sha",
        engine_args=(("tensor_parallel_size", 1), ("dtype", "bfloat16")),
        environment=(("VLLM_USE_V1", "1"),),
        process_tree=(100, 101),
        cuda_holders=(101,),
        oracle_token_ids=(12095,),
        oracle_text=" Paris",
    )
    return dataclasses.replace(manifest, **changes)


def test_manifest_round_trip_is_private(tmp_path: Path):
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    manifest = make_manifest()

    write_manifest_atomic(artifact, manifest)

    assert read_manifest(artifact) == manifest
    assert stat.S_IMODE((artifact / "manifest.json").stat().st_mode) == 0o600


def test_identity_mismatch_names_field():
    expected = make_manifest()
    actual = dataclasses.replace(expected, model_revision="different-model-sha")

    with pytest.raises(SnapshotCompatibilityError, match="model_revision"):
        validate_identity(expected, actual)


def test_recording_metadata_does_not_change_identity():
    expected = make_manifest()
    actual = dataclasses.replace(
        expected,
        created_at="2026-08-07T00:00:00Z",
        artifact_bytes=9999,
    )

    validate_identity(expected, actual)


def test_symlink_artifact_is_rejected(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(SnapshotSecurityError, match="symlink"):
        validate_artifact_root(linked, creating=False)


def test_world_writable_parent_is_rejected(tmp_path: Path):
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir()
    unsafe.chmod(0o777)

    with pytest.raises(SnapshotSecurityError, match="world-writable"):
        validate_artifact_root(unsafe / "snapshot", creating=True)


def test_manifest_writer_refuses_existing_symlink(tmp_path: Path):
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    target = tmp_path / "target.json"
    target.write_text("not a manifest")
    (artifact / "manifest.json").symlink_to(target)

    with pytest.raises(SnapshotSecurityError, match="already exists"):
        write_manifest_atomic(artifact, make_manifest())


def test_inspect_is_json_safe(tmp_path: Path):
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    write_manifest_atomic(artifact, make_manifest())

    inspected = inspect_snapshot(artifact)

    assert inspected["complete"] is True
    assert inspected["support_boundary"] == "same-host Linux x86_64 TP1"
    assert inspected["oracle_token_ids"] == [12095]


def test_manifest_temp_file_is_not_left_behind(tmp_path: Path):
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)

    write_manifest_atomic(artifact, make_manifest())

    assert not (artifact / "manifest.json.tmp").exists()
    assert os.listdir(artifact) == ["manifest.json"]
