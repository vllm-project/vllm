# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import struct
from pathlib import Path

import pytest

from vllm.transformers_utils.utils import (
    is_azure,
    is_cloud_storage,
    is_gcs,
    is_s3,
    parse_safetensors_file_metadata,
)

pytestmark = pytest.mark.skip_global_cleanup


def test_is_gcs():
    assert is_gcs("gs://model-path")
    assert not is_gcs("s3://model-path/path-to-model")
    assert not is_gcs("/unix/local/path")
    assert not is_gcs("nfs://nfs-fqdn.local")


def test_is_s3():
    assert is_s3("s3://model-path/path-to-model")
    assert not is_s3("gs://model-path")
    assert not is_s3("/unix/local/path")
    assert not is_s3("nfs://nfs-fqdn.local")


def test_is_azure():
    assert is_azure("az://model-container/path")
    assert not is_azure("s3://model-path/path-to-model")
    assert not is_azure("/unix/local/path")
    assert not is_azure("nfs://nfs-fqdn.local")


def test_is_cloud_storage():
    assert is_cloud_storage("gs://model-path")
    assert is_cloud_storage("s3://model-path/path-to-model")
    assert is_cloud_storage("az://model-container/path")
    assert not is_cloud_storage("/unix/local/path")
    assert not is_cloud_storage("nfs://nfs-fqdn.local")


@pytest.mark.parametrize("include_metadata", [False, True])
def test_parse_safetensors_metadata_preserves_header(tmp_path: Path, include_metadata):
    metadata = {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    if include_metadata:
        metadata["__metadata__"] = {"format": "pt", "description": "checkpoint"}
    header = json.dumps(metadata).encode("utf-8")
    header += b" " * (-len(header) % 8)
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0))

    assert parse_safetensors_file_metadata(path) == metadata


@pytest.mark.parametrize(
    "contents",
    [
        b"",
        b"\x00" * 7,
        struct.pack("<Q", 1024) + b"{}",
        struct.pack("<Q", 2**64 - 1) + b"{}",
    ],
    ids=["empty", "short-length", "truncated-header", "unreadable-length"],
)
def test_parse_safetensors_metadata_rejects_invalid_header(tmp_path: Path, contents):
    """Even a length too large for read() must produce a checkpoint error."""
    path = tmp_path / "model.safetensors"
    path.write_bytes(contents)

    with pytest.raises(ValueError, match="header") as exc_info:
        parse_safetensors_file_metadata(path)

    assert str(path) in str(exc_info.value)
    assert "incomplete" in str(exc_info.value)


def test_parse_safetensors_metadata_rejects_lfs_pointer(tmp_path: Path):
    path = tmp_path / "model.safetensors"
    path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{'0' * 64}\n"
        "size 1024\n"
    )

    with pytest.raises(ValueError, match="Git LFS") as exc_info:
        parse_safetensors_file_metadata(str(path))

    assert str(path) in str(exc_info.value)
    assert "download" in str(exc_info.value).lower()
