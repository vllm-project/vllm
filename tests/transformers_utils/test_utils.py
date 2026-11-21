# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import struct

import pytest

from vllm.transformers_utils.utils import (
    is_azure,
    is_cloud_storage,
    is_gcs,
    is_s3,
    parse_safetensors_file_metadata,
)


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


def test_parse_safetensors_file_metadata(tmp_path):
    path = tmp_path / "model.safetensors"
    payload = b'{"weight": {"dtype": "F16"}}'
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)

    assert parse_safetensors_file_metadata(path) == {"weight": {"dtype": "F16"}}


def test_parse_safetensors_file_metadata_rejects_bogus_length(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", 2**63) + b"{}")

    with pytest.raises(ValueError, match="exceeds the 10-byte file"):
        parse_safetensors_file_metadata(path)
