# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.transformers_utils.utils import (
    is_cloud_storage,
    is_gcs,
    is_s3,
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


def test_is_cloud_storage():
    assert is_cloud_storage("gs://model-path")
    assert is_cloud_storage("s3://model-path/path-to-model")
    assert not is_cloud_storage("/unix/local/path")
    assert not is_cloud_storage("nfs://nfs-fqdn.local")
