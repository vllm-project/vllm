# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from vllm.transformers_utils.utils import (
    convert_model_repo_to_path,
    is_azure,
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


def test_convert_model_repo_to_path_without_modelscope(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("VLLM_USE_MODELSCOPE", raising=False)

    assert convert_model_repo_to_path("org/model") == "org/model"


def test_convert_model_repo_to_path_preserves_local_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", "true")

    assert convert_model_repo_to_path(str(tmp_path)) == str(tmp_path)


def test_convert_model_repo_to_path_uses_modelscope_snapshot(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", "true")
    snapshot_path = "/cache/modelscope/models/org--model/snapshots/master"
    snapshot_download = Mock(return_value=snapshot_path)
    modelscope = ModuleType("modelscope")
    modelscope.__path__ = []
    modelscope_hub = ModuleType("modelscope.hub")
    modelscope_hub.__path__ = []
    snapshot_download_module = ModuleType("modelscope.hub.snapshot_download")
    snapshot_download_module.__dict__["snapshot_download"] = snapshot_download
    monkeypatch.setitem(sys.modules, "modelscope", modelscope)
    monkeypatch.setitem(sys.modules, "modelscope.hub", modelscope_hub)
    monkeypatch.setitem(
        sys.modules, "modelscope.hub.snapshot_download", snapshot_download_module
    )

    assert convert_model_repo_to_path("org/model", revision="v1") == snapshot_path
    snapshot_download.assert_called_once_with(
        model_id="org/model",
        revision="v1",
        local_files_only=True,
    )
