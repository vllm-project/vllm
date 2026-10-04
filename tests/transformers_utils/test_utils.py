# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm.transformers_utils.utils import (
    convert_model_repo_to_path,
    is_azure,
    is_cloud_storage,
    is_gcs,
    is_s3,
)


@pytest.fixture
def modelscope_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", "1")
    legacy_root = tmp_path / "hub" / "models"
    file_utils = ModuleType("modelscope.utils.file_utils")
    monkeypatch.setattr(
        file_utils, "get_model_cache_root", lambda: str(legacy_root), raising=False
    )
    hub = ModuleType("modelscope_hub")
    monkeypatch.setattr(
        hub,
        "get_default_config",
        lambda: SimpleNamespace(cache_dir=tmp_path),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "modelscope", ModuleType("modelscope"))
    monkeypatch.setitem(sys.modules, "modelscope.utils", ModuleType("modelscope.utils"))
    monkeypatch.setitem(sys.modules, "modelscope.utils.file_utils", file_utils)
    monkeypatch.setitem(sys.modules, "modelscope_hub", hub)
    return tmp_path


@pytest.mark.parametrize("revision", [None, "release/v1"])
def test_modelscope_resolves_requested_snapshot(modelscope_cache, revision):
    """Resolve the requested revision even when another snapshot is cached."""
    snapshots = modelscope_cache / "models" / "org--model" / "snapshots"
    (snapshots / "unrelated").mkdir(parents=True)
    expected = snapshots / (revision or "master")
    expected.mkdir(parents=True)
    assert convert_model_repo_to_path("org/model", revision) == str(expected)


def test_modelscope_prefers_existing_legacy_cache(modelscope_cache):
    legacy = modelscope_cache / "hub" / "models" / "org" / "model"
    legacy.mkdir(parents=True)
    (modelscope_cache / "models" / "org--model" / "snapshots" / "master").mkdir(
        parents=True
    )
    assert convert_model_repo_to_path("org/model") == str(legacy)


def test_modelscope_missing_cache_preserves_legacy_path(modelscope_cache):
    """Missing snapshots must not trigger downloads or select another revision."""
    (modelscope_cache / "models" / "org--model" / "snapshots" / "other").mkdir(
        parents=True
    )
    expected = modelscope_cache / "hub" / "models" / "org" / "model"
    assert convert_model_repo_to_path("org/model", "missing") == str(expected)
    assert not expected.exists()


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
