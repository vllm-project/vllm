# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import vllm.envs as envs
from vllm.transformers_utils.utils import (
    is_azure,
    is_cloud_storage,
    is_gcs,
    is_s3,
    maybe_model_redirect,
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


def test_maybe_model_redirect_ignores_non_dict_json(tmp_path, monkeypatch):
    # A redirect file that is valid JSON but not an object (e.g. a list) must
    # not crash; it should be treated as "no redirect".
    redirect_file = tmp_path / "redirect.json"
    redirect_file.write_text(json.dumps(["a", "b"]))
    monkeypatch.setattr(envs, "VLLM_MODEL_REDIRECT_PATH", str(redirect_file))
    maybe_model_redirect.cache_clear()

    assert maybe_model_redirect("some/model") == "some/model"


def test_maybe_model_redirect_uses_dict_json(tmp_path, monkeypatch):
    redirect_file = tmp_path / "redirect.json"
    redirect_file.write_text(json.dumps({"some/model": "/local/model"}))
    monkeypatch.setattr(envs, "VLLM_MODEL_REDIRECT_PATH", str(redirect_file))
    maybe_model_redirect.cache_clear()

    assert maybe_model_redirect("some/model") == "/local/model"
