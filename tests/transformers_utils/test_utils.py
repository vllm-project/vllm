# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from types import ModuleType

import pytest

from vllm.transformers_utils.utils import (
    is_azure,
    is_cloud_storage,
    is_gcs,
    is_s3,
    modelscope_list_repo_files,
)


class _FakeModelScopeApiModule(ModuleType):
    HubApi: type[object]


def _install_fake_modelscope_hub_api(
    monkeypatch: pytest.MonkeyPatch,
    hub_api: type[object],
) -> None:
    api_module = _FakeModelScopeApiModule("modelscope.hub.api")
    api_module.HubApi = hub_api
    monkeypatch.setitem(sys.modules, "modelscope", ModuleType("modelscope"))
    monkeypatch.setitem(sys.modules, "modelscope.hub", ModuleType("modelscope.hub"))
    monkeypatch.setitem(sys.modules, "modelscope.hub.api", api_module)


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


def test_modelscope_list_repo_files_old_api(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[object, ...]] = []

    class HubApi:
        def login(self, token: str | bool | None) -> None:
            calls.append(("login", token))

        def get_model_files(
            self,
            model_id: str,
            revision: str | None = None,
            recursive: bool = False,
        ) -> list[dict[str, object]]:
            calls.append(("get_model_files", model_id, revision, recursive))
            return [
                {"Path": "config.json", "Type": "blob"},
                {"Path": "nested", "Type": "tree"},
                {"Path": "model.safetensors", "Type": "blob"},
            ]

    _install_fake_modelscope_hub_api(monkeypatch, HubApi)

    files = modelscope_list_repo_files(
        "qwen/Qwen1.5-0.5B-Chat",
        revision="master",
        token="fake-token",
    )

    assert files == ["config.json", "model.safetensors"]
    assert calls == [
        ("login", "fake-token"),
        ("get_model_files", "qwen/Qwen1.5-0.5B-Chat", "master", True),
    ]


def test_modelscope_list_repo_files_new_api(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[object, ...]] = []

    class HubApi:
        def login(self, token: str | bool | None) -> None:
            calls.append(("login", token))

        def get_model_files(
            self,
            model_id: str,
            recursive: bool = True,
        ) -> list[dict[str, object]]:
            calls.append(("get_model_files", model_id, recursive))
            return [
                {"Path": ".gitattributes", "Size": 1519},
                {"Path": "config.json", "Size": 661},
                {"Path": "nested", "Type": "tree"},
            ]

    _install_fake_modelscope_hub_api(monkeypatch, HubApi)

    files = modelscope_list_repo_files("qwen/Qwen1.5-0.5B-Chat")

    assert files == [".gitattributes", "config.json"]
    assert calls == [
        ("login", None),
        ("get_model_files", "qwen/Qwen1.5-0.5B-Chat", True),
    ]


def test_modelscope_list_repo_files_new_api_rejects_revision(
    monkeypatch: pytest.MonkeyPatch,
):
    class HubApi:
        def login(self, token: str | bool | None) -> None:
            pass

        def get_model_files(
            self,
            model_id: str,
            recursive: bool = True,
        ) -> list[dict[str, object]]:
            return []

    _install_fake_modelscope_hub_api(monkeypatch, HubApi)

    with pytest.raises(ValueError, match="does not support listing files"):
        modelscope_list_repo_files(
            "qwen/Qwen1.5-0.5B-Chat",
            revision="master",
        )
