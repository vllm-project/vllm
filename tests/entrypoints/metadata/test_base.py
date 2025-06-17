# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from http import HTTPStatus

import pytest
import requests

from vllm.entrypoints.metadata.generate import GenerateBrief

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

MODEL_NAME = "Qwen/Qwen3-0.6B"
task = "generate"
max_model_len = 1024
enable_prefix_caching = False
architectures = ["Qwen3ForCausalLM"]

expected_brief = GenerateBrief(task=task,
                               served_model_name=MODEL_NAME,
                               architectures=architectures,
                               max_model_len=max_model_len,
                               enable_prefix_caching=enable_prefix_caching)


@pytest.fixture(scope="module")
def server():
    from tests.utils import RemoteOpenAIServer
    args = [
        "--task", "generate", "--max-model-len", f"{max_model_len}",
        "--enforce-eager", "--disable-uvicorn-access-log"
    ]

    if enable_prefix_caching:
        args.append("--enable-prefix-caching")
    else:
        args.append("--no-enable-prefix-caching")

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


def test_brief_metadata_only(server):
    # default brief metadata only,
    # unless --disable-brief-metadata-only is set

    url = server.url_for("metadata/brief")
    response = requests.get(url)
    assert response.json() == expected_brief.model_dump()

    url = server.url_for("metadata/detail")
    response = requests.get(url)
    assert response.status_code == HTTPStatus.NOT_FOUND

    url = server.url_for("metadata/hf_config")
    response = requests.get(url)
    assert response.status_code == HTTPStatus.NOT_FOUND


def test_quick_access(server):
    url = server.url_for("metadata/brief")

    for k, v in expected_brief.model_dump().items():
        response = requests.get(url + f"/{k}")
        assert response.json() == {k: v}

    response = requests.get(url + "/foo")
    assert response.status_code == HTTPStatus.NOT_FOUND
