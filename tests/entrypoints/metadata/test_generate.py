# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest
import requests

from vllm.entrypoints.metadata.generate import (GenerateBrief, GenerateDetail,
                                                GenerateMetadata)

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

expected_detail = GenerateDetail(task=task,
                                 served_model_name=MODEL_NAME,
                                 architectures=architectures,
                                 max_model_len=max_model_len,
                                 enable_prefix_caching=enable_prefix_caching)


def test_generate_offline_metadata(vllm_runner):
    with vllm_runner(
            MODEL_NAME,
            max_model_len=max_model_len,
            task=task,
            enable_prefix_caching=enable_prefix_caching) as vllm_model:
        metadata: GenerateMetadata = vllm_model.model.metadata

        assert isinstance(metadata, GenerateMetadata)

        assert metadata.brief == expected_brief
        assert metadata.detail == expected_detail

        assert metadata.hf_config["architectures"] == architectures


@pytest.fixture(scope="module")
def server():
    from tests.utils import RemoteOpenAIServer
    args = [
        "--task", "generate", "--max-model-len", f"{max_model_len}",
        "--enforce-eager", "--disable-uvicorn-access-log",
        "--disable-brief-metadata-only"
    ]

    if enable_prefix_caching:
        args.append("--enable-prefix-caching")
    else:
        args.append("--no-enable-prefix-caching")

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


def test_generate_online_metadata(server):
    url = server.url_for("metadata/brief")
    response = requests.get(url)
    assert response.json() == expected_brief.model_dump()

    url = server.url_for("metadata/detail")
    response = requests.get(url)
    assert response.json() == expected_detail.model_dump()

    url = server.url_for("metadata/hf_config")
    response = requests.get(url)
    assert response.json()
