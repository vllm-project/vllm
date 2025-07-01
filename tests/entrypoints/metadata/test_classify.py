# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest
import requests

from vllm.entrypoints.metadata.base import PoolerConfigMetadata
from vllm.entrypoints.metadata.classify import ClassifyBrief, ClassifyMetadata

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

MODEL_NAME = "jason9693/Qwen2.5-1.5B-apeach"

expected_brief = ClassifyBrief(
    task="classify",
    served_model_name=MODEL_NAME,
    architectures=["Qwen2ForSequenceClassification"],
    max_model_len=131072,
    num_labels=2)

expected_pooler_config = PoolerConfigMetadata(pooling_type=None,
                                              normalize=None,
                                              softmax=None,
                                              step_tag_id=None,
                                              returned_token_ids=None)


def test_classify_offline_metadata(vllm_runner):
    with vllm_runner(MODEL_NAME, task="classify",
                     max_model_len=None) as vllm_model:
        metadata: ClassifyMetadata = vllm_model.model.metadata

        assert isinstance(metadata, ClassifyMetadata)

        assert metadata.brief == expected_brief

        assert metadata.hf_config["architectures"][
            0] == "Qwen2ForSequenceClassification"
        assert metadata.pooler_config == expected_pooler_config


@pytest.fixture(scope="module")
def server():
    from tests.utils import RemoteOpenAIServer
    args = [
        "--task", "classify", "--enforce-eager",
        "--disable-uvicorn-access-log", "--disable-brief-metadata-only"
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


def test_classify_online_metadata(server):
    url = server.url_for("metadata/brief")
    response = requests.get(url)
    assert response.json() == expected_brief.model_dump()

    url = server.url_for("metadata/hf_config")
    response = requests.get(url)
    assert response.json()

    url = server.url_for("metadata/pooler_config")
    response = requests.get(url)
    assert response.json() == vars(expected_pooler_config)
