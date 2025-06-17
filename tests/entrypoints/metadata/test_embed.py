# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest
import requests

from vllm.config import PoolerConfig
from vllm.entrypoints.metadata.embed import (EmbedBrief, EmbedDetail,
                                             EmbedMetadata)

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

MODEL_NAME = "intfloat/e5-small"

expected_brief = EmbedBrief(
    task="embed",
    served_model_name=MODEL_NAME,
    architectures=["BertModel"],
    embedding_dim=384,
    max_model_len=512,
    is_matryoshka=False,
    matryoshka_dimensions=None,
    truncation_side="right",
)

expected_detail = EmbedDetail(
    task="embed",
    served_model_name=MODEL_NAME,
    architectures=["BertModel"],
    embedding_dim=384,
    max_model_len=512,
    is_matryoshka=False,
    matryoshka_dimensions=None,
    truncation_side="right",
)

expected_pooler_config = PoolerConfig(pooling_type='MEAN',
                                      normalize=True,
                                      softmax=None,
                                      step_tag_id=None,
                                      returned_token_ids=None)


def test_embed_offline_metadata(vllm_runner):
    from vllm.config import PoolerConfig

    with vllm_runner(MODEL_NAME, task="embed",
                     max_model_len=None) as vllm_model:
        metadata: EmbedMetadata = vllm_model.model.metadata

        assert isinstance(metadata, EmbedMetadata)

        assert metadata.brief == expected_brief
        assert metadata.detail == expected_detail

        assert metadata.hf_config["architectures"][0] == "BertModel"
        assert metadata.pooler_config == PoolerConfig(pooling_type='MEAN',
                                                      normalize=True,
                                                      softmax=None,
                                                      step_tag_id=None,
                                                      returned_token_ids=None)


@pytest.fixture(scope="module")
def server():
    from tests.utils import RemoteOpenAIServer
    args = [
        "--task", "embed", "--enforce-eager", "--disable-uvicorn-access-log",
        "--disable-brief-metadata-only"
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


def test_embed_online_metadata(server):
    url = server.url_for("metadata/brief")
    response = requests.get(url)
    assert response.json() == expected_brief.model_dump()

    url = server.url_for("metadata/detail")
    response = requests.get(url)
    assert response.json() == expected_detail.model_dump()

    url = server.url_for("metadata/hf_config")
    response = requests.get(url)
    assert response.json()

    url = server.url_for("metadata/pooler_config")
    response = requests.get(url)
    assert response.json() == vars(expected_pooler_config)
