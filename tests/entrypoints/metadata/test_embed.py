# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest
import requests

from vllm.entrypoints.metadata.base import PoolerConfigMetadata
from vllm.entrypoints.metadata.embed import EmbedBrief, EmbedMetadata

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

MODEL_NAME = "intfloat/e5-small"

expected_brief = EmbedBrief(
    task="embed",
    served_model_name=MODEL_NAME,
    embedding_dim=384,
    max_model_len=512,
    is_matryoshka=False,
    matryoshka_dimensions=None,
    truncation_side="right",
)

expected_pooler_config = PoolerConfigMetadata(pooling_type='MEAN',
                                              normalize=True,
                                              softmax=None,
                                              step_tag_id=None,
                                              returned_token_ids=None)


def test_embed_offline_metadata(vllm_runner):
    with vllm_runner(MODEL_NAME, task="embed",
                     max_model_len=None) as vllm_model:
        metadata: EmbedMetadata = vllm_model.model.metadata

        assert isinstance(metadata, EmbedMetadata)

        assert metadata.brief == expected_brief
        assert metadata.pooler_config == expected_pooler_config


@pytest.fixture(scope="module")
def server():
    from tests.utils import RemoteOpenAIServer
    args = [
        "--task", "embed", "--enforce-eager", "--disable-uvicorn-access-log",
        "--enable-metadata-dev-mode"
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


def test_embed_online_metadata(server):
    url = server.url_for("metadata/brief")
    response = requests.get(url)
    assert response.json() == expected_brief.model_dump()

    url = server.url_for("metadata/hf_config")
    response = requests.get(url)
    assert response.json()

    url = server.url_for("metadata/pooler_config")
    response = requests.get(url)
    assert response.json() == vars(expected_pooler_config)
