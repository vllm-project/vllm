# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest

from tests.models.language.pooling_mteb_test.mteb_score_utils import (
    MTEB_RERANK_LANGS,
    MTEB_RERANK_TASKS,
    MTEB_RERANK_TOL,
    RerankClientMtebEncoder,
    ScoreClientMtebEncoder,
    run_mteb_rerank,
)
from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
st_main_score = 0.33457


@pytest.fixture(scope="module")
def server():
    args = ["--runner", "pooling", "--enforce-eager", "--disable-uvicorn-access-log"]

    # ROCm: Use Flex Attention to support encoder-only self-attention.
    if current_platform.is_rocm():
        args.extend(["--attention-backend", "FLEX_ATTENTION"])

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


def test_mteb_score(server):
    url = server.url_for("score")
    encoder = ScoreClientMtebEncoder(MODEL_NAME, url)
    vllm_main_score = run_mteb_rerank(encoder, MTEB_RERANK_TASKS, MTEB_RERANK_LANGS)

    print("VLLM main score: ", vllm_main_score)
    print("SentenceTransformer main score: ", st_main_score)
    print("Difference: ", st_main_score - vllm_main_score)

    # We are not concerned that the vllm mteb results are better
    # than SentenceTransformers, so we only perform one-sided testing.
    assert st_main_score - vllm_main_score < MTEB_RERANK_TOL


def test_mteb_rerank(server):
    url = server.url_for("rerank")
    encoder = RerankClientMtebEncoder(MODEL_NAME, url)
    vllm_main_score = run_mteb_rerank(encoder, MTEB_RERANK_TASKS, MTEB_RERANK_LANGS)

    print("VLLM main score: ", vllm_main_score)
    print("SentenceTransformer main score: ", st_main_score)
    print("Difference: ", st_main_score - vllm_main_score)

    # We are not concerned that the vllm mteb results are better
    # than SentenceTransformers, so we only perform one-sided testing.
    assert st_main_score - vllm_main_score < MTEB_RERANK_TOL
