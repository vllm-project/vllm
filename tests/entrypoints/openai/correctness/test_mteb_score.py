# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest

# yapf conflicts with isort for this block
# yapf: disable
from tests.models.language.pooling.mteb_utils import (MTEB_RERANK_LANGS,
                                                      MTEB_RERANK_TASKS,
                                                      MTEB_RERANK_TOL,
                                                      RerankClientMtebEncoder,
                                                      ScoreClientMtebEncoder,
                                                      run_mteb_rerank)
# yapf: enable
from tests.utils import RemoteOpenAIServer

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAIN_SCORE = 0.33437


@pytest.fixture(scope="module")
def server():
    args = [
        "--task", "score", "--enforce-eager", "--disable-uvicorn-access-log"
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


def test_mteb_score(server):
    url = server.url_for("score")
    encoder = ScoreClientMtebEncoder(MODEL_NAME, url)
    vllm_main_score = run_mteb_rerank(encoder, MTEB_RERANK_TASKS,
                                      MTEB_RERANK_LANGS)
    st_main_score = MAIN_SCORE

    print("VLLM main score: ", vllm_main_score)
    print("SentenceTransformer main score: ", st_main_score)
    print("Difference: ", st_main_score - vllm_main_score)

    assert st_main_score == pytest.approx(vllm_main_score, abs=MTEB_RERANK_TOL)


def test_mteb_rerank(server):
    url = server.url_for("rerank")
    encoder = RerankClientMtebEncoder(MODEL_NAME, url)
    vllm_main_score = run_mteb_rerank(encoder, MTEB_RERANK_TASKS,
                                      MTEB_RERANK_LANGS)
    st_main_score = MAIN_SCORE

    print("VLLM main score: ", vllm_main_score)
    print("SentenceTransformer main score: ", st_main_score)
    print("Difference: ", st_main_score - vllm_main_score)

    assert st_main_score == pytest.approx(vllm_main_score, abs=MTEB_RERANK_TOL)
