# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from tests.models.language.pooling.mteb_utils import (MTEB_EMBED_TASKS,
                                                      MTEB_EMBED_TOL,
                                                      OpenAIClientMtebEncoder,
                                                      run_mteb_embed_task,
                                                      run_mteb_embed_task_st)
from tests.utils import RemoteOpenAIServer

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

MODEL_NAME = "BAAI/bge-m3"
DTYPE = "float16"
MAIN_SCORE = 0.7873427091972599


@pytest.fixture(scope="module")
def server():
    args = [
        "--task", "embed", "--dtype", DTYPE, "--enforce-eager",
        "--max-model-len", "512"
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


def test_mteb(server):
    client = server.get_client()
    encoder = OpenAIClientMtebEncoder(MODEL_NAME, client)
    vllm_main_score = run_mteb_embed_task(encoder, MTEB_EMBED_TASKS)
    st_main_score = MAIN_SCORE or run_mteb_embed_task_st(
        MODEL_NAME, MTEB_EMBED_TASKS)

    print("VLLM main score: ", vllm_main_score)
    print("SentenceTransformer main score: ", st_main_score)
    print("Difference: ", st_main_score - vllm_main_score)

    assert st_main_score == pytest.approx(vllm_main_score, abs=MTEB_EMBED_TOL)
