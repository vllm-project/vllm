# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest

from tests.models.language.pooling.mteb_utils import (MTEB_EMBED_TASKS,
                                                      MTEB_EMBED_TOL,
                                                      OpenAIClientMtebEncoder,
                                                      run_mteb_embed_task)
from tests.utils import RemoteOpenAIServer

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

MODEL_NAME = "intfloat/e5-small"
MAIN_SCORE = 0.7422994752439667


@pytest.fixture(scope="module")
def server():
    args = [
        "--task", "embed", "--enforce-eager", "--disable-uvicorn-access-log"
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


def test_mteb_embed(server):
    client = server.get_client()
    encoder = OpenAIClientMtebEncoder(MODEL_NAME, client)
    vllm_main_score = run_mteb_embed_task(encoder, MTEB_EMBED_TASKS)
    st_main_score = MAIN_SCORE

    print("VLLM main score: ", vllm_main_score)
    print("SentenceTransformer main score: ", st_main_score)
    print("Difference: ", st_main_score - vllm_main_score)

    assert st_main_score == pytest.approx(vllm_main_score, abs=MTEB_EMBED_TOL)
