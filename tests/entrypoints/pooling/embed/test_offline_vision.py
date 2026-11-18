# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import weakref

import pytest

from vllm import LLM, EmbeddingRequestOutput

from ..basic.test_encode_vision import EMBED_DIM, _conversation

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"


@pytest.fixture(scope="module")
def llm(vllm_runner):
    with vllm_runner(
        MODEL_NAME,
        runner="pooling",
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=True,
        seed=0,
    ) as runner:
        # pytest caches yielded fixtures until after teardown, so use a proxy to
        # avoid retaining the LLM while VllmRunner.__exit__ releases ROCm memory.
        yield weakref.proxy(runner.llm)


# More tests are now covered in basic/test_encode_vision.py.
def test_chat_vision(llm: LLM):
    outputs = llm.embed(_conversation(), use_tqdm=False)

    assert len(outputs) == 1
    output = outputs[0]

    assert output.finished
    assert isinstance(output, EmbeddingRequestOutput)
    assert len(output.outputs.embedding) == EMBED_DIM
