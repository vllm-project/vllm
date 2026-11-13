# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.entrypoints.pooling.basic.test_encode_vision import (
    EMBED_DIM,
    _conversation,
)
from vllm import LLM


@pytest.mark.skip_global_cleanup
def test_chat_vision(llm: LLM):
    """`LLM.embed()` convenience-wrapper smoke test.

    The full parameter matrix (`chat_template`, `chat_template_kwargs`,
    `media_io_kwargs`, `mm_processor_kwargs`, `cache_salt`,
    `add_special_tokens`, `add_generation_prompt`, `continue_final_message`,
    plus their combinations and mutual-exclusion validation) is exercised
    in `pooling/basic/test_encode_vision.py` against
    `llm.encode(..., pooling_task="embed")`, which is the underlying path
    that `llm.embed` wraps.

    This test only proves that the `LLM.embed()` convenience method accepts
    the same chat + multimodal input shape and that its returned
    `EmbeddingRequestOutput.outputs.embedding` has the expected dimension.
    """
    outputs = llm.embed(_conversation(), use_tqdm=False)

    assert len(outputs) == 1
    assert outputs[0].finished
    assert len(outputs[0].outputs.embedding) == EMBED_DIM
