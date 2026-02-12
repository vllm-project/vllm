# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import ModelConfig
from vllm.inputs.preprocess import InputPreprocessor

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("model_id", ["facebook/chameleon-7b"])
@pytest.mark.parametrize("prompt", ["", {"prompt_token_ids": []}])
@pytest.mark.skip(
    reason=(
        "Applying huggingface processor on text inputs results in "
        "significant performance regression for multimodal models. "
        "See https://github.com/vllm-project/vllm/issues/26320"
    )
)
def test_preprocessor_always_mm_code_path(model_id, prompt):
    model_config = ModelConfig(model=model_id)
    input_preprocessor = InputPreprocessor(model_config)

    # HF processor adds sep token
    tokenizer = input_preprocessor.get_tokenizer()
    sep_token_id = tokenizer.vocab[tokenizer.sep_token]

    processed_inputs = input_preprocessor.preprocess(prompt)
    assert sep_token_id in processed_inputs["prompt_token_ids"]
