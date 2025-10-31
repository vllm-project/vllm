# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.utils import GenerateModelInfo

from .ppl_utils import wikitext_ppl_test

MODELS = [
    GenerateModelInfo("Qwen/Qwen3-0.6B"),
    GenerateModelInfo("Qwen/Qwen3-0.6B-FP8"),
    # transformers:
    # Loading a GPTQ quantized model requires optimum, gptqmodel
    # GenerateModelInfo("Qwen/Qwen3-0.6B-GPTQ-Int8"),
]


@pytest.mark.parametrize("model_info", MODELS)
def test_ppl(hf_runner, vllm_runner, model_info: GenerateModelInfo):
    wikitext_ppl_test(hf_runner, vllm_runner, model_info)
