# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.models.utils import GenerateModelInfo

from .ppl_utils import wikitext_ppl_test

MODELS = [GenerateModelInfo("openai-community/gpt2-large")]


@pytest.mark.parametrize("model_info", MODELS)
def test_ppl(hf_runner, vllm_runner, model_info: GenerateModelInfo):
    wikitext_ppl_test(hf_runner, vllm_runner, model_info)
