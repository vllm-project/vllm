# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.utils import ModelInfo

from .ppl_utils import wikitext_ppl_test

MODELS = [ModelInfo("Qwen/Qwen3-0.6B")]


@pytest.mark.parametrize("model_info", MODELS)
def test_ppl(hf_runner, vllm_runner, model_info: ModelInfo):
    wikitext_ppl_test(hf_runner, vllm_runner, model_info)
