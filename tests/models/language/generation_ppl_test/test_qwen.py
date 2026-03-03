# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.utils import GenerateModelInfo

from .ppl_utils import wikitext_ppl_test

MODELS = [
    # for Qwen3
    GenerateModelInfo("Qwen/Qwen3-0.6B", hf_ppl=23.864173889160156),
    GenerateModelInfo("Qwen/Qwen3-0.6B-FP8", hf_ppl=24.313045501708984),
    # for Qwen3.5
    GenerateModelInfo("Qwen/Qwen3.5-0.8B", hf_ppl=19.38858413696289),
]


@pytest.mark.parametrize("model_info", MODELS)
def test_ppl(hf_runner, vllm_runner, model_info: GenerateModelInfo):
    vllm_extra_kwargs = {}
    if model_info.name == "Qwen/Qwen3.5-0.8B":
        vllm_extra_kwargs["language_model_only"] = True

    wikitext_ppl_test(
        hf_runner, vllm_runner, model_info, vllm_extra_kwargs=vllm_extra_kwargs
    )
