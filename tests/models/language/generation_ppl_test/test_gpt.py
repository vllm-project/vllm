# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.models.utils import GenerateModelInfo

from .ppl_utils import wikitext_ppl_test

MODELS = [GenerateModelInfo("openai-community/gpt2-large", hf_ppl=19.457056045532227)]


@pytest.mark.parametrize("model_info", MODELS)
def test_ppl(hf_runner, vllm_runner, model_info: GenerateModelInfo):
    bf16_ppl = wikitext_ppl_test(hf_runner, vllm_runner, model_info)
    fp32_ppl = wikitext_ppl_test(
        hf_runner,
        vllm_runner,
        model_info,
        vllm_extra_kwargs={"hf_overrides": {"head_dtype": "float32"}},
    )

    differ = ((fp32_ppl - bf16_ppl) / bf16_ppl) * 100
    print("fp32 head difference (%):", differ)
    assert differ < 0
