# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F

from vllm.config import PoolerConfig


@pytest.mark.parametrize(
    "model",
    [
        "jason9693/Qwen2.5-1.5B-apeach",
        "papluca/xlm-roberta-base-language-detection"
    ],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:

    with vllm_runner(
            model,
            max_model_len=512,
            dtype=dtype,
            override_pooler_config=PoolerConfig(softmax=False)) as vllm_model:
        wo_softmax_out = vllm_model.classify(example_prompts)

    with vllm_runner(
            model,
            max_model_len=512,
            dtype=dtype,
            override_pooler_config=PoolerConfig(softmax=True)) as vllm_model:
        w_softmax_out = vllm_model.classify(example_prompts)

    for wo_softmax, w_softmax in zip(wo_softmax_out, w_softmax_out):
        wo_softmax = torch.tensor(wo_softmax)
        w_softmax = torch.tensor(w_softmax)

        assert not torch.allclose(
            wo_softmax, w_softmax,
            atol=1e-2), "override_pooler_config is not working"
        assert torch.allclose(F.softmax(wo_softmax, dim=-1), w_softmax,
                              1e-3 if dtype == "float" else 1e-2)