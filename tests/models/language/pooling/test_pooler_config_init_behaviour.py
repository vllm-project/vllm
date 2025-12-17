# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F

from tests.models.utils import softmax
from vllm.config import PoolerConfig


@pytest.mark.parametrize(
    "model",
    ["jason9693/Qwen2.5-1.5B-apeach", "papluca/xlm-roberta-base-language-detection"],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_classify_models_using_activation(
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
        pooler_config=PoolerConfig(use_activation=False),
    ) as vllm_model:
        wo_activation_out = vllm_model.classify(example_prompts)

    with vllm_runner(
        model,
        max_model_len=512,
        dtype=dtype,
        pooler_config=PoolerConfig(use_activation=True),
    ) as vllm_model:
        w_activation_out = vllm_model.classify(example_prompts)

    for wo_activation, w_activation in zip(wo_activation_out, w_activation_out):
        wo_activation = torch.tensor(wo_activation)
        w_activation = torch.tensor(w_activation)

        assert not torch.allclose(wo_activation, w_activation, atol=1e-2), (
            "pooler_config is not working"
        )
        assert torch.allclose(
            softmax(wo_activation), w_activation, 1e-3 if dtype == "float" else 1e-2
        )


@pytest.mark.parametrize(
    "model",
    [
        "intfloat/multilingual-e5-small",
    ],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_embed_models_using_normalize(
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
        pooler_config=PoolerConfig(normalize=False),
    ) as vllm_model:
        wo_normalize = torch.tensor(vllm_model.embed(example_prompts))

    with vllm_runner(
        model,
        max_model_len=512,
        dtype=dtype,
        pooler_config=PoolerConfig(normalize=True),
    ) as vllm_model:
        w_normalize = torch.tensor(vllm_model.embed(example_prompts))

    assert not torch.allclose(wo_normalize, w_normalize, atol=1e-2), (
        "pooler_config normalize is not working"
    )
    assert torch.allclose(
        F.normalize(wo_normalize, p=2, dim=-1), w_normalize, atol=1e-2
    ), "w_normal should be close to normal(wo_normal)."


@pytest.mark.parametrize(
    "model",
    [
        "internlm/internlm2-1_8b-reward",
    ],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_reward_models_using_activation(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    with vllm_runner(
        model,
        max_model_len=1024,
        dtype=dtype,
        pooler_config=PoolerConfig(use_activation=False),
    ) as vllm_model:
        wo_activation = vllm_model.reward(example_prompts)

    with vllm_runner(
        model,
        max_model_len=1024,
        dtype=dtype,
        pooler_config=PoolerConfig(use_activation=True),
    ) as vllm_model:
        w_activation = vllm_model.reward(example_prompts)

    for wo, w in zip(wo_activation, w_activation):
        wo = torch.tensor(wo)
        w = torch.tensor(w)

        assert not torch.allclose(wo, w, atol=1e-2), (
            "pooler_config activation is not working"
        )
        assert torch.allclose(softmax(wo), w, atol=1e-2), (
            "w_activation should be close to activation(wo_activation)."
        )


@pytest.mark.parametrize(
    "model",
    [
        "intfloat/multilingual-e5-small",
    ],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_multi_vector_retrieval_models_using_normalize(
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
        pooler_config=PoolerConfig(normalize=False),
    ) as vllm_model:
        wo_normalize = vllm_model.token_embed(example_prompts)

    with vllm_runner(
        model,
        max_model_len=512,
        dtype=dtype,
        pooler_config=PoolerConfig(normalize=True),
    ) as vllm_model:
        w_normalize = vllm_model.token_embed(example_prompts)

    for wo, w in zip(wo_normalize, w_normalize):
        assert not torch.allclose(wo, w, atol=1e-2), (
            "pooler_config normalize is not working"
        )
        assert torch.allclose(F.normalize(wo, p=2, dim=-1), w, atol=1e-2), (
            "w_normal should be close to normal(wo_normal)."
        )
