# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import PoolerConfig

from ...utils import check_embeddings_close


@pytest.mark.parametrize(
    "model",
    [
        # Be careful of the order of models, decoder-only models should be
        # placed before encoder-only models, otherwise `Qwen2.5-0.5B-Instruct`
        # case won't pass because gte-Qwen2-1.5B-instruct will cache custom
        # model code with bidirectional attention.
        # [Decoder-only]
        pytest.param(
            "BAAI/bge-multilingual-gemma2",
            marks=[pytest.mark.core_model, pytest.mark.slow_test],
        ),
        pytest.param(
            "intfloat/e5-mistral-7b-instruct",
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param(
            "ssmits/Qwen2-7B-Instruct-embed-base", marks=[pytest.mark.cpu_model]
        ),
        # [Encoder-only]
        pytest.param(
            "BAAI/bge-base-en-v1.5",
            marks=[
                pytest.mark.core_model,
                pytest.mark.cpu_model,
                pytest.mark.slow_test,
            ],
        ),
        pytest.param("sentence-transformers/all-MiniLM-L12-v2"),
        pytest.param("intfloat/multilingual-e5-small"),
        # [Cross-Encoder]
        pytest.param(
            "sentence-transformers/stsb-roberta-base-v2",
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
    ],
)
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model,
) -> None:
    vllm_extra_kwargs = {}
    if model == "ssmits/Qwen2-7B-Instruct-embed-base":
        vllm_extra_kwargs["pooler_config"] = PoolerConfig(
            pooling_type="MEAN", normalize=False
        )

    max_model_len: int | None = 512
    if model in [
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/stsb-roberta-base-v2",
    ]:
        max_model_len = None

    # The example_prompts has ending "\n", for example:
    # "Write a short story about a robot that dreams for the first time.\n"
    # sentence_transformers will strip the input texts, see:
    # https://github.com/UKPLab/sentence-transformers/blob/v3.1.1/sentence_transformers/models/Transformer.py#L159
    # This makes the input_ids different between hf_model and vllm_model.
    # So we need to strip the input texts to avoid test failing.
    example_prompts = [str(s).strip() for s in example_prompts]

    with hf_runner(model, is_sentence_transformer=True) as hf_model:
        hf_outputs = hf_model.encode(example_prompts)

    with vllm_runner(
        model, runner="pooling", max_model_len=max_model_len, **vllm_extra_kwargs
    ) as vllm_model:
        vllm_outputs = vllm_model.embed(example_prompts)

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
        tol=1e-2,
    )
