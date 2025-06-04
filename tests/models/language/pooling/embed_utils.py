# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import Optional

import pytest

from tests.conftest import HfRunner
from tests.models.utils import (EmbedModelInfo, check_embeddings_close,
                                matryoshka_fy)


def run_embedding_correctness_test(
    hf_model: "HfRunner",
    inputs: list[str],
    vllm_outputs: Sequence[list[float]],
    dimensions: Optional[int] = None,
):
    hf_outputs = hf_model.encode(inputs)
    if dimensions:
        hf_outputs = matryoshka_fy(hf_outputs, dimensions)

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
        tol=1e-2,
    )


def correctness_test_embed_models(hf_runner,
                                  vllm_runner,
                                  model_info: EmbedModelInfo,
                                  example_prompts,
                                  vllm_extra_kwargs=None,
                                  hf_model_callback=None):
    if not model_info.enable_test:
        # A model family has many models with the same architecture,
        # and we don't need to test each one.
        pytest.skip("Skipping test.")

    # The example_prompts has ending "\n", for example:
    # "Write a short story about a robot that dreams for the first time.\n"
    # sentence_transformers will strip the input texts, see:
    # https://github.com/UKPLab/sentence-transformers/blob/v3.1.1/sentence_transformers/models/Transformer.py#L159
    # This makes the input_ids different between hf_model and vllm_model.
    # So we need to strip the input texts to avoid test failing.
    example_prompts = [str(s).strip() for s in example_prompts]

    vllm_extra_kwargs = vllm_extra_kwargs or {}
    vllm_extra_kwargs["dtype"] = model_info.dtype

    with vllm_runner(model_info.name,
                     task="embed",
                     max_model_len=None,
                     **vllm_extra_kwargs) as vllm_model:
        vllm_outputs = vllm_model.encode(example_prompts)

    with hf_runner(
            model_info.name,
            dtype="float32",
            is_sentence_transformer=True,
    ) as hf_model:

        if hf_model_callback is not None:
            hf_model_callback(hf_model)

        run_embedding_correctness_test(hf_model, example_prompts, vllm_outputs)
