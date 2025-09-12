# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch.nn.functional as F
from transformers import SiglipTextModel

from ....conftest import HfRunner, PromptImageInput, VllmRunner
from ...utils import check_embeddings_close

HF_TEXT_PROMPTS = [
    "Find me an everyday image that matches the given caption: The label of the object is stop sign",  # noqa: E501
    "Retrieve an image of this caption: cherry blossom",
]

MODELS = ["google/siglip2-so400m-patch14-384"]


def _run_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    input_texts: list[str],
    input_images: PromptImageInput,
    model: str,
    *,
    dtype: str,
) -> None:
    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).
    with vllm_runner(model,
                     runner="pooling",
                     dtype=dtype,
                     max_model_len=64,
                     enable_prefix_caching=False,
                     hf_overrides={"architectures":
                                   ["Siglip2TextModel"]}) as vllm_model:
        vllm_outputs = vllm_model.embed(input_texts, images=input_images)

    with hf_runner(model_name=model, dtype=dtype,
                   auto_cls=SiglipTextModel) as hf_model:
        all_inputs = hf_model.get_inputs(input_texts)

        all_outputs = []
        for inputs in all_inputs:
            outputs = hf_model.model(
                **hf_model.wrap_device(inputs),
                return_dict=True,
            )
            pooled_output = F.normalize(outputs.pooler_output, p=2,
                                        dim=-1)[0, :]

            all_outputs.append(pooled_output.tolist())

        hf_outputs = all_outputs

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.core_model
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_models_text(
    hf_runner,
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    input_texts_images = [(text, None) for text in HF_TEXT_PROMPTS]
    input_texts = [text for text, _ in input_texts_images]
    input_images = [image for _, image in input_texts_images]

    _run_test(
        hf_runner,
        vllm_runner,
        input_texts,
        input_images,
        model,
        dtype=dtype,
    )
