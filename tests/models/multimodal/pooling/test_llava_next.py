# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText

from vllm.platforms import current_platform

from ....conftest import IMAGE_ASSETS, HfRunner, PromptImageInput, VllmRunner
from ....utils import large_gpu_test
from ...utils import check_embeddings_close

# Llava Next embedding implementation is only supported by CUDA.
# If run on ROCm, hf_model.model.resize_token_embeddings will
# cause the following error:
#    RuntimeError: Calling torch.linalg.cholesky on a CUDA tensor
#    requires compiling PyTorch with MAGMA. Please use PyTorch
#    built with MAGMA support.
# If run on CPU, hf_model.model.resize_token_embeddings will
# cause the following error:
#    RuntimeError: Calling torch.linalg.cholesky on a CPU tensor
#    requires compiling PyTorch with LAPACK. Please use PyTorch
#    built with LAPACK support.
pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Llava Next model uses op that is only supported in CUDA")

llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'  # noqa: E501

HF_TEXT_PROMPTS = [
    # T -> X
    llama3_template.format(
        "The label of the object is stop sign\nSummary above sentence in one word: "  # noqa: E501
    ),
    # T -> X
    llama3_template.format(
        "cherry blossom\nSummary above sentence in one word: "),
]

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    # I -> X
    "stop_sign":
    llama3_template.format("<image>\nSummary above image in one word: "),
    # I -> X
    "cherry_blossom":
    llama3_template.format("<image>\nSummary above image in one word: "),
})

MODELS = ["royokong/e5-v"]


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
                     task="embed",
                     dtype=dtype,
                     max_model_len=4096,
                     enforce_eager=True) as vllm_model:
        vllm_outputs = vllm_model.embed(input_texts, images=input_images)

    with hf_runner(model, dtype=dtype,
                   auto_cls=AutoModelForImageTextToText) as hf_model:
        # Patch the issue where generation_config.json is missing
        hf_model.processor.patch_size = \
            hf_model.model.config.vision_config.patch_size

        # Patch the issue where image_token_id
        # exceeds the maximum allowed vocab size
        hf_model.model.resize_token_embeddings(
            hf_model.model.language_model.vocab_size + 1)

        all_inputs = hf_model.get_inputs(input_texts, images=input_images)

        all_outputs = []
        for inputs in all_inputs:
            # Based on: https://huggingface.co/royokong/e5-v
            outputs = hf_model.model(
                **hf_model.wrap_device(inputs),
                return_dict=True,
                output_hidden_states=True,
            )
            pooled_output = F.normalize(outputs.hidden_states[-1][0, -1, :],
                                        dim=-1)

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
    image_assets,
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
        input_images,  # type: ignore
        model,
        dtype=dtype,
    )


@large_gpu_test(min_gb=48)
@pytest.mark.core_model
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_models_image(
    hf_runner,
    vllm_runner,
    image_assets,
    model: str,
    dtype: str,
) -> None:
    input_texts_images = [
        (text, asset.pil_image)
        for text, asset in zip(HF_IMAGE_PROMPTS, image_assets)
    ]
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
