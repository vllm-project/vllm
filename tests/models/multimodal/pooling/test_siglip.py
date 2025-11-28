# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import SiglipModel

from ....conftest import IMAGE_ASSETS, HfRunner, PromptImageInput, VllmRunner
from ...utils import check_embeddings_close

HF_TEXT_PROMPTS = [
    "a photo of a stop sign",
    "a photo of a cherry blossom",
]

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts(
    {
        "stop_sign": "",
        "cherry_blossom": "",
    }
)

MODELS = ["google/siglip-base-patch16-224", "google/siglip2-base-patch16-224"]


def _run_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    input_texts: list[str],
    input_images: PromptImageInput,
    model: str,
    *,
    dtype: str,
) -> None:
    with vllm_runner(
        model, runner="pooling", dtype=dtype, enforce_eager=True, max_model_len=64
    ) as vllm_model:
        vllm_outputs = vllm_model.embed(input_texts, images=input_images)

    with hf_runner(model, dtype=dtype, auto_cls=SiglipModel) as hf_model:
        all_inputs = hf_model.get_inputs(input_texts, images=input_images)

        all_outputs = []
        for inputs in all_inputs:
            inputs = hf_model.wrap_device(inputs)

            if "pixel_values" in inputs:
                pooled_output = hf_model.model.get_image_features(
                    pixel_values=inputs.pixel_values,
                ).squeeze(0)
            else:
                pooled_output = hf_model.model.get_text_features(
                    input_ids=inputs.input_ids,
                ).squeeze(0)

            all_outputs.append(pooled_output.tolist())

        hf_outputs = all_outputs

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
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


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
def test_models_image(
    hf_runner,
    vllm_runner,
    image_assets,
    model: str,
    dtype: str,
) -> None:
    input_texts_images = [
        (text, asset.pil_image) for text, asset in zip(HF_IMAGE_PROMPTS, image_assets)
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


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
def test_models_text_image_no_crash(
    vllm_runner,
    image_assets,
    model: str,
    dtype: str,
) -> None:
    texts = [HF_TEXT_PROMPTS[0]]
    images = [image_assets[0].pil_image]

    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        enforce_eager=True,
        max_model_len=64,
    ) as vllm_model:
        with pytest.raises(ValueError, match="not both"):
            vllm_model.embed(texts, images=images)

        vllm_model.embed(texts)
        vllm_model.embed([""], images=images)
