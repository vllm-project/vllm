# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import CLIPModel

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

MODELS = ["openai/clip-vit-base-patch32"]


def _run_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    input_cases: list[tuple[list[str], PromptImageInput]],
    model: str,
    *,
    dtype: str,
) -> None:
    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).
    with vllm_runner(
        model, runner="pooling", dtype=dtype, enforce_eager=True, max_model_len=77
    ) as vllm_model:
        vllm_outputs_per_case = [
            vllm_model.embed(input_texts, images=input_images)
            for input_texts, input_images in input_cases
        ]

        texts = [HF_TEXT_PROMPTS[0]]
        images = [input_cases[1][1][0]]
        with pytest.raises(ValueError, match="not both"):
            vllm_model.embed(texts, images=images)

        # Should still be able to run subsequent requests
        vllm_model.embed(texts)
        vllm_model.embed([""], images=images)

    with hf_runner(model, dtype=dtype, auto_cls=CLIPModel) as hf_model:
        hf_outputs_per_case = []
        for input_texts, input_images in input_cases:
            all_inputs = hf_model.get_inputs(input_texts, images=input_images)

            hf_outputs = []
            for inputs in all_inputs:
                inputs = hf_model.wrap_device(inputs)

                if "pixel_values" in inputs:
                    pooled_output = hf_model.model.get_image_features(
                        pixel_values=inputs.pixel_values,
                    )
                else:
                    pooled_output = hf_model.model.get_text_features(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                    )

                if not isinstance(pooled_output, torch.Tensor):
                    pooled_output = pooled_output.pooler_output
                pooled_output = pooled_output.squeeze(0)
                hf_outputs.append(pooled_output.tolist())

            hf_outputs_per_case.append(hf_outputs)

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_case, vllm_outputs_per_case):
        check_embeddings_close(
            embeddings_0_lst=hf_outputs,
            embeddings_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
def test_models(
    hf_runner,
    vllm_runner,
    image_assets,
    model: str,
    dtype: str,
) -> None:
    text_images = [None] * len(HF_TEXT_PROMPTS)
    images = [asset.pil_image for asset in image_assets]
    input_cases = [
        (HF_TEXT_PROMPTS, text_images),
        (HF_IMAGE_PROMPTS, images),
    ]

    _run_test(
        hf_runner,
        vllm_runner,
        input_cases,  # type: ignore[arg-type]
        model,
        dtype=dtype,
    )
