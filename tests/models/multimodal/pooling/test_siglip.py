# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
import torch
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

MODELS = [
    "google/siglip-base-patch16-224",
    "google/siglip2-base-patch16-224",
    # Different image embedding dim than text_config.hidden_size
    "google/siglip2-giant-opt-patch16-384",
]


def _run_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    input_cases: list[tuple[list[str], PromptImageInput, dict[str, Any]]],
    model: str,
    *,
    dtype: str,
) -> None:
    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        enforce_eager=True,
        max_model_len=64,
        gpu_memory_utilization=0.7,
    ) as vllm_model:
        vllm_outputs_per_case = [
            vllm_model.embed(
                input_texts,
                images=input_images,
                tokenization_kwargs=tokenization_kwargs,
            )
            for input_texts, input_images, tokenization_kwargs in input_cases
        ]

        texts = [HF_TEXT_PROMPTS[0]]
        images = [input_cases[1][1][0]]
        with pytest.raises(ValueError, match="not both"):
            vllm_model.embed(texts, images=images)

        vllm_model.embed(texts)
        vllm_model.embed([""], images=images)

        mixed_outputs = vllm_outputs_per_case[2]
        check_embeddings_close(
            embeddings_0_lst=[vllm_outputs_per_case[0][0]],
            embeddings_1_lst=[mixed_outputs[0]],
            name_0="text_only",
            name_1="mixed_text",
        )
        check_embeddings_close(
            embeddings_0_lst=[vllm_outputs_per_case[1][0]],
            embeddings_1_lst=[mixed_outputs[1]],
            name_0="image_only",
            name_1="mixed_image",
        )

    with hf_runner(model, dtype=dtype, auto_cls=SiglipModel) as hf_model:
        hf_outputs_per_case = []
        for input_texts, input_images, tokenization_kwargs in input_cases:
            all_inputs = hf_model.get_inputs(
                input_texts,
                images=input_images,
                tokenization_kwargs=tokenization_kwargs,
            )

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
        (
            HF_TEXT_PROMPTS,
            text_images,
            {
                "padding": "max_length",
                "max_length": 64,
            },
        ),
        (HF_IMAGE_PROMPTS, images, {}),
        (
            [HF_TEXT_PROMPTS[0], HF_IMAGE_PROMPTS[0]],
            [None, images[0]],
            {
                "padding": "max_length",
                "max_length": 64,
            },
        ),
    ]

    _run_test(
        hf_runner,
        vllm_runner,
        input_cases,  # type: ignore[arg-type]
        model,
        dtype=dtype,
    )
