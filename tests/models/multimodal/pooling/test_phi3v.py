# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch.nn.functional as F
import transformers.utils
from PIL import Image

from vllm.assets.base import get_vllm_public_assets
from vllm.assets.image import VLM_IMAGES_DIR
from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

from ....conftest import IMAGE_ASSETS, HfRunner, PromptImageInput, VllmRunner
from ....utils import large_gpu_test
from ...utils import check_embeddings_close

# BC for method that was deleted in Transformers v5.
# Only needed for generating the HF reference.
transformers.utils.is_flash_attn_greater_or_equal_2_10 = (
    lambda: transformers.utils.is_flash_attn_greater_or_equal("2.1.0")
)

HF_TEXT_PROMPTS = [
    # T -> X
    "Find me an everyday image that matches the given caption: The label of the object is stop sign",  # noqa: E501
    # T -> X
    "Retrieve an image of this caption: cherry blossom",
]

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts(
    {
        # T + I -> X
        "stop_sign": "<|image_1|> Select the portion of the image that isolates the object of the given label: The label of the object is stop sign",  # noqa: E501
        # I -> X
        "cherry_blossom": "<|image_1|> Represent the given image for classification",  # noqa: E501
    }
)

MODELS = ["TIGER-Lab/VLM2Vec-Full"]

SPECIAL_TOKEN_IMAGE_PROMPT = (
    "\n<s><|user|>\n <|image_1|>\n\t <s>"
    "Represent the given image for classification<|end|>"
    "\n<|assistant|>\n"
)


def _get_cherry_blossom_image() -> Image.Image:
    return Image.open(
        get_vllm_public_assets(filename="cherry_blossom.jpg", s3_prefix=VLM_IMAGES_DIR)
    )


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
    with vllm_runner(
        model, runner="pooling", dtype=dtype, enforce_eager=True
    ) as vllm_model:
        vllm_outputs = vllm_model.embed(input_texts, images=input_images)

    # use eager mode for hf runner, since phi3_v didn't work with flash_attn
    hf_model_kwargs = {"_attn_implementation": "eager"}
    with hf_runner(model, dtype=dtype, model_kwargs=hf_model_kwargs) as hf_model:
        all_inputs = hf_model.get_inputs(input_texts, images=input_images)

        all_outputs = []
        for inputs in all_inputs:
            # Based on: https://github.com/TIGER-AI-Lab/VLM2Vec/blob/db3b951bccabba220c1f53ab46a734e50dd2fc08/src/model.py
            outputs = hf_model.model(
                **hf_model.wrap_device(inputs),
                return_dict=True,
                output_hidden_states=True,
            )
            last_hidden_state = outputs.hidden_states[-1][0]
            reps = last_hidden_state[inputs.attention_mask[0].sum() - 1]
            pooled_output = F.normalize(reps, p=2, dim=-1)

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


@pytest.mark.core_model
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_models_image_special_tokens_processing(
    model: str,
    dtype: str,
) -> None:
    model_config = ModelConfig(
        model,
        runner="pooling",
        trust_remote_code=True,
        dtype=dtype,
        max_model_len=1024,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    image = _get_cherry_blossom_image()

    processed_inputs = processor(
        SPECIAL_TOKEN_IMAGE_PROMPT,
        mm_items=processor.info.parse_mm_data({"image": image}),
        hf_processor_mm_kwargs={},
    )

    hf_processor = processor.info.get_hf_processor()
    hf_inputs = hf_processor(
        SPECIAL_TOKEN_IMAGE_PROMPT,
        images=image,
        return_tensors="pt",
    )

    image_token_id = hf_processor.get_special_image_token_id()
    hf_prompt_token_ids = [
        image_token_id if token_id < 0 else token_id
        for token_id in hf_inputs["input_ids"][0].tolist()
    ]

    prompt_token_ids = processed_inputs["prompt_token_ids"]

    assert prompt_token_ids == hf_prompt_token_ids
    assert prompt_token_ids.count(image_token_id) == hf_prompt_token_ids.count(
        image_token_id
    )
    assert prompt_token_ids.count(image_token_id) > 0
