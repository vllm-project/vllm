from typing import List, Optional, Type

import pytest
from PIL.Image import Image

from vllm.model_executor.models.internvl import (IMG_CONTEXT, IMG_END,
                                                 IMG_START,
                                                 image_to_pixel_values)
from vllm.multimodal.utils import rescale_image_size
from vllm.utils import is_cpu

from ..conftest import IMAGE_ASSETS, HfRunner, VllmRunner, _ImageAssets
from .utils import check_logprobs_close

pytestmark = pytest.mark.vlm

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "<image>\nWhat's the content of the image?\n",
    "cherry_blossom":
    "<image>\nWhat is the season?\n",
})

models = ["OpenGVLab/InternVL2-4B", "OpenGVLab/InternVL2-8B"]


class InternVLProcessor:

    def __init__(self, hf_runner: HfRunner):
        self.num_image_token = hf_runner.model.num_image_token
        self.tokenizer = hf_runner.tokenizer
        self.dtype = hf_runner.model.dtype

    def __call__(self, text: str, images: Image, **kwargs):
        pixel_values = image_to_pixel_values(images).to(self.dtype)
        num_patches_list = [pixel_values.shape[0]]
        for num_patches in num_patches_list:
            context_tokens = IMG_CONTEXT * self.num_image_token * num_patches
            image_tokens = IMG_START + context_tokens + IMG_END
            text = text.replace('<image>', image_tokens, 1)
        prompt = self.tokenizer(text, return_tensors="pt")
        prompt.update({"pixel_values": pixel_values})
        return prompt


def run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    image_assets: _ImageAssets,
    model: str,
    *,
    size_factors: List[float],
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    """Inference result should be the same between hf and vllm.

    All the image fixtures for the test is under tests/images.
    For huggingface runner, we provide the PIL images as input.
    For vllm runner, we provide MultiModalDataDict objects 
    and corresponding vision language config as input.
    Note, the text input is also adjusted to abide by vllm contract.
    The text output is sanitized to be able to compare with hf.
    """
    images = [asset.pil_image for asset in image_assets]

    inputs_per_image = [(
        [prompt for _ in size_factors],
        [rescale_image_size(image, factor) for factor in size_factors],
    ) for image, prompt in zip(images, HF_IMAGE_PROMPTS)]

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    # max_model_len should be greater than image_feature_size
    with vllm_runner(model,
                     max_model_len=2048,
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True) as vllm_model:
        vllm_outputs_per_image = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images)
            for prompts, images in inputs_per_image
        ]

    with hf_runner(model, dtype=dtype) as hf_model:
        img_context_token_id = hf_model.tokenizer.convert_tokens_to_ids(
            "<IMG_CONTEXT>")
        hf_model.model.img_context_token_id = img_context_token_id
        hf_model.processor = InternVLProcessor(hf_model)
        hf_outputs_per_image = []
        hf_model.model.get_output_embeddings = lambda: \
            hf_model.model.language_model.get_output_embeddings()
        hf_outputs_per_image = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    images=hf_images)
            for prompts, hf_images in inputs_per_image
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_image,
                                        vllm_outputs_per_image):
        # TODO: Check whether using original CLIPVisionModel can improve
        # consistency against HF
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )


target_dtype = "half"
if is_cpu():
    target_dtype = "bfloat16"


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "size_factors",
    [
        # No image
        [],
        # Single-scale
        [1.0],
        # Single-scale, batched
        [1.0, 1.0, 1.0],
        # Multi-scale
        [0.25, 0.5, 1.0],
    ],
)
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(hf_runner, vllm_runner, image_assets, model, size_factors,
                dtype: str, max_tokens: int, num_logprobs: int) -> None:
    run_test(
        hf_runner,
        vllm_runner,
        image_assets,
        model,
        size_factors=size_factors,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
    )
