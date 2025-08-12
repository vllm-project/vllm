import types
from typing import List, Optional, Type

import pytest
import torch
from huggingface_hub import snapshot_download
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
    "<|im_start|>User\n<image>\nWhat's the content in the center of the image?<|im_end|>\n<|im_start|>Assistant\n",  # noqa: E501
    "cherry_blossom":
    "<|im_start|>User\n<image>\nWhat is the season?<|im_end|>\n<|im_start|>Assistant\n",  # noqa: E501
})

# we use snapshot_download to prevent conflicts between
# dynamic_module and trust_remote_code for hf_runner
models = [
    snapshot_download("OpenGVLab/InternVL2-1B"),
    snapshot_download("OpenGVLab/InternVL2-2B"),
    # snapshot_download("OpenGVLab/InternVL2-4B"),  # broken
]


class InternVLProcessor:
    """A simple processor for InternVL2 HF model which misses a processor."""

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


# adapted from https://huggingface.co/OpenGVLab/InternVL2-1B/blob/main/modeling_internvl_chat.py
def generate(
    self,
    pixel_values: torch.FloatTensor,
    input_ids: torch.FloatTensor,
    attention_mask: Optional[torch.LongTensor] = None,
    **generate_kwargs,
) -> torch.LongTensor:
    """Generate method for InternVL2 model without fixed use_cache."""
    assert self.img_context_token_id is not None
    vit_embeds = self.extract_feature(pixel_values)
    input_embeds = self.language_model.get_input_embeddings()(input_ids)
    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    input_ids = input_ids.reshape(B * N)
    selected = (input_ids == self.img_context_token_id)
    assert selected.sum() != 0
    input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

    input_embeds = input_embeds.reshape(B, N, C)

    outputs = self.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        **generate_kwargs,
    )

    return outputs


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
                     max_model_len=4096,
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
        hf_model.model.get_output_embeddings = lambda: \
            hf_model.model.language_model.get_output_embeddings()
        hf_model.model.generate = types.MethodType(generate, hf_model.model)
        eos_token_id = hf_model.tokenizer.eos_token_id
        hf_outputs_per_image = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    images=hf_images,
                                                    eos_token_id=eos_token_id)
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
@torch.inference_mode()
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
