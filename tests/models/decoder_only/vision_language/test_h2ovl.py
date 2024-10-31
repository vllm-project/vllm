import types
from typing import List, Optional, Tuple, Type, Union
import pytest
import torch
from PIL.Image import Image
from transformers import AutoConfig
from vllm.multimodal.utils import rescale_image_size
from ....conftest import (IMAGE_ASSETS, HfRunner, PromptImageInput, VllmRunner,
                          _ImageAssets)
from ...utils import check_logprobs_close
from .test_internvl import generate

# Import the functions to test
from vllm.model_executor.models.h2ovl import (
    image_to_pixel_values_wrapper,
    calculate_num_blocks,
    IMG_CONTEXT, IMG_END, IMG_START, image_to_pixel_values
)

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "<|prompt|><image>\nWhat's the content in the center of the image?<|end|><|answer|>",  # noqa: E501
    "cherry_blossom":
    "<|prompt|><image>\nWhat is the season?<|end|><|answer|>",  # noqa: E501
})
HF_MULTIIMAGE_IMAGE_PROMPT = "<|prompt|>Image-1: <image>\nImage-2: <image>\nDescribe the two images in short.<|end|><|answer|>"  # noqa: E501

models = [
    "h2oai/h2ovl-mississippi-800m",  # Replace with your actual model names
    "h2oai/h2ovl-mississippi-2b",
]
target_dtype = "bfloat16"


def run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    inputs: List[Tuple[List[str], PromptImageInput]],
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    mm_limit: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    """Inference result should be the same between hf and vllm.

    All the image fixtures for the test are from IMAGE_ASSETS.
    For huggingface runner, we provide the PIL images as input.
    For vllm runner, we provide MultiModalDataDict objects
    and corresponding MultiModalConfig as input.
    Note, the text input is also adjusted to abide by vllm contract.
    The text output is sanitized to be able to compare with hf.
    """

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    class H2OVLProcessor:
        """A simple processor for H2OVL models."""

        def __init__(self, hf_runner: HfRunner):
            self.num_image_token = hf_runner.model.num_image_token
            self.tokenizer = hf_runner.tokenizer
            self.dtype = hf_runner.model.dtype

            self.config = AutoConfig.from_pretrained(hf_runner.model_name,
                                                     trust_remote_code=True)
            self.vision_config = self.config.vision_config
            self.use_thumbnail = self.config.use_thumbnail
            self.min_num = self.config.min_dynamic_patch
            self.max_num = self.config.max_dynamic_patch
            self.image_size = self.vision_config.image_size

        def __call__(self, text: str, images: Union[Image, List[Image]],
                     **kwargs):
            images = [images] if isinstance(images, Image) else images
            pixel_values = [
                image_to_pixel_values(image, self.image_size, self.min_num,
                                      self.max_num,
                                      self.use_thumbnail,
                                      use_MSAC=self.config.use_msac).to(self.dtype)
                for image in images
            ]
            num_patches_list = [
                pixel_value.shape[0] for pixel_value in pixel_values
            ]
            pixel_values = torch.cat(pixel_values, dim=0)
            for num_patches in num_patches_list:
                context_tokens = IMG_CONTEXT * self.num_image_token \
                    * num_patches
                image_tokens = IMG_START + context_tokens + IMG_END
                text = text.replace('<image>', image_tokens, 1)
            prompt = self.tokenizer(text, return_tensors="pt")
            prompt.update({"pixel_values": pixel_values})
            return prompt

    # max_model_len should be greater than image_feature_size
    with vllm_runner(model,
                     max_model_len=8192,
                     dtype=dtype,
                     limit_mm_per_prompt={"image": mm_limit},
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True) as vllm_model:
        vllm_outputs_per_image = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images)
            for prompts, images in inputs
        ]

    with hf_runner(model, dtype=dtype) as hf_model:
        img_context_token_id = hf_model.tokenizer.convert_tokens_to_ids(
            "<IMG_CONTEXT>")
        hf_model.model.img_context_token_id = img_context_token_id
        hf_model.processor = H2OVLProcessor(hf_model)
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
            for prompts, hf_images in inputs
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_image,
                                        vllm_outputs_per_image):
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )


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
    images = [asset.pil_image for asset in image_assets]

    inputs_per_image = [(
        [prompt for _ in size_factors],
        [rescale_image_size(image, factor) for factor in size_factors],
    ) for image, prompt in zip(images, HF_IMAGE_PROMPTS)]

    run_test(
        hf_runner,
        vllm_runner,
        inputs_per_image,
        model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        mm_limit=1,
        tensor_parallel_size=1,
    )


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
        [0.5, 0.75, 1.0],
    ],
)
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
@torch.inference_mode()
def test_multi_images_models(hf_runner, vllm_runner, image_assets, model,
                             size_factors, dtype: str, max_tokens: int,
                             num_logprobs: int) -> None:
    images = [asset.pil_image for asset in image_assets]

    inputs_per_case = [
        ([HF_MULTIIMAGE_IMAGE_PROMPT for _ in size_factors],
         [[rescale_image_size(image, factor) for image in images]
          for factor in size_factors])
    ]

    run_test(
        hf_runner,
        vllm_runner,
        inputs_per_case,
        model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        mm_limit=2,
        tensor_parallel_size=1,
    )


def run_preprocessing_test(
    image: Image,
    config,
    max_dynamic_patch: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """Test the image preprocessing and calculate expected blocks."""

    if max_dynamic_patch is None:
        max_dynamic_patch = config.max_dynamic_patch

    width, height = image.size
    use_MSAC = config.use_msac

    # Create the mapper function with the provided configuration
    mapper = image_to_pixel_values_wrapper(config, max_dynamic_patch, use_MSAC)
    pixel_values = mapper(image)

    # Calculate the expected number of blocks
    if use_MSAC:
        # First pass
        blocks1, _, _, aspect_ratio = calculate_num_blocks(
            width,
            height,
            config.min_dynamic_patch,
            max_dynamic_patch,
            config.vision_config.image_size,
            use_thumbnail=False,  # Thumbnail is handled separately
            prior_aspect_ratio=None,
        )

        # Second pass
        blocks2, _, _, _ = calculate_num_blocks(
            width,
            height,
            config.min_dynamic_patch,
            max_dynamic_patch,
            config.vision_config.image_size,
            use_thumbnail=False,
            prior_aspect_ratio=aspect_ratio,
        )

        # Add thumbnail if use_thumbnail is True and total_blocks > 1
        if config.use_thumbnail:
            blocks1 +=1 if blocks1 > 1 else 0
            blocks2 +=1 if blocks2 > 1 else 0

            
        # Total blocks is the sum of blocks from both passes minus overlapping
        total_blocks = blocks1 + blocks2 -1    
            
        expected_blocks = total_blocks

    else:
        blocks, _, _, _ = calculate_num_blocks(
            width,
            height,
            config.min_dynamic_patch,
            max_dynamic_patch,
            config.vision_config.image_size,
            use_thumbnail=False,
            prior_aspect_ratio=None,
        )
        expected_blocks = blocks

        if config.use_thumbnail and expected_blocks > 1:
            expected_blocks += 1

    return pixel_values, expected_blocks

@pytest.mark.parametrize("model_name", models)
@pytest.mark.parametrize(
    "size_factors",
    [
        # Single-scale
        [1.0],
        # Single-scale, batched
        [1.0, 1.0, 1.0],
        # Multi-scale
        [0.25, 0.5, 1.0],
    ],
)
@pytest.mark.parametrize("max_dynamic_patch", [None, 2,4,8])
def test_image_preprocessing(image_assets, model_name, size_factors, max_dynamic_patch):
    """Test image preprocessing pipeline with different configurations."""
    # Load the configuration from the model
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    for asset in image_assets:
        image = asset.pil_image
        for factor in size_factors:
            scaled_image = rescale_image_size(image, factor)

            # Test preprocessing and get expected number of blocks
            pixel_values, expected_blocks = run_preprocessing_test(
                scaled_image, config, max_dynamic_patch
            )

            # Verify output shapes and properties
            actual_blocks = pixel_values.shape[0]
            assert actual_blocks == expected_blocks, (
                f"Expected {expected_blocks} blocks, got {actual_blocks}"
            )

            # Check image dimensions
            expected_size = (
                3,  # Number of channels (C, H, W)
                config.vision_config.image_size,
                config.vision_config.image_size,
            )
            for img in pixel_values:
                assert img.shape == expected_size, (
                    f"Expected image size {expected_size}, got {img.shape}"
                )