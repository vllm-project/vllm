"""Custom input builders for edge-cases in different models."""
from typing import Callable

from vllm.multimodal.utils import rescale_image_size

from .core_utils import build_single_image_inputs, build_multi_image_inputs
from .....conftest import IMAGE_ASSETS
from ..vlm_test_types import ImageSizeWrapper, SizeType

def multi_image_multi_aspect_ratio_inputs(formatter: Callable[[str], str]):
    """Builds inputs for multi-image (varied sizes/aspect ratio) testing.
    
    Args:
        formatter: model-specific prompt formatter.
    """
    stop_sign = IMAGE_ASSETS[0].pil_image
    cherry_blossom = IMAGE_ASSETS[1].pil_image

    # Apply the selected formatter to the base prompts
    img_prompts = [
        "<image><image>\nDescribe 2 images.",
        "<image><image>\nDescribe 2 images.",
        "<image><image><image><image>\nDescribe 4 images.",  # noqa: E501
        "<image>\nWhat is the season?",
    ]
    formatted_prompts = [formatter(prompt) for prompt in img_prompts]

    return [(
        formatted_prompts,
        [
            [stop_sign, cherry_blossom],
            # Images with different sizes and aspect-ratios
            [
                rescale_image_size(stop_sign, 0.1),
                stop_sign,
            ],
            [
                stop_sign,
                rescale_image_size(stop_sign, 0.25),
                cherry_blossom.resize((183, 488)),
                cherry_blossom.resize((488, 183))
            ],
            cherry_blossom,
        ])]


def different_patch_input_cases_internvl():
    images = [asset.pil_image.resize((896, 896)) for asset in IMAGE_ASSETS]
    formatter = lambda img_prompt: f"<|im_start|>User\n{img_prompt}<|im_end|>\n<|im_start|>Assistant\n"  # noqa: E501
    single_img_prompts = [
        "<image>\nWhat's the content in the center of the image?",
        "<image>\nWhat is the season?",
    ]
    multi_img_prompts = [
        "Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.\n",  # noqa: E501
    ]
    formatted_sprompts = [formatter(prompt) for prompt in single_img_prompts]
    formatted_mprompts = [formatter(prompt) for prompt in multi_img_prompts]

    wrapped_sf = ImageSizeWrapper(type=SizeType.SIZE_FACTOR, data=[0.5, 1.0])
    return [
        build_single_image_inputs(images, formatted_sprompts, wrapped_sf),
        build_multi_image_inputs([images], formatted_mprompts, wrapped_sf),
    ]
