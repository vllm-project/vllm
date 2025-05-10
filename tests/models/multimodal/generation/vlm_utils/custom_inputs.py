# SPDX-License-Identifier: Apache-2.0
"""Custom input builders for edge-cases in different models."""
from io import BytesIO
from typing import Callable

import requests
from PIL import Image

from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.multimodal.image import rescale_image_size
from vllm.multimodal.video import (rescale_video_size, resize_video,
                                   sample_frames_from_video)

from .....conftest import IMAGE_ASSETS, VIDEO_ASSETS
from .builders import build_multi_image_inputs, build_single_image_inputs
from .types import ImageSizeWrapper, PromptWithMultiModalInput, SizeType


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
        "<image><image><image><image>\nDescribe 4 images.",
        "<image>\nWhat is the season?",
    ]
    formatted_prompts = [formatter(prompt) for prompt in img_prompts]
    aspect_ratio_images = [
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
    ]

    return [
        PromptWithMultiModalInput.create(
            prompts=formatted_prompts,
            image_data=aspect_ratio_images,
        )
    ]


def multi_video_multi_aspect_ratio_inputs(formatter: Callable[[str], str],
                                          num_frames: int = 16):
    """Builds inputs for multi-video (varied sizes/aspect ratio) testing.
    
    Args:
        formatter: model-specific prompt formatter.
    """
    video = sample_frames_from_video(VIDEO_ASSETS[0].np_ndarrays, num_frames)
    # Apply the selected formatter to the base prompts
    video_prompts = [
        "<video><video>\nDescribe 2 videos.",
        "<video><video>\nDescribe 2 videos.",
        "<video><video><video><video>\nDescribe 4 videos.",
        "<video>\nWhy is this video funny?",
    ]
    formatted_prompts = [formatter(prompt) for prompt in video_prompts]
    aspect_ratio_videos = [
        [video, video],
        # Videos with different sizes and aspect-ratios
        [
            rescale_video_size(video, 0.1),
            video,
        ],
        [
            video,
            rescale_video_size(video, 0.25),
            resize_video(video, (183, 488)),
            resize_video(video, (488, 183))
        ],
        video,
    ]

    return [
        PromptWithMultiModalInput.create(
            prompts=formatted_prompts,
            video_data=aspect_ratio_videos,
        )
    ]


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


def windows_attention_image_qwen2_5_vl():
    # image from regression issue: https://github.com/vllm-project/vllm/issues/15122
    image_url = "https://aomediacodec.github.io/av1-avif/testFiles/Link-U/hato.jpg"
    image = Image.open(BytesIO(requests.get(image_url).content))

    question = "Describe the image."
    img_prompt = "<|vision_start|><|image_pad|><|vision_end|>"
    prompt = (f"<|im_start|>User\n{img_prompt}{question}<|im_end|>\n"
              "<|im_start|>assistant\n")

    wrapped_sf = ImageSizeWrapper(type=SizeType.SIZE_FACTOR, data=[0.5])
    return build_single_image_inputs([image], [prompt], wrapped_sf)


def mixed_modality_qwen2_5_omni(size_factor: float = 0.25):
    default_system = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
        "Group, capable of perceiving auditory and visual inputs, as well as "
        "generating text and speech.")
    question = ("What is recited in the audio? "
                "What is the content of this image? Why is this video funny?")
    prompt = (f"<|im_start|>system\n{default_system}<|im_end|>\n"
              "<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>"
              "<|vision_bos|><|IMAGE|><|vision_eos|>"
              "<|vision_bos|><|VIDEO|><|vision_eos|>"
              f"{question}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    audio = AudioAsset("mary_had_lamb").audio_and_sample_rate
    video = VideoAsset(name="baby_reading", num_frames=4).np_ndarrays
    image = ImageAsset("cherry_blossom").pil_image.convert("RGB")
    return [
        PromptWithMultiModalInput.create(
            prompts=[prompt],
            image_data=[rescale_image_size(image, size_factor=size_factor)],
            video_data=[video],
            audio_data=[audio],
        )
    ]
