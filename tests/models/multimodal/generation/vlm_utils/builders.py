# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for building inputs that can be leveraged for different test types.
"""
from collections.abc import Iterable
from pathlib import PosixPath
from typing import Callable, Optional, Union

import torch

from vllm.multimodal.audio import AudioResampler
from vllm.multimodal.image import rescale_image_size
from vllm.multimodal.video import (rescale_video_size, resize_video,
                                   sample_frames_from_video)

from .....conftest import AudioTestAssets, ImageTestAssets, VideoTestAssets
from .types import (SINGLE_AUDIO_BASE_PROMPT, SINGLE_IMAGE_BASE_PROMPTS,
                    TEST_AUDIO_PLACEHOLDER, TEST_IMG_PLACEHOLDER,
                    TEST_VIDEO_PLACEHOLDER, VIDEO_BASE_PROMPT,
                    ImageSizeWrapper, PromptWithMultiModalInput, SizeType,
                    VLMTestInfo)


def replace_test_placeholder(prompt: str, mm_idx_to_prompt: Callable[[int],
                                                                     str],
                             test_placeholder: str) -> str:
    """Given a prompt, replaces each test placeholder with the
    model-specific tag.
    """
    prompt_segments = prompt.split(test_placeholder)
    img_prompt = prompt_segments[0]
    for placeholder_idx, next_seg in enumerate(prompt_segments[1:], start=1):
        img_prompt += mm_idx_to_prompt(placeholder_idx)
        img_prompt += next_seg
    return img_prompt


def get_model_prompts(base_prompts: Iterable[str],
                      img_idx_to_prompt: Optional[Callable[[int], str]],
                      video_idx_to_prompt: Optional[Callable[[int], str]],
                      audio_idx_to_prompt: Optional[Callable[[int], str]],
                      prompt_formatter: Callable[[str], str]) -> list[str]:
    """Given a model-agnostic base prompt and test configuration for a model(s)
    to be tested, update the media placeholders and apply the prompt formatting
    to get the test prompt string for this model.

    Example for phi3v, given the base_prompt: "<image>What is the season?"
        1. Replace img placeholder(s)
          -> "<|image_1|>\nWhat is the season?"
        2. Apply prompt formatter:
          -> <|user|>\n<|image_1|>\nWhat is the season?<|end|>\n<|assistant|>\n
    """
    assert isinstance(base_prompts, (list, tuple))
    model_prompts = []
    for base_prompt in base_prompts:
        # Replace the multimodal placeholders in the base prompt with
        # the correct ones for the model that we are testing
        if img_idx_to_prompt:
            base_prompt = replace_test_placeholder(base_prompt,
                                                   img_idx_to_prompt,
                                                   TEST_IMG_PLACEHOLDER)

        if video_idx_to_prompt:
            base_prompt = replace_test_placeholder(base_prompt,
                                                   video_idx_to_prompt,
                                                   TEST_VIDEO_PLACEHOLDER)

        if audio_idx_to_prompt:
            base_prompt = replace_test_placeholder(base_prompt,
                                                   audio_idx_to_prompt,
                                                   TEST_AUDIO_PLACEHOLDER)

        # Apply the prompt formatter to wrap the base prompt with
        # the correct media placeholders to get the model test prompt
        model_prompt = prompt_formatter(base_prompt)
        model_prompts.append(model_prompt)
    return model_prompts


def build_single_image_inputs_from_test_info(
    test_info: VLMTestInfo,
    image_assets: ImageTestAssets,
    size_wrapper: ImageSizeWrapper,
    tmp_path: Optional[PosixPath] = None,
) -> list[PromptWithMultiModalInput]:
    if test_info.prompt_formatter is None:
        raise ValueError(
            "Prompt formatter must be set to build single image inputs")

    model_prompts = get_model_prompts(test_info.single_image_prompts,
                                      test_info.img_idx_to_prompt,
                                      test_info.video_idx_to_prompt,
                                      test_info.audio_idx_to_prompt,
                                      test_info.prompt_formatter)

    # For models that require a local path / URL encoded in the image; export
    # assets and encode into tmp_path for this test. This should be avoided
    # where possible (currently needed for Qwen-VL).
    if test_info.prompt_path_encoder is not None:
        if tmp_path is None:
            raise ValueError("Prompt path encoder requires setting local path")
        model_prompts = [
            test_info.prompt_path_encoder(tmp_path, prompt, [asset])
            for prompt, asset in zip(model_prompts, image_assets)
        ]

    images = [asset.pil_image for asset in image_assets]
    assert len(images) == len(model_prompts)
    return build_single_image_inputs(images, model_prompts, size_wrapper)


def build_single_image_inputs(
        images, model_prompts,
        size_wrapper: ImageSizeWrapper) -> list[PromptWithMultiModalInput]:
    # For every image / prompt pair, get a pair containing two lists of
    # length size_factors, where the first contains duplicates of the model
    # prompt [str], and the second contains copies of the image after being
    # scaled by one of the size factors.
    #
    # NOTE: rescaling preserves the image aspect ratio.
    return [
        PromptWithMultiModalInput(
            prompts=[prompt for _ in size_wrapper.data],
            image_data=[
                apply_image_size_scaling(image, size, size_wrapper.type)
                for size in size_wrapper.data
            ],
        ) for image, prompt in zip(images, model_prompts)
    ]


def build_multi_image_inputs_from_test_info(
    test_info: VLMTestInfo,
    image_assets: ImageTestAssets,
    size_wrapper: ImageSizeWrapper,
    tmp_path: Optional[PosixPath] = None,
) -> list[PromptWithMultiModalInput]:
    if test_info.prompt_formatter is None:
        raise ValueError(
            "Prompt formatter must be set to build multi image inputs")

    model_prompts = get_model_prompts([test_info.multi_image_prompt],
                                      test_info.img_idx_to_prompt,
                                      test_info.video_idx_to_prompt,
                                      test_info.audio_idx_to_prompt,
                                      test_info.prompt_formatter)

    if test_info.prompt_path_encoder is not None:
        if tmp_path is None:
            raise ValueError("Prompt path encoder requires setting local path")
        model_prompts = [
            test_info.prompt_path_encoder(tmp_path, model_prompt, image_assets)
            for model_prompt in model_prompts
        ]

    images = [asset.pil_image for asset in image_assets]

    # Currently, we only have one multi-image list & one multi-image prompt
    return build_multi_image_inputs(
        image_lists=[images],
        model_prompts=model_prompts,
        size_wrapper=size_wrapper,
    )


def build_multi_image_inputs(
        image_lists, model_prompts,
        size_wrapper: ImageSizeWrapper) -> list[PromptWithMultiModalInput]:
    return [
        PromptWithMultiModalInput(
            prompts=[prompt for _ in size_wrapper.data],
            image_data=[[
                apply_image_size_scaling(image, size, size_wrapper.type)
                for image in images
            ] for size in size_wrapper.data],
        ) for images, prompt in zip(image_lists, model_prompts)
    ]


def build_embedding_inputs_from_test_info(
    test_info: VLMTestInfo,
    image_assets: ImageTestAssets,
    size_wrapper: ImageSizeWrapper,
):
    # These conditions will always be true if invoked through filtering,
    # but we still check them in case this is ever called directly
    if test_info.prompt_formatter is None:
        raise ValueError(
            "Prompt formatter must be set to build image embedding inputs")
    if size_wrapper.type != SizeType.SIZE_FACTOR or not \
            all(factor == 1.0 for factor in size_wrapper.data):
        raise ValueError("Embedding tests require constant (1.0) size factors")
    if test_info.convert_assets_to_embeddings is None:
        raise ValueError("No conversion func for getting embeddings found")

    model_prompts = get_model_prompts(
        SINGLE_IMAGE_BASE_PROMPTS,
        test_info.img_idx_to_prompt,
        test_info.video_idx_to_prompt,
        test_info.audio_idx_to_prompt,
        test_info.prompt_formatter,
    )

    images = [asset.pil_image for asset in image_assets]
    embeds = test_info.convert_assets_to_embeddings(image_assets)
    assert len(images) == len(model_prompts)

    inputs = build_single_image_inputs(images, model_prompts, size_wrapper)
    vllm_embeddings = build_single_image_inputs(embeds, model_prompts,
                                                size_wrapper)
    return inputs, vllm_embeddings


def build_video_inputs_from_test_info(
    test_info: VLMTestInfo,
    video_assets: VideoTestAssets,
    size_wrapper: ImageSizeWrapper,
    num_frames: int,
) -> list[PromptWithMultiModalInput]:
    if test_info.prompt_formatter is None:
        raise ValueError("Prompt formatter must be set to build video inputs")
    model_prompts = get_model_prompts(
        [VIDEO_BASE_PROMPT],
        test_info.img_idx_to_prompt,
        test_info.video_idx_to_prompt,
        test_info.audio_idx_to_prompt,
        test_info.prompt_formatter,
    )

    sampled_vids = [
        sample_frames_from_video(asset.np_ndarrays, num_frames)
        for asset in video_assets
    ]

    video_scaler = (resize_video if size_wrapper.type == SizeType.FIXED_SIZE
                    else rescale_video_size)

    return [
        PromptWithMultiModalInput(
            prompts=[prompt for _ in size_wrapper.data],
            video_data=[
                video_scaler(video, size) for size in size_wrapper.data
            ],
        ) for video, prompt in zip(sampled_vids, model_prompts)
    ]


def apply_image_size_scaling(image, size: Union[float, tuple[int, int]],
                             size_type: SizeType):
    """Applies a size scaler to one image; this can be a an image size factor,
    which scales the image while maintaining the aspect ratio"""
    # Special case for embeddings; if it's a tensor, it's only valid if we
    # are considering size factors at constant scale, i.e., we just clone
    # the tensor
    if isinstance(image, torch.Tensor):
        assert size_type == SizeType.SIZE_FACTOR and size == 1
        return image
    if size_type == SizeType.SIZE_FACTOR:
        # We have a list of image size factors
        return rescale_image_size(image, size)
    elif size_type == SizeType.FIXED_SIZE:
        # We have a list of fixed sizes
        return image.resize(size)
    raise ValueError("ImageSizeWrapper type must be FIXED_SIZE or SIZE_FACTOR")


def build_audio_inputs_from_test_info(
    test_info: VLMTestInfo,
    audio_assets: AudioTestAssets,
) -> list[PromptWithMultiModalInput]:
    if test_info.prompt_formatter is None:
        raise ValueError("Prompt formatter must be set to build audio inputs")
    model_prompts = get_model_prompts(
        SINGLE_AUDIO_BASE_PROMPT,
        test_info.img_idx_to_prompt,
        test_info.video_idx_to_prompt,
        test_info.audio_idx_to_prompt,
        test_info.prompt_formatter,
    )
    resampler = AudioResampler(
        target_sr=16000,
        method="librosa",
    )
    audios = [asset.audio_and_sample_rate for asset in audio_assets]
    resampled_audios = [(
        resampler.resample(
            audio,
            orig_sr=sr,
        ),
        int(resampler.target_sr),
    ) for audio, sr in audios]

    return [
        PromptWithMultiModalInput(
            prompts=model_prompts,
            audio_data=resampled_audios,
        )
    ]
