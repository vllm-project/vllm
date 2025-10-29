# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Types for writing multimodal model tests."""

from collections.abc import Callable, Iterable
from enum import Enum
from pathlib import PosixPath
from typing import Any, NamedTuple

import torch
from pytest import MarkDecorator
from transformers import AutoModelForCausalLM
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from vllm.config.model import RunnerOption
from vllm.logprobs import SampleLogprobs
from vllm.transformers_utils.tokenizer import AnyTokenizer

from .....conftest import (
    AUDIO_ASSETS,
    IMAGE_ASSETS,
    HfRunner,
    ImageAsset,
    ImageTestAssets,
    PromptAudioInput,
    PromptImageInput,
    PromptVideoInput,
)
from ....utils import check_logprobs_close

# meta image tag; will be replaced by the appropriate tag for the model
TEST_IMG_PLACEHOLDER = "<vlm_image>"
TEST_VIDEO_PLACEHOLDER = "<vlm_video>"
TEST_AUDIO_PLACEHOLDER = "<lmm_audio>"

SINGLE_IMAGE_BASE_PROMPTS = IMAGE_ASSETS.prompts(
    {
        "stop_sign": f"{TEST_IMG_PLACEHOLDER}What's the content of the image?",
        "cherry_blossom": f"{TEST_IMG_PLACEHOLDER}What is the season?",
    }
)
SINGLE_AUDIO_BASE_PROMPT = AUDIO_ASSETS.prompts(
    {
        "mary_had_lamb": f"{TEST_AUDIO_PLACEHOLDER}Transcribe this audio into English.",  # noqa: E501
        "winning_call": f"{TEST_AUDIO_PLACEHOLDER}What is happening in this audio clip?",  # noqa: E501
    }
)

MULTI_IMAGE_BASE_PROMPT = f"Image-1: {TEST_IMG_PLACEHOLDER}Image-2: {TEST_IMG_PLACEHOLDER}Describe the two images in detail.\n"  # noqa: E501
VIDEO_BASE_PROMPT = f"{TEST_VIDEO_PLACEHOLDER}Why is this video funny?"


IMAGE_SIZE_FACTORS = [(), (1.0,), (1.0, 1.0, 1.0), (0.25, 0.5, 1.0)]
EMBEDDING_SIZE_FACTORS = [(), (1.0,), (1.0, 1.0, 1.0)]
RunnerOutput = tuple[list[int], str, SampleLogprobs | None]


class PromptWithMultiModalInput(NamedTuple):
    """Holds the multimodal input for a single test case."""

    prompts: list[str]
    image_data: PromptImageInput | None = None
    video_data: PromptVideoInput | None = None
    audio_data: PromptAudioInput | None = None


class VLMTestType(Enum):
    IMAGE = 1
    MULTI_IMAGE = 2
    EMBEDDING = 3
    VIDEO = 4
    AUDIO = 5
    CUSTOM_INPUTS = 6


class SizeType(Enum):
    SIZE_FACTOR = 1
    FIXED_SIZE = 2


class CustomTestOptions(NamedTuple):
    inputs: list[PromptWithMultiModalInput]
    limit_mm_per_prompt: dict[str, int]


class ImageSizeWrapper(NamedTuple):
    type: SizeType
    # A size factor is a wrapper of 0+ floats,
    # while a fixed size contains an iterable of integer pairs
    data: Iterable[float] | Iterable[tuple[int, int]]


class VLMTestInfo(NamedTuple):
    """Holds the configuration for 1+ tests for one model architecture."""

    models: list[str]
    test_type: VLMTestType | Iterable[VLMTestType]

    # Should be None only if this is a CUSTOM_INPUTS test
    prompt_formatter: Callable[[str], str] | None = None
    img_idx_to_prompt: Callable[[int], str] = lambda idx: "<image>\n"
    video_idx_to_prompt: Callable[[int], str] = lambda idx: "<video>\n"
    audio_idx_to_prompt: Callable[[int], str] = lambda idx: "<audio>\n"

    # Most models work on the single / multi-image prompts above, but in some
    # cases the log prob check fails, e.g., for paligemma. We allow passing
    # an override for the single image prompts / multi-image prompt for this
    # reason.
    single_image_prompts: Iterable[str] = SINGLE_IMAGE_BASE_PROMPTS
    multi_image_prompt: str = MULTI_IMAGE_BASE_PROMPT

    # Function for converting ImageAssets to image embeddings;
    # We need to define this explicitly for embedding tests
    convert_assets_to_embeddings: (
        Callable[[ImageTestAssets], list[torch.Tensor]] | None
    ) = None

    # Exposed options for vLLM runner; we change these in a several tests,
    # but the defaults are derived from VllmRunner & the engine defaults
    # These settings are chosen to avoid OOMs when running in the CI
    enforce_eager: bool = True
    max_model_len: int = 1024
    max_num_seqs: int = 256
    runner: RunnerOption = "auto"
    tensor_parallel_size: int = 1
    vllm_runner_kwargs: dict[str, Any] | None = None

    # Optional callable which gets a list of token IDs from the model tokenizer
    get_stop_token_ids: Callable[[AnyTokenizer], list[int]] | None = None
    # Optional list of strings to stop generation, useful when stop tokens are
    # not special tokens in the tokenizer
    stop_str: list[str] | None = None

    # Exposed options for HF runner
    hf_model_kwargs: dict[str, Any] | None = None
    # Indicates we should explicitly pass the EOS from the tokenizer
    use_tokenizer_eos: bool = False
    auto_cls: type[_BaseAutoModelClass] = AutoModelForCausalLM
    patch_hf_runner: Callable[[HfRunner], HfRunner] | None = None

    # Post processors that if defined, will run oun the outputs of the
    # vLLM and HF runner, respectively (useful for sanitization, etc).
    vllm_output_post_proc: Callable[[RunnerOutput, str], Any] | None = None
    hf_output_post_proc: Callable[[RunnerOutput, str], Any] | None = None

    # Consumes the output of the callables above and checks if they're equal
    comparator: Callable[..., None] = check_logprobs_close

    # Default expandable params per test; these defaults can be overridden in
    # instances of this object; the complete set of test cases for the model
    # is all combinations of .models + all fields below
    max_tokens: int = 128
    num_logprobs: int = 5
    dtype: str = "auto"
    distributed_executor_backend: str | None = None
    # Only expanded in video tests
    num_video_frames: int | tuple[int] = 16
    needs_video_metadata: bool = False

    # Fixed image sizes / image size factors; most tests use image_size_factors
    # The values provided for these two fields will be stacked and expanded
    # such that each model will consider each image size factor / image size
    # once per tests (much like concatenating and wrapping in one parametrize
    # call)
    image_size_factors: Iterable[Iterable[float]] = IMAGE_SIZE_FACTORS
    image_sizes: Iterable[Iterable[tuple[int, int]]] | None = None

    # Hack for updating a prompt to take into a local path; currently only used
    # for Qwen-VL, which requires encoding the image path / url into the prompt
    # for HF runner
    prompt_path_encoder: (
        Callable[[PosixPath, str, list[ImageAsset] | ImageTestAssets], str] | None
    ) = None  # noqa: E501

    # Allows configuring a test to run with custom inputs
    custom_test_opts: list[CustomTestOptions] | None = None

    marks: list[MarkDecorator] | None = None

    def get_non_parametrized_runner_kwargs(self):
        """Returns a dictionary of expandable kwargs for items that are used
        in all test types, which are NOT used when creating the parametrized
        test cases.
        """
        return {
            "enforce_eager": self.enforce_eager,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "runner": self.runner,
            "tensor_parallel_size": self.tensor_parallel_size,
            "vllm_runner_kwargs": self.vllm_runner_kwargs,
            "hf_output_post_proc": self.hf_output_post_proc,
            "vllm_output_post_proc": self.vllm_output_post_proc,
            "auto_cls": self.auto_cls,
            "use_tokenizer_eos": self.use_tokenizer_eos,
            "comparator": self.comparator,
            "get_stop_token_ids": self.get_stop_token_ids,
            "hf_model_kwargs": self.hf_model_kwargs,
            "stop_str": self.stop_str,
            "patch_hf_runner": self.patch_hf_runner,
        }


class ExpandableVLMTestArgs(NamedTuple):
    """The expanded kwargs which correspond to a single test case."""

    model: str
    max_tokens: int
    num_logprobs: int
    dtype: str
    distributed_executor_backend: str | None
    # Sizes are used for everything except for custom input tests
    size_wrapper: ImageSizeWrapper | None = None
    # Video only
    num_video_frames: int | None = None
    needs_video_metadata: bool = False
    # Custom inputs only
    custom_test_opts: CustomTestOptions | None = None
