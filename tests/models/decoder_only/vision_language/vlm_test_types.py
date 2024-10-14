from enum import Enum
from pathlib import PosixPath
from typing import (Any, Callable, Dict, Iterable, List, NamedTuple, Optional,
                    Tuple, Type, Union)

import torch
from PIL.Image import Image
from transformers import AutoModelForCausalLM, BatchEncoding
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from vllm.sequence import SampleLogprobs
from vllm.utils import identity

from ....conftest import IMAGE_ASSETS, ImageAsset, _ImageAssets, HfRunner
from ...utils import check_logprobs_close

# meta image tag; will be replaced by the appropriate tag for the model
TEST_IMG_PLACEHOLDER = "<vlm_image>"
TEST_VIDEO_PLACEHOLDER = "<vlm_video>"

# yapf: disable
SINGLE_IMAGE_BASE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign": f"{TEST_IMG_PLACEHOLDER}What's the content of the image?",
    "cherry_blossom": f"{TEST_IMG_PLACEHOLDER}What is the season?",
})

MULTI_IMAGE_BASE_PROMPT = f"Image-1: {TEST_IMG_PLACEHOLDER}Image-2: {TEST_IMG_PLACEHOLDER}Describe the two images in detail.\n"  # noqa: E501
VIDEO_BASE_PROMPT = f"${TEST_VIDEO_PLACEHOLDER}why is this video funny?"


IMAGE_SIZE_FACTORS = ((), (1.0, ), (1.0, 1.0, 1.0), (0.25, 0.5, 1.0))
EMBEDDING_SIZE_FACTORS = ((), (1.0, ), (1.0, 1.0, 1.0))
VllmOutput = Tuple[List[int], str, Optional[SampleLogprobs]]
# yapf: enable


class VlmTestType(Enum):
    IMAGE = 1
    MULTI_IMAGE = 2
    EMBEDDING = 3
    VIDEO = 4  # TODO
    CUSTOM_INPUTS = 5


class SizeType(Enum):
    SIZE_FACTOR = 1
    FIXED_SIZE = 2


class CustomTestOptions(NamedTuple):
    inputs: List[Tuple[List[str], List[Union[List[Image], Image]]]]
    limit_mm_per_prompt: Dict[str, int]


class ImageSizeWrapper(NamedTuple):
    type: SizeType
    # A size factor is a wrapper of 0+ floats,
    # while a fixed size contains 2 integers
    data: Union[Iterable[float], Iterable[int]]


class VLMTestInfo(NamedTuple):
    models: Union[Iterable[str], str]
    # Should be None only if this is a CUSTOM_INPUTS test
    prompt_formatter: Optional[Callable]
    test_type: Union[VlmTestType, Iterable[VlmTestType]]
    # Indicates whether or not we need to run every case in a new process
    fork_new_process_for_each_test: bool = False

    img_idx_to_prompt: Callable = lambda idx: "<image>\n"
    video_idx_to_prompt: Callable = lambda idx: "<video>\n"

    # HACK - currently, this is exposed so that we can pass an override for the
    # prompt to paligemma so that we can match the existing prompt, because the
    # default prompt fails the logprobs check.
    # If a different value is passed here, it should be of length 2 so that it
    # can be zipped up with the same assets.
    single_image_prompts: Tuple[str, str] = SINGLE_IMAGE_BASE_PROMPTS

    # Function for converting ImageAssets to image embeddings;
    # We need to define this explicitly for embedding tests
    convert_assets_to_embeddings: Optional[Callable[[_ImageAssets],
                                                    torch.Tensor]] = None

    # Exposed options for vLLM runner; we change these in a several tests,
    # but the defaults are derived from VllmRunner & the engine defaults
    enforce_eager: bool = True
    max_model_len: int = 1024
    max_num_seqs: int = 256
    tensor_parallel_size: int = 1

    # Optional callable which gets a list of token IDs from the model tokenizer
    get_stop_token_ids: Optional[Callable] = None

    # Exposed options for HF runner
    model_kwargs: Optional[Dict[str, Any]] = None
    # Indicates we should explicitly pass the EOS from the tokeniezr
    use_tokenizer_eos: bool = False
    auto_cls: Type[_BaseAutoModelClass] = AutoModelForCausalLM
    # Callable to pass to the HF runner to run on inputs
    postprocess_inputs: Callable[[BatchEncoding], BatchEncoding] = identity
    patch_hf_runner: Optional[Callable[[HfRunner], HfRunner]] = None

    # Post processors that if defined, will run oun the outputs of the
    # vLLM and HF runner, respectively (useful for sanitization, etc).
    vllm_output_post_proc: Optional[Callable] = None
    hf_output_post_proc: Optional[Callable] = None
    comparator: Callable = check_logprobs_close
    # Default expandable params per test; these defaults can be overridden in
    # instances of this object; the complete set of test cases for the model
    # is all combinations of .models + all fields below
    max_tokens: Union[int, Tuple[int]] = 128
    num_logprobs: Union[int, Tuple[int]] = 5
    dtype: Union[str, Iterable[str]] = "half"
    distributed_executor_backend: Optional[Union[str, Iterable[str]]] = None
    # Only expanded in video tests
    num_video_frames: Union[int, Tuple[int]] = 16

    # Fixed image sizes / image size factors; most tests use image_size_factors
    # The values provided for these two fields will be stacked and expanded
    # such that each model will consider each image size factor / image size
    # once per tests (much like concatenating and wrapping in one parametrize
    # call)
    image_size_factors: Iterable[Iterable[float]] = IMAGE_SIZE_FACTORS
    image_sizes: Optional[Iterable[Tuple[int, int]]] = None

    # Hack for updating a prompt to take into a local path; currently only used
    # for Qwen-VL, which requires encoding the image path / url into the prompt
    # for HF runner
    prompt_path_encoder: Optional[
        Callable[[PosixPath, str, Union[List[ImageAsset], _ImageAssets]],
                 str]] = None  # noqa: E501

    # Allows configuring a test to run with custom inputs
    custom_test_opts: Optional[Iterable[CustomTestOptions]] = None

    # Toggle for disabling instances of this class
    skip: bool = True  # TODO - flip me after done testing...

    def get_non_parametrized_runner_kwargs(self):
        """Returns a dictionary of expandable kwargs for items that are used
        in all test types, which are NOT used when creating the parametrized
        test cases.
        """
        return {
            "enforce_eager": self.enforce_eager,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "hf_output_post_proc": self.hf_output_post_proc,
            "vllm_output_post_proc": self.vllm_output_post_proc,
            "auto_cls": self.auto_cls,
            "use_tokenizer_eos": self.use_tokenizer_eos,
            "postprocess_inputs": self.postprocess_inputs,
            "comparator": self.comparator,
            "get_stop_token_ids": self.get_stop_token_ids,
            "model_kwargs": self.model_kwargs,
            "patch_hf_runner": self.patch_hf_runner,
        }
