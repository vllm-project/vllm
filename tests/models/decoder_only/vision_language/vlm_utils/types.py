"""Types for writing multimodal model tests."""
from enum import Enum
from pathlib import PosixPath
from typing import (Any, Callable, Dict, Iterable, List, NamedTuple, Optional,
                    Tuple, Type, Union)

import torch
from PIL.Image import Image
from pytest import MarkDecorator
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from vllm.sequence import SampleLogprobs
from vllm.utils import identity

from .....conftest import IMAGE_ASSETS, HfRunner, ImageAsset, _ImageAssets
from ....utils import check_logprobs_close

# meta image tag; will be replaced by the appropriate tag for the model
TEST_IMG_PLACEHOLDER = "<vlm_image>"
TEST_VIDEO_PLACEHOLDER = "<vlm_video>"

# yapf: disable
SINGLE_IMAGE_BASE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign": f"{TEST_IMG_PLACEHOLDER}What's the content of the image?",
    "cherry_blossom": f"{TEST_IMG_PLACEHOLDER}What is the season?",
})

MULTI_IMAGE_BASE_PROMPT = f"Image-1: {TEST_IMG_PLACEHOLDER}Image-2: {TEST_IMG_PLACEHOLDER}Describe the two images in detail.\n"  # noqa: E501
VIDEO_BASE_PROMPT = f"{TEST_VIDEO_PLACEHOLDER}Why is this video funny?"


IMAGE_SIZE_FACTORS = [(), (1.0, ), (1.0, 1.0, 1.0), (0.25, 0.5, 1.0)]
EMBEDDING_SIZE_FACTORS = [(), (1.0, ), (1.0, 1.0, 1.0)]
RunnerOutput = Tuple[List[int], str, Optional[SampleLogprobs]]
# yapf: enable


class VLMTestType(Enum):
    IMAGE = 1
    MULTI_IMAGE = 2
    EMBEDDING = 3
    VIDEO = 4
    CUSTOM_INPUTS = 5


class SizeType(Enum):
    SIZE_FACTOR = 1
    FIXED_SIZE = 2


class CustomTestOptions(NamedTuple):
    inputs: List[Tuple[List[str], List[Union[List[Image], Image]]]]
    limit_mm_per_prompt: Dict[str, int]
    # kwarg to pass multimodal data in as to vllm/hf runner instances.
    runner_mm_key: str = "images"


class ImageSizeWrapper(NamedTuple):
    type: SizeType
    # A size factor is a wrapper of 0+ floats,
    # while a fixed size contains an iterable of integer pairs
    data: Union[Iterable[float], Iterable[Tuple[int, int]]]


class VLMTestInfo(NamedTuple):
    """Holds the configuration for 1+ tests for one model architecture."""

    models: Union[List[str]]
    test_type: Union[VLMTestType, Iterable[VLMTestType]]

    # Should be None only if this is a CUSTOM_INPUTS test
    prompt_formatter: Optional[Callable[[str], str]] = None
    img_idx_to_prompt: Callable[[int], str] = lambda idx: "<image>\n"
    video_idx_to_prompt: Callable[[int], str] = lambda idx: "<video>\n"

    # Most models work on the single / multi-image prompts above, but in some
    # cases the log prob check fails, e.g., for paligemma. We allow passing
    # an override for the single image prompts / multi-image prompt for this
    # reason.
    single_image_prompts: Iterable[str] = SINGLE_IMAGE_BASE_PROMPTS
    multi_image_prompt: str = MULTI_IMAGE_BASE_PROMPT

    # Function for converting ImageAssets to image embeddings;
    # We need to define this explicitly for embedding tests
    convert_assets_to_embeddings: Optional[Callable[[_ImageAssets],
                                                    torch.Tensor]] = None

    # Exposed options for vLLM runner; we change these in a several tests,
    # but the defaults are derived from VllmRunner & the engine defaults
    # These settings are chosen to avoid OOMs when running in the CI
    enforce_eager: bool = True
    max_model_len: int = 1024
    max_num_seqs: int = 256
    task: str = "auto"
    tensor_parallel_size: int = 1

    # Optional callable which gets a list of token IDs from the model tokenizer
    get_stop_token_ids: Optional[Callable[[AutoTokenizer], List[int]]] = None

    # Exposed options for HF runner
    model_kwargs: Optional[Dict[str, Any]] = None
    # Indicates we should explicitly pass the EOS from the tokeniezr
    use_tokenizer_eos: bool = False
    auto_cls: Type[_BaseAutoModelClass] = AutoModelForCausalLM
    # Callable to pass to the HF runner to run on inputs; for now, we also pass
    # the data type to input post processing, because almost all of the uses of
    # postprocess_inputs are to fix the data types of BatchEncoding values.
    postprocess_inputs: Callable[[BatchEncoding, str],
                                 BatchEncoding] = identity
    patch_hf_runner: Optional[Callable[[HfRunner], HfRunner]] = None

    # Post processors that if defined, will run oun the outputs of the
    # vLLM and HF runner, respectively (useful for sanitization, etc).
    vllm_output_post_proc: Optional[Callable[[RunnerOutput, str], Any]] = None
    hf_output_post_proc: Optional[Callable[[RunnerOutput, str], Any]] = None

    # Consumes the output of the callables above and checks if they're equal
    comparator: Callable[..., None] = check_logprobs_close

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
    image_sizes: Optional[Iterable[Iterable[Tuple[int, int]]]] = None

    # Hack for updating a prompt to take into a local path; currently only used
    # for Qwen-VL, which requires encoding the image path / url into the prompt
    # for HF runner
    prompt_path_encoder: Optional[
        Callable[[PosixPath, str, Union[List[ImageAsset], _ImageAssets]],
                 str]] = None  # noqa: E501

    # Allows configuring a test to run with custom inputs
    custom_test_opts: Optional[List[CustomTestOptions]] = None

    marks: Optional[List[MarkDecorator]] = None

    def get_non_parametrized_runner_kwargs(self):
        """Returns a dictionary of expandable kwargs for items that are used
        in all test types, which are NOT used when creating the parametrized
        test cases.
        """
        return {
            "enforce_eager": self.enforce_eager,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "task": self.task,
            "tensor_parallel_size": self.tensor_parallel_size,
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


class ExpandableVLMTestArgs(NamedTuple):
    """The expanded kwargs which correspond to a single test case."""
    model: str
    max_tokens: int
    num_logprobs: int
    dtype: str
    distributed_executor_backend: Optional[str]
    # Sizes are used for everything except for custom input tests
    size_wrapper: Optional[ImageSizeWrapper] = None
    # Video only
    num_video_frames: Optional[int] = None
    # Custom inputs only
    custom_test_opts: Optional[CustomTestOptions] = None
