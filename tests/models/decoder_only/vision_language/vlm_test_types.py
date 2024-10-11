from vllm.sequence import SampleLogprobs

from enum import Enum
from pathlib import PosixPath
from typing import Union, Tuple, Callable, Optional, Dict, Any, NamedTuple, Type, List
import torch
from transformers import AutoModelForCausalLM, BatchEncoding
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from ...utils import check_logprobs_close, identity
from ....conftest import _ImageAssets, IMAGE_ASSETS

# meta image tag; will be replaced by the appropriate tag for the model
TEST_IMG_PLACEHOLDER = "<vlm_image>"

SINGLE_IMAGE_BASE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign": f"{TEST_IMG_PLACEHOLDER}What's the content of the image?",
    "cherry_blossom": f"{TEST_IMG_PLACEHOLDER}What is the season?",
})
MULTI_IMAGE_BASE_PROMPT = f"Image-1: {TEST_IMG_PLACEHOLDER}Image-2: {TEST_IMG_PLACEHOLDER}Describe the two images in detail.\n"  # noqa: E501

IMAGE_SIZE_FACTORS=((), (1.0,), (1.0, 1.0, 1.0), (0.25, 0.5, 1.0))
EMBEDDING_SIZE_FACTORS=((), (1.0,), (1.0, 1.0, 1.0),)

VllmOutput = Tuple[List[int], str, Optional[SampleLogprobs]]

class VlmTestType(Enum):
    IMAGE = 1
    MULTI_IMAGE = 2
    EMBEDDING = 3
    VIDEO = 4 # TODO
    QUANTIZED_IMAGE = 5 # TODO

class SizeType: # TODO - integate this
    SIZE_FACTOR = 1
    FIXED_SIZE = 2

class ImageSizeWrapper:
    type: SizeType
    # A size factor is a wrapper of 0+ floats,
    # while a fixed size contains 2 integers
    data: Union[Tuple[float], Tuple[int ,int]]


class VLMTestInfo(NamedTuple):
    models: Union[Tuple[str], str]
    prompt_formatter: Callable
    img_idx_to_prompt: Callable = lambda idx: "<image>\n"

    # TODO - Overrides for single / multi-image prompts, respectively
    single_image_prompt: Tuple[str] = None
    multi_image_prompt: str = None

    # Function for converting ImageAssets to image embeddings; if a VLMTestInfo
    # object defines this, we run a separate test for embedding with
    # size_factors 
    convert_assets_to_embeddings: Callable[[_ImageAssets], torch.Tensor] = None
    supports_multi_image: bool = False

    # Exposed options for vLLM runner; we change these in a several tests,
    # but the defaults are derived from VllmRunner & the engine defaults
    tensor_parallel_size: int = 1
    enforce_eager: bool = True
    max_model_len: int = 1024
    max_num_seqs: int = 256
    # Optional callable which gets a list of token IDs from the model tokenizer
    get_stop_token_ids: Optional[Callable] = None

    # Exposed options for HF runner
    model_kwargs: Optional[Dict[str, Any]] = None
     # Indicates we should explicitly pass the EOS from the tokeniezr
    use_tokenizer_eos: bool = False
    auto_cls: Type[_BaseAutoModelClass] = AutoModelForCausalLM
    # Callable to pass to the HF runner to run on inputs 
    postprocess_inputs: Callable[[BatchEncoding], BatchEncoding] = identity

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
    dtype: Union[str] = "half"

    # Fixed image sizes / image size factors; most tests use image_size_factors
    # The values provided for these two fields will be stacked and expanded
    # such that each model will consider each image size factor / image size
    # once per tests (much like concatenating and wrapping in one parametrize
    # call)
    image_size_factors: Tuple[float] = IMAGE_SIZE_FACTORS
    image_sizes: Tuple[float] = None

    # Hack for updating a prompt to take into a local path; currently only used
    # for Qwen-VL, which requires encoding the image path / url into the prompt
    # for HF runner
    prompt_path_encoder: Optional[Callable[[PosixPath, str, List[_ImageAssets]], str]] = None  # noqa: E501

    # Toggle for disabling instances of this class
    skip: bool = True
