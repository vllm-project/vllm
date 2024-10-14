"""Common utility functions relating to different models that are useful
for running tests, e.g., output sanitizers to make HF outputs more easily
comparable to vLLM, etc.
"""
import itertools
import re
from pathlib import PosixPath
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

from transformers import AutoConfig, AutoTokenizer, BatchEncoding

from vllm.multimodal.utils import rescale_image_size
from vllm.sequence import SampleLogprobs
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE

from ....conftest import IMAGE_ASSETS, ImageAsset, _ImageAssets
from .vlm_test_types import (EMBEDDING_SIZE_FACTORS, TEST_IMG_PLACEHOLDER,
                             ImageSizeWrapper, SizeType, VllmOutput,
                             VLMTestInfo, VlmTestType)


####### vLLM output processors functions
def blip2_vllm_to_hf_output(vllm_output: VllmOutput, model: str):
    """Sanitize vllm output [blip2 models] to be comparable with hf output."""
    _, output_str, out_logprobs = vllm_output

    hf_output_str = output_str + "\n"

    tokenizer = AutoTokenizer.from_pretrained(model)
    hf_output_ids = tokenizer.encode(hf_output_str)
    assert hf_output_ids[0] == tokenizer.bos_token_id
    hf_output_ids = hf_output_ids[1:]

    return hf_output_ids, hf_output_str, out_logprobs


def fuyu_vllm_to_hf_output(vllm_output: VllmOutput, model: str):
    """Sanitize vllm output [fuyu models] to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    hf_output_str = output_str.lstrip() + "|ENDOFTEXT|"

    return output_ids, hf_output_str, out_logprobs


def qwen_vllm_to_hf_output(vllm_output: VllmOutput, model: str):
    """Sanitize vllm output [qwen models] to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    hf_output_str = output_str + "<|endoftext|>"

    return output_ids, hf_output_str, out_logprobs


def llava_image_vllm_to_hf_output(vllm_output: VllmOutput, model: str):
    config = AutoConfig.from_pretrained(model)
    mm_token_id = config.image_token_index
    return _llava_vllm_to_hf_output(vllm_output, model, mm_token_id)


def llava_video_vllm_to_hf_output(vllm_output: VllmOutput, model: str):
    config = AutoConfig.from_pretrained(model)
    mm_token_id = config.video_token_index
    return _llava_vllm_to_hf_output(vllm_output, model, mm_token_id)


def _llava_vllm_to_hf_output(vllm_output: VllmOutput, model: str, mm_token_id: int):
    """Sanitize vllm output [Llava models] to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_token_id = tokenizer.eos_token_id

    hf_output_ids = [
        token_id for idx, token_id in enumerate(output_ids)
        if token_id != mm_token_id or output_ids[idx - 1] != mm_token_id
    ]

    assert output_str[0] == " "
    hf_output_str = output_str[1:]
    if hf_output_ids[-1] == eos_token_id:
        hf_output_str = hf_output_str + tokenizer.decode(eos_token_id)

    return hf_output_ids, hf_output_str, out_logprobs


def phi3v_vllm_to_hf_output(vllm_output: VllmOutput, model: str):
    """Sanitize vllm output [phi3v] to be comparable with hf output."""
    _, output_str, out_logprobs = vllm_output

    output_str_without_image = re.sub(r"(<\|image_\d+\|>)+", "", output_str)
    assert output_str_without_image[0] == " "
    output_str_without_image = output_str_without_image[1:]

    hf_output_str = output_str_without_image + "<|end|><|endoftext|>"

    tokenizer = AutoTokenizer.from_pretrained(model)
    hf_output_ids = tokenizer.encode(output_str_without_image)
    assert hf_output_ids[0] == 1
    hf_output_ids = hf_output_ids[1:]

    return hf_output_ids, hf_output_str, out_logprobs


def paligemma_vllm_to_hf_output(vllm_output: VllmOutput, model: str):
    """Sanitize vllm output to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    config = AutoConfig.from_pretrained(model)
    image_token_id = config.image_token_index

    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_token_id = tokenizer.eos_token_id

    hf_output_ids = [
        token_id for idx, token_id in enumerate(output_ids)
        if token_id != image_token_id or output_ids[idx - 1] != image_token_id
    ]

    hf_output_str = output_str

    if hf_output_ids[-1] == eos_token_id:
        hf_output_str = hf_output_str + tokenizer.decode(eos_token_id)

    return hf_output_ids, hf_output_str, out_logprobs


####### Post-processors for HF outputs
def minicmpv_trunc_hf_output(hf_output: Tuple[List[int], str,
                                              Optional[SampleLogprobs]],
                             model: str):
    output_ids, output_str, out_logprobs = hf_output
    if output_str.endswith("<|eot_id|>"):
        output_str = output_str.split("<|eot_id|>")[0]
    return output_ids, output_str, out_logprobs


####### Functions for converting image assets to embeddings
def get_llava_embeddings(image_assets: _ImageAssets):
    return [asset.image_embeds for asset in image_assets]


####### postprocessors to run on HF BatchEncoding
# NOTE: It would be helpful to rewrite this to be configured in the test info,
# but built inside of the test so that we don't need to specify the dtype twice
def get_key_type_post_processor(
        hf_inp_key: str,
        dtype: str) -> Callable[[BatchEncoding], BatchEncoding]:
    """Gets a handle to a post processor which converts a """
    torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]

    def process(hf_inputs: BatchEncoding):
        hf_inputs[hf_inp_key] = hf_inputs[hf_inp_key].to(torch_dtype)
        return hf_inputs

    return process


def wrap_inputs_post_processor(hf_inputs: BatchEncoding) -> BatchEncoding:
    return BatchEncoding({"model_inputs": hf_inputs})


####### Prompt path encoders for models that need models on disk
def qwen_prompt_path_encoder(
        tmp_path: PosixPath, prompt: str, assets: Union[List[ImageAsset],
                                                        _ImageAssets]) -> str:
    """Given a temporary dir path, export one or more image assets into the
    tempdir & replace its contents with the local path to the string so that
    the HF version of Qwen-VL can resolve the path and load the image in its
    forward() call.

    Args:
        tmp_path: Tempdir for test under consideration.
        prompt: Prompt with image placeholders.
        assets: List of image assets whose len equals the num placeholders.
    """
    # Ensure that the number of placeholders matches the number of assets;
    # If this is not true, the test is probably written incorrectly.
    assert prompt.count("<img></img>") == len(assets)

    # Replace the placeholders with local paths to the exported assets
    for asset in assets:
        image_tmp_path = tmp_path / f"{asset.name}.jpg"
        asset.pil_image.save(image_tmp_path)
        prompt = prompt.replace(
            "<img></img>",
            f"<img>{image_tmp_path}</img>",
            1,
        )
    return prompt


####### Utils for model-agnostic prompt manipulation / case filtering
# Most of these help us handle image tags and configure things
# that would normally be handled by parametrize(), since we want
# to be able to adapt settings like num_logprobs on a per-model basis


def replace_test_img_placeholders(prompt: str,
                                  img_idx_to_prompt: Callable) -> str:
    """Given a prompt, replaces each TEST_IMG_PLACEHOLDER with the
    model-specific image prompt.
    """
    prompt_segments = prompt.split(TEST_IMG_PLACEHOLDER)
    img_prompt = prompt_segments[0]
    for placeholder_idx, next_seg in enumerate(prompt_segments[1:], start=1):
        img_prompt += img_idx_to_prompt(placeholder_idx)
        img_prompt += next_seg
    return img_prompt


def get_model_prompts(base_prompts: Union[List[str], Tuple[str]],
                      img_idx_to_prompt: Callable,
                      prompt_formatter: Callable) -> List[str]:
    """Given a model-agnostic base prompt and test configuration for a model(s)
    to be tested, update the image placeholders and apply the prompt formatting
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
        # Replace the image placeholders in the base prompt with
        # the correct ones for the model that we are testing
        base_prompt_with_imgs = replace_test_img_placeholders(
            base_prompt, img_idx_to_prompt)
        # Apply the prompt formatter to wrap the base prompt with
        # the correct img placeholders to get the model test prompt
        model_prompt = prompt_formatter(base_prompt_with_imgs)
        model_prompts.append(model_prompt)
    return model_prompts


def get_filtered_test_settings(test_settings: Dict[str, VLMTestInfo],
                               test_type: VlmTestType):
    filtered_test_settings = {}
    for test_name, test_info in test_settings.items():
        # Skip if it's explicitly disabled
        if test_info.skip:
            continue
        # Otherwise check if the test has the right type & keep if it does
        if test_type == test_info.test_type or (
                isinstance(test_info.test_type, Iterable)
                and test_type in test_info.test_type):
            # Embedding tests need to have a conversion func in their test info
            if test_type == VlmTestType.EMBEDDING:
                assert test_info.convert_assets_to_embeddings is not None
            # Custom test inputs need to explicitly define the mm limit/inputs
            if test_type == VlmTestType.CUSTOM_INPUTS:
                assert test_info.custom_test_opts is not None

            filtered_test_settings[test_name] = test_info
    return filtered_test_settings


def get_parametrized_options(test_settings: Dict[str, VLMTestInfo],
                             test_type: VlmTestType):
    """Converts all of our VLMTestInfo into an expanded list of parameters."""
    test_settings = get_filtered_test_settings(test_settings, test_type)

    # Ensure that something is wrapped as an iterable it's not already
    ensure_wrapped = lambda e: e if isinstance(e, (list, tuple)) else (e, )

    def get_model_type_cases(model_type: str, test_info: VLMTestInfo):
        # This is essentially the same as nesting a bunch of mark.parametrize
        # decorators, but we do it programmatically to allow overrides for on
        # a per-model basis, while still being able to execute each of these
        # as individual test cases in pytest.
        iterables = [
            ensure_wrapped(model_type),
            ensure_wrapped(test_info.models),
            ensure_wrapped(test_info.max_tokens),
            ensure_wrapped(test_info.num_logprobs),
            ensure_wrapped(test_info.dtype),
        ]
        # No sizes passed for custom inputs, since inputs are directly provided
        if test_type != VlmTestType.CUSTOM_INPUTS:
            iterables.append(get_wrapped_test_sizes(test_info, test_type))
        return list(itertools.product(*iterables))

    # Get a list per model type, where each entry contains a tuple of all of
    # that model type's cases, then flatten them into the top level so that
    # we can consume them in one mark.parametrize call.
    cases_by_model_type = [
        get_model_type_cases(model_type, test_info)
        for model_type, test_info in test_settings.items()
    ]
    return list(itertools.chain(*cases_by_model_type))


def get_wrapped_test_sizes(
        test_info: VLMTestInfo,
        test_type: VlmTestType) -> Tuple[ImageSizeWrapper, ...]:
    """Given a test info which may have size factors or fixed sizes, wrap them
    and combine them into an iterable, each of which will be used in parameter
    expansion.

    Args:
        test_info: Test configuration to be expanded.
        test_type: The type of test being filtered for.
    """
    # If it is an embedding test, we always use the EMBEDDING_SIZE_FACTORS
    if test_type == VlmTestType.EMBEDDING:
        return tuple([
            ImageSizeWrapper(type=SizeType.SIZE_FACTOR, data=factor)
            for factor in EMBEDDING_SIZE_FACTORS
        ])
    # Custom inputs have preprocessed inputs
    elif test_type == VlmTestType.CUSTOM_INPUTS:
        return tuple()

    size_factors = test_info.image_size_factors \
        if test_info.image_size_factors else []
    fixed_sizes = test_info.image_sizes \
        if test_info.image_sizes else []

    wrapped_factors = [
        ImageSizeWrapper(type=SizeType.SIZE_FACTOR, data=factor)
        for factor in size_factors
    ]

    wrapped_sizes = [
        ImageSizeWrapper(type=SizeType.FIXED_SIZE, data=size)
        for size in fixed_sizes
    ]

    return tuple(wrapped_factors + wrapped_sizes)


def multi_image_multi_aspect_ratio_inputs_llava(is_llava):
    """Builds inputs for multi-image (varied sizes/aspect ratio) testing."""
    stop_sign = IMAGE_ASSETS[0].pil_image
    cherry_blossom = IMAGE_ASSETS[1].pil_image
    llava_formatter = lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:"
    llava_next_formatter = lambda img_prompt: f"[INST] {img_prompt} [/INST]"
    # Apply the selected formatter to the base prompts
    img_prompts = [
        "<image><image>\nDescribe 2 images.",
        "<image><image>\nDescribe 2 images.",
        "<image><image><image><image>\nDescribe 4 images.",  # noqa: E501
        "<image>\nWhat is the season?",
    ]
    formatter = llava_formatter if is_llava else llava_next_formatter 
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


### Utilities for local export
def export_test(model,
                size_info,
                export_info,
                is_new,
                write_dir="/u/brooks/vllm/compare_tests",
                terminate_test=False):
    import json
    import os
    import sys
    if size_info is None:
        size_info = ("custom", )
    default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
    size_str = "_".join([str(x) for x in size_info])
    size_str = size_str if size_str else 'NONE'
    filename = f"{model.split('/')[-1]}_sf_{size_str}.json"
    subdir_name = "common" if is_new else "legacy"
    subdir = os.path.join(write_dir, subdir_name)
    full_path = os.path.join(subdir, filename)

    if not os.path.isdir(subdir):
        os.mkdir(subdir)
    if os.path.exists(full_path):
        # Delete it if already exists
        os.remove(full_path)

    with open(full_path, "w") as f:
        json.dump(export_info, f, sort_keys=True, indent=4, default=default)
    if terminate_test:
        sys.exit(0)
