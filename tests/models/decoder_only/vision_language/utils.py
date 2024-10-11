"""Common utility functions relating to different models that are useful
for running tests, e.g., output sanitizers to make HF outputs more easily
comparable to vLLM, etc.
"""
import re
import itertools
from pathlib import PosixPath
from typing import Tuple, List, Callable, Union, Dict, Optional
from transformers import AutoTokenizer, AutoConfig, BatchEncoding

from ....conftest import _ImageAssets
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.sequence import SampleLogprobs
from vlm_test_types import (VllmOutput, VLMTestInfo, VlmTestType,
                            EMBEDDING_SIZE_FACTORS, TEST_IMG_PLACEHOLDER)

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


def llava_vllm_to_hf_output(vllm_output: VllmOutput, model: str):
    """Sanitize vllm output [Llava models] to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    config = AutoConfig.from_pretrained(model)
    image_token_id = config.image_token_index

    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_token_id = tokenizer.eos_token_id

    hf_output_ids = [
        token_id for idx, token_id in enumerate(output_ids)
        if token_id != image_token_id or output_ids[idx - 1] != image_token_id
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
        dtype: str
    ) -> Callable[[BatchEncoding], BatchEncoding]:
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
    tmp_path: PosixPath,
    prompt: str,
    assets: Union[_ImageAssets, List[_ImageAssets]]
) -> str:
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
    # Filter based on type of test; assume single image is supported everywhere
    filter_func = lambda _: True

    if test_type == VlmTestType.MULTI_IMAGE:
        # Multi-image requires explicit enablement
        filter_func = lambda info: info.supports_multi_image
    elif test_type == VlmTestType.EMBEDDING:
        # Embedding requires an explicit func to get embeddings to pass to vLLM
        filter_func = lambda info: info.convert_assets_to_embeddings is not None

    # Drop anything that is either unimplemented for the test config/disabled
    test_settings = {
        model_type: test_info
        for model_type, test_info in test_settings.items()
        if filter_func(test_info) and not test_info.skip
    }
    return test_settings


def get_parametrized_options(test_settings: Dict[str, VLMTestInfo],
                             test_type: VlmTestType):
    """Converts all of our VLMTestInfo into an expanded list of parameters."""
    test_settings = get_filtered_test_settings(test_settings, test_type)

    # Ensure that something is wrapped as an iterable it's not already
    ensure_wrapped = lambda e: e if isinstance(e, (list, tuple)) else (e, )

    def get_model_type_cases(model_type: str, test_info: VLMTestInfo):
        size_factors = test_info.image_size_factors
        # All embedding tests use the same size factors; we currently
        # don't configure this per test since embeddings can't be
        # heterogeneous, etc
        if test_type == VlmTestType.EMBEDDING:
            size_factors = EMBEDDING_SIZE_FACTORS

        # This is essentially the same as nesting a bunch of mark.parametrize
        # decorators, but we do it programmatically to allow overrides for on
        # a per-model basis, while still being able to execute each of these
        # as individual test cases in pytest.
        return list(
            itertools.product(
                ensure_wrapped(model_type),
                ensure_wrapped(test_info.models),
                ensure_wrapped(test_info.max_tokens),
                ensure_wrapped(test_info.num_logprobs),
                ensure_wrapped(test_info.dtype),
                ensure_wrapped(size_factors),
            ))

    # Get a list per model type, where each entry contains a tuple of all of
    # that model type's cases, then flatten them into the top level so that
    # we can consume them in one mark.parametrize call.
    cases_by_model_type = [
        get_model_type_cases(model_type, test_info)
        for model_type, test_info in test_settings.items()
    ]
    return list(itertools.chain(*cases_by_model_type))
