"""Common utility functions relating to different models that are useful
for running tests, e.g., output sanitizers to make HF outputs more easily
comparable to vLLM, wrappers for running tests for different modalities, etc.
"""
import itertools
import re
import types
from pathlib import PosixPath
from typing import (Any, Callable, Dict, Iterable, List, Optional, Tuple, Type,
                    Union)

import torch
from PIL.Image import Image
from transformers import AutoConfig, AutoTokenizer, BatchEncoding
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from vllm.multimodal.utils import (rescale_image_size, rescale_video_size,
                                   resize_video, sample_frames_from_video)
from vllm.sequence import SampleLogprobs
from vllm.transformers_utils.tokenizer import patch_padding_side
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE

from ....conftest import (IMAGE_ASSETS, HfRunner, ImageAsset, VllmRunner,
                          _ImageAssets, _VideoAssets)
from .vlm_test_types import (EMBEDDING_SIZE_FACTORS, MULTI_IMAGE_BASE_PROMPT,
                             SINGLE_IMAGE_BASE_PROMPTS, TEST_IMG_PLACEHOLDER,
                             TEST_VIDEO_PLACEHOLDER, VIDEO_BASE_PROMPT,
                             CustomTestOptions, ImageSizeWrapper, SizeType,
                             VllmOutput, VLMTestInfo, VLMTestType)


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


def _llava_vllm_to_hf_output(vllm_output: VllmOutput, model: str,
                             mm_token_id: int):
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


def llava_onevision_vllm_to_hf_output(vllm_output: VllmOutput, model: str):
    """Sanitize vllm output [llava-onevision] to compare with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    config = AutoConfig.from_pretrained(model)
    video_token_id = config.video_token_index

    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_token_id = tokenizer.eos_token_id

    hf_output_ids = [
        token_id for idx, token_id in enumerate(output_ids)
        if token_id != video_token_id or output_ids[idx - 1] != video_token_id
    ]

    hf_output_str = output_str
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


####### Model-specific HuggingFace runner patchers
def glm_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patches and returns an instance of the HfRunner to use for GLM4."""
    hf_processor = hf_model.processor
    patch_padding_side(hf_processor)

    def processor(*args, text="", images=None, **kwargs):
        if images is None:
            return hf_processor(*args, **kwargs)

        return hf_processor.apply_chat_template(
            [{
                "role": "user",
                "image": images,
                "content": text
            }],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            **kwargs,
        )

    hf_model.processor = processor
    hf_model.model.get_output_embeddings = lambda: \
        hf_model.model.transformer.output_layer
    return hf_model


def internvl_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patches and returns an instance of the HfRunner to use for InternVL."""

    class InternVLProcessor:
        """A simple processor for InternVL2 which misses a processor."""

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
            from vllm.model_executor.models.internvl import (
                IMG_CONTEXT, IMG_END, IMG_START, image_to_pixel_values)
            images = [images] if isinstance(images, Image) else images
            pixel_values = [
                image_to_pixel_values(image, self.image_size, self.min_num,
                                      self.max_num,
                                      self.use_thumbnail).to(self.dtype)
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

    img_context_token_id = hf_model.tokenizer.convert_tokens_to_ids(
        "<IMG_CONTEXT>")
    hf_model.model.img_context_token_id = img_context_token_id
    hf_model.processor = InternVLProcessor(hf_model)
    hf_model.model.get_output_embeddings = lambda: \
        hf_model.model.language_model.get_output_embeddings()
    hf_model.model.generate = types.MethodType(_internvl_generate,
                                               hf_model.model)
    return hf_model


def _internvl_generate(
    self,
    pixel_values: torch.FloatTensor,
    input_ids: torch.FloatTensor,
    attention_mask: Optional[torch.LongTensor] = None,
    **generate_kwargs,
) -> torch.LongTensor:
    """Generate method for InternVL2 model without fixed use_cache."""
    assert self.img_context_token_id is not None
    vit_embeds = self.extract_feature(pixel_values)
    input_embeds = self.language_model.get_input_embeddings()(input_ids)
    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    input_ids = input_ids.reshape(B * N)
    selected = (input_ids == self.img_context_token_id)
    assert selected.sum() != 0
    input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

    input_embeds = input_embeds.reshape(B, N, C)

    return self.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        **generate_kwargs,
    )


####### Utils for model-agnostic prompt manipulation / case filtering
# Most of these help us handle mm tags and configure things
# that would normally be handled by parametrize(), since we want
# to be able to adapt settings like num_logprobs on a per-model basis
def replace_test_placeholder(prompt: str, img_idx_to_prompt: Callable[[int],
                                                                      str],
                             test_placeholder: str) -> str:
    """Given a prompt, replaces each TEST_IMG_PLACEHOLDER with the
    model-specific image prompt.
    """
    prompt_segments = prompt.split(test_placeholder)
    img_prompt = prompt_segments[0]
    for placeholder_idx, next_seg in enumerate(prompt_segments[1:], start=1):
        img_prompt += img_idx_to_prompt(placeholder_idx)
        img_prompt += next_seg
    return img_prompt


def get_model_prompts(base_prompts: Iterable[str],
                      img_idx_to_prompt: Optional[Callable[[int], str]],
                      video_idx_to_prompt: Optional[Callable[[int], str]],
                      prompt_formatter: Callable) -> List[str]:
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

        # Apply the prompt formatter to wrap the base prompt with
        # the correct media placeholders to get the model test prompt
        model_prompt = prompt_formatter(base_prompt)
        model_prompts.append(model_prompt)
    return model_prompts


def get_filtered_test_settings(test_settings: Dict[str, VLMTestInfo],
                               test_type: VLMTestType,
                               fork_per_test: bool) -> Dict[str, VLMTestInfo]:
    """Given the dict of potential test settings to run, return a subdict
    of tests who have the current test type enabled, with the matching val for
    fork_per_test.
    """

    def matches_test_type(test_info: VLMTestInfo, test_type: VLMTestType):
        return test_info.test_type == test_type or (
            isinstance(test_info.test_type, Iterable)
            and test_type in test_info.test_type)

    filtered_test_settings = {}
    for test_name, test_info in test_settings.items():
        # Skip if it's explicitly disabled
        if test_info.skip:
            continue
        # Otherwise check if the test has the right type & keep if it does
        if matches_test_type(test_info, test_type):
            # Embedding tests need to have a conversion func in their test info
            if matches_test_type(test_info, VLMTestType.EMBEDDING):
                assert test_info.convert_assets_to_embeddings is not None
            # Custom test inputs need to explicitly define the mm limit/inputs
            if matches_test_type(test_info, VLMTestType.CUSTOM_INPUTS):
                assert (test_info.custom_test_opts is not None
                        and isinstance(test_info.custom_test_opts, Iterable))
            # For all types besides custom inputs, we need a prompt formatter
            else:
                assert test_info.prompt_formatter is not None

            # Everything looks okay; keep if this is has correct proc handling
            if test_info.fork_new_process_for_each_test == fork_per_test:
                filtered_test_settings[test_name] = test_info

    return filtered_test_settings


def get_parametrized_options(test_settings: Dict[str, VLMTestInfo],
                             test_type: VLMTestType,
                             fork_new_process_for_each_test: bool):
    """Converts all of our VLMTestInfo into an expanded list of parameters.
    This is similar to nesting pytest parametrize calls, but done directly
    through an itertools product so that each test can set things like
    size factors etc, while still running in isolated test cases.
    """
    test_settings = get_filtered_test_settings(test_settings, test_type,
                                               fork_new_process_for_each_test)

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
            ensure_wrapped(test_info.distributed_executor_backend),
        ]
        # num_frames is video only
        if test_type == VLMTestType.VIDEO:
            iterables.append(ensure_wrapped(test_info.num_video_frames))

        # No sizes passed for custom inputs, since inputs are directly provided
        if test_type != VLMTestType.CUSTOM_INPUTS:
            iterables.append(get_wrapped_test_sizes(test_info, test_type))
        #Otherwise expand the custom test options instead
        else:
            if test_info.custom_test_opts is None:
                raise ValueError("Test has type CUSTOM_INPUTS, but none given")

            iterables.append(test_info.custom_test_opts)
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
        test_type: VLMTestType) -> Tuple[ImageSizeWrapper, ...]:
    """Given a test info which may have size factors or fixed sizes, wrap them
    and combine them into an iterable, each of which will be used in parameter
    expansion.

    Args:
        test_info: Test configuration to be expanded.
        test_type: The type of test being filtered for.
    """
    # If it is an embedding test, we always use the EMBEDDING_SIZE_FACTORS
    if test_type == VLMTestType.EMBEDDING:
        return tuple([
            ImageSizeWrapper(type=SizeType.SIZE_FACTOR, data=factor)
            for factor in EMBEDDING_SIZE_FACTORS
        ])
    # Custom inputs have preprocessed inputs
    elif test_type == VLMTestType.CUSTOM_INPUTS:
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


def multi_image_multi_aspect_ratio_inputs(formatter):
    """Builds inputs for multi-image (varied sizes/aspect ratio) testing.
    
    Args:
        is_llava: Indicates whether we should use llava or llava-next format.
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


####### Useful builders for setting up different types of tests


def build_single_image_inputs_from_test_info(
        test_info: VLMTestInfo,
        image_assets: _ImageAssets,
        size_wrapper: ImageSizeWrapper,
        tmp_path: Optional[PosixPath] = None):
    if test_info.prompt_formatter is None:
        raise ValueError(
            "Prompt formatter must be set to build single image inputs")

    model_prompts = get_model_prompts(test_info.single_image_prompts,
                                      test_info.img_idx_to_prompt,
                                      test_info.video_idx_to_prompt,
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


def build_single_image_inputs(images, model_prompts,
                              size_wrapper: ImageSizeWrapper):
    # For every image / prompt pair, get a pair containing two lists of
    # length size_factors, where the first contains duplicates of the model
    # prompt [str], and the second contains copies of the image after being
    # scaled by one of the size factors.
    #
    # NOTE: rescaling preserves the image aspect ratio.
    return [(
        [prompt for _ in size_wrapper.data],
        [
            apply_image_size_scaling(image, size, size_wrapper.type)
            for size in size_wrapper.data
        ],
    ) for image, prompt in zip(images, model_prompts)]


def build_multi_image_inputs_from_test_info(
        test_info: VLMTestInfo,
        image_assets: _ImageAssets,
        size_wrapper: ImageSizeWrapper,
        tmp_path: Optional[PosixPath] = None):
    if test_info.prompt_formatter is None:
        raise ValueError(
            "Prompt formatter must be set to build multi image inputs")

    model_prompts = get_model_prompts([MULTI_IMAGE_BASE_PROMPT],
                                      test_info.img_idx_to_prompt,
                                      test_info.video_idx_to_prompt,
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


def build_multi_image_inputs(image_lists, model_prompts,
                             size_wrapper: ImageSizeWrapper):
    return [(
        [prompt for _ in size_wrapper.data],
        [[
            apply_image_size_scaling(image, size, size_wrapper.type)
            for image in images
        ] for size in size_wrapper.data],
    ) for images, prompt in zip(image_lists, model_prompts)]


def build_embedding_inputs_from_test_info(
    test_info: VLMTestInfo,
    image_assets: _ImageAssets,
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
    video_assets: _VideoAssets,
    size_wrapper: ImageSizeWrapper,
    num_frames: int,
):
    if test_info.prompt_formatter is None:
        raise ValueError("Prompt formatter must be set to build video inputs")
    model_prompts = get_model_prompts(
        [VIDEO_BASE_PROMPT],
        test_info.img_idx_to_prompt,
        test_info.video_idx_to_prompt,
        test_info.prompt_formatter,
    )

    sampled_vids = [
        sample_frames_from_video(asset.np_ndarrays, num_frames)
        for asset in video_assets
    ]

    video_scaler = (resize_video if size_wrapper.type == SizeType.FIXED_SIZE
                    else rescale_video_size)

    return [(
        [prompt for _ in size_wrapper.data],
        [video_scaler(video, size) for size in size_wrapper.data],
    ) for video, prompt in zip(sampled_vids, model_prompts)]


def apply_image_size_scaling(image, size: Union[float, Tuple[int, int]],
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


####### Entrypoints for running different test types
# Wrappers for all the above, where the only difference
# is that each test runs in a separate process
def run_single_image_test(
        *, tmp_path: PosixPath, test_info: VLMTestInfo, model: str,
        max_tokens: int, num_logprobs: int, dtype: str,
        distributed_executor_backend: Optional[str],
        size_wrapper: ImageSizeWrapper, hf_runner: Type[HfRunner],
        vllm_runner: Type[VllmRunner], image_assets: _ImageAssets):
    # Grab the model type's global model config to leverage callables
    inputs = build_single_image_inputs_from_test_info(test_info, image_assets,
                                                      size_wrapper, tmp_path)

    run_test(hf_runner=hf_runner,
             vllm_runner=vllm_runner,
             inputs=inputs,
             model=model,
             dtype=dtype,
             max_tokens=max_tokens,
             num_logprobs=num_logprobs,
             limit_mm_per_prompt={"image": 1},
             distributed_executor_backend=distributed_executor_backend,
             **test_info.get_non_parametrized_runner_kwargs())


def run_multi_image_test(
        *, tmp_path: PosixPath, test_info: VLMTestInfo, model: str,
        max_tokens: int, num_logprobs: int, dtype: str,
        distributed_executor_backend: Optional[str],
        size_wrapper: ImageSizeWrapper, hf_runner: Type[HfRunner],
        vllm_runner: Type[VllmRunner], image_assets: _ImageAssets):
    # Grab the model type's global model config to leverage callables
    inputs = build_multi_image_inputs_from_test_info(test_info, image_assets,
                                                     size_wrapper, tmp_path)

    run_test(hf_runner=hf_runner,
             vllm_runner=vllm_runner,
             inputs=inputs,
             model=model,
             dtype=dtype,
             max_tokens=max_tokens,
             num_logprobs=num_logprobs,
             limit_mm_per_prompt={"image": len(image_assets)},
             distributed_executor_backend=distributed_executor_backend,
             **test_info.get_non_parametrized_runner_kwargs())


def run_embedding_test(*, test_info: VLMTestInfo, model: str, max_tokens: int,
                       num_logprobs: int, dtype: str,
                       distributed_executor_backend: Optional[str],
                       size_wrapper: ImageSizeWrapper,
                       hf_runner: Type[HfRunner],
                       vllm_runner: Type[VllmRunner],
                       image_assets: _ImageAssets):
    inputs, vllm_embeddings = build_embedding_inputs_from_test_info(
        test_info, image_assets, size_wrapper)

    run_test(hf_runner=hf_runner,
             vllm_runner=vllm_runner,
             inputs=inputs,
             model=model,
             dtype=dtype,
             max_tokens=max_tokens,
             num_logprobs=num_logprobs,
             limit_mm_per_prompt={"image": 1},
             vllm_embeddings=vllm_embeddings,
             distributed_executor_backend=distributed_executor_backend,
             **test_info.get_non_parametrized_runner_kwargs())


def run_video_test(
    *,
    test_info: VLMTestInfo,
    model: str,
    num_frames: int,
    max_tokens: int,
    num_logprobs: int,
    dtype: str,
    distributed_executor_backend: Optional[str],
    size_wrapper: ImageSizeWrapper,
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    video_assets: _VideoAssets,
):
    inputs = build_video_inputs_from_test_info(test_info, video_assets,
                                               size_wrapper, num_frames)

    run_test(hf_runner=hf_runner,
             vllm_runner=vllm_runner,
             inputs=inputs,
             model=model,
             dtype=dtype,
             max_tokens=max_tokens,
             num_logprobs=num_logprobs,
             limit_mm_per_prompt={"video": len(video_assets)},
             distributed_executor_backend=distributed_executor_backend,
             **test_info.get_non_parametrized_runner_kwargs())


def run_custom_inputs_test(*, test_info: VLMTestInfo, model: str,
                           max_tokens: int, num_logprobs: int,
                           distributed_executor_backend: Optional[str],
                           dtype: str, custom_test_opts: CustomTestOptions,
                           hf_runner: Type[HfRunner],
                           vllm_runner: Type[VllmRunner]):
    # Custom test cases can provide inputs directly, but they need to
    # explicitly provided a CustomTestConfig, which wraps the inputs and
    # the limit_mm_per_prompt
    assert custom_test_opts is not None

    inputs = custom_test_opts.inputs

    limit_mm_per_prompt = custom_test_opts.limit_mm_per_prompt

    assert inputs is not None and limit_mm_per_prompt is not None
    run_test(hf_runner=hf_runner,
             vllm_runner=vllm_runner,
             inputs=inputs,
             model=model,
             dtype=dtype,
             max_tokens=max_tokens,
             num_logprobs=num_logprobs,
             limit_mm_per_prompt=limit_mm_per_prompt,
             distributed_executor_backend=distributed_executor_backend,
             **test_info.get_non_parametrized_runner_kwargs())


####### Core test implementation & details
def run_test(
    *,
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    inputs: List[Tuple[List[str], List[Union[List[Image], Image]]]],
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    enforce_eager: bool,
    max_model_len: int,
    max_num_seqs: int,
    hf_output_post_proc: Optional[Callable],
    vllm_output_post_proc: Optional[Callable],
    auto_cls: Type[_BaseAutoModelClass],
    use_tokenizer_eos: bool,
    postprocess_inputs: Callable[[BatchEncoding], BatchEncoding],
    comparator: Callable,
    get_stop_token_ids: Optional[Callable],
    limit_mm_per_prompt: Dict[str, int],
    model_kwargs: Optional[Dict[str, Any]],
    patch_hf_runner: Optional[Callable[[HfRunner], HfRunner]],
    runner_mm_key: str = "images",
    distributed_executor_backend: Optional[str] = None,
    tensor_parallel_size: int = 1,
    vllm_embeddings: Optional[torch.Tensor] = None,
):
    # In the case of embeddings, vLLM takes separate input tensors
    vllm_inputs = vllm_embeddings if vllm_embeddings is not None else inputs
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    vllm_outputs_per_mm = []
    hf_outputs_per_mm = []

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).
    vllm_kwargs = {}
    if get_stop_token_ids is not None:
        vllm_kwargs["stop_token_ids"] = get_stop_token_ids(tokenizer)

    with vllm_runner(model,
                     max_model_len=max_model_len,
                     max_num_seqs=max_num_seqs,
                     dtype=dtype,
                     limit_mm_per_prompt=limit_mm_per_prompt,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=enforce_eager) as vllm_model:
        for prompts, media in vllm_inputs:
            vllm_kwargs[runner_mm_key] = media
            vllm_output = vllm_model.generate_greedy_logprobs(
                prompts, max_tokens, num_logprobs=num_logprobs, **vllm_kwargs)
            vllm_outputs_per_mm.append(vllm_output)

    hf_model = hf_runner(model,
                         dtype=dtype,
                         auto_cls=auto_cls,
                         postprocess_inputs=postprocess_inputs,
                         model_kwargs=model_kwargs)

    # Some models need to patch things like the model processor, e.g., internvl
    if patch_hf_runner is not None:
        hf_model = patch_hf_runner(hf_model)

    # Some models need to explicitly pass the eos_token_id off the tokenizer or
    # processor for a good comparison; currently assume processor/tokenizer
    # agree on the EOS, and pull it off the tokenizer if requested.
    hf_kwargs = {}
    if use_tokenizer_eos:
        hf_kwargs["eos_token_id"] = tokenizer.eos_token_id

    with hf_model, torch.no_grad():
        for prompts, media in inputs:
            hf_kwargs[runner_mm_key] = media
            hf_output = hf_model.generate_greedy_logprobs_limit(
                prompts,
                max_tokens,
                num_logprobs=num_logprobs,
                tokenizer=tokenizer,
                **hf_kwargs)
            hf_outputs_per_mm.append(hf_output)

    # Apply output processing / sanitation to the vLLM and HF runner results
    hf_outputs_per_mm, vllm_outputs_per_mm = process_runner_outputs(
        model,
        first_runner_outputs=hf_outputs_per_mm,
        second_runner_outputs=vllm_outputs_per_mm,
        first_runner_processor=hf_output_post_proc,
        second_runner_processor=vllm_output_post_proc,
    )

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_mm,
                                        vllm_outputs_per_mm):
        # This is usually check_logprobs_close, but it's passed through to
        # allow things like check_outputs_equal where needed
        comparator(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )


def process_runner_outputs(
    model,
    first_runner_outputs,
    second_runner_outputs,
    first_runner_processor=None,
    second_runner_processor=None,
):
    if first_runner_processor is not None:
        first_runner_outputs = process_outputs(first_runner_processor, model,
                                               first_runner_outputs)
    if second_runner_processor is not None:
        second_runner_outputs = process_outputs(second_runner_processor, model,
                                                second_runner_outputs)
    return first_runner_outputs, second_runner_outputs


def process_outputs(output_processor, model, outputs_per_image):
    """Applies a model specific post-processor function to a runner's output"""
    return [[output_processor(res, model) for res in outputs]
            for outputs in outputs_per_image]
