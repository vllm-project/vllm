"""Common tests for testing .generate() functionality for single / multiple
image support for different VLMs in vLLM.
"""
import os
from pathlib import PosixPath
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import pytest
import torch
from PIL.Image import Image
from transformers import AutoModelForVision2Seq, AutoTokenizer, BatchEncoding
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from vllm.multimodal.utils import rescale_image_size
from vllm.utils import identity, is_cpu, is_hip

from ....conftest import HfRunner, VllmRunner, _ImageAssets, IMAGE_ASSETS
from ...utils import check_outputs_equal
from ....utils import get_memory_gb
from . import utils as vlm_utils
from .vlm_test_types import (MULTI_IMAGE_BASE_PROMPT,
                             SINGLE_IMAGE_BASE_PROMPTS, CustomTestOptions,
                             ImageSizeWrapper, SizeType, VLMTestInfo,
                             VlmTestType)

# This hack is needed for phi3v & paligemma models
# ROCm Triton FA can run into shared memory issues with these models,
# use other backends in the meantime
# FIXME (mattwong, gshtrasb, hongxiayan)
if is_hip():
    os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"

### Test configuration for specific models;
# NOTE: the key in the dict below is not mostly used as an identifier;
# it will be first in all of the expanded parametrizations, so it will
# tell you which test configuration failed.

# yapf: disable
VLM_TEST_SETTINGS = {
    "blip2": VLMTestInfo(
        models="Salesforce/blip2-opt-2.7b",
        prompt_formatter=lambda img_prompt: f"Question: {img_prompt} Answer:",
        test_type=VlmTestType.IMAGE,
        img_idx_to_prompt=lambda idx: "",
        auto_cls=AutoModelForVision2Seq,
        vllm_output_post_proc=vlm_utils.blip2_vllm_to_hf_output,
    ),
    "chameleon": VLMTestInfo(
        models="facebook/chameleon-7b",
        prompt_formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:",
        test_type=VlmTestType.IMAGE,
        max_model_len=4096,
        auto_cls=AutoModelForVision2Seq,
        postprocess_inputs=vlm_utils.get_key_type_post_processor(
            "pixel_values",
            "bfloat16"
        ),
        # For chameleon, we only compare the sequences
        vllm_output_post_proc = lambda vllm_output, model: vllm_output[:2],
        hf_output_post_proc = lambda hf_output, model: hf_output[:2],
        comparator=check_outputs_equal,
        max_tokens=8,
        dtype="bfloat16",
    ),
    "fuyu": VLMTestInfo(
        models="adept/fuyu-8b",
        prompt_formatter=lambda img_prompt: f"{img_prompt}\n",
        test_type=VlmTestType.IMAGE,
        img_idx_to_prompt=lambda idx: "",
        max_model_len=2048,
        max_num_seqs=2,
        use_tokenizer_eos=True,
        vllm_output_post_proc=vlm_utils.fuyu_vllm_to_hf_output,
        num_logprobs=10,
        dtype="bfloat16" if is_cpu() else "half",
        image_size_factors=((), (0.25,), (0.25, 0.25, 0.25), (0.25, 0.2, 0.15)),
    ),
    "glm4": VLMTestInfo(
        models="THUDM/glm-4v-9b",
        prompt_formatter=identity,
        test_type=VlmTestType.IMAGE,
        img_idx_to_prompt=lambda idx: "",
        max_model_len=2048,
        max_num_seqs=2,
        dtype="bfloat16",
        get_stop_token_ids=lambda tok: [151329, 151336, 151338],
        skip=(get_memory_gb() < 48), # large GPU test
        patch_hf_runner=vlm_utils.glm_patch_hf_runner,
    ),
    "llava": VLMTestInfo(
        models="llava-hf/llava-1.5-7b-hf",
        prompt_formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:",
        test_type=(
            VlmTestType.EMBEDDING,
            VlmTestType.IMAGE,
            VlmTestType.CUSTOM_INPUTS
        ),
        convert_assets_to_embeddings=vlm_utils.get_llava_embeddings,
        max_model_len=4096,
        auto_cls=AutoModelForVision2Seq,
        vllm_output_post_proc=vlm_utils.llava_image_vllm_to_hf_output,
        custom_test_opts=CustomTestOptions(
            inputs=vlm_utils.multi_image_multi_aspect_ratio_inputs_llava(is_llava=True),
            limit_mm_per_prompt={"image": 4},
        ),
    ),
    "llava-next": VLMTestInfo(
        models="llava-hf/llava-v1.6-mistral-7b-hf",
        prompt_formatter=lambda img_prompt: f"[INST] {img_prompt} [/INST]",
        test_type=(VlmTestType.IMAGE, VlmTestType.CUSTOM_INPUTS),
        max_model_len=10240,
        auto_cls=AutoModelForVision2Seq,
        vllm_output_post_proc=vlm_utils.llava_image_vllm_to_hf_output,
        custom_test_opts=CustomTestOptions(
            inputs=vlm_utils.multi_image_multi_aspect_ratio_inputs_llava(is_llava=False),
            limit_mm_per_prompt={"image": 4},
        ),
        # Llava-next tests fixed sizes & the default size factors
        image_sizes=(((1669, 2560), (2560, 1669), (183, 488), (488, 183),),),
    ),
    "minicpmv": VLMTestInfo(
        models="openbmb/MiniCPM-Llama3-V-2_5",
        prompt_formatter=lambda img_prompt: f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{img_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",  # noqa: E501
        test_type=(VlmTestType.MULTI_IMAGE),
        img_idx_to_prompt=lambda idx: "(<image>./</image>)\n",
        max_model_len=4096,
        max_num_seqs=2,
        get_stop_token_ids=lambda tok: [tok.eos_id, tok.eot_id],
        postprocess_inputs=vlm_utils.wrap_inputs_post_processor,
        hf_output_post_proc=vlm_utils.minicmpv_trunc_hf_output,
    ),
    "paligemma": VLMTestInfo(
        models="google/paligemma-3b-mix-224",
        prompt_formatter=identity,
        test_type=VlmTestType.IMAGE,
        img_idx_to_prompt = lambda idx: "",
        # Paligemma uses its own sample prompts because the default one fails
        single_image_prompts=IMAGE_ASSETS.prompts({
            "stop_sign": "caption es",
            "cherry_blossom": "What is in the picture?",
        }),
        auto_cls=AutoModelForVision2Seq,
        vllm_output_post_proc=vlm_utils.paligemma_vllm_to_hf_output,
        dtype="half" if is_hip() else ("half", "float"),
    ),
    "phi3v": VLMTestInfo(
        models="microsoft/Phi-3.5-vision-instruct",
        prompt_formatter=lambda img_prompt: f"<|user|>\n{img_prompt}<|end|>\n<|assistant|>\n", # noqa: E501
        test_type=(VlmTestType.IMAGE, VlmTestType.MULTI_IMAGE),
        img_idx_to_prompt=lambda idx: f"<|image_{idx}|>\n",
        max_model_len=4096,
        max_num_seqs=2,
        # use eager mode for hf runner, since phi3v didn't work with flash_attn
        model_kwargs={"_attn_implementation": "eager"},
        use_tokenizer_eos=True,
        vllm_output_post_proc=vlm_utils.phi3v_vllm_to_hf_output,
        num_logprobs=10,
    ),
    "qwen": VLMTestInfo(
        models="Qwen/Qwen-VL",
        prompt_formatter=identity,
        test_type=(VlmTestType.IMAGE, VlmTestType.MULTI_IMAGE),
        img_idx_to_prompt=lambda idx: f"Picture {idx}: <img></img>\n",
        max_model_len=1024,
        max_num_seqs=2,
        vllm_output_post_proc=vlm_utils.qwen_vllm_to_hf_output,
        prompt_path_encoder=vlm_utils.qwen_prompt_path_encoder,
    ),
    # Tests above this point have been validated to align with current tests
    "intern_vl": VLMTestInfo(
        models=("OpenGVLab/InternVL2-1B", "OpenGVLab/InternVL2-2B"),
        prompt_formatter=lambda img_prompt: f"<|im_start|>User\n{img_prompt}<|im_end|>\n<|im_start|>Assistant\n", # noqa: E501
        test_type=(VlmTestType.IMAGE, VlmTestType.MULTI_IMAGE),
        max_model_len=4096,
        num_logprobs=10,
        dtype="bfloat16" if is_cpu() else "half",
        use_tokenizer_eos=True,
        patch_hf_runner=vlm_utils.internvl_patch_hf_runner,
    ),
}
# yapf: enable

### Test wrappers
# Wrappers around the core test running func for:
# - single image
# - multi-image
# - image embeddings
# - video [TODO]
# All wrappers (except single image) have a filter for dropping
# models that don't have applicable tests, and expanding the
# relevant VLMTestInfo object into a combination that can be
# consumed by parametrize()
@pytest.mark.parametrize(
    "model_type,model,max_tokens,num_logprobs,dtype,size_wrapper",
    vlm_utils.get_parametrized_options(VLM_TEST_SETTINGS,
                                       test_type=VlmTestType.IMAGE))
def test_single_image_generation(tmp_path: PosixPath, model_type: str,
                                 model: str, max_tokens: int,
                                 num_logprobs: int, dtype: str,
                                 size_wrapper: ImageSizeWrapper, hf_runner,
                                 vllm_runner, image_assets: _ImageAssets):
    # Grab the model type's global model config to leverage callables
    test_info = VLM_TEST_SETTINGS[model_type]
    model_prompts = vlm_utils.get_model_prompts(test_info.single_image_prompts,
                                                test_info.img_idx_to_prompt,
                                                test_info.prompt_formatter)

    # For models that require a local path / URL encoded in the image; export
    # assets and encode into tmp_path for this test. This should be avoided
    # where possible (currently needed for Qwen-VL).
    if test_info.prompt_path_encoder is not None:
        model_prompts = [
            test_info.prompt_path_encoder(tmp_path, prompt, [asset])
            for prompt, asset in zip(model_prompts, image_assets)
        ]

    images = [asset.pil_image for asset in image_assets]
    assert len(images) == len(model_prompts)

    inputs = build_single_image_inputs(images, model_prompts, size_wrapper)

    run_test(hf_runner=hf_runner,
             vllm_runner=vllm_runner,
             inputs=inputs,
             model=model,
             dtype=dtype,
             max_tokens=max_tokens,
             num_logprobs=num_logprobs,
             limit_mm_per_prompt={"image": 1},
             size_factors=size_wrapper,
             **test_info.get_non_parametrized_runner_kwargs())


# def test_video_generation():
#     raise NotImplementedError("Video model test wrapper not implemented")

# def test_quantized_models():
#     raise NotImplementedError("Quantized model test wrapper not implemented")


@pytest.mark.parametrize(
    "model_type,model,max_tokens,num_logprobs,dtype,size_wrapper",
    vlm_utils.get_parametrized_options(VLM_TEST_SETTINGS,
                                       test_type=VlmTestType.EMBEDDING))
def test_embedding_generation(model_type: str, model: str, max_tokens: int,
                              num_logprobs: int, dtype: str,
                              size_wrapper: ImageSizeWrapper, hf_runner,
                              vllm_runner, image_assets: _ImageAssets):
    # Grab the model type's global model config to leverage callables
    test_info = VLM_TEST_SETTINGS[model_type]
    # These checks will always be true due to the way the test is invoked
    assert size_wrapper.type != SizeType.SIZE_FACTOR or not \
            all(factor == 1.0 for factor in size_wrapper.data)
    assert test_info.convert_assets_to_embeddings is not None

    model_prompts = vlm_utils.get_model_prompts(
        SINGLE_IMAGE_BASE_PROMPTS,
        test_info.img_idx_to_prompt,
        test_info.prompt_formatter,
    )

    images = [asset.pil_image for asset in image_assets]
    embeds = test_info.convert_assets_to_embeddings(image_assets)
    assert len(images) == len(model_prompts)

    inputs = build_single_image_inputs(images, model_prompts, size_wrapper)
    vllm_embeddings = build_single_image_inputs(embeds, model_prompts,
                                                size_wrapper)

    run_test(hf_runner=hf_runner,
             vllm_runner=vllm_runner,
             inputs=inputs,
             model=model,
             dtype=dtype,
             max_tokens=max_tokens,
             num_logprobs=num_logprobs,
             limit_mm_per_prompt={"image": 1},
             size_factors=size_wrapper,
             vllm_embeddings=vllm_embeddings,
             **test_info.get_non_parametrized_runner_kwargs())


@pytest.mark.parametrize(
    "model_type,model,max_tokens,num_logprobs,dtype,size_wrapper",
    vlm_utils.get_parametrized_options(VLM_TEST_SETTINGS,
                                       test_type=VlmTestType.MULTI_IMAGE))
def test_multi_image_generation(tmp_path: PosixPath, model_type: str,
                                model: str, max_tokens: int, num_logprobs: int,
                                dtype: str, size_wrapper: ImageSizeWrapper,
                                hf_runner, vllm_runner,
                                image_assets: _ImageAssets):
    test_info = VLM_TEST_SETTINGS[model_type]

    model_prompts = vlm_utils.get_model_prompts([MULTI_IMAGE_BASE_PROMPT],
                                                test_info.img_idx_to_prompt,
                                                test_info.prompt_formatter)

    if test_info.prompt_path_encoder is not None:
        model_prompts = [
            test_info.prompt_path_encoder(tmp_path, model_prompt, image_assets)
            for model_prompt in model_prompts
        ]

    images = [asset.pil_image for asset in image_assets]

    # Currently, we only have one multi-image list & one multi-image prompt
    inputs = build_multi_image_inputs(
        image_lists=[images],
        model_prompts=model_prompts,
        size_wrapper=size_wrapper,
    )

    run_test(hf_runner=hf_runner,
             vllm_runner=vllm_runner,
             inputs=inputs,
             model=model,
             dtype=dtype,
             max_tokens=max_tokens,
             num_logprobs=num_logprobs,
             limit_mm_per_prompt={"image": len(images)},
             size_factors=size_wrapper,
             **test_info.get_non_parametrized_runner_kwargs())


@pytest.mark.parametrize("model_type,model,max_tokens,num_logprobs,dtype",
                         vlm_utils.get_parametrized_options(
                             VLM_TEST_SETTINGS,
                             test_type=VlmTestType.CUSTOM_INPUTS))
def test_custom_inputs(model_type: str, model: str, max_tokens: int,
                       num_logprobs: int, dtype: str, hf_runner, vllm_runner):
    test_info = VLM_TEST_SETTINGS[model_type]
    custom_test_opts = test_info.custom_test_opts
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
             size_factors=None,
             **test_info.get_non_parametrized_runner_kwargs())


### Core test implementation & details
# run_test() is does the heavy lifting of every test type here.
def run_test(
    *,
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    inputs: List[Tuple[List[str], List[Union[List[Image], Image]]]],
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
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
    size_factors,
    model_kwargs: Optional[Dict[str, Any]],
    patch_hf_runner: Optional[Callable[[HfRunner], HfRunner]],
    vllm_embeddings: Optional[torch.Tensor]=None,
):
    # In the case of embeddings, vLLM takes separate input tensors
    vllm_inputs = vllm_embeddings if vllm_embeddings is not None else inputs
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).
    opt_vllm_kwargs = {}
    if get_stop_token_ids is not None:
        opt_vllm_kwargs["stop_token_ids"] = get_stop_token_ids(tokenizer)

    with vllm_runner(model,
                     max_model_len=max_model_len,
                     max_num_seqs=max_num_seqs,
                     dtype=dtype,
                     limit_mm_per_prompt=limit_mm_per_prompt,
                     tensor_parallel_size=tensor_parallel_size,
                     enforce_eager=enforce_eager) as vllm_model:
        vllm_outputs_per_image = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images,
                                                **opt_vllm_kwargs)
            for prompts, images in vllm_inputs
        ]

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
    opt_hf_kwargs = {}
    if use_tokenizer_eos:
        opt_hf_kwargs["eos_token_id"] = tokenizer.eos_token_id

    with hf_model, torch.no_grad():
        hf_outputs_per_image = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    images=images,
                                                    tokenizer=tokenizer,
                                                    **opt_hf_kwargs)
            for prompts, images in inputs
        ]

    # Apply output processing / sanitation to the vLLM and HF runner results
    hf_outputs_per_image, vllm_outputs_per_image = _process_runner_outputs(
        model,
        first_runner_outputs=hf_outputs_per_image,
        second_runner_outputs=vllm_outputs_per_image,
        first_runner_processor=hf_output_post_proc,
        second_runner_processor=vllm_output_post_proc,
    )

    vlm_utils.export_test(
        model,
        size_factors.data if size_factors is not None else None,
        is_new=True,
        export_info=[
            {"config": {"inputs": inputs, "max_tokens": max_tokens, "num_logprobs": num_logprobs}},
            {"hf_runner": {"dtype": dtype, "model": model}},
            {"hf_out": hf_outputs_per_image},
            {"vllm_out": vllm_outputs_per_image},
        ],
    )

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_image,
                                        vllm_outputs_per_image):
        # This is usually check_logprobs_close, but it's passed through to
        # allow things like check_outputs_equal where needed
        comparator(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )


def _process_runner_outputs(
    model,
    first_runner_outputs,
    second_runner_outputs,
    quantized_model=None,
    first_runner_processor=None,
    second_runner_processor=None,
):
    # In the case of quantization tests, we have two vLLM runners with one
    # running the quantized model, so they should have the same post-processor
    if quantized_model is not None:
        # This should be easy, but for now disabled since we don't have the
        # wrapper or second vLLM runner added; will enable when porting
        # internVL
        # assert second_runner_processor is None
        # second_runner_processor = first_runner_processor
        raise NotImplementedError("WIP - quantized test wrapper not added yet")

    if first_runner_processor is not None:
        first_runner_outputs = _process_outputs(first_runner_processor, model,
                                                first_runner_outputs)
    if second_runner_processor is not None:
        second_runner_outputs = _process_outputs(second_runner_processor,
                                                 model, second_runner_outputs)
    return first_runner_outputs, second_runner_outputs


def _process_outputs(output_processor, model, outputs_per_image):
    """Applies a model specific post-processor function to a runner's output"""
    return [[output_processor(res, model) for res in outputs]
            for outputs in outputs_per_image]


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
            apply_size_scaling(image, size, size_wrapper.type)
            for size in size_wrapper.data
        ],
    ) for image, prompt in zip(images, model_prompts)]


def build_multi_image_inputs(image_lists, model_prompts,
                             size_wrapper: ImageSizeWrapper):
    return [(
        [prompt for _ in size_wrapper.data],
        [[
            apply_size_scaling(image, size, size_wrapper.type)
            for image in images
        ] for size in size_wrapper.data],
    ) for images, prompt in zip(image_lists, model_prompts)]


def apply_size_scaling(image, size: Union[float, Tuple[int, int]],
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
