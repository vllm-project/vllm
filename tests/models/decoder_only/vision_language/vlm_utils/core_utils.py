from pathlib import PosixPath
from typing import (Any, Callable, Dict, Iterable, List, Optional, Tuple, Type,
                    Union)

import torch
from PIL.Image import Image
from transformers import AutoTokenizer, BatchEncoding
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from vllm.multimodal.utils import (rescale_image_size, rescale_video_size,
                                   resize_video, sample_frames_from_video)

from .....conftest import HfRunner, VllmRunner, _ImageAssets, _VideoAssets
from .mm_test_types import (MULTI_IMAGE_BASE_PROMPT, SINGLE_IMAGE_BASE_PROMPTS,
                            TEST_IMG_PLACEHOLDER, TEST_VIDEO_PLACEHOLDER,
                            VIDEO_BASE_PROMPT, CustomTestOptions,
                            ImageSizeWrapper, RunnerOutput, SizeType,
                            VLMTestInfo)


####### Helpers for building the inputs to different test types
def replace_test_placeholder(prompt: str, img_idx_to_prompt: Callable[[int],
                                                                      str],
                             test_placeholder: str) -> str:
    """Given a prompt, replaces each test placeholder with the
    model-specific tag.
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
                      prompt_formatter: Callable[[str], str]) -> List[str]:
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
    hf_output_post_proc: Optional[Callable[[RunnerOutput, str], Any]],
    vllm_output_post_proc: Optional[Callable[[RunnerOutput, str], Any]],
    auto_cls: Type[_BaseAutoModelClass],
    use_tokenizer_eos: bool,
    postprocess_inputs: Callable[[BatchEncoding], BatchEncoding],
    comparator: Callable[..., None],
    get_stop_token_ids: Optional[Callable[[AutoTokenizer], List[int]]],
    limit_mm_per_prompt: Dict[str, int],
    model_kwargs: Optional[Dict[str, Any]],
    patch_hf_runner: Optional[Callable[[HfRunner], HfRunner]],
    runner_mm_key: str = "images",
    distributed_executor_backend: Optional[str] = None,
    tensor_parallel_size: int = 1,
    vllm_embeddings: Optional[torch.Tensor] = None,
):
    """Modality agnostic test test executor for comparing HF/vLLM outputs."""
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
    """Applies the runner processor(s) to the runner outputs, if any."""
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
