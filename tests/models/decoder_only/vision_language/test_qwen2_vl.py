from typing import Any, List, Optional, Tuple, Type, TypedDict, Union

import numpy.typing as npt
import pytest
import torch
from PIL import Image

from vllm.entrypoints.llm import LLM
from vllm.multimodal.utils import (rescale_image_size, rescale_video_size,
                                   sample_frames_from_video)

from ....conftest import (IMAGE_ASSETS, VIDEO_ASSETS, PromptImageInput,
                          PromptVideoInput, VllmRunner)
from ...utils import check_logprobs_close

models = ["Qwen/Qwen2-VL-2B-Instruct"]
target_dtype = "half"

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"
MODEL_HIDDEN_SIZE = 1536


def qwen2_vl_chat_template(*query):
    return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{''.join(query)}<|im_end|><|im_start|>assistant\n"  # noqa: E501


IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    qwen2_vl_chat_template(
        IMAGE_PLACEHOLDER,
        "What is the biggest text's content in this image?",
    ),
    "cherry_blossom":
    qwen2_vl_chat_template(
        IMAGE_PLACEHOLDER,
        "What is the season shown in this image? ",
        "Reply with a short sentence (no more than 20 words)",
    ),
})

VIDEO_PROMPTS = VIDEO_ASSETS.prompts({
    "sample_demo_1":
    qwen2_vl_chat_template(
        VIDEO_PLACEHOLDER,
        "Describe this video with a short sentence ",
        "(no more than 20 words)",
    ),
})

MULTIIMAGE_PROMPT = qwen2_vl_chat_template(
    IMAGE_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    "Describe these two images separately. ",
    "For each image, reply with a short sentence ",
    "(no more than 10 words).",
)


class Qwen2VLPromptImageEmbeddingInput(TypedDict):
    image_embeds: torch.Tensor
    image_grid_thw: torch.Tensor


class Qwen2VLPromptVideoEmbeddingInput(TypedDict):
    video_embeds: torch.Tensor
    video_grid_thw: torch.Tensor


def batch_make_image_embeddings(
        image_batches: List[Union[Image.Image, List[Image.Image]]], processor,
        llm: LLM) -> List[Qwen2VLPromptImageEmbeddingInput]:
    """batched image embeddings for Qwen2-VL

    This will infer all images' embeddings in a single batch, 
      and split the result according to input batches.

    image_batches:
      - Single-image batches: `List[Image.Image]`
      - Multiple-image batches: `List[List[Image.Image]]]`
    
    returns: `List[Qwen2VLPromptImageEmbeddingInput]`
    """

    image_batches_: List[Any] = image_batches[:]

    # convert single-image batches to multiple-image batches
    for idx in range(len(image_batches_)):
        if not isinstance(image_batches_[idx], list):
            image_batches_[idx] = [image_batches_[idx]]

        assert isinstance(image_batches_[idx], list)

    # append all images into a list (as a batch)
    images: List[Image.Image] = []
    for image_batch in image_batches_:
        images += image_batch

    # image to pixel values
    image_processor = processor.image_processor

    preprocess_result = image_processor \
        .preprocess(images=images, return_tensors="pt") \
        .data
    pixel_values = preprocess_result["pixel_values"]
    image_grid_thw = preprocess_result["image_grid_thw"]

    # pixel values to embeddinds & grid_thws
    with torch.no_grad():
        visual = llm.llm_engine.model_executor.driver_worker. \
            model_runner.model.visual

        pixel_values_on_device = pixel_values.to(visual.device,
                                                 dtype=visual.dtype)
        image_grid_thw_on_device = image_grid_thw.to(visual.device,
                                                     dtype=torch.int64)
        image_embeds = visual(pixel_values_on_device,
                              grid_thw=image_grid_thw_on_device)

    # split into original batches
    result: List[Qwen2VLPromptImageEmbeddingInput] = []
    image_counter = 0
    embed_counter = 0
    for image_batch in image_batches_:
        cur_batch_image_count = len(image_batch)
        merge_size = image_processor.merge_size
        cur_batch_embed_len = sum([
            grid_thw.prod() // merge_size // merge_size
            for grid_thw in image_grid_thw[image_counter:image_counter +
                                           cur_batch_image_count]
        ])

        result.append({
            "image_embeds":
            image_embeds[embed_counter:embed_counter + cur_batch_embed_len],
            "image_grid_thw":
            image_grid_thw[image_counter:image_counter +
                           cur_batch_image_count],
        })

        embed_counter += cur_batch_embed_len
        image_counter += cur_batch_image_count

    # ensure we don't lost any images or embeddings
    assert embed_counter == image_embeds.size(0)
    assert image_counter == image_grid_thw.size(0)
    assert len(image_batches) == len(result)

    return result


def batch_make_video_embeddings(
        video_batches: PromptVideoInput, processor,
        llm: LLM) -> List[Qwen2VLPromptVideoEmbeddingInput]:
    """batched video embeddings for Qwen2-VL

    A NDArray represents a single video's all frames.

    This will infer all videos' embeddings in a single batch, 
      and split the result according to input batches.

    video_batches:
      - Single-video batches: `List[NDArray]`
      - Multiple-video batches: `List[List[NDArray]]`
    """

    video_batches_: List[Any] = video_batches[:]

    for idx in range(len(video_batches_)):
        if not isinstance(video_batches_[idx], list):
            single_video_batch: List[npt.NDArray] = [video_batches_[idx]]
            video_batches_[idx] = single_video_batch

        assert isinstance(video_batches_[idx], list)

    # append all videos into a list (as a batch)
    videos: List[npt.NDArray] = []
    for video_batch in video_batches_:
        videos += video_batch

    # video to pixel values
    image_processor = processor.image_processor

    preprocess_result = image_processor \
        .preprocess(images=None, videos=videos, return_tensors="pt") \
        .data
    pixel_values = preprocess_result["pixel_values_videos"]
    video_grid_thw = preprocess_result["video_grid_thw"]

    # pixel values to embeddinds & grid_thws
    with torch.no_grad():
        visual = llm.llm_engine.model_executor.driver_worker.\
            model_runner.model.visual

        pixel_values_on_device = pixel_values.to(visual.device,
                                                 dtype=visual.dtype)
        video_grid_thw_on_device = video_grid_thw.to(visual.device,
                                                     dtype=torch.int64)
        video_embeds = visual(pixel_values_on_device,
                              grid_thw=video_grid_thw_on_device)

    # split into original batches
    result: List[Qwen2VLPromptVideoEmbeddingInput] = []
    video_counter = 0
    embed_counter = 0
    for video_batch in video_batches_:
        cur_batch_video_count = len(video_batch)
        merge_size = image_processor.merge_size
        cur_batch_embed_len = sum([
            grid_thw.prod() // merge_size // merge_size
            for grid_thw in video_grid_thw[video_counter:video_counter +
                                           cur_batch_video_count]
        ])

        result.append({
            "video_embeds":
            video_embeds[embed_counter:embed_counter + cur_batch_embed_len],
            "video_grid_thw":
            video_grid_thw[video_counter:video_counter +
                           cur_batch_video_count],
        })

        embed_counter += cur_batch_embed_len
        video_counter += cur_batch_video_count

    # ensure we don't lost any videos or embeddings
    assert embed_counter == video_embeds.size(0)
    assert video_counter == video_grid_thw.size(0)
    assert len(video_batches) == len(result)

    return result


def run_embedding_input_test(
    vllm_runner: Type[VllmRunner],
    inputs: List[Tuple[List[str], PromptImageInput, PromptVideoInput]],
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    mm_limit: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    """Inference result should be the same between
    original image/video input and image/video embeddings input.
    """
    from transformers import AutoProcessor  # noqa: F401

    processor = AutoProcessor.from_pretrained(model)

    # NOTE:
    # max_model_len should be greater than image_feature_size
    with vllm_runner(model,
                     task="generate",
                     max_model_len=4000,
                     max_num_seqs=3,
                     dtype=dtype,
                     limit_mm_per_prompt={
                         "image": mm_limit,
                         "video": mm_limit
                     },
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend
                     ) as vllm_model:

        outputs_per_case_for_original_input = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images or None,
                                                videos=videos or None)
            for prompts, images, videos in inputs
        ]

        outputs_per_case_for_embeddings_input = [
            vllm_model.generate_greedy_logprobs(
                prompts,
                max_tokens,
                num_logprobs=num_logprobs,
                images=batch_make_image_embeddings(
                    images, processor, vllm_model.model) if images else None,
                videos=batch_make_video_embeddings(
                    videos, processor, vllm_model.model) if videos else None)
            for prompts, images, videos in inputs
        ]

    for outputs_for_original_input, \
        outputs_for_embeddings_input \
        in zip(outputs_per_case_for_original_input,
            outputs_per_case_for_embeddings_input):
        check_logprobs_close(
            outputs_0_lst=outputs_for_original_input,
            outputs_1_lst=outputs_for_embeddings_input,
            name_0="original_input",
            name_1="embeddings_input",
        )


@pytest.mark.core_model
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "size_factors",
    [
        # Single-scale
        [0.5],
        # Single-scale, batched
        [0.5, 0.5],
        # Multi-scale
        [0.25, 0.5, 0.5],
    ],
)
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [10])
def test_qwen2_vl_image_embeddings_input(vllm_runner, image_assets, model,
                                         size_factors, dtype: str,
                                         max_tokens: int,
                                         num_logprobs: int) -> None:
    images = [asset.pil_image for asset in image_assets]

    inputs_per_case: List[Tuple[
        List[str], PromptImageInput, PromptVideoInput]] = [(
            [prompt for _ in size_factors],
            [rescale_image_size(image, factor) for factor in size_factors],
            [],
        ) for image, prompt in zip(images, IMAGE_PROMPTS)]

    run_embedding_input_test(
        vllm_runner,
        inputs_per_case,
        model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        mm_limit=1,
        tensor_parallel_size=1,
    )


@pytest.mark.core_model
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "size_factors",
    [
        [],
        # Single-scale
        [0.5],
        # Single-scale, batched
        [0.5, 0.5],
        # Multi-scale
        [0.25, 0.5, 0.5],
    ],
)
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [10])
def test_qwen2_vl_multiple_image_embeddings_input(vllm_runner, image_assets,
                                                  model, size_factors,
                                                  dtype: str, max_tokens: int,
                                                  num_logprobs: int) -> None:
    images = [asset.pil_image for asset in image_assets]

    inputs_per_case: List[Tuple[List[str], PromptImageInput,
                                PromptVideoInput]] = [(
                                    [MULTIIMAGE_PROMPT for _ in size_factors],
                                    [[
                                        rescale_image_size(image, factor)
                                        for image in images
                                    ] for factor in size_factors],
                                    [],
                                )]

    run_embedding_input_test(
        vllm_runner,
        inputs_per_case,
        model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        mm_limit=2,
        tensor_parallel_size=1,
    )


@pytest.mark.core_model
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "size_factors",
    [
        # Single-scale
        [0.5],
        # Single-scale, batched
        [0.5, 0.5],
        # Multi-scale
        [0.25, 0.25, 0.5],
    ],
)
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [10])
def test_qwen2_vl_video_embeddings_input(vllm_runner, video_assets, model,
                                         size_factors, dtype: str,
                                         max_tokens: int,
                                         num_logprobs: int) -> None:
    num_frames = 4
    sampled_vids = [
        sample_frames_from_video(asset.np_ndarrays, num_frames)
        for asset in video_assets
    ]

    inputs_per_case: List[Tuple[
        List[str], PromptImageInput, PromptVideoInput]] = [(
            [prompt for _ in size_factors],
            [],
            [rescale_video_size(video, factor) for factor in size_factors],
        ) for video, prompt in zip(sampled_vids, VIDEO_PROMPTS)]

    run_embedding_input_test(
        vllm_runner,
        inputs_per_case,
        model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        mm_limit=1,
        tensor_parallel_size=1,
    )


def run_chunked_prefill_test(
    vllm_runner: Type[VllmRunner],
    inputs: List[Tuple[List[str], PromptImageInput, PromptVideoInput]],
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    mm_limit: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    """Compare inference result between
    chunked prefill disabled and chunked prefill enabled
    """

    # NOTE:
    # max_model_len should be greater than image_feature_size
    with vllm_runner(model,
                     task="generate",
                     max_model_len=4000,
                     max_num_seqs=4,
                     dtype=dtype,
                     limit_mm_per_prompt={
                         "image": mm_limit,
                         "video": mm_limit
                     },
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend
                     ) as vllm_model:

        outputs_per_case = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images or None,
                                                videos=videos or None)
            for prompts, images, videos in inputs
        ]

    with vllm_runner(
            model,
            task="generate",
            max_model_len=4000,
            max_num_seqs=4,
            dtype=dtype,
            limit_mm_per_prompt={
                "image": mm_limit,
                "video": mm_limit
            },
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=distributed_executor_backend,
            enable_chunked_prefill=True,
            # should be small enough to ensure prefilling is chunked
            max_num_batched_tokens=32,
            mm_processor_kwargs={
                "max_pixels": 16 * 28 * 28,
            }) as vllm_model_chunked:
        outputs_per_case_chunked = [
            vllm_model_chunked.generate_greedy_logprobs(
                prompts,
                max_tokens,
                num_logprobs=num_logprobs,
                images=images or None,
                videos=videos or None) for prompts, images, videos in inputs
        ]

    for outputs, \
        outputs_chunked \
        in zip(outputs_per_case,
            outputs_per_case_chunked):
        check_logprobs_close(
            outputs_0_lst=outputs,
            outputs_1_lst=outputs_chunked,
            name_0="non_chunked",
            name_1="chunked",
        )


@pytest.mark.core_model
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [1])
@pytest.mark.parametrize("num_logprobs", [10])
def test_qwen2_vl_mrope_chunked_prefill(vllm_runner, example_prompts,
                                        model: str, dtype: str,
                                        max_tokens: int,
                                        num_logprobs: int) -> None:
    """
    Test Qwen2-VL's chunked prefill with M-RoPE
    """
    prompts = [
        qwen2_vl_chat_template(IMAGE_PLACEHOLDER, prompt)
        for prompt in example_prompts[:1]
    ]

    # 1. Qwen2-VL's M-RoPE works only when there are some multi-modal inputs,
    #    so an image is included in the inputs
    # 2. however, Qwen2-VL currently won't work properly
    #    when chunked prefill is enabled and there are some multi-modal inputs,
    #    here use a hacky way: provide a **zero-length** image to make it happy
    #
    # and finally we achieved:
    # (1) chunked_prefill enabled; (2) M-RoPE works; to continue our tests
    zero_len_image = {
        "image_embeds": torch.empty((0, MODEL_HIDDEN_SIZE)),
        "image_grid_thw": torch.tensor([[0, 0, 0]])
    }
    images = [zero_len_image] * len(prompts)

    inputs_per_case: List[Tuple[List[str], PromptImageInput,
                                PromptVideoInput]] = [
                                    (prompts, images, []),
                                ]

    run_chunked_prefill_test(
        vllm_runner,
        inputs_per_case,
        model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        mm_limit=1,
        tensor_parallel_size=1,
    )
