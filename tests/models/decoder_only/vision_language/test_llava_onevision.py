from typing import List, Optional, Tuple, Type, overload

import pytest
from transformers import (AutoConfig, AutoModelForVision2Seq, AutoTokenizer,
                          BatchEncoding)

from vllm.multimodal.utils import (rescale_image_size, rescale_video_size,
                                   resize_video, sample_frames_from_video)
from vllm.sequence import SampleLogprobs
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE

from ....conftest import (VIDEO_ASSETS, HfRunner, PromptImageInput, VllmRunner,
                          _VideoAssets)
from ....utils import large_gpu_test
from ...utils import check_logprobs_close

# Video test
HF_VIDEO_PROMPTS = VIDEO_ASSETS.prompts({
    "sample_demo_1":
    "<|im_start|>user\n<video>\nwhy is this video funny?<|im_end|>\n<|im_start|>assistant\n"  # noqa: E501
})

models = ["llava-hf/llava-onevision-qwen2-7b-ov-hf"]


def vllm_to_hf_output(vllm_output: Tuple[List[int], str,
                                         Optional[SampleLogprobs]],
                      model: str):
    """Sanitize vllm output to be comparable with hf output."""
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


@overload
def run_video_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    video_assets: _VideoAssets,
    model: str,
    *,
    size_factors: List[float],
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    num_frames: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    ...


@overload
def run_video_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    video_assets: _VideoAssets,
    model: str,
    *,
    sizes: List[Tuple[int, int]],
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    num_frames: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    ...


def run_video_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    video_assets: _VideoAssets,
    model: str,
    *,
    size_factors: Optional[List[float]] = None,
    sizes: Optional[List[Tuple[int, int]]] = None,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    num_frames: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]

    videos = [
        sample_frames_from_video(asset.np_ndarrays, num_frames)
        for asset in video_assets
    ]

    if size_factors is not None:
        inputs_per_video = [(
            [prompt for _ in size_factors],
            [rescale_video_size(video, factor) for factor in size_factors],
        ) for video, prompt in zip(videos, HF_VIDEO_PROMPTS)]
    elif sizes is not None:
        inputs_per_video = [(
            [prompt for _ in sizes],
            [resize_video(video, size) for size in sizes],
        ) for video, prompt in zip(videos, HF_VIDEO_PROMPTS)]
    else:
        raise ValueError("You must provide either `size_factors` or `sizes`")

    # max_model_len should be greater than image_feature_size
    with vllm_runner(model,
                     dtype=dtype,
                     max_model_len=4096,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True) as vllm_model:
        vllm_outputs_per_video = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                videos=videos)
            for prompts, videos in inputs_per_video
        ]

    def process(hf_inputs: BatchEncoding):
        hf_inputs["pixel_values_videos"] = hf_inputs["pixel_values_videos"] \
            .to(torch_dtype)  # type: ignore
        return hf_inputs

    with hf_runner(model,
                   dtype=dtype,
                   postprocess_inputs=process,
                   auto_cls=AutoModelForVision2Seq) as hf_model:
        hf_outputs_per_video = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    videos=videos)
            for prompts, videos in inputs_per_video
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_video,
                                        vllm_outputs_per_video):
        # TODO: Check whether using original CLIPVisionModel can improve
        # consistency against HF
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=[
                vllm_to_hf_output(vllm_output, model)
                for vllm_output in vllm_outputs
            ],
            name_0="hf",
            name_1="vllm",
        )


@large_gpu_test(min_gb=48)
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "size_factors",
    [
        # No video
        [],
        # Single-scale
        [1.0],
        # Single-scale, batched
        [1.0, 1.0, 1.0],
        # Multi-scale
        [0.25, 0.5, 1.0],
    ],
)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("num_frames", [16])
def test_models(hf_runner, vllm_runner, video_assets, model, size_factors,
                dtype, max_tokens, num_logprobs, num_frames) -> None:
    """Inference result should be the same between hf and vllm.

    All the image fixtures for the test is under tests/videos.
    For huggingface runner, we provide the np.ndarray as input.
    For vllm runner, we provide MultiModalDataDict objects 
    and corresponding MultiModalConfig as input.
    Note, the text input is also adjusted to abide by vllm contract.
    The text output is sanitized to be able to compare with hf.
    """
    run_video_test(
        hf_runner,
        vllm_runner,
        video_assets,
        model,
        size_factors=size_factors,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        num_frames=num_frames,
        tensor_parallel_size=1,
    )


@large_gpu_test(min_gb=48)
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "sizes",
    [[(1669, 2560), (2560, 1669), (183, 488), (488, 183)]],
)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("num_frames", [16])
def test_models_fixed_sizes(hf_runner, vllm_runner, video_assets, model, sizes,
                            dtype, max_tokens, num_logprobs,
                            num_frames) -> None:
    run_video_test(
        hf_runner,
        vllm_runner,
        video_assets,
        model,
        sizes=sizes,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        num_frames=num_frames,
        tensor_parallel_size=1,
    )


# Image test
_LIMIT_IMAGE_PER_PROMPT = 4


def run_image_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    inputs: List[Tuple[List[str], PromptImageInput]],
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]

    # max_model_len should be greater than image_feature_size
    with vllm_runner(model,
                     dtype=dtype,
                     max_model_len=16384,
                     max_num_seqs=2,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True,
                     limit_mm_per_prompt={"image": _LIMIT_IMAGE_PER_PROMPT
                                          }) as vllm_model:
        vllm_outputs_per_image = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images)
            for prompts, images in inputs
        ]

    def process(hf_inputs: BatchEncoding):
        hf_inputs["pixel_values"] = hf_inputs["pixel_values"] \
            .to(torch_dtype)  # type: ignore
        return hf_inputs

    with hf_runner(model,
                   dtype=dtype,
                   postprocess_inputs=process,
                   auto_cls=AutoModelForVision2Seq) as hf_model:
        hf_outputs_per_image = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    images=images)
            for prompts, images in inputs
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_image,
                                        vllm_outputs_per_image):
        # TODO: Check whether using original CLIPVisionModel can improve
        # consistency against HF
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=[
                vllm_to_hf_output(vllm_output, model)
                for vllm_output in vllm_outputs
            ],
            name_0="hf",
            name_1="vllm",
        )


@large_gpu_test(min_gb=48)
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models_multiple_image_inputs(hf_runner, vllm_runner, image_assets,
                                      model, dtype, max_tokens,
                                      num_logprobs) -> None:
    stop_sign = image_assets[0].pil_image
    cherry_blossom = image_assets[1].pil_image

    inputs = [(
        [
            "<|im_start|>user\n<image><image>\nDescribe 2 images.<|im_end|>\n<|im_start|>assistant\n",  # noqa: E501
            "<|im_start|>user\n<image><image>\nDescribe 2 images.<|im_end|>\n<|im_start|>assistant\n",  # noqa: E501
            "<|im_start|>user\n<image><image><image><image>\nDescribe 4 images.<|im_end|>\n<|im_start|>assistant\n",  # noqa: E501
            "<|im_start|>user\n<image>\nWhat is the season?<|im_end|>\n<|im_start|>assistant\n",  # noqa: E501
        ],
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

    run_image_test(
        hf_runner,
        vllm_runner,
        inputs,
        model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
    )
