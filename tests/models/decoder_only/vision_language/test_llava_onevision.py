from typing import List, Optional, Tuple, Type

import pytest
from transformers import (AutoConfig, AutoModelForVision2Seq, AutoTokenizer,
                          BatchEncoding)

from vllm.multimodal.utils import (rescale_image_size, rescale_video_size,
                                   resize_video, sample_frames_from_video)
from vllm.sequence import SampleLogprobs
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE

from ....conftest import (VIDEO_ASSETS, HfRunner, PromptImageInput,
                          PromptVideoInput, VllmRunner)
from ...utils import check_logprobs_close

# Video test
HF_VIDEO_PROMPTS = VIDEO_ASSETS.prompts({
    "sample_demo_1":
    "<|im_start|>user\n<video>\nwhy is this video funny?<|im_end|>\n<|im_start|>assistant\n"  # noqa: E501
})

models = ["llava-hf/llava-onevision-qwen2-0.5b-ov-hf"]


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


# Video test
_LIMIT_VIDEO_PER_PROMPT = 4


def run_video_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    inputs: List[Tuple[List[str], PromptVideoInput]],
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    num_frames: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]
    with vllm_runner(model,
                     dtype=dtype,
                     max_model_len=16384,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True,
                     limit_mm_per_prompt={"video": _LIMIT_VIDEO_PER_PROMPT
                                          }) as vllm_model:
        vllm_outputs_per_input = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                videos=videos)
            for prompts, videos in inputs
        ]

    def process(hf_inputs: BatchEncoding):
        hf_inputs["pixel_values_videos"] = hf_inputs["pixel_values_videos"] \
            .to(torch_dtype)  # type: ignore
        return hf_inputs

    with hf_runner(model,
                   dtype=dtype,
                   postprocess_inputs=process,
                   auto_cls=AutoModelForVision2Seq) as hf_model:
        hf_outputs_per_input = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    videos=videos)
            for prompts, videos in inputs
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_input,
                                        vllm_outputs_per_input):
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


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("num_frames", [16])
def test_models_multiple_video_inputs(hf_runner, vllm_runner, video_assets,
                                      model, dtype, max_tokens, num_logprobs,
                                      num_frames) -> None:
    video = sample_frames_from_video(video_assets[0].np_ndarrays, num_frames)
    inputs = [(
        [
            "<|im_start|>user <video><video>\nDescribe 2 videos. \
                <|im_end|><|im_start|>assistant\n",
            "<|im_start|>user <video><video>\nDescribe 2 videos. \
                <|im_end|><|im_start|>assistant\n",
            "<|im_start|>user <video><video><video><video>\nDescribe 4 videos. \
                <|im_end|><|im_start|>assistant\n",
            "<|im_start|>user <video>\nwhy is this video funny? \
                <|im_end|><|im_start|>assistant\n",
        ],
        [
            [video, video],
            # Images with different sizes and aspect-ratios
            [
                rescale_video_size(video, 0.1),
                video,
            ],
            [
                video,
                rescale_video_size(video, 0.25),
                resize_video(video, (183, 488)),
                resize_video(video, (488, 183))
            ],
            video,
        ])]
    run_video_test(
        hf_runner,
        vllm_runner,
        inputs,
        model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
        num_frames=num_frames,
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
