# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.multimodal.image import convert_image_mode

models = ["llava-hf/llava-onevision-qwen2-0.5b-ov-hf"]


def base_prompt(modalities_str: str) -> str:
    return f"<|im_start|>user {modalities_str}\nDescribe what you see from these items.<|im_end|><|im_start|>assistant\n"  # noqa: E501


INTERLEAVED_PROMPT = base_prompt("<image><video><image>\n")
NONINTERLEAVED_PROMPT = base_prompt("<image><image><video>\n")


@pytest.mark.core_model
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize("max_tokens", [128])
def test_models(vllm_runner, model, dtype: str, max_tokens: int) -> None:
    """
    This is a simple test to check if interleaved and non-interleaved prompts
    give the same result.
    """

    image_cherry = convert_image_mode(
        ImageAsset("cherry_blossom").pil_image, "RGB")
    image_stop = convert_image_mode(ImageAsset("stop_sign").pil_image, "RGB")
    images = [image_cherry, image_stop]
    video = VideoAsset(name="baby_reading", num_frames=16).np_ndarrays

    inputs = [
        (
            [INTERLEAVED_PROMPT],
            [images],
            [video],
        ),
        (
            [NONINTERLEAVED_PROMPT],
            [images],
            [video],
        ),
    ]

    with vllm_runner(model,
                     task="generate",
                     dtype=dtype,
                     limit_mm_per_prompt={"image": 2},
                     max_model_len=32768,
                     max_num_seqs=2,
                     tensor_parallel_size=1,
                     enforce_eager=True) as vllm_model:
        vllm_outputs_per_case = [
            vllm_model.generate_greedy(prompts,
                                       max_tokens,
                                       images=images,
                                       videos=videos)
            for prompts, images, videos in inputs
        ]

    all_results = [output[0][1] for output in vllm_outputs_per_case]
    outputs = [(total_str, total_str.find("assistant\n") + len("assistant\n"))
               for total_str in all_results]
    prompt_lengths = [prompt_len for _, prompt_len in outputs]
    generated_strs = [
        total_str[prompt_len:] for total_str, prompt_len in outputs
    ]
    interleaved_prompt_len, noninterleaved_prompt_len = prompt_lengths
    interleaved_output_str, noninterleaved_output_str = generated_strs

    # The two prompts are identical except for the order of modality tokens.
    assert interleaved_prompt_len == noninterleaved_prompt_len

    # The two generated strings should be different because of the
    # interleaved modality tokens.
    assert interleaved_output_str != noninterleaved_output_str
