# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM for running offline inference
with the correct prompt format on Qwen2.5-Omni (thinker only).
"""

from typing import NamedTuple

import vllm.envs as envs
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.multimodal.image import convert_image_mode
from vllm.utils import FlexibleArgumentParser


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.

default_system = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)


def get_mixed_modalities_query() -> QueryResult:
    question = (
        "What is recited in the audio? "
        "What is the content of this image? Why is this video funny?"
    )
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>"
        "<|vision_bos|><|IMAGE|><|vision_eos|>"
        "<|vision_bos|><|VIDEO|><|vision_eos|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "audio": AudioAsset("mary_had_lamb").audio_and_sample_rate,
                "image": convert_image_mode(
                    ImageAsset("cherry_blossom").pil_image, "RGB"
                ),
                "video": VideoAsset(name="baby_reading", num_frames=16).np_ndarrays,
            },
        },
        limit_mm_per_prompt={"audio": 1, "image": 1, "video": 1},
    )


def get_use_audio_in_video_query() -> QueryResult:
    question = (
        "Describe the content of the video, then convert what the baby say into text."
    )
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    asset = VideoAsset(name="baby_reading", num_frames=16)
    audio = asset.get_audio(sampling_rate=16000)
    assert not envs.VLLM_USE_V1, (
        "V1 does not support use_audio_in_video. "
        "Please launch this example with "
        "`VLLM_USE_V1=0`."
    )
    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "video": asset.np_ndarrays,
                "audio": audio,
            },
            "mm_processor_kwargs": {
                "use_audio_in_video": True,
            },
        },
        limit_mm_per_prompt={"audio": 1, "video": 1},
    )


def get_multi_audios_query() -> QueryResult:
    question = "Are these two audio clips the same?"
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>"
        "<|audio_bos|><|AUDIO|><|audio_eos|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "audio": [
                    AudioAsset("winning_call").audio_and_sample_rate,
                    AudioAsset("mary_had_lamb").audio_and_sample_rate,
                ],
            },
        },
        limit_mm_per_prompt={
            "audio": 2,
        },
    )


query_map = {
    "mixed_modalities": get_mixed_modalities_query,
    "use_audio_in_video": get_use_audio_in_video_query,
    "multi_audios": get_multi_audios_query,
}


def main(args):
    model_name = "Qwen/Qwen2.5-Omni-7B"
    query_result = query_map[args.query_type]()

    llm = LLM(
        model=model_name,
        max_model_len=5632,
        max_num_seqs=5,
        limit_mm_per_prompt=query_result.limit_mm_per_prompt,
        seed=args.seed,
    )

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(temperature=0.2, max_tokens=64)

    outputs = llm.generate(query_result.inputs, sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "audio language models"
    )
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="mixed_modalities",
        choices=query_map.keys(),
        help="Query type.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set the seed when initializing `vllm.LLM`.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
