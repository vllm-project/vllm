# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM for running offline inference
with the correct prompt format on Qwen3-Omni (thinker only).
"""

from typing import NamedTuple

from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.lora.request import LoRARequest
from vllm.multimodal.image import convert_image_mode
from vllm.utils.argparse_utils import FlexibleArgumentParser


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
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "<|vision_start|><|video_pad|><|vision_end|>"
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
        "Describe the content of the video in details, then convert what the "
        "baby say into text."
    )
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    asset = VideoAsset(name="baby_reading", num_frames=16)
    audio = asset.get_audio(sampling_rate=16000)
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
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
        "<|audio_start|><|audio_pad|><|audio_end|>"
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


def get_multi_images_query() -> QueryResult:
    question = "What are the differences between these two images?"
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "image": [
                    convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB"),
                    convert_image_mode(ImageAsset("stop_sign").pil_image, "RGB"),
                ],
            },
        },
        limit_mm_per_prompt={
            "image": 2,
        },
    )


query_map = {
    "mixed_modalities": get_mixed_modalities_query,
    "use_audio_in_video": get_use_audio_in_video_query,
    "multi_audios": get_multi_audios_query,
    "multi_images": get_multi_images_query,
}


def main(args):
    model_name = args.model
    query_result = query_map[args.query_type]()

    enable_lora = args.lora_path is not None

    llm = LLM(
        model=model_name,
        max_model_len=args.max_model_len,
        max_num_seqs=5,
        limit_mm_per_prompt=query_result.limit_mm_per_prompt,
        seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=enable_lora,
        max_lora_rank=args.max_lora_rank,
        enable_tower_connector_lora=enable_lora,
        mm_processor_cache_gb=0 if enable_lora else None,
    )

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(temperature=0.2, max_tokens=256)

    lora_request = None
    if enable_lora:
        lora_request = LoRARequest(args.lora_name, args.lora_id, args.lora_path)

    outputs = llm.generate(
        query_result.inputs,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )

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
        default=0,
        help="Set the seed when initializing `vllm.LLM`.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Model name or path.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "-tp",
        type=int,
        default=1,
        help="Tensor parallel size for distributed inference.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0 to 1.0).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=12800,
        help="Maximum model context length.",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Optional path to the LoRA adapter weights.",
    )
    parser.add_argument(
        "--lora-name",
        type=str,
        default="vision",
        help="LoRA adapter name.",
    )
    parser.add_argument(
        "--lora-id",
        type=int,
        default=1,
        help="LoRA adapter ID.",
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=8,
        help="Maximum LoRA rank.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
