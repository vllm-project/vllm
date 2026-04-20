# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM for running offline inference with
the explicit/implicit prompt format on enc-dec LMMs for text generation.
"""

import os
import time
from collections.abc import Sequence
from typing import NamedTuple

from vllm import LLM, EngineArgs, PromptType, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.utils.argparse_utils import FlexibleArgumentParser


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: Sequence[PromptType]


def run_whisper():
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    engine_args = EngineArgs(
        model="openai/whisper-large-v3-turbo",
        max_model_len=448,
        max_num_seqs=16,
        limit_mm_per_prompt={"audio": 1},
        dtype="half",
    )

    prompts = [
        {  # Test implicit prompt
            "prompt": "<|startoftranscript|>",
            "multi_modal_data": {
                "audio": AudioAsset("mary_had_lamb").audio_and_sample_rate,
            },
        },
        {  # Test explicit encoder/decoder prompt
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "audio": AudioAsset("winning_call").audio_and_sample_rate,
                },
            },
            "decoder_prompt": "<|startoftranscript|>",
        },
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


def run_fireredasr2():
    """
    FireRedASR2 – Automatic Speech Recognition model.

    This model uses a Conformer encoder + Qwen2 LLM decoder architecture
    for speech-to-text transcription.  Audio is passed via the implicit
    prompt format with the ``<|AUDIO|>`` placeholder token.
    """
    engine_args = EngineArgs(
        model="allendou/FireRedASR2-LLM-vllm",
        max_model_len=448,
        max_num_seqs=16,
        limit_mm_per_prompt={"audio": 1},
    )

    prompt_str = (
        "<|im_start|>user\n<|AUDIO|>请转写音频为文字<|im_end|>\n<|im_start|>assistant\n"
    )

    prompts = [
        {  # Implicit prompt with audio
            "prompt": prompt_str,
            "multi_modal_data": {
                "audio": AudioAsset("mary_had_lamb").audio_and_sample_rate,
            },
        },
        {  # Another audio sample
            "prompt": prompt_str,
            "multi_modal_data": {
                "audio": AudioAsset("winning_call").audio_and_sample_rate,
            },
        },
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


def run_fireredlid():
    """
    FireRedLID – Language Identification model.

    This encoder-decoder model identifies the spoken language of an audio
    clip. It outputs at most 2 tokens representing the detected language
    (e.g. "en", "zh mandarin").
    """
    engine_args = EngineArgs(
        model="PatchyTisa/FireRedLID-vllm",
        max_model_len=8,
        max_num_seqs=16,
        limit_mm_per_prompt={"audio": 1},
    )

    prompts = [
        {  # Test explicit encoder/decoder prompt
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "audio": AudioAsset("mary_had_lamb").audio_and_sample_rate,
                },
            },
            "decoder_prompt": "<sos>",
        },
        {  # Another audio sample
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "audio": AudioAsset("winning_call").audio_and_sample_rate,
                },
            },
            "decoder_prompt": "<sos>",
        },
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


model_example_map = {
    "fireredasr2": run_fireredasr2,
    "fireredlid": run_fireredlid,
    "whisper": run_whisper,
}


def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "vision language models for text generation"
    )
    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        default="whisper",
        choices=model_example_map.keys(),
        help='Huggingface "model_type".',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set the seed when initializing `vllm.LLM`.",
    )
    return parser.parse_args()


def main(args):
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    req_data = model_example_map[model]()

    # Disable other modalities to save memory
    engine_args = req_data.engine_args
    default_limits = {"image": 0, "video": 0, "audio": 0}
    limit_mm_per_prompt = default_limits | (engine_args.limit_mm_per_prompt or {})
    engine_args.limit_mm_per_prompt = limit_mm_per_prompt
    engine_args.seed = args.seed
    llm = LLM.from_engine_args(engine_args)

    prompts = req_data.prompts

    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=64,
        skip_special_tokens=False,
    )

    start = time.time()

    # Generate output tokens from the prompts. The output is a list of
    # RequestOutput objects that contain the prompt, generated
    # text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Decoder prompt: {prompt!r}, Generated text: {generated_text!r}")

    duration = time.time() - start

    print("Duration:", duration)
    print("RPS:", len(prompts) / duration)


if __name__ == "__main__":
    args = parse_args()
    main(args)
