# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use vLLM for running offline inference with
the explicit/implicit prompt format on enc-dec LMMs for text generation.
"""
import time
from collections.abc import Sequence
from dataclasses import asdict
from typing import NamedTuple

from vllm import LLM, EngineArgs, PromptType, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.utils import FlexibleArgumentParser


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: Sequence[PromptType]


def run_florence2():
    engine_args = EngineArgs(
        model="microsoft/Florence-2-large",
        tokenizer="facebook/bart-large",
        max_num_seqs=8,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
        dtype="half",
    )

    prompts = [
        {   # implicit prompt with task token
            "prompt": "<DETAILED_CAPTION>",
            "multi_modal_data": {
                "image": ImageAsset("stop_sign").pil_image
            },
        },
        {   # explicit encoder/decoder prompt
            "encoder_prompt": {
                "prompt": "Describe in detail what is shown in the image.",
                "multi_modal_data": {
                    "image": ImageAsset("cherry_blossom").pil_image
                },
            },
            "decoder_prompt": "",
        },
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


def run_mllama():
    engine_args = EngineArgs(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct",
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": 1},
        dtype="half",
    )

    prompts = [
        {   # Implicit prompt
            "prompt": "<|image|><|begin_of_text|>What is the content of this image?",   # noqa: E501
            "multi_modal_data": {
                "image": ImageAsset("stop_sign").pil_image,
            },
        },
        {   # Explicit prompt
            "encoder_prompt": {
                "prompt": "<|image|>",
                "multi_modal_data": {
                    "image": ImageAsset("stop_sign").pil_image,
                },
            },
            "decoder_prompt": "<|image|><|begin_of_text|>Please describe the image.",   # noqa: E501
        },
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


def run_whisper():
    engine_args = EngineArgs(
        model="openai/whisper-large-v3-turbo",
        max_model_len=448,
        max_num_seqs=16,
        limit_mm_per_prompt={"audio": 1},
        dtype="half",
    )

    prompts = [
        {   # Test implicit prompt
            "prompt": "<|startoftranscript|>",
            "multi_modal_data": {
                "audio": AudioAsset("mary_had_lamb").audio_and_sample_rate,
            },
        },
        {   # Test explicit encoder/decoder prompt
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "audio": AudioAsset("winning_call").audio_and_sample_rate,
                },
            },
            "decoder_prompt": "<|startoftranscript|>",
        }
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


model_example_map = {
    "florence2": run_florence2,
    "mllama": run_mllama,
    "whisper": run_whisper,
}


def main(args):
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    req_data = model_example_map[model]()

    engine_args = asdict(req_data.engine_args) | {"seed": args.seed}
    llm = LLM(**engine_args)

    prompts = req_data.prompts

    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=64,
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
        print(f"Decoder prompt: {prompt!r}, "
              f"Generated text: {generated_text!r}")

    duration = time.time() - start

    print("Duration:", duration)
    print("RPS:", len(prompts) / duration)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models for text generation')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="mllama",
                        choices=model_example_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="Set the seed when initializing `vllm.LLM`.")

    args = parser.parse_args()
    main(args)
