# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use vLLM for running offline inference with
the explicit/implicit prompt format on enc-dec LMMs for text generation.
"""
import time

from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.utils import FlexibleArgumentParser


def run_mllama():
    # Create a Mllama encoder/decoder model instance
    llm = LLM(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct",
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": 1},
        dtype="half",
    )

    # yapf: disable
    prompts = [
        # Implicit prompt
        {
            "prompt": "<|image|><|begin_of_text|>What is the content of this image?",   # noqa: E501
            "multi_modal_data": {
                "image": ImageAsset("stop_sign").pil_image,
            },
        },
        # Explicit prompt
        {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "image": ImageAsset("stop_sign").pil_image,
                },
            },
            "decoder_prompt": "<|image|><|begin_of_text|>Please describe the image.",   # noqa: E501
        },
    ]
    # yapf: enable
    return llm, prompts


def run_whisper():
    # Create a Whisper encoder/decoder model instance
    llm = LLM(
        model="openai/whisper-large-v3",
        max_model_len=448,
        max_num_seqs=2,
        limit_mm_per_prompt={"audio": 1},
        dtype="half",
    )

    prompts = [
        # Test implicit prompt
        {
            "prompt": "<|startoftranscript|>",
            "multi_modal_data": {
                "audio": AudioAsset("mary_had_lamb").audio_and_sample_rate,
            },
        },
        # Test explicit encoder/decoder prompt
        {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "audio": AudioAsset("winning_call").audio_and_sample_rate,
                },
            },
            "decoder_prompt": "<|startoftranscript|>",
        }
    ]
    return llm, prompts


model_example_map = {
    "mllama": run_mllama,
    "whisper": run_whisper,
}


def main(args):
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    llm, prompts = model_example_map[model]()

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
        encoder_prompt = output.encoder_prompt
        generated_text = output.outputs[0].text
        print(f"Encoder prompt: {encoder_prompt!r}, "
              f"Decoder prompt: {prompt!r}, "
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

    args = parser.parse_args()
    main(args)
