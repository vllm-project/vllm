# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
from typing import NamedTuple

from vllm import LLM, SamplingParams
from vllm.inputs import (
    ExplicitEncoderDecoderPrompt,
    TextPrompt,
    TokensPrompt,
    zip_enc_dec_prompts,
)


class ModelConfig(NamedTuple):
    model_id: str
    encoder_prompts: list
    decoder_prompts: list


def get_bart_config() -> ModelConfig:
    encoder_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "An encoder prompt",
    ]
    decoder_prompts = [
        "A decoder prompt",
        "Another decoder prompt",
    ]
    return ModelConfig(
        model_id="facebook/bart-large-cnn",
        encoder_prompts=encoder_prompts,
        decoder_prompts=decoder_prompts,
    )


def get_mbart_config() -> ModelConfig:
    """
    Returns the configuration for facebook/mbart-large-en-ro.
    This uses prompts suitable for an English-to-Romanian translation task.
    """
    encoder_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "How are you today?",
    ]
    decoder_prompts = ["", ""]
    return ModelConfig(
        model_id="facebook/mbart-large-en-ro",
        encoder_prompts=encoder_prompts,
        decoder_prompts=decoder_prompts,
    )


# A dictionary to map model short names to their configuration getters.
MODEL_GETTERS = {
    "bart": get_bart_config,
    "mbart": get_mbart_config,
}


def create_all_prompt_types(
    encoder_prompts_raw: list,
    decoder_prompts_raw: list,
    tokenizer,
) -> list:
    """
    Generates a list of diverse prompt types for demonstration.
    This function is generic and uses the provided raw prompts
    to create various vLLM input objects.
    """
    # 1. Create basic prompt objects from the raw text.
    text_prompt_raw = encoder_prompts_raw[0]
    text_prompt = TextPrompt(prompt=encoder_prompts_raw[1])
    tokens_prompt = TokensPrompt(
        prompt_token_ids=tokenizer.encode(encoder_prompts_raw[2])
    )

    decoder_tokens_prompt = TokensPrompt(
        prompt_token_ids=tokenizer.encode(decoder_prompts_raw[0])
    )

    # 2. Demonstrate passing a single prompt (implicitly for the encoder).
    single_prompt_examples = [
        text_prompt_raw,  # Pass a raw string
        text_prompt,  # Pass a TextPrompt object
        tokens_prompt,  # Pass a TokensPrompt object
    ]

    # 3. Demonstrate passing explicit encoder/decoder pairs.
    #    Note the flexibility in mixing prompt types.
    explicit_pair_examples = [
        ExplicitEncoderDecoderPrompt(
            encoder_prompt=text_prompt_raw,
            decoder_prompt=decoder_tokens_prompt,
        ),
        ExplicitEncoderDecoderPrompt(
            encoder_prompt=text_prompt,
            decoder_prompt=decoder_prompts_raw[1],
        ),
        ExplicitEncoderDecoderPrompt(
            encoder_prompt=tokens_prompt,
            decoder_prompt=text_prompt,
        ),
    ]

    # 4. Demonstrate the convenience helper for zipping lists of prompts.
    zipped_prompt_list = zip_enc_dec_prompts(
        encoder_prompts=encoder_prompts_raw,
        decoder_prompts=decoder_prompts_raw,
    )

    # 5. Combine all examples into a single list for processing.
    return single_prompt_examples + explicit_pair_examples + zipped_prompt_list


def create_sampling_params() -> SamplingParams:
    """Create a sampling params object."""
    return SamplingParams(
        temperature=0,
        top_p=1.0,
        min_tokens=0,
        max_tokens=30,
    )


def print_outputs(outputs: list):
    """Formats and prints the generation outputs."""
    print("-" * 80)
    for i, output in enumerate(outputs):
        prompt = output.prompt
        encoder_prompt = output.encoder_prompt
        generated_text = output.outputs[0].text
        print(f"Output {i + 1}:")
        print(f"Encoder Prompt: {encoder_prompt!r}")
        print(f"Decoder Prompt: {prompt!r}")
        print(f"Generated Text: {generated_text!r}")
        print("-" * 80)


def main(args):
    """Main execution function."""
    model_key = args.model

    # 1. Get the configuration for the selected model.
    if model_key not in MODEL_GETTERS:
        raise ValueError(
            f"Unknown model: {model_key}. "
            f"Available models: {list(MODEL_GETTERS.keys())}"
        )
    config_getter = MODEL_GETTERS[model_key]
    model_config = config_getter()

    print(f"🚀 Running demo for model: {model_config.model_id}")

    # 2. Create the vLLM engine instance.
    llm = LLM(
        model=model_config.model_id,
        dtype="float",
    )

    # 3. Get the tokenizer and create the list of prompts.
    tokenizer = llm.llm_engine.get_tokenizer_group()
    prompts = create_all_prompt_types(
        encoder_prompts_raw=model_config.encoder_prompts,
        decoder_prompts_raw=model_config.decoder_prompts,
        tokenizer=tokenizer,
    )

    # 4. Create sampling parameters.
    sampling_params = create_sampling_params()

    # 5. Generate text from the prompts and print the outputs.
    outputs = llm.generate(prompts, sampling_params)
    print_outputs(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A flexible demo for vLLM encoder-decoder models."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="bart",
        choices=MODEL_GETTERS.keys(),
        help="The short name of the model to run.",
    )
    args = parser.parse_args()
    main(args)
