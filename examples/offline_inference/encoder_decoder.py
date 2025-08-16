# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrate prompting of text-to-text
encoder/decoder models, specifically BART and mBART.

This script is refactored to allow model selection via command-line arguments.
"""

import argparse
from typing import NamedTuple, Optional

from vllm import LLM, SamplingParams
from vllm.inputs import (
    ExplicitEncoderDecoderPrompt,
    TextPrompt,
    TokensPrompt,
    zip_enc_dec_prompts,
)


class ModelRequestData(NamedTuple):
    """
    Holds the configuration for a specific model, including its
    HuggingFace ID and the prompts to use for the demo.
    """

    model_id: str
    encoder_prompts: list
    decoder_prompts: list
    hf_overrides: Optional[dict] = None


def get_bart_config() -> ModelRequestData:
    """
    Returns the configuration for facebook/bart-large-cnn.
    This uses the exact test cases from the original script.
    """
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
    return ModelRequestData(
        model_id="facebook/bart-large-cnn",
        encoder_prompts=encoder_prompts,
        decoder_prompts=decoder_prompts,
    )


def get_mbart_config() -> ModelRequestData:
    """
    Returns the configuration for facebook/mbart-large-en-ro.
    This uses prompts suitable for an English-to-Romanian translation task.
    """
    encoder_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "How are you today?",
    ]
    decoder_prompts = ["", ""]
    hf_overrides = {"architectures": ["MBartForConditionalGeneration"]}
    return ModelRequestData(
        model_id="facebook/mbart-large-en-ro",
        encoder_prompts=encoder_prompts,
        decoder_prompts=decoder_prompts,
        hf_overrides=hf_overrides,
    )


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
    text_prompt_raw = encoder_prompts_raw[0]
    text_prompt = TextPrompt(prompt=encoder_prompts_raw[1 % len(encoder_prompts_raw)])
    tokens_prompt = TokensPrompt(
        prompt_token_ids=tokenizer.encode(
            encoder_prompts_raw[2 % len(encoder_prompts_raw)]
        )
    )

    decoder_tokens_prompt = TokensPrompt(
        prompt_token_ids=tokenizer.encode(decoder_prompts_raw[0])
    )
    single_prompt_examples = [
        text_prompt_raw,
        text_prompt,
        tokens_prompt,
    ]
    explicit_pair_examples = [
        ExplicitEncoderDecoderPrompt(
            encoder_prompt=text_prompt_raw,
            decoder_prompt=decoder_tokens_prompt,
        ),
        ExplicitEncoderDecoderPrompt(
            encoder_prompt=text_prompt,
            decoder_prompt=decoder_prompts_raw[1 % len(decoder_prompts_raw)],
        ),
        ExplicitEncoderDecoderPrompt(
            encoder_prompt=tokens_prompt,
            decoder_prompt=text_prompt,
        ),
    ]
    zipped_prompt_list = zip_enc_dec_prompts(
        encoder_prompts_raw,
        decoder_prompts_raw,
    )
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
    if model_key not in MODEL_GETTERS:
        raise ValueError(
            f"Unknown model: {model_key}. "
            f"Available models: {list(MODEL_GETTERS.keys())}"
        )
    config_getter = MODEL_GETTERS[model_key]
    model_config = config_getter()

    print(f"🚀 Running demo for model: {model_config.model_id}")
    llm = LLM(
        model=model_config.model_id,
        dtype="float",
        hf_overrides=model_config.hf_overrides,
    )
    tokenizer = llm.llm_engine.get_tokenizer_group()
    prompts = create_all_prompt_types(
        encoder_prompts_raw=model_config.encoder_prompts,
        decoder_prompts_raw=model_config.decoder_prompts,
        tokenizer=tokenizer,
    )
    sampling_params = create_sampling_params()
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
