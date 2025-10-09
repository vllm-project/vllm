# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://huggingface.co/boltuix/NeuroBERT-NER

from argparse import Namespace

from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser


def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    # Set example specific arguments
    parser.set_defaults(
        model="boltuix/NeuroBERT-NER",
        runner="pooling",
        enforce_eager=True,
        trust_remote_code=True,
    )
    return parser.parse_args()


def main(args: Namespace):
    # Sample prompts.
    prompts = [
        "Barack Obama visited Microsoft headquarters in Seattle on January 2025."
    ]

    # Create an LLM.
    llm = LLM(**vars(args))
    tokenizer = llm.get_tokenizer()
    label_map = llm.llm_engine.vllm_config.model_config.hf_config.id2label

    # Run inference
    outputs = llm.encode(prompts)

    for prompt, output in zip(prompts, outputs):
        logits = output.outputs.data
        predictions = logits.argmax(dim=-1)

        # Map predictions to labels
        tokens = tokenizer.convert_ids_to_tokens(output.prompt_token_ids)
        labels = [label_map[p.item()] for p in predictions]

        # Print results
        for token, label in zip(tokens, labels):
            if token not in tokenizer.all_special_tokens:
                print(f"{token:15} → {label}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
