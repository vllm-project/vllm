# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace

from vllm import LLM, EngineArgs, PoolingParams
from vllm.utils import FlexibleArgumentParser


def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    # Set example specific arguments
    parser.set_defaults(model="jinaai/jina-embeddings-v3",
                        task="embed",
                        trust_remote_code=True)
    return parser.parse_args()


def main(args: Namespace):
    # Sample prompts.
    prompts = [
        "Follow the white rabbit.",  # English
        "Sigue al conejo blanco.",  # Spanish
        "Suis le lapin blanc.",  # French
        "跟着白兔走。",  # Chinese
        "اتبع الأرنب الأبيض.",  # Arabic
        "Folge dem weißen Kaninchen.",  # German
    ]

    # Create an LLM.
    # You should pass task="embed" for embedding models
    model = LLM(**vars(args))

    # Generate embedding. The output is a list of EmbeddingRequestOutputs.
    outputs = model.embed(prompts, pooling_params=PoolingParams(dimensions=32))

    # Print the outputs.
    print("\nGenerated Outputs:")
    print("-" * 60)
    for prompt, output in zip(prompts, outputs):
        embeds = output.outputs.embedding
        embeds_trimmed = ((str(embeds[:16])[:-1] +
                           ", ...]") if len(embeds) > 16 else embeds)
        print(f"Prompt: {prompt!r} \n"
              f"Embeddings: {embeds_trimmed} "
              f"(size={len(embeds)})")
        print("-" * 60)


if __name__ == "__main__":
    args = parse_args()
    main(args)
