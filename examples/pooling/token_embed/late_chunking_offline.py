# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace

from vllm import LLM, EngineArgs, PoolingParams
from vllm.config import PoolerConfig
from vllm.pooling_params import LateChunkingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser


def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Number of model-input tokens per chunk, including special tokens.",
    )
    parser.set_defaults(
        model="intfloat/multilingual-e5-small",
        runner="pooling",
        pooler_config=PoolerConfig(task="token_embed"),
        max_model_len=512,
        enforce_eager=True,
    )
    return parser.parse_args()


def main(args: Namespace):
    pooling_params = PoolingParams(
        late_chunking_params=LateChunkingParams(chunk_size=args.chunk_size)
    )
    engine_args = vars(args).copy()
    del engine_args["chunk_size"]
    llm = LLM(**engine_args)

    # E5 document inputs use the "passage: " prefix; it also counts toward chunks.
    documents = [
        "passage: Berlin is the capital of Germany. The city has many museums. "
        "They display art and historical collections from around the world.",
        "passage: Kyoto is a city in Japan. Its historic temples attract visitors "
        "throughout the year.",
    ]
    outputs = llm.encode(
        documents,
        pooling_task="token_embed",
        pooling_params=pooling_params,
    )

    for doc_index, (document, output) in enumerate(zip(documents, outputs)):
        vectors = output.outputs.data
        metadata = output.late_chunking
        assert metadata is not None
        print(f"\nDocument {doc_index}: {metadata.input_tokens} input tokens")
        print(f"Chunk vectors: {tuple(vectors.shape)}")
        for index, chunk in enumerate(metadata.chunks):
            # vectors[index] corresponds to this chunk and its source-text range.
            if chunk.char_range is None:
                excerpt = "<special tokens only>"
            else:
                start, end = chunk.char_range
                excerpt = document[start:end]
            print(
                f"  Chunk {index}: tokens={chunk.token_range}, chars={chunk.char_range}"
            )
            print(f"    {excerpt!r}")


if __name__ == "__main__":
    main(parse_args())
