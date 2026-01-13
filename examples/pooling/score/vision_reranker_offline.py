# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM for running offline inference with
vision language reranker models for multimodal scoring tasks.

Vision language rerankers score the relevance between a text query and
multimodal documents (text + images/videos).
"""

from argparse import Namespace
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple

from vllm import LLM, EngineArgs
from vllm.entrypoints.score_utils import ScoreMultiModalParam
from vllm.utils.argparse_utils import FlexibleArgumentParser

TEMPLATE_HOME = Path(__file__).parent / "template"


class RerankModelData(NamedTuple):
    engine_args: EngineArgs
    chat_template: str | None = None


def run_jinavl_reranker(modality: str) -> RerankModelData:
    assert modality == "image"

    engine_args = EngineArgs(
        model="jinaai/jina-reranker-m0",
        runner="pooling",
        max_model_len=32768,
        trust_remote_code=True,
        mm_processor_kwargs={
            "min_pixels": 3136,
            "max_pixels": 602112,
        },
        limit_mm_per_prompt={modality: 1},
    )
    return RerankModelData(
        engine_args=engine_args,
    )


def run_qwen3_vl_reranker(modality: str) -> RerankModelData:
    engine_args = EngineArgs(
        model="Qwen/Qwen3-VL-Reranker-2B",
        runner="pooling",
        max_model_len=16384,
        limit_mm_per_prompt={modality: 1},
        # HuggingFace model configuration overrides required for compatibility
        hf_overrides={
            # Manually route to sequence classification architecture
            # This tells vLLM to use Qwen3VLForSequenceClassification instead of
            # the default Qwen3VLForConditionalGeneration
            "architectures": ["Qwen3VLForSequenceClassification"],
            # Specify which token logits to extract from the language model head
            # The original reranker uses "no" and "yes" token logits for scoring
            "classifier_from_token": ["no", "yes"],
            # Enable special handling for original Qwen3-Reranker models
            # This flag triggers conversion logic that transforms the two token
            # vectors into a single classification vector
            "is_original_qwen3_reranker": True,
        },
    )
    chat_template_path = "qwen3_vl_reranker.jinja"
    chat_template = (TEMPLATE_HOME / chat_template_path).read_text()
    return RerankModelData(
        engine_args=engine_args,
        chat_template=chat_template,
    )


model_example_map: dict[str, Callable[[str], RerankModelData]] = {
    "jinavl_reranker": run_jinavl_reranker,
    "qwen3_vl_reranker": run_qwen3_vl_reranker,
}


def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "vision language reranker models for multimodal scoring tasks."
    )
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="jinavl_reranker",
        choices=model_example_map.keys(),
        help="The name of the reranker model.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="image",
        choices=["image", "video"],
        help="Modality of the multimodal input (image or video).",
    )
    return parser.parse_args()


def get_multi_modal_input(modality: str) -> tuple[str, ScoreMultiModalParam]:
    # Sample query for testing the reranker
    if modality == "image":
        query = "A woman playing with her dog on a beach at sunset."
        # Sample multimodal documents to be scored against the query
        # Each document contains an image URL that will be fetched and processed
        documents: ScoreMultiModalParam = {
            "content": [
                {
                    "type": "text",
                    "text": (
                        "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, "  # noqa: E501
                        "as the dog offers its paw in a heartwarming display of companionship and trust."  # noqa: E501
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                    },
                },
            ]
        }
    elif modality == "video":
        query = "A girl is drawing pictures on an ipad."
        # Sample video documents to be scored against the query
        documents: ScoreMultiModalParam = {
            "content": [
                {
                    "type": "text",
                    "text": "A girl is drawing a guitar on her ipad with Apple Pencil.",
                },
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"
                    },
                },
            ]
        }
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    return query, documents


def main(args: Namespace):
    # Run the selected reranker model
    modality = args.modality
    model_request = model_example_map[args.model_name](modality)
    engine_args = model_request.engine_args

    llm = LLM(**asdict(engine_args))

    query, documents = get_multi_modal_input(modality)
    outputs = llm.score(query, documents, chat_template=model_request.chat_template)

    print("-" * 50)
    print(f"Model: {engine_args.model}")
    print(f"Modality: {modality}")
    print(f"Query: {query}")
    print("Relevance scores:", [output.outputs.score for output in outputs])
    print("-" * 50)


if __name__ == "__main__":
    args = parse_args()
    main(args)
