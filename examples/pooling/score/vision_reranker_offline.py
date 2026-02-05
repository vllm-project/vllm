# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
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
from vllm.multimodal.utils import encode_image_url, fetch_image
from vllm.utils.argparse_utils import FlexibleArgumentParser

TEMPLATE_HOME = Path(__file__).parent / "template"


query = "A woman playing with her dog on a beach at sunset."
document = (
    "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, "
    "as the dog offers its paw in a heartwarming display of companionship and trust."
)
image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
video_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"
documents = [
    {
        "type": "text",
        "text": document,
    },
    {
        "type": "image_url",
        "image_url": {"url": image_url},
    },
    {
        "type": "image_url",
        "image_url": {"url": encode_image_url(fetch_image(image_url))},
    },
    {
        "type": "video_url",
        "video_url": {"url": video_url},
    },
]


class RerankModelData(NamedTuple):
    engine_args: EngineArgs
    chat_template: str | None = None
    modality: set[str] = {}


def run_jinavl_reranker() -> RerankModelData:
    engine_args = EngineArgs(
        model="jinaai/jina-reranker-m0",
        runner="pooling",
        max_model_len=32768,
        trust_remote_code=True,
        mm_processor_kwargs={
            "min_pixels": 3136,
            "max_pixels": 602112,
        },
    )
    return RerankModelData(engine_args=engine_args, modality={"image"})


def run_qwen3_vl_reranker() -> RerankModelData:
    engine_args = EngineArgs(
        model="Qwen/Qwen3-VL-Reranker-2B",
        runner="pooling",
        max_model_len=16384,
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
        modality={"image", "video"},
    )


model_example_map: dict[str, Callable[[], RerankModelData]] = {
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
    return parser.parse_args()


def main(args: Namespace):
    # Run the selected reranker model
    model_request = model_example_map[args.model_name]()
    engine_args = model_request.engine_args

    llm = LLM(**asdict(engine_args))

    print("Query: string & Document: string")
    outputs = llm.score(query, document)
    print("Relevance scores:", [output.outputs.score for output in outputs])

    print("Query: string & Document: text")
    outputs = llm.score(
        query, {"content": [documents[0]]}, chat_template=model_request.chat_template
    )
    print("Relevance scores:", [output.outputs.score for output in outputs])

    print("Query: string & Document: image url")
    outputs = llm.score(
        query, {"content": [documents[1]]}, chat_template=model_request.chat_template
    )
    print("Relevance scores:", [output.outputs.score for output in outputs])

    print("Query: string & Document: image base64")
    outputs = llm.score(
        query, {"content": [documents[2]]}, chat_template=model_request.chat_template
    )
    print("Relevance scores:", [output.outputs.score for output in outputs])

    if "video" in model_request.modality:
        print("Query: string & Document: video url")
        outputs = llm.score(
            query,
            {"content": [documents[3]]},
            chat_template=model_request.chat_template,
        )
        print("Relevance scores:", [output.outputs.score for output in outputs])

    print("Query: string & Document: text + image url")
    outputs = llm.score(
        query,
        {"content": [documents[0], documents[1]]},
        chat_template=model_request.chat_template,
    )
    print("Relevance scores:", [output.outputs.score for output in outputs])

    print("Query: string & Document: list")
    outputs = llm.score(
        query,
        [
            document,
            {"content": [documents[0]]},
            {"content": [documents[1]]},
            {"content": [documents[0], documents[1]]},
        ],
        chat_template=model_request.chat_template,
    )
    print("Relevance scores:", [output.outputs.score for output in outputs])


if __name__ == "__main__":
    args = parse_args()
    main(args)
