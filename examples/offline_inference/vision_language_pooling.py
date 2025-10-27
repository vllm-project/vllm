# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for multimodal pooling.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""

from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from typing import Literal, NamedTuple, TypeAlias, TypedDict, get_args

from PIL.Image import Image

from vllm import LLM, EngineArgs
from vllm.entrypoints.score_utils import ScoreMultiModalParam
from vllm.multimodal.utils import fetch_image
from vllm.utils.argparse_utils import FlexibleArgumentParser

ROOT_DIR = Path(__file__).parent.parent.parent
EXAMPLES_DIR = ROOT_DIR / "examples"


class TextQuery(TypedDict):
    modality: Literal["text"]
    text: str


class ImageQuery(TypedDict):
    modality: Literal["image"]
    image: Image


class TextImageQuery(TypedDict):
    modality: Literal["text+image"]
    text: str
    image: Image


class TextImagesQuery(TypedDict):
    modality: Literal["text+images"]
    text: str
    image: ScoreMultiModalParam


QueryModality = Literal["text", "image", "text+image", "text+images"]
Query: TypeAlias = TextQuery | ImageQuery | TextImageQuery | TextImagesQuery


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str | None = None
    image: Image | None = None
    query: str | None = None
    documents: ScoreMultiModalParam | None = None


def run_clip(query: Query) -> ModelRequestData:
    if query["modality"] == "text":
        prompt = query["text"]
        image = None
    elif query["modality"] == "image":
        prompt = ""  # For image input, make sure that the prompt text is empty
        image = query["image"]
    else:
        modality = query["modality"]
        raise ValueError(f"Unsupported query modality: '{modality}'")

    engine_args = EngineArgs(
        model="openai/clip-vit-base-patch32",
        runner="pooling",
        limit_mm_per_prompt={"image": 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image=image,
    )


def run_e5_v(query: Query) -> ModelRequestData:
    llama3_template = "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n"  # noqa: E501

    if query["modality"] == "text":
        text = query["text"]
        prompt = llama3_template.format(f"{text}\nSummary above sentence in one word: ")
        image = None
    elif query["modality"] == "image":
        prompt = llama3_template.format("<image>\nSummary above image in one word: ")
        image = query["image"]
    else:
        modality = query["modality"]
        raise ValueError(f"Unsupported query modality: '{modality}'")

    engine_args = EngineArgs(
        model="royokong/e5-v",
        runner="pooling",
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image=image,
    )


def run_jinavl_reranker(query: Query) -> ModelRequestData:
    if query["modality"] != "text+images":
        raise ValueError(f"Unsupported query modality: '{query['modality']}'")

    engine_args = EngineArgs(
        model="jinaai/jina-reranker-m0",
        runner="pooling",
        max_model_len=32768,
        trust_remote_code=True,
        mm_processor_kwargs={
            "min_pixels": 3136,
            "max_pixels": 602112,
        },
        limit_mm_per_prompt={"image": 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        query=query["text"],
        documents=query["image"],
    )


def run_siglip(query: Query) -> ModelRequestData:
    if query["modality"] == "text":
        prompt = query["text"]
        image = None
    elif query["modality"] == "image":
        prompt = ""  # For image input, make sure that the prompt text is empty
        image = query["image"]
    else:
        modality = query["modality"]
        raise ValueError(f"Unsupported query modality: '{modality}'")

    engine_args = EngineArgs(
        model="google/siglip-base-patch16-224",
        runner="pooling",
        limit_mm_per_prompt={"image": 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image=image,
    )


def _get_vlm2vec_prompt_image(query: Query, image_token: str):
    if query["modality"] == "text":
        text = query["text"]
        prompt = f"Find me an everyday image that matches the given caption: {text}"
        image = None
    elif query["modality"] == "image":
        prompt = f"{image_token} Find a day-to-day image that looks similar to the provided image."  # noqa: E501
        image = query["image"]
    elif query["modality"] == "text+image":
        text = query["text"]
        prompt = f"{image_token} Represent the given image with the following question: {text}"  # noqa: E501
        image = query["image"]
    else:
        modality = query["modality"]
        raise ValueError(f"Unsupported query modality: {modality!r}")

    return prompt, image


def run_vlm2vec_phi3v(query: Query) -> ModelRequestData:
    prompt, image = _get_vlm2vec_prompt_image(query, "<|image_1|>")

    engine_args = EngineArgs(
        model="TIGER-Lab/VLM2Vec-Full",
        runner="pooling",
        max_model_len=4096,
        trust_remote_code=True,
        mm_processor_kwargs={"num_crops": 4},
        limit_mm_per_prompt={"image": 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image=image,
    )


def run_vlm2vec_qwen2vl(query: Query) -> ModelRequestData:
    # vLLM does not support LoRA adapters on multi-modal encoder,
    # so we merge the weights first
    from huggingface_hub.constants import HF_HUB_CACHE
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from vllm.entrypoints.chat_utils import load_chat_template

    model_id = "TIGER-Lab/VLM2Vec-Qwen2VL-2B"

    base_model = AutoModelForImageTextToText.from_pretrained(model_id)
    lora_model = PeftModel.from_pretrained(
        base_model,
        model_id,
        config=PeftConfig.from_pretrained(model_id),
    )
    model = lora_model.merge_and_unload().to(dtype=base_model.dtype)
    model._hf_peft_config_loaded = False  # Needed to save the merged model

    processor = AutoProcessor.from_pretrained(
        model_id,
        # `min_pixels` and `max_pixels` are deprecated for
        # transformers `preprocessor_config.json`
        size={"shortest_edge": 3136, "longest_edge": 12845056},
    )
    processor.chat_template = load_chat_template(
        # The original chat template is not correct
        EXAMPLES_DIR / "template_vlm2vec_qwen2vl.jinja",
    )

    merged_path = str(
        Path(HF_HUB_CACHE) / ("models--" + model_id.replace("/", "--") + "-vllm")
    )
    print(f"Saving merged model to {merged_path}...")
    print(
        "NOTE: This directory is not tracked by `huggingface_hub` "
        "so you have to delete this manually if you don't want it anymore."
    )
    model.save_pretrained(merged_path)
    processor.save_pretrained(merged_path)
    print("Done!")

    prompt, image = _get_vlm2vec_prompt_image(query, "<|image_pad|>")

    engine_args = EngineArgs(
        model=merged_path,
        runner="pooling",
        max_model_len=4096,
        mm_processor_kwargs={
            "min_pixels": 3136,
            "max_pixels": 12845056,
        },
        limit_mm_per_prompt={"image": 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image=image,
    )


def get_query(modality: QueryModality):
    if modality == "text":
        return TextQuery(modality="text", text="A dog sitting in the grass")

    if modality == "image":
        return ImageQuery(
            modality="image",
            image=fetch_image(
                "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/American_Eskimo_Dog.jpg/360px-American_Eskimo_Dog.jpg"  # noqa: E501
            ),
        )

    if modality == "text+image":
        return TextImageQuery(
            modality="text+image",
            text="A cat standing in the snow.",
            image=fetch_image(
                "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Felis_catus-cat_on_snow.jpg/179px-Felis_catus-cat_on_snow.jpg"  # noqa: E501
            ),
        )

    if modality == "text+images":
        return TextImagesQuery(
            modality="text+images",
            text="slm markdown",
            image={
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png"
                        },
                    },
                ]
            },
        )

    msg = f"Modality {modality} is not supported."
    raise ValueError(msg)


def run_encode(model: str, modality: QueryModality, seed: int | None):
    query = get_query(modality)
    req_data = model_example_map[model](query)

    # Disable other modalities to save memory
    default_limits = {"image": 0, "video": 0, "audio": 0}
    req_data.engine_args.limit_mm_per_prompt = default_limits | dict(
        req_data.engine_args.limit_mm_per_prompt or {}
    )

    engine_args = asdict(req_data.engine_args) | {"seed": seed}
    llm = LLM(**engine_args)

    mm_data = {}
    if req_data.image is not None:
        mm_data["image"] = req_data.image

    outputs = llm.embed(
        {
            "prompt": req_data.prompt,
            "multi_modal_data": mm_data,
        }
    )

    print("-" * 50)
    for output in outputs:
        print(output.outputs.embedding)
        print("-" * 50)


def run_score(model: str, modality: QueryModality, seed: int | None):
    query = get_query(modality)
    req_data = model_example_map[model](query)

    engine_args = asdict(req_data.engine_args) | {"seed": seed}
    llm = LLM(**engine_args)

    outputs = llm.score(req_data.query, req_data.documents)

    print("-" * 30)
    print([output.outputs.score for output in outputs])
    print("-" * 30)


model_example_map = {
    "clip": run_clip,
    "e5_v": run_e5_v,
    "jinavl_reranker": run_jinavl_reranker,
    "siglip": run_siglip,
    "vlm2vec_phi3v": run_vlm2vec_phi3v,
    "vlm2vec_qwen2vl": run_vlm2vec_qwen2vl,
}


def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "vision language models for multimodal pooling tasks."
    )
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="vlm2vec_phi3v",
        choices=model_example_map.keys(),
        help="The name of the embedding model.",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default="embedding",
        choices=["embedding", "scoring"],
        help="The task type.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="image",
        choices=get_args(QueryModality),
        help="Modality of the input.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set the seed when initializing `vllm.LLM`.",
    )
    return parser.parse_args()


def main(args: Namespace):
    if args.task == "embedding":
        run_encode(args.model_name, args.modality, args.seed)
    elif args.task == "scoring":
        run_score(args.model_name, args.modality, args.seed)
    else:
        raise ValueError(f"Unsupported task: {args.task}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
