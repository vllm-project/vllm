"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for multimodal embedding.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
from argparse import Namespace
from typing import Literal, NamedTuple, Optional, TypedDict, Union, get_args

from PIL.Image import Image

from vllm import LLM
from vllm.multimodal.utils import fetch_image
from vllm.utils import FlexibleArgumentParser


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


QueryModality = Literal["text", "image", "text+image"]
Query = Union[TextQuery, ImageQuery, TextImageQuery]


class ModelRequestData(NamedTuple):
    llm: LLM
    prompt: str
    image: Optional[Image]


def run_e5_v(query: Query):
    llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'  # noqa: E501

    if query["modality"] == "text":
        text = query["text"]
        prompt = llama3_template.format(
            f"{text}\nSummary above sentence in one word: ")
        image = None
    elif query["modality"] == "image":
        prompt = llama3_template.format(
            "<image>\nSummary above image in one word: ")
        image = query["image"]
    else:
        modality = query['modality']
        raise ValueError(f"Unsupported query modality: '{modality}'")

    llm = LLM(
        model="royokong/e5-v",
        task="embedding",
        max_model_len=4096,
    )

    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        image=image,
    )


def run_vlm2vec(query: Query):
    if query["modality"] == "text":
        text = query["text"]
        prompt = f"Find me an everyday image that matches the given caption: {text}"  # noqa: E501
        image = None
    elif query["modality"] == "image":
        prompt = "<|image_1|> Find a day-to-day image that looks similar to the provided image."  # noqa: E501
        image = query["image"]
    elif query["modality"] == "text+image":
        text = query["text"]
        prompt = f"<|image_1|> Represent the given image with the following question: {text}"  # noqa: E501
        image = query["image"]
    else:
        modality = query['modality']
        raise ValueError(f"Unsupported query modality: '{modality}'")

    llm = LLM(
        model="TIGER-Lab/VLM2Vec-Full",
        task="embedding",
        trust_remote_code=True,
        mm_processor_kwargs={"num_crops": 4},
    )

    return ModelRequestData(
        llm=llm,
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

    msg = f"Modality {modality} is not supported."
    raise ValueError(msg)


def run_encode(model: str, modality: QueryModality):
    query = get_query(modality)
    req_data = model_example_map[model](query)

    mm_data = {}
    if req_data.image is not None:
        mm_data["image"] = req_data.image

    outputs = req_data.llm.encode({
        "prompt": req_data.prompt,
        "multi_modal_data": mm_data,
    })

    for output in outputs:
        print(output.outputs.embedding)


def main(args: Namespace):
    run_encode(args.model_name, args.modality)


model_example_map = {
    "e5_v": run_e5_v,
    "vlm2vec": run_vlm2vec,
}

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models for multimodal embedding')
    parser.add_argument('--model-name',
                        '-m',
                        type=str,
                        default="vlm2vec",
                        choices=model_example_map.keys(),
                        help='The name of the embedding model.')
    parser.add_argument('--modality',
                        type=str,
                        default="image",
                        choices=get_args(QueryModality),
                        help='Modality of the input.')
    args = parser.parse_args()
    main(args)
