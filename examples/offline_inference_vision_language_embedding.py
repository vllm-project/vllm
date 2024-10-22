"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for multimodal embedding.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
from argparse import Namespace
from typing import List, NamedTuple, Optional, Union

from PIL.Image import Image

from vllm import LLM
from vllm.multimodal.utils import fetch_image
from vllm.utils import FlexibleArgumentParser


class ModelRequestData(NamedTuple):
    llm: LLM
    prompt: str
    stop_token_ids: Optional[List[str]]
    image: Optional[Image]


def run_e5_v(text_or_image: Union[str, Image]):
    llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'  # noqa: E501

    if isinstance(text_or_image, str):
        prompt = llama3_template.format(
            f"{text_or_image}\nSummary above sentence in one word: ")
        image = None
    else:
        prompt = llama3_template.format(
            "<image>\nSummary above image in one word: ")
        image = text_or_image

    llm = LLM(
        model="royokong/e5-v-2",
        task="embedding",
    )

    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        stop_token_ids=None,
        image=image,
    )


def run_vlm2vec(text_or_image: Union[str, Image]):
    if isinstance(text_or_image, str):
        prompt = f"Find me an everyday image that matches the given caption: {text_or_image}"  # noqa: E501
        image = None
    else:
        prompt = "<|image_1|> Represent the given image with the following question: What is in the image"  # noqa: E501
        image = text_or_image

    llm = LLM(
        model="TIGER-Lab/VLM2Vec-Full",
        task="embedding",
        trust_remote_code=True,
        mm_processor_kwargs={"num_crops": 4},
    )

    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        stop_token_ids=None,
        image=image,
    )


def get_text_or_image(modality: str):
    if modality == "text":
        return "A dog sitting in the grass"

    if modality == "image":
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/American_Eskimo_Dog.jpg/360px-American_Eskimo_Dog.jpg"
        return fetch_image(image_url)

    msg = f"Modality {modality} is not supported."
    raise ValueError(msg)


def run_encode(model: str, modality: str):
    text_or_image = get_text_or_image(modality)
    req_data = model_example_map[model](text_or_image)

    # Generate embedding. The output is a list of EmbeddingRequestOutputs.
    outputs = req_data.llm.encode(
        {
            "prompt": req_data.prompt,
            "multi_modal_data": {
                "image": req_data.image
            },
        }, )

    for output in outputs:
        print(output.outputs.embedding)


def main(args: Namespace):
    run_encode(args.model, args.modality)


model_example_map = {
    "e5_v": run_e5_v,
    "vlm2vec": run_vlm2vec,
}

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models for multimodal embedding')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="vlm2vec",
                        choices=model_example_map.keys(),
                        help='The name of the embedding model.')
    parser.add_argument('--modality',
                        type=str,
                        default="image",
                        choices=['text', 'image'],
                        help='Modality of the input.')
    args = parser.parse_args()
    main(args)
