"""
This example shows how to use vLLM for running offline inference with
multi-image input on vision language models, using the chat template defined
by the model.
"""
from argparse import Namespace
from typing import List

from vllm import LLM
from vllm.multimodal.utils import fetch_image
from vllm.utils import FlexibleArgumentParser

QUESTION = "What is the content of each image?"
IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg",
]


def _load_phi3v(image_urls: List[str]):
    return LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": len(image_urls)},
    )


def run_phi3v_generate(question: str, image_urls: List[str]):
    llm = _load_phi3v(image_urls)

    placeholders = "\n".join(f"<|image_{i}|>"
                             for i, _ in enumerate(image_urls, start=1))
    prompt = f"<|user|>\n{placeholders}\n{question}<|end|>\n<|assistant|>\n"

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {
            "image": [fetch_image(url) for url in image_urls]
        },
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def run_phi3v_chat(question: str, image_urls: List[str]):
    llm = _load_phi3v(image_urls)

    outputs = llm.chat([{
        "role":
        "user",
        "content": [
            {
                "type": "text",
                "text": question,
            },
            *({
                "type": "image_url",
                "image_url": {
                    "url": image_url
                },
            } for image_url in image_urls),
        ],
    }])

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def main(args: Namespace):
    method = args.method

    if method == "generate":
        run_phi3v_generate(QUESTION, IMAGE_URLS)
    elif method == "chat":
        run_phi3v_chat(QUESTION, IMAGE_URLS)
    else:
        raise ValueError(f"Invalid method: {method}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models that support multi-image input')
    parser.add_argument("--method",
                        type=str,
                        default="generate",
                        choices=["generate", "chat"],
                        help="The method to run in `vllm.LLM`.")

    args = parser.parse_args()
    main(args)
