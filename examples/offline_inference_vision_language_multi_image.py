"""
This example shows how to use vLLM for running offline inference with
multi-image input on vision language models, using the chat template defined
by the model.
"""
from typing import List

from vllm import LLM

IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg",
]


def run_phi3v(image_urls: List[str]):
    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,  # Required to load Phi-3.5-vision
        max_model_len=4096,  # Otherwise, it may not fit in smaller GPUs
        limit_mm_per_prompt={"image": 2},  # The maximum number to accept
    )

    # It's quite tedious to create the prompt with multiple image placeholders
    # Let's instead use the chat template that is built into Phi-3.5-vision
    outputs = llm.chat([{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What are the animals in these images?"
            },
            *(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
                for image_url in image_urls
            ),
        ],
    }])

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    run_phi3v(IMAGE_URLS)
