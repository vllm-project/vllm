"""
This example shows how to use vLLM for running offline inference with
multi-image input on vision language models, using the chat template defined
by the model.
"""
from argparse import Namespace
from typing import List

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image
from vllm.utils import FlexibleArgumentParser

QUESTION = "What is the content of each image?"
IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg",
]


def load_phi3v(question, image_urls: List[str]):
    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": len(image_urls)},
    )
    placeholders = "\n".join(f"<|image_{i}|>"
                             for i, _ in enumerate(image_urls, start=1))
    prompt = f"<|user|>\n{placeholders}\n{question}<|end|>\n<|assistant|>\n"
    stop_token_ids = None
    return llm, prompt, stop_token_ids


def load_internvl(question, image_urls: List[str]):
    model_name = "OpenGVLab/InternVL2-2B"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_num_seqs=5,
        max_model_len=4096,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = "\n".join(f"Image-{i}: <image>\n"
                             for i, _ in enumerate(image_urls, start=1))
    messages = [{'role': 'user', 'content': f"{placeholders}\n{question}"}]

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B#service
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    return llm, prompt, stop_token_ids


model_example_map = {
    "phi3_v": load_phi3v,
    "internvl_chat": load_internvl,
}


def run_generate(model, question: str, image_urls: List[str]):
    llm, prompt, stop_token_ids = model_example_map[model](question,
                                                           image_urls)

    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=128,
                                     stop_token_ids=stop_token_ids)

    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {
                "image": [fetch_image(url) for url in image_urls]
            },
        },
        sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def run_chat(model: str, question: str, image_urls: List[str]):
    llm, _, stop_token_ids = model_example_map[model](question, image_urls)

    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=128,
                                     stop_token_ids=stop_token_ids)

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
    }],
                       sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def main(args: Namespace):
    model = args.model_type
    method = args.method

    if method == "generate":
        run_generate(model, QUESTION, IMAGE_URLS)
    elif method == "chat":
        run_chat(model, QUESTION, IMAGE_URLS)
    else:
        raise ValueError(f"Invalid method: {method}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models that support multi-image input')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="phi3_v",
                        choices=model_example_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument("--method",
                        type=str,
                        default="generate",
                        choices=["generate", "chat"],
                        help="The method to run in `vllm.LLM`.")

    args = parser.parse_args()
    main(args)
