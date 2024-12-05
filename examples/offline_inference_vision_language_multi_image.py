"""
This example shows how to use vLLM for running offline inference with
multi-image input on vision language models for text generation,
using the chat template defined by the model.
"""
from argparse import Namespace
from typing import List, NamedTuple, Optional

from PIL.Image import Image
from transformers import AutoProcessor, AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image
from vllm.utils import FlexibleArgumentParser

QUESTION = "What is the content of each image?"
IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg",
]


class ModelRequestData(NamedTuple):
    llm: LLM
    prompt: str
    stop_token_ids: Optional[List[str]]
    image_data: List[Image]
    chat_template: Optional[str]


# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.


def load_qwenvl_chat(question: str, image_urls: List[str]) -> ModelRequestData:
    model_name = "Qwen/Qwen-VL-Chat"
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=1024,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": len(image_urls)},
    )
    placeholders = "".join(f"Picture {i}: <img></img>\n"
                           for i, _ in enumerate(image_urls, start=1))

    # This model does not have a chat_template attribute on its tokenizer,
    # so we need to explicitly pass it. We use ChatML since it's used in the
    # generation utils of the model:
    # https://huggingface.co/Qwen/Qwen-VL-Chat/blob/main/qwen_generation_utils.py#L265
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)

    # Copied from: https://huggingface.co/docs/transformers/main/en/chat_templating
    chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"  # noqa: E501

    messages = [{'role': 'user', 'content': f"{placeholders}\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True,
                                           chat_template=chat_template)

    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
        image_data=[fetch_image(url) for url in image_urls],
        chat_template=chat_template,
    )


def load_phi3v(question: str, image_urls: List[str]) -> ModelRequestData:
    # num_crops is an override kwarg to the multimodal image processor;
    # For some models, e.g., Phi-3.5-vision-instruct, it is recommended
    # to use 16 for single frame scenarios, and 4 for multi-frame.
    #
    # Generally speaking, a larger value for num_crops results in more
    # tokens per image instance, because it may scale the image more in
    # the image preprocessing. Some references in the model docs and the
    # formula for image tokens after the preprocessing
    # transform can be found below.
    #
    # https://huggingface.co/microsoft/Phi-3.5-vision-instruct#loading-the-model-locally
    # https://huggingface.co/microsoft/Phi-3.5-vision-instruct/blob/main/processing_phi3_v.py#L194
    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": len(image_urls)},
        mm_processor_kwargs={"num_crops": 4},
    )
    placeholders = "\n".join(f"<|image_{i}|>"
                             for i, _ in enumerate(image_urls, start=1))
    prompt = f"<|user|>\n{placeholders}\n{question}<|end|>\n<|assistant|>\n"
    stop_token_ids = None

    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
        image_data=[fetch_image(url) for url in image_urls],
        chat_template=None,
    )


def load_h2onvl(question: str, image_urls: List[str]) -> ModelRequestData:
    model_name = "h2oai/h2ovl-mississippi-2b"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={"image": len(image_urls)},
        mm_processor_kwargs={"max_dynamic_patch": 4},
    )

    placeholders = "\n".join(f"Image-{i}: <image>\n"
                             for i, _ in enumerate(image_urls, start=1))
    messages = [{'role': 'user', 'content': f"{placeholders}\n{question}"}]

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    # Stop tokens for H2OVL-Mississippi
    # https://huggingface.co/h2oai/h2ovl-mississippi-2b
    stop_token_ids = [tokenizer.eos_token_id]

    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
        image_data=[fetch_image(url) for url in image_urls],
        chat_template=None,
    )


def load_internvl(question: str, image_urls: List[str]) -> ModelRequestData:
    model_name = "OpenGVLab/InternVL2-2B"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": len(image_urls)},
        mm_processor_kwargs={"max_dynamic_patch": 4},
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

    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
        image_data=[fetch_image(url) for url in image_urls],
        chat_template=None,
    )


def load_nvlm_d(question: str, image_urls: List[str]):
    model_name = "nvidia/NVLM-D-72B"

    # Adjust this as necessary to fit in GPU
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        tensor_parallel_size=4,
        limit_mm_per_prompt={"image": len(image_urls)},
        mm_processor_kwargs={"max_dynamic_patch": 4},
    )

    placeholders = "\n".join(f"Image-{i}: <image>\n"
                             for i, _ in enumerate(image_urls, start=1))
    messages = [{'role': 'user', 'content': f"{placeholders}\n{question}"}]

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    stop_token_ids = None

    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
        image_data=[fetch_image(url) for url in image_urls],
        chat_template=None,
    )


def load_qwen2_vl(question, image_urls: List[str]) -> ModelRequestData:
    try:
        from qwen_vl_utils import process_vision_info
    except ModuleNotFoundError:
        print('WARNING: `qwen-vl-utils` not installed, input images will not '
              'be automatically resized. You can enable this functionality by '
              '`pip install qwen-vl-utils`.')
        process_vision_info = None

    model_name = "Qwen/Qwen2-VL-7B-Instruct"

    # Tested on L40
    llm = LLM(
        model=model_name,
        max_model_len=32768 if process_vision_info is None else 4096,
        max_num_seqs=5,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role":
        "user",
        "content": [
            *placeholders,
            {
                "type": "text",
                "text": question
            },
        ],
    }]

    processor = AutoProcessor.from_pretrained(model_name)

    prompt = processor.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    stop_token_ids = None

    if process_vision_info is None:
        image_data = [fetch_image(url) for url in image_urls]
    else:
        image_data, _ = process_vision_info(messages)

    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
        image_data=image_data,
        chat_template=None,
    )


def load_mllama(question, image_urls: List[str]) -> ModelRequestData:
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    # The configuration below has been confirmed to launch on a single L40 GPU.
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=16,
        enforce_eager=True,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    prompt = f"<|image|><|image|><|begin_of_text|>{question}"
    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        stop_token_ids=None,
        image_data=[fetch_image(url) for url in image_urls],
        chat_template=None,
    )


def load_idefics3(question, image_urls: List[str]) -> ModelRequestData:
    model_name = "HuggingFaceM4/Idefics3-8B-Llama3"

    # The configuration below has been confirmed to launch on a single L40 GPU.
    llm = LLM(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=16,
        enforce_eager=True,
        limit_mm_per_prompt={"image": len(image_urls)},
        # if you are running out of memory, you can reduce the "longest_edge".
        # see: https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3#model-optimizations
        mm_processor_kwargs={
            "size": {
                "longest_edge": 2 * 364
            },
        },
    )

    placeholders = "\n".join(f"Image-{i}: <image>\n"
                             for i, _ in enumerate(image_urls, start=1))
    prompt = f"<|begin_of_text|>User:{placeholders}\n{question}<end_of_utterance>\nAssistant:"  # noqa: E501
    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        stop_token_ids=None,
        image_data=[fetch_image(url) for url in image_urls],
        chat_template=None,
    )


def load_aria(question, image_urls: List[str]) -> ModelRequestData:
    model_name = "rhymes-ai/Aria"
    llm = LLM(model=model_name,
              tokenizer_mode="slow",
              trust_remote_code=True,
              dtype="bfloat16",
              limit_mm_per_prompt={"image": len(image_urls)})
    placeholders = "<fim_prefix><|img|><fim_suffix>\n" * len(image_urls)
    prompt = (f"<|im_start|>user\n{placeholders}{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    stop_token_ids = [93532, 93653, 944, 93421, 1019, 93653, 93519]
    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
        image_data=[fetch_image(url) for url in image_urls],
        chat_template=None)


model_example_map = {
    "phi3_v": load_phi3v,
    "h2ovl_chat": load_h2onvl,
    "internvl_chat": load_internvl,
    "NVLM_D": load_nvlm_d,
    "qwen2_vl": load_qwen2_vl,
    "qwen_vl_chat": load_qwenvl_chat,
    "mllama": load_mllama,
    "idefics3": load_idefics3,
    "aria": load_aria,
}


def run_generate(model, question: str, image_urls: List[str]):
    req_data = model_example_map[model](question, image_urls)

    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=128,
                                     stop_token_ids=req_data.stop_token_ids)

    outputs = req_data.llm.generate(
        {
            "prompt": req_data.prompt,
            "multi_modal_data": {
                "image": req_data.image_data
            },
        },
        sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def run_chat(model: str, question: str, image_urls: List[str]):
    req_data = model_example_map[model](question, image_urls)

    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=128,
                                     stop_token_ids=req_data.stop_token_ids)
    outputs = req_data.llm.chat(
        [{
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
        sampling_params=sampling_params,
        chat_template=req_data.chat_template,
    )

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
        'vision language models that support multi-image input for text '
        'generation')
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
