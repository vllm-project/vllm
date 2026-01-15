# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM for running offline inference with
multi-image input on vision language models for text generation,
using the chat template defined by the model.
"""

import os
from argparse import Namespace
from dataclasses import asdict
from typing import NamedTuple

from huggingface_hub import snapshot_download
from PIL.Image import Image
from transformers import AutoProcessor, AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.multimodal.utils import fetch_image
from vllm.utils.argparse_utils import FlexibleArgumentParser

QUESTION = "What is the content of each image?"
IMAGE_URLS = [
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/duck.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/lion.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/flycatcher.jpeg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/somefish.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/starfish.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/snail.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/thistle.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/husky.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/orangetabbycat.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/guineapig.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/rabbit.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/horsepony.jpg",
]


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str
    image_data: list[Image]
    stop_token_ids: list[int] | None = None
    chat_template: str | None = None
    lora_requests: list[LoRARequest] | None = None
    sampling_params: SamplingParams | None = None


# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.


def load_aria(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "rhymes-ai/Aria"
    engine_args = EngineArgs(
        model=model_name,
        tokenizer_mode="slow",
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": len(image_urls)},
    )
    placeholders = "<fim_prefix><|img|><fim_suffix>\n" * len(image_urls)
    prompt = (
        f"<|im_start|>user\n{placeholders}{question}<|im_end|>\n<|im_start|>assistant\n"
    )
    stop_token_ids = [93532, 93653, 944, 93421, 1019, 93653, 93519]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_aya_vision(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "CohereLabs/aya-vision-8b"

    engine_args = EngineArgs(
        model=model_name,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        }
    ]

    processor = AutoProcessor.from_pretrained(model_name)

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_bee(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "Open-Bee/Bee-8B-RL"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=16384,
        max_num_seqs=16,
        limit_mm_per_prompt={"image": len(image_urls)},
        trust_remote_code=True,
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        }
    ]

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_command_a_vision(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "CohereLabs/command-a-vision-07-2025"

    # NOTE: This model is 122B parameters and requires tensor parallelism
    # Recommended to use tp=4 on H100 GPUs
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=32768,
        tensor_parallel_size=4,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        }
    ]

    processor = AutoProcessor.from_pretrained(model_name)

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_deepseek_vl2(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "deepseek-ai/deepseek-vl2-tiny"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholder = "".join(
        f"image_{i}:<image>\n" for i, _ in enumerate(image_urls, start=1)
    )
    prompt = f"<|User|>: {placeholder}{question}\n\n<|Assistant|>:"

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_deepseek_ocr(question: str, image_urls: list[str]) -> ModelRequestData:
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

    model_name = "deepseek-ai/DeepSeek-OCR"

    engine_args = EngineArgs(
        model=model_name,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": len(image_urls)},
        logits_processors=[NGramPerReqLogitsProcessor],
    )

    placeholder = "<image>\n" * len(image_urls)
    prompt = placeholder + question

    # The following sampling params config is taken from
    # the official Deepseek-OCR inference example.
    # (IMPORTANT) Use the custom logits processor and avoid skipping
    # special tokens for this model for the optimal OCR performance.
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        # ngram logit processor args
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            # whitelist: <td>, </td>
            whitelist_token_ids={128821, 128822},
        ),
        skip_special_tokens=False,
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
        sampling_params=sampling_params,
    )


def load_gemma3(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "google/gemma-3-4b-it"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        }
    ]

    processor = AutoProcessor.from_pretrained(model_name)

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_h2ovl(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "h2oai/h2ovl-mississippi-800m"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={"image": len(image_urls)},
        mm_processor_kwargs={"max_dynamic_patch": 4},
    )

    placeholders = "\n".join(
        f"Image-{i}: <image>\n" for i, _ in enumerate(image_urls, start=1)
    )
    messages = [{"role": "user", "content": f"{placeholders}\n{question}"}]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Stop tokens for H2OVL-Mississippi
    # https://huggingface.co/h2oai/h2ovl-mississippi-800m
    stop_token_ids = [tokenizer.eos_token_id]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
        image_data=[fetch_image(url) for url in image_urls],
    )


# HunyuanOCR
def load_hunyuan_vl(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "tencent/HunyuanOCR"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholder = (
        "<｜hy_place▁holder▁no▁100｜><｜hy_place▁holder▁no▁102｜><｜hy_place▁holder▁no▁101｜>"  # noqa: E501
    ) * len(image_urls)
    prompt = f"<｜hy_begin▁of▁sentence｜>{placeholder}{question}<｜hy_User｜>"

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_hyperclovax_seed_vision(
    question: str, image_urls: list[str]
) -> ModelRequestData:
    model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=16384,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    message = {"role": "user", "content": list()}
    for _image_url in image_urls:
        message["content"].append(
            {
                "type": "image",
                "image": _image_url,
                "ocr": "",
                "lens_keywords": "",
                "lens_local_keywords": "",
            }
        )
    message["content"].append(
        {
            "type": "text",
            "text": question,
        }
    )

    prompt = tokenizer.apply_chat_template(
        [
            message,
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        stop_token_ids=None,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_idefics3(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "HuggingFaceM4/Idefics3-8B-Llama3"

    # The configuration below has been confirmed to launch on a single L40 GPU.
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=16,
        enforce_eager=True,
        limit_mm_per_prompt={"image": len(image_urls)},
        # if you are running out of memory, you can reduce the "longest_edge".
        # see: https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3#model-optimizations
        mm_processor_kwargs={
            "size": {"longest_edge": 2 * 364},
        },
    )

    placeholders = "\n".join(
        f"Image-{i}: <image>\n" for i, _ in enumerate(image_urls, start=1)
    )
    prompt = f"<|begin_of_text|>User:{placeholders}\n{question}<end_of_utterance>\nAssistant:"  # noqa: E501
    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_interns1(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "internlm/Intern-S1-mini"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = "\n".join(
        f"Image-{i}: <IMG_CONTEXT>\n" for i, _ in enumerate(image_urls, start=1)
    )
    messages = [{"role": "user", "content": f"{placeholders}\n{question}"}]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_internvl(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "OpenGVLab/InternVL2-2B"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": len(image_urls)},
        mm_processor_kwargs={"max_dynamic_patch": 4},
    )

    placeholders = "\n".join(
        f"Image-{i}: <image>\n" for i, _ in enumerate(image_urls, start=1)
    )
    messages = [{"role": "user", "content": f"{placeholders}\n{question}"}]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B/blob/main/conversation.py
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_keye_vl(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "Kwai-Keye/Keye-VL-8B-Preview"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        max_num_seqs=5,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        },
    ]

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_data = [fetch_image(url) for url in image_urls]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=image_data,
    )


def load_keye_vl1_5(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "Kwai-Keye/Keye-VL-1_5-8B"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=32768,
        max_num_seqs=5,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        },
    ]

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_data = [fetch_image(url) for url in image_urls]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=image_data,
    )


def load_kimi_vl(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "moonshotai/Kimi-VL-A3B-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=4,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        }
    ]

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_llama4(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=131072,
        tensor_parallel_size=8,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        }
    ]

    processor = AutoProcessor.from_pretrained(model_name)

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_llava(question: str, image_urls: list[str]) -> ModelRequestData:
    # NOTE: CAUTION! Original Llava models wasn't really trained on multi-image inputs,
    # it will generate poor response for multi-image inputs!
    model_name = "llava-hf/llava-1.5-7b-hf"
    engine_args = EngineArgs(
        model=model_name,
        max_num_seqs=16,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        }
    ]

    processor = AutoProcessor.from_pretrained(model_name)

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_llava_next(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=16,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        }
    ]

    processor = AutoProcessor.from_pretrained(model_name)

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_llava_onevision(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=16384,
        max_num_seqs=16,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        }
    ]

    processor = AutoProcessor.from_pretrained(model_name)

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_mistral3(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    # Adjust this as necessary to fit in GPU
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        tensor_parallel_size=2,
        limit_mm_per_prompt={"image": len(image_urls)},
        ignore_patterns=["consolidated.safetensors"],
    )

    placeholders = "[IMG]" * len(image_urls)
    prompt = f"<s>[INST]{question}\n{placeholders}[/INST]"

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_nvlm_d(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "nvidia/NVLM-D-72B"

    # Adjust this as necessary to fit in GPU
    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        tensor_parallel_size=4,
        limit_mm_per_prompt={"image": len(image_urls)},
        mm_processor_kwargs={"max_dynamic_patch": 4},
    )

    placeholders = "\n".join(
        f"Image-{i}: <image>\n" for i, _ in enumerate(image_urls, start=1)
    )
    messages = [{"role": "user", "content": f"{placeholders}\n{question}"}]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


# Ovis
def load_ovis(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "AIDC-AI/Ovis2-1B"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        trust_remote_code=True,
        dtype="half",
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = "\n".join(
        f"Image-{i}: <image>\n" for i, _ in enumerate(image_urls, start=1)
    )
    messages = [{"role": "user", "content": f"{placeholders}\n{question}"}]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


# ovis2_5
def load_ovis2_5(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "AIDC-AI/Ovis2.5-2B"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        trust_remote_code=True,
        dtype="half",
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = "\n".join(
        f"Image-{i}: <image>\n" for i, _ in enumerate(image_urls, start=1)
    )
    prompt = (
        f"<|im_start|>user\n\n{placeholders}\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_paddleocr_vl(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "PaddlePaddle/PaddleOCR-VL"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>" * len(image_urls)
    prompt = f"<|begin_of_sentence|>User: {question}{placeholders}\nAssistant: "

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_pixtral_hf(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "mistral-community/pixtral-12b"

    # Adjust this as necessary to fit in GPU
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        tensor_parallel_size=2,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = "[IMG]" * len(image_urls)
    prompt = f"<s>[INST]{question}\n{placeholders}[/INST]"

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_phi3v(question: str, image_urls: list[str]) -> ModelRequestData:
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
    engine_args = EngineArgs(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": len(image_urls)},
        mm_processor_kwargs={"num_crops": 4},
    )
    placeholders = "\n".join(
        f"<|image_{i}|>" for i, _ in enumerate(image_urls, start=1)
    )
    prompt = f"<|user|>\n{placeholders}\n{question}<|end|>\n<|assistant|>\n"

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_phi4mm(question: str, image_urls: list[str]) -> ModelRequestData:
    """
    Phi-4-multimodal-instruct supports both image and audio inputs. Here, we
    show how to process multi images inputs.
    """

    model_path = snapshot_download("microsoft/Phi-4-multimodal-instruct")
    # Since the vision-lora and speech-lora co-exist with the base model,
    # we have to manually specify the path of the lora weights.
    vision_lora_path = os.path.join(model_path, "vision-lora")
    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": len(image_urls)},
        enable_lora=True,
        max_lora_rank=320,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={"dynamic_hd": 4},
    )

    placeholders = "".join(f"<|image_{i}|>" for i, _ in enumerate(image_urls, start=1))
    prompt = f"<|user|>{placeholders}{question}<|end|><|assistant|>"

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
        lora_requests=[LoRARequest("vision", 1, vision_lora_path)],
    )


def load_qwen_vl_chat(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "Qwen/Qwen-VL-Chat"
    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=1024,
        max_num_seqs=2,
        hf_overrides={"architectures": ["QwenVLForConditionalGeneration"]},
        limit_mm_per_prompt={"image": len(image_urls)},
    )
    placeholders = "".join(
        f"Picture {i}: <img></img>\n" for i, _ in enumerate(image_urls, start=1)
    )

    # This model does not have a chat_template attribute on its tokenizer,
    # so we need to explicitly pass it. We use ChatML since it's used in the
    # generation utils of the model:
    # https://huggingface.co/Qwen/Qwen-VL-Chat/blob/main/qwen_generation_utils.py#L265
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Copied from: https://huggingface.co/docs/transformers/main/en/chat_templating
    chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"  # noqa: E501

    messages = [{"role": "user", "content": f"{placeholders}\n{question}"}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=chat_template,
    )

    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
        image_data=[fetch_image(url) for url in image_urls],
        chat_template=chat_template,
    )


def load_qwen2_vl(question: str, image_urls: list[str]) -> ModelRequestData:
    try:
        from qwen_vl_utils import smart_resize
    except ModuleNotFoundError:
        print(
            "WARNING: `qwen-vl-utils` not installed, input images will not "
            "be automatically resized. You can enable this functionality by "
            "`pip install qwen-vl-utils`."
        )
        smart_resize = None

    model_name = "Qwen/Qwen2-VL-7B-Instruct"

    # Tested on L40
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=32768 if smart_resize is None else 4096,
        max_num_seqs=5,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        },
    ]

    processor = AutoProcessor.from_pretrained(model_name)

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if smart_resize is None:
        image_data = [fetch_image(url) for url in image_urls]
    else:

        def post_process_image(image: Image) -> Image:
            width, height = image.size
            resized_height, resized_width = smart_resize(
                height, width, max_pixels=1024 * 28 * 28
            )
            return image.resize((resized_width, resized_height))

        image_data = [post_process_image(fetch_image(url)) for url in image_urls]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=image_data,
    )


def load_qwen2_5_vl(question: str, image_urls: list[str]) -> ModelRequestData:
    try:
        from qwen_vl_utils import smart_resize
    except ModuleNotFoundError:
        print(
            "WARNING: `qwen-vl-utils` not installed, input images will not "
            "be automatically resized. You can enable this functionality by "
            "`pip install qwen-vl-utils`."
        )
        smart_resize = None

    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=32768 if smart_resize is None else 4096,
        max_num_seqs=5,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        },
    ]

    processor = AutoProcessor.from_pretrained(model_name)

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if smart_resize is None:
        image_data = [fetch_image(url) for url in image_urls]
    else:

        def post_process_image(image: Image) -> Image:
            width, height = image.size
            resized_height, resized_width = smart_resize(
                height, width, max_pixels=1024 * 28 * 28
            )
            return image.resize((resized_width, resized_height))

        image_data = [post_process_image(fetch_image(url)) for url in image_urls]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=image_data,
    )


def load_r_vl(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "YannQi/R-4B"
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=16384,
        max_num_seqs=16,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        }
    ]

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_smolvlm(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

    # The configuration below has been confirmed to launch on a single L40 GPU.
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=16,
        enforce_eager=True,
        limit_mm_per_prompt={"image": len(image_urls)},
        mm_processor_kwargs={
            "max_image_size": {"longest_edge": 384},
        },
    )

    placeholders = "\n".join(
        f"Image-{i}: <image>\n" for i, _ in enumerate(image_urls, start=1)
    )
    prompt = (
        f"<|im_start|>User:{placeholders}\n{question}<end_of_utterance>\nAssistant:"  # noqa: E501
    )
    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_step3(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "stepfun-ai/step3-fp8"

    # NOTE: Below are verified configurations for step3-fp8
    # on 8xH100 GPUs.
    engine_args = EngineArgs(
        model=model_name,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=8,
        limit_mm_per_prompt={"image": len(image_urls)},
        reasoning_parser="step3",
    )

    prompt = (
        "<｜begin▁of▁sentence｜> You are a helpful assistant. <|BOT|>user\n "
        f"{'<im_patch>' * len(image_urls)}{question} <|EOT|><|BOT|"
        ">assistant\n<think>\n"
    )
    image_data = [fetch_image(url) for url in image_urls]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=image_data,
    )


def load_tarsier(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "omni-research/Tarsier-7b"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    prompt = f"USER: {'<image>' * len(image_urls)}\n{question}\n ASSISTANT:"
    image_data = [fetch_image(url) for url in image_urls]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=image_data,
    )


def load_tarsier2(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "omni-research/Tarsier2-Recap-7b"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=32768,
        limit_mm_per_prompt={"image": len(image_urls)},
        hf_overrides={
            "architectures": ["Tarsier2ForConditionalGeneration"],
            "model_type": "tarsier2",
        },
    )

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{'<|image_pad|>' * len(image_urls)}"
        f"<|vision_end|>{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    image_data = [fetch_image(url) for url in image_urls]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=image_data,
    )


# GLM-4.5V
def load_glm4_5v(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "zai-org/GLM-4.5V"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=32768,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": len(image_urls)},
        enforce_eager=True,
        tensor_parallel_size=4,
    )
    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        }
    ]
    processor = AutoProcessor.from_pretrained(model_name)
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_data = [fetch_image(url) for url in image_urls]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=image_data,
    )


# GLM-4.5V-FP8
def load_glm4_5v_fp8(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "zai-org/GLM-4.5V-FP8"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=32768,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": len(image_urls)},
        enforce_eager=True,
        tensor_parallel_size=4,
    )
    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        }
    ]
    processor = AutoProcessor.from_pretrained(model_name)
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_data = [fetch_image(url) for url in image_urls]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=image_data,
    )


def load_molmo2(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "allenai/Molmo2-8B"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": len(image_urls)},
        max_num_batched_tokens=36864,
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        },
    ]

    processor = AutoProcessor.from_pretrained(model_name)

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_data = [fetch_image(url) for url in image_urls]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=image_data,
    )


model_example_map = {
    "aria": load_aria,
    "aya_vision": load_aya_vision,
    "bee": load_bee,
    "command_a_vision": load_command_a_vision,
    "deepseek_vl_v2": load_deepseek_vl2,
    "deepseek_ocr": load_deepseek_ocr,
    "gemma3": load_gemma3,
    "h2ovl_chat": load_h2ovl,
    "hunyuan_vl": load_hunyuan_vl,
    "hyperclovax_seed_vision": load_hyperclovax_seed_vision,
    "idefics3": load_idefics3,
    "interns1": load_interns1,
    "internvl_chat": load_internvl,
    "keye_vl": load_keye_vl,
    "keye_vl1_5": load_keye_vl1_5,
    "kimi_vl": load_kimi_vl,
    "llama4": load_llama4,
    "llava": load_llava,
    "llava-next": load_llava_next,
    "llava-onevision": load_llava_onevision,
    "mistral3": load_mistral3,
    "molmo2": load_molmo2,
    "NVLM_D": load_nvlm_d,
    "ovis": load_ovis,
    "ovis2_5": load_ovis2_5,
    "paddleocr_vl": load_paddleocr_vl,
    "phi3_v": load_phi3v,
    "phi4_mm": load_phi4mm,
    "pixtral_hf": load_pixtral_hf,
    "qwen_vl_chat": load_qwen_vl_chat,
    "qwen2_vl": load_qwen2_vl,
    "qwen2_5_vl": load_qwen2_5_vl,
    "rvl": load_r_vl,
    "smolvlm": load_smolvlm,
    "step3": load_step3,
    "tarsier": load_tarsier,
    "tarsier2": load_tarsier2,
    "glm4_5v": load_glm4_5v,
    "glm4_5v_fp8": load_glm4_5v_fp8,
}


def run_generate(
    model,
    question: str,
    image_urls: list[str],
    seed: int,
    tensor_parallel_size: int | None,
):
    req_data = model_example_map[model](question, image_urls)

    engine_args = asdict(req_data.engine_args) | {"seed": seed}
    if tensor_parallel_size is not None:
        engine_args["tensor_parallel_size"] = tensor_parallel_size
    llm = LLM(**engine_args)

    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=256, stop_token_ids=req_data.stop_token_ids
    )

    outputs = llm.generate(
        {
            "prompt": req_data.prompt,
            "multi_modal_data": {"image": req_data.image_data},
        },
        sampling_params=sampling_params,
        lora_request=req_data.lora_requests,
    )

    print("-" * 50)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        print("-" * 50)


def run_chat(
    model: str,
    question: str,
    image_urls: list[str],
    seed: int,
    tensor_parallel_size: int | None,
):
    req_data = model_example_map[model](question, image_urls)

    # Disable other modalities to save memory
    default_limits = {"image": 0, "video": 0, "audio": 0}
    req_data.engine_args.limit_mm_per_prompt = default_limits | dict(
        req_data.engine_args.limit_mm_per_prompt or {}
    )

    engine_args = asdict(req_data.engine_args) | {"seed": seed}
    if tensor_parallel_size is not None:
        engine_args["tensor_parallel_size"] = tensor_parallel_size
    llm = LLM(**engine_args)

    sampling_params = (
        SamplingParams(
            temperature=0.0, max_tokens=256, stop_token_ids=req_data.stop_token_ids
        )
        if req_data.sampling_params is None
        else req_data.sampling_params
    )
    outputs = llm.chat(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question,
                    },
                    *(
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        }
                        for image_url in image_urls
                    ),
                ],
            }
        ],
        sampling_params=sampling_params,
        chat_template=req_data.chat_template,
        lora_request=req_data.lora_requests,
    )

    print("-" * 50)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        print("-" * 50)


def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "vision language models that support multi-image input for text "
        "generation"
    )
    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        default="phi3_v",
        choices=model_example_map.keys(),
        help='Huggingface "model_type".',
    )
    parser.add_argument(
        "--method",
        type=str,
        default="generate",
        choices=["generate", "chat"],
        help="The method to run in `vllm.LLM`.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set the seed when initializing `vllm.LLM`.",
    )
    parser.add_argument(
        "--num-images",
        "-n",
        type=int,
        choices=list(range(1, len(IMAGE_URLS) + 1)),  # the max number of images
        default=2,
        help="Number of images to use for the demo.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "-tp",
        type=int,
        default=None,
        help="Tensor parallel size to override the model's default setting. ",
    )
    return parser.parse_args()


def main(args: Namespace):
    model = args.model_type
    method = args.method
    seed = args.seed
    tensor_parallel_size = args.tensor_parallel_size

    if tensor_parallel_size is not None and tensor_parallel_size < 1:
        raise ValueError(
            f"tensor_parallel_size must be a positive integer, "
            f"got {tensor_parallel_size}"
        )

    image_urls = IMAGE_URLS[: args.num_images]

    if method == "generate":
        run_generate(model, QUESTION, image_urls, seed, tensor_parallel_size)
    elif method == "chat":
        run_chat(model, QUESTION, image_urls, seed, tensor_parallel_size)
    else:
        raise ValueError(f"Invalid method: {method}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
