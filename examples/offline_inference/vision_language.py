# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""

import os
import random
from contextlib import contextmanager
from dataclasses import asdict
from typing import NamedTuple

from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.lora.request import LoRARequest
from vllm.multimodal.image import convert_image_mode
from vllm.utils.argparse_utils import FlexibleArgumentParser


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: list[str]
    stop_token_ids: list[int] | None = None
    lora_requests: list[LoRARequest] | None = None
    sampling_params: list[SamplingParams] | None = None


# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.


# Aria
def run_aria(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "rhymes-ai/Aria"

    # NOTE: Need L40 (or equivalent) to avoid OOM
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        dtype="bfloat16",
        limit_mm_per_prompt={modality: 1},
    )

    prompts = [
        (
            f"<|im_start|>user\n<fim_prefix><|img|><fim_suffix>{question}"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        for question in questions
    ]

    stop_token_ids = [93532, 93653, 944, 93421, 1019, 93653, 93519]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


# Aya Vision
def run_aya_vision(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "CohereLabs/aya-vision-8b"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=2048,
        max_num_seqs=2,
        mm_processor_kwargs={"crop_to_patches": True},
        limit_mm_per_prompt={modality: 1},
    )
    prompts = [
        f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|><image>{question}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        for question in questions
    ]
    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Bee-8B
def run_bee(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "Open-Bee/Bee-8B-RL"

    prompts = [
        (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<image>\n{question}<|im_end|>"
            f"<|im_start|>assistant\n<think>\n"
        )
        for question in questions
    ]

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=16384,
        limit_mm_per_prompt={modality: 1},
        trust_remote_code=True,
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


def run_bagel(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "ByteDance-Seed/BAGEL-7B-MoT"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={modality: 1},
    )

    prompts = [
        (
            f"<|im_start|>user\n<|image_pad|>\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# BLIP-2
def run_blip2(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    # BLIP-2 prompt format is inaccurate on HuggingFace model repository.
    # See https://huggingface.co/Salesforce/blip2-opt-2.7b/discussions/15#64ff02f3f8cf9e4f5b038262 #noqa
    prompts = [f"Question: {question} Answer:" for question in questions]
    engine_args = EngineArgs(
        model="Salesforce/blip2-opt-2.7b",
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Chameleon
def run_chameleon(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    prompts = [f"{question}<image>" for question in questions]
    engine_args = EngineArgs(
        model="facebook/chameleon-7b",
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


def run_command_a_vision(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "CohereLabs/command-a-vision-07-2025"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=32768,
        tensor_parallel_size=4,
        limit_mm_per_prompt={modality: 1},
    )

    prompts = [
        f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|><|IMG_PATCH|>{question}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Deepseek-VL2
def run_deepseek_vl2(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "deepseek-ai/deepseek-vl2-tiny"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
        limit_mm_per_prompt={modality: 1},
    )

    prompts = [
        f"<|User|>: <image>\n{question}\n\n<|Assistant|>:" for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


def run_deepseek_ocr(questions: list[str], modality: str) -> ModelRequestData:
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

    assert modality == "image"

    model_name = "deepseek-ai/DeepSeek-OCR"

    engine_args = EngineArgs(
        model=model_name,
        limit_mm_per_prompt={modality: 1},
        logits_processors=[NGramPerReqLogitsProcessor],
    )

    # deepseek-ocr use plain prompt template
    prompts = [f"<image>\n{question}" for question in questions]

    # The following sampling params config is taken from
    # the official Deepseek-OCR inference example.
    # (IMPORTANT) Use the custom logits processor and avoid skipping
    # special tokens for this model for the optimal OCR performance.
    sampling_params = [
        SamplingParams(
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
        for _ in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        sampling_params=sampling_params,
    )


# Dots-OCR
def run_dots_ocr(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    prompts = [f"<|img|><|imgpad|><|endofimg|>{question}" for question in questions]
    engine_args = EngineArgs(
        model="rednote-hilab/dots.ocr",
        limit_mm_per_prompt={modality: 1},
        trust_remote_code=True,
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Eagle2.5-VL
def run_eagle2_5(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "nvidia/Eagle2.5-8B"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        trust_remote_code=True,
        limit_mm_per_prompt={modality: 1},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [
        [{"role": "user", "content": f"<image>\n{question}"}] for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Stop tokens for Eagle2.5 (Qwen2 based)
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    stop_token_ids = [token_id for token_id in stop_token_ids if token_id is not None]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


# Ernie4.5-VL
def run_ernie45_vl(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "baidu/ERNIE-4.5-VL-28B-A3B-PT"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        limit_mm_per_prompt={modality: 1},
        trust_remote_code=True,
    )

    if modality == "image":
        placeholder = "Picture 1:<|IMAGE_START|><|image@placeholder|><|IMAGE_END|>"
    elif modality == "video":
        placeholder = "Video 1:<|VIDEO_START|><|video@placeholder|><|VIDEO_END|>"

    prompts = [
        (
            f"<|begin_of_sentence|>User: {question}{placeholder}\n"
            "Assistant: <think></think>"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Fuyu
def run_fuyu(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    prompts = [f"{question}\n" for question in questions]
    engine_args = EngineArgs(
        model="adept/fuyu-8b",
        max_model_len=2048,
        max_num_seqs=2,
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Gemma 3
def run_gemma3(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "google/gemma-3-4b-it"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=2048,
        max_num_seqs=2,
        mm_processor_kwargs={"do_pan_and_scan": True},
        limit_mm_per_prompt={modality: 1},
    )

    prompts = [
        (
            "<bos><start_of_turn>user\n"
            f"<start_of_image>{question}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        for question in questions
    ]
    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Gemma3N
def run_gemma3n(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "google/gemma-3n-E2B-it"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=2048,
        max_num_seqs=2,
        limit_mm_per_prompt={modality: 1},
        enforce_eager=True,
    )

    prompts = [
        (
            "<start_of_turn>user\n"
            f"<image_soft_token>{question}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        for question in questions
    ]
    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# GLM-4v
def run_glm4v(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "zai-org/glm-4v-9b"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=2048,
        max_num_seqs=2,
        trust_remote_code=True,
        enforce_eager=True,
        hf_overrides={"architectures": ["GLM4VForCausalLM"]},
        limit_mm_per_prompt={modality: 1},
    )

    prompts = [
        (
            "<|user|>\n<|begin_of_image|><|endoftext|><|end_of_image|>"
            f"{question}<|assistant|>"
        )
        for question in questions
    ]

    stop_token_ids = [151329, 151336, 151338]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


# GLM-4.1V
def run_glm4_1v(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "zai-org/GLM-4.1V-9B-Thinking"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        mm_processor_kwargs={
            "size": {"shortest_edge": 12544, "longest_edge": 47040000},
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
        enforce_eager=True,
    )

    if modality == "image":
        placeholder = "<|begin_of_image|><|image|><|end_of_image|>"
    elif modality == "video":
        placeholder = "<|begin_of_video|><|video|><|end_of_video|>"

    prompts = [
        (
            "[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n"
            f"{placeholder}"
            f"{question}<|assistant|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# GLM-4.5V
def run_glm4_5v(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "zai-org/GLM-4.5V"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        mm_processor_kwargs={
            "size": {"shortest_edge": 12544, "longest_edge": 47040000},
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
        enforce_eager=True,
        tensor_parallel_size=4,
    )

    if modality == "image":
        placeholder = "<|begin_of_image|><|image|><|end_of_image|>"
    elif modality == "video":
        placeholder = "<|begin_of_video|><|video|><|end_of_video|>"

    prompts = [
        (
            "[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n"
            f"{placeholder}"
            f"{question}<|assistant|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# GLM-4.5V-FP8
def run_glm4_5v_fp8(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "zai-org/GLM-4.5V-FP8"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        mm_processor_kwargs={
            "size": {"shortest_edge": 12544, "longest_edge": 47040000},
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
        enforce_eager=True,
        tensor_parallel_size=4,
    )

    if modality == "image":
        placeholder = "<|begin_of_image|><|image|><|end_of_image|>"
    elif modality == "video":
        placeholder = "<|begin_of_video|><|video|><|end_of_video|>"

    prompts = [
        (
            "[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n"
            f"{placeholder}"
            f"{question}<|assistant|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# GLM-OCR
def run_glm_ocr(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "zai-org/GLM-OCR"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        mm_processor_kwargs={
            "size": {"shortest_edge": 12544, "longest_edge": 47040000},
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
        enforce_eager=True,
    )

    if modality == "image":
        placeholder = "<|begin_of_image|><|image|><|end_of_image|>"
    elif modality == "video":
        placeholder = "<|begin_of_video|><|video|><|end_of_video|>"

    prompts = [
        (
            "[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n"
            f"{placeholder}"
            f"{question}<|assistant|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# H2OVL-Mississippi
def run_h2ovl(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "h2oai/h2ovl-mississippi-800m"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={modality: 1},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [
        [{"role": "user", "content": f"<image>\n{question}"}] for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Stop tokens for H2OVL-Mississippi
    # https://huggingface.co/h2oai/h2ovl-mississippi-800m
    stop_token_ids = [tokenizer.eos_token_id]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


# HunyuanOCR
def run_hunyuan_vl(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "tencent/HunyuanOCR"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        limit_mm_per_prompt={modality: 1},
    )

    placeholder = "<｜hy_place▁holder▁no▁100｜><｜hy_place▁holder▁no▁102｜><｜hy_place▁holder▁no▁101｜>"  # noqa: E501
    prompts = [
        f"<｜hy_begin▁of▁sentence｜>{placeholder}{question}<｜hy_User｜>"
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=None,
    )


# naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B
def run_hyperclovax_seed_vision(
    questions: list[str], modality: str
) -> ModelRequestData:
    model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192 if modality == "image" else 16384,
        limit_mm_per_prompt={modality: 1},
    )

    messages = list()
    for question in questions:
        if modality == "image":
            """
            ocr: List the words in the image in raster order.
                Even if the word order feels unnatural for reading,
                the model will handle it as long as it follows raster order.
                e.g. "Naver, CLOVA, bigshane"
            lens_keywords: List the entity names in the image.
                e.g. "iPhone"
            lens_local_keywords: List the entity names with quads in the image.
                e.g. "[0.07, 0.21, 0.92, 0.90] iPhone"
            """
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "ocr": "",
                                "lens_keywords": "",
                                "lens_local_keywords": "",
                            },
                            {
                                "type": "text",
                                "text": question,
                            },
                        ],
                    }
                ]
            )
        elif modality == "video":
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                            },
                            {
                                "type": "text",
                                "text": question,
                            },
                        ],
                    }
                ]
            )
        else:
            raise ValueError(f"Unsupported modality: {modality}")

    prompts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=None,
    )


# Idefics3-8B-Llama3
def run_idefics3(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "HuggingFaceM4/Idefics3-8B-Llama3"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        enforce_eager=True,
        # if you are running out of memory, you can reduce the "longest_edge".
        # see: https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3#model-optimizations
        mm_processor_kwargs={
            "size": {"longest_edge": 3 * 364},
        },
        limit_mm_per_prompt={modality: 1},
    )
    prompts = [
        (f"<|begin_of_text|>User:<image>{question}<end_of_utterance>\nAssistant:")
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Intern-S1
def run_interns1(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "internlm/Intern-S1-mini"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={modality: 1},
        enforce_eager=True,
    )

    if modality == "image":
        placeholder = "<IMG_CONTEXT>"
    elif modality == "video":
        placeholder = "<video>"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [
        [{"role": "user", "content": f"{placeholder}\n{question}"}]
        for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# InternVL
def run_internvl(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "OpenGVLab/InternVL3-2B"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<image>"
    elif modality == "video":
        placeholder = "<video>"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [
        [{"role": "user", "content": f"{placeholder}\n{question}"}]
        for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B/blob/main/conversation.py
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    stop_token_ids = [token_id for token_id in stop_token_ids if token_id is not None]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


# Kanana-V
def run_kanana_v(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "kakaocorp/kanana-1.5-v-3b-instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        trust_remote_code=True,
        limit_mm_per_prompt={modality: 1},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [
        [{"role": "user", "content": f"<image>\n{question}"}] for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Keye-VL
def run_keye_vl(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "Kwai-Keye/Keye-VL-8B-Preview"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        trust_remote_code=True,
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        (
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Keye-VL-1.5
def run_keye_vl1_5(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "Kwai-Keye/Keye-VL-1.5-8B"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        trust_remote_code=True,
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        (
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Kimi-VL
def run_kimi_vl(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    prompts = [
        "<|im_user|>user<|im_middle|><|media_start|>image<|media_content|>"
        f"<|media_pad|><|media_end|>{question}<|im_end|>"
        "<|im_assistant|>assistant<|im_middle|>"
        for question in questions
    ]

    engine_args = EngineArgs(
        model="moonshotai/Kimi-VL-A3B-Instruct",
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# LightOnOCR
def run_lightonocr(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    prompts = [
        "<|im_start|>system<|im_end|>\n<|im_start|>user\n<|image_pad|><|im_end|>\n<|im_start|>assistant\n"
        for _ in questions
    ]

    engine_args = EngineArgs(
        model="lightonai/LightOnOCR-1B",
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


def run_lfm2_vl(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "LiquidAI/LFM2-VL-450M"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        limit_mm_per_prompt={modality: 1},
    )

    processor = AutoProcessor.from_pretrained(model_name)
    messages = [
        [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": question}],
            }
        ]
        for question in questions
    ]
    prompts = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


def run_llama4(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=4,
        tensor_parallel_size=8,
        gpu_memory_utilization=0.4,
        limit_mm_per_prompt={modality: 1},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [
        [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": f"{question}"}],
            }
        ]
        for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    stop_token_ids = None
    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


# LLaVA-1.5
def run_llava(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    prompts = [f"USER: <image>\n{question}\nASSISTANT:" for question in questions]

    engine_args = EngineArgs(
        model="llava-hf/llava-1.5-7b-hf",
        max_model_len=4096,
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# LLaVA-1.6/LLaVA-NeXT
def run_llava_next(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    prompts = [f"[INST] <image>\n{question} [/INST]" for question in questions]
    engine_args = EngineArgs(
        model="llava-hf/llava-v1.6-mistral-7b-hf",
        max_model_len=8192,
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# LlaVA-NeXT-Video
# Currently only support for video input
def run_llava_next_video(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "video"

    prompts = [f"USER: <video>\n{question} ASSISTANT:" for question in questions]
    engine_args = EngineArgs(
        model="llava-hf/LLaVA-NeXT-Video-7B-hf",
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# LLaVA-OneVision
def run_llava_onevision(questions: list[str], modality: str) -> ModelRequestData:
    if modality == "video":
        prompts = [
            f"<|im_start|>user <video>\n{question}<|im_end|><|im_start|>assistant\n"
            for question in questions
        ]

    elif modality == "image":
        prompts = [
            f"<|im_start|>user <image>\n{question}<|im_end|><|im_start|>assistant\n"
            for question in questions
        ]

    engine_args = EngineArgs(
        model="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        max_model_len=16384,
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Mantis
def run_mantis(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    llama3_template = "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"  # noqa: E501
    prompts = [llama3_template.format(f"{question}\n<image>") for question in questions]

    engine_args = EngineArgs(
        model="TIGER-Lab/Mantis-8B-siglip-llama3",
        max_model_len=4096,
        hf_overrides={"architectures": ["MantisForConditionalGeneration"]},
        limit_mm_per_prompt={modality: 1},
    )
    stop_token_ids = [128009]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


# MiniCPM-V
def run_minicpmv_base(questions: list[str], modality: str, model_name):
    assert modality in ["image", "video"]
    # If you want to use `MiniCPM-o-2_6` with audio inputs, check `audio_language.py` # noqa

    # 2.0
    # The official repo doesn't work yet, so we need to use a fork for now
    # For more details, please see: See: https://github.com/vllm-project/vllm/pull/4087#issuecomment-2250397630 # noqa
    # model_name = "HwwwH/MiniCPM-V-2"

    # 2.5
    # model_name = "openbmb/MiniCPM-Llama3-V-2_5"

    # 2.6
    # model_name = "openbmb/MiniCPM-V-2_6"
    # o2.6

    # modality supports
    # 2.0: image
    # 2.5: image
    # 2.6: image, video
    # o2.6: image, video, audio
    # model_name = "openbmb/MiniCPM-o-2_6"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        trust_remote_code=True,
        limit_mm_per_prompt={modality: 1},
    )
    # NOTE The stop_token_ids are different for various versions of MiniCPM-V
    # 2.0
    # stop_token_ids = [tokenizer.eos_id]

    # 2.5
    # stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]

    # 2.6 / o2.6
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    modality_placeholder = {
        "image": "(<image>./</image>)",
        "video": "(<video>./</video>)",
    }

    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": f"{modality_placeholder[modality]}\n{question}",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


def run_minicpmo(questions: list[str], modality: str) -> ModelRequestData:
    return run_minicpmv_base(questions, modality, "openbmb/MiniCPM-o-2_6")


def run_minicpmv(questions: list[str], modality: str) -> ModelRequestData:
    return run_minicpmv_base(questions, modality, "openbmb/MiniCPM-V-2_6")


def run_minimax_vl_01(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "MiniMaxAI/MiniMax-VL-01"

    engine_args = EngineArgs(
        model=model_name,
        max_num_seqs=2,
        limit_mm_per_prompt={modality: 1},
        trust_remote_code=True,
        tensor_parallel_size=8,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [
        [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": question}],
            }
        ]
        for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Mistral-3 HF-format
def run_mistral3(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    # NOTE: Need L40 (or equivalent) to avoid OOM
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        tensor_parallel_size=2,
        limit_mm_per_prompt={modality: 1},
        ignore_patterns=["consolidated.safetensors"],
    )

    prompts = [f"<s>[INST]{question}\n[IMG][/INST]" for question in questions]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Molmo
def run_molmo(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "allenai/Molmo-7B-D-0924"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={modality: 1},
    )

    prompts = [
        f"<|im_start|>user <image>\n{question}<|im_end|><|im_start|>assistant\n"
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Molmo2
def run_molmo2(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "allenai/Molmo2-8B"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={modality: 1},
        max_num_batched_tokens=36864,
    )

    if modality == "image":
        placeholder = "<|image|>"
    elif modality == "video":
        placeholder = "<|video|>"
    else:
        raise ValueError(f"Unsupported modality for molmo2: {modality}")

    prompts = [
        f"{placeholder}<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Nemontron_VL
def run_nemotron_vl(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={modality: 1},
    )

    assert modality == "image"
    placeholder = "<image>"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [
        [{"role": "user", "content": f"{placeholder}\n{question}"}]
        for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B/blob/main/conversation.py
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    stop_token_ids = [token_id for token_id in stop_token_ids if token_id is not None]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


# NVLM-D
def run_nvlm_d(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "nvidia/NVLM-D-72B"

    # Adjust this as necessary to fit in GPU
    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        tensor_parallel_size=4,
        limit_mm_per_prompt={modality: 1},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [
        [{"role": "user", "content": f"<image>\n{question}"}] for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# OpenPangu
def run_openpangu_vl(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "FreedomIntelligence/openPangu-VL-7B"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=4,
        trust_remote_code=True,
        enforce_eager=True,
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "[unused19]"
    elif modality == "video":
        placeholder = "[unused32]"

    prompts = [
        (
            f"<s>[unused9]系统：[unused10][unused9]用户：[unused18]{placeholder}[unused20]{question}[unused10][unused9]助手："
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Ovis
def run_ovis(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "AIDC-AI/Ovis2-1B"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        trust_remote_code=True,
        dtype="half",
        limit_mm_per_prompt={modality: 1},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [
        [{"role": "user", "content": f"<image>\n{question}"}] for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Ovis2_5
def run_ovis2_5(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "AIDC-AI/Ovis2.5-2B"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        trust_remote_code=True,
        dtype="half",
        limit_mm_per_prompt={modality: 1},
    )
    if modality == "image":
        placeholder = "<image>"
    elif modality == "video":
        placeholder = "<video>"

    prompts = [
        f"<|im_start|>user\n\n{placeholder}\n{question}<|im_end|>\n<|im_start|>assistant\n"
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# PaddleOCR-VL
def run_paddleocr_vl(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "PaddlePaddle/PaddleOCR-VL"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={modality: 1},
        trust_remote_code=True,
    )

    placeholder = "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>"
    prompts = [
        (f"<|begin_of_sentence|>User: {question}{placeholder}\nAssistant: ")
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# PaliGemma
def run_paligemma(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    # PaliGemma has special prompt format for VQA
    prompts = ["caption en" for _ in questions]
    engine_args = EngineArgs(
        model="google/paligemma-3b-mix-224",
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# PaliGemma 2
def run_paligemma2(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    # PaliGemma 2 has special prompt format for VQA
    prompts = ["caption en" for _ in questions]
    engine_args = EngineArgs(
        model="google/paligemma2-3b-ft-docci-448",
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Phi-3-Vision
def run_phi3v(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    prompts = [
        f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"
        for question in questions
    ]

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
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={"num_crops": 16},
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Phi-4-multimodal-instruct
def run_phi4mm(questions: list[str], modality: str) -> ModelRequestData:
    """
    Phi-4-multimodal-instruct supports both image and audio inputs. Here, we
    show how to process image inputs.
    """
    assert modality == "image"
    model_path = snapshot_download("microsoft/Phi-4-multimodal-instruct")
    # Since the vision-lora and speech-lora co-exist with the base model,
    # we have to manually specify the path of the lora weights.
    vision_lora_path = os.path.join(model_path, "vision-lora")
    prompts = [
        f"<|user|><|image_1|>{question}<|end|><|assistant|>" for question in questions
    ]
    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True,
        max_model_len=5120,
        max_num_seqs=2,
        max_num_batched_tokens=12800,
        enable_lora=True,
        max_lora_rank=320,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={"dynamic_hd": 16},
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        lora_requests=[LoRARequest("vision", 1, vision_lora_path)],
    )


# Pixtral HF-format
def run_pixtral_hf(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "mistral-community/pixtral-12b"

    # NOTE: Need L40 (or equivalent) to avoid OOM
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=6144,
        max_num_seqs=2,
        limit_mm_per_prompt={modality: 1},
    )

    prompts = [f"<s>[INST]{question}\n[IMG][/INST]" for question in questions]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Qwen-VL
def run_qwen_vl(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    engine_args = EngineArgs(
        model="Qwen/Qwen-VL",
        trust_remote_code=True,
        max_model_len=1024,
        max_num_seqs=2,
        hf_overrides={"architectures": ["QwenVLForConditionalGeneration"]},
        limit_mm_per_prompt={modality: 1},
    )

    prompts = [f"{question}Picture 1: <img></img>\n" for question in questions]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Qwen2-VL
def run_qwen2_vl(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "Qwen/Qwen2-VL-7B-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Qwen2.5-VL
def run_qwen2_5_vl(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Qwen2.5-Omni
def run_qwen2_5_omni(questions: list[str], modality: str):
    model_name = "Qwen/Qwen2.5-Omni-7B"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<|IMAGE|>"
    elif modality == "video":
        placeholder = "<|VIDEO|>"

    default_system = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
        "Group, capable of perceiving auditory and visual inputs, as well as "
        "generating text and speech."
    )

    prompts = [
        (
            f"<|im_start|>system\n{default_system}<|im_end|>\n"
            f"<|im_start|>user\n<|vision_bos|>{placeholder}<|vision_eos|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]
    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Qwen3-VL-Dense
def run_qwen3_vl(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "Qwen/Qwen3-VL-4B-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Qwen3-VL-MOE
def run_qwen3_vl_moe(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# R-4B
def run_r_vl(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "YannQi/R-4B"

    prompts = [
        f"<|im_start|>user <image>\n{question}<|im_end|><|im_start|>assistant\n"
        for question in questions
    ]

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=16384,
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# SkyworkR1V
def run_skyworkr1v(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "Skywork/Skywork-R1V-38B"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={modality: 1},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [
        [{"role": "user", "content": f"<image>\n{question}"}] for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Stop tokens for SkyworkR1V
    # https://huggingface.co/Skywork/Skywork-R1V-38B/blob/main/conversation.py
    stop_tokens = ["<｜end▁of▁sentence｜>", "<|endoftext|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


# SmolVLM2-2.2B-Instruct
def run_smolvlm(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        enforce_eager=True,
        mm_processor_kwargs={
            "max_image_size": {"longest_edge": 384},
        },
        limit_mm_per_prompt={modality: 1},
    )
    prompts = [
        (f"<|im_start|>User:<image>{question}<end_of_utterance>\nAssistant:")
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Step3
def run_step3(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "stepfun-ai/step3-fp8"

    # NOTE: Below are verified configurations for step3-fp8
    # on 8xH100 GPUs.
    engine_args = EngineArgs(
        model=model_name,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=8,
        limit_mm_per_prompt={modality: 1},
        reasoning_parser="step3",
    )

    prompts = [
        "<｜begin▁of▁sentence｜> You are a helpful assistant. <|BOT|>user\n "
        f"<im_patch>{question} <|EOT|><|BOT|>assistant\n<think>\n"
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# StepVL10B
def run_step_vl(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "stepfun-ai/Step3-VL-10B"
    engine_args = EngineArgs(
        model=model_name,
        max_num_batched_tokens=4096,
        tensor_parallel_size=1,
        trust_remote_code=True,
        limit_mm_per_prompt={modality: 1},
        reasoning_parser="deepseek_r1",
    )

    prompts = [
        "<｜begin▁of▁sentence｜> You are a helpful assistant.<|BOT|>user\n "
        f"<im_patch>{question} <|EOT|><|BOT|>assistant\n<think>\n"
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# omni-research/Tarsier-7b
def run_tarsier(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "omni-research/Tarsier-7b"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={modality: 1},
    )
    prompts = [(f"USER: <image>\n{question} ASSISTANT:") for question in questions]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


def run_tarsier2(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "omni-research/Tarsier2-Recap-7b"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        hf_overrides={
            "architectures": ["Tarsier2ForConditionalGeneration"],
            "model_type": "tarsier2",
        },
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


model_example_map = {
    "aria": run_aria,
    "aya_vision": run_aya_vision,
    "bagel": run_bagel,
    "bee": run_bee,
    "blip-2": run_blip2,
    "chameleon": run_chameleon,
    "command_a_vision": run_command_a_vision,
    "deepseek_vl_v2": run_deepseek_vl2,
    "deepseek_ocr": run_deepseek_ocr,
    "dots_ocr": run_dots_ocr,
    "eagle2_5": run_eagle2_5,
    "ernie45_vl": run_ernie45_vl,
    "fuyu": run_fuyu,
    "gemma3": run_gemma3,
    "gemma3n": run_gemma3n,
    "glm4v": run_glm4v,
    "glm4_1v": run_glm4_1v,
    "glm4_5v": run_glm4_5v,
    "glm4_5v_fp8": run_glm4_5v_fp8,
    "glm_ocr": run_glm_ocr,
    "h2ovl_chat": run_h2ovl,
    "hunyuan_vl": run_hunyuan_vl,
    "hyperclovax_seed_vision": run_hyperclovax_seed_vision,
    "idefics3": run_idefics3,
    "interns1": run_interns1,
    "internvl_chat": run_internvl,
    "kanana_v": run_kanana_v,
    "keye_vl": run_keye_vl,
    "keye_vl1_5": run_keye_vl1_5,
    "kimi_vl": run_kimi_vl,
    "lightonocr": run_lightonocr,
    "lfm2_vl": run_lfm2_vl,
    "llama4": run_llama4,
    "llava": run_llava,
    "llava-next": run_llava_next,
    "llava-next-video": run_llava_next_video,
    "llava-onevision": run_llava_onevision,
    "mantis": run_mantis,
    "minicpmo": run_minicpmo,
    "minicpmv": run_minicpmv,
    "minimax_vl_01": run_minimax_vl_01,
    "mistral3": run_mistral3,
    "molmo": run_molmo,
    "molmo2": run_molmo2,
    "nemotron_vl": run_nemotron_vl,
    "NVLM_D": run_nvlm_d,
    "openpangu_vl": run_openpangu_vl,
    "ovis": run_ovis,
    "ovis2_5": run_ovis2_5,
    "paddleocr_vl": run_paddleocr_vl,
    "paligemma": run_paligemma,
    "paligemma2": run_paligemma2,
    "phi3_v": run_phi3v,
    "phi4_mm": run_phi4mm,
    "pixtral_hf": run_pixtral_hf,
    "qwen_vl": run_qwen_vl,
    "qwen2_vl": run_qwen2_vl,
    "qwen2_5_vl": run_qwen2_5_vl,
    "qwen2_5_omni": run_qwen2_5_omni,
    "qwen3_vl": run_qwen3_vl,
    "qwen3_vl_moe": run_qwen3_vl_moe,
    "rvl": run_r_vl,
    "skywork_chat": run_skyworkr1v,
    "smolvlm": run_smolvlm,
    "step3": run_step3,
    "stepvl": run_step_vl,
    "tarsier": run_tarsier,
    "tarsier2": run_tarsier2,
}


MODELS_NEED_VIDEO_METADATA = [
    "glm4_1v",
    "glm_ocr",
    "glm4_5v",
    "glm4_5v_fp8",
    "molmo2",
    "qwen3_vl",
    "qwen3_vl_moe",
]


def get_multi_modal_input(args):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """
    if args.modality == "image":
        # Input image and question
        image = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
        img_questions = [
            "What is the content of this image?",
            "Describe the content of this image in detail.",
            "What's in the image?",
            "Where is this image taken?",
        ]

        return {
            "data": image,
            "questions": img_questions,
        }

    if args.modality == "video":
        # Input video and question
        needs_metadata = args.model_type in MODELS_NEED_VIDEO_METADATA
        video = VideoAsset(name="baby_reading", num_frames=args.num_frames).np_ndarrays
        metadata = VideoAsset(name="baby_reading", num_frames=args.num_frames).metadata
        vid_questions = ["Why is this video funny?"]

        return {
            "data": ([(video, metadata)] if needs_metadata else video),
            "questions": vid_questions,
        }

    msg = f"Modality {args.modality} is not supported."
    raise ValueError(msg)


def apply_image_repeat(
    image_repeat_prob, num_prompts, data, prompts: list[str], modality
):
    """Repeats images with provided probability of "image_repeat_prob".
    Used to simulate hit/miss for the MM preprocessor cache.
    """
    assert image_repeat_prob <= 1.0 and image_repeat_prob >= 0
    no_yes = [0, 1]
    probs = [1.0 - image_repeat_prob, image_repeat_prob]

    inputs = []
    inputs_with_empty_media = []
    cur_image = data
    for i in range(num_prompts):
        if image_repeat_prob is not None:
            res = random.choices(no_yes, probs)[0]
            if res == 0:
                # No repeat => Modify one pixel
                cur_image = cur_image.copy()
                new_val = (i // 256 // 256, i // 256, i % 256)
                cur_image.putpixel((0, 0), new_val)

        uuid = "uuid_{}".format(i)

        inputs.append(
            {
                "prompt": prompts[i % len(prompts)],
                "multi_modal_data": {modality: cur_image},
                "multi_modal_uuids": {modality: uuid},
            }
        )

        inputs_with_empty_media.append(
            {
                "prompt": prompts[i % len(prompts)],
                "multi_modal_data": {modality: None},
                "multi_modal_uuids": {modality: uuid},
            }
        )

    return inputs, inputs_with_empty_media


@contextmanager
def time_counter(enable: bool):
    if enable:
        import time

        start_time = time.time()
        yield
        elapsed_time = time.time() - start_time
        print("-" * 50)
        print("-- generate time = {}".format(elapsed_time))
        print("-" * 50)
    else:
        yield


def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "vision language models for text generation"
    )
    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        default="llava",
        choices=model_example_map.keys(),
        help='Huggingface "model_type".',
    )
    parser.add_argument(
        "--num-prompts", type=int, default=4, help="Number of prompts to run."
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="image",
        choices=["image", "video"],
        help="Modality of the input.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to extract from the video.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set the seed when initializing `vllm.LLM`.",
    )

    parser.add_argument(
        "--image-repeat-prob",
        type=float,
        default=None,
        help="Simulates the hit-ratio for multi-modal preprocessor cache (if enabled)",
    )

    parser.add_argument(
        "--disable-mm-processor-cache",
        action="store_true",
        help="If True, disables caching of multi-modal processor.",
    )

    parser.add_argument(
        "--time-generate",
        action="store_true",
        help="If True, then print the total generate() call time",
    )

    parser.add_argument(
        "--use-different-prompt-per-request",
        action="store_true",
        help="If True, then use different prompt (with the same multi-modal "
        "data) for each request.",
    )

    parser.add_argument(
        "--verify-mm-cache-hit-with-uuids",
        action="store_true",
        help="If True, will send all requests in a second batch with empty mm "
        "data to verify cache hits with UUIDs.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "-tp",
        type=int,
        default=None,
        help="Tensor parallel size to override the model's default setting. ",
    )
    return parser.parse_args()


def main(args):
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    if args.tensor_parallel_size is not None and args.tensor_parallel_size < 1:
        raise ValueError(
            f"tensor_parallel_size must be a positive integer, "
            f"got {args.tensor_parallel_size}"
        )

    modality = args.modality
    mm_input = get_multi_modal_input(args)
    data = mm_input["data"]
    questions = mm_input["questions"]

    req_data = model_example_map[model](questions, modality)

    # Disable other modalities to save memory
    default_limits = {"image": 0, "video": 0, "audio": 0}
    req_data.engine_args.limit_mm_per_prompt = default_limits | dict(
        req_data.engine_args.limit_mm_per_prompt or {}
    )

    engine_args = asdict(req_data.engine_args) | {
        "seed": args.seed,
        "mm_processor_cache_gb": 0 if args.disable_mm_processor_cache else 4,
    }
    if args.tensor_parallel_size is not None:
        engine_args["tensor_parallel_size"] = args.tensor_parallel_size
    llm = LLM(**engine_args)

    # Don't want to check the flag multiple times, so just hijack `prompts`.
    prompts = (
        req_data.prompts
        if args.use_different_prompt_per_request
        else [req_data.prompts[0]]
    )

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = (
        SamplingParams(
            temperature=0.2, max_tokens=64, stop_token_ids=req_data.stop_token_ids
        )
        if req_data.sampling_params is None
        else req_data.sampling_params
    )

    assert args.num_prompts > 0
    if args.num_prompts == 1:
        # Single inference
        uuid = "uuid_0"
        inputs = {
            "prompt": prompts[0],
            "multi_modal_data": {modality: data},
            "multi_modal_uuids": {modality: uuid},
        }
        inputs_with_empty_media = {
            "prompt": prompts[0],
            "multi_modal_data": {modality: None},
            "multi_modal_uuids": {modality: uuid},
        }
    else:
        # Batch inference
        if args.image_repeat_prob is not None:
            # Repeat images with specified probability of "image_repeat_prob"
            inputs, inputs_with_empty_media = apply_image_repeat(
                args.image_repeat_prob,
                args.num_prompts,
                data,
                prompts,
                modality,
            )
        else:
            # Use the same image for all prompts
            inputs = []
            inputs_with_empty_media = []
            for i in range(args.num_prompts):
                uuid = "uuid_{}".format(i)
                inputs.append(
                    {
                        "prompt": prompts[i % len(prompts)],
                        "multi_modal_data": {modality: data},
                        "multi_modal_uuids": {modality: uuid},
                    }
                )
                inputs_with_empty_media.append(
                    {
                        "prompt": prompts[i % len(prompts)],
                        "multi_modal_data": {modality: None},
                        "multi_modal_uuids": {modality: uuid},
                    }
                )

    # Add LoRA request if applicable
    lora_request = (
        req_data.lora_requests * args.num_prompts if req_data.lora_requests else None
    )

    with time_counter(args.time_generate):
        outputs = llm.generate(
            inputs,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )

    print("-" * 50)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        print("-" * 50)

    if args.verify_mm_cache_hit_with_uuids:
        try:
            # Verify cache hits with UUIDs
            print(
                "Sending a second batch of requests with empty media"
                " and matching UUIDs."
            )
            outputs = llm.generate(
                inputs_with_empty_media,
                sampling_params=sampling_params,
                lora_request=lora_request,
            )
            print("-" * 50)
            for o in outputs:
                generated_text = o.outputs[0].text
                print(generated_text)
                print("-" * 50)
        except Exception as e:
            print(f"Failed to verify cache hits with UUIDs. Error: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
