# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Consolidated test for ViT attention backend functionality across multiple models.

This test validates that each multimodal model can successfully generate outputs
using different ViT attention backends. Tests are parametrized by model and backend.
"""

from dataclasses import asdict
from typing import Any

import pytest
from transformers import AutoProcessor

from vllm import LLM, EngineArgs, SamplingParams
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.multimodal.utils import encode_image_base64
from vllm.multimodal.video import sample_frames_from_video
from vllm.platforms import current_platform

from ....utils import create_new_process_for_each_test
from ...utils import dummy_hf_overrides

# Dots.OCR prompt from official repository
# https://github.com/rednote-hilab/dots.ocr/blob/d72d1d8c5bdd0362eb264f714cdbd1e5daa7cdff/dots_ocr/utils/prompts.py#L3
# ruff: noqa: E501
DOTS_OCR_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"


# Model configurations
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "dots_ocr": {
        "model_name": "rednote-hilab/dots.ocr",
        "interface": "llm_chat",
        "max_model_len": 32768,
        "max_num_seqs": 1,
        "limit_mm_per_prompt": {"image": 1},
        "sampling_params": {
            "temperature": 0.1,
            "max_tokens": 16384,
            "top_p": 0.9,
            "stop_token_ids": None,
        },
        "use_specific_image": "stop_sign",
        "prompt_builder": "build_dots_ocr_prompt",
        "output_validator": lambda x: len(x) > 10 and "stop" in x.lower(),
    },
    "ernie45_vl": {
        "model_name": "baidu/ERNIE-4.5-VL-28B-A3B-PT",
        "interface": "llm_generate",
        "max_model_len": 16384,
        "max_num_seqs": 2,
        "sampling_params": {
            "temperature": 0.0,
            "max_tokens": 256,
            "stop_token_ids": None,
        },
        "use_processor": True,
        "question": "What is the content of each image?",
    },
    "glm4_1v": {
        "model_name": "zai-org/GLM-4.1V-9B-Thinking",
        "interface": "llm_generate",
        "max_model_len": 32768,
        "max_num_seqs": 2,
        "sampling_params": {
            "temperature": 0.0,
            "max_tokens": 256,
            "stop_token_ids": None,
        },
        "use_processor": True,
        "question": "What is the content of each image?",
    },
    "keye_vl": {
        "model_name": "Kwai-Keye/Keye-VL-8B-Preview",
        "interface": "llm_generate",
        "max_model_len": 8192,
        "max_num_seqs": 5,
        "sampling_params": {
            "temperature": 0.0,
            "max_tokens": 256,
            "stop_token_ids": None,
        },
        "supported_backends": {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
        },
        "use_processor": True,
        "question": "What is the content of each image?",
    },
    "ovis2_5": {
        "model_name": "AIDC-AI/Ovis2.5-2B",
        "interface": "llm_generate",
        "max_model_len": 8192,
        "max_num_seqs": 2,
        "sampling_params": {
            "temperature": 0.0,
            "max_tokens": 256,
            "stop_token_ids": None,
        },
        "prompt_builder": "build_ovis_prompt",
        "question": "What is the content of each image?",
    },
    "qwen2_5_vl": {
        "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "interface": "vllm_runner",
        "media_type": "video",
        "max_model_len": 4000,
        "max_num_seqs": 1,
        "limit_mm_per_prompt": {"video": 1},
        "sampling_params": {
            "max_tokens": 128,
        },
        "runner_kwargs": {
            "runner": "generate",
            "dtype": "bfloat16",
        },
        "video_params": {
            "num_frames": 16,
            "pruning_rates": [0.0, 0.75],
        },
    },
    "qwen2_5_omni": {
        "model_name": "Qwen/Qwen2.5-Omni-3B",
        "interface": "llm_generate",
        "max_model_len": 32768,
        "max_num_seqs": 2,
        "limit_mm_per_prompt": {"image": 3, "video": 3, "audio": 3},
        "sampling_params": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 16384,
        },
        "use_processor": True,
        "question": "What is the content of each image?",
    },
    "qwen3_omni": {
        "model_name": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "interface": "llm_generate",
        "max_model_len": 32768,
        "max_num_seqs": 2,
        "limit_mm_per_prompt": {"image": 3, "video": 3, "audio": 3},
        "sampling_params": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 16384,
        },
        "use_processor": True,
        "question": "What is the content of each image?",
    },
}


# Prompt builder functions
def build_dots_ocr_prompt(images, config):
    """Build Dots.OCR specific prompt with OCR instructions."""
    # Use only stop_sign image for Dots.OCR
    image = images[0]  # Already filtered to stop_sign

    image_url = f"data:image/jpeg;base64,{encode_image_base64(image)}"

    placeholders = [{"type": "image_url", "image_url": {"url": image_url}}]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {
                    "type": "text",
                    "text": f"<|img|><|imgpad|><|endofimg|>{DOTS_OCR_PROMPT}",
                },
            ],
        },
    ]

    return messages


def build_processor_prompt(images, config):
    """Build prompt using AutoProcessor.apply_chat_template()."""
    processor = AutoProcessor.from_pretrained(
        config["model_name"], trust_remote_code=True
    )

    image_urls = [
        f"data:image/jpeg;base64,{encode_image_base64(img)}" for img in images
    ]
    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": config["question"]},
            ],
        },
    ]

    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_ovis_prompt(images, config):
    """Build Ovis2.5 specific prompt with custom format."""
    image_urls = [
        f"data:image/jpeg;base64,{encode_image_base64(img)}" for img in images
    ]

    placeholders = "\n".join(
        f"Image-{i}: <image>\n" for i, _ in enumerate(image_urls, start=1)
    )

    return (
        f"<|im_start|>user\n\n{placeholders}\n{config['question']}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_qwen2_5_video_prompt():
    """Build Qwen2.5-VL video prompt with EVS placeholder."""
    return (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{VIDEO_PLACEHOLDER}"
        "Describe this video with a short sentence (no more than 20 words)"
        "<|im_end|><|im_start|>assistant\n"
    )


# Handler functions
def run_llm_generate_test(config, mm_encoder_attn_backend, image_assets):
    """Standard LLM.generate() interface handler."""
    images = [asset.pil_image for asset in image_assets]

    # Build prompt
    if config.get("use_processor"):
        prompt = build_processor_prompt(images, config)
    else:
        prompt_builder_name = config.get("prompt_builder", "build_ovis_prompt")
        prompt_builder = globals()[prompt_builder_name]
        prompt = prompt_builder(images, config)

    # Determine limit_mm_per_prompt
    limit_mm_per_prompt = config.get("limit_mm_per_prompt", {"image": len(images)})

    # Create engine
    engine_args = EngineArgs(
        model=config["model_name"],
        trust_remote_code=True,
        max_model_len=config["max_model_len"],
        max_num_seqs=config["max_num_seqs"],
        limit_mm_per_prompt=limit_mm_per_prompt,
        mm_encoder_attn_backend=mm_encoder_attn_backend,
        hf_overrides=dummy_hf_overrides,
        load_format="dummy",
    )

    engine_dict = asdict(engine_args) | {"seed": 42}
    llm = LLM(**engine_dict)

    # Generate
    sampling_params = SamplingParams(**config["sampling_params"])
    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"image": images},
        },
        sampling_params=sampling_params,
    )

    # Validate
    for o in outputs:
        generated_text = o.outputs[0].text
        validator = config.get("output_validator", lambda x: len(x) > 10)
        assert validator(generated_text), (
            f"Validation failed for {config['model_name']}: {generated_text}"
        )


def run_llm_chat_test(config, mm_encoder_attn_backend, image_assets):
    """LLM.chat() interface handler for Dots.OCR."""
    # Filter to stop_sign image only
    stop_sign_image = [
        asset.pil_image for asset in image_assets if asset.name == "stop_sign"
    ][0]

    # Build messages
    messages = build_dots_ocr_prompt([stop_sign_image], config)

    # Create engine
    engine_args = EngineArgs(
        model=config["model_name"],
        trust_remote_code=True,
        max_model_len=config["max_model_len"],
        max_num_seqs=config["max_num_seqs"],
        limit_mm_per_prompt=config["limit_mm_per_prompt"],
        mm_encoder_attn_backend=mm_encoder_attn_backend,
        hf_overrides=dummy_hf_overrides,
        load_format="dummy",
    )

    engine_dict = asdict(engine_args) | {"seed": 42}
    llm = LLM(**engine_dict)

    # Generate using chat
    sampling_params = SamplingParams(**config["sampling_params"])
    outputs = llm.chat(messages=messages, sampling_params=sampling_params)

    # Validate
    for o in outputs:
        generated_text = o.outputs[0].text
        validator = config.get("output_validator", lambda x: len(x) > 10)
        assert validator(generated_text), (
            f"Validation failed for {config['model_name']}: {generated_text}"
        )


def run_video_test(config, mm_encoder_attn_backend, video_assets, vllm_runner):
    """Video test with EVS (Efficient Video Sampling) handler."""
    for pruning_rate in config["video_params"]["pruning_rates"]:
        num_frames = config["video_params"]["num_frames"]

        # Sample frames from video
        sampled_vids = [
            sample_frames_from_video(asset.np_ndarrays, num_frames)
            for asset in video_assets
        ]

        # Build prompt and prepare video
        prompt = build_qwen2_5_video_prompt()
        prompts = [prompt]
        videos = [sampled_vids[0]]

        # Run with vllm_runner context manager
        with vllm_runner(
            config["model_name"],
            max_model_len=config["max_model_len"],
            max_num_seqs=config["max_num_seqs"],
            limit_mm_per_prompt=config["limit_mm_per_prompt"],
            tensor_parallel_size=1,
            video_pruning_rate=pruning_rate,
            mm_encoder_attn_backend=mm_encoder_attn_backend,
            hf_overrides=dummy_hf_overrides,
            load_format="dummy",
            **config["runner_kwargs"],
        ) as vllm_model:
            outputs = vllm_model.generate_greedy(
                prompts,
                config["sampling_params"]["max_tokens"],
                videos=videos,
            )

            # Validate output
            assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"
            output_ids, output_text = outputs[0]
            assert len(output_ids) > 0, "Generated no output IDs"
            assert len(output_text) > 0, "Generated empty text"
            assert isinstance(output_text, str), (
                f"Output is not string: {type(output_text)}"
            )


# Main test function
@pytest.mark.parametrize("model_key", list(MODEL_CONFIGS.keys()))
@pytest.mark.parametrize(
    "mm_encoder_attn_backend",
    [None] + current_platform.get_supported_vit_attn_backends(),
)
@create_new_process_for_each_test()
def test_vit_backend_functionality(
    model_key: str,
    mm_encoder_attn_backend: AttentionBackendEnum | None,
    image_assets,
    video_assets,
    vllm_runner,
    request,
):
    """Test ViT attention backend functionality for multimodal models.

    This test validates that each model can successfully generate outputs
    using different ViT attention backends. The test:
    1. Filters unsupported backends per model
    2. Applies appropriate GPU marks
    3. Routes to the correct test handler based on interface
    4. Validates output meets minimum requirements
    """
    config = MODEL_CONFIGS[model_key]

    # Step 1: Backend filtering
    if (
        "supported_backends" in config
        and mm_encoder_attn_backend is not None
        and mm_encoder_attn_backend not in config["supported_backends"]
    ):
        pytest.skip(
            f"{model_key} does not support {mm_encoder_attn_backend} backend now."
        )

    # Step 2: Apply GPU marks dynamically
    if "gpu_marks" in config:
        for mark in config["gpu_marks"]:
            request.applymarker(mark)

    # Step 3: Route to appropriate handler
    if config.get("media_type") == "video":
        run_video_test(config, mm_encoder_attn_backend, video_assets, vllm_runner)
    elif config["interface"] == "llm_chat":
        run_llm_chat_test(config, mm_encoder_attn_backend, image_assets)
    elif config["interface"] == "llm_generate":
        run_llm_generate_test(config, mm_encoder_attn_backend, image_assets)
    else:
        raise ValueError(f"Unknown interface: {config['interface']}")
