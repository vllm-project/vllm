#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Vision model test for Command-A Vision model.
Tests that the model can correctly identify images of a duck and a lion.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from PIL import Image
from test_utils import (
    RunMode,
    _create_reasoning_config,
    make_speculative_config,
    validate_output,
)
from test_utils_engine_args import get_async_engine_args_with_overrides
from transformers import AutoProcessor

from vllm import SamplingParams
from vllm.cohere.guided_decoding.convert_to_structural_tag_format import (  # noqa: E501
    convert_schema_to_structural_tags,
)
from vllm.cohere.guided_decoding.tool_grammar import get_text_model_name
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.engine.async_llm import AsyncLLM

# Get the directory where this test file is located
TEST_DIR = Path(__file__).parent
FIXTURES_DIR = TEST_DIR / "fixtures"
# Local image paths
IMAGE_PATHS = [
    FIXTURES_DIR / "duck.jpg",  # duck
    FIXTURES_DIR / "lion.jpg",  # lion
]
QUESTION = (
    "What is the content of each image?, Generate a JSON object with "
    "the fields 'image_index' and 'description'."
)
PROMPTS = [QUESTION] * 32


async def generate_response(
    engine, prompt: dict, images, request_id: int, thinking_token_budget: int = 30000
) -> str:
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.75,
        max_tokens=3000,
        thinking_token_budget=thinking_token_budget,
    )
    schema = {"type": "object"}
    structural_tag = convert_schema_to_structural_tags(schema=schema, engine=engine)
    if structural_tag is not None:
        sampling_params.structured_outputs = StructuredOutputsParams(
            structural_tag=structural_tag, backend="xgrammar"
        )
    stream = engine.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"image": images},
        },
        sampling_params=sampling_params,
        request_id=str(request_id),
    )
    final_output = None
    async for output in stream:  # iterate over streamed chunks
        final_output = output  # keep updating until final chunk
    text = final_output.outputs[0].text if final_output else ""
    return text


async def run_vision_test(
    model_path: str,
    tensor_parallel_size: int = 1,
    args=None,
) -> None:
    """
    Run vision test similar to load_command_a_vision from examples.
    Args:
        model_path: Path to the model checkpoint
        tensor_parallel_size: Number of GPUs for tensor parallelism
    """
    print(f"Loading model from: {model_path}")
    print(f"Using tensor_parallel_size: {tensor_parallel_size}")
    # Load images from local files
    print("Loading images from local files...")
    images = [Image.open(path) for path in IMAGE_PATHS]
    # Set up engine args similar to load_command_a_vision
    engine_args = get_async_engine_args_with_overrides(
        test_kwargs={
            "model": model_path,
            "max_model_len": 32768,
            "tensor_parallel_size": tensor_parallel_size,
            "limit_mm_per_prompt": {"image": len(images)},
            "structured_outputs_config": {"backend": "xgrammar"},
            "enable_prefix_caching": False,
            "speculative_config": make_speculative_config(args),
            "reasoning_config": _create_reasoning_config(),
            "async_scheduling": True,
        },
        engine_args_override=getattr(args, "engine_args", None),
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    model_arch = get_text_model_name(engine.model_config)
    # Create message with image placeholders
    placeholders = [{"type": "image"} for _ in images]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": QUESTION},
            ],
        }
    ]
    # Load processor and apply chat template
    print("Loading processor and preparing prompt...")
    processor = AutoProcessor.from_pretrained(model_path)
    prompt_template = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Run inference across different thinking budgets
    thinking_budgets = args.thinking_budgets
    print(f"Running inference with thinking_budgets={thinking_budgets}...")
    for budget in thinking_budgets:
        print(f"  Testing thinking_token_budget={budget}...")
        generated_text = await asyncio.gather(
            *[
                generate_response(
                    engine, prompt_template, images, i, thinking_token_budget=budget
                )
                for i, prompt in enumerate(PROMPTS)
            ]
        )
        bad_json, _ = validate_output(generated_text, {}, model_arch)
        assert len(bad_json) <= 5, (
            f"Too many invalid JSON objects (budget={budget}): {len(bad_json)}"
        )
        print(f"  ✅ budget={budget}: {len(bad_json)}/{len(PROMPTS)} bad JSON")
    print(f"✅ MM + GG + TB validation passed ({args.mode.value})")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Test vision model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Number of GPUs for tensor parallelism (default: 2)",
    )
    parser.add_argument(
        "--mode", type=RunMode, choices=list(RunMode), default=RunMode.NON_SPECULATIVE
    )
    parser.add_argument("--method", type=str, default="eagle")
    parser.add_argument("--draft_model", type=str, default=None)
    parser.add_argument("--num_spec_tokens", type=int, default=4)
    parser.add_argument("--draft_tp", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=32000)
    parser.add_argument(
        "--thinking-budgets",
        type=int,
        nargs="+",
        default=[30000],
        help="Thinking token budgets to test (default: [30000])",
    )
    parser.add_argument(
        "--engine-args",
        type=str,
        default=None,
        help=(
            "CLI-style engine args to pass to AsyncLLM (e.g., '--max-model-len 32768 "
            "--enable-chunked-prefill'). "
            "If not provided, uses VLLM_HARDWARE_PROFILE_ARGS "
            "environment variable."
        ),
    )
    args = parser.parse_args()
    asyncio.run(run_vision_test(args.model, args.tensor_parallel_size, args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
