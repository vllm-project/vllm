# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import asyncio

from test_utils import (
    RunMode,
    _create_reasoning_config,
    generate_guided_output,
    get_input_text,
    make_speculative_config,
    validate_output,
)
from test_utils_engine_args import get_async_engine_args_with_overrides
from transformers import AutoTokenizer

from vllm.cohere.guided_decoding.tool_grammar import get_text_model_name
from vllm.v1.engine.async_llm import AsyncLLM

# ----------------------- Validation Runner --------------------- #


async def check_long_context_validation(args):
    """Run long context validation for one model."""
    spec_config = (
        make_speculative_config(args) if args.mode != RunMode.NON_SPECULATIVE else None
    )

    # Get effective engine args with hardware profile args + test-specific overrides
    engine_args = get_async_engine_args_with_overrides(
        test_kwargs={
            "model": args.model,
            "dtype": "auto",
            "max_model_len": args.max_model_len,
            "tensor_parallel_size": args.tensor_parallel_size,
            "structured_outputs_config": {"backend": "xgrammar"},
            "speculative_config": spec_config,
            "reasoning_config": _create_reasoning_config(),
            "async_scheduling": True,
        },
        engine_args_override=getattr(args, "engine_args", None),
    )

    engine = AsyncLLM.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    long_context_texts, schema = get_input_text()
    schema_list = [schema] * len(long_context_texts)
    model_arch = get_text_model_name(engine.model_config)
    try:
        tasks: list[asyncio.Task] = [
            asyncio.create_task(
                generate_guided_output(engine, text, str(i), schema, tokenizer)
            )
            for i, text in enumerate(long_context_texts)
        ]

        results = await asyncio.gather(*tasks)
        invalid_json, invalid_schema_json = validate_output(
            results, schema_list, model_arch
        )

        assert not invalid_json, "Invalid JSON outputs in long context test"
        assert not invalid_schema_json, "Invalid JSON schemas in long context test"

        print("✅ Long context validation passed (%s)", args.mode.value)
    finally:
        engine.shutdown()


# --------------------------- Test Runner ----------------------- #
async def run_long_context_tests(args):
    if args.mode in (RunMode.NON_SPECULATIVE, RunMode.BOTH):
        args.mode = RunMode.NON_SPECULATIVE
        await check_long_context_validation(args)

    if args.mode in (RunMode.SPECULATIVE, RunMode.BOTH):
        args.mode = RunMode.SPECULATIVE
        await check_long_context_validation(args)

    print("------------- Long context test finished -------------")


# ------------------------------ CLI ---------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description="Long context guided generation validation"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)

    # Mode selection
    parser.add_argument(
        "--mode", type=RunMode, choices=list(RunMode), default=RunMode.BOTH
    )

    # Speculative decoding args
    parser.add_argument("--method", type=str, default="eagle")
    parser.add_argument(
        "--draft_model",
        type=str,
        default=None,
        help="Draft model for speculative decoding",
    )
    parser.add_argument(
        "--num_spec_tokens", type=int, default=4, help="Number of speculative tokens"
    )
    parser.add_argument(
        "--draft_tp",
        type=int,
        default=1,
        help="Tensor parallel size for speculative model",
    )
    parser.add_argument("--max_model_len", type=int, default=32_000)

    return parser.parse_args()


def main():
    args = parse_args()
    asyncio.run(run_long_context_tests(args))


if __name__ == "__main__":
    main()
