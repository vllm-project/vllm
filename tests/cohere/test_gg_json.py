# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import asyncio

from test_schema_data import synth_prompts
from test_utils import (
    RunMode,
    _create_reasoning_config,
    generate_guided_output,
    make_speculative_config,
    validate_output,
)
from test_utils_engine_args import get_async_engine_args_with_overrides
from transformers import AutoTokenizer

from vllm.cohere.guided_decoding.tool_grammar import get_text_model_name
from vllm.v1.engine.async_llm import AsyncLLM


async def run_json_validation_tests(args):
    # Get effective engine args with hardware profile args + test-specific overrides
    engine_args = get_async_engine_args_with_overrides(
        test_kwargs={
            "model": args.model,
            "dtype": "auto",
            "max_model_len": args.max_model_len,
            "tensor_parallel_size": args.tensor_parallel_size,
            "structured_outputs_config": {"backend": "xgrammar"},
            "speculative_config": make_speculative_config(args),
            "reasoning_config": _create_reasoning_config(),
            "async_scheduling": True,
        },
        engine_args_override=getattr(args, "engine_args", None),
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model_arch = get_text_model_name(engine.model_config)
    try:
        # JSON Object validation
        json_results = await asyncio.gather(
            *[
                generate_guided_output(
                    engine, p["prompt"], f"json-{i}", None, tokenizer
                )
                for i, p in enumerate(synth_prompts)
            ]
        )
        bad_json, _ = validate_output(json_results, {}, model_arch)
        assert len(bad_json) <= 5, f"Too many invalid JSON objects: {len(bad_json)}"
        print("✅ JSON object validation passed (%s)", args.mode.value)

        # JSON Schema validation
        schema_map = {i: p["schema"] for i, p in enumerate(synth_prompts)}
        schema_results = await asyncio.gather(
            *[
                generate_guided_output(
                    engine, p["prompt"], f"schema-{i}", p["schema"], tokenizer
                )
                for i, p in enumerate(synth_prompts)
            ]
        )
        bad_json, bad_schema = validate_output(schema_results, schema_map, model_arch)
        assert len(bad_json) <= 5, f"Too many invalid JSON: {len(bad_json)}"
        assert len(bad_schema) <= 5, f"Too many invalid schema JSON: {len(bad_schema)}"
        print("✅ JSON schema validation passed (%s)", args.mode.value)

    finally:
        engine.shutdown()


def parse_args():
    p = argparse.ArgumentParser(description="Guided JSON & JSON Schema validation")
    p.add_argument("--model", type=str, default="CohereForAI/c4ai-command-r-v01")
    p.add_argument("--tensor_parallel_size", type=int, default=2)
    p.add_argument(
        "--mode", type=RunMode, choices=list(RunMode), default=RunMode.NON_SPECULATIVE
    )
    p.add_argument("--method", type=str, default="eagle")
    p.add_argument("--draft_model", type=str, default=None)
    p.add_argument("--num_spec_tokens", type=int, default=4)
    p.add_argument("--draft_tp", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=32000)
    return p.parse_args()


def main():
    args = parse_args()
    asyncio.run(run_json_validation_tests(args))


if __name__ == "__main__":
    main()
