# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import asyncio
import time
from enum import Enum

from test_const import REASONING_PROMPT
from test_schema_data import synth_prompts
from test_utils import (
    RunMode,
    _create_reasoning_config,
    add_reasoning_prompt,
    make_speculative_config,
    validate_output,
)
from test_utils_engine_args import get_async_engine_args_with_overrides
from transformers import AutoTokenizer

from vllm import SamplingParams, TokensPrompt
from vllm.cohere.guided_decoding.cohere_constants import END_THINKING_TOKEN
from vllm.cohere.guided_decoding.convert_to_structural_tag_format import (  # noqa: E501
    convert_schema_to_structural_tags,
)
from vllm.cohere.guided_decoding.tool_grammar import get_text_model_name
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.engine.async_llm import AsyncLLM

st = time.time()

MAX_OUTPUT_TOKEN = 32000  # Global variable for maximum output tokens


class TestMode(Enum):
    REASONING = "reasoning"
    GUIDED_GENERATION = "guided_generation"


# Define the test cases
thinking_budgets_reasoning = [300, 400, -1, 0, 0, -1, 5000, 6000, -1, -1, -1, 45, 652]
continue_thinking_budgets_reasoning = [0, 300, 400, 500, 600, -1, 700]


def validate_logprobs(batch_results):
    """Validate that logprobs are returned for each request"""
    print("Validating logprobs...")

    for request_id, result in enumerate(batch_results):
        logprobs = result.get("logprobs")

        # Basic validation that logprobs exist
        assert logprobs is not None, f"Request {request_id}: No logprobs returned"
        assert len(logprobs) > 0, f"Request {request_id}: Empty logprobs list"

        print(f"Request {request_id}: Got {len(logprobs)} token logprobs")

    print("✅ Logprobs validation passed!")


def validate_outputs(
    batch_results,
    tokenizer,
    reasoning_mode,
    thinking_token_budgets=None,
    model_arch=None,
    continue_thinking=False,
):
    # First validate logprobs
    validate_logprobs(batch_results)

    # Extract text from results for existing validation
    text_outputs = [result["text"] for result in batch_results]
    text_tokens = [result["token_ids"] for result in batch_results]
    tokenized_prompts = [result["tokenized_prompt"] for result in batch_results]
    if reasoning_mode == TestMode.REASONING:
        for request_id, (output, prompt) in enumerate(
            zip(text_tokens, tokenized_prompts)
        ):
            end_thinking_token_id = tokenizer.encode(
                END_THINKING_TOKEN, add_special_tokens=True
            )[1]

            if thinking_token_budgets[request_id] == -1:
                continue
            else:
                index_et = output.index(end_thinking_token_id)

            print(
                f"Request ID: {request_id}, Thinking Budget: \
                    {thinking_token_budgets[request_id]}, \
                        Thinking Token Index: {index_et}"
            )
            # The reason we add 2 is because the tokenizer
            # while encode adds a <BOS> token
            # We also count the <START_THINKING TOKEN> token
            # as well along with all the other budget
            # tokens
            index_offset = 1
            if continue_thinking:
                start_thinking_ids = tokenizer.encode(
                    "<|START_THINKING|>", add_special_tokens=False
                )[0]
                start_thinking_pos = (
                    len(prompt) - 1 - prompt[::-1].index(start_thinking_ids)
                )
                index_et += len(prompt) - (start_thinking_pos + 1)
                index_offset = 0
            assert index_et == thinking_token_budgets[request_id] + index_offset, (
                f"Expected thinking budget \
                    {thinking_token_budgets[request_id] + index_offset}\
                    , but got {index_et} for request {request_id}"
            )

    elif reasoning_mode == TestMode.GUIDED_GENERATION:
        validate_output(text_outputs, {}, model_arch)
        print("✅ Guided generation with reasoning output validation passed.")

    else:
        raise ValueError("Unsupported mode for validation.")


async def gen(
    engine,
    example_input,
    id,
    tokenizer,
    model,
    thinking_budget,
    reasoning_mode,
    continue_thinking=False,
):
    print(
        f"[DEBUG] gen() called with thinking_budget: {thinking_budget} for request {id}"
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=-1,
        max_tokens=MAX_OUTPUT_TOKEN,
        thinking_token_budget=thinking_budget,
        logprobs=5,  # Request 5 logprobs per token
    )

    print(
        f"[DEBUG] SamplingParams created with thinking_token_budget: "
        f"{sampling_params.thinking_token_budget}"
    )

    if reasoning_mode == TestMode.GUIDED_GENERATION:
        schema = {"type": "object"}
        structural_tag = convert_schema_to_structural_tags(schema=schema, engine=engine)
        sampling_params.structured_outputs = StructuredOutputsParams(
            structural_tag=structural_tag, backend="xgrammar"
        )
    if continue_thinking:
        sampling_params.continue_thinking = True
        example_input = add_reasoning_prompt(example_input, continue_thinking)
    else:
        example_input = add_reasoning_prompt(example_input)

    tokenized_prompt = tokenizer.encode(example_input)
    results_generator = engine.generate(
        sampling_params=sampling_params,
        request_id=str(id),
        prompt=TokensPrompt(prompt_token_ids=tokenized_prompt),
    )

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    assert final_output is not None, "No output generated"
    out = final_output.outputs[0]

    # Return both text and logprobs for validation
    text = (
        tokenizer.decode(
            out.token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
        )
        if getattr(out, "token_ids", None)
        else out.text
    )

    logprobs = getattr(out, "logprobs", None)

    return {
        "text": text,
        "logprobs": logprobs,
        "token_ids": out.token_ids,
        "tokenized_prompt": tokenized_prompt,
    }


async def validate_thinking_token_budget(
    model, tensor_parallel_size, reasoning_mode, args
):
    tasks = []
    all_prompt: dict[int, str] = {}
    # Get effective engine args with hardware profile args + test-specific overrides
    engine_args = get_async_engine_args_with_overrides(
        test_kwargs={
            "model": model,
            "dtype": "auto",
            "max_model_len": 32000,
            "tensor_parallel_size": tensor_parallel_size,
            "structured_outputs_config": {"backend": "xgrammar"},
            "speculative_config": make_speculative_config(args),
            "max_logprobs": 5,  # Enable logprobs collection
            "reasoning_config": _create_reasoning_config(),
            "async_scheduling": True,
        },
        engine_args_override=getattr(args, "engine_args", None),
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    # Debug: Check speculative config

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    model_arch = get_text_model_name(engine.model_config)
    if reasoning_mode == TestMode.REASONING:
        for request_id, thinking_budget in enumerate(thinking_budgets_reasoning):
            prompt = REASONING_PROMPT
            all_prompt[request_id] = prompt
            tasks.append(
                asyncio.create_task(
                    gen(
                        engine,
                        prompt,
                        request_id,
                        tokenizer,
                        model,
                        thinking_budget=thinking_budget,
                        reasoning_mode=reasoning_mode,
                    )
                )
            )
        batch_results = await asyncio.gather(*tasks)
        validate_outputs(
            batch_results,
            tokenizer,
            reasoning_mode,
            thinking_budgets_reasoning,
            model_arch,
        )

        # Test the continue thinking functionality
        # for the the retried requests
        continue_tasks = []
        for request_id, thinking_budget in enumerate(
            continue_thinking_budgets_reasoning
        ):
            prompt = REASONING_PROMPT
            all_prompt[request_id] = prompt
            continue_tasks.append(
                asyncio.create_task(
                    gen(
                        engine,
                        prompt,
                        request_id,
                        tokenizer,
                        model,
                        thinking_budget=thinking_budget,
                        reasoning_mode=reasoning_mode,
                        continue_thinking=True,
                    )
                )
            )
        batch_results = await asyncio.gather(*continue_tasks)
        validate_outputs(
            batch_results,
            tokenizer,
            reasoning_mode,
            continue_thinking_budgets_reasoning,
            model_arch,
            continue_thinking=True,
        )

    elif reasoning_mode == TestMode.GUIDED_GENERATION:
        for prompt_id, prompt in enumerate(synth_prompts):
            thinking_budget = 500
            all_prompt[prompt_id] = prompt["prompt"]
            tasks.append(
                asyncio.create_task(
                    gen(
                        engine,
                        all_prompt[prompt_id],
                        prompt_id,
                        tokenizer,
                        model,
                        thinking_budget=thinking_budget,
                        reasoning_mode=reasoning_mode,
                    )
                )
            )
        batch_results = await asyncio.gather(*tasks)
        validate_outputs(
            batch_results, tokenizer, reasoning_mode, model_arch=model_arch
        )

    engine.shutdown()
    print("Async vLLM inference time: ", time.time() - st)


async def test_thinking_token_budget(
    model_name, tensor_parallel_size, reasoning_mode, args
):
    await validate_thinking_token_budget(
        model_name, tensor_parallel_size, reasoning_mode, args
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Thinking Budget CI Test")
    parser.add_argument("--model", type=str)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument(
        "--reasoning_mode",
        type=str,
        choices=[m.name.lower() for m in TestMode],
        required=True,
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

    reasoning_mode_enum = TestMode[args.reasoning_mode.upper()]

    models = [args.model]

    for model in models:
        print(f"Running {reasoning_mode_enum} test for model: {model}")
        asyncio.run(
            test_thinking_token_budget(
                model, args.tensor_parallel_size, reasoning_mode_enum, args
            )
        )
