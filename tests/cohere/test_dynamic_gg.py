# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import ast
import asyncio
import json

from jsonschema import validate
from test_const import (
    C2_tool_prompt_1,
    C3_tool_prompt_1,
    C3_tool_prompt_2,
    tool_schema_1,
    tool_schema_2,
)
from test_utils import (
    RunMode,
    _create_reasoning_config,
    find_text_between,
    get_tool_schema,
    make_speculative_config,
)
from test_utils_engine_args import get_async_engine_args_with_overrides
from transformers import AutoTokenizer

from vllm import SamplingParams, TokensPrompt
from vllm.cohere.guided_decoding.convert_to_structural_tag_format import (  # noqa: E501
    convert_schema_to_structural_tags,
)
from vllm.cohere.guided_decoding.tool_grammar import get_text_model_name
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.engine.async_llm import AsyncLLM


def check_if_tool_output_is_valid(batch_results, model_architecture, tool_schemas):
    """Check each tool output against its schema."""
    invalid = []
    schema_list = [get_tool_schema(model_architecture, s) for s in tool_schemas]

    for request_id, output in enumerate(batch_results):
        if (
            model_architecture == "Cohere2ForCausalLM"
            or model_architecture == "Cohere2MoeForCausalLM"
        ):
            output = find_text_between(output, "<|START_ACTION|>", "<|END_ACTION|>")
        else:
            output = find_text_between(output, "Action: ```json", "```")

        try:
            output_list = ast.literal_eval(output)
        except Exception:
            invalid.append(request_id)
            continue

        for tool in output_list:
            matching_strings = None
            for s in schema_list:
                for sch in s:
                    if tool.get("tool_name") in sch:
                        matching_strings = sch
            try:
                if matching_strings:
                    validate(tool, json.loads(matching_strings))
                    print("✅ JSON is valid")
            except Exception:
                invalid.append(request_id)

    return invalid


async def generate_tool_output(engine, prompt, request_id, response_schema, tokenizer):
    """Run model for one prompt and decode result safely."""
    sampling_params = SamplingParams(
        temperature=0.3, top_p=0.75, top_k=-1, max_tokens=1000
    )
    structural_tag = convert_schema_to_structural_tags(
        tools=json.dumps(response_schema), engine=engine
    )
    print("Structural Tag:", structural_tag)
    sampling_params.structured_outputs = StructuredOutputsParams(
        structural_tag=structural_tag
    )

    tokens = tokenizer.encode(prompt)
    final_output = None
    async for output in engine.generate(
        sampling_params=sampling_params,
        request_id=request_id,
        prompt=TokensPrompt(prompt_token_ids=tokens),
    ):
        final_output = output

    if not final_output or not final_output.outputs:
        return None

    output = final_output.outputs[0]
    return (
        tokenizer.decode(
            output.token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        if getattr(output, "token_ids", None)
        else output.text
    )


async def check_tools_are_valid(args, speculative: bool = False):
    """Check all prompts + tool schemas for given model."""
    spec_config = make_speculative_config(args) if speculative else None

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
    model_arch = get_text_model_name(engine.model_config)
    # Select prompts and schemas based on model
    if model_arch == "CohereForCausalLM":
        prompts, schemas = [C2_tool_prompt_1], [tool_schema_2]
    else:
        prompts, schemas = (
            [C3_tool_prompt_1, C3_tool_prompt_2],
            [tool_schema_1, tool_schema_2],
        )

    tasks: list[asyncio.Task] = []
    for idx, prompt in enumerate(prompts):
        tasks.append(
            asyncio.create_task(
                generate_tool_output(engine, prompt, str(idx), schemas[idx], tokenizer)
            )
        )

    batch_results = await asyncio.gather(*tasks)
    invalid = check_if_tool_output_is_valid(batch_results, model_arch, schemas)

    assert not invalid, f"Tool validation produced invalid JSON ({invalid})"
    engine.shutdown()


async def run_tool_validation_tests(args):
    """Run tool validation based on chosen mode."""
    if args.mode in (RunMode.NON_SPECULATIVE, RunMode.BOTH):
        await check_tools_are_valid(args, speculative=False)
        print("✅ Non-speculative tool validation passed")

    if args.mode in (RunMode.SPECULATIVE, RunMode.BOTH):
        await check_tools_are_valid(args, speculative=True)
        print("✅ Speculative tool validation passed")

    print("------------- Tool validation finished -------------")


# ----------------- CLI ----------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Tool grammar validation")
    p.add_argument("--model", type=str, default="CohereLabs/c4ai-command-r7b-12-2024")
    p.add_argument("--tensor_parallel_size", type=int, default=2)
    p.add_argument("--mode", type=RunMode, choices=list(RunMode), default=RunMode.BOTH)

    # Speculative decoding parameters
    p.add_argument("--method", type=str, default="eagle")
    p.add_argument(
        "--draft_model",
        type=str,
        default=None,
        help="Draft model used in speculative decoding",
    )
    p.add_argument(
        "--num_spec_tokens",
        type=int,
        default=4,
        help="Number of tokens drafted in speculative mode",
    )
    p.add_argument(
        "--draft_tp",
        type=int,
        default=1,
        help="Tensor parallel size for speculative model",
    )
    p.add_argument("--max_model_len", type=int, default=32_000)
    return p.parse_args()


def main():
    args = parse_args()
    asyncio.run(run_tool_validation_tests(args))


if __name__ == "__main__":
    main()
