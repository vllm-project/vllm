# ruff: noqa: E501
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Guided generation tests: JSON/schema validation, structural tool tags, and long
context. Default ``--suite merged`` builds **one** AsyncLLM engine and runs JSON,
tool, and long-context suites against it in order (default ``--max-concurrent`` 16).

Use ``--suite json`` / ``tool`` / ``long-context`` for standalone single-suite CLIs.
Use ``--suite sweep`` for a concurrency load test that fires all requests simultaneously
(no semaphore throttling) at configurable concurrency levels.
"""

import argparse
import ast
import asyncio
import copy
import itertools
import json
import sys

from jsonschema import validate
from test_const import (
    C2_tool_prompt_1,
    C3_tool_prompt_1,
    C3_tool_prompt_2,
    tool_schema_1,
    tool_schema_2,
)
from test_schema_data import synth_prompts
from test_utils import (
    RunMode,
    _create_reasoning_config,
    find_text_between,
    generate_guided_output,
    get_input_text,
    get_tool_schema,
    make_speculative_config,
    validate_output,
)
from transformers import AutoTokenizer

from vllm import SamplingParams, TokensPrompt
from vllm.cohere.guided_decoding.convert_to_structural_tag_format import (  # noqa: E501
    convert_schema_to_structural_tags,
)
from vllm.cohere.utils import get_text_model_name
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.engine.async_llm import AsyncLLM

THINKING_TOKEN_BUDGET = 50

# ---------------------------------------------------------------------------
# JSON / JSON-schema
# ---------------------------------------------------------------------------


async def run_json_validation_with_engine(
    engine,
    tokenizer,
    args,
    model_arch,
    *,
    max_concurrent: int = 16,
    max_failures: int = 5,
):
    """JSON object + JSON schema checks using an existing engine (bounded concurrency)."""
    sem = asyncio.Semaphore(max_concurrent)

    async def _json_object(i: int, p: dict) -> str | None:
        async with sem:
            return await generate_guided_output(
                engine,
                p["prompt"],
                f"json-{i}",
                None,
                tokenizer,
                thinking_token_budget=THINKING_TOKEN_BUDGET,
            )

    json_results = await asyncio.gather(
        *[_json_object(i, p) for i, p in enumerate(synth_prompts)]
    )
    bad_json, _ = validate_output(json_results, {}, model_arch)
    assert len(bad_json) <= max_failures, (
        f"Too many invalid JSON objects: {len(bad_json)}"
    )
    print(f"✅ JSON object validation passed ({args.mode.value})")

    schema_map = {i: p["schema"] for i, p in enumerate(synth_prompts)}

    async def _json_schema(i: int, p: dict) -> str | None:
        async with sem:
            return await generate_guided_output(
                engine,
                p["prompt"],
                f"schema-{i}",
                p["schema"],
                tokenizer,
                thinking_token_budget=THINKING_TOKEN_BUDGET,
            )

    schema_results = await asyncio.gather(
        *[_json_schema(i, p) for i, p in enumerate(synth_prompts)]
    )
    bad_json, bad_schema = validate_output(schema_results, schema_map, model_arch)
    assert len(bad_json) <= max_failures, f"Too many invalid JSON: {len(bad_json)}"
    assert len(bad_schema) <= max_failures, (
        f"Too many invalid schema JSON: {len(bad_schema)}"
    )
    print(f"✅ JSON schema validation passed ({args.mode.value})")


async def run_json_validation_tests(
    args, *, max_concurrent: int = 16, max_failures: int = 5
):
    engine_args = AsyncEngineArgs(
        model=args.model,
        dtype="auto",
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        structured_outputs_config={"backend": "xgrammar"},
        speculative_config=make_speculative_config(args),
        reasoning_config=_create_reasoning_config(),
        async_scheduling=True,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model_arch = get_text_model_name(engine.model_config)
    try:
        await run_json_validation_with_engine(
            engine,
            tokenizer,
            args,
            model_arch,
            max_concurrent=max_concurrent,
            max_failures=max_failures,
        )
    finally:
        engine.shutdown()


# ---------------------------------------------------------------------------
# Dynamic / tool grammar
# ---------------------------------------------------------------------------


def check_if_tool_output_is_valid(batch_results, model_architecture, tool_schemas):
    """Check each tool output against its schema."""
    invalid = []
    schema_list = [get_tool_schema(model_architecture, s) for s in tool_schemas]

    for request_id, output in enumerate(batch_results):
        if model_architecture in (
            "Cohere2ForCausalLM",
            "Cohere2VisionForConditionalGeneration",
            "Cohere2MoeForCausalLM",
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
                print("❌ JSON is invalid", tool)
                invalid.append(request_id)

    return invalid


async def generate_tool_output(engine, prompt, request_id, response_schema, tokenizer):
    """Run model for one prompt and decode result safely."""
    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.75,
        top_k=-1,
        max_tokens=1000,
        thinking_token_budget=THINKING_TOKEN_BUDGET,
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


async def run_tool_validation_with_engine(
    engine,
    tokenizer,
    args,
    model_arch,
    *,
    max_concurrent: int = 16,
):
    """Tool grammar checks using an existing engine (bounded concurrency)."""
    if model_arch == "CohereForCausalLM":
        prompts, schemas = [C2_tool_prompt_1], [tool_schema_2]
    else:
        prompts, schemas = (
            [C3_tool_prompt_1, C3_tool_prompt_2],
            [tool_schema_1, tool_schema_2],
        )

    sem = asyncio.Semaphore(max_concurrent)

    async def _one(idx: int, prompt: str, response_schema):
        async with sem:
            return await generate_tool_output(
                engine, prompt, str(idx), response_schema, tokenizer
            )

    batch_results = await asyncio.gather(
        *(_one(i, p, s) for i, (p, s) in enumerate(zip(prompts, schemas)))
    )
    invalid = check_if_tool_output_is_valid(batch_results, model_arch, schemas)
    assert not invalid, f"Tool validation produced invalid JSON ({invalid})"


async def check_tools_are_valid(
    args, speculative: bool = False, *, max_concurrent: int = 16
):
    """Check all prompts + tool schemas for given model."""
    spec_config = make_speculative_config(args) if speculative else None

    engine_args = AsyncEngineArgs(
        model=args.model,
        dtype="auto",
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        structured_outputs_config={"backend": "xgrammar"},
        speculative_config=spec_config,
        reasoning_config=_create_reasoning_config(),
        async_scheduling=True,
    )

    engine = AsyncLLM.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model_arch = get_text_model_name(engine.model_config)
    try:
        await run_tool_validation_with_engine(
            engine,
            tokenizer,
            args,
            model_arch,
            max_concurrent=max_concurrent,
        )
    finally:
        engine.shutdown()


async def run_tool_validation_tests(args, *, max_concurrent: int = 16):
    """Run tool validation based on chosen mode."""
    if args.mode in (RunMode.NON_SPECULATIVE, RunMode.BOTH):
        await check_tools_are_valid(
            args, speculative=False, max_concurrent=max_concurrent
        )
        print("✅ Non-speculative tool validation passed")

    if args.mode in (RunMode.SPECULATIVE, RunMode.BOTH):
        await check_tools_are_valid(
            args, speculative=True, max_concurrent=max_concurrent
        )
        print("✅ Speculative tool validation passed")

    print("------------- Tool validation finished -------------")


# ---------------------------------------------------------------------------
# Long context
# ---------------------------------------------------------------------------


async def run_long_context_with_engine(
    engine,
    tokenizer,
    args,
    model_arch,
    *,
    max_concurrent: int = 16,
):
    """Long-context guided JSON checks using an existing engine (bounded concurrency)."""
    long_context_texts, schema = get_input_text()
    schema_list = [schema] * len(long_context_texts)
    sem = asyncio.Semaphore(max_concurrent)

    async def _one(i: int, text: str):
        async with sem:
            return await generate_guided_output(
                engine,
                text,
                str(i),
                schema,
                tokenizer,
                thinking_token_budget=THINKING_TOKEN_BUDGET,
            )

    results = await asyncio.gather(
        *(_one(i, t) for i, t in enumerate(long_context_texts))
    )
    invalid_json, invalid_schema_json = validate_output(
        results, schema_list, model_arch
    )

    assert not invalid_json, "Invalid JSON outputs in long context test"
    assert not invalid_schema_json, "Invalid JSON schemas in long context test"
    print(f"✅ Long context validation passed ({args.mode.value})")


async def check_long_context_validation(args, *, max_concurrent: int = 16):
    """Run long context validation for one model."""
    spec_config = (
        make_speculative_config(args) if args.mode != RunMode.NON_SPECULATIVE else None
    )

    engine_args = AsyncEngineArgs(
        model=args.model,
        dtype="auto",
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        structured_outputs_config={"backend": "xgrammar"},
        speculative_config=spec_config,
        reasoning_config=_create_reasoning_config(),
        async_scheduling=True,
    )

    engine = AsyncLLM.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model_arch = get_text_model_name(engine.model_config)
    try:
        await run_long_context_with_engine(
            engine,
            tokenizer,
            args,
            model_arch,
            max_concurrent=max_concurrent,
        )
    finally:
        engine.shutdown()


async def run_long_context_tests(args, *, max_concurrent: int = 16):
    if args.mode in (RunMode.NON_SPECULATIVE, RunMode.BOTH):
        phase_args = copy.copy(args)
        phase_args.mode = RunMode.NON_SPECULATIVE
        await check_long_context_validation(phase_args, max_concurrent=max_concurrent)

    if args.mode in (RunMode.SPECULATIVE, RunMode.BOTH):
        phase_args = copy.copy(args)
        phase_args.mode = RunMode.SPECULATIVE
        await check_long_context_validation(phase_args, max_concurrent=max_concurrent)

    print("------------- Long context test finished -------------")


# ---------------------------------------------------------------------------
# Concurrency sweep (no semaphore — true simultaneous load)
# ---------------------------------------------------------------------------

_TAG_PAIRS = [
    ("<|START_RESPONSE|>", "<|END_RESPONSE|>"),
    ("<|START_TEXT|>", "<|END_TEXT|>"),
]


def _extract_json_text(raw_output: str | None) -> str | None:
    """Extract JSON text from model output by trying known tag pairs."""
    if raw_output is None:
        return None
    for prefix, postfix in _TAG_PAIRS:
        if prefix in raw_output and postfix in raw_output:
            start = raw_output.find(prefix) + len(prefix)
            end = raw_output.find(postfix, start)
            return raw_output[start:end].strip()
        elif prefix in raw_output:
            return raw_output[raw_output.find(prefix) + len(prefix) :].strip()
    return raw_output.strip()


def _validate_sweep_output(
    engine_output: list[str | None], schema_map: dict[int, dict]
) -> tuple[list[int], list[int]]:
    """Validate outputs: parse JSON and check schema conformance."""
    invalid_json: list[int] = []
    invalid_schema: list[int] = []
    for idx, raw in enumerate(engine_output):
        text = _extract_json_text(raw)
        if not text:
            invalid_json.append(idx)
            continue
        try:
            json_obj = json.loads(text)
        except Exception:
            invalid_json.append(idx)
            continue
        if idx in schema_map:
            try:
                validate(json_obj, schema_map[idx])
            except Exception:
                invalid_schema.append(idx)
    return invalid_json, invalid_schema


def _select_prompts(concurrency: int) -> list[dict]:
    """Cycle synth_prompts to fill the requested concurrency level."""
    return list(itertools.islice(itertools.cycle(synth_prompts), concurrency))


async def _run_json_sweep_for_engine(
    engine: AsyncLLM,
    tokenizer,
    concurrency_levels: list[int],
    max_failures: int,
    mode_label: str,
) -> None:
    """Run the JSON schema concurrency sweep (no semaphore)."""
    for concurrency in concurrency_levels:
        print(f"\n--- JSON concurrency {concurrency} ({mode_label}) ---")
        prompts = _select_prompts(concurrency)
        schema_map = {i: p["schema"] for i, p in enumerate(prompts)}

        results = await asyncio.gather(
            *[
                generate_guided_output(
                    engine,
                    p["prompt"],
                    f"json-{mode_label}-c{concurrency}-{i}",
                    p["schema"],
                    tokenizer,
                )
                for i, p in enumerate(prompts)
            ]
        )

        bad_json, bad_schema = _validate_sweep_output(results, schema_map)
        total_failures = len(bad_json) + len(bad_schema)
        print(
            f"  Results: {concurrency - total_failures}/{concurrency} valid "
            f"(bad_json={len(bad_json)}, bad_schema={len(bad_schema)})"
        )
        assert len(bad_json) <= max_failures, (
            f"JSON concurrency {concurrency} ({mode_label}): "
            f"too many invalid JSON objects: {bad_json}"
        )
        assert len(bad_schema) <= max_failures, (
            f"JSON concurrency {concurrency} ({mode_label}): "
            f"too many schema violations: {bad_schema}"
        )
        print(f"  PASSED json concurrency={concurrency} ({mode_label})")


async def run_sweep(args) -> None:
    """Run JSON concurrency sweep, respecting --mode for which passes to run."""
    concurrency_levels = args.max_concurrent
    max_failures = args.max_failures
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # --- Non-speculative sweep ---
    if args.mode in (RunMode.NON_SPECULATIVE, RunMode.BOTH):
        print("\n=== GG JSON Concurrency Sweep: Non-Speculative ===")
        engine_args = AsyncEngineArgs(
            model=args.model,
            dtype="auto",
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            structured_outputs_config={"backend": "xgrammar"},
            speculative_config=None,
            reasoning_config=_create_reasoning_config(),
            async_scheduling=True,
        )
        engine = AsyncLLM.from_engine_args(engine_args)
        try:
            await _run_json_sweep_for_engine(
                engine, tokenizer, concurrency_levels, max_failures, "non-spec"
            )
        finally:
            engine.shutdown()
        print("\n Non-speculative sweep passed")

    # --- Speculative sweep ---
    if args.mode in (RunMode.SPECULATIVE, RunMode.BOTH):
        if not args.draft_model:
            raise ValueError("--draft_model is required for speculative sweep")
        spec_config = {
            "method": args.method,
            "model": args.draft_model,
            "num_speculative_tokens": args.num_spec_tokens,
            "draft_tensor_parallel_size": args.draft_tp,
            "max_model_len": args.max_model_len,
        }
        print("\n=== GG JSON Concurrency Sweep: Speculative ===")
        engine_args = AsyncEngineArgs(
            model=args.model,
            dtype="auto",
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            structured_outputs_config={"backend": "xgrammar"},
            speculative_config=spec_config,
            reasoning_config=_create_reasoning_config(),
            async_scheduling=True,
        )
        engine = AsyncLLM.from_engine_args(engine_args)
        try:
            await _run_json_sweep_for_engine(
                engine, tokenizer, concurrency_levels, max_failures, "spec"
            )
        finally:
            engine.shutdown()
        print("\n Speculative sweep passed")

    print("\n=== All GG concurrency sweeps passed ===")


# ---------------------------------------------------------------------------
# Merged suite (single shared engine)
# ---------------------------------------------------------------------------


def _merged_single_engine_phase_args(args):
    """
    Choose engine kwargs for one AsyncLLM that runs JSON, tools, and long context.

    ``RunMode.BOTH`` cannot use two different speculative configs on one process
    engine; with ``--draft_model`` we enable speculative decoding, otherwise we fall
    back to a non-speculative engine and log why.
    """
    phase_args = copy.copy(args)
    if args.mode == RunMode.NON_SPECULATIVE:
        phase_args.mode = RunMode.NON_SPECULATIVE
        label = "non-speculative"
    elif args.mode == RunMode.SPECULATIVE:
        phase_args.mode = RunMode.SPECULATIVE
        label = "speculative"
    else:
        if args.draft_model:
            phase_args.mode = RunMode.BOTH
            label = "speculative decoding (mode=both)"
        else:
            phase_args.mode = RunMode.NON_SPECULATIVE
            label = (
                "non-speculative (mode=both but no --draft_model; "
                "add draft to exercise speculative on this single engine)"
            )
            print(
                "Note: merged suite uses one engine; BOTH without --draft_model "
                "runs as non-speculative."
            )
    return phase_args, label


async def run_merged_guided_generation_tests(
    args, *, max_concurrent: int = 16, max_failures: int = 5
):
    """Run JSON, tool, and long-context suites on **one** shared engine, in order."""
    phase_args, phase_label = _merged_single_engine_phase_args(args)

    engine_args = AsyncEngineArgs(
        model=args.model,
        dtype="auto",
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        structured_outputs_config={"backend": "xgrammar"},
        speculative_config=make_speculative_config(phase_args),
        reasoning_config=_create_reasoning_config(),
        async_scheduling=True,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model_arch = get_text_model_name(engine.model_config)
    print(f"\n=== Guided generation merged suite ({phase_label}) ===\n")
    try:
        print("1/3 JSON validation...")
        await run_json_validation_with_engine(
            engine,
            tokenizer,
            phase_args,
            model_arch,
            max_concurrent=max_concurrent,
            max_failures=max_failures,
        )
        print("2/3 Tool validation...")
        await run_tool_validation_with_engine(
            engine,
            tokenizer,
            phase_args,
            model_arch,
            max_concurrent=max_concurrent,
        )
        print("3/3 Long context validation...")
        await run_long_context_with_engine(
            engine,
            tokenizer,
            phase_args,
            model_arch,
            max_concurrent=max_concurrent,
        )
        print(f"✅ Merged guided generation passed ({phase_label})\n")
    finally:
        engine.shutdown()

    print("------------- All merged guided generation tests finished -------------")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Cohere guided generation tests")
    parser.add_argument(
        "--suite",
        choices=["merged", "json", "tool", "long-context", "sweep"],
        default="merged",
        help="merged (default): one engine runs JSON+tool+long in order; sweep: concurrency load test",
    )
    parser.add_argument("--model", type=str, default="CohereForAI/c4ai-command-r-v01")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument(
        "--mode",
        type=RunMode,
        choices=list(RunMode),
        default=None,
        help="If omitted: non-spec for --suite json, speculative for sweep with --draft_model, else both",
    )
    parser.add_argument("--method", type=str, default="eagle")
    parser.add_argument(
        "--draft_model",
        type=str,
        default=None,
        help="Draft model for speculative decoding",
    )
    parser.add_argument("--num_spec_tokens", type=int, default=4)
    parser.add_argument(
        "--draft_tp",
        type=int,
        default=1,
        help="Tensor parallel size for speculative model",
    )
    parser.add_argument("--max_model_len", type=int, default=32_000)
    parser.add_argument(
        "--max-concurrent",
        type=int,
        nargs="+",
        default=[16],
        help=(
            "For sweep: concurrency levels to test (e.g. 32 64 128). "
            "For other suites: single value used as semaphore cap (default: 16)."
        ),
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=5,
        help="Max failures allowed per concurrency level (only used with --suite sweep)",
    )
    args = parser.parse_args()
    if args.mode is None:
        if args.suite == "json":
            args.mode = RunMode.NON_SPECULATIVE
        elif args.suite == "sweep" and args.draft_model:
            args.mode = RunMode.BOTH
        elif args.suite == "sweep":
            args.mode = RunMode.NON_SPECULATIVE
        else:
            args.mode = RunMode.BOTH
    # For non-sweep suites, max_concurrent is a single-value semaphore cap
    if args.suite != "sweep" and len(args.max_concurrent) > 1:
        args.max_concurrent = args.max_concurrent[:1]
    return args


async def _dispatch(args):
    if args.suite == "sweep":
        await run_sweep(args)
        return
    mc = args.max_concurrent[0]
    mf = args.max_failures
    if args.suite == "merged":
        await run_merged_guided_generation_tests(
            args, max_concurrent=mc, max_failures=mf
        )
    elif args.suite == "json":
        await run_json_validation_tests(args, max_concurrent=mc, max_failures=mf)
    elif args.suite == "tool":
        await run_tool_validation_tests(args, max_concurrent=mc)
    elif args.suite == "long-context":
        await run_long_context_tests(args, max_concurrent=mc)


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(_dispatch(args))
    except Exception as e:
        print(f"\n❌ Guided generation test failed: {e}")
        sys.exit(1)
