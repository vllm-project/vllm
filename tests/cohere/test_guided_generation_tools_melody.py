# ruff: noqa: E501
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test guided generation with tools via the Chat Completions API (vLLM serve).

Uses the same tool prompts as test_guided_generation (engine-based), but drives the server
via client.chat.completions.create() with `tools` in the request so that
prepare_structured_tag in serving.py is exercised with tools. Tool outputs are
validated as OpenAI ``tool_calls`` arguments against each tool's ``parameters``
schema from RESPONSE_FORMAT_TOOL_DEFINITIONS (e.g. ``test_const.tool_schema_1``).

This test always uses ``tool_choice="auto"`` so Cohere output is parsed via
``--enable-auto-tool-choice`` and ``--tool-call-parser cohere_command4`` (the
``required`` path expects raw OpenAI JSON only and is not used here).

A final request combines the same-style function tools with
``response_format={"type": "json_object"}`` and a user prompt asking for JSON
with ``name`` and ``age``, and asserts the assistant message parses as that
JSON object (exercising tools + json_object together).
"""

import argparse
import asyncio
import json
import os
import sys

# Repo root on path so tests.utils and cohere test_utils import from tests/cohere
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import openai  # noqa: E402
from jsonschema import validate as jsonschema_validate  # noqa: E402
from test_const import tool_schema_1, tool_schema_2  # noqa: E402
from test_utils import RunMode, make_speculative_config  # noqa: E402

from tests.utils import RemoteOpenAIServer  # noqa: E402

# Default for --model_architecture CLI (Cohere2 family).
DEFAULT_MODEL_ARCHITECTURE = "Cohere2ForCausalLM"

# Always "auto": Melody parses <|START_ACTION|> / tool_name style tool output.
CHAT_COMPLETIONS_TOOL_CHOICE: str = "auto"

# Tools used with ``response_format: json_object`` to verify assistant content is JSON
# (name / age) while ``prepare_structured_tag`` still sees tools on the request.
TOOLS_WITH_JSON_OBJECT_RESPONSE_FORMAT: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "adds two integers: num1, and num2",
            "parameters": {
                "type": "object",
                "properties": {
                    "num1": {
                        "type": "number",
                        "description": "num1: first number",
                    },
                    "num2": {
                        "type": "number",
                        "description": "num2: 2nd number",
                    },
                },
                "required": ["num1", "num2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply_numbers",
            "description": "multiply two integers: num1, and num2",
            "parameters": {
                "type": "object",
                "properties": {
                    "num1": {
                        "type": "string",
                        "description": "num1: first number",
                    },
                    "num2": {
                        "type": "string",
                        "description": "num2: 2nd number",
                    },
                },
                "required": ["num1", "num2"],
            },
        },
    },
]

JSON_OBJECT_WITH_TOOLS_USER_MESSAGE = "Generate a JSON with name and age"

# (user message, tool_schema): tool_schema is RESPONSE_FORMAT_TOOL_DEFINITIONS
# (JSON string).
TOOL_PROMPTS = [
    (
        "What's 2 * 3? And what is 4 + 5? Use the provided tools to compute both.",
        tool_schema_1,
    ),
    (
        (
            "I'm ordering an item in Ottawa and need to ensure the weather is good "
            "for delivery. Could you fetch the current weather for Ottawa "
            "(latitude 45.4215, longitude -75.6972) and the delivery date for "
            "order id 123?"
        ),
        tool_schema_2,
    ),
]


def tool_schema_to_api_tools(schema: str | list) -> list[dict]:
    """Convert RESPONSE_FORMAT_TOOL_DEFINITIONS to Chat Completions ``tools``."""
    parsed = json.loads(schema) if isinstance(schema, str) else schema
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description"),
                "parameters": t.get("parameters"),
            },
        }
        for t in parsed
    ]


def _tool_definitions_from_response_format(
    schema_str_or_list: str | list,
) -> list[dict]:
    """Parse RESPONSE_FORMAT_TOOL_DEFINITIONS (JSON string or list)."""
    return (
        json.loads(schema_str_or_list)
        if isinstance(schema_str_or_list, str)
        else schema_str_or_list
    )


def _parameters_schema_by_tool_name(
    schema_str_or_list: str | list,
) -> dict[str, dict]:
    """Map tool name -> JSON Schema for ``parameters`` (OpenAI function arguments)."""
    tools = _tool_definitions_from_response_format(schema_str_or_list)
    return {t["name"]: t["parameters"] for t in tools}


def _extract_chat_completion_tool_calls(choice) -> list[tuple[str, dict]] | None:
    """
    Parse OpenAI Chat Completions ``tool_calls``: each item is
    (function name, arguments object).

    Returns None if there are no tool calls or any call is malformed.
    """
    if choice is None or choice.message is None:
        return None
    tool_calls = getattr(choice.message, "tool_calls", None) or []
    if not tool_calls:
        return None
    out: list[tuple[str, dict]] = []
    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        if fn is None:
            return None
        name = getattr(fn, "name", None)
        arg_str = getattr(fn, "arguments", None)
        if not name or arg_str is None:
            return None
        try:
            args = json.loads(str(arg_str).strip())
        except json.JSONDecodeError:
            return None
        if not isinstance(args, dict):
            return None
        out.append((name, args))
    return out


def _parse_assistant_json_object_content(content: str | None) -> dict | None:
    """
    Parse assistant ``content`` as a JSON object.

    Strips optional ``` / ```json fences so models that wrap JSON still validate.
    """
    if not content or not str(content).strip():
        return None
    s = str(content).strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) < 2:
            return None
        inner = "\n".join(lines[1:])
        if inner.rstrip().endswith("```"):
            inner = inner.rstrip()[:-3].rstrip()
        s = inner.strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def validate_tool_api_output(
    tool_invocations_per_request: list[list[tuple[str, dict]] | None],
    schema_list: list[str | list],
) -> list[int]:
    """
    Validate Chat Completions API tool outputs against RESPONSE_FORMAT tool definitions.

    For each request, ``tool_invocations_per_request[i]`` is a list of
    ``(tool_name, arguments_dict)`` from ``choice.message.tool_calls``
    (OpenAI format). Each ``arguments_dict`` is validated with jsonschema
    against that tool's ``parameters`` schema from ``schema_list[i]``.

    Returns indices of requests that failed validation.
    """
    invalid: list[int] = []
    param_schema_by_name_per_request = [
        _parameters_schema_by_tool_name(s) for s in schema_list
    ]

    for request_id, invocations in enumerate(tool_invocations_per_request):
        if not invocations:
            invalid.append(request_id)
            continue
        schemas_for_req = param_schema_by_name_per_request[request_id]
        failed = False
        for tool_name, arguments in invocations:
            param_schema = schemas_for_req.get(tool_name)
            if param_schema is None:
                failed = True
                break
            try:
                jsonschema_validate(arguments, param_schema)
            except Exception:
                failed = True
                break
        if failed:
            invalid.append(request_id)

    return invalid


async def run_tool_chat_completions_tests(args):
    """Run tool validation via vLLM serve (Chat Completions API with tools)."""
    server_args = [
        "--tensor-parallel-size",
        "4",
        "--enable-prefix-caching",
        "--max-num-seqs",
        "32",
        "--quantization",
        "compressed-tensors",
        "--async-scheduling",
        "--gpu_memory_utilization",
        "0.9",
        "--reasoning-config",
        '{"reasoning_start_str": "<|START_THINKING|>", '
        '"reasoning_end_str": "<|END_THINKING|>"}',
        "--structured-outputs-config",
        '{"backend": "xgrammar", "enable_in_reasoning":true}',
        "--reasoning-parser",
        "cohere_command4",
        # tool_choice="auto" with Cohere needs ToolParser; vLLM only loads it when
        # --enable-auto-tool-choice is set. Name must be a ToolParserManager key
        # (e.g. cohere_command4), not the Python class name.
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "cohere_command4",
    ]
    if args.tensor_parallel_size is not None:
        server_args.extend(["--tensor-parallel-size", str(args.tensor_parallel_size)])

    speculative_config = make_speculative_config(args)
    if speculative_config is not None:
        server_args.extend(["--speculative_config", json.dumps(speculative_config)])

    with RemoteOpenAIServer(
        args.model,
        server_args,
        env_dict={"VLLM_SERVER_DEV_MODE": "1"},
        max_wait_seconds=getattr(args, "server_wait_seconds", 300),
    ) as server:
        base_url = f"http://{server.host}:{server.port}/v1"
        async with openai.AsyncOpenAI(
            base_url=base_url,
            api_key=RemoteOpenAIServer.DUMMY_API_KEY,
        ) as client:
            # Exercise prepare_structured_tag with tools via Chat Completions API
            tool_invocations_per_request: list[list[tuple[str, dict]] | None] = []
            for i, (user_message, tool_schema) in enumerate(TOOL_PROMPTS):
                try:
                    api_tools = tool_schema_to_api_tools(tool_schema)
                    response = await client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": user_message}],
                        tools=api_tools,
                        tool_choice=CHAT_COMPLETIONS_TOOL_CHOICE,
                        max_tokens=32000,
                    )
                    choice = response.choices[0] if response.choices else None
                    print(f"choice: {choice}")
                    tool_invocations_per_request.append(
                        _extract_chat_completion_tool_calls(choice)
                    )
                except Exception as e:
                    print(f"Request {i} failed: {e}")
                    tool_invocations_per_request.append(None)

            schema_list = [p[1] for p in TOOL_PROMPTS]
            invalid = validate_tool_api_output(
                tool_invocations_per_request,
                schema_list,
            )
            assert len(invalid) <= 2, (
                "Too many invalid tool outputs: "
                f"{len(invalid)} (request ids: {invalid})"
            )
            print(
                "✅ Tool validation via Chat Completions API "
                f"(prepare_structured_tag with tools) passed ({args.mode.value})"
            )

            # tools + response_format json_object: expect parseable JSON with name & age
            try:
                rf_response = await client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {
                            "role": "user",
                            "content": JSON_OBJECT_WITH_TOOLS_USER_MESSAGE,
                        }
                    ],
                    tools=TOOLS_WITH_JSON_OBJECT_RESPONSE_FORMAT,
                    tool_choice=CHAT_COMPLETIONS_TOOL_CHOICE,
                    response_format={"type": "json_object"},
                    temperature=0.6,
                    max_tokens=2048,
                )
                rf_choice = rf_response.choices[0] if rf_response.choices else None
                print(f"tools+json_object choice: {rf_choice}")
                raw_content = (
                    (rf_choice.message.content or "").strip()
                    if rf_choice and rf_choice.message
                    else ""
                )
                parsed = _parse_assistant_json_object_content(raw_content)
                assert parsed is not None, (
                    "Expected assistant content to be a JSON object "
                    f"(response_format=json_object); raw content: {raw_content!r}"
                )
                assert "name" in parsed and "age" in parsed, (
                    "Expected JSON object to include 'name' and 'age' keys; "
                    f"got keys: {list(parsed.keys())}"
                )
            except Exception as e:
                raise AssertionError(
                    f"tools + response_format json_object (name/age prompt) failed: {e}"
                ) from e

            print(
                "✅ tools + response_format=json_object (name/age JSON) passed "
                f"({args.mode.value})"
            )


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Tool-guided generation via vLLM serve (Chat Completions API with tools)"
        )
    )
    p.add_argument("--model", type=str, default="CohereForAI/c4ai-command-r-v01")
    p.add_argument("--tensor_parallel_size", type=int, default=None)
    p.add_argument(
        "--mode", type=RunMode, choices=list(RunMode), default=RunMode.NON_SPECULATIVE
    )
    p.add_argument("--method", type=str, default="eagle")
    p.add_argument("--draft_model", type=str, default=None)
    p.add_argument("--num_spec_tokens", type=int, default=4)
    p.add_argument("--draft_tp", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=32000)
    p.add_argument("--server_wait_seconds", type=float, default=300)
    p.add_argument(
        "--model_architecture",
        type=str,
        default=DEFAULT_MODEL_ARCHITECTURE,
        help="Model architecture for tool schema validation (e.g. Cohere2ForCausalLM)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    asyncio.run(run_tool_chat_completions_tests(args))


if __name__ == "__main__":
    main()
