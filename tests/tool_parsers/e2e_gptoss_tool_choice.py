#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E test for GPT-OSS tool_choice=required.

Usage:
    vllm serve <gpt-oss-model> --tool-parser-plugin openai --enable-auto-tool-choice
    python tests/tool_parsers/e2e_gptoss_tool_choice.py [-v] [--scenario NAME]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger("e2e_tool_choice")

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOL_GET_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}

TOOL_SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

TOOL_CALCULATE = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a math expression",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}

TOOL_DATABASE_QUERY = {
    "type": "function",
    "function": {
        "name": "database_query",
        "description": "Execute a database query",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string"},
                "database": {"type": "string"},
            },
            "required": ["sql", "database"],
        },
    },
}

TOOL_GET_TIME = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current time",
        "parameters": {"type": "object", "properties": {}},
    },
}

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@dataclass
class TestScenario:
    name: str
    messages: list[dict]
    tools: list[dict]
    expected_tool_names: list[str] | None = None
    min_tool_calls: int = 1


SCENARIOS: list[TestScenario] = [
    TestScenario(
        name="simple_weather",
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        tools=[TOOL_GET_WEATHER],
        expected_tool_names=["get_weather"],
    ),
    TestScenario(
        name="select_from_multiple",
        messages=[{"role": "user", "content": "What is the weather in Seoul?"}],
        tools=[TOOL_GET_WEATHER, TOOL_SEARCH, TOOL_CALCULATE],
        expected_tool_names=["get_weather"],
    ),
    TestScenario(
        name="nested_json_args",
        messages=[
            {
                "role": "user",
                "content": "Query the users database: "
                "SELECT * FROM users WHERE age > 18 AND status = 'active'",
            }
        ],
        tools=[TOOL_DATABASE_QUERY],
        expected_tool_names=["database_query"],
    ),
    TestScenario(
        name="multi_turn_with_tool_result",
        messages=[
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_001",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "content": '{"temperature": 18, "condition": "cloudy"}',
            },
            {"role": "user", "content": "Now search for indoor activities."},
        ],
        tools=[TOOL_GET_WEATHER, TOOL_SEARCH],
        expected_tool_names=["search"],
    ),
    TestScenario(
        name="special_chars",
        messages=[
            {
                "role": "user",
                "content": "My code has if x < 10 && y >= 20. Search for help.",
            }
        ],
        tools=[TOOL_SEARCH],
        expected_tool_names=["search"],
    ),
    TestScenario(
        name="korean_unicode",
        messages=[
            {
                "role": "user",
                "content": "서울의 현재 날씨를 알려주세요.",
            }
        ],
        tools=[TOOL_GET_WEATHER],
        expected_tool_names=["get_weather"],
    ),
    TestScenario(
        name="no_arg_tool",
        messages=[{"role": "user", "content": "What time is it right now?"}],
        tools=[TOOL_GET_TIME],
        expected_tool_names=["get_current_time"],
    ),
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    scenario: str
    passed: bool
    tool_calls: list[dict] = field(default_factory=list)
    error: str | None = None
    latency_ms: float = 0.0


def _validate_tool_args(tool_call: dict, tool_defs: list[dict]) -> str | None:
    name = tool_call["name"]
    try:
        args = json.loads(tool_call["arguments"])
    except json.JSONDecodeError:
        return f"{name}: invalid JSON"
    tool_def = next((t for t in tool_defs if t["function"]["name"] == name), None)
    if tool_def is None:
        return f"{name}: not in tool definitions"
    for req in tool_def["function"].get("parameters", {}).get("required", []):
        if req not in args:
            return f"{name}: missing required field '{req}'"
    return None


def _detect_model(client: OpenAI) -> str:
    models = [m.id for m in client.models.list().data]
    if len(models) == 1:
        return models[0]
    logger.error("Expected 1 model, got %s", models)
    sys.exit(1)


def run_scenario(
    client: OpenAI,
    model: str,
    scenario: TestScenario,
    verbose: bool,
) -> TestResult:
    logger.info("--- [%s] ---", scenario.name)
    if verbose:
        logger.debug(
            "Request:\n%s", json.dumps(scenario.messages, indent=2, ensure_ascii=False)
        )

    t0 = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=scenario.messages,
            tools=scenario.tools,
            tool_choice="required",
            temperature=0,
            max_tokens=4096,
        )
    except Exception as e:
        return TestResult(scenario=scenario.name, passed=False, error=str(e))
    latency = (time.monotonic() - t0) * 1000

    choice = response.choices[0]
    msg = choice.message
    if verbose:
        logger.debug(
            "Response:\n%s",
            json.dumps(response.model_dump(), indent=2, ensure_ascii=False),
        )

    tc_data = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            tc_data.append(
                {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            )
            logger.info("  Tool: %s(%s)", tc.function.name, tc.function.arguments)

    passed = True
    errors: list[str] = []

    if len(tc_data) < scenario.min_tool_calls:
        passed = False
        errors.append(
            f"Expected >= {scenario.min_tool_calls} calls, got {len(tc_data)}"
        )

    if scenario.expected_tool_names and tc_data:
        actual = {tc["name"] for tc in tc_data}
        if not actual & set(scenario.expected_tool_names):
            passed = False
            errors.append(f"Expected {scenario.expected_tool_names}, got {actual}")

    if choice.finish_reason in ("error", "length"):
        passed = False
        errors.append(f"finish_reason={choice.finish_reason}")

    for tc in tc_data:
        err = _validate_tool_args(tc, scenario.tools)
        if err:
            passed = False
            errors.append(err)

    logger.info("  %s (%.0fms)", "PASS" if passed else "FAIL", latency)
    if not passed:
        logger.error("  Error: %s", "; ".join(errors))

    return TestResult(
        scenario=scenario.name,
        passed=passed,
        tool_calls=tc_data,
        error="; ".join(errors) if errors else None,
        latency_ms=latency,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="E2E test: GPT-OSS tool_choice=required")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--scenario", default=None)
    args = ap.parse_args()

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(
                Path(args.log_dir) / f"e2e_{time.strftime('%Y%m%d_%H%M%S')}.log"
            )
        )
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=handlers,
    )

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    model = args.model or _detect_model(client)
    logger.info("Model: %s", model)

    scenarios = SCENARIOS
    if args.scenario:
        scenarios = [s for s in SCENARIOS if s.name == args.scenario]
        if not scenarios:
            logger.error("Scenario '%s' not found", args.scenario)
            sys.exit(1)

    results = [run_scenario(client, model, s, args.verbose) for s in scenarios]
    passed = sum(1 for r in results if r.passed)
    logger.info(
        "\n%s\nSUMMARY: %d/%d passed\n%s", "=" * 40, passed, len(results), "=" * 40
    )
    for r in results:
        logger.info(
            "  %-30s %s", r.scenario, "PASS" if r.passed else f"FAIL: {r.error}"
        )
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
