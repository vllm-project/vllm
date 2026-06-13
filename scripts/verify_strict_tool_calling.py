#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Verify strict tool calling and structured output enforcement against a live
vLLM server.

Start the server with strict tool calling enabled (the default):

    VLLM_ENFORCE_STRICT_TOOL_CALLING=true vllm serve <model> \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3

Then run this script:

    python scripts/verify_strict_tool_calling.py [--base-url URL] [-v] [-k FILTER]

NOTE: The tool-call-parser name for Qwen3 models is "qwen3_coder" (or
"qwen3_xml"), NOT "qwen3". The name "qwen3" is only valid for
--reasoning-parser.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

from openai import OpenAI

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to find the weather for",
                },
                "state": {
                    "type": "string",
                    "description":
                        "The two-letter abbreviation for the state "
                        "that the city is in, e.g. 'CA' for California",
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["city", "state", "unit"],
        },
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the internet and get a summary of results",
        "parameters": {
            "type": "object",
            "properties": {
                "search_term": {
                    "type": "string",
                    "description": "The search query keywords",
                },
            },
            "required": ["search_term"],
        },
    },
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform a mathematical operation on two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "The math operation to perform",
                    "enum": ["add", "subtract", "multiply", "divide"],
                },
                "a": {"type": "number", "description": "First operand"},
                "b": {"type": "number", "description": "Second operand"},
            },
            "required": ["operation", "a", "b"],
        },
    },
}

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

SEED = 42


@dataclass
class TestResult:
    name: str
    category: str
    passed: bool
    details: str = ""
    duration: float = 0.0


@dataclass
class TestRunner:
    client: OpenAI
    model: str
    verbose: bool = False
    name_filter: str | None = None
    results: list[TestResult] = field(default_factory=list)

    def run(self, category: str, name: str, fn):
        if self.name_filter and self.name_filter not in name:
            return
        t0 = time.time()
        try:
            fn()
            dur = time.time() - t0
            self.results.append(
                TestResult(name, category, True, duration=dur))
            print(f"  PASS  {name} ({dur:.1f}s)")
        except Exception as e:
            dur = time.time() - t0
            detail = str(e)
            self.results.append(
                TestResult(name, category, False, detail, dur))
            print(f"  FAIL  {name} ({dur:.1f}s)")
            if self.verbose:
                for line in detail.splitlines():
                    print(f"        {line}")

    def print_category(self, category: str):
        print(f"\n[{category}]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_json(s: str) -> dict:
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise AssertionError(
            f"Invalid JSON: {e}\nRaw string: {s!r}") from None


def assert_eq(actual, expected, label: str = ""):
    if actual != expected:
        raise AssertionError(
            f"{label}: expected {expected!r}, got {actual!r}")


def assert_in(value, allowed, label: str = ""):
    if value not in allowed:
        raise AssertionError(
            f"{label}: expected one of {allowed!r}, got {value!r}")


def assert_is_type(value, typ, label: str = ""):
    if not isinstance(value, typ):
        raise AssertionError(
            f"{label}: expected type {typ.__name__}, "
            f"got {type(value).__name__} ({value!r})")


def assert_tool_calls_present(choice):
    if not choice.message.tool_calls:
        parts = [f"No tool calls produced (finish_reason={choice.finish_reason!r})"]
        if choice.message.content:
            preview = choice.message.content[:200]
            parts.append(f"Content instead: {preview!r}")
        raise AssertionError(". ".join(parts))


def assert_content_present(choice):
    if not choice.message.content:
        parts = [f"No content produced (finish_reason={choice.finish_reason!r})"]
        if choice.message.tool_calls:
            names = [tc.function.name for tc in choice.message.tool_calls]
            parts.append(f"Got tool calls instead: {names}")
        raise AssertionError(". ".join(parts))


def validate_tool_call(tc, expected_name: str, required_keys: list[str],
                       enum_checks: dict[str, list] | None = None):
    assert_eq(tc.function.name, expected_name, "function name")
    args = parse_json(tc.function.arguments)
    for key in required_keys:
        if key not in args:
            raise AssertionError(
                f"Required key {key!r} missing from args: {args}")
    if enum_checks:
        for key, allowed in enum_checks.items():
            assert_in(args[key], allowed, f"args.{key}")
    return args


def collect_streaming_tool_calls(stream):
    calls: dict[int, dict] = {}
    finish_reason = None
    for chunk in stream:
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        if choice.delta.tool_calls:
            for tc in choice.delta.tool_calls:
                idx = tc.index
                if idx not in calls:
                    calls[idx] = {"name": "", "arguments": ""}
                if tc.function:
                    if tc.function.name:
                        calls[idx]["name"] += tc.function.name
                    if tc.function.arguments:
                        calls[idx]["arguments"] += tc.function.arguments
    return calls, finish_reason


def collect_streaming_content(stream):
    content = ""
    finish_reason = None
    for chunk in stream:
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        if choice.delta.content:
            content += choice.delta.content
    return content, finish_reason


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

def register_tests(runner: TestRunner):
    client = runner.client
    model = runner.model

    common = dict(model=model, temperature=0, seed=SEED,
                   max_completion_tokens=2048)

    # -----------------------------------------------------------------------
    # Category 1: Basic Tool Calling
    # -----------------------------------------------------------------------
    cat = "TOOL CALLING"
    runner.print_category(cat)

    def basic_tool_call():
        resp = client.chat.completions.create(
            messages=[{"role": "user",
                       "content": "What is the weather in Dallas, Texas "
                                  "in Fahrenheit?"}],
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            **common,
        )
        choice = resp.choices[0]
        assert_tool_calls_present(choice)
        validate_tool_call(
            choice.message.tool_calls[0],
            "get_current_weather",
            ["city", "state", "unit"],
            {"unit": ["celsius", "fahrenheit"]},
        )

    runner.run(cat, "basic_tool_call", basic_tool_call)

    def basic_tool_call_streaming():
        stream = client.chat.completions.create(
            messages=[{"role": "user",
                       "content": "What is the weather in Dallas, Texas "
                                  "in Fahrenheit?"}],
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            stream=True,
            **common,
        )
        calls, finish = collect_streaming_tool_calls(stream)
        assert_eq(finish, "tool_calls", "finish_reason")
        assert calls, "No tool calls in streamed response"
        tc = calls[min(calls)]
        assert_eq(tc["name"], "get_current_weather", "function name")
        args = parse_json(tc["arguments"])
        for key in ["city", "state", "unit"]:
            if key not in args:
                raise AssertionError(f"Missing required key {key!r}: {args}")
        assert_in(args["unit"], ["celsius", "fahrenheit"], "args.unit")

    runner.run(cat, "basic_tool_call_streaming", basic_tool_call_streaming)

    def required_fields_present():
        resp = client.chat.completions.create(
            messages=[{"role": "user",
                       "content": "What's the weather like in New York?"}],
            tools=[WEATHER_TOOL],
            tool_choice="required",
            **common,
        )
        choice = resp.choices[0]
        assert_tool_calls_present(choice)
        args = validate_tool_call(
            choice.message.tool_calls[0],
            "get_current_weather",
            ["city", "state", "unit"],
        )
        assert_is_type(args["city"], str, "args.city")
        assert_is_type(args["state"], str, "args.state")

    runner.run(cat, "required_fields_present", required_fields_present)

    def parallel_tool_calls():
        resp = client.chat.completions.create(
            messages=[{"role": "user",
                       "content": "What is the weather in Dallas, Texas "
                                  "and Orlando, Florida in Fahrenheit?"}],
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            max_completion_tokens=4096,
            **{k: v for k, v in common.items()
               if k != "max_completion_tokens"},
        )
        choice = resp.choices[0]
        assert_tool_calls_present(choice)
        tcs = choice.message.tool_calls
        assert len(tcs) >= 2, (
            f"Expected >=2 tool calls, got {len(tcs)}")
        for tc in tcs:
            validate_tool_call(
                tc, "get_current_weather",
                ["city", "state", "unit"],
                {"unit": ["celsius", "fahrenheit"]},
            )

    runner.run(cat, "parallel_tool_calls", parallel_tool_calls)

    def parallel_tool_calls_streaming():
        stream = client.chat.completions.create(
            messages=[{"role": "user",
                       "content": "What is the weather in Dallas, Texas "
                                  "and Orlando, Florida in Fahrenheit?"}],
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            max_completion_tokens=4096,
            stream=True,
            **{k: v for k, v in common.items()
               if k != "max_completion_tokens"},
        )
        calls, finish = collect_streaming_tool_calls(stream)
        assert_eq(finish, "tool_calls", "finish_reason")
        assert len(calls) >= 2, (
            f"Expected >=2 streamed tool calls, got {len(calls)}")
        for idx in sorted(calls):
            tc = calls[idx]
            assert_eq(tc["name"], "get_current_weather", "function name")
            args = parse_json(tc["arguments"])
            for key in ["city", "state", "unit"]:
                if key not in args:
                    raise AssertionError(
                        f"Tool call {idx}: missing key {key!r}: {args}")
            assert_in(args["unit"], ["celsius", "fahrenheit"],
                      f"tool call {idx} args.unit")

    runner.run(cat, "parallel_tool_calls_streaming",
               parallel_tool_calls_streaming)

    # -----------------------------------------------------------------------
    # Category 2: Constraint Enforcement
    # -----------------------------------------------------------------------
    cat = "CONSTRAINT ENFORCEMENT"
    runner.print_category(cat)

    def enum_enforcement_kelvin():
        resp = client.chat.completions.create(
            messages=[{"role": "user",
                       "content": "What is the temperature in Berlin, "
                                  "Germany in kelvin?"}],
            tools=[WEATHER_TOOL],
            tool_choice="required",
            **common,
        )
        choice = resp.choices[0]
        assert_tool_calls_present(choice)
        args = validate_tool_call(
            choice.message.tool_calls[0],
            "get_current_weather",
            ["city", "state", "unit"],
        )
        assert_in(args["unit"], ["celsius", "fahrenheit"],
                  "ENFORCEMENT: args.unit must not be 'kelvin'")

    runner.run(cat, "enum_enforcement_kelvin", enum_enforcement_kelvin)

    def enum_enforcement_kelvin_streaming():
        stream = client.chat.completions.create(
            messages=[{"role": "user",
                       "content": "What is the temperature in Berlin, "
                                  "Germany in kelvin?"}],
            tools=[WEATHER_TOOL],
            tool_choice="required",
            stream=True,
            **common,
        )
        calls, _ = collect_streaming_tool_calls(stream)
        assert calls, "No tool calls in streamed response"
        tc = calls[min(calls)]
        args = parse_json(tc["arguments"])
        assert_in(args.get("unit"), ["celsius", "fahrenheit"],
                  "ENFORCEMENT (streaming): args.unit must not be 'kelvin'")

    runner.run(cat, "enum_enforcement_kelvin_streaming",
               enum_enforcement_kelvin_streaming)

    def enum_enforcement_calculator():
        resp = client.chat.completions.create(
            messages=[{"role": "user",
                       "content": "Please raise 2 to the power of 8."}],
            tools=[CALCULATOR_TOOL],
            tool_choice="required",
            **common,
        )
        choice = resp.choices[0]
        assert_tool_calls_present(choice)
        args = validate_tool_call(
            choice.message.tool_calls[0],
            "calculate",
            ["operation", "a", "b"],
        )
        assert_in(
            args["operation"],
            ["add", "subtract", "multiply", "divide"],
            "ENFORCEMENT: operation must not be 'power'/'exponentiate'",
        )

    runner.run(cat, "enum_enforcement_calculator",
               enum_enforcement_calculator)

    def named_tool_choice():
        resp = client.chat.completions.create(
            messages=[{"role": "user",
                       "content": "Search for recent news about AI."}],
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            tool_choice={
                "type": "function",
                "function": {"name": "web_search"},
            },
            **common,
        )
        choice = resp.choices[0]
        assert_tool_calls_present(choice)
        assert_eq(choice.message.tool_calls[0].function.name,
                  "web_search", "function name")
        args = parse_json(choice.message.tool_calls[0].function.arguments)
        assert "search_term" in args, f"Missing 'search_term': {args}"

    runner.run(cat, "named_tool_choice", named_tool_choice)

    # -----------------------------------------------------------------------
    # Category 3: Structured Output (response_format)
    # -----------------------------------------------------------------------
    cat = "STRUCTURED OUTPUT"
    runner.print_category(cat)

    calendar_schema = {
        "type": "object",
        "properties": {
            "event_name": {"type": "string"},
            "date": {"type": "string"},
            "participants": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["event_name", "date", "participants"],
        "additionalProperties": False,
    }

    so_common = dict(model=model, temperature=0, seed=SEED,
                     max_completion_tokens=4096)

    def json_schema_basic():
        resp = client.chat.completions.create(
            messages=[{"role": "user",
                       "content": "Alice and Bob are going to a science "
                                  "fair on Friday."}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "calendar_event",
                    "schema": calendar_schema,
                },
            },
            **so_common,
        )
        choice = resp.choices[0]
        assert_content_present(choice)
        data = parse_json(choice.message.content)
        for key in ["event_name", "date", "participants"]:
            if key not in data:
                raise AssertionError(f"Missing required key {key!r}: {data}")
        assert_is_type(data["participants"], list, "participants")
        for i, p in enumerate(data["participants"]):
            assert_is_type(p, str, f"participants[{i}]")
        extra = set(data.keys()) - {"event_name", "date", "participants"}
        if extra:
            raise AssertionError(
                f"Unexpected extra keys {extra} in response: {data}")

    runner.run(cat, "json_schema_basic", json_schema_basic)

    def json_schema_maxitems_enforcement():
        resp = client.chat.completions.create(
            messages=[{"role": "user",
                       "content": "List 10 different colors of the rainbow."}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "color_list",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "colors": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 3,
                            },
                        },
                        "required": ["colors"],
                        "additionalProperties": False,
                    },
                },
            },
            **so_common,
        )
        choice = resp.choices[0]
        assert_content_present(choice)
        data = parse_json(choice.message.content)
        colors = data.get("colors", [])
        assert_is_type(colors, list, "colors")
        if len(colors) > 3:
            raise AssertionError(
                f"ENFORCEMENT: maxItems=3 but got {len(colors)} items: "
                f"{colors}")

    runner.run(cat, "json_schema_maxitems_enforcement",
               json_schema_maxitems_enforcement)

    def json_schema_type_enforcement():
        resp = client.chat.completions.create(
            messages=[{"role": "user",
                       "content": "What is the value of pi? "
                                  "Return the numeric value."}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "math_result",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "integer"},
                            "description": {"type": "string"},
                        },
                        "required": ["value", "description"],
                        "additionalProperties": False,
                    },
                },
            },
            **so_common,
        )
        choice = resp.choices[0]
        assert_content_present(choice)
        data = parse_json(choice.message.content)
        assert_is_type(data["value"], int,
                       "ENFORCEMENT: value must be integer, not float")
        assert_is_type(data["description"], str, "description")

    runner.run(cat, "json_schema_type_enforcement",
               json_schema_type_enforcement)

    def json_schema_streaming():
        stream = client.chat.completions.create(
            messages=[{"role": "user",
                       "content": "Alice and Bob are going to a science "
                                  "fair on Friday."}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "calendar_event",
                    "schema": calendar_schema,
                },
            },
            stream=True,
            **so_common,
        )
        content, finish = collect_streaming_content(stream)
        assert_eq(finish, "stop", "finish_reason")
        data = parse_json(content)
        for key in ["event_name", "date", "participants"]:
            if key not in data:
                raise AssertionError(
                    f"Missing required key {key!r} in streamed JSON: {data}")

    runner.run(cat, "json_schema_streaming", json_schema_streaming)

    # -----------------------------------------------------------------------
    # Category 4: Negative / Edge Cases
    # -----------------------------------------------------------------------
    cat = "NEGATIVE TESTS"
    runner.print_category(cat)

    def no_extra_fields_in_tool_args():
        resp = client.chat.completions.create(
            messages=[{"role": "user",
                       "content": "What is the weather in Paris, France? "
                                  "Also tell me the humidity and wind speed."}],
            tools=[WEATHER_TOOL],
            tool_choice="required",
            **common,
        )
        choice = resp.choices[0]
        assert_tool_calls_present(choice)
        args = parse_json(choice.message.tool_calls[0].function.arguments)
        allowed_keys = {"city", "state", "unit"}
        extra = set(args.keys()) - allowed_keys
        if extra:
            raise AssertionError(
                f"Extra keys not in schema: {extra}. Full args: {args}")

    runner.run(cat, "no_extra_fields_in_tool_args",
               no_extra_fields_in_tool_args)

    def tool_call_roundtrip():
        messages = [
            {"role": "user",
             "content": "What is the weather in Dallas, Texas "
                        "in Fahrenheit?"},
            {"role": "assistant",
             "tool_calls": [{
                 "id": "call_1234",
                 "type": "function",
                 "function": {
                     "name": "get_current_weather",
                     "arguments": json.dumps({
                         "city": "Dallas", "state": "TX",
                         "unit": "fahrenheit",
                     }),
                 },
             }]},
            {"role": "tool",
             "tool_call_id": "call_1234",
             "content": "The weather in Dallas is 98 degrees fahrenheit, "
                        "with partly cloudy skies."},
        ]
        resp = client.chat.completions.create(
            messages=messages,
            tools=[WEATHER_TOOL],
            **common,
        )
        choice = resp.choices[0]
        assert_content_present(choice)
        if choice.finish_reason == "tool_calls":
            raise AssertionError(
                "Model produced more tool calls instead of a text response "
                "after receiving the tool result")

    runner.run(cat, "tool_call_roundtrip", tool_call_roundtrip)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify strict tool calling and structured output "
                    "enforcement against a live vLLM server.",
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000/v1",
        help="vLLM server base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name (auto-detected if omitted)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show full details on test failure",
    )
    parser.add_argument(
        "-k", "--filter", default=None,
        help="Only run tests whose name contains this substring",
    )
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    try:
        models = client.models.list()
    except Exception as e:
        print(f"ERROR: Cannot connect to server at {args.base_url}: {e}",
              file=sys.stderr)
        sys.exit(1)

    model = args.model or models.data[0].id

    print("=" * 50)
    print(" Strict Tool Calling Verification")
    print("=" * 50)
    print(f"Server:  {args.base_url}")
    print(f"Model:   {model}")
    print("=" * 50)

    runner = TestRunner(
        client=client,
        model=model,
        verbose=args.verbose,
        name_filter=args.filter,
    )

    register_tests(runner)

    passed = sum(1 for r in runner.results if r.passed)
    failed = sum(1 for r in runner.results if not r.passed)
    total = len(runner.results)

    print()
    print("=" * 50)
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print("=" * 50)

    if failed:
        print()
        print("Failed tests:")
        for r in runner.results:
            if not r.passed:
                print(f"  - {r.name}: {r.details}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
