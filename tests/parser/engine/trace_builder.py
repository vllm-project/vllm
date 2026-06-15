# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On-demand trace builder for parser engine testing and benchmarks.

Generates token sequences programmatically from model-agnostic scenario
definitions.  Each model format handler knows how to render scenarios
into the model's output format, tokenize them with correct special token
IDs, and compute expected parse outputs.

Every generated sample is self-validated by replaying it through the
real parser before being returned.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from typing import Any

from tests.parser.engine.replay_harness import (
    MockTokenizer,
    Sample,
    assert_parse_output,
    collect_output,
    replay_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.parser.engine.registered_adapters import (
    Qwen3Parser,
)

# ── Data structures ──────────────────────────────────────────────────


@dataclass
class ToolCallSpec:
    name: str
    arguments: dict[str, Any]


@dataclass
class Scenario:
    id: str
    description: str
    reasoning: str | None = None
    content: str | None = None
    tool_calls: list[ToolCallSpec] | None = None


# ── Scenarios ────────────────────────────────────────────────────────

_READ_TOOL = ToolCallSpec("read_file", {"path": "/tmp/test.txt"})
_BASH_TOOL = ToolCallSpec(
    "bash", {"command": "hostname", "description": "Get hostname"}
)
_WEATHER_TOOL = ToolCallSpec(
    "get_weather",
    {"city": "Dallas", "state": "TX", "unit": "fahrenheit"},
)
_COMPLEX_TOOL = ToolCallSpec(
    "search",
    {
        "query": "vllm parser",
        "filters": {"language": "python", "min_stars": 100},
        "tags": ["ml", "inference"],
        "limit": 10,
        "verbose": True,
    },
)

SCENARIOS: list[Scenario] = [
    Scenario(
        id="think-then-tool",
        description="Reasoning then single tool call",
        reasoning="Let me check the file.",
        tool_calls=[_READ_TOOL],
    ),
    Scenario(
        id="think-then-parallel-tools",
        description="Reasoning then two parallel tool calls",
        reasoning="I need to run both commands.",
        tool_calls=[_BASH_TOOL, _WEATHER_TOOL],
    ),
    Scenario(
        id="think-then-content",
        description="Reasoning then content response",
        reasoning="Let me think about this carefully.",
        content="The answer is 42.",
    ),
    Scenario(
        id="content-only",
        description="Plain content response without reasoning",
        content="Hello! How can I help you today?",
    ),
    Scenario(
        id="tool-only",
        description="Tool call without reasoning",
        tool_calls=[_READ_TOOL],
    ),
    Scenario(
        id="complex-json-args",
        description="Tool call with nested objects, arrays, numbers, booleans",
        reasoning="This needs a complex query.",
        tool_calls=[_COMPLEX_TOOL],
    ),
    Scenario(
        id="whitespace-before-tool",
        description="Whitespace-only content before tool call",
        content="\n\n",
        tool_calls=[_WEATHER_TOOL],
    ),
    Scenario(
        id="think-content-tool",
        description="Reasoning, content, then tool call",
        reasoning="Let me analyze and then fetch data.",
        content="Checking the weather now.",
        tool_calls=[_WEATHER_TOOL],
    ),
    Scenario(
        id="think-whitespace-tool",
        description="Reasoning, whitespace-only gap, then tool call",
        reasoning="Let me check the file contents.",
        content="\n\n",
        tool_calls=[_READ_TOOL],
    ),
    Scenario(
        id="empty-reasoning-content",
        description="Empty reasoning section followed by content",
        reasoning="",
        content="The epoch timestamp is 1779111346.",
    ),
]


# ── Tokenization ─────────────────────────────────────────────────────


def _word_split(text: str) -> list[str]:
    """Split text into word-like tokens, preserving all characters."""
    if not text:
        return []
    parts: list[str] = []
    current = ""
    for ch in text:
        if ch in " \t\n\r" and current and current[-1] not in " \t\n\r":
            parts.append(current)
            current = ch
        else:
            current += ch
    if current:
        parts.append(current)
    return parts


def _tokenize(
    segments: list[tuple[str, bool]],
    vocab: dict[str, int],
    start_id: int = 100,
) -> list[tuple[int, str]]:
    """Build token list from segments.

    Each segment is ``(text, is_special)``.  Special segments use vocab
    IDs; content segments are word-split with sequential IDs.
    """
    tokens: list[tuple[int, str]] = []
    next_id = start_id

    for text, is_special in segments:
        if not text:
            continue
        if is_special:
            tid = vocab.get(text)
            if tid is None:
                raise ValueError(f"Special token {text!r} not in vocab")
            tokens.append((tid, text))
        else:
            for word in _word_split(text):
                tokens.append((next_id, word))
                next_id += 1

    return tokens


# ── Tool definitions ─────────────────────────────────────────────────


def _infer_schema(value: object) -> dict:
    """Infer a JSON Schema from a Python value, recursing into dicts/lists."""
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, dict):
        return {
            "type": "object",
            "properties": {k: _infer_schema(v) for k, v in value.items()},
        }
    if isinstance(value, list) and value:
        return {"type": "array", "items": _infer_schema(value[0])}
    if isinstance(value, list):
        return {"type": "array"}
    return {}


def _tool_defs(tool_calls: list[ToolCallSpec]) -> list[dict]:
    """Generate OpenAI-style tool definitions from tool call specs."""
    seen: set[str] = set()
    tools: list[dict] = []
    for tc in tool_calls:
        if tc.name in seen:
            continue
        seen.add(tc.name)
        properties = {k: _infer_schema(v) for k, v in tc.arguments.items()}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tc.name,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                    },
                },
            }
        )
    return tools


# ── Format handlers ──────────────────────────────────────────────────


def _expected_tc(scenario: Scenario) -> list[dict] | None:
    if not scenario.tool_calls:
        return None
    return [{"name": tc.name, "arguments": tc.arguments} for tc in scenario.tool_calls]


def _expected_tools(scenario: Scenario) -> list[dict] | None:
    return _tool_defs(scenario.tool_calls) if scenario.tool_calls else None


def _validate_sample(sample: Sample, parser_cls: type, **kwargs) -> None:
    """Replay sample through the real parser and assert correctness."""
    tokenizer = MockTokenizer(vocab=dict(sample.vocab), tokens=sample.tokens)
    parser = parser_cls(tokenizer, sample.tools, **kwargs)
    deltas = replay_streaming(parser, sample.tokens, chunk_size=1, tools=sample.tools)
    output = collect_output(deltas)
    assert_parse_output(output, sample)


def _validate_tools(
    tools: list[dict] | None,
) -> list[ChatCompletionToolsParam] | None:
    if not tools:
        return None
    return [ChatCompletionToolsParam.model_validate(t) for t in tools]


def _make_sample(
    sample_id: str,
    description: str,
    vocab: dict[str, int],
    segments: list[tuple[str, bool]],
    expected_reasoning: str | None,
    expected_content: str | None,
    expected_tool_calls: list[dict] | None,
    tools: list[dict] | None,
    chat_template_kwargs: dict | None = None,
) -> Sample:
    tokens = _tokenize(segments, vocab)
    return Sample(
        id=sample_id,
        description=description,
        source="trace-builder",
        vocab=dict(vocab),
        tokens=tokens,
        expected_reasoning=expected_reasoning,
        expected_content=expected_content,
        expected_tool_calls=expected_tool_calls,
        tools=_validate_tools(tools),
        chat_template_kwargs=chat_template_kwargs,
    )


# ── Qwen3 / NemotronV3 (XML tool format, starts in REASONING) ───────

_QWEN3_VOCAB: dict[str, int] = {
    "<think>": 50,
    "</think>": 51,
    "<tool_call>": 60,
    "</tool_call>": 61,
}


def _qwen3_arg_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _qwen3_tool_segments(tc: ToolCallSpec) -> list[tuple[str, bool]]:
    parts = [f"\n<function={tc.name}>"]
    for key, value in tc.arguments.items():
        parts.append(f"\n<parameter={key}>{_qwen3_arg_value(value)}</parameter>")
    parts.append("\n</function>\n")
    return [
        ("<tool_call>", True),
        ("".join(parts), False),
        ("</tool_call>", True),
    ]


def _qwen3_segments(scenario: Scenario) -> list[tuple[str, bool]]:
    segs: list[tuple[str, bool]] = []
    if scenario.reasoning is not None:
        segs.append((scenario.reasoning, False))
    if scenario.content is not None or scenario.tool_calls:
        segs.append(("</think>", True))
    if scenario.content is not None:
        segs.append((scenario.content, False))
    if scenario.tool_calls:
        for tc in scenario.tool_calls:
            segs.extend(_qwen3_tool_segments(tc))
    return segs


def _qwen3_expected_content(scenario: Scenario) -> str | None:
    if (
        scenario.content is not None
        and scenario.tool_calls
        and not scenario.content.strip()
    ):
        return ""
    return scenario.content


def _build_qwen3(
    scenario: Scenario,
    name: str = "qwen3",
    parser_cls: type = Qwen3Parser,
    strip_trailing_ws: bool = False,
    validate: bool = True,
) -> Sample:
    expected_reasoning: str | None
    if scenario.reasoning is not None:
        r = scenario.reasoning
        if strip_trailing_ws:
            r = r.rstrip()
        expected_reasoning = r
    else:
        expected_reasoning = ""

    sample = _make_sample(
        sample_id=f"{name}-{scenario.id}",
        description=scenario.description,
        vocab=_QWEN3_VOCAB,
        segments=_qwen3_segments(scenario),
        expected_reasoning=expected_reasoning,
        expected_content=_qwen3_expected_content(scenario),
        expected_tool_calls=_expected_tc(scenario),
        tools=_expected_tools(scenario),
    )
    if validate:
        _validate_sample(sample, parser_cls)
    return sample


# ── Registry and public API ──────────────────────────────────────────

_BUILDERS: dict[str, Any] = {
    "qwen3": _build_qwen3,
}


@functools.cache
def build_samples(model: str) -> tuple[Sample, ...]:
    """Build all scenario samples for a model, self-validated."""
    builder = _BUILDERS[model]
    return tuple(builder(s) for s in SCENARIOS)


def build_sample(model: str, scenario: Scenario) -> Sample:
    """Build a single sample for one model + scenario."""
    return _BUILDERS[model](scenario)


def build_scaling_sample(
    model: str, token_count: int, validate: bool = False
) -> Sample:
    """Build a sample with approximately *token_count* tokens."""
    sentence = "The quick brown fox jumps over the lazy dog. "
    text = sentence * (token_count // 10 + 1)
    scenario = Scenario(
        id=f"scaling-{token_count}",
        description=f"Scaling test with ~{token_count} tokens",
        reasoning=text,
        tool_calls=[_READ_TOOL],
    )
    return _BUILDERS[model](scenario, validate=validate)
