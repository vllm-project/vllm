# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for include_reasoning suppression in the unified Parser interface.

Covers both non-streaming (extract_response_outputs) and streaming
(parse_delta) paths, plus the legacy ResponsesParser.process() path.
"""

import json

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.abstract_parser import _WrappedParser
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser


class ThinkReasoningParser(BaseThinkingReasoningParser):
    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"


MODEL_OUTPUT_REASONING_AND_CONTENT = (
    "<think>let me think about this</think>The answer is 42."
)

MODEL_OUTPUT_REASONING_AND_TOOL = (
    "<think>I need to call a tool</think>"
    '<tool_call>\n{"name": "get_weather", '
    '"arguments": {"city": "Dallas"}}\n</tool_call>'
)

MODEL_OUTPUT_CONTENT_ONLY = "The answer is 42."


@pytest.fixture(scope="module")
def tokenizer():
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer("Qwen/Qwen3-32B")


def make_responses_request(**kwargs) -> ResponsesRequest:
    defaults = dict(model="test-model", input="test input")
    defaults.update(kwargs)
    return ResponsesRequest(**defaults)


def make_chat_request(**kwargs) -> ChatCompletionRequest:
    defaults = dict(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


def make_parser(tokenizer, reasoning=False, tool=False):
    _WrappedParser.reasoning_parser_cls = ThinkReasoningParser if reasoning else None
    _WrappedParser.tool_parser_cls = Hermes2ProToolParser if tool else None
    return _WrappedParser(tokenizer)


# ── Non-streaming: extract_response_outputs ──────────────────────────


class TestExtractResponseOutputsIncludeReasoning:
    def test_include_reasoning_true_has_reasoning_item(self, tokenizer):
        """Default: reasoning items appear in output."""
        parser = make_parser(tokenizer, reasoning=True)
        request = make_responses_request(include_reasoning=True)

        outputs = parser.extract_response_outputs(
            model_output=MODEL_OUTPUT_REASONING_AND_CONTENT,
            model_output_token_ids=tokenizer.encode(
                MODEL_OUTPUT_REASONING_AND_CONTENT,
                add_special_tokens=False,
            ),
            request=request,
        )

        types = [o.type for o in outputs]
        assert "reasoning" in types
        assert "message" in types

    def test_include_reasoning_false_no_reasoning_item(self, tokenizer):
        """Reasoning item is suppressed when include_reasoning=False."""
        parser = make_parser(tokenizer, reasoning=True)
        request = make_responses_request(include_reasoning=False)

        outputs = parser.extract_response_outputs(
            model_output=MODEL_OUTPUT_REASONING_AND_CONTENT,
            model_output_token_ids=tokenizer.encode(
                MODEL_OUTPUT_REASONING_AND_CONTENT,
                add_special_tokens=False,
            ),
            request=request,
        )

        types = [o.type for o in outputs]
        assert "reasoning" not in types
        assert "message" in types
        assert outputs[0].content[0].text == "The answer is 42."

    def test_include_reasoning_false_content_preserved(self, tokenizer):
        """Content is extracted correctly even when reasoning is suppressed."""
        parser = make_parser(tokenizer, reasoning=True)
        request = make_responses_request(include_reasoning=False)

        outputs = parser.extract_response_outputs(
            model_output=MODEL_OUTPUT_REASONING_AND_CONTENT,
            model_output_token_ids=tokenizer.encode(
                MODEL_OUTPUT_REASONING_AND_CONTENT,
                add_special_tokens=False,
            ),
            request=request,
        )

        message = next(o for o in outputs if o.type == "message")
        assert message.content[0].text == "The answer is 42."

    def test_include_reasoning_false_tool_calls_preserved(self, tokenizer):
        """Tool calls still work when reasoning is suppressed."""
        parser = make_parser(tokenizer, reasoning=True, tool=True)
        request = make_responses_request(
            include_reasoning=False,
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
        )

        outputs = parser.extract_response_outputs(
            model_output=MODEL_OUTPUT_REASONING_AND_TOOL,
            model_output_token_ids=tokenizer.encode(
                MODEL_OUTPUT_REASONING_AND_TOOL,
                add_special_tokens=False,
            ),
            request=request,
            enable_auto_tools=True,
        )

        types = [o.type for o in outputs]
        assert "reasoning" not in types
        assert "function_call" in types
        fc = next(o for o in outputs if o.type == "function_call")
        assert fc.name == "get_weather"
        assert json.loads(fc.arguments) == {"city": "Dallas"}

    def test_no_reasoning_parser_include_false_is_noop(self, tokenizer):
        """include_reasoning=False is harmless when no reasoning parser."""
        parser = make_parser(tokenizer, reasoning=False)
        request = make_responses_request(include_reasoning=False)

        outputs = parser.extract_response_outputs(
            model_output=MODEL_OUTPUT_CONTENT_ONLY,
            model_output_token_ids=tokenizer.encode(
                MODEL_OUTPUT_CONTENT_ONLY, add_special_tokens=False
            ),
            request=request,
        )

        assert len(outputs) == 1
        assert outputs[0].type == "message"
        assert outputs[0].content[0].text == MODEL_OUTPUT_CONTENT_ONLY

    def test_default_include_reasoning_is_true(self, tokenizer):
        """ResponsesRequest defaults to include_reasoning=True."""
        request = make_responses_request()
        assert request.include_reasoning is True

    def test_chat_completion_request_also_works(self, tokenizer):
        """extract_response_outputs respects ChatCompletionRequest too."""
        parser = make_parser(tokenizer, reasoning=True)
        request = make_responses_request(include_reasoning=False)

        outputs = parser.extract_response_outputs(
            model_output=MODEL_OUTPUT_REASONING_AND_CONTENT,
            model_output_token_ids=tokenizer.encode(
                MODEL_OUTPUT_REASONING_AND_CONTENT,
                add_special_tokens=False,
            ),
            request=request,
        )

        assert all(o.type != "reasoning" for o in outputs)


# ── Streaming: parse_delta ───────────────────────────────────────────


def stream_text(parser, tokenizer, text, request, prompt_token_ids=None):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    results: list[DeltaMessage | None] = []
    for tid in token_ids:
        delta_text = tokenizer.decode([tid])
        result = parser.parse_delta(
            delta_text, [tid], request, prompt_token_ids=prompt_token_ids
        )
        prompt_token_ids = None
        results.append(result)
    return results


def collect_fields(results):
    all_reasoning = "".join(r.reasoning for r in results if r and r.reasoning)
    all_content = "".join(r.content for r in results if r and r.content)
    all_tool_calls = [tc for r in results if r and r.tool_calls for tc in r.tool_calls]
    return all_reasoning, all_content, all_tool_calls


class TestParseDeltaIncludeReasoning:
    def test_streaming_include_true_emits_reasoning(self, tokenizer):
        """With include_reasoning=True, reasoning deltas are emitted."""
        parser = make_parser(tokenizer, reasoning=True)
        request = make_responses_request(include_reasoning=True)

        results = stream_text(
            parser,
            tokenizer,
            MODEL_OUTPUT_REASONING_AND_CONTENT,
            request,
            prompt_token_ids=[],
        )
        reasoning, content, _ = collect_fields(results)

        assert "let me think about this" in reasoning
        assert "42" in content

    def test_streaming_include_false_suppresses_reasoning(self, tokenizer):
        """With include_reasoning=False, no reasoning deltas are emitted."""
        parser = make_parser(tokenizer, reasoning=True)
        request = make_responses_request(include_reasoning=False)

        results = stream_text(
            parser,
            tokenizer,
            MODEL_OUTPUT_REASONING_AND_CONTENT,
            request,
            prompt_token_ids=[],
        )
        reasoning, content, _ = collect_fields(results)

        assert reasoning == ""
        assert "42" in content

    def test_streaming_include_false_content_still_works(self, tokenizer):
        """Content is correctly extracted in streaming even with suppression."""
        parser = make_parser(tokenizer, reasoning=True)
        request = make_responses_request(include_reasoning=False)

        results = stream_text(
            parser,
            tokenizer,
            MODEL_OUTPUT_REASONING_AND_CONTENT,
            request,
            prompt_token_ids=[],
        )
        _, content, _ = collect_fields(results)

        assert "The answer is 42" in content

    def test_streaming_include_false_tool_calls_preserved(self, tokenizer):
        """Tool calls stream correctly when reasoning is suppressed."""
        parser = make_parser(tokenizer, reasoning=True, tool=True)
        request = make_responses_request(include_reasoning=False)

        results = stream_text(
            parser,
            tokenizer,
            MODEL_OUTPUT_REASONING_AND_TOOL,
            request,
            prompt_token_ids=[],
        )
        reasoning, content, tool_calls = collect_fields(results)

        assert reasoning == ""
        assert len(tool_calls) > 0
        assert tool_calls[0].function.name == "get_weather"
        tool_args = "".join(
            tc.function.arguments for tc in tool_calls if tc.function.arguments
        )
        assert json.loads(tool_args) == {"city": "Dallas"}

    def test_streaming_no_reasoning_parser_include_false(self, tokenizer):
        """No crash when reasoning parser absent and include_reasoning=False."""
        parser = make_parser(tokenizer, reasoning=False)
        request = make_responses_request(include_reasoning=False)

        results = stream_text(
            parser,
            tokenizer,
            MODEL_OUTPUT_CONTENT_ONLY,
            request,
            prompt_token_ids=[],
        )
        reasoning, content, _ = collect_fields(results)

        assert reasoning == ""
        assert "42" in content

    def test_streaming_chat_completion_include_false(self, tokenizer):
        """parse_delta also respects ChatCompletionRequest.include_reasoning."""
        parser = make_parser(tokenizer, reasoning=True)
        request = make_chat_request(include_reasoning=False)

        results = stream_text(
            parser,
            tokenizer,
            MODEL_OUTPUT_REASONING_AND_CONTENT,
            request,
            prompt_token_ids=[],
        )
        reasoning, content, _ = collect_fields(results)

        assert reasoning == ""
        assert "42" in content

    def test_streaming_reasoning_only_deltas_become_none(self, tokenizer):
        """Deltas that carry only reasoning become None (not empty)."""
        parser = make_parser(tokenizer, reasoning=True)
        request = make_responses_request(include_reasoning=False)

        results = stream_text(
            parser,
            tokenizer,
            MODEL_OUTPUT_REASONING_AND_CONTENT,
            request,
            prompt_token_ids=[],
        )

        for r in results:
            if r is not None:
                assert r.reasoning is None


# ── Legacy ResponsesParser.process() ─────────────────────────────────


class TestResponsesParserIncludeReasoning:
    def _make_responses_parser(self, tokenizer, request):
        from vllm.entrypoints.openai.parser.responses_parser import (
            ResponsesParser,
        )

        _WrappedParser.reasoning_parser_cls = ThinkReasoningParser
        _WrappedParser.tool_parser_cls = None
        response_messages: list = []

        return ResponsesParser(
            tokenizer=tokenizer,
            parser_cls=_WrappedParser,
            response_messages=response_messages,
            request=request,
            chat_template=None,
            chat_template_content_format="auto",
        )

    def test_process_include_false_suppresses_reasoning(self, tokenizer):
        """ResponsesParser.process() suppresses reasoning items."""
        from vllm.outputs import CompletionOutput

        request = make_responses_request(include_reasoning=False)
        parser = self._make_responses_parser(tokenizer, request)

        output = CompletionOutput(
            index=0,
            text=MODEL_OUTPUT_REASONING_AND_CONTENT,
            token_ids=tokenizer.encode(
                MODEL_OUTPUT_REASONING_AND_CONTENT,
                add_special_tokens=False,
            ),
            cumulative_logprob=None,
            logprobs=None,
            finish_reason="stop",
        )

        parser.process(output)

        types = [getattr(m, "type", None) for m in parser.response_messages]
        assert "reasoning" not in types
        assert "message" in types

    def test_process_include_true_has_reasoning(self, tokenizer):
        """ResponsesParser.process() includes reasoning by default."""
        from vllm.outputs import CompletionOutput

        request = make_responses_request(include_reasoning=True)
        parser = self._make_responses_parser(tokenizer, request)

        output = CompletionOutput(
            index=0,
            text=MODEL_OUTPUT_REASONING_AND_CONTENT,
            token_ids=tokenizer.encode(
                MODEL_OUTPUT_REASONING_AND_CONTENT,
                add_special_tokens=False,
            ),
            cumulative_logprob=None,
            logprobs=None,
            finish_reason="stop",
        )

        parser.process(output)

        types = [getattr(m, "type", None) for m in parser.response_messages]
        assert "reasoning" in types
        assert "message" in types
