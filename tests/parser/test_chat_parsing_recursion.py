# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for RecursionError handling in the response_template parser.

``json.loads`` raises ``RecursionError`` (not ``ValueError``) once the region body
is nested past the C scanner's depth limit, and the same is true for the schema
driven casts in ``_coerce``.  ``RecursionError`` derives from ``RuntimeError``, so
it is invisible to the ``except ValueError`` / ``except (KeyError, TypeError,
ValueError)`` guards that the parser uses to treat a value as "malformed but
recoverable".  Left unhandled it escapes the parser entirely: the region never
emits ``region_malformed``, the parser is left mid-region, ``finalize()`` raises
again, and request handlers surface an unhandled exception instead of the
documented degrade-in-place behaviour.

Deep nesting is a property of the *input*, so these tests drive the real public
entry points (``ResponseParser.feed``/``finalize`` and ``parse_response``) with
genuinely over-deep payloads rather than mocking the recursion failure.
"""

import sys
import unittest
from unittest import mock

from vllm.parser.chat_parsing import ResponseParser, parse_response
from vllm.parser.chat_parsing.content_parsers import _json, process_field
from vllm.parser.chat_parsing.response_parser import _coerce

# Deep enough to exceed the CPython C scanner's depth for JSON at the default
# recusion limit, and at any lowered limit these tests use.
DEPTH = 5000
DEEP_ARRAY = "[" * DEPTH + "]" * DEPTH

# A region whose body must parse as JSON; no fallback is declared.
JSON_TEMPLATE = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|assistant|>",
    "fields": {
        "tool_call": {"open": "<tool>", "close": "</tool>", "content": "json"},
        "content": {"open": "<say>", "close": "</say>", "content": "text"},
    },
}

# Same, but the field opts into keeping raw text when the value is not JSON.
JSON_FALLBACK_TEMPLATE = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|assistant|>",
    "fields": {
        "tool_call": {
            "open": "<tool>",
            "close": "</tool>",
            "content": "json",
            "content_args": {"allow_non_json": True},
        },
    },
}


def _feed_all(parser: ResponseParser, chunks) -> list[dict]:
    events: list[dict] = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    _message, final_events = parser.finalize()
    return events + final_events


def _types(events) -> list[str]:
    return [event["type"] for event in events]


class DeepRegionDegradesTest(unittest.TestCase):
    """A too-deeply nested region must degrade in place, never escape."""

    def test_deep_json_region_is_malformed_not_raised(self):
        parser = ResponseParser(JSON_TEMPLATE, prefix="<|assistant|>")
        events = parser.feed("<tool>" + DEEP_ARRAY)
        events += parser.feed("</tool>")

        self.assertEqual(_types(events).count("region_malformed"), 1)
        self.assertNotIn("region_close", _types(events))
        malformed = next(e for e in events if e["type"] == "region_malformed")
        self.assertEqual(malformed["field"], "tool_call")
        # The body is not JSON-parseable, so the recorded cause stays a ValueError
        # regardless of which layer noticed the depth.
        self.assertIsInstance(malformed["error"], ValueError)

    def test_parser_state_is_consistent_after_a_deep_region(self):
        parser = ResponseParser(JSON_TEMPLATE, prefix="<|assistant|>")
        parser.feed("<tool>" + DEEP_ARRAY + "</tool>")

        # Back in the implicit region, body cleared, and the field recorded so a
        # required field is not reported as missing on top of being malformed.
        self.assertFalse(parser._opened)
        self.assertEqual(parser._body, "")
        self.assertIn("tool_call", parser._malformed_fields)

        # finalize() must not re-raise, and later regions must still work.
        message, final_events = parser.finalize()
        self.assertEqual(_types(final_events).count("region_malformed"), 0)
        self.assertEqual(message.get("role"), "assistant")

    def test_parsing_continues_after_a_deep_region(self):
        # Streaming API: the parser keeps going after a malformed region.
        parser = ResponseParser(JSON_TEMPLATE, prefix="<|assistant|>")
        parser.feed("<tool>" + DEEP_ARRAY + "</tool>")
        parser.feed("<say>still working</say>")
        message, _events = parser.finalize()
        self.assertEqual(message["content"], "still working")
        self.assertNotIn("tool_call", message)

    def test_parse_response_reports_a_malformed_region_as_an_error(self):
        # parse_response() re-raises malformed-region errors by design, and must
        # do so with a ValueError here rather than leaking a RecursionError.
        with self.assertRaises(ValueError):
            parse_response(
                "<tool>" + DEEP_ARRAY + "</tool><say>after</say>",
                JSON_TEMPLATE,
                prefix="<|assistant|>",
            )

    def test_two_deep_regions_then_a_valid_one(self):
        parser = ResponseParser(JSON_TEMPLATE, prefix="<|assistant|>")
        parser.feed("<tool>" + DEEP_ARRAY + "</tool>")
        parser.feed("<tool>" + DEEP_ARRAY + "</tool>")
        parser.feed("<say>ok</say>")
        message, _events = parser.finalize()
        self.assertEqual(message["content"], "ok")

    def test_deep_region_at_end_of_stream_is_recorded(self):
        # The close delimiter is implied by end-of-stream here, exercising the
        # other path into _close_current.
        parser = ResponseParser(JSON_TEMPLATE, prefix="<|assistant|>")
        events = parser.feed("<tool>" + DEEP_ARRAY)
        _message, final_events = parser.finalize()
        self.assertEqual(_types(events + final_events).count("region_malformed"), 1)

    def test_deep_region_does_not_escape_for_any_chunking(self):
        text = "<|assistant|><tool>" + DEEP_ARRAY + "</tool>"
        for size in (1, 3, 7, 64, 1000, len(text)):
            with self.subTest(chunk_size=size):
                parser = ResponseParser(JSON_TEMPLATE, prefix="<|assistant|>")
                chunks = [text[i : i + size] for i in range(0, len(text), size)]
                # Must not raise: no RecursionError may reach the caller.
                events = _feed_all(parser, chunks)
                self.assertIn("region_malformed", _types(events))


class DeepRegionFallbackTest(unittest.TestCase):
    """With a fallback declared, a too-deep value must keep its raw text."""

    def test_allow_non_json_keeps_raw_text_instead_of_dropping(self):
        message = parse_response(
            "<tool>" + DEEP_ARRAY + "</tool>",
            JSON_FALLBACK_TEMPLATE,
            prefix="<|assistant|>",
        )
        self.assertEqual(message["tool_call"], DEEP_ARRAY)

    def test_deep_value_in_a_text_region_is_unaffected(self):
        message = parse_response(
            "<say>" + DEEP_ARRAY + "</say>",
            JSON_TEMPLATE,
            prefix="<|assistant|>",
        )
        self.assertEqual(message["content"], DEEP_ARRAY)


class ExceptionBoundaryTest(unittest.TestCase):
    """Only the recoverable input-error classes may be swallowed."""

    def test_other_runtimeerror_still_escapes(self):
        class OtherRuntime(RuntimeError):
            pass

        parser = ResponseParser(JSON_TEMPLATE, prefix="<|assistant|>")
        with (
            mock.patch(
                "vllm.parser.chat_parsing.response_parser.process_field",
                side_effect=OtherRuntime("boom"),
            ),
            self.assertRaises(OtherRuntime),
        ):
            parser.feed("<tool>{}")
            parser.feed("</tool>")

    def test_recoverable_classes_become_malformed(self):
        for exc in (RecursionError(), KeyError("k"), TypeError("t"), ValueError("v")):
            with self.subTest(exc=type(exc).__name__):
                parser = ResponseParser(JSON_TEMPLATE, prefix="<|assistant|>")
                with mock.patch(
                    "vllm.parser.chat_parsing.response_parser.process_field",
                    side_effect=type(exc)("injected"),
                ):
                    events = parser.feed("<tool>{}")
                    events += parser.feed("</tool>")
                self.assertEqual(_types(events).count("region_malformed"), 1)


class InnerCoercionFallbackTest(unittest.TestCase):
    """Schema driven casts must not turn a deep value into a raised exception."""

    def test_coerce_keeps_raw_text_for_a_deep_array(self):
        self.assertEqual(_coerce(DEEP_ARRAY, ("array",)), DEEP_ARRAY)

    def test_coerce_keeps_raw_text_for_a_deep_object(self):
        deep_object = "{" * DEPTH + "}" * DEPTH
        self.assertEqual(_coerce(deep_object, ("object",)), deep_object)


class ContentParserLayerTest(unittest.TestCase):
    """The json content parser reports depth as a parse failure, not a crash."""

    def test_json_parser_reports_depth_as_a_parse_failure(self):
        with self.assertRaises(ValueError):
            _json(DEEP_ARRAY, {})

    def test_json_parser_honours_allow_non_json_for_a_deep_value(self):
        self.assertEqual(_json(DEEP_ARRAY, {"allow_non_json": True}), DEEP_ARRAY)

    def test_process_field_surfaces_a_value_error_for_a_deep_json_field(self):
        template = JSON_TEMPLATE
        field = mock.Mock(
            name="tool_call",
            content="json",
            content_args={},
            captures={},
            transform=None,
            transform_each=False,
        )
        with self.assertRaises(ValueError):
            process_field(DEEP_ARRAY, field, {})
        self.assertIsNotNone(template)


class RecursionLimitIndependenceTest(unittest.TestCase):
    """The fix must hold under a lowered limit, and must not tune the limit."""

    def test_holds_when_the_recursion_limit_is_lowered(self):
        original = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(300)
            parser = ResponseParser(JSON_TEMPLATE, prefix="<|assistant|>")
            parser.feed("<tool>" + DEEP_ARRAY + "</tool><say>after</say>")
            message, _events = parser.finalize()
            self.assertEqual(message["content"], "after")
        finally:
            sys.setrecursionlimit(original)

    def test_recursion_limit_is_never_tuned_by_the_parser(self):
        import inspect

        from vllm.parser.chat_parsing import content_parsers, response_parser

        for module in (response_parser, content_parsers):
            source = inspect.getsource(module)
            self.assertNotIn("setrecursionlimit", source)


class ValidInputUnchangedTest(unittest.TestCase):
    """The change must not alter behaviour for values that parse normally."""

    def test_valid_json_region_still_closes(self):
        message = parse_response(
            '<tool>{"name": "set_alarm", "hour": 7}</tool><say>done</say>',
            JSON_TEMPLATE,
            prefix="<|assistant|>",
        )
        self.assertEqual(message["tool_call"], {"name": "set_alarm", "hour": 7})
        self.assertEqual(message["content"], "done")

    def test_moderately_nested_json_still_parses(self):
        message = parse_response(
            "<tool>" + "[" * 100 + "]" * 100 + "</tool>",
            JSON_TEMPLATE,
            prefix="<|assistant|>",
        )
        self.assertIsInstance(message["tool_call"], list)

    def test_malformed_event_shape_is_unchanged(self):
        parser = ResponseParser(JSON_TEMPLATE, prefix="<|assistant|>")
        events = parser.feed("<tool>{not json")
        events += parser.feed("</tool>")
        malformed = next(e for e in events if e["type"] == "region_malformed")
        self.assertEqual(set(malformed), {"type", "field", "start", "end", "error"})
        self.assertEqual(malformed["field"], "tool_call")
        self.assertIsInstance(malformed["error"], ValueError)


if __name__ == "__main__":
    unittest.main()
