# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.parser.json_scan import JsonWalker, walk_json


def _decode_key(raw_slice: str) -> str:
    if "\\" not in raw_slice:
        return raw_slice
    try:
        return json.loads(f'"{raw_slice}"')
    except json.JSONDecodeError:
        return raw_slice


class TestJsonWalkerBasics:
    def test_object_end_via_handler(self):
        raw = '{"a": 1, "b": [2, {"c": "d"}]}'
        end: list[int] = []

        def on_outside(i: int, ch: str, depth: int) -> bool:
            if ch == "}" and depth == 0:
                end.append(i + 1)
                return False
            return True

        pos = walk_json(
            raw, 0, on_outside_char=on_outside, brackets="object", key_depth=None
        )
        assert end == [len(raw)]
        assert pos == len(raw)

    def test_escaped_quote_does_not_close_string(self):
        raw = r'{"k": "has \"quote\" inside"}'
        keys: list[str] = []

        def on_outside(i: int, ch: str, depth: int) -> bool:
            if ch == ":" and depth == 1:
                keys.append(walker.last_key or "")
            return True

        walker = JsonWalker(brackets="any", key_depth=1)
        walker.advance(raw, on_outside_char=on_outside)
        assert keys == ["k"]
        assert walker.depth == 0
        assert walker.incomplete_string_at < 0

    def test_braces_inside_string_ignored(self):
        raw = '{"k": "brace } inside"}'
        closed = False

        def on_outside(i: int, ch: str, depth: int) -> bool:
            nonlocal closed
            if ch == "}" and depth == 0:
                closed = True
                return False
            return True

        walk_json(raw, 0, on_outside_char=on_outside, brackets="any", key_depth=None)
        assert closed

    def test_unicode_escape_key_with_decode(self):
        raw = r'{"na\u006de": 1}'
        seen: list[str] = []

        def on_outside(i: int, ch: str, depth: int) -> bool:
            if ch == ":" and depth == 1 and walker.last_key is not None:
                seen.append(walker.last_key)
            return True

        walker = JsonWalker(brackets="any", decode_keys=_decode_key, key_depth=1)
        walker.advance(raw, on_outside_char=on_outside)
        assert seen == ["name"]

    def test_raw_key_without_decode(self):
        raw = r'{"na\u006de": 1}'
        seen: list[str] = []

        def on_outside(i: int, ch: str, depth: int) -> bool:
            if ch == ":" and depth == 1 and walker.last_key is not None:
                seen.append(walker.last_key)
            return True

        walker = JsonWalker(brackets="any", decode_keys=None, key_depth=1)
        walker.advance(raw, on_outside_char=on_outside)
        assert seen == [r"na\u006de"]

    def test_truncated_leaves_incomplete_string(self):
        walker = JsonWalker(brackets="any", key_depth=None)
        walker.advance('{"a": "hel')
        assert walker.incomplete_string_at == 6
        assert walker.suspend_pos == len('{"a": "hel')

    def test_clear_last_key(self):
        walker = JsonWalker(brackets="any", key_depth=1)
        walker.advance('{"a"')
        assert walker.last_key == "a"
        walker.clear_last_key()
        assert walker.last_key is None


class TestBracketsMode:
    def test_object_brackets_ignore_array_nesting(self):
        # Starting at the outer `{`: arrays must not change object depth.
        raw = '{"a": [{}, {"b": 1}]}'
        end: list[int] = []

        def on_outside(i: int, ch: str, depth: int) -> bool:
            if ch == "}" and depth == 0:
                end.append(i + 1)
                return False
            return True

        walk_json(raw, 0, on_outside_char=on_outside, brackets="object", key_depth=None)
        assert end == [len(raw)]

    def test_any_brackets_closes_array(self):
        raw = '[{"x": 1}]'
        end: list[int] = []

        def on_outside(i: int, ch: str, depth: int) -> bool:
            if ch in "}]" and depth == 0:
                end.append(i + 1)
                return False
            return True

        walk_json(raw, 0, on_outside_char=on_outside, brackets="any", key_depth=None)
        assert end == [len(raw)]

    def test_object_brackets_on_array_never_hits_depth_zero_close(self):
        raw = "[{}]"
        hits = 0

        def on_outside(i: int, ch: str, depth: int) -> bool:
            nonlocal hits
            if ch == "}" and depth == 0:
                hits += 1
            return True

        # Depth starts 0; `[` ignored; `{` -> 1; `}` -> 0 (hit); `]` ignored.
        walk_json(raw, 0, on_outside_char=on_outside, brackets="object", key_depth=None)
        assert hits == 1


class TestResumability:
    CORPUS = [
        '{"a": 1, "b": [2, {"c": "d"}]}',
        r'{"k": "has \"quote\" inside"}',
        '{"k": "brace } inside"}',
        r'{"na\u006de": "escaped key"}',
        '[{"name": "f"}]',
        '{"a": "hel',
        "{",
        "",
        "prose before {",
    ]

    @pytest.mark.parametrize("text", CORPUS)
    def test_char_by_char_matches_one_shot(self, text: str):
        one = JsonWalker(brackets="any", decode_keys=_decode_key, key_depth=1)
        one.advance(text)

        inc = JsonWalker(brackets="any", decode_keys=_decode_key, key_depth=1)
        for end in range(len(text) + 1):
            inc.advance(text[:end])

        assert (
            inc.pos
            == one.pos
            == walk_json(text, 0, brackets="any", decode_keys=_decode_key, key_depth=1)
        )
        assert inc.depth == one.depth
        assert inc.last_key == one.last_key
        assert inc.incomplete_string_at == one.incomplete_string_at

    @pytest.mark.parametrize("text", CORPUS)
    @pytest.mark.parametrize("chunk", [1, 2, 3, 7, 10_000])
    def test_chunked_matches_one_shot(self, text: str, chunk: int):
        one = JsonWalker(brackets="any", key_depth=None)
        one.advance(text)

        inc = JsonWalker(brackets="any", key_depth=None)
        ends = (
            list(range(0, len(text) + 1, chunk)) if chunk < len(text) else [len(text)]
        )
        for end in ends:
            inc.advance(text[:end])
        inc.advance(text)

        assert (inc.pos, inc.depth, inc.incomplete_string_at) == (
            one.pos,
            one.depth,
            one.incomplete_string_at,
        )
