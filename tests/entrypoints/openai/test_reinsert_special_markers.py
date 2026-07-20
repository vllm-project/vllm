# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the special-token reinsertion helper used to keep markers
like ``</think>`` inline in chat content when the request runs with
``skip_special_tokens=True`` and no reasoning parser is configured.

Uses a minimal fake tokenizer so the splice logic is exercised hermetically
(no model download / GPU)."""

import pytest

from vllm.entrypoints.openai.chat_completion.serving import (
    _reasoning_parser_text,
    _reinsert_markers_by_ids,
    _reinsert_special_markers,
)

# Fixed toy vocab. Special tokens are stripped when skip_special_tokens=True.
THINK_END = 100
EOS = 101
TOOL_CALL_START = 102
TOOL_CALL_END = 103
SPECIAL_IDS = {THINK_END, EOS, TOOL_CALL_START, TOOL_CALL_END}
ID_TO_TOK = {
    THINK_END: "</think>",
    EOS: "<|im_end|>",
    TOOL_CALL_START: "<tool_call>",
    TOOL_CALL_END: "</tool_call>",
}
# A few normal tokens for building text.
WORDS = {1: "a", 2: "b", 3: "c", 4: "\n"}
TOK_TO_ID = {v: k for k, v in {**ID_TO_TOK, **WORDS}.items()}


class FakeTokenizer:
    """Decodes id lists; drops special tokens when skip_special_tokens=True."""

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for i in ids:
            if i in SPECIAL_IDS:
                if not skip_special_tokens:
                    out.append(ID_TO_TOK[i])
                continue
            out.append(WORDS.get(i, ""))
        return "".join(out)

    def get_vocab(self):
        return dict(TOK_TO_ID)

    def convert_tokens_to_ids(self, tok):
        return TOK_TO_ID.get(tok, 0)

    def convert_ids_to_tokens(self, tid):
        return {i: t for t, i in TOK_TO_ID.items()}.get(tid)


@pytest.fixture
def tok():
    return FakeTokenizer()


MARKERS = [("</think>", THINK_END)]


def test_single_token_delta(tok):
    ids = [THINK_END]
    stripped = tok.decode(ids)  # ""
    assert _reinsert_special_markers(tok, stripped, ids, True, MARKERS) == "</think>"


def test_multi_token_delta_preserves_both_sides(tok):
    # "a" </think> "b"
    ids = [1, THINK_END, 2]
    stripped = tok.decode(ids)  # "ab"
    assert _reinsert_special_markers(tok, stripped, ids, True, MARKERS) == "a</think>b"


def test_idempotent_when_literal_present(tok):
    ids = [1, THINK_END, 2]
    text = "a</think>b"
    assert _reinsert_special_markers(tok, text, ids, True, MARKERS) == text


def test_eos_not_leaked(tok):
    # a </think> b <|im_end|>
    ids = [1, THINK_END, 2, EOS]
    stripped = tok.decode(ids)  # "ab"
    res = _reinsert_special_markers(tok, stripped, ids, True, MARKERS)
    assert res == "a</think>b"
    assert "<|im_end|>" not in res


def test_skip_special_tokens_false_is_noop(tok):
    ids = [1, THINK_END, 2]
    assert _reinsert_special_markers(tok, "ab", ids, False, MARKERS) == "ab"


def test_no_marker_in_output_unchanged(tok):
    ids = [1, 2, 3]
    stripped = tok.decode(ids)
    assert _reinsert_special_markers(tok, stripped, ids, True, MARKERS) == stripped


def test_empty_token_ids_unchanged(tok):
    assert _reinsert_special_markers(tok, "ab", [], True, MARKERS) == "ab"


def test_none_tokenizer_unchanged():
    assert _reinsert_special_markers(None, "ab", [1], True, MARKERS) == "ab"


def test_reasoning_parser_text_delegates(tok):
    class FakeRP:
        end_token_id = THINK_END
        end_token = "</think>"

    ids = [1, THINK_END, 2]
    stripped = tok.decode(ids)  # "ab"
    res = _reasoning_parser_text(tok, FakeRP(), stripped, ids, True)
    assert res == "a</think>b"


def test_reasoning_parser_text_no_end_token_id(tok):
    class FakeRP:
        end_token_id = None
        end_token = None

    ids = [1, 2]
    assert _reasoning_parser_text(tok, FakeRP(), "ab", ids, True) == "ab"


# --- _reinsert_markers_by_ids: multi-marker / repeated-occurrence variant ----

TOOL_MARKERS = {
    TOOL_CALL_START: "<tool_call>",
    TOOL_CALL_END: "</tool_call>",
}


def test_by_ids_lone_start_token(tok):
    # A lone <tool_call> delta decodes to "" under skip_special_tokens; it must
    # come back so the string-based tool parser can enter its tool-call state.
    ids = [TOOL_CALL_START]
    stripped = tok.decode(ids)  # ""
    assert (
        _reinsert_markers_by_ids(tok, stripped, ids, True, TOOL_MARKERS)
        == "<tool_call>"
    )


def test_by_ids_fused_separator_between_calls(tok):
    # </tool_call> "\n" <tool_call> fused into one delta between parallel calls.
    ids = [TOOL_CALL_END, 4, TOOL_CALL_START]
    stripped = tok.decode(ids)  # "\n"
    assert (
        _reinsert_markers_by_ids(tok, stripped, ids, True, TOOL_MARKERS)
        == "</tool_call>\n<tool_call>"
    )


def test_by_ids_marker_with_surrounding_words(tok):
    ids = [1, TOOL_CALL_START, 2]
    stripped = tok.decode(ids)  # "ab"
    assert (
        _reinsert_markers_by_ids(tok, stripped, ids, True, TOOL_MARKERS)
        == "a<tool_call>b"
    )


def test_by_ids_no_marker_delta_unchanged(tok):
    # A function-body delta carries no wrapper token ids and must be untouched.
    ids = [1, 2, 3]
    stripped = tok.decode(ids)  # "abc"
    assert _reinsert_markers_by_ids(tok, stripped, ids, True, TOOL_MARKERS) == "abc"


def test_by_ids_skip_special_tokens_false_noop(tok):
    ids = [TOOL_CALL_START]
    assert (
        _reinsert_markers_by_ids(tok, "<tool_call>", ids, False, TOOL_MARKERS)
        == "<tool_call>"
    )


def test_by_ids_empty_token_ids_unchanged(tok):
    assert _reinsert_markers_by_ids(tok, "ab", [], True, TOOL_MARKERS) == "ab"


def test_by_ids_none_tokenizer_unchanged():
    assert (
        _reinsert_markers_by_ids(None, "ab", [TOOL_CALL_START], True, TOOL_MARKERS)
        == "ab"
    )


def test_by_ids_does_not_leak_other_special_tokens(tok):
    # An EOS in the same delta must stay stripped; only the tool wrappers return.
    ids = [1, TOOL_CALL_END, EOS]
    stripped = tok.decode(ids)  # "a"
    res = _reinsert_markers_by_ids(tok, stripped, ids, True, TOOL_MARKERS)
    assert res == "a</tool_call>"
    assert "<|im_end|>" not in res
