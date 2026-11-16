# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for _resolve_token_ids in vllm.config.reasoning.

Verifies that, for Mistral tokenizers, markers that are special tokens
resolve to one token ID via ``convert_tokens_to_ids``, while other
markers, such as ``<think>`` and ordinary vocabulary pieces, fall back to
``tokenizer.encode``, as every marker does for other tokenizers.
"""

from unittest.mock import MagicMock

import pytest

from vllm.config import reasoning
from vllm.config.reasoning import _resolve_token_ids


def _make_tokenizer(
    vocab: dict[str, int],
    special_ids: set[int] | None = None,
    unk_id: int | None = None,
):
    """Build a mock tokenizer with convert_tokens_to_ids and encode.

    As in Mistral tokenizers, ``convert_tokens_to_ids`` maps a string that
    is not in ``vocab`` to ``unk_id``.
    """
    special_ids = special_ids or set()
    tok = MagicMock(
        spec=[
            "all_special_tokens",
            "all_special_ids",
            "convert_tokens_to_ids",
            "encode",
        ]
    )
    tok.all_special_ids = sorted(special_ids)
    tok.all_special_tokens = [t for t, i in vocab.items() if i in special_ids]

    def convert(token: str) -> int | None:
        return vocab.get(token, unk_id)

    tok.convert_tokens_to_ids = MagicMock(side_effect=convert)

    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        return [vocab.get(ch, unk_id) for ch in text]

    tok.encode = MagicMock(side_effect=encode)
    return tok


_THINK_CHARS = {"<": 10, "t": 11, "h": 12, "i": 13, "n": 14, "k": 15, ">": 16}


@pytest.fixture
def mistral_like(monkeypatch):
    """Treat the mock tokenizers as Mistral tokenizers."""
    monkeypatch.setattr(reasoning, "is_mistral_tokenizer", lambda tokenizer: True)


@pytest.mark.usefixtures("mistral_like")
class TestResolveTokenIdsMistral:
    def test_control_token_returns_single_id(self):
        """A special control token in the vocab resolves to a single ID."""
        tok = _make_tokenizer(
            {"[THINK]": 34, "[/THINK]": 35},
            special_ids={34, 35},
        )
        assert _resolve_token_ids(tok, "[THINK]") == [34]
        assert _resolve_token_ids(tok, "[/THINK]") == [35]
        tok.encode.assert_not_called()

    def test_unknown_marker_falls_back_to_encode(self):
        """A marker outside the vocab is encoded, although
        convert_tokens_to_ids maps it to the unknown token, which Tekken
        tokenizers count as special."""
        tok = _make_tokenizer({"<unk>": 0, **_THINK_CHARS}, special_ids={0}, unk_id=0)
        assert tok.convert_tokens_to_ids("<think>") in tok.all_special_ids
        assert _resolve_token_ids(tok, "<think>") == [10, 11, 12, 13, 14, 15, 16]
        tok.encode.assert_called_once_with("<think>", add_special_tokens=False)

    def test_ordinary_vocab_token_falls_back_to_encode(self):
        """An ordinary vocab token (not special) must fall back to encode."""
        tok = _make_tokenizer(
            {"Hello": 16998, "H": 72, "e": 101, "l": 108, "o": 111},
        )
        ids = _resolve_token_ids(tok, "Hello")
        tok.encode.assert_called_once_with("Hello", add_special_tokens=False)
        assert ids == [72, 101, 108, 108, 111]


def test_other_tokenizers_encode_every_marker():
    """Tokenizers other than Mistral's encode special markers themselves."""
    tok = _make_tokenizer(
        {"[THINK]": 34, **{c: ord(c) for c in "[THINK]"}}, special_ids={34}
    )
    assert _resolve_token_ids(tok, "[THINK]") == [ord(c) for c in "[THINK]"]
    tok.convert_tokens_to_ids.assert_not_called()


@pytest.fixture(scope="module")
def tekken_tokenizer():
    from vllm.tokenizers.mistral import MistralTokenizer

    try:
        return MistralTokenizer.from_pretrained("mistralai/Magistral-Small-2509")
    except Exception:
        pytest.skip("Magistral tokenizer unavailable")


def test_mistral_tekken_tokenizer(tekken_tokenizer):
    """Mistral's control tokens resolve to their IDs, while other markers
    keep their encode() IDs rather than the unknown token's."""
    tok = tekken_tokenizer
    assert _resolve_token_ids(tok, "[THINK]") == [34]
    assert _resolve_token_ids(tok, "[/THINK]") == [35]
    # Tekken maps strings outside its vocabulary to a special unknown token.
    assert tok.convert_tokens_to_ids("<think>") in tok.all_special_ids
    for marker in ("<think>", "</think>", "Hello"):
        assert _resolve_token_ids(tok, marker) == tok.encode(
            marker, add_special_tokens=False
        )
