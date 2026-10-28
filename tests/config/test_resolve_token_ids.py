# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for _resolve_token_ids in vllm.config.reasoning.

Verifies that control tokens registered as single vocabulary entries
(e.g. Mistral's ``[THINK]``) resolve to one token ID via
``convert_tokens_to_ids``, while multi-token markers like ``<think>``
fall back to ``tokenizer.encode``.
"""

from unittest.mock import MagicMock

from vllm.config.reasoning import _resolve_token_ids


def _make_tokenizer(vocab: dict[str, int], unk_token_id: int = 0):
    """Build a mock tokenizer with convert_tokens_to_ids and encode."""
    tok = MagicMock()
    tok.unk_token_id = unk_token_id

    def convert(token: str) -> int | None:
        return vocab.get(token)

    tok.convert_tokens_to_ids = MagicMock(side_effect=convert)

    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for ch in text:
            ids.append(vocab.get(ch, unk_token_id))
        return ids

    tok.encode = MagicMock(side_effect=encode)
    return tok


class TestResolveTokenIds:

    def test_control_token_returns_single_id(self):
        """A control token in the vocab resolves to a single ID."""
        tok = _make_tokenizer({"[THINK]": 34, "[/THINK]": 35})
        assert _resolve_token_ids(tok, "[THINK]") == [34]
        assert _resolve_token_ids(tok, "[/THINK]") == [35]
        tok.encode.assert_not_called()

    def test_unknown_marker_falls_back_to_encode(self):
        """A marker not in the vocab falls back to tokenizer.encode."""
        tok = _make_tokenizer({"<": 10, "t": 11, "h": 12, "i": 13,
                               "n": 14, "k": 15, ">": 16})
        ids = _resolve_token_ids(tok, "<think>")
        assert len(ids) > 1
        tok.encode.assert_called_once_with("<think>", add_special_tokens=False)

    def test_unk_id_falls_back_to_encode(self):
        """If convert_tokens_to_ids returns unk_token_id, fall back."""
        tok = _make_tokenizer({"<": 10, "t": 11}, unk_token_id=0)
        tok.convert_tokens_to_ids = MagicMock(return_value=0)
        _resolve_token_ids(tok, "<think>")
        tok.encode.assert_called_once()

    def test_none_falls_back_to_encode(self):
        """If convert_tokens_to_ids returns None, fall back."""
        tok = _make_tokenizer({"<": 10}, unk_token_id=0)
        tok.convert_tokens_to_ids = MagicMock(return_value=None)
        _resolve_token_ids(tok, "<think>")
        tok.encode.assert_called_once()
