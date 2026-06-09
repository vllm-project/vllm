# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for VocabMapping (TLI vocabulary intersection)."""

import pytest
import torch

from vllm.v1.spec_decode.vocab_mapping import (
    VocabMapping,
    _detect_space_prefix,
    _normalize_token,
)


class MockTokenizer:
    def __init__(
        self,
        vocab,
        unk_token_id=None,
        eos_token_id=0,
        encode_fn=None,
    ):
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.unk_token_id = unk_token_id
        self.eos_token_id = eos_token_id
        self._encode_fn = encode_fn

    def get_vocab(self):
        return dict(self.vocab)

    def encode(self, text, add_special_tokens=False):
        if self._encode_fn is not None:
            return self._encode_fn(text)
        return []

    def convert_ids_to_tokens(self, token_id):
        return self.id_to_token.get(token_id, "")


class TestDetectSpacePrefix:
    def test_bpe_style(self):
        tokenizer = MockTokenizer(
            {"Ġa": 1, "hello": 2},
            encode_fn=lambda text: [1] if text == " a" else [],
        )
        result = _detect_space_prefix(tokenizer)
        assert result == ("Ġ",)

    def test_fallback(self):
        tokenizer = MockTokenizer(
            {"a": 1},
            encode_fn=lambda text: [1] if text == " a" else [],
        )
        result = _detect_space_prefix(tokenizer)
        assert result == ("Ġ", "▁")


class TestNormalizeToken:
    def test_bpe_prefix(self):
        assert _normalize_token("Ġhello", ("Ġ",)) == " hello"

    def test_sp_prefix(self):
        assert _normalize_token("▁hello", ("▁",)) == " hello"

    def test_no_prefix(self):
        assert _normalize_token("hello", ("Ġ",)) == "hello"


class TestVocabMapping:
    @pytest.fixture
    def mapping(self):
        target_tok = MockTokenizer(
            {
                " hello": 0,
                " world": 1,
                " foo": 2,
                " bar": 3,
            },
            unk_token_id=99,
            eos_token_id=0,
        )
        draft_tok = MockTokenizer(
            {
                " hello": 0,
                " world": 1,
                " baz": 2,
                " bar": 3,
            },
            unk_token_id=88,
            eos_token_id=0,
        )

        import vllm.v1.spec_decode.vocab_mapping as vm

        orig = vm._detect_space_prefix
        vm._detect_space_prefix = lambda tok: ()  # type: ignore[assignment]
        m = VocabMapping(target_tok, draft_tok, 4, 4, torch.device("cpu"))
        vm._detect_space_prefix = orig  # type: ignore[assignment]
        return m

    def test_intersection_size(self, mapping):
        assert mapping.intersection_size == 3

    def test_map_draft_to_target(self, mapping):
        draft_ids = torch.tensor([0, 1, 3], dtype=torch.long)
        target_ids = mapping.map_draft_to_target_ids(draft_ids)
        assert target_ids.tolist() == [0, 1, 3]

    def test_map_draft_to_target_unmapped(self, mapping):
        draft_ids = torch.tensor([2], dtype=torch.long)
        target_ids = mapping.map_draft_to_target_ids(draft_ids)
        assert target_ids[0].item() == 99

    def test_map_target_to_draft(self, mapping):
        target_ids = torch.tensor([0, 1, 3], dtype=torch.long)
        draft_ids = mapping.map_target_to_draft_ids(target_ids)
        assert draft_ids.tolist() == [0, 1, 3]

    def test_map_target_to_draft_unmapped(self, mapping):
        target_ids = torch.tensor([2], dtype=torch.long)
        draft_ids = mapping.map_target_to_draft_ids(target_ids)
        assert draft_ids[0].item() == 88

    def test_constrain_draft_logits(self, mapping):
        logits = torch.ones(1, 4)
        constrained = mapping.constrain_draft_logits(logits)
        assert constrained[0, 0].item() == 1.0
        assert constrained[0, 1].item() == 1.0
        assert constrained[0, 2].item() == float("-inf")
        assert constrained[0, 3].item() == 1.0
