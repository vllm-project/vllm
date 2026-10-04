# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SLEM (String-Level Exact Match) speculative decoding."""

import pytest
import torch


class MockTokenizer:
    """Minimal tokenizer mock for testing text roundtrip logic."""

    def __init__(self, vocab: dict[str, int], eos_token_id: int = 0):
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.eos_token_id = eos_token_id
        self.unk_token_id = None
        self.vocab_size = max(vocab.values()) + 1

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        return "".join(self.id_to_token.get(i, "") for i in ids)

    def encode(self, text, add_special_tokens=False, **kwargs):
        ids = []
        i = 0
        while i < len(text):
            best_len = 0
            best_id = -1
            for token, tid in self.vocab.items():
                if text[i:].startswith(token) and len(token) > best_len:
                    best_len = len(token)
                    best_id = tid
            if best_len == 0:
                i += 1
                continue
            ids.append(best_id)
            i += best_len
        return ids

    def get_vocab(self):
        return dict(self.vocab)


@pytest.fixture
def bpe_tokenizer():
    """BPE-style tokenizer with fine-grained tokens."""
    return MockTokenizer(
        {
            "good": 1,
            " morn": 2,
            "ing": 3,
            "!": 4,
            "g": 5,
            "o": 6,
            "d": 7,
            " ": 8,
            "m": 9,
            "r": 10,
            "n": 11,
            "i": 12,
        },
        eos_token_id=0,
    )


@pytest.fixture
def sp_tokenizer():
    """SentencePiece-style tokenizer with coarser tokens."""
    return MockTokenizer(
        {
            "good morning": 1,
            "!": 2,
            "g": 3,
            "o": 4,
            "d": 5,
            " ": 6,
            "m": 7,
            "r": 8,
            "n": 9,
            "i": 10,
        },
        eos_token_id=0,
    )


class TestSlemMapDraftToTargetIds:
    """Tests for string-level draft→target mapping."""

    def test_basic_roundtrip(self, bpe_tokenizer, sp_tokenizer):
        from vllm.v1.spec_decode.slem import SlemMapper

        mapper = SlemMapper(
            target_tokenizer=sp_tokenizer,
            draft_tokenizer=bpe_tokenizer,
            device=torch.device("cpu"),
        )

        # "good" + " morn" + "ing" = "good morning" → target token [1]
        # With K=3, result is [1, eos, eos] (1 target token, padded)
        draft_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        target_ids = mapper.map_draft_to_target_ids(draft_ids)

        assert target_ids.shape == (1, 3)
        assert target_ids[0, 0].item() == 1  # "good morning"
        assert target_ids[0, 1].item() == mapper.target_eos_id  # padding
        assert target_ids[0, 2].item() == mapper.target_eos_id  # padding

    def test_more_target_tokens_than_draft(self, bpe_tokenizer, sp_tokenizer):
        """When target re-encoding produces more tokens, result is truncated."""
        from vllm.v1.spec_decode.slem import SlemMapper

        # Use sp as draft (coarse), bpe as target (fine)
        mapper = SlemMapper(
            target_tokenizer=bpe_tokenizer,
            draft_tokenizer=sp_tokenizer,
            device=torch.device("cpu"),
        )

        # "good morning" (1 draft token) → "good" + " morn" + "ing" (3 target tokens)
        # With K=1, only first target token fits
        draft_ids = torch.tensor([[1]], dtype=torch.long)
        target_ids = mapper.map_draft_to_target_ids(draft_ids)

        assert target_ids.shape == (1, 1)
        assert target_ids[0, 0].item() == 1  # "good" (truncated)

    def test_fewer_target_tokens_than_draft(self, bpe_tokenizer, sp_tokenizer):
        """When target re-encoding produces fewer tokens, result is padded."""
        from vllm.v1.spec_decode.slem import SlemMapper

        mapper = SlemMapper(
            target_tokenizer=sp_tokenizer,
            draft_tokenizer=bpe_tokenizer,
            device=torch.device("cpu"),
        )

        # 3 draft tokens → 1 target token, padded to width 3
        draft_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        target_ids = mapper.map_draft_to_target_ids(draft_ids)

        assert target_ids[0, 0].item() == 1
        assert target_ids[0, 1].item() == mapper.target_eos_id
        assert target_ids[0, 2].item() == mapper.target_eos_id

    def test_batch(self, bpe_tokenizer, sp_tokenizer):
        """Batch of sequences mapped correctly."""
        from vllm.v1.spec_decode.slem import SlemMapper

        mapper = SlemMapper(
            target_tokenizer=sp_tokenizer,
            draft_tokenizer=bpe_tokenizer,
            device=torch.device("cpu"),
        )

        # Batch: ["good morning", "!"]
        draft_ids = torch.tensor(
            [
                [1, 2, 3],  # "good" + " morn" + "ing" → "good morning" → [1]
                [4, 0, 0],  # "!" + eos + eos → "!" → [2]
            ],
            dtype=torch.long,
        )
        target_ids = mapper.map_draft_to_target_ids(draft_ids)

        assert target_ids.shape == (2, 3)
        assert target_ids[0, 0].item() == 1  # "good morning"
        assert target_ids[1, 0].item() == 2  # "!"

    def test_all_eos_input(self, bpe_tokenizer, sp_tokenizer):
        """All-EOS input produces all-EOS output."""
        from vllm.v1.spec_decode.slem import SlemMapper

        mapper = SlemMapper(
            target_tokenizer=sp_tokenizer,
            draft_tokenizer=bpe_tokenizer,
            device=torch.device("cpu"),
        )

        draft_ids = torch.tensor([[0, 0, 0]], dtype=torch.long)
        target_ids = mapper.map_draft_to_target_ids(draft_ids)

        assert (target_ids == mapper.target_eos_id).all()

    def test_text_preservation(self, bpe_tokenizer, sp_tokenizer):
        """The mapped target tokens decode to a prefix of the draft text."""
        from vllm.v1.spec_decode.slem import SlemMapper

        mapper = SlemMapper(
            target_tokenizer=sp_tokenizer,
            draft_tokenizer=bpe_tokenizer,
            device=torch.device("cpu"),
        )

        # "good morning!" = draft tokens [1, 2, 3, 4]
        draft_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        target_ids = mapper.map_draft_to_target_ids(draft_ids)

        # Get non-EOS target tokens
        non_eos_mask = target_ids[0] != mapper.target_eos_id
        active_target_ids = target_ids[0][non_eos_mask].tolist()

        draft_text = bpe_tokenizer.decode([1, 2, 3, 4])
        target_text = sp_tokenizer.decode(active_target_ids)

        # Target text must be a prefix of draft text (possibly truncated)
        assert draft_text.startswith(target_text)


class TestSlemMapTargetToDraftIds:
    """Tests for target→draft 1:1 lookup (used for context feeding)."""

    def test_1to1_mapping(self, bpe_tokenizer, sp_tokenizer):
        from vllm.v1.spec_decode.slem import SlemMapper

        mapper = SlemMapper(
            target_tokenizer=sp_tokenizer,
            draft_tokenizer=bpe_tokenizer,
            device=torch.device("cpu"),
        )

        # "!" maps 1:1 between vocabs: sp[2] ↔ bpe[4]
        target_ids = torch.tensor([2], dtype=torch.long)
        draft_ids = mapper.map_target_to_draft_ids(target_ids)
        assert draft_ids[0].item() == 4

    def test_no_mapping_falls_back_to_eos(self, bpe_tokenizer, sp_tokenizer):
        from vllm.v1.spec_decode.slem import SlemMapper

        mapper = SlemMapper(
            target_tokenizer=sp_tokenizer,
            draft_tokenizer=bpe_tokenizer,
            device=torch.device("cpu"),
        )

        # "good morning" (sp token 1) doesn't exist as single bpe token
        target_ids = torch.tensor([1], dtype=torch.long)
        draft_ids = mapper.map_target_to_draft_ids(target_ids)
        assert draft_ids[0].item() == mapper.draft_eos_id
