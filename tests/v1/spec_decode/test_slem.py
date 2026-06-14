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


class TestSlemDraftToTargetCandidates:
    def test_basic_roundtrip(self, bpe_tokenizer, sp_tokenizer):
        from vllm.v1.spec_decode.slem import SlemMapper

        mapper = SlemMapper(
            target_tokenizer=sp_tokenizer,
            draft_tokenizer=bpe_tokenizer,
            device=torch.device("cpu"),
        )

        # "good" + " morn" + "ing" = "good morning"
        draft_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        num_draft = torch.tensor([3], dtype=torch.long)

        target_ids, num_tokens = mapper.draft_to_target_candidates(draft_ids, num_draft)

        # "good morning" → ["good morning"] = [1]
        assert num_tokens[0].item() == 1
        assert target_ids[0, 0].item() == 1

    def test_fewer_target_tokens_than_draft(self, bpe_tokenizer, sp_tokenizer):
        from vllm.v1.spec_decode.slem import SlemMapper

        mapper = SlemMapper(
            target_tokenizer=sp_tokenizer,
            draft_tokenizer=bpe_tokenizer,
            device=torch.device("cpu"),
        )

        # 3 draft tokens → 1 target token
        draft_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        num_draft = torch.tensor([3], dtype=torch.long)
        _, num_tokens = mapper.draft_to_target_candidates(draft_ids, num_draft)
        assert num_tokens[0].item() == 1


class TestSlemVerifyAndAccept:
    def _make_mapper(self):
        from vllm.v1.spec_decode.slem import SlemMapper

        mapper = SlemMapper.__new__(SlemMapper)
        mapper.target_eos_id = 0
        mapper.device = torch.device("cpu")
        return mapper

    def test_all_match(self):
        mapper = self._make_mapper()
        candidates = torch.tensor([[10, 20, 30]], dtype=torch.long)
        sampled = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        num_tokens = torch.tensor([3], dtype=torch.long)

        accepted, lens = mapper.verify_and_accept(candidates, sampled, num_tokens)

        assert lens[0].item() == 4  # 3 matched + 1 bonus
        assert accepted[0, :4].tolist() == [10, 20, 30, 40]

    def test_first_mismatch(self):
        mapper = self._make_mapper()
        candidates = torch.tensor([[10, 20, 30]], dtype=torch.long)
        sampled = torch.tensor([[99, 20, 30, 40]], dtype=torch.long)
        num_tokens = torch.tensor([3], dtype=torch.long)

        accepted, lens = mapper.verify_and_accept(candidates, sampled, num_tokens)

        assert lens[0].item() == 1  # only bonus
        assert accepted[0, 0].item() == 99

    def test_middle_mismatch(self):
        mapper = self._make_mapper()
        candidates = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        sampled = torch.tensor([[10, 20, 99, 40, 50]], dtype=torch.long)
        num_tokens = torch.tensor([4], dtype=torch.long)

        accepted, lens = mapper.verify_and_accept(candidates, sampled, num_tokens)

        assert lens[0].item() == 3  # 2 matched + bonus
        assert accepted[0, :3].tolist() == [10, 20, 99]

    def test_batch_mixed(self):
        mapper = self._make_mapper()
        candidates = torch.tensor(
            [
                [10, 20, 30],
                [10, 20, 30],
            ],
            dtype=torch.long,
        )
        sampled = torch.tensor(
            [
                [10, 20, 30, 40],  # all match -> 4 accepted
                [10, 99, 30, 40],  # mismatch at pos 1 -> 2 accepted
            ],
            dtype=torch.long,
        )
        num_tokens = torch.tensor([3, 3], dtype=torch.long)

        accepted, lens = mapper.verify_and_accept(candidates, sampled, num_tokens)

        assert lens[0].item() == 4
        assert lens[1].item() == 2
        assert accepted[1, :2].tolist() == [10, 99]

    def test_empty_candidates(self):
        mapper = self._make_mapper()
        candidates = torch.zeros((1, 0), dtype=torch.long)
        sampled = torch.tensor([[42]], dtype=torch.long)
        num_tokens = torch.tensor([0], dtype=torch.long)

        accepted, lens = mapper.verify_and_accept(candidates, sampled, num_tokens)

        assert lens[0].item() == 1
        assert accepted[0, 0].item() == 42

    def test_single_token_match(self):
        mapper = self._make_mapper()
        candidates = torch.tensor([[10]], dtype=torch.long)
        sampled = torch.tensor([[10, 55]], dtype=torch.long)
        num_tokens = torch.tensor([1], dtype=torch.long)

        accepted, lens = mapper.verify_and_accept(candidates, sampled, num_tokens)

        assert lens[0].item() == 2
        assert accepted[0, :2].tolist() == [10, 55]


class TestSlemTargetToDraftInput:
    def test_roundtrip_back_to_draft(self, bpe_tokenizer, sp_tokenizer):
        from vllm.v1.spec_decode.slem import SlemMapper

        mapper = SlemMapper(
            target_tokenizer=sp_tokenizer,
            draft_tokenizer=bpe_tokenizer,
            device=torch.device("cpu"),
        )

        # Accepted: ["good morning"] = [1] in target vocab
        accepted = torch.tensor([[1]], dtype=torch.long)
        lens = torch.tensor([1], dtype=torch.long)

        draft_ids, num_ids = mapper.target_to_draft_input(accepted, lens)

        # "good morning" re-encoded with BPE -> ["good", " morn", "ing"] = [1, 2, 3]
        assert draft_ids[0, :3].tolist() == [1, 2, 3]
        assert num_ids[0].item() == 3
