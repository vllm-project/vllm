# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for score mode: SamplingParams validation, gather_target_logprobs,
and the fast-path prompt logprobs processor."""

import pytest
import torch

from vllm import SamplingParams
from vllm.logprobs import create_prompt_logprobs
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.sampler import Sampler

# ---------------------------------------------------------------------------
# 1. SamplingParams validation
# ---------------------------------------------------------------------------


class TestSamplingParamsValidation:
    def test_score_mode_requires_prompt_logprobs(self):
        with pytest.raises(ValueError, match="score_mode requires prompt_logprobs"):
            SamplingParams(score_mode=True, max_tokens=1)

    def test_score_mode_valid(self):
        params = SamplingParams(score_mode=True, prompt_logprobs=1, max_tokens=1)
        assert params.score_mode is True
        assert params.prompt_logprobs == 1

    def test_kld_mode_mutually_exclusive_with_return_prompt_logits(self):
        with pytest.raises(
            ValueError,
            match="return_prompt_logits and kld_mode are mutually exclusive",
        ):
            SamplingParams(
                return_prompt_logits=True,
                kld_mode=True,
                prompt_logprobs=1,
                max_tokens=1,
            )

    def test_return_prompt_logits_valid(self):
        params = SamplingParams(
            return_prompt_logits=True, prompt_logprobs=1, max_tokens=1
        )
        assert params.return_prompt_logits is True

    def test_kld_mode_valid(self):
        params = SamplingParams(kld_mode=True, prompt_logprobs=1, max_tokens=1)
        assert params.kld_mode is True


# ---------------------------------------------------------------------------
# 2. Sampler.gather_target_logprobs
# ---------------------------------------------------------------------------


def _make_logprobs_tensor(num_tokens: int, vocab_size: int) -> torch.Tensor:
    """Create a logprobs tensor from random logits."""
    logits = torch.randn(num_tokens, vocab_size)
    return Sampler.compute_logprobs(logits)


class TestGatherTargetLogprobs:
    def test_basic(self):
        vocab_size = 100
        num_tokens = 4
        logprobs = _make_logprobs_tensor(num_tokens, vocab_size)
        target_ids = torch.tensor([3, 42, 0, 99], dtype=torch.int64)

        result = Sampler.gather_target_logprobs(logprobs, target_ids)
        indices, lps, ranks, _ = result

        assert indices.shape == (num_tokens, 1)
        assert lps.shape == (num_tokens, 1)

        for i, tid in enumerate(target_ids.tolist()):
            assert indices[i, 0].item() == tid
            torch.testing.assert_close(
                lps[i, 0], logprobs[i, tid], atol=1e-6, rtol=1e-5
            )

    @pytest.mark.parametrize("num_tokens", [1, 8, 64])
    def test_output_shapes(self, num_tokens: int):
        vocab_size = 256
        logprobs = _make_logprobs_tensor(num_tokens, vocab_size)
        target_ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int64)

        indices, lps, ranks, _ = Sampler.gather_target_logprobs(logprobs, target_ids)

        assert indices.shape == (num_tokens, 1)
        assert lps.shape == (num_tokens, 1)
        assert ranks.shape == (num_tokens, 1)

    def test_output_dtypes(self):
        logprobs = _make_logprobs_tensor(4, 50)
        target_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int64)

        indices, lps, ranks, _ = Sampler.gather_target_logprobs(logprobs, target_ids)

        assert indices.dtype == torch.int32
        assert lps.dtype == torch.float32

    def test_rank_correctness(self):
        """The rank of the highest-logprob token should be 0."""
        logprobs = _make_logprobs_tensor(1, 50)
        best_token = logprobs[0].argmax().item()
        target_ids = torch.tensor([best_token], dtype=torch.int64)

        _, _, ranks, _ = Sampler.gather_target_logprobs(logprobs, target_ids)
        assert ranks[0, 0].item() == 0


# ---------------------------------------------------------------------------
# 3. LogprobsProcessor._update_prompt_logprobs_fast_path
# ---------------------------------------------------------------------------


class TestFastPathLogprobs:
    @staticmethod
    def _make_processor(
        target_token_ids: list[int],
    ):
        """Build a minimal LogprobsProcessor for fast-path testing."""
        from vllm.v1.engine.logprobs import LogprobsProcessor

        return LogprobsProcessor(
            tokenizer=None,
            logprobs=None,
            prompt_logprobs=create_prompt_logprobs(flat_logprobs=False),
            cumulative_logprob=None,
            num_logprobs=None,
            num_prompt_logprobs=1,
            target_token_ids=target_token_ids,
        )

    def test_produces_correct_logprobs(self):
        target_ids = [10, 20, 30]
        processor = self._make_processor(target_ids)

        token_ids_tensor = torch.tensor([[10], [20], [30]], dtype=torch.int32)
        logprobs_tensor = torch.tensor([[-1.5], [-2.0], [-0.5]], dtype=torch.float32)
        ranks_tensor = torch.tensor([[3], [7], [0]], dtype=torch.int64)

        tensors = LogprobsTensors(token_ids_tensor, logprobs_tensor, ranks_tensor)
        processor._update_prompt_logprobs_fast_path(tensors, target_ids)

        prompt_lps = processor.prompt_logprobs
        # First entry is always None (position 0 has no logprobs).
        assert prompt_lps[0] is None
        # Subsequent entries should have one logprob each.
        assert len(prompt_lps) == 4  # None + 3 positions
        for i, tid in enumerate(target_ids):
            entry = prompt_lps[i + 1]
            assert tid in entry
            assert entry[tid].logprob == logprobs_tensor[i, 0].item()

    def test_token_id_mismatch_raises(self):
        target_ids = [10, 20, 30]
        processor = self._make_processor(target_ids)

        wrong_ids_tensor = torch.tensor([[10], [99], [30]], dtype=torch.int32)
        logprobs_tensor = torch.tensor([[-1.0], [-1.0], [-1.0]], dtype=torch.float32)
        ranks_tensor = torch.tensor([[0], [0], [0]], dtype=torch.int64)

        tensors = LogprobsTensors(wrong_ids_tensor, logprobs_tensor, ranks_tensor)

        with pytest.raises(ValueError, match="Token ID mismatch at position 1"):
            processor._update_prompt_logprobs_fast_path(tensors, target_ids)
