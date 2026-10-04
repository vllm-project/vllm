# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

current_platform.import_kernels()

VOCAB_SIZES = [32000, 49152, 128256]
BATCH_SIZES = [1, 4, 16]


def reference_greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1).view(-1)


class TestGreedyArgmax:
    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_exact_match(self, vocab_size: int, batch_size: int):
        logits = torch.randn(batch_size, vocab_size, dtype=torch.float32)
        expected = reference_greedy_sample(logits)
        result = torch.ops._C.greedy_argmax(logits)
        torch.testing.assert_close(result, expected)

    def test_single_dominant(self):
        logits = torch.full((1, 50000), -1e9, dtype=torch.float32)
        logits[0, 42] = 100.0
        assert torch.ops._C.greedy_argmax(logits).item() == 42

    def test_negative_logits(self):
        logits = torch.randn(8, 32000, dtype=torch.float32) - 10.0
        expected = reference_greedy_sample(logits)
        result = torch.ops._C.greedy_argmax(logits)
        torch.testing.assert_close(result, expected)


class TestBucketedRejectionSampling:
    def test_small_support_distribution(self):
        from scipy.stats import chisquare

        logits = torch.tensor([0.0, -0.2, -0.7, -1.5, -3.0, -float("inf")])
        n_samples = 20_000
        generator = torch.Generator().manual_seed(432)
        seeds = torch.empty(n_samples, dtype=torch.int64).random_(
            -(2**63), None, generator=generator
        )
        tokens = torch.ops._C.bucketed_rejection_sample(
            logits.expand(n_samples, -1), seeds
        )
        counts = torch.bincount(tokens, minlength=logits.numel()).double()
        probs = logits.double().softmax(-1)
        assert counts[-1] == 0
        _, p_value = chisquare(counts[:-1].numpy(), (probs[:-1] * n_samples).numpy())
        assert p_value > 1e-7

    def test_full_vocabulary_zipf_tail_distribution(self):
        """The old fixed table loses tail mass even with unlimited draws."""
        from scipy.stats import chisquare

        vocab_size, batch_size, n_samples = 151_936, 128, 16_384
        logits = (-1.1 * torch.arange(1, vocab_size + 1).double().log()).float()
        probs = logits.double().softmax(-1)
        counts = torch.zeros(vocab_size, dtype=torch.int64)
        generator = torch.Generator().manual_seed(59786)
        for _ in range(n_samples // batch_size):
            seeds = torch.empty(batch_size, dtype=torch.int64).random_(
                -(2**63), None, generator=generator
            )
            tokens = torch.ops._C.bucketed_rejection_sample(
                logits.expand(batch_size, -1), seeds
            )
            counts += torch.bincount(tokens, minlength=vocab_size)

        # Fixed rank groups have enough expected observations for chi-square;
        # individually rare tokens need not appear in a finite sample.
        observed = torch.stack([group.sum() for group in counts.tensor_split(32)])
        expected = torch.stack([group.sum() for group in probs.tensor_split(32)])
        expected *= n_samples
        assert expected.min() >= 10
        _, p_value = chisquare(observed.numpy(), expected.numpy())
        assert p_value > 1e-7, f"Zipf rank-group p-value: {p_value}"

        # This set has about 5% target mass, versus about 3.3% with the table.
        tail = probs < 2**-20
        probability = probs[tail].sum().item()
        tail_count = counts[tail].sum().item()
        sigma = (n_samples * probability * (1 - probability)) ** 0.5
        assert abs(tail_count - n_samples * probability) <= 6 * sigma, (
            f"tail frequency={tail_count / n_samples}, expected={probability}"
        )

    def test_strided_inputs_preserve_row_seed_determinism(self):
        generator = torch.Generator().manual_seed(723)
        logits = torch.randn(8, 514, generator=generator)[:, ::2]
        logits[:, 3::7] = -float("inf")
        seeds = torch.empty(16, dtype=torch.int64).random_(
            -(2**63), None, generator=generator
        )[::2]
        assert not logits.is_contiguous() and not seeds.is_contiguous()
        expected = torch.ops._C.bucketed_rejection_sample(
            logits.contiguous(), seeds.contiguous()
        )
        original_threads = torch.get_num_threads()
        try:
            for threads in (1, 4):
                torch.set_num_threads(threads)
                actual = torch.ops._C.bucketed_rejection_sample(logits, seeds)
                torch.testing.assert_close(actual, expected)
                order = torch.tensor([3, 7, 1, 6, 0, 5, 2, 4])
                actual = torch.ops._C.bucketed_rejection_sample(
                    logits[order], seeds[order]
                )
                torch.testing.assert_close(actual, expected[order])
        finally:
            torch.set_num_threads(original_threads)
        assert torch.isfinite(logits[torch.arange(8), expected]).all()

    def test_high_seed_bits_change_streams(self):
        logits = torch.zeros(128, 97, dtype=torch.float32)
        seeds = torch.arange(128, dtype=torch.int64) << 40
        tokens = torch.ops._C.bucketed_rejection_sample(logits, seeds)
        # All low bits match: the old table returns one token for every row.
        assert tokens.unique().numel() > 20
        negative_seeds = seeds ^ -(2**63)
        negative_tokens = torch.ops._C.bucketed_rejection_sample(logits, negative_seeds)
        assert (negative_seeds < 0).all()
        assert (tokens != negative_tokens).sum() > 100

    @pytest.mark.parametrize(
        "row", [[0.0, float("nan")], [0.0, float("inf")], [-float("inf")] * 2]
    )
    @pytest.mark.parametrize("batch_size", [1, 16])
    def test_invalid_rows_preserve_other_rows(self, row, batch_size):
        logits = torch.zeros(batch_size, 2)
        seeds = torch.arange(batch_size, dtype=torch.int64)
        expected = torch.ops._C.bucketed_rejection_sample(logits, seeds)
        invalid_row = batch_size // 2
        logits[invalid_row] = torch.tensor(row)
        expected[invalid_row] = -1
        original_threads = torch.get_num_threads()
        try:
            for threads in (1, 4):
                torch.set_num_threads(threads)
                actual = torch.ops._C.bucketed_rejection_sample(logits, seeds)
                torch.testing.assert_close(actual, expected)
                actual = torch.ops._C.bucketed_rejection_sample(
                    logits.flip(0), seeds.flip(0)
                )
                torch.testing.assert_close(actual, expected.flip(0))
        finally:
            torch.set_num_threads(original_threads)

    def test_empty_batch_and_single_unmasked_token(self):
        tokens = torch.ops._C.bucketed_rejection_sample(
            torch.empty(0, 7), torch.empty(0, dtype=torch.int64)
        )
        assert tokens.shape == (0,) and tokens.dtype == torch.int64
        logits = torch.full((4, 7), -float("inf"))
        logits[:, 3] = -torch.finfo(torch.float32).max
        tokens = torch.ops._C.bucketed_rejection_sample(
            logits, torch.tensor([0, -1, -(2**63), 2**63 - 1])
        )
        torch.testing.assert_close(tokens, torch.full((4,), 3))

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_output_in_range(self, vocab_size: int, batch_size: int):
        logits = torch.randn(batch_size, vocab_size, dtype=torch.float32)
        seeds = torch.arange(batch_size, dtype=torch.long)
        result = torch.ops._C.bucketed_rejection_sample(logits, seeds)
        assert result.min() >= 0
        assert result.max() < vocab_size
