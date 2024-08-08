"""Tests for rejection sampling."""
from typing import List, Tuple

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from vllm.model_executor.utils import set_random_seed

CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]


def mock_causal_accepted_tensor(
        k: int, last_accepted_indices: torch.Tensor) -> torch.Tensor:
    """Generate an "accepted" tensor which should yield causally-accepted tokens
    up to last accepted indices.

    Tokens after last_accepted_indices+1 may also be accepted, although they
    will not be causally accepted.
    """
    batch_size = last_accepted_indices.shape[0]

    accepted = (torch.arange(k).expand(batch_size, k) <=
                last_accepted_indices.unsqueeze(-1).broadcast_to(
                    batch_size, k))

    # Sprinkle accepted values after the contiguous initial accepted values.
    # This replicates the behavior of rejection sampling, which may "accept"
    # a token that cannot be accepted because of causality.
    sprinkle_candidates = (
        torch.arange(k).expand(batch_size, k) >
        last_accepted_indices.unsqueeze(-1).broadcast_to(batch_size, k) + 1)
    sprinkle = torch.rand(batch_size, k) > 0.5
    accepted[sprinkle_candidates] = sprinkle[sprinkle_candidates]
    return accepted


@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize(
    "which_tokens_accepted",
    ["all_tokens_accepted", "no_tokens_accepted", "some_tokens_accepted"])
@pytest.mark.parametrize("disable_bonus_tokens", [True, False])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_correct_output_format(which_tokens_accepted: str,
                               disable_bonus_tokens: bool, seed: int,
                               device: str):
    """Verify the output has correct format given predetermined accepted matrix.
    """
    set_random_seed(seed)
    torch.set_default_device(device)

    batch_size = 10
    k = 5
    vocab_size = 3000

    if which_tokens_accepted == "all_tokens_accepted":
        accepted = mock_causal_accepted_tensor(
            k, -1 + k * torch.ones((batch_size, ), dtype=torch.long))
    elif which_tokens_accepted == "no_tokens_accepted":
        accepted = mock_causal_accepted_tensor(
            k, -torch.ones((batch_size, ), dtype=torch.long))
    elif which_tokens_accepted == "some_tokens_accepted":
        last_accepted_indices = torch.randint(low=-1,
                                              high=k,
                                              size=(batch_size, ))
        accepted = mock_causal_accepted_tensor(k, last_accepted_indices)
    else:
        raise AssertionError()

    recovered_token_ids = torch.randint(low=0,
                                        high=vocab_size,
                                        size=(batch_size, k),
                                        dtype=torch.int64)
    draft_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, k),
                                    dtype=torch.int64)
    bonus_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, 1),
                                    dtype=torch.int64)

    rejection_sampler = RejectionSampler(
        disable_bonus_tokens=disable_bonus_tokens)
    rejection_sampler.init_gpu_tensors(device=device)
    output_token_ids = rejection_sampler._create_output(  # pylint: disable=protected-access
        accepted,
        recovered_token_ids,
        draft_token_ids,
        bonus_token_ids,
    )

    expected_bonus_token_ids = bonus_token_ids.clone()
    # If bonus tokens disabled. Verify they are set to -1.
    # See https://github.com/vllm-project/vllm/issues/4212
    if disable_bonus_tokens:
        expected_bonus_token_ids = expected_bonus_token_ids * 0 - 1

    if which_tokens_accepted == "all_tokens_accepted":
        # Expect all tokens to be equal to draft tokens.
        assert torch.equal(output_token_ids[:, :-1], draft_token_ids)

        # Expect all bonus tokens to be included.
        assert torch.equal(output_token_ids[:, -1:], expected_bonus_token_ids)
    elif which_tokens_accepted == "no_tokens_accepted":
        # Expect first token to be equal to recovered tokens.
        assert torch.equal(output_token_ids[:, 0], recovered_token_ids[:, 0])

        # Expect everything else to be -1.
        assert torch.equal(output_token_ids[:, 1:],
                           torch.ones_like(output_token_ids[:, 1:]) * -1)
    elif which_tokens_accepted == "some_tokens_accepted":
        recovered_plus_bonus = torch.cat(
            (recovered_token_ids, expected_bonus_token_ids), dim=-1)
        # Assert first rejected token is a recovered token or bonus token.
        assert torch.equal(
            recovered_plus_bonus[torch.arange(0, batch_size),
                                 last_accepted_indices + 1],
            output_token_ids[torch.arange(0, batch_size),
                             last_accepted_indices + 1])

        # Assert every subsequent token is -1.
        subsequent_mask = torch.arange(0, k + 1).expand(
            batch_size, k + 1) >= (last_accepted_indices + 2).unsqueeze(-1)
        assert torch.all(output_token_ids[subsequent_mask] == -1)


@pytest.mark.parametrize("k", list(range(1, 6)))
@pytest.mark.parametrize("vocab_size", [30_000, 50_000])
@pytest.mark.parametrize("batch_size", list(range(1, 32)))
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_no_crash_with_varying_dims(k: int, vocab_size: int, batch_size: int,
                                    device: str):
    torch.set_default_device(device)
    rejection_sampler = RejectionSampler()
    rejection_sampler.init_gpu_tensors(device=device)

    draft_probs = torch.rand(batch_size, k, vocab_size, dtype=torch.float32)
    target_probs = torch.rand(batch_size, k, vocab_size, dtype=torch.float32)
    bonus_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, 1),
                                    dtype=torch.int64)
    draft_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, k),
                                    dtype=torch.int64)

    rejection_sampler(target_probs, bonus_token_ids, draft_probs,
                      draft_token_ids)


@pytest.mark.parametrize("frac_seeded", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("k", [1, 3, 6])
@pytest.mark.parametrize("vocab_size", [30_000, 50_000])
@pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
@pytest.mark.parametrize("n_rep", [100])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_deterministic_when_seeded(k: int, vocab_size: int, batch_size: int,
                                   frac_seeded: float, n_rep: int,
                                   device: str):
    torch.set_default_device(device)
    rejection_sampler = RejectionSampler()
    rejection_sampler.init_gpu_tensors(device=device)

    draft_probs = torch.rand(batch_size, k, vocab_size, dtype=torch.float32)
    target_probs = torch.rand(batch_size, k, vocab_size, dtype=torch.float32)
    bonus_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, 1),
                                    dtype=torch.int64)
    draft_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, k),
                                    dtype=torch.int64)

    seeded_mask = torch.rand(batch_size, dtype=torch.float32) <= frac_seeded

    results = []
    for _ in range(n_rep):
        seeded_seqs = {
            i: torch.Generator(device=device).manual_seed(i)
            for i in range(batch_size) if seeded_mask[i]
        }
        results.append(
            rejection_sampler(target_probs, bonus_token_ids, draft_probs,
                              draft_token_ids, seeded_seqs))

    for i in range(batch_size):
        if seeded_mask[i]:
            for j in range(1, n_rep):
                assert torch.equal(results[j][i], results[0][i])


@pytest.mark.parametrize("above_or_below_vocab_range", ["above", "below"])
@pytest.mark.parametrize("which_token_ids",
                         ["bonus_token_ids", "draft_token_ids"])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_raises_when_vocab_oob(above_or_below_vocab_range: str,
                               which_token_ids: str, device: str):
    k = 3
    batch_size = 5
    vocab_size = 30_000
    torch.set_default_device(device)

    rejection_sampler = RejectionSampler(strict_mode=True)
    rejection_sampler.init_gpu_tensors(device=device)

    draft_probs = torch.rand(batch_size, k, vocab_size, dtype=torch.float32)
    target_probs = torch.rand(batch_size, k, vocab_size, dtype=torch.float32)
    bonus_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, 1),
                                    dtype=torch.int64)
    draft_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, k),
                                    dtype=torch.int64)

    oob_token_ids = None
    if which_token_ids == "bonus_token_ids":
        oob_token_ids = bonus_token_ids
    elif which_token_ids == "draft_token_ids":
        oob_token_ids = draft_token_ids
    else:
        raise AssertionError()

    if above_or_below_vocab_range == "above":
        rogue_token_id = vocab_size + 1
    elif above_or_below_vocab_range == "below":
        rogue_token_id = -1
    else:
        raise AssertionError()

    oob_token_ids[0][0] = rogue_token_id

    with pytest.raises(AssertionError):
        rejection_sampler(target_probs, bonus_token_ids, draft_probs,
                          draft_token_ids)


@pytest.mark.parametrize("draft_and_target_probs_equal", [True, False])
@pytest.mark.parametrize("seed", list(range(5)))
@torch.inference_mode()
def test_rejection_sampling_approximates_target_distribution(
        seed: int, draft_and_target_probs_equal: bool):
    """Verify rejection sampling approximates target distribution,
    despite sampling from a potentially distinct draft distribution.

    This is done by first creating a random target probability
    distribution and a random draft probability distribution. We then
    sample token ids from the rejection sampler using these draft
    and target distributions. The samples are used to estimate
    the output probability distribution, which we expect to approximate
    the target distribution.

    A basic distance metric is used to determine similarity between
    distributions.

    We expect that as we increase the number of samples,
    the distance between the observed distribution and the target
    distribution decreases. To measure this, we compare the distance
    of the observed distribution against both the target distribution
    and a uniform random distribution. We expect the distance between
    the observed distribution and the target distribution to improve
    much more than the distance improvement between the observed
    distribution and the random distribution.

    When draft_and_target_probs_equal=True, the draft and target
    probabilities are exactly equal. Rejection sampling should
    still work without any NaNs or exceptions.
    """
    torch.set_default_device("cpu")
    set_random_seed(seed)

    helper = _CorrectnessTestHelper(
        vocab_size=10,
        rejection_sampler=RejectionSampler(),
    )

    draft_probs, target_probs, reference_probs = helper.generate_probs_for_test(
        draft_and_target_probs_equal)

    sample_sizes = [10, 100, 1_000, 10_000, 100_000]
    distance_wrt_reference: List[float] = []
    distance_wrt_target: List[float] = []

    for num_samples in sample_sizes:
        (reference_vs_rejsample_dist,
         target_vs_rejsample_dist) = helper.run_and_compare_distributions(
             draft_probs,
             target_probs,
             reference_probs,
             num_samples,
         )

        distance_wrt_reference.append(reference_vs_rejsample_dist)
        distance_wrt_target.append(target_vs_rejsample_dist)

        relative_change_in_distance_wrt_target = get_ratio_first_to_last(
            distance_wrt_target)
        relative_change_in_distance_wrt_reference = get_ratio_first_to_last(
            distance_wrt_reference)

        print(f"{num_samples=} {target_vs_rejsample_dist=:.05f} "
              f"{reference_vs_rejsample_dist=:.05f}")
        print(f"{num_samples=} {relative_change_in_distance_wrt_target=:.02f} "
              f"{relative_change_in_distance_wrt_reference=:.02f}")

    relative_change_in_distance_wrt_target = get_ratio_first_to_last(
        distance_wrt_target)
    relative_change_in_distance_wrt_reference = get_ratio_first_to_last(
        distance_wrt_reference)

    expected_improvement_multiplier = 20
    assert (relative_change_in_distance_wrt_target >
            relative_change_in_distance_wrt_reference *
            expected_improvement_multiplier)


def get_ratio_first_to_last(elements: List[float]) -> float:
    return elements[0] / elements[-1]


class _CorrectnessTestHelper:
    """Class that packages together logic required for the unit-level
    rejection sampling correctness test.
    """

    def __init__(self, vocab_size: int, rejection_sampler: RejectionSampler):
        self.rejection_sampler = rejection_sampler
        self.vocab_size = vocab_size
        self.vocab_range = (0, vocab_size)

        self.rejection_sampler.init_gpu_tensors(device=0)

        # Keep test simple, use k=1
        self.k = 1

        # Bonus tokens not used, but rejection sampler requires
        # correct shape.
        self.num_bonus_tokens = 1

    def generate_probs_for_test(
        self, draft_and_target_probs_equal: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        draft_probs, target_probs = [
            F.softmax(
                torch.rand(self.vocab_size, dtype=torch.float32),
                dim=-1,
            ) for _ in range(2)
        ]

        num_reference_probs = 100
        reference_probs = F.softmax(
            torch.rand(num_reference_probs,
                       self.vocab_size,
                       dtype=torch.float32),
            dim=-1,
        )

        if draft_and_target_probs_equal:
            target_probs = draft_probs.clone()

        return draft_probs, target_probs, reference_probs

    def run_and_compare_distributions(self, draft_probs: torch.Tensor,
                                      target_probs: torch.Tensor,
                                      reference_probs: torch.Tensor,
                                      num_samples: int) -> Tuple[float, float]:
        # Sample using rejection sampling.
        rej_sample_probs = self._estimate_rejection_sampling_pdf(
            draft_probs, target_probs, num_samples)

        # Average distance from reference probs.
        reference_vs_rejsample_dist = torch.dist(
            reference_probs,
            rej_sample_probs).item() / reference_probs.shape[0]
        target_vs_rejsample_dist = torch.dist(target_probs,
                                              rej_sample_probs).item()

        return reference_vs_rejsample_dist, target_vs_rejsample_dist

    def _estimate_rejection_sampling_pdf(
        self,
        draft_probs: torch.Tensor,
        target_probs: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        # Repeat draft probs num_samples times.
        draft_probs = draft_probs.reshape(1, self.k, self.vocab_size).repeat(
            num_samples, 1, 1)

        # Repeat target probs num_samples * k times.
        # Rejection sampler requires bonus token probs, but they aren't used.
        target_probs = target_probs.reshape(1, 1, self.vocab_size).repeat(
            num_samples, self.k, 1)

        # Randomly sample draft token ids from draft probs.
        draft_token_ids = torch.multinomial(draft_probs[:, 0, :],
                                            num_samples=1,
                                            replacement=True).reshape(
                                                num_samples, self.k)

        # Bonus tokens not used but required.
        bonus_token_ids = torch.zeros((1, self.num_bonus_tokens),
                                      dtype=torch.int64,
                                      device="cuda").repeat(num_samples, 1)

        # Get output tokens via rejection sampling.
        output_token_ids = self.rejection_sampler(target_probs.to("cuda"),
                                                  bonus_token_ids.to("cuda"),
                                                  draft_probs.to("cuda"),
                                                  draft_token_ids.to("cuda"))

        # Remove bonus tokens
        output_token_ids = output_token_ids[:, :-1].flatten()

        # Estimate probability density function
        hist = torch.histogram(output_token_ids.to(dtype=torch.float,
                                                   device="cpu"),
                               bins=self.vocab_size,
                               range=self.vocab_range,
                               density=True)

        return hist.hist
