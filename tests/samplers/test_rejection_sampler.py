# SPDX-License-Identifier: Apache-2.0
"""Tests for rejection sampling."""

import random
from typing import Any, Optional

import pytest
import torch
import torch.nn.functional as F

import vllm.model_executor.layers.rejection_sampler as v0_rej_sampler
import vllm.v1.sample.rejection_sampler as v1_rej_sampler
from vllm.model_executor.utils import set_random_seed

CUDA_DEVICES = ["cuda:0"]
TEST_KS = [1, 3, 5]
TEST_BATCH_SIZES = [1, 4, 8, 32]
TEST_VOCAB_SIZES = [30_000, 50_000]


@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize(
    "which_tokens_accepted",
    ["all_tokens_accepted", "no_tokens_accepted", "some_tokens_accepted"])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("use_flashinfer", [True, False])
@torch.inference_mode()
def test_correct_output_format(which_tokens_accepted: str, seed: int,
                               device: str, use_flashinfer: bool):
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

    rejection_sampler = get_sampler(use_v1=False,
                                    use_flashinfer=use_flashinfer,
                                    device=device)
    output_token_ids = rejection_sampler._create_output(  # pylint: disable=protected-access
        accepted,
        recovered_token_ids,
        draft_token_ids,
        bonus_token_ids,
    )

    expected_bonus_token_ids = bonus_token_ids.clone()

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


@pytest.mark.parametrize("k", TEST_KS)
@pytest.mark.parametrize("vocab_size", TEST_VOCAB_SIZES)
@pytest.mark.parametrize("batch_size", TEST_BATCH_SIZES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("use_flashinfer", [True, False])
@pytest.mark.parametrize("use_v1", [True, False])
@torch.inference_mode()
def test_no_crash_with_varying_dims(k: int, vocab_size: int, batch_size: int,
                                    device: str, use_flashinfer: bool,
                                    use_v1: bool):
    torch.set_default_device(device)
    rejection_sampler = get_sampler(use_v1, use_flashinfer, device)

    draft_probs = torch.rand(batch_size, k, vocab_size, dtype=torch.float32)
    target_probs = torch.rand(batch_size,
                              k + 1,
                              vocab_size,
                              dtype=torch.float32)
    if use_v1:
        if use_flashinfer:
            pytest.skip("Flashinfer will cause illegal memory access"
                        "skip before fixing.")
        draft_token_ids = [[
            random.randint(0, vocab_size - 1) for _ in range(k)
        ] for _ in range(batch_size)]
        rejection_sampler(draft_token_ids, draft_probs, target_probs,
                          create_v1_sampling_metadata(all_greedy=False))
    else:
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
@pytest.mark.parametrize("k", TEST_KS)
@pytest.mark.parametrize("vocab_size", TEST_VOCAB_SIZES)
@pytest.mark.parametrize("batch_size", TEST_BATCH_SIZES)
@pytest.mark.parametrize("n_rep", [100])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("use_flashinfer", [False])
@pytest.mark.parametrize("use_v1", [True])
@torch.inference_mode()
def test_deterministic_when_seeded(k: int, vocab_size: int, batch_size: int,
                                   frac_seeded: float, n_rep: int, device: str,
                                   use_flashinfer: bool, use_v1: bool):
    '''
    Test that the rejection sampler generates the same output when seeded.
    '''
    if use_flashinfer and use_v1:
        pytest.skip("Flashinfer will cause illegal memory access"
                    "skip before fixing for v1.")

    torch.set_default_device(device)
    rejection_sampler = get_sampler(use_v1, use_flashinfer, device)

    draft_probs = torch.rand(batch_size, k, vocab_size, dtype=torch.float32)
    target_probs = torch.rand(batch_size,
                              k + 1,
                              vocab_size,
                              dtype=torch.float32)
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
        if use_v1:
            sampling_metadata = create_v1_sampling_metadata(
                all_greedy=False, generators=seeded_seqs)
            rep_result = rejection_sampler(draft_token_ids.tolist(),
                                           draft_probs, target_probs,
                                           sampling_metadata).sampled_token_ids
        else:
            rep_result = rejection_sampler(target_probs, bonus_token_ids,
                                           draft_probs, draft_token_ids,
                                           seeded_seqs)
        results.append(rep_result)

    for i in range(batch_size):
        if seeded_mask[i]:
            for j in range(1, n_rep):
                assert torch.equal(results[j][i], results[0][i])


@pytest.mark.parametrize("k", TEST_KS)
@pytest.mark.parametrize("vocab_size", TEST_VOCAB_SIZES)
@pytest.mark.parametrize("batch_size", TEST_BATCH_SIZES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("use_flashinfer", [True, False])
@pytest.mark.parametrize("use_v1", [True, False])
@torch.inference_mode()
def test_mixed_seeded_batch(k: int, vocab_size: int, batch_size: int,
                            device: str, use_flashinfer: bool, use_v1: bool):
    if use_flashinfer and use_v1:
        pytest.skip("Flashinfer will cause illegal memory access"
                    "skip before fixing for v1.")
    torch.set_default_device(device)
    set_random_seed(0)
    draft_probs = torch.rand(batch_size, k, vocab_size, dtype=torch.float32)
    target_probs = torch.rand(batch_size,
                              k + 1,
                              vocab_size,
                              dtype=torch.float32)
    bonus_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, 1),
                                    dtype=torch.int64)
    draft_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, k),
                                    dtype=torch.int64)

    single_batches = []
    for i in range(batch_size):
        single_batches.append((draft_probs[i].clone().unsqueeze(0),
                               draft_token_ids[i].clone().unsqueeze(0),
                               target_probs[i].clone().unsqueeze(0),
                               bonus_token_ids[i].clone().unsqueeze(0),
                               draft_token_ids[i].clone().unsqueeze(0)))

    set_random_seed(0)
    rejection_sampler = get_sampler(use_v1, use_flashinfer, device)

    results = []
    seeded_seqs = {
        i: torch.Generator(device=device).manual_seed(i)
        for i in range(1, batch_size)  # 0 is seed None
    }
    if use_v1:
        sampling_metadata = create_v1_sampling_metadata(all_greedy=False,
                                                        generators=seeded_seqs)
        batch_result = rejection_sampler(draft_token_ids.tolist(),
                                         draft_probs.clone(),
                                         target_probs.clone(),
                                         sampling_metadata).sampled_token_ids
    else:
        batch_result = rejection_sampler(target_probs.clone(),
                                         bonus_token_ids.clone(),
                                         draft_probs.clone(),
                                         draft_token_ids.clone(), seeded_seqs)

    set_random_seed(0)
    rejection_sampler = get_sampler(use_v1, use_flashinfer, device)
    for i in range(batch_size):
        request_seeded_seqs = {
            0: torch.Generator(device=device).manual_seed(i)
        } if seeded_seqs.get(i) is not None else None
        (draft_probs, draft_token_ids, target_probs, bonus_token_ids,
         draft_token_ids) = single_batches[i]
        if use_v1:
            request_seeded_seqs = request_seeded_seqs or {}
            sampling_metadata = create_v1_sampling_metadata(
                all_greedy=False, generators=request_seeded_seqs)
            results.append(
                rejection_sampler(draft_token_ids.tolist(), draft_probs,
                                  target_probs,
                                  sampling_metadata).sampled_token_ids)
        else:
            results.append(
                rejection_sampler(target_probs, bonus_token_ids, draft_probs,
                                  draft_token_ids, request_seeded_seqs))
    for i in range(batch_size):
        assert torch.equal(batch_result[i], results[i].squeeze(0))


@pytest.mark.parametrize("k", TEST_KS)
@pytest.mark.parametrize("vocab_size", TEST_VOCAB_SIZES)
@pytest.mark.parametrize("batch_size", TEST_BATCH_SIZES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_compare_nonflashinfer_backend(k: int, vocab_size: int,
                                       batch_size: int, device: str):
    """
    Test the flashinfer and nonflashinfer backend generate 
    the same output metrics.
    """
    torch.set_default_device(device)
    torch.manual_seed(0)
    draft_probs = torch.rand(batch_size, k, vocab_size, dtype=torch.float32)
    target_probs = torch.rand(batch_size,
                              k + 1,
                              vocab_size,
                              dtype=torch.float32)
    bonus_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, 1),
                                    dtype=torch.int64)
    draft_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, k),
                                    dtype=torch.int64)

    num_accepted_tokens = []
    num_emitted_tokens = []
    num_draft_tokens = []

    def get_seeded_seqs():
        return {
            i: torch.Generator(device=device).manual_seed(i)
            for i in range(batch_size)
        }

    for use_flashinfer in [True, False]:
        rejection_sampler = get_sampler(use_v1=False,
                                        use_flashinfer=use_flashinfer,
                                        device=device)
        # We use seeded sequences to ensure the same tokens are accepted
        # for both flashinfer and nonflashinfer backends.
        seeded_seqs = get_seeded_seqs()
        rejection_sampler(target_probs, bonus_token_ids, draft_probs,
                          draft_token_ids, seeded_seqs)
        num_accepted_tokens.append(rejection_sampler.num_accepted_tokens)
        num_emitted_tokens.append(rejection_sampler.num_emitted_tokens)
        num_draft_tokens.append(rejection_sampler.num_draft_tokens)

    assert num_accepted_tokens[0] == num_accepted_tokens[1]
    assert num_emitted_tokens[0] == num_emitted_tokens[1]
    assert num_draft_tokens[0] == num_draft_tokens[1]


@pytest.mark.parametrize("above_or_below_vocab_range", ["above", "below"])
@pytest.mark.parametrize("which_token_ids",
                         ["bonus_token_ids", "draft_token_ids"])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("use_flashinfer", [True, False])
@torch.inference_mode()
def test_raises_when_vocab_oob(above_or_below_vocab_range: str,
                               which_token_ids: str, device: str,
                               use_flashinfer: bool):
    k = 3
    batch_size = 5
    vocab_size = 30_000
    torch.set_default_device(device)

    rejection_sampler = get_sampler(use_v1=False,
                                    use_flashinfer=use_flashinfer,
                                    device=device,
                                    strict_mode=True)

    draft_probs = torch.rand(batch_size, k, vocab_size, dtype=torch.float32)
    target_probs = torch.rand(batch_size,
                              k + 1,
                              vocab_size,
                              dtype=torch.float32)
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
@pytest.mark.parametrize("seed", [1, 3, 5])
@pytest.mark.parametrize("use_v1", [True, False])
@torch.inference_mode()
def test_rejection_sampling_approximates_target_distribution(
        seed: int, draft_and_target_probs_equal: bool, use_v1: bool):
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
    device = CUDA_DEVICES[0]
    torch.set_default_device(device)
    set_random_seed(seed)
    # Rejection sampler has to be on GPU because all metrics recorded
    # in the rejection sampler are on GPU.
    helper = _CorrectnessTestHelper(
        vocab_size=10,
        use_v1=use_v1,
        device=device,
    )

    draft_probs, target_probs, reference_probs = helper.generate_probs_for_test(
        draft_and_target_probs_equal)

    sample_sizes = [10, 100, 1_000, 10_000, 100_000]
    distance_wrt_reference: list[float] = []
    distance_wrt_target: list[float] = []

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
    assert (relative_change_in_distance_wrt_target
            > relative_change_in_distance_wrt_reference *
            expected_improvement_multiplier)


def get_ratio_first_to_last(elements: list[float]) -> float:
    return elements[0] / elements[-1]


class _CorrectnessTestHelper:
    """Class that packages together logic required for the unit-level
    rejection sampling correctness test.
    """

    def __init__(self, vocab_size: int, use_v1: bool, device: str):
        self.rejection_sampler = get_sampler(use_v1,
                                             use_flashinfer=False,
                                             device=device)
        self.device = device
        self.vocab_size = vocab_size
        self.vocab_range = (0, vocab_size)
        self.use_v1 = use_v1

        # Keep test simple, use k=1
        self.k = 1

        # Bonus tokens not used, but rejection sampler requires
        # correct shape.
        self.num_bonus_tokens = 1

    def generate_probs_for_test(
        self, draft_and_target_probs_equal: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        draft_probs, target_probs = (F.softmax(
            torch.rand(self.vocab_size, dtype=torch.float32),
            dim=-1,
        ) for _ in range(2))

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
                                      num_samples: int) -> tuple[float, float]:
        # Sample using rejection sampling.
        rej_sample_probs = self._estimate_rejection_sampling_pdf(
            draft_probs, target_probs, num_samples)
        rej_sample_probs = rej_sample_probs.to(self.device)

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

        # Repeat target probs num_samples * (k + 1) times.
        # Rejection sampler requires bonus token probs, but they aren't used.
        target_probs = target_probs.reshape(1, 1, self.vocab_size).repeat(
            num_samples, self.k + 1, 1)

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
        if self.use_v1:
            draft_token_ids_list = draft_token_ids.tolist()
            output_token_ids = self.rejection_sampler(
                draft_token_ids_list, draft_probs, target_probs,
                create_v1_sampling_metadata(
                    all_greedy=False)).sampled_token_ids
        else:
            output_token_ids = self.rejection_sampler(
                target_probs.to("cuda"), bonus_token_ids.to("cuda"),
                draft_probs.to("cuda"), draft_token_ids.to("cuda"))

        # Remove bonus tokens
        output_token_ids = output_token_ids[:, :-1].flatten()

        # Estimate probability density function
        # torch.histogram can only be used with CUDA backend.
        # Therefore, we move the output token ids to CPU.
        hist = torch.histogram(output_token_ids.to(dtype=torch.float,
                                                   device="cpu"),
                               bins=self.vocab_size,
                               range=self.vocab_range,
                               density=True)

        return hist.hist


def mock_causal_accepted_tensor(
        k: int, last_accepted_indices: torch.Tensor) -> torch.Tensor:
    """Generate an "accepted" tensor which should yield causally-accepted tokens
    up to last accepted indices.

    Tokens after last_accepted_indices+1 may also be accepted, although they
    will not be causally accepted.
    """
    batch_size = last_accepted_indices.shape[0]

    accepted = (torch.arange(k).expand(batch_size, k)
                <= last_accepted_indices.unsqueeze(-1).broadcast_to(
                    batch_size, k))

    # Sprinkle accepted values after the contiguous initial accepted values.
    # This replicates the behavior of rejection sampling, which may "accept"
    # a token that cannot be accepted because of causality.
    sprinkle_candidates = (torch.arange(k).expand(
        batch_size,
        k) > last_accepted_indices.unsqueeze(-1).broadcast_to(batch_size, k) +
                           1)
    sprinkle = torch.rand(batch_size, k) > 0.5
    accepted[sprinkle_candidates] = sprinkle[sprinkle_candidates]
    return accepted


def create_v1_sampling_metadata(
    all_greedy: bool,
    generators: Optional[dict[int, Any]] = None
) -> v1_rej_sampler.SamplingMetadata:
    """Create a v1 sampling metadata object with all_greedy set 
        to the given value. Either all greedy or all random sampling 
        is used.
    """
    generators = generators or {}
    return v1_rej_sampler.SamplingMetadata(
        temperature=torch.tensor([]),
        all_greedy=all_greedy,
        all_random=not all_greedy,
        top_p=None,
        top_k=None,
        min_p=torch.empty(1, ),
        generators=generators,
        max_num_logprobs=0,
        no_penalties=False,
        prompt_token_ids=None,
        frequency_penalties=torch.tensor([]),
        presence_penalties=torch.tensor([]),
        repetition_penalties=torch.tensor([]),
        output_token_ids=[],
        min_tokens={},
        logit_bias=[None],
        allowed_token_ids_mask=None,
    )


def get_sampler(use_v1: bool,
                use_flashinfer: bool,
                device: str = "cuda:0",
                strict_mode: bool = False) -> Any:
    if use_v1:
        import os
        os.environ[
            "VLLM_USE_FLASHINFER_SAMPLER"] = "1" if use_flashinfer else "0"
        return v1_rej_sampler.RejectionSampler()
    sampler = v0_rej_sampler.RejectionSampler(use_flashinfer=use_flashinfer,
                                              strict_mode=strict_mode)
    sampler.init_gpu_tensors(device=device)
    return sampler
