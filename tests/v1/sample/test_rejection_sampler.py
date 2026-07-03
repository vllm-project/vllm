# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from tests.v1.sample.utils import create_allowed_token_ids
from vllm.platforms import current_platform
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import (
    PLACEHOLDER_TOKEN_ID,
    RejectionSampler,
    rejection_sample,
    sample_recovered_tokens,
)
from vllm.v1.sample.sampler import Sampler, SamplerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

DEVICE_TYPE = current_platform.device_type
# Repo root (…/tests/v1/sample/test_rejection_sampler.py -> parents[3]) so the
# CPU accept-rule subprocess can import the ``tests`` package.
REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def rejection_sampler():
    mock_sampler = Mock(spec=Sampler)
    mock_sampler.logprobs_mode = "raw_logprobs"
    return RejectionSampler(mock_sampler)


def mock_sampler_output(
    rejection_sampler: RejectionSampler, bonus_token_ids: torch.Tensor
):
    rejection_sampler.sampler.return_value = SamplerOutput(
        sampled_token_ids=bonus_token_ids, logprobs_tensors=None
    )


def create_spec_decode_metadata(
    spec_tokens: list[list[int]], logits: torch.Tensor
) -> SpecDecodeMetadata:
    metadata = SpecDecodeMetadata.make_dummy(spec_tokens, device=logits.device)
    metadata.target_logits_indices = torch.arange(logits.shape[0])
    # Output bonus token ids are mocked, so the bonus logit indices should
    # be empty.
    metadata.bonus_logits_indices = torch.empty(0, dtype=torch.int32)
    return metadata


def create_logits_tensor(
    output_token_ids: list[list[int]],
    vocab_size: int = 100,
    token_idx_to_override: int | None = None,
) -> torch.Tensor:
    """Helper function to create logits tensor that
    will produce desired token ids on argmax"""
    token_ids = [tokens[:-1] for tokens in output_token_ids]
    num_total_tokens = sum(len(tokens) for tokens in token_ids)
    logits = torch.full((num_total_tokens, vocab_size), -100.0, device=DEVICE_TYPE)
    start_loc = 0
    for tokens in token_ids:
        for j, token_id in enumerate(tokens):
            logits[start_loc + j, token_id] = 100.0
        start_loc += len(tokens)
    if token_idx_to_override:
        logits[:, token_idx_to_override] = 99.0
    return logits


def create_sampling_metadata(
    all_greedy: bool,
    output_token_ids: list[list[int]] | None = None,
    prompt_token_ids: torch.Tensor | None = None,
    spec_token_ids: torch.Tensor | None = None,
    temperature: torch.Tensor | None = None,
    top_k: torch.Tensor | None = None,
    top_p: torch.Tensor | None = None,
    generators: dict[int, Any] | None = None,
    frequency_penalties: list[float] | None = None,
    presence_penalties: list[float] | None = None,
    repetition_penalties: list[float] | None = None,
    bad_words_token_ids: dict[int, list[list[int]]] | None = None,
    allowed_token_ids_mask: torch.Tensor | None = None,
) -> SamplingMetadata:
    """Create a v1 sampling metadata object with all_greedy set
    to the given value. Either all greedy or all random sampling
    is used.
    """
    generators = generators or {}
    if all_greedy:
        temperature = None
    else:
        assert temperature is not None

    if any([frequency_penalties, presence_penalties, repetition_penalties]):
        no_penalties = False

        assert output_token_ids
        assert len(output_token_ids) > 0

        frequency_penalties = torch.tensor(frequency_penalties, device=DEVICE_TYPE)
        presence_penalties = torch.tensor(presence_penalties, device=DEVICE_TYPE)
        repetition_penalties = torch.tensor(repetition_penalties, device=DEVICE_TYPE)
    else:
        no_penalties = True
        frequency_penalties = torch.tensor([])
        presence_penalties = torch.tensor([])
        repetition_penalties = torch.tensor([])

    return SamplingMetadata(
        temperature=temperature,
        all_greedy=all_greedy,
        all_random=not all_greedy,
        top_p=top_p,
        top_k=top_k,
        generators=generators,
        max_num_logprobs=None,
        no_penalties=no_penalties,
        prompt_token_ids=prompt_token_ids,
        frequency_penalties=frequency_penalties,
        presence_penalties=presence_penalties,
        repetition_penalties=repetition_penalties,
        output_token_ids=[] if output_token_ids is None else output_token_ids,
        spec_token_ids=[] if spec_token_ids is None else spec_token_ids,
        allowed_token_ids_mask=allowed_token_ids_mask,
        bad_words_token_ids={} if bad_words_token_ids is None else bad_words_token_ids,
        logitsprocs=LogitsProcessors(),
    )


def run_relaxed_rejection_sample(
    draft_token_ids: list[int],
    target_logits: torch.Tensor,
    thinking_state: bool | None,
    *,
    relax_ratio: float = 0.5,
    relax_top_k: int = 3,
    think_start_token_id: int = 10,
    think_end_token_id: int = 11,
    sampling_metadata: SamplingMetadata | None = None,
    device: str = DEVICE_TYPE,
) -> list[int]:
    num_draft_tokens = [len(draft_token_ids)]
    thinking_states = (
        None
        if thinking_state is None
        else torch.tensor([thinking_state], dtype=torch.bool, device=device)
    )
    output = rejection_sample(
        draft_token_ids=torch.tensor(draft_token_ids, dtype=torch.int32, device=device),
        num_draft_tokens=num_draft_tokens,
        max_spec_len=len(draft_token_ids),
        cu_num_draft_tokens=torch.tensor(
            num_draft_tokens, dtype=torch.int32, device=device
        ),
        draft_probs=None,
        target_logits=target_logits,
        bonus_token_ids=torch.tensor([12], dtype=torch.int64, device=device),
        sampling_metadata=sampling_metadata
        if sampling_metadata is not None
        else create_sampling_metadata(all_greedy=True),
        relaxed_thinking=True,
        relax_ratio=relax_ratio,
        relax_top_k=relax_top_k,
        thinking_states=thinking_states,
        think_start_token_id=think_start_token_id,
        think_end_token_id=think_end_token_id,
    )
    return output[0].tolist()


def test_relaxed_thinking_accepts_top_k_token_above_ratio():
    target_logits = torch.full((1, 16), -100.0, device=DEVICE_TYPE)
    target_logits[0, 5] = 10.0
    target_logits[0, 7] = 9.4

    output = run_relaxed_rejection_sample([7], target_logits, True)

    assert output == [7, 12]


def test_relaxed_thinking_false_uses_strict_argmax():
    target_logits = torch.full((1, 16), -100.0, device=DEVICE_TYPE)
    target_logits[0, 5] = 10.0
    target_logits[0, 7] = 9.4

    output = run_relaxed_rejection_sample([7], target_logits, False)

    assert output == [5, PLACEHOLDER_TOKEN_ID]


def test_relaxed_thinking_fallback_truncates_after_boundary_token():
    target_logits = torch.full((2, 16), -100.0, device=DEVICE_TYPE)
    target_logits[0, 10] = 10.0
    target_logits[1, 8] = 10.0

    output = run_relaxed_rejection_sample([10, 8], target_logits, None)

    assert output == [
        10,
        PLACEHOLDER_TOKEN_ID,
        PLACEHOLDER_TOKEN_ID,
    ]


def test_relaxed_thinking_rejects_token_below_ratio_floor():
    target_logits = torch.full((1, 16), -100.0, device=DEVICE_TYPE)
    target_logits[0, 5] = 10.0
    target_logits[0, 7] = 8.0

    output = run_relaxed_rejection_sample([7], target_logits, True)

    assert output == [5, PLACEHOLDER_TOKEN_ID]


def test_relaxed_thinking_truncates_after_accepted_boundary_token():
    target_logits = torch.full((3, 16), -100.0, device=DEVICE_TYPE)
    target_logits[0, 7] = 10.0
    target_logits[1, 11] = 10.0
    target_logits[2, 8] = 10.0

    output = run_relaxed_rejection_sample([7, 11, 8], target_logits, True)

    assert output == [7, 11, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]


def test_relaxed_thinking_rejects_non_greedy_sampling():
    target_logits = torch.full((1, 16), -100.0, device=DEVICE_TYPE)
    target_logits[0, 7] = 10.0

    with pytest.raises(ValueError, match="greedy sampling"):
        run_relaxed_rejection_sample(
            [7],
            target_logits,
            True,
            sampling_metadata=create_sampling_metadata(
                all_greedy=False,
                temperature=torch.ones(1, device=DEVICE_TYPE),
            ),
        )


def test_relaxed_thinking_rejects_misaligned_thinking_states():
    target_logits = torch.full((1, 16), -100.0, device=DEVICE_TYPE)
    target_logits[0, 7] = 10.0

    with pytest.raises(ValueError, match="thinking_states"):
        rejection_sample(
            draft_token_ids=torch.tensor([7], dtype=torch.int32, device=DEVICE_TYPE),
            num_draft_tokens=[1],
            max_spec_len=1,
            cu_num_draft_tokens=torch.tensor(
                [1], dtype=torch.int32, device=DEVICE_TYPE
            ),
            draft_probs=None,
            target_logits=target_logits,
            bonus_token_ids=torch.tensor([12], dtype=torch.int64, device=DEVICE_TYPE),
            sampling_metadata=create_sampling_metadata(all_greedy=True),
            relaxed_thinking=True,
            thinking_states=torch.tensor(
                [True, False], dtype=torch.bool, device=DEVICE_TYPE
            ),
        )


def _cpu_accept_rule_checks() -> None:
    """Assert the relaxed accept rule against the REAL kernel on CPU.

    Run in a subprocess by ``test_relaxed_thinking_accept_rule_runs_on_cpu``
    with ``TRITON_INTERPRET=1`` set (before ``triton.jit`` decorates the
    kernel) so ``relaxed_thinking_sample_kernel`` executes on CPU tensors with
    no GPU. Raises ``AssertionError`` (non-zero exit) if the rule regresses.
    """
    # top1 = id5 @ 10.0 -> ratio floor = 10.0 + log(0.5) ~= 9.307.
    # Draft id7 @ 9.4 is in the target top-3 AND >= floor, so the relaxed rule
    # ACCEPTS id7 even though argmax is id5. Strict argmax would emit id5, so
    # this case is what distinguishes the relaxed rule from strict decoding.
    logits = torch.full((1, 16), -100.0)
    logits[0, 5] = 10.0
    logits[0, 7] = 9.4
    accepted = run_relaxed_rejection_sample([7], logits, True, device="cpu")
    assert accepted == [7, 12], f"relaxed should accept id7, got {accepted}"

    # Draft id7 @ 8.0 is below the ratio floor -> REJECTED, strict argmax id5.
    logits = torch.full((1, 16), -100.0)
    logits[0, 5] = 10.0
    logits[0, 7] = 8.0
    rejected = run_relaxed_rejection_sample([7], logits, True, device="cpu")
    assert rejected == [5, PLACEHOLDER_TOKEN_ID], (
        f"draft below floor should fall back to argmax, got {rejected}"
    )


def test_relaxed_thinking_accept_rule_runs_on_cpu():
    """Defend the relaxed accept rule on a CPU-only box (no GPU).

    The existing relaxed tests dispatch to the Triton kernel and need CUDA/HIP,
    so a revert of the accept rule to strict argmax is invisible without a GPU.
    This runs the SAME production kernel under Triton's interpreter on CPU
    tensors. ``TRITON_INTERPRET`` must be set before ``triton.jit`` decorates
    the kernel, so the checks run in a fresh subprocess with the env applied
    and all GPUs hidden. Fails if the relaxed rule is reverted to strict argmax.
    """
    env = {
        **os.environ,
        "TRITON_INTERPRET": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.v1.sample.test_rejection_sampler import "
            "_cpu_accept_rule_checks; _cpu_accept_rule_checks()",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "CPU relaxed accept-rule checks failed "
        f"(rc={result.returncode})\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


########################### Tests for Greedy Sampling ###################
def test_perfect_match(rejection_sampler):
    """Test when output tokens perfectly match speculated tokens"""
    spec_tokens = [[1, 2, 3]]
    output_tokens = [[1, 2, 3, 4]]  # 4 is the bonus token

    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus_token_tensor = torch.tensor([output_tokens[0][-1]], device=logits.device)
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected = torch.tensor([[1, 2, 3, 4]], dtype=torch.int, device=logits.device)
    assert torch.equal(output.sampled_token_ids, expected)


def test_early_mismatch(rejection_sampler):
    """Test when there's an early mismatch in tokens"""
    spec_tokens = [[1, 2, 3]]
    output_tokens = [[1, 5, 3, 4]]  # Mismatch at position 1

    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus_token_tensor = torch.tensor([output_tokens[0][-1]], device=logits.device)
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected = torch.tensor(
        [[1, 5, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]],
        dtype=torch.int,
        device=logits.device,
    )
    assert torch.equal(output.sampled_token_ids, expected)


def test_multiple_sequences(rejection_sampler):
    """Test handling multiple sequences of speculated tokens"""
    spec_tokens = [[1, 2], [3]]
    output_tokens = [[1, 2, 5], [3, 4]]  # Two sequences with bonus tokens 5 and 4

    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus_token_tensor = torch.tensor(
        [output_tokens[0][-1], output_tokens[1][-1]], device=logits.device
    )
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected = torch.tensor(
        [[1, 2, 5], [3, 4, PLACEHOLDER_TOKEN_ID]], dtype=torch.int, device=logits.device
    )
    assert torch.equal(output.sampled_token_ids, expected)


def test_single_token_sequence(rejection_sampler):
    """Test handling sequences with single token"""
    spec_tokens = [[1]]
    output_tokens = [[1, 2]]  # Single token with bonus token 2

    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus_token_tensor = torch.tensor([output_tokens[0][-1]], device=logits.device)
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected = torch.tensor([[1, 2]], dtype=torch.int, device=logits.device)
    assert torch.equal(output.sampled_token_ids, expected)


def test_empty_sequence(rejection_sampler):
    """Test handling empty sequence of speculated tokens"""
    spec_tokens: list[list[int]] = [[]]
    output_tokens = [[5]]  # Just the bonus token

    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus_token_tensor = torch.tensor([output_tokens[0][-1]], device=logits.device)
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected = torch.tensor([[5]], dtype=torch.int, device=logits.device)
    assert torch.equal(output.sampled_token_ids, expected)


def test_multiple_mismatches(rejection_sampler):
    """Test handling multiple sequences with mismatches"""
    spec_tokens = [[1, 2, 3], [4, 5, 6]]
    output_tokens = [[1, 2, 7, 6], [4, 8, 6, 9]]  # Mismatches in both sequences

    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus_token_tensor = torch.tensor(
        [output_tokens[0][-1], output_tokens[1][-1]], device=logits.device
    )
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected = torch.tensor(
        [
            [1, 2, 7, PLACEHOLDER_TOKEN_ID],
            [4, 8, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID],
        ],
        dtype=torch.int,
        device=logits.device,
    )
    assert torch.equal(output.sampled_token_ids, expected)


@pytest.mark.parametrize(
    "spec_tokens,output_tokens,expected",
    [
        ([[1, 2]], [[1, 2, 3]], [[1, 2, 3]]),  # Perfect match with bonus
        ([[1]], [[2, 3]], [[2, PLACEHOLDER_TOKEN_ID]]),  # First mismatch
        (
            [[1, 2], [3, 4]],
            [[1, 5, 6], [3, 4, 7]],
            [[1, 5, PLACEHOLDER_TOKEN_ID], [3, 4, 7]],
        ),  # Mixed matches
    ],
)
def test_parametrized_cases(rejection_sampler, spec_tokens, output_tokens, expected):
    """Parametrized test for various matching scenarios"""
    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus_token_tensor = torch.tensor(
        [tokens[-1] for tokens in output_tokens], device=logits.device
    )
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected_tensor = torch.tensor(expected, dtype=torch.int, device=logits.device)
    assert torch.equal(output.sampled_token_ids, expected_tensor)


########################### Tests for Random Sampling ###################
@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("frac_seeded", [0.0, 0.5])
@pytest.mark.parametrize("n_rep", [20])
def test_deterministic_when_seeded(
    rejection_sampler,
    k: int,
    vocab_size: int,
    batch_size: int,
    frac_seeded: float,
    n_rep: int,
):
    num_tokens = batch_size * k
    draft_probs = torch.rand(
        num_tokens,
        vocab_size,
        dtype=torch.float32,
        device=DEVICE_TYPE,
    )
    draft_probs = F.softmax(draft_probs, dim=-1)
    target_logits = torch.rand_like(draft_probs)
    bonus_token_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, 1),
        dtype=torch.int64,
        device=DEVICE_TYPE,
    )
    draft_token_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, k),
        dtype=torch.int64,
        device=DEVICE_TYPE,
    )

    seeded_mask = torch.rand(batch_size, dtype=torch.float32) <= frac_seeded

    results = []
    for _ in range(n_rep):
        seeded_seqs = {
            i: torch.Generator(device=DEVICE_TYPE).manual_seed(i)
            for i in range(batch_size)
            if seeded_mask[i]
        }

        temperature = torch.ones(batch_size, dtype=torch.float32, device=DEVICE_TYPE)
        sampling_metadata = create_sampling_metadata(
            all_greedy=False, temperature=temperature, generators=seeded_seqs
        )
        spec_decode_metadata = create_spec_decode_metadata(
            draft_token_ids.tolist(), target_logits
        )

        mock_sampler_output(rejection_sampler, bonus_token_ids)
        rep_result = rejection_sampler(
            spec_decode_metadata,
            draft_probs=None,
            logits=target_logits,
            sampling_metadata=sampling_metadata,
        )

        results.append(rep_result.sampled_token_ids)

    for i in range(batch_size):
        if seeded_mask[i]:
            for j in range(1, n_rep):
                assert torch.equal(results[j][i], results[0][i])


def test_rejection_sampling_approximates_target_distribution():
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
    """
    torch.set_default_device(DEVICE_TYPE)
    vocab_size = 10
    k = 2
    num_reference_probs = 100

    # Prepare draft, target, and reference probability distributions
    draft_probs = F.softmax(torch.rand(vocab_size, dtype=torch.float32), dim=-1)
    target_logits = torch.rand(vocab_size, dtype=torch.float32)
    target_probs = F.softmax(target_logits, dim=-1)
    reference_probs = F.softmax(
        torch.rand(num_reference_probs, vocab_size, dtype=torch.float32),
        dim=-1,
    )

    sample_sizes = [10, 100, 1_000, 10_000, 100_000]
    distance_wrt_reference: list[float] = []
    distance_wrt_target: list[float] = []

    for num_samples in sample_sizes:
        # Sample using rejection sampling.
        rej_sample_probs = estimate_rejection_sampling_pdf(
            draft_probs, target_logits, k, vocab_size, num_samples
        )
        rej_sample_probs = rej_sample_probs.to(DEVICE_TYPE)

        # Average distance from reference probs.
        reference_vs_rejsample_dist = (
            torch.dist(reference_probs, rej_sample_probs).item()
            / reference_probs.shape[0]
        )
        target_vs_rejsample_dist = torch.dist(target_probs, rej_sample_probs).item()

        distance_wrt_reference.append(reference_vs_rejsample_dist)
        distance_wrt_target.append(target_vs_rejsample_dist)

        relative_change_in_distance_wrt_target = get_ratio_first_to_last(
            distance_wrt_target
        )
        relative_change_in_distance_wrt_reference = get_ratio_first_to_last(
            distance_wrt_reference
        )

        print(
            f"{num_samples=} {target_vs_rejsample_dist=:.05f} "
            f"{reference_vs_rejsample_dist=:.05f}"
        )
        print(
            f"{num_samples=} {relative_change_in_distance_wrt_target=:.02f} "
            f"{relative_change_in_distance_wrt_reference=:.02f}"
        )

    relative_change_in_distance_wrt_target = get_ratio_first_to_last(
        distance_wrt_target
    )
    relative_change_in_distance_wrt_reference = get_ratio_first_to_last(
        distance_wrt_reference
    )

    expected_improvement_multiplier = 20
    assert (
        relative_change_in_distance_wrt_target
        > relative_change_in_distance_wrt_reference * expected_improvement_multiplier
    )


def get_ratio_first_to_last(elements: list[float]) -> float:
    return elements[0] / elements[-1]


def estimate_rejection_sampling_pdf(
    draft_probs: torch.Tensor,
    target_logits: torch.Tensor,
    k: int,
    vocab_size: int,
    num_samples: int,
) -> torch.Tensor:
    """Estimate the probability distribution of the output tokens
    using rejection sampling.

    Args:
        draft_probs: Draft probability distribution.
        target_logits: Target logits.
        num_samples: Number of samples to draw.

    Returns:
        Estimated probability distribution of the output tokens.
    """
    mock_sampler = Mock(spec=Sampler)
    mock_sampler.logprobs_mode = "raw_logprobs"
    rejection_sampler = RejectionSampler(mock_sampler)
    num_tokens = num_samples * k
    # Repeat draft probs num_samples * k times.
    draft_probs = draft_probs.reshape(1, 1, vocab_size).repeat(num_samples, k, 1)

    # Repeat target probs num_tokens times.
    target_logits = target_logits.reshape(1, vocab_size).repeat(num_tokens, 1)

    # Randomly sample draft token ids from draft probs.
    draft_token_ids = torch.multinomial(
        draft_probs[:, 0, :], num_samples=k, replacement=True
    ).reshape(num_samples, k)
    draft_probs = draft_probs.view(num_tokens, vocab_size)

    # Bonus tokens not used but required.
    bonus_token_ids = torch.zeros((1, 1), dtype=torch.int64, device=DEVICE_TYPE).repeat(
        num_samples, 1
    )

    temperature = torch.ones(num_samples, dtype=torch.float32, device=DEVICE_TYPE)
    sampling_metadata = create_sampling_metadata(
        all_greedy=False, temperature=temperature
    )
    spec_decode_metadata = create_spec_decode_metadata(
        draft_token_ids.tolist(), target_logits
    )

    mock_sampler_output(rejection_sampler, bonus_token_ids)
    sampler_output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=draft_probs,
        logits=target_logits,
        sampling_metadata=sampling_metadata,
    )
    output_token_ids = sampler_output.sampled_token_ids[:, :-1].flatten()

    hist = torch.histogram(
        output_token_ids.to(dtype=torch.float, device="cpu"),
        bins=vocab_size,
        range=(0, vocab_size),
        density=True,
    )

    return hist.hist


def native_sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    draft_token_ids: torch.Tensor,  # [num_tokens]
    draft_probs: torch.Tensor | None,  # [num_tokens, vocab_size]
    target_probs: torch.Tensor,  # [num_tokens, vocab_size]
    sampling_metadata: SamplingMetadata,
    device: torch.device,
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    q_dtype = torch.float64 if use_fp64_gumbel else torch.float32

    q = torch.empty(
        (batch_size, vocab_size),
        dtype=q_dtype,
        device=device,
    )
    q.exponential_()

    states = {
        i: generator.get_state()
        for i, generator in sampling_metadata.generators.items()
    }
    for i, generator in sampling_metadata.generators.items():
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)

        # In order to generate the same exponential later, reset the CUDA RNG
        # state because RNG state advances after each call.
        generator.set_state(states[i])

    inv_q = q.reciprocal()

    out = torch.empty_like(draft_token_ids)

    for req_idx in range(batch_size):
        start_idx = 0 if req_idx == 0 else int(cu_num_draft_tokens[req_idx - 1].item())
        end_idx = int(cu_num_draft_tokens[req_idx].item())
        num_tokens = end_idx - start_idx

        for pos in range(max_spec_len):
            if pos >= num_tokens:
                continue
            token_idx = start_idx + pos

            if draft_probs is None:
                # prob is target_probs[token_idx] except draft_token_id is zeroed
                prob = target_probs[token_idx].clone()
                draft_token_id = draft_token_ids[token_idx]
                prob[draft_token_id] = 0.0
            else:
                prob = (target_probs[token_idx] - draft_probs[token_idx]).clamp_min_(
                    0.0
                )

            score = prob * inv_q[req_idx]
            recovered_id = torch.argmax(score, dim=-1)
            out[token_idx] = recovered_id
    return out


def _test_masked_logits(
    rejection_sampler,
    batch_size: int,
    num_draft_tokens: int,
    vocab_size: int,
    target_logits: torch.Tensor,
    unmasked_indices: torch.Tensor,
    sampling_metadata: SamplingMetadata,
):
    # Set up test parameters
    num_tokens = batch_size * num_draft_tokens

    # Create random draft probabilities.
    draft_probs = torch.rand(
        (num_tokens, vocab_size), dtype=torch.float32, device=DEVICE_TYPE
    )
    draft_probs = F.softmax(draft_probs, dim=-1)

    # Randomly sample draft token ids from draft probs
    draft_token_ids = torch.multinomial(draft_probs, num_samples=1)
    draft_token_ids = draft_token_ids.reshape(batch_size, num_draft_tokens)
    draft_token_ids = draft_token_ids.tolist()

    # Bonus tokens not used but required
    bonus_token_ids = torch.zeros(
        (batch_size, 1),
        dtype=torch.int64,
        device=DEVICE_TYPE,
    )

    # Create spec decode metadata
    spec_decode_metadata = create_spec_decode_metadata(draft_token_ids, target_logits)

    # Run rejection sampling
    mock_sampler_output(rejection_sampler, bonus_token_ids)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=draft_probs,
        logits=target_logits,
        sampling_metadata=sampling_metadata,
    )

    # Remove bonus tokens and reshape
    output_token_ids = output.sampled_token_ids[:, :-1].flatten().tolist()

    # Check that all sampled tokens are within the unmasked indices.
    for i in range(num_tokens):
        token_id = output_token_ids[i]
        if token_id == PLACEHOLDER_TOKEN_ID:
            continue
        assert token_id in unmasked_indices[i]


@pytest.mark.parametrize("top_k", [1, 5, 99])
def test_top_k(rejection_sampler, top_k):
    """Test rejection sampling with top-k sampling"""
    vocab_size = 100
    batch_size = 100
    num_draft_tokens = 3
    num_tokens = batch_size * num_draft_tokens

    # Randomly create top-k indices.
    top_k_indices = [
        torch.randperm(vocab_size, device=DEVICE_TYPE)[:top_k]
        for _ in range(num_tokens)
    ]
    top_k_indices = torch.stack(top_k_indices)

    # Create logits with the uniform distribution.
    target_logits = torch.zeros((num_tokens, vocab_size), device=DEVICE_TYPE)

    # Increment the logits for top-k indices, a little bit more than the other
    # ones. If the masking is effective, the non-topk indices will never be
    # sampled despite the small difference in logits.
    for i in range(num_tokens):
        target_logits[i, top_k_indices[i]] += 0.1

    # Create sampling metadata
    temperature = torch.ones(batch_size, dtype=torch.float32, device=DEVICE_TYPE)
    sampling_metadata = create_sampling_metadata(
        all_greedy=False,
        temperature=temperature,
        top_k=torch.tensor([top_k] * batch_size, device=DEVICE_TYPE, dtype=torch.int64),
    )

    _test_masked_logits(
        rejection_sampler,
        batch_size=batch_size,
        num_draft_tokens=num_draft_tokens,
        vocab_size=vocab_size,
        target_logits=target_logits,
        unmasked_indices=top_k_indices,
        sampling_metadata=sampling_metadata,
    )


@pytest.mark.parametrize("top_p", [0.5, 0.9, 0.99])
def test_top_p(rejection_sampler, top_p):
    """Test rejection sampling with top-p sampling"""
    vocab_size = 100
    batch_size = 100
    num_draft_tokens = 3
    num_tokens = batch_size * num_draft_tokens

    # Create logits with the uniform distribution.
    target_logits = torch.randn((num_tokens, vocab_size), device=DEVICE_TYPE)
    temperature = torch.ones(batch_size, dtype=torch.float32, device=DEVICE_TYPE)
    rescaled_logits = target_logits / temperature

    logits_sort, logits_idx = rescaled_logits.sort(dim=-1, descending=False)
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)
    top_p_mask = probs_sum <= 1 - top_p
    # at least one
    top_p_mask[:, -1] = False

    # Get the top-p indices.
    top_p_indices = []
    for i in range(num_tokens):
        top_p_indices.append(logits_idx[i][~top_p_mask[i]].tolist())

    # Create sampling metadata
    sampling_metadata = create_sampling_metadata(
        all_greedy=False,
        temperature=temperature,
        top_p=torch.tensor(
            [top_p] * batch_size,
            device=DEVICE_TYPE,
            dtype=torch.float32,
        ),
    )

    _test_masked_logits(
        rejection_sampler,
        batch_size=batch_size,
        num_draft_tokens=num_draft_tokens,
        vocab_size=vocab_size,
        target_logits=target_logits,
        unmasked_indices=top_p_indices,
        sampling_metadata=sampling_metadata,
    )


########################### Tests for Logit Processors ###################
def test_frequency_penalties(rejection_sampler):
    """Test rejection sampling with frequency penalties"""
    spec_tokens = [[1, 1, 1], [], [1, 1, 1]]
    output_tokens = [[1, 1, 1, 1], [7], [1, 1, 1, 1]]  # 1, 7 and 1 are the bonus tokens

    num_requests = len(spec_tokens)
    logits = create_logits_tensor(output_tokens, token_idx_to_override=15)
    metadata = create_sampling_metadata(
        all_greedy=True,
        output_token_ids=[[2], [3], [4]],
        spec_token_ids=spec_tokens,
        prompt_token_ids=torch.tensor(
            [[5, 6, 7], [6, 7, 8], [7, 8, 9]],
            device=DEVICE_TYPE,
        ),
        frequency_penalties=[1.5, 1.5, 0.7],
        presence_penalties=[0.0] * num_requests,
        repetition_penalties=[1.0] * num_requests,
    )
    bonus_token_tensor = torch.tensor(
        [output_tokens[i][-1] for i in range(len(output_tokens))], device=logits.device
    )
    spec_decode_metadata = SpecDecodeMetadata.make_dummy(
        spec_tokens, device=logits.device
    )
    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected = torch.tensor(
        [[1, 15, -1, -1], [7, -1, -1, -1], [1, 1, 15, -1]],
        dtype=torch.int,
        device=logits.device,
    )
    assert torch.equal(output.sampled_token_ids, expected)


def test_bad_words(rejection_sampler):
    """Test rejection sampling with bad words constraints.

    This test applies bad words to non-consecutive requests (0 and 2, but not 1)
    to verify correct logit indexing when iterating over requests with bad words.
    """
    spec_tokens = [[1, 2, 3], [1, 15, 3], [1, 2, 3]]
    output_tokens = [[1, 2, 3, 4], [1, 15, 3, 4], [1, 2, 3, 4]]

    logits = create_logits_tensor(output_tokens, token_idx_to_override=15)
    metadata = create_sampling_metadata(
        all_greedy=True,
        output_token_ids=[[2], [3], [4]],
        spec_token_ids=spec_tokens,
        bad_words_token_ids={
            0: [[2]],
            # Request 1 has no bad words (to test non-consecutive request handling)
            2: [[2]],
        },
    )
    bonus_token_tensor = torch.tensor(
        [output_tokens[i][-1] for i in range(len(output_tokens))], device=logits.device
    )
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)
    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )

    # Request 0: bad word [2] matches prefix, so token 2 is rejected -> 15
    # Request 1: no bad words, all tokens match -> [1, 15, 3, 4]
    # Request 2: bad word [2] matches prefix, so token 2 is rejected -> 15
    expected = torch.tensor(
        [[1, 15, -1, -1], [1, 15, 3, 4], [1, 15, -1, -1]],
        dtype=torch.int,
        device=logits.device,
    )
    assert torch.equal(output.sampled_token_ids, expected)


def test_allowed_token_ids(rejection_sampler):
    """Test rejection sampling with allowed token ids"""
    spec_tokens = [[1, 2, 10], [10, 5, 3], [7, 10, 12]]
    output_tokens = [[1, 2, 10, 5], [10, 5, 10, 5], [7, 10, 12, 5]]
    # Not allowed tokens:
    # 0: 0-4
    # 1: 1-5
    # 2: 2-6
    num_allowed_token_ids = 5

    # Use the token 15 as the sampler choose if a token rejected
    logits = create_logits_tensor(output_tokens, token_idx_to_override=15)

    batch_size = len(output_tokens)
    _, vocab_size = logits.size()
    mask = create_allowed_token_ids(
        batch_size=batch_size,
        vocab_size=vocab_size,
        num_allowed_token_ids=num_allowed_token_ids,
        device=logits.device,
    )
    metadata = create_sampling_metadata(
        all_greedy=True,
        output_token_ids=[[], [], []],
        spec_token_ids=spec_tokens,
        allowed_token_ids_mask=mask,
    )
    bonus_token_tensor = torch.tensor(
        [output_tokens[i][-1] for i in range(len(output_tokens))], device=logits.device
    )
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)
    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )

    expected = torch.tensor(
        [[15, -1, -1, -1], [10, 5, 10, -1], [7, 10, 12, 5]],
        dtype=torch.int,
        device=logits.device,
    )
    assert torch.equal(output.sampled_token_ids, expected)


@pytest.mark.parametrize("batch_size", [1, 100])
@pytest.mark.parametrize("vocab_size", [100, 8192, 10000])
@pytest.mark.parametrize("max_spec_len", [1, 3])
@pytest.mark.parametrize("no_draft_probs", [True, False])
def test_sample_recovered_tokens(
    batch_size: int, vocab_size: int, max_spec_len: int, no_draft_probs: bool
):
    num_tokens = batch_size * max_spec_len

    # Create random draft probabilities.
    draft_probs = torch.rand(
        num_tokens,
        vocab_size,
        dtype=torch.float32,
        device=DEVICE_TYPE,
    )
    draft_probs = F.softmax(draft_probs, dim=-1)

    # Create random target probabilities.
    target_logits = torch.rand(
        num_tokens, vocab_size, dtype=torch.float32, device=DEVICE_TYPE
    )
    target_probs = F.softmax(target_logits, dim=-1)

    # Randomly sample draft token ids from draft probs
    draft_token_ids = torch.multinomial(draft_probs, num_samples=1).to(torch.int32)

    temperature = torch.ones(batch_size, dtype=torch.float32, device=DEVICE_TYPE)
    generators = {
        i: torch.Generator(device=DEVICE_TYPE).manual_seed(i) for i in range(batch_size)
    }
    sampling_metadata = create_sampling_metadata(
        all_greedy=False, temperature=temperature, generators=generators
    )

    spec_decode_metadata = create_spec_decode_metadata(
        draft_token_ids.reshape(batch_size, max_spec_len).tolist(), target_logits
    )

    ref_recovered_token_ids = native_sample_recovered_tokens(
        max_spec_len,
        spec_decode_metadata.num_draft_tokens,
        spec_decode_metadata.cu_num_draft_tokens,
        draft_token_ids,
        None if no_draft_probs else draft_probs,
        target_probs,
        sampling_metadata,
        device=DEVICE_TYPE,
    )
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        spec_decode_metadata.num_draft_tokens,
        spec_decode_metadata.cu_num_draft_tokens,
        draft_token_ids,
        None if no_draft_probs else draft_probs,
        target_probs,
        sampling_metadata,
        device=DEVICE_TYPE,
    )
    assert torch.equal(recovered_token_ids, ref_recovered_token_ids)


def test_sample_recovered_tokens_uses_fp64_exponential_race_when_requested():
    batch_size = 2
    vocab_size = 64
    max_spec_len = 2
    num_tokens = batch_size * max_spec_len

    draft_probs = torch.rand(
        num_tokens,
        vocab_size,
        dtype=torch.float32,
        device=DEVICE_TYPE,
    )
    draft_probs = F.softmax(draft_probs, dim=-1)
    target_probs = torch.rand(
        num_tokens,
        vocab_size,
        dtype=torch.float32,
        device=DEVICE_TYPE,
    )
    target_probs = F.softmax(target_probs, dim=-1)
    draft_token_ids = torch.multinomial(draft_probs, num_samples=1).to(torch.int32)

    generators = {
        i: torch.Generator(device=DEVICE_TYPE).manual_seed(i) for i in range(batch_size)
    }
    sampling_metadata = create_sampling_metadata(
        all_greedy=False,
        temperature=torch.ones(batch_size, dtype=torch.float32, device=DEVICE_TYPE),
        generators=generators,
    )
    spec_decode_metadata = create_spec_decode_metadata(
        draft_token_ids.reshape(batch_size, max_spec_len).tolist(),
        target_probs.log(),
    )

    expected = native_sample_recovered_tokens(
        max_spec_len,
        spec_decode_metadata.num_draft_tokens,
        spec_decode_metadata.cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device=torch.device(DEVICE_TYPE),
        use_fp64_gumbel=True,
    )
    actual = sample_recovered_tokens(
        max_spec_len,
        spec_decode_metadata.num_draft_tokens,
        spec_decode_metadata.cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device=torch.device(DEVICE_TYPE),
        use_fp64_gumbel=True,
    )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("no_draft_probs", [True, False])
@pytest.mark.parametrize(
    "vocab_size",
    [
        100,  # below BLOCK_SIZE: single partial tile with many padding entries
        8193,  # BLOCK_SIZE + 1: only 1 valid entry in the last tile
        10000,  # non-aligned, moderate tail
        151936,  # real-world Qwen3 vocab size from the CVE report
    ],
)
def test_sample_recovered_tokens_vocab_boundary(vocab_size: int, no_draft_probs: bool):
    """Regression test for GHSA-8wr5-jm2h-8r4f.

    When vocab_size is not a multiple of BLOCK_SIZE (8192), the last Triton
    tile extends beyond the vocabulary.  If all valid entries in that tail tile
    have zero target probability, the out-of-range masked positions (score 0)
    could win the tl.max tie-break, producing recovered_id >= vocab_size.
    This test forces that scenario and asserts every recovered token is valid.
    """
    BLOCK_SIZE = 8192
    batch_size = 2
    max_spec_len = 3
    num_tokens = batch_size * max_spec_len

    last_tile_start = (vocab_size // BLOCK_SIZE) * BLOCK_SIZE

    target_probs = torch.rand(
        num_tokens, vocab_size, dtype=torch.float32, device=DEVICE_TYPE
    )
    if last_tile_start > 0:
        # Zero out valid entries in the last partial tile so the only
        # non-zero scores come from earlier, fully-covered tiles.
        target_probs[:, last_tile_start:] = 0.0
    else:
        # vocab_size < BLOCK_SIZE: single tile. Concentrate all mass on
        # entry 0 so the NO_DRAFT_PROBS path (which zeroes the draft
        # token entry) can drive all valid scores to zero.
        target_probs = torch.zeros_like(target_probs)
        target_probs[:, 0] = 1.0
    # Re-normalize so it's a valid distribution.
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)

    draft_probs = torch.rand(
        num_tokens, vocab_size, dtype=torch.float32, device=DEVICE_TYPE
    )
    draft_probs = torch.nn.functional.softmax(draft_probs, dim=-1)

    if last_tile_start == 0:
        # Force draft token to 0 so the NO_DRAFT_PROBS path zeroes the
        # only non-zero entry, leaving all valid scores at zero.
        draft_token_ids = torch.zeros(
            num_tokens, 1, dtype=torch.int32, device=DEVICE_TYPE
        )
    else:
        draft_token_ids = torch.randint(
            0, vocab_size, (num_tokens, 1), dtype=torch.int32, device=DEVICE_TYPE
        )

    temperature = torch.ones(batch_size, dtype=torch.float32, device=DEVICE_TYPE)
    generators = {
        i: torch.Generator(device=DEVICE_TYPE).manual_seed(42 + i)
        for i in range(batch_size)
    }
    sampling_metadata = create_sampling_metadata(
        all_greedy=False, temperature=temperature, generators=generators
    )

    spec_decode_metadata = create_spec_decode_metadata(
        draft_token_ids.reshape(batch_size, max_spec_len).tolist(),
        torch.rand(num_tokens, vocab_size, device=DEVICE_TYPE),
    )

    recovered = sample_recovered_tokens(
        max_spec_len,
        spec_decode_metadata.num_draft_tokens,
        spec_decode_metadata.cu_num_draft_tokens,
        draft_token_ids.squeeze(-1),
        None if no_draft_probs else draft_probs,
        target_probs,
        sampling_metadata,
        device=DEVICE_TYPE,
    )

    assert (recovered >= 0).all(), (
        f"Recovered token IDs contain negative values: "
        f"{recovered[recovered < 0].tolist()}"
    )
    assert (recovered < vocab_size).all(), (
        f"Recovered token IDs >= vocab_size ({vocab_size}): "
        f"{recovered[recovered >= vocab_size].tolist()}"
    )


########################### Tests for Synthetic Rejection Sampling #########


def _make_synthetic_sampler(rates: list[float]) -> RejectionSampler:
    mock_sampler = Mock(spec=Sampler)
    mock_sampler.logprobs_mode = "raw_logprobs"
    spec_config = Mock()
    spec_config.rejection_sample_method = "synthetic"
    spec_config.synthetic_acceptance_rates = rates
    return RejectionSampler(mock_sampler, spec_config, torch.device(DEVICE_TYPE))


def _make_sampling_metadata(all_greedy: bool) -> SamplingMetadata:
    temperature = None if all_greedy else torch.tensor([1.0, 1.0], device=DEVICE_TYPE)
    return create_sampling_metadata(all_greedy=all_greedy, temperature=temperature)


@pytest.mark.parametrize("all_greedy", [True, False])
def test_synthetic_all_accepted(all_greedy: bool):
    """With all rates=1.0, every draft token is accepted."""
    sampler = _make_synthetic_sampler([1.0, 1.0])
    spec_tokens = [[1, 2], [3]]
    output_tokens = [[10, 20, 50], [30, 40]]

    metadata = _make_sampling_metadata(all_greedy)
    logits = create_logits_tensor(output_tokens)
    bonus = torch.tensor([50, 40], device=DEVICE_TYPE)
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(sampler, bonus)
    output = sampler(spec_decode_metadata, None, logits, metadata)
    expected = torch.tensor(
        [[1, 2, 50], [3, 40, PLACEHOLDER_TOKEN_ID]],
        dtype=torch.int,
        device=DEVICE_TYPE,
    )
    assert torch.equal(output.sampled_token_ids, expected)


@pytest.mark.parametrize("all_greedy", [True, False])
def test_synthetic_all_rejected(all_greedy: bool):
    """With all rates=0.0, the first token is always rejected."""
    sampler = _make_synthetic_sampler([0.0, 0.0])
    spec_tokens = [[1, 2], [3]]
    output_tokens = [[10, 20, 50], [30, 40]]

    metadata = _make_sampling_metadata(all_greedy)
    logits = create_logits_tensor(output_tokens)
    bonus = torch.tensor([50, 40], device=DEVICE_TYPE)
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(sampler, bonus)
    output = sampler(spec_decode_metadata, None, logits, metadata)
    result = output.sampled_token_ids
    # Exactly one token emitted per sequence (the rejection fallback),
    # followed by placeholders.
    for row in result:
        assert row[0] != PLACEHOLDER_TOKEN_ID
        assert (row[1:] == PLACEHOLDER_TOKEN_ID).all()


def test_placeholder_draft_token_rejected_random(rejection_sampler):
    """A placeholder draft id (-1) must be rejected in non-greedy sampling
    without indexing the probability tensors by the invalid id.
    """
    vocab_size = 100
    spec_tokens = [[1, vocab_size - 1, PLACEHOLDER_TOKEN_ID]]
    output_tokens = [[1, vocab_size - 1, 7, 9]]

    temperature = torch.ones(1, dtype=torch.float32, device=DEVICE_TYPE)
    metadata = create_sampling_metadata(
        all_greedy=False,
        temperature=temperature,
        generators={0: torch.Generator(device=DEVICE_TYPE).manual_seed(0)},
    )
    logits = create_logits_tensor(output_tokens, vocab_size=vocab_size)
    bonus_token_tensor = torch.tensor([output_tokens[0][-1]], device=logits.device)
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    sampled = output.sampled_token_ids

    assert sampled[0, 0].item() == 1
    assert sampled[0, 1].item() == vocab_size - 1
    recovered = sampled[0, 2].item()
    assert 0 <= recovered < vocab_size
