# ruff: noqa: E402
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from unittest.mock import MagicMock

sys.modules["uvloop"] = MagicMock()

import torch

from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler


def test_monolingual_sampler_masking():
    # 1. Initialize Sampler
    sampler = Sampler()

    # Define a vocab size of 100
    vocab_size = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Setup the Chinese CJK mask on the sampler
    # Let's say tokens 10, 20, 30 are Chinese
    chinese_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    chinese_mask[10] = True
    chinese_mask[20] = True
    chinese_mask[30] = True
    sampler.chinese_mask = chinese_mask

    # 3. Create dummy logits (e.g. all ones)
    logits = torch.ones((1, vocab_size), dtype=torch.float32, device=device)

    # Test case A: Monolingual mask enabled with soft-masking bias -100.0,
    # and in thinking phase
    metadata_thinking = SamplingMetadata(
        temperature=torch.tensor([1.0], device=device),
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        prompt_token_ids=torch.zeros((1, 5), dtype=torch.int64, device=device),
        frequency_penalties=torch.tensor([0.0], device=device),
        presence_penalties=torch.tensor([0.0], device=device),
        repetition_penalties=torch.tensor([1.0], device=device),
        output_token_ids=[[]],  # empty, so currently thinking
        spec_token_ids=None,
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        thinking_budget_state_holder=None,
        monolingual_drift_mask=[True],
        think_token_ids=[151648],
        end_think_token_ids=[151649],
        chinese_token_ids_paths=[None],
        monolingual_drift_biases=[-100.0],
    )

    # Apply logits processors
    processed_logits_thinking = sampler.apply_logits_processors(
        logits.clone(), metadata_thinking, predict_bonus_token=False
    )

    # Assert Chinese tokens are soft-masked (1.0 - 100.0 = -99.0)
    assert processed_logits_thinking[0, 10].item() == -99.0
    assert processed_logits_thinking[0, 20].item() == -99.0
    assert processed_logits_thinking[0, 30].item() == -99.0
    # Non-Chinese tokens should be untouched
    assert processed_logits_thinking[0, 0].item() == 1.0
    assert processed_logits_thinking[0, 99].item() == 1.0

    # Test case B: Monolingual mask enabled, but thinking has finished (end
    # think token seen)
    metadata_finished = SamplingMetadata(
        temperature=torch.tensor([1.0], device=device),
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        prompt_token_ids=torch.zeros((1, 5), dtype=torch.int64, device=device),
        frequency_penalties=torch.tensor([0.0], device=device),
        presence_penalties=torch.tensor([0.0], device=device),
        repetition_penalties=torch.tensor([1.0], device=device),
        output_token_ids=[
            [151648, 5, 151649]
        ],  # has end think token, so thinking completed
        spec_token_ids=None,
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        thinking_budget_state_holder=None,
        monolingual_drift_mask=[True],
        think_token_ids=[151648],
        end_think_token_ids=[151649],
        chinese_token_ids_paths=[None],
        monolingual_drift_biases=[-100.0],
    )

    # Apply logits processors
    processed_logits_finished = sampler.apply_logits_processors(
        logits.clone(), metadata_finished, predict_bonus_token=False
    )

    # Assert Chinese tokens are NOT masked
    assert processed_logits_finished[0, 10].item() == 1.0
    assert processed_logits_finished[0, 20].item() == 1.0
    assert processed_logits_finished[0, 30].item() == 1.0
    assert processed_logits_finished[0, 0].item() == 1.0

    # Test case C: Monolingual mask disabled
    metadata_disabled = SamplingMetadata(
        temperature=torch.tensor([1.0], device=device),
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        prompt_token_ids=torch.zeros((1, 5), dtype=torch.int64, device=device),
        frequency_penalties=torch.tensor([0.0], device=device),
        presence_penalties=torch.tensor([0.0], device=device),
        repetition_penalties=torch.tensor([1.0], device=device),
        output_token_ids=[[]],
        spec_token_ids=None,
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        thinking_budget_state_holder=None,
        monolingual_drift_mask=[False],
        think_token_ids=[151648],
        end_think_token_ids=[151649],
        chinese_token_ids_paths=[None],
        monolingual_drift_biases=[-100.0],
    )

    processed_logits_disabled = sampler.apply_logits_processors(
        logits.clone(), metadata_disabled, predict_bonus_token=False
    )

    # Assert Chinese tokens are NOT masked
    assert processed_logits_disabled[0, 10].item() == 1.0
    assert processed_logits_disabled[0, 20].item() == 1.0
    assert processed_logits_disabled[0, 30].item() == 1.0

    # Test case D: Monolingual mask enabled with hard-masking bias -inf,
    # and in thinking phase
    metadata_hard = SamplingMetadata(
        temperature=torch.tensor([1.0], device=device),
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        prompt_token_ids=torch.zeros((1, 5), dtype=torch.int64, device=device),
        frequency_penalties=torch.tensor([0.0], device=device),
        presence_penalties=torch.tensor([0.0], device=device),
        repetition_penalties=torch.tensor([1.0], device=device),
        output_token_ids=[[]],
        spec_token_ids=None,
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        thinking_budget_state_holder=None,
        monolingual_drift_mask=[True],
        think_token_ids=[151648],
        end_think_token_ids=[151649],
        chinese_token_ids_paths=[None],
        monolingual_drift_biases=[float("-inf")],
    )

    processed_logits_hard = sampler.apply_logits_processors(
        logits.clone(), metadata_hard, predict_bonus_token=False
    )

    # Assert Chinese tokens are hard-masked (-inf)
    assert processed_logits_hard[0, 10].item() == float("-inf")
    assert processed_logits_hard[0, 20].item() == float("-inf")
    assert processed_logits_hard[0, 30].item() == float("-inf")

    # Test case E: Monolingual mask enabled with unsafe path in thinking phase
    metadata_unsafe_path = SamplingMetadata(
        temperature=torch.tensor([1.0], device=device),
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        prompt_token_ids=torch.zeros((1, 5), dtype=torch.int64, device=device),
        frequency_penalties=torch.tensor([0.0], device=device),
        presence_penalties=torch.tensor([0.0], device=device),
        repetition_penalties=torch.tensor([1.0], device=device),
        output_token_ids=[[]],
        spec_token_ids=None,
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        thinking_budget_state_holder=None,
        monolingual_drift_mask=[True],
        think_token_ids=[151648],
        end_think_token_ids=[151649],
        chinese_token_ids_paths=["../../etc/passwd"],
        monolingual_drift_biases=[-100.0],
    )

    try:
        sampler.apply_logits_processors(
            logits.clone(), metadata_unsafe_path, predict_bonus_token=False
        )
        raise AssertionError("Should have raised ValueError for unsafe path")
    except ValueError as e:
        assert "Unsafe or unauthorized JSON file path" in str(e)

    print("All monolingual sampler constraint tests passed successfully!")


if __name__ == "__main__":
    test_monolingual_sampler_masking()
