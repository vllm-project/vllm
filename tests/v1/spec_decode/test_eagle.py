# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import compute_probs_and_sample_next_token


# Minimal mock for SamplingMetadata
def create_minimal_metadata(
    generators_dict: dict[int, torch.Generator], ) -> SamplingMetadata:
    batch_size = len(generators_dict)

    return SamplingMetadata(
        temperature=torch.tensor([0.8] * batch_size, dtype=torch.float32),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        min_p=None,
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(batch_size, dtype=torch.float32),
        presence_penalties=torch.zeros(batch_size, dtype=torch.float32),
        repetition_penalties=torch.zeros(batch_size, dtype=torch.float32),
        output_token_ids=[[] for _ in range(batch_size)],
        min_tokens={i: (0, set())
                    for i in range(batch_size)},
        logit_bias=[None] * batch_size,
        allowed_token_ids_mask=None,
        bad_words_token_ids={i: []
                             for i in range(batch_size)},
        generators=generators_dict,
    )


# Simplified test focusing only on the seed effect
def test_compute_probs_and_sample_seed_determinism():
    vocab_size = 1024
    batch_size = 16

    torch.manual_seed(0)
    logits = torch.rand(batch_size, vocab_size, dtype=torch.float32)

    # --- Run 1: Seed 42 ---
    seed1 = 42
    generators1 = {
        i: torch.Generator().manual_seed(seed1)
        for i in range(batch_size)
    }
    metadata1 = create_minimal_metadata(generators_dict=generators1)
    tokens1, probs1 = compute_probs_and_sample_next_token(
        logits.clone(), metadata1)

    # --- Run 2: Seed 42 (Again) ---
    generators2 = {
        i: torch.Generator().manual_seed(seed1)
        for i in range(batch_size)
    }
    metadata2 = create_minimal_metadata(generators_dict=generators2)
    tokens2, probs2 = compute_probs_and_sample_next_token(
        logits.clone(), metadata2)

    # --- Run 3: Seed 123 (Different) ---
    seed3 = 123
    generators3 = {
        i: torch.Generator().manual_seed(seed3)
        for i in range(batch_size)
    }
    metadata3 = create_minimal_metadata(generators_dict=generators3)
    tokens3, probs3 = compute_probs_and_sample_next_token(
        logits.clone(), metadata3)

    # 1. Same seed should yield same results
    assert torch.equal(tokens1, tokens2)
    assert torch.equal(probs1, probs2)

    # 2. Different seeds should yield different results (highly likely)
    assert not torch.equal(tokens1, tokens3)
    assert not torch.equal(probs1, probs3)
