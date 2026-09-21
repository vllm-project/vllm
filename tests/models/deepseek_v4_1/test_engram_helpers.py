# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.models.deepseek_v4_1.common.engram import (
    EngramLayout,
    compute_hash_multipliers,
    find_next_prime,
)


def test_find_next_prime_skips_used_values():
    assert find_next_prime(100, {101}) == 103


def test_engram_layout_assigns_disjoint_bucket_ranges():
    config = SimpleNamespace(
        engram_layer_ids=[1, 3],
        engram_num_embeddings=[1000, 1000],
        engram_max_ngram_size=3,
        engram_n_heads=2,
        engram_head_dim=8,
        engram_compressed_vocab_size=128,
        engram_pad_token_id=0,
        engram_vocab_size=100,
    )

    layout = EngramLayout.from_config(config)

    assert layout is not None
    assert layout.layer_ids == (1, 3)
    assert layout.n_hash_cols == 4
    assert layout.offsets.shape == (2, 4)
    all_primes = {prime for layer in layout.primes for row in layer for prime in row}
    assert len(all_primes) == 8


def test_hash_multipliers_are_deterministic_odd_and_int64_safe():
    first = compute_hash_multipliers((1, 3), 3, 128)
    second = compute_hash_multipliers((1, 3), 3, 128)

    torch.testing.assert_close(first, second)
    assert first.shape == (2, 3)
    assert torch.all(first % 2 == 1)
    assert torch.all(first <= torch.iinfo(torch.int64).max // 128)
