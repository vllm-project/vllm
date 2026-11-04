# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the `vllm bench sweep tune` CLI glue (no GPU)."""

from vllm.benchmarks.sweep.tune import TuneArgs, speculative_config


def test_speculative_config_covers_every_k():
    cfg = speculative_config([[1, 51, 3], [52, 128, 2]], "drafter", "eagle3")
    assert cfg["num_speculative_tokens"] == 3
    assert cfg["num_speculative_tokens_per_batch_size"] == [[1, 51, 3], [52, 128, 2]]
    # a table that only uses K=1 still needs num_speculative_tokens >= 1
    cfg = speculative_config([[1, 128, 1]], "drafter", "eagle3")
    assert cfg["num_speculative_tokens"] == 1


def test_k0_excluded_by_default():
    # without --include-k0 the K values measured are 1..k_max
    args = TuneArgs(target="t", drafter="d", prompts_file="p.jsonl")
    ks = ([0] if args.include_k0 else []) + list(range(1, args.k_max + 1))
    assert 0 not in ks and ks == [1, 2, 3]
