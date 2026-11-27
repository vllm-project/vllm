# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM90 indexer's prefill row split: a step splits once its scoring
work reaches the threshold, into shares that cover every row. Host
arithmetic only."""

import torch

from vllm.models.deepseek_v41.nvidia.specialized_decode.hopper.indexer_metadata import (  # noqa: E501
    ROW_SHARD_MIN_WORK,
    prefill_row_shard,
)


def test_small_steps_stay_replicated() -> None:
    # 4096 new tokens of a 4096-token prompt: 4096 x 1024 compressed keys.
    seq_lens = torch.tensor([4096])
    query_lens = torch.tensor([4096])
    assert ROW_SHARD_MIN_WORK > 4096 * 1024
    assert prefill_row_shard(seq_lens, query_lens, 4, 8) is None


def test_large_steps_split_every_row() -> None:
    # A 4096-token chunk at 96K tokens of context, beside a short prompt.
    seq_lens = torch.tensor([98304, 512])
    query_lens = torch.tensor([4096, 512])
    sizes = prefill_row_shard(seq_lens, query_lens, 4, 8)
    assert sizes is not None and len(sizes) == 8
    assert sum(sizes) == 4608 and min(sizes) > 0
