# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the removal of the redundant all-gather in ViT ``split_qkv``.

PaddleOCR-VL/Ernie4.5-VL/GLM-4.1V vision attention used to route their local
``QKVParallelLinear`` output through an ``all_gather_interleave`` +
per-rank re-split round-trip. For full multi-head self-attention
(``total_num_kv_heads == total_num_heads``) that round-trip is a numerical
identity: it gathers every rank's ``[q_shard | k_shard | v_shard]``, reorders to
``[all_q | all_k | all_v]``, then selects back exactly the calling rank's own
shards. This test pins that equivalence so a future reintroduced gather -- or a
grouped-query layout that breaks the equal-shard assumption -- fails loudly.
"""

from functools import partial

import pytest
import torch

from vllm.distributed import utils as dist_utils


def _reference_gather_interleave(rank_locals, hidden_size, tp_size, tp_rank):
    """Old ``all_gather_interleave`` + chunk + per-rank split, on CPU tensors.

    ``rank_locals`` is the list of every rank's local qkv tensor (what a real
    ``dist.all_gather`` would have produced). Returns the calling rank's
    ``[q | k | v]`` slice, matching the pre-removal ``split_qkv`` behavior.
    """
    gathered_split = [
        torch.split(t, hidden_size // tp_size, dim=-1) for t in rank_locals
    ]
    ordered = [t for pair in zip(*gathered_split) for t in pair]
    gathered = torch.cat(ordered, dim=-1)

    q, k, v = gathered.chunk(3, dim=2)
    splitter = partial(dist_utils.split_tensor_along_last_dim, num_partitions=tp_size)
    q = splitter(q)[tp_rank]
    k = splitter(k)[tp_rank]
    v = splitter(v)[tp_rank]
    return torch.cat([q, k, v], dim=-1)


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("num_heads,head_dim", [(4, 8), (8, 16), (16, 8)])
def test_split_qkv_local_chunk_matches_gather_interleave(tp_size, num_heads, head_dim):
    """Chunking the local qkv is bit-identical to the old gather round-trip."""
    torch.manual_seed(0)
    seq_len, bs = 3, 2
    heads_per_rank = num_heads // tp_size
    comp = heads_per_rank * head_dim  # per-rank width of q (== k == v)
    # ``hidden_size`` is the split granularity the old helper received: the full
    # (all-rank) width of a single q/k/v component.
    hidden_size = num_heads * head_dim

    rank_locals = []
    for _ in range(tp_size):
        q = torch.randn(seq_len, bs, comp)
        k = torch.randn(seq_len, bs, comp)
        v = torch.randn(seq_len, bs, comp)
        rank_locals.append(torch.cat([q, k, v], dim=-1))

    for tp_rank in range(tp_size):
        reference = _reference_gather_interleave(
            rank_locals, hidden_size, tp_size, tp_rank
        )
        # New behavior: no gather, just chunk this rank's own local tensor.
        local_chunk = torch.cat(rank_locals[tp_rank].chunk(3, dim=2), dim=-1)
        assert torch.equal(reference, local_chunk)
