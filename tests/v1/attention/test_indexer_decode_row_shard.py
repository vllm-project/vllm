# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerDecodeRowShard,
    decode_row_shard_plan,
    restore_decode_row_order,
)


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize(
    "num_reqs,rows_per_req", [(1, 1), (3, 6), (8, 6), (13, 2), (48, 6)]
)
def test_decode_row_shard_scores_every_row_once_in_order(
    tp_size: int, num_reqs: int, rows_per_req: int
) -> None:
    """Each decode row is scored by exactly one rank, a request's rows stay
    together under one id, and the all-gathered results land back in order."""
    num_tokens = num_reqs * rows_per_req
    plans = [
        decode_row_shard_plan(num_tokens, rows_per_req, tp_size, rank)
        for rank in range(tp_size)
    ]
    owned = torch.cat([rows[valid] for rows, valid, _ in plans])
    assert torch.equal(owned.sort().values, torch.arange(num_tokens))
    for rows, _, ids in plans:
        assert ids.numel() == rows.numel() + 1
        per_req = ids[: rows.numel()].view(-1, rows_per_req)
        assert (per_req == per_req[:, :1]).all()
        assert (per_req[1:, 0] != per_req[:-1, 0]).all()

    # What each rank all-gathers: the (valid) row it scored.
    gathered = torch.cat([rows.masked_fill(~valid, -1) for rows, valid, _ in plans])
    num_local_reqs = plans[0][0].numel() // rows_per_req
    row_shard = DeepseekV32IndexerDecodeRowShard(
        rows=plans[0][0],
        seq_lens=torch.empty(0),
        block_table=torch.empty(0),
        indices=torch.empty(0),
        schedule_metadata=torch.empty(0),
        tp_size=tp_size,
        rows_per_req=rows_per_req,
        num_local_reqs=num_local_reqs,
    )
    restored = restore_decode_row_order(gathered[:, None], row_shard, num_tokens)
    assert torch.equal(restored[:, 0], torch.arange(num_tokens))
