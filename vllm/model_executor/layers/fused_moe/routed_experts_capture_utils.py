# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context


def collect_local_routing_rows(
    topk_ids: torch.Tensor,
    *,
    dp_rank: int,
    sp_size: int,
    is_gathered: bool,
) -> torch.Tensor:
    """Collect this DP rank's rows from evenly padded SP shards.

    is_gathered selects a DP-major concatenation instead of local SP input.
    Only TP0 publishes rows; its zero/one-token SP shard is already complete.
    """
    ctx = get_forward_context()
    if ctx.dp_metadata is not None:
        counts = ctx.dp_metadata.num_tokens_across_dp_cpu.tolist()
    elif ctx.batch_descriptor is not None:
        counts = [ctx.batch_descriptor.num_tokens]
    elif sp_size == 1 and not is_gathered:
        return topk_ids
    else:
        raise ValueError("Routed-experts restoration requires forward token counts")

    num_local_tokens = counts[dp_rank]
    if is_gathered:
        rows_per_shard = [(n + sp_size - 1) // sp_size for n in counts]
        expected_rows = sum(rows_per_shard) * sp_size
        start = sum(rows_per_shard[:dp_rank]) * sp_size
    else:
        expected_rows = (num_local_tokens + sp_size - 1) // sp_size
        start = 0
    if topk_ids.shape[0] != expected_rows:
        raise ValueError(
            f"Routed-experts restoration expected {expected_rows} rows, "
            f"got {topk_ids.shape[0]} (sp_size={sp_size}, gathered={is_gathered})"
        )

    if not is_gathered and expected_rows != num_local_tokens:
        topk_ids = get_tp_group().all_gather(topk_ids.contiguous(), dim=0)
    return topk_ids[start : start + num_local_tokens]
