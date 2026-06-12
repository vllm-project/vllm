# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.attention.ops.common import CPTritonContext


def dcp_lse_reduce(
    local_out: torch.Tensor,
    local_lse: torch.Tensor,
    group: "GroupCoordinator",
    ctx: "CPTritonContext | None" = None,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reduce partial attention output/LSE across DCP ranks."""
    return dcp_a2a_lse_reduce(
        local_out,
        local_lse,
        group,
        ctx=ctx,
        return_lse=return_lse,
        is_lse_base_on_e=is_lse_base_on_e,
    )


class AttentionOutputReducer:
    """Merge DCP partial attention output and write the final local heads."""

    def __init__(self, group: "GroupCoordinator") -> None:
        self.group = group

    def reduce_lse_out(
        self,
        local_out: torch.Tensor,
        local_lse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, lse = dcp_lse_reduce(
            local_out,
            local_lse,
            self.group,
            return_lse=True,
        )
        return out, lse

    @staticmethod
    def apply_attn_sink(
        out: torch.Tensor,
        lse: torch.Tensor,
        attn_sink: torch.Tensor,
    ) -> torch.Tensor:
        sink = attn_sink[: out.shape[1]].to(dtype=lse.dtype)
        output_lse = torch.logaddexp(lse, sink.unsqueeze(0))
        scale = torch.exp(lse - output_lse).to(dtype=out.dtype)
        return out * scale.unsqueeze(-1)

    @staticmethod
    def scatter_heads_back(out: torch.Tensor, output: torch.Tensor) -> None:
        output[:, : out.shape[1], :].copy_(out)

    def reduce_with_sink(
        self,
        local_out: torch.Tensor,
        local_lse: torch.Tensor,
        attn_sink: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        out, lse = self.reduce_lse_out(local_out, local_lse)
        out = self.apply_attn_sink(out, lse, attn_sink)
        self.scatter_heads_back(out, output)


def dcp_global_topk(
    local_values: torch.Tensor,
    local_global_indices: torch.Tensor,
    k: int,
    group: "GroupCoordinator",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select global top-k candidates contributed by every DCP rank."""
    candidate_values = group.all_gather(local_values.contiguous(), dim=1)
    candidate_indices = group.all_gather(local_global_indices.contiguous(), dim=1)
    candidate_values = torch.where(
        candidate_indices >= 0,
        candidate_values,
        torch.full_like(candidate_values, float("-inf")),
    )

    values, offsets = torch.topk(candidate_values, k=k, dim=-1)
    indices = torch.gather(candidate_indices, 1, offsets)
    indices = torch.where(
        values == float("-inf"),
        torch.full_like(indices, -1),
        indices,
    )
    return values, indices.to(torch.int32)


def dcp_softmax_reduce(
    local_max: torch.Tensor,
    local_sum: torch.Tensor,
    local_weighted_value: torch.Tensor,
    group: "GroupCoordinator",
) -> torch.Tensor:
    """Merge numerically stable partial softmax statistics across DCP ranks."""
    valid = local_sum > 0
    local_max = torch.where(
        valid,
        local_max,
        torch.full_like(local_max, -float("inf")),
    )
    gathered_max = group.all_gather(local_max, dim=0).reshape(
        (group.world_size,) + local_max.shape
    )
    global_max = gathered_max.max(dim=0).values

    scale = torch.exp(local_max - global_max)
    scale = torch.where(valid, scale, torch.zeros_like(scale))
    reduce_payload = torch.stack(
        (
            torch.where(valid, local_sum * scale, torch.zeros_like(local_sum)),
            torch.where(
                valid,
                local_weighted_value * scale,
                torch.zeros_like(local_weighted_value),
            ),
        )
    )
    global_sum, global_weighted_value = group.all_reduce(reduce_payload).unbind(0)
    return torch.where(
        global_sum > 0,
        global_weighted_value / global_sum,
        torch.zeros_like(global_weighted_value),
    )
