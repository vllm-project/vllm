# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP collectives with PyTorch fallbacks and pluggable CUDA launches."""

import torch
import torch.distributed as dist

from vllm.model_executor.custom_op import CustomOp


def _validate_dcp_empty_shard_args(
    seq_lens: torch.Tensor | None,
    query_start_loc: torch.Tensor | None,
) -> bool:
    """Validate the empty-shard mask inputs.

    Returns True when masking is requested, False when it is disabled. Raises on
    an inconsistent pair. Performs no device work so it is safe to call from a
    CUDA-graph-captured region.
    """
    if seq_lens is None and query_start_loc is None:
        return False
    if seq_lens is None or query_start_loc is None:
        raise ValueError("seq_lens and query_start_loc must be provided together")
    if (
        seq_lens.ndim != 1
        or query_start_loc.ndim != 1
        or query_start_loc.shape[0] != seq_lens.shape[0] + 1
    ):
        raise ValueError("query_start_loc must contain one boundary per sequence")
    return True


def mask_dcp_empty_shards_(
    lse: torch.Tensor,
    seq_lens: torch.Tensor | None,
    query_start_loc: torch.Tensor | None,
) -> None:
    if not _validate_dcp_empty_shard_args(seq_lens, query_start_loc):
        return
    assert seq_lens is not None and query_start_loc is not None

    # A DCP rank can receive no local sequences during CUDA graph warmup even
    # though the padded LSE buffer still has rows. In that case every row is an
    # empty shard; avoid indexing the empty seq_lens tensor below.
    if seq_lens.shape[0] == 0:
        lse.fill_(float("-inf"))
        return

    row_indices = torch.arange(
        lse.shape[0], device=lse.device, dtype=query_start_loc.dtype
    )
    sequence_indices = torch.searchsorted(
        query_start_loc[1:], row_indices, right=True
    ).clamp_max(seq_lens.shape[0] - 1)
    empty_rows = (row_indices >= query_start_loc[-1]) | (
        seq_lens[sequence_indices] == 0
    )
    lse.masked_fill_(empty_rows[:, None], float("-inf"))


def _lse_weighted_combine(
    outputs: torch.Tensor,
    lses: torch.Tensor,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch LSE-weighted combination used by native DCP and references.

    Args:
        outputs: Partial attention outputs [N, B, H, D]
                 N = number of KV shards (ranks)
                 B = batch size (num_tokens)
                 H = number of heads per rank
                 D = head dimension
        lses: Log-sum-exp values [N, B, H]
        return_lse: If True, also return the global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2

    Returns:
        Combined output [B, H, D], and optionally global LSE [B, H]

    """
    N, B, H, D = outputs.shape

    # Handle NaN and inf in LSEs
    lses = torch.where(
        torch.isnan(lses) | torch.isinf(lses),
        float("-inf"),
        lses,
    )

    # Compute max LSE for numerical stability
    lse_max, _ = lses.max(dim=0)  # [B, H]
    lse_max = torch.where(
        lse_max == float("-inf"),
        torch.zeros_like(lse_max),
        lse_max,
    )

    # Compute weights: softmax over the N dimension
    if is_lse_base_on_e:
        weights = torch.exp(lses - lse_max.unsqueeze(0))  # [N, B, H]
    else:
        weights = torch.pow(2.0, lses - lse_max.unsqueeze(0))  # [N, B, H]

    # Handle NaN weights
    weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)

    # Normalize weights
    weight_sum = weights.sum(dim=0, keepdim=True)  # [1, B, H]
    weights = weights / weight_sum.clamp(min=1e-10)  # [N, B, H]

    # Weighted combination: sum over N dimension
    weights = weights.unsqueeze(-1)
    outputs = torch.where(weights == 0, torch.zeros_like(outputs), outputs)
    result = (outputs * weights).sum(dim=0)  # [B, H, D]

    if return_lse:
        if is_lse_base_on_e:
            global_lse = torch.log(weight_sum.squeeze(0)) + lse_max  # [B, H]
        else:
            global_lse = torch.log2(weight_sum.squeeze(0)) + lse_max  # [B, H]
        return result, global_lse

    return result


@CustomOp.register("direct_dcp_q_gather")
class DirectDCPQGather(CustomOp):
    """Gather query shards into the core-owned output view."""

    def __init__(self, group: dist.ProcessGroup) -> None:
        super().__init__()
        self.group = group

    def forward_native(
        self,
        local_query: torch.Tensor,
        final_query: torch.Tensor,
        received_signal: torch.Tensor,
        completion: torch.Tensor,
        epoch: torch.Tensor,
        world_size: int,
        rank: int,
        max_num_tokens: int,
        padded_num_heads: int,
        query_mc_ptr: int,
        signal_mc_ptr: int,
    ) -> None:
        gathered = local_query.new_empty((world_size, *local_query.shape))
        dist.all_gather_into_tensor(
            gathered.flatten(0, 1), local_query.contiguous(), group=self.group
        )
        final_query.copy_(gathered.movedim(0, 1).reshape(final_query.shape))

    def forward_cuda(
        self,
        local_query: torch.Tensor,
        final_query: torch.Tensor,
        received_signal: torch.Tensor,
        completion: torch.Tensor,
        epoch: torch.Tensor,
        world_size: int,
        rank: int,
        max_num_tokens: int,
        padded_num_heads: int,
        query_mc_ptr: int,
        signal_mc_ptr: int,
    ) -> None:
        torch.ops._C.direct_dcp_q_gather(
            local_query,
            final_query,
            received_signal,
            completion,
            epoch,
            world_size,
            rank,
            max_num_tokens,
            padded_num_heads,
            query_mc_ptr,
            signal_mc_ptr,
        )


@CustomOp.register("direct_dcp_a2a_lse_reduce")
class DirectDCPA2AReduce(CustomOp):
    """Exchange partial attention outputs and reduce them using peer LSEs."""

    def __init__(self, group: dist.ProcessGroup) -> None:
        super().__init__()
        self.group = group

    def forward_native(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        seq_lens: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        peer_output_ptrs: torch.Tensor,
        peer_lse_ptrs: torch.Tensor,
        peer_signal_ptrs: torch.Tensor,
        received_output: torch.Tensor,
        received_lse: torch.Tensor,
        received_signal: torch.Tensor,
        epoch: torch.Tensor,
        combined_output: torch.Tensor,
        world_size: int,
        rank: int,
        max_num_tokens: int,
        is_lse_base_on_e: bool,
    ) -> None:
        num_tokens, num_heads, head_dim = partial_output.shape
        heads_per_rank = num_heads // world_size
        lse = partial_lse.to(dtype=torch.float32, copy=True)
        mask_dcp_empty_shards_(lse, seq_lens, query_start_loc)
        send_buffer = (
            torch.cat((partial_output.float(), lse.unsqueeze(-1)), dim=-1)
            .reshape(num_tokens, world_size, heads_per_rank, head_dim + 1)
            .movedim(1, 0)
            .contiguous()
        )
        recv_buffer = torch.empty_like(send_buffer)
        dist.all_to_all_single(recv_buffer, send_buffer, group=self.group)
        output = _lse_weighted_combine(
            recv_buffer[..., :head_dim],
            recv_buffer[..., head_dim],
            is_lse_base_on_e=is_lse_base_on_e,
        )
        assert isinstance(output, torch.Tensor), "DCP combine must return its output"
        combined_output.copy_(output)

    def forward_cuda(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        seq_lens: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        peer_output_ptrs: torch.Tensor,
        peer_lse_ptrs: torch.Tensor,
        peer_signal_ptrs: torch.Tensor,
        received_output: torch.Tensor,
        received_lse: torch.Tensor,
        received_signal: torch.Tensor,
        epoch: torch.Tensor,
        combined_output: torch.Tensor,
        world_size: int,
        rank: int,
        max_num_tokens: int,
        is_lse_base_on_e: bool,
    ) -> None:
        torch.ops._C.direct_dcp_a2a_lse_reduce(
            partial_output,
            partial_lse,
            seq_lens,
            query_start_loc,
            peer_output_ptrs,
            peer_lse_ptrs,
            peer_signal_ptrs,
            received_output,
            received_lse,
            received_signal,
            epoch,
            combined_output,
            world_size,
            rank,
            max_num_tokens,
            is_lse_base_on_e,
        )


@CustomOp.register("direct_dcp_kv_gather")
class DirectDCPKVGather(CustomOp):
    """Gather KV rows through the double-buffered symmetric workspace."""

    def __init__(self, group: dist.ProcessGroup) -> None:
        super().__init__()
        self.group = group

    def forward_native(
        self,
        local_kv: torch.Tensor,
        received_kv: torch.Tensor,
        received_signal: torch.Tensor,
        completion: torch.Tensor,
        epoch: torch.Tensor,
        gathered_kv: torch.Tensor,
        world_size: int,
        rank: int,
        max_gathered_tokens: int,
        kv_mc_ptr: int,
        signal_mc_ptr: int,
    ) -> None:
        dist.all_gather_into_tensor(
            gathered_kv, local_kv.contiguous(), group=self.group
        )

    def forward_cuda(
        self,
        local_kv: torch.Tensor,
        received_kv: torch.Tensor,
        received_signal: torch.Tensor,
        completion: torch.Tensor,
        epoch: torch.Tensor,
        gathered_kv: torch.Tensor,
        world_size: int,
        rank: int,
        max_gathered_tokens: int,
        kv_mc_ptr: int,
        signal_mc_ptr: int,
    ) -> None:
        torch.ops._C.direct_dcp_kv_gather(
            local_kv,
            received_kv,
            received_signal,
            completion,
            epoch,
            gathered_kv,
            world_size,
            rank,
            max_gathered_tokens,
            kv_mc_ptr,
            signal_mc_ptr,
        )
