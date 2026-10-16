# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable launches using the direct DCP workspace and synchronization ABI."""

import torch

from vllm.model_executor.custom_op import CustomOp


@CustomOp.register("direct_dcp_q_gather")
class DirectDCPQGather(CustomOp):
    """Publish query shards and wait for peers in core-owned symmetric memory."""

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

    forward_cuda = forward_native


@CustomOp.register("direct_dcp_a2a_lse_reduce")
class DirectDCPA2AReduce(CustomOp):
    """Exchange partial attention outputs and reduce them using peer LSEs."""

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

    forward_cuda = forward_native


@CustomOp.register("direct_dcp_kv_gather")
class DirectDCPKVGather(CustomOp):
    """Gather KV rows through the double-buffered symmetric workspace."""

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

    forward_cuda = forward_native
