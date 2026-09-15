# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One MoE layer on the global expert pool: device-planned decode step and
the two-partition (bank + host) path for wider batches, both without host
code in the forward, so a CUDA graph can capture them.

Marlin consumer only in this version: the bank holds Marlin-format rows for
every layer, so the bank's row count exceeds the layer's expert count.
Alignment therefore happens on logical expert ids (absent experts already
padding) and the aligned blocks are mapped to physical bank rows
afterwards; the align op never sees a row id (it histograms by id in a
num_experts+1 buffer). Ported from the lab expert tier (runtime.py
split_global / _run_marlin_chains).
"""

from __future__ import annotations

import math
from typing import Any

import torch

from vllm.model_executor.layers.fused_moe.expert_pool.pool import GlobalPool, copy_in
from vllm.model_executor.layers.fused_moe.expert_pool.tables import (
    TENSORS,
    StepBuffers,
    step,
)


def mask_routes(ids: torch.Tensor, expert_map: torch.Tensor) -> torch.Tensor:
    """Routes whose expert is absent from `expert_map` become padding (-1)."""
    num_experts = expert_map.shape[0]
    safe = ids.clamp(0, num_experts - 1).long()
    present = (ids >= 0) & (ids < num_experts) & (expert_map[safe] >= 0)
    return torch.where(present, ids, torch.full_like(ids, -1))


def physical_block_experts(
    logical_ids: torch.Tensor,
    post_padded: torch.Tensor,
    block: int,
    expert_map: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Map per-block logical expert ids to physical bank rows.

    Blocks at or beyond `post_padded` tokens are unused by the GEMM and may
    hold uninitialized ids; they become -1 without indexing anything.
    """
    blocks = torch.arange(
        logical_ids.numel(), device=logical_ids.device, dtype=torch.int32
    )
    valid = (blocks * block) < post_padded.reshape(1)
    safe = torch.where(valid, logical_ids, torch.zeros_like(logical_ids))
    safe = safe.clamp(0, num_experts - 1)
    return torch.where(valid, expert_map[safe.long()], torch.full_like(logical_ids, -1))


def marlin_block_size(tokens, top_k, local_experts, global_experts, input_dtype):
    """The stock fused_marlin_moe M-block choice for one expert partition."""
    estimated = math.ceil(tokens * local_experts / global_experts)
    block = 8
    for block in (8, 16, 32, 48, 64):
        if estimated * top_k / local_experts / block < 0.9:
            break
    if input_dtype is not None and input_dtype.itemsize == 1:
        block = max(block, 16)
    return block


class PoolLayer:
    """Per-layer view of the pool plus the consumer state."""

    def __init__(
        self,
        index: int,
        pool: GlobalPool,
        slots: int,
        sources: dict[str, torch.Tensor],
        host_views: dict[str, torch.Tensor],
        buffers: StepBuffers,
        experts: Any,
        marlin_workspace: torch.Tensor,
        num_experts: int,
        top_k: int,
        activation: Any,
        apply_router_weight_on_input: bool,
    ) -> None:
        self.index = index
        self.pool = pool
        self.slots = slots
        self.offset = pool.offset(index)
        self.sources = sources  # pinned host rows, final layout
        self.host = host_views  # accelerator (UVA) views of the sources
        self.buffers = buffers
        self.experts = experts  # MarlinExperts bound to the bank tensors
        self.marlin_workspace = marlin_workspace
        self.num_experts = num_experts
        self.top_k = top_k
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.bank = pool.bank
        self.bank_rows = pool.rows
        self.staging_rows = pool.staging_slots
        self.width = buffers.gather_src.shape[0]
        self.hot_map = pool.tables.layer_slice(pool.tables.hot_phys, index)
        self.cold_map = pool.tables.layer_slice(pool.tables.cold_phys, index)
        self.decode_steps = 0
        self.partition_steps = 0

    # --- forward -----------------------------------------------------------

    def apply(
        self, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor
    ) -> torch.Tensor:
        self._check_routes(x, weights, ids)
        lanes = ids.shape[0] * ids.shape[1]
        if lanes <= self.width and lanes <= self.staging_rows:
            self.decode_steps += 1
            step(self.pool.tables, self.index, ids, self.buffers)
            copy_in(self.host, self.bank, self.buffers)
            return self._run_marlin(
                x, weights, ids, ((self.bank, self.buffers.step_map, self.bank_rows),)
            )
        # Wider batches (prefill): resident rows from the bank, the rest read
        # straight from the pinned host source through its accelerator view.
        self.partition_steps += 1
        return self._run_marlin(
            x,
            weights,
            ids,
            (
                (self.bank, self.hot_map, self.bank_rows),
                (self.host, self.cold_map, self.num_experts),
            ),
        )

    def _check_routes(self, x, weights, ids) -> None:
        if (
            x.ndim != 2
            or ids.ndim != 2
            or weights.shape != ids.shape
            or ids.shape[0] != x.shape[0]
            or ids.shape[1] != self.top_k
            or ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Unexpected pool routing/input shape or dtype")
        allowed = (ids >= -1) & (ids < self.num_experts)
        finite = torch.isfinite(weights) & (weights >= 0)
        torch._assert_async(
            (allowed & (finite | (ids == -1))).all(),
            "Invalid routing: ids must be in [-1, num_experts) and weights "
            "finite/nonnegative",
        )

    def _run_marlin(self, x, weights, ids, partitions):
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
            _fused_marlin_moe,
            marlin_moe_intermediate_size,
        )
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )
        from vllm.scalar_type import ScalarType
        from vllm.v1.worker.workspace import current_workspace_manager

        experts = self.experts
        tokens, hidden = x.shape
        top_k = ids.shape[1]
        rows_count = tokens * top_k
        inner = marlin_moe_intermediate_size(
            self.bank["w13_weight"], self.bank["w2_weight"]
        )
        # Same manager as the stock kernels: stable addresses once locked.
        cache13, cache2, rows = current_workspace_manager().get_simultaneous(
            ((rows_count * max(2 * inner, hidden),), x.dtype),
            ((rows_count, inner), x.dtype),
            ((rows_count, hidden), x.dtype),
        )
        rows.zero_()
        for tensors, expert_map, slots in partitions:
            block = marlin_block_size(
                tokens, top_k, slots, self.num_experts, experts.input_dtype
            )
            routed = ids
            if slots > self.num_experts:
                # Align by logical id (absent experts already padding), then
                # map the blocks to rows; the align op never sees a row id.
                routed = mask_routes(ids, expert_map)
                sorted_ids, logical_ids, post_padded = moe_align_block_size(
                    routed, block, self.num_experts, None, ignore_invalid_experts=True
                )
                expert_ids = physical_block_experts(
                    logical_ids, post_padded, block, expert_map, self.num_experts
                )
            else:
                sorted_ids, expert_ids, post_padded = moe_align_block_size(
                    ids,
                    block,
                    self.num_experts,
                    expert_map,
                    ignore_invalid_experts=True,
                )
            _fused_marlin_moe(
                hidden_states=x,
                w1=tensors["w13_weight"],
                w2=tensors["w2_weight"],
                bias1=experts.w1_bias,
                bias2=experts.w2_bias,
                w1_scale=self._scale(tensors, "w13_weight_scale"),
                w2_scale=self._scale(tensors, "w2_weight_scale"),
                topk_weights=weights,
                num_topk=top_k,
                quant_type=ScalarType.from_id(experts.quant_type_id),
                apply_router_weight_on_input=self.apply_router_weight_on_input,
                expert_map=expert_map,
                block_size_m=block,
                sorted_token_ids=sorted_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=post_padded,
                activation=self.activation,
                activation_func=experts.activation,
                topk_ids=routed,
                input_global_scale1=experts.a1_gscale,
                input_global_scale2=experts.a2_gscale,
                global_scale1=self._scale(tensors, "w13_weight_scale_2"),
                global_scale2=self._scale(tensors, "w2_weight_scale_2"),
                w1_zeros=experts.w1_zp,
                w2_zeros=experts.w2_zp,
                workspace=self.marlin_workspace,
                intermediate_cache13=cache13,
                intermediate_cache2=cache2,
                output=rows,
                input_dtype=experts.input_dtype,
                activation_config=experts.activation_config,
            )
        # Rows already carry the router weights (second GEMM multiplies them).
        return torch.sum(rows.view(tokens, top_k, hidden), dim=1)

    @staticmethod
    def _scale(tensors: dict[str, torch.Tensor], name: str) -> torch.Tensor:
        # Scales and globals are indexed by the same physical row as the
        # weights of the partition being run (bank rows or host rows).
        return tensors[name]

    def stats(self) -> dict[str, int]:
        return {
            "decode_steps": self.decode_steps,
            "partition_steps": self.partition_steps,
            "slots": self.slots,
        }


__all__ = [
    "TENSORS",
    "PoolLayer",
    "marlin_block_size",
    "mask_routes",
    "physical_block_experts",
]
