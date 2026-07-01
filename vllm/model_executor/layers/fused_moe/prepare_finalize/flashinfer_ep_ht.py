# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer moe_ep high-throughput prepare/finalize (FLAT / Standard).

FlashInfer's HT dispatch is token-major: it returns a `[num_recv, hidden]` receive
buffer plus the per-received-token routing (`recv_topk_idx` in LOCAL expert space with
`-1` for non-local/padding picks, and `recv_topk_weights`). That matches vLLM's
`Standard` activation format, so the fused experts run over `[num_recv, hidden]` and
finalize calls combine.

GAP 3 (recv-count): this adapter opts in to `HandleAlgoKnobNumReceivedTokens`, so
`DispatchOutput.recv_total_counter` carries the actual received-token count. On nccl-ep
v0.1 the transport buffer stays statically sized to `max_recv_tokens_per_rank`; the count
enables a future compute-view trim (`recv_x[:actual_recv]`) without resizing the buffer.

⚠ ON-GPU VALIDATION REQUIRED. Three points in this path are structurally implemented but
must be verified/iterated on real hardware (lyris), and cannot be exercised on a
CPU/host-only box:
  1. Feeding `recv_topk_idx` (local indices, `-1` for non-local) into vLLM's Standard
     `fused_moe` — confirm `-1` picks are masked (contribute 0), i.e. expert_map/`-1`
     handling matches.
  2. finalize's `fused_expert_output` arrives `(M, topk, K)` per the modular contract,
     while FlashInfer HT combine consumes `[num_recv, hidden]`; the reshape/reduce
     reconciliation below assumes the combine owns the cross-rank reduce and weights were
     bound at dispatch.
  3. The optional `recv_total_counter` compute-view trim (left off here for correctness;
     the full static buffer is used, with `-1` masking handling padding).
The low-latency adapter (`flashinfer_ep_ll.py`) is the clean, directly-mapped path and
should be validated first.
"""

from __future__ import annotations

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)

from .flashinfer_ep_common import FlashInferEPPrepareAndFinalizeBase


class FlashInferEPHTPrepareAndFinalize(FlashInferEPPrepareAndFinalizeBase):
    """Prepare/Finalize using FlashInfer moe_ep high-throughput (FLAT)."""

    def __init__(
        self,
        fleet,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        num_local_experts: int,
        hidden_size: int,
    ):
        super().__init__(fleet, num_dispatchers, num_local_experts)
        self.max_tokens_per_rank = max_tokens_per_rank
        self.hidden_size = hidden_size

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        # HT recv buffer is statically sized (max_tokens_per_rank * world); the
        # Standard path does not bound the per-rank token count the way batched does.
        return None

    def _assert_bf16(self, quant_config: FusedMoEQuantConfig) -> None:
        if quant_config is not None and quant_config.quant_dtype is not None:
            raise NotImplementedError(
                "FlashInferEPHTPrepareAndFinalize currently supports bf16 dispatch "
                f"only; got quant_dtype={quant_config.quant_dtype!r}."
            )

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        self._assert_bf16(quant_config)

        if apply_router_weight_on_input:
            assert topk_ids.size(1) == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        from flashinfer.moe_ep import (
            DispatchInputParams,
            HandleAlgoKnobNumReceivedTokens,
        )

        # GAP 3 opt-in: bind a scalar recv-total counter; populated by the HT metadata
        # step at create_handle. Used later for the compute-view trim.
        recv_total = torch.zeros(1, dtype=torch.int32, device=a1.device)
        combine_weights = self._combine_weights(
            topk_weights, apply_router_weight_on_input
        )
        handle = self._make_handle(
            topk_ids,
            combine_weights,
            extra_knobs=[HandleAlgoKnobNumReceivedTokens(target=recv_total)],
        )
        d = handle.dispatch(DispatchInputParams(x=[a1]))
        self._recv_total = recv_total  # kept for a future actual_recv trim

        # d.expert_tensors is a 3D [world, max_per_rank, hidden] view of the token-major
        # FLAT recv; flatten to the [num_recv, hidden] Standard format.
        expert_x = d.expert_tensors.reshape(-1, self.hidden_size)

        # Return the received per-token routing (local expert ids, -1 = non-local).
        # NOTE(on-gpu): vLLM's Standard fused_moe must treat -1 picks as skipped.
        return expert_x, None, None, d.recv_topk_idx, d.recv_topk_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate), (
            "FlashInfer moe_ep HT reduces across ranks inside combine; weights were "
            "bound at dispatch."
        )
        from flashinfer.moe_ep import CombineInputParams

        handle = self._pop_handle()
        # HT combine consumes [num_recv, hidden]; it reshapes x to 2D internally, so a
        # contiguous [num_recv, hidden] (or a view that flattens to it) is expected.
        # NOTE(on-gpu): reconcile with the (M, topk, K) contract shape if the fused
        # experts return an un-reduced tensor for this path.
        comb_in = fused_expert_output.reshape(-1, self.hidden_size)
        try:
            handle.combine(CombineInputParams(x=[comb_in], out=output))
        finally:
            handle.complete()
