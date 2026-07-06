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
    TopKWeightAndReduceNoOP,
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
        # Static local->global expert-id remap cache (built on first prepare()).
        self._l2g_cache: tuple | None = None
        # Persistent full-size combine input buffer (recv-trim support).
        self._comb_full: torch.Tensor | None = None
        self._num_recv_full: int = 0
        self._trim_m: int = 0

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
        self._recv_total = recv_total

        # d.expert_tensors is a 3D [world, max_per_rank, hidden] view of the token-major
        # FLAT recv; flatten to the [num_recv, hidden] Standard format.
        expert_x = d.expert_tensors.reshape(-1, self.hidden_size)
        self._num_recv_full = expert_x.shape[0]

        # FlashInfer HT FLAT recv carries LOCAL expert ids with -1 for non-local /
        # padding picks. vLLM's Standard experts, however, feed topk_ids straight
        # into moe_align_block_size + expert_map[...] expecting *global* ids in
        # [0, num_experts): the skip for non-local picks is done via expert_map
        # (global->local, -1), never by a -1 in topk_ids. Passing local ids (and
        # especially -1) corrupts the alignment histogram and OOBs expert_map[...]
        # (illegal memory access). Rebuild global ids from expert_map so this path
        # matches the DeepEP HT contract: valid local picks -> their global id;
        # non-local (-1) picks -> a non-owned global id, which expert_map then
        # re-tags -1 (skipped by the experts).
        recv_idx = d.recv_topk_idx
        recv_w = d.recv_topk_weights
        if expert_map is not None:
            # The local->global mapping is static for this layer; cache it (the
            # nonzero() calls each force a device sync, twice per layer per step).
            cached = self._l2g_cache
            if cached is None or cached[0] is not expert_map:
                owned = (expert_map >= 0).nonzero(as_tuple=True)[0]  # global ids
                local_to_global = torch.empty(
                    self._num_local_experts,
                    dtype=recv_idx.dtype,
                    device=recv_idx.device,
                )
                local_to_global[expert_map[owned]] = owned.to(recv_idx.dtype)
                not_owned = (expert_map < 0).nonzero(as_tuple=True)[0]
                skip_id = (not_owned[0] if not_owned.numel() else owned[0]).to(
                    recv_idx.dtype
                )
                self._l2g_cache = (expert_map, local_to_global, skip_id)
            _, local_to_global, skip_id = self._l2g_cache
            valid = recv_idx >= 0
            recv_idx = torch.where(
                valid, local_to_global[recv_idx.clamp_min(0)], skip_id
            )

        # Trim the compute view to the actually-received rows. The recv buffer is
        # statically sized to max_tokens_per_rank * world (65k+ rows), but a step
        # typically receives a tiny fraction — without the trim every downstream
        # kernel (moe_align, sort, GEMMs, activation, moe_sum) pads to the full
        # buffer. recv_total was written by the HT metadata step at create_handle
        # (GAP 3); .item() is a host sync — eager-only, matching this path (the
        # nccl.ep HT combine below still gets the full static buffer).
        actual = int(recv_total.item())
        m = self._num_recv_full
        if 0 < actual < m:
            # Round up for allocator/kernel-config stability; padded rows within
            # the round-up carry topk_idx -1 -> skip_id -> expert_map -1 (masked).
            m = min(self._num_recv_full, -(-actual // 128) * 128)
            expert_x = expert_x[:m]
            recv_idx = recv_idx[:m]
            recv_w = recv_w[:m]
        self._trim_m = m

        # NOTE(on-gpu): expert_map re-tags the non-owned skip id to -1; the Standard
        # experts then contribute 0 for those picks, matching combine's masking.
        return expert_x, None, None, recv_idx, recv_w

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        # The Standard experts either delegate weight+reduce to us
        # (TopKWeightAndReduceDelegate) or have already applied the dispatched
        # per-token routing weights and reduced their local picks
        # (TopKWeightAndReduceNoOP, e.g. the Triton experts). Both are correct here:
        # FlashInfer HT combine applies NO weights (routing weights were captured at
        # dispatch and consumed by the experts) and only reduces the per-rank partial
        # sums across ranks — so there is no double weighting in either case.
        assert isinstance(
            weight_and_reduce_impl,
            (TopKWeightAndReduceDelegate, TopKWeightAndReduceNoOP),
        ), (
            "FlashInfer moe_ep HT combine reduces per-rank partials across ranks "
            "(weights bound at dispatch); the experts must not request a different "
            f"weight/reduce, got {type(weight_and_reduce_impl).__name__}."
        )
        from flashinfer.moe_ep import CombineInputParams

        handle = self._pop_handle()
        # HT combine consumes the FULL static [num_recv, hidden] buffer (the library
        # sized its staging to max_recv and, in cached mode, expects address-stable
        # buffers). When prepare() trimmed the compute view, copy the expert output
        # for the valid rows into a persistent full-size buffer; rows beyond the
        # actual recv count carry no routing state and are never sent by combine.
        comb_in = fused_expert_output.reshape(-1, self.hidden_size)
        if comb_in.shape[0] < self._num_recv_full:
            full = self._comb_full
            if full is None or full.shape[0] != self._num_recv_full:
                full = torch.empty(
                    self._num_recv_full,
                    self.hidden_size,
                    dtype=comb_in.dtype,
                    device=comb_in.device,
                )
                self._comb_full = full
            full[: comb_in.shape[0]].copy_(comb_in)
            comb_in = full
        try:
            handle.combine(CombineInputParams(x=[comb_in], out=output))
        finally:
            handle.complete()
