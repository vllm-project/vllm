# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-latency routed all-to-all prepare/finalize for the batched Triton MoE.

Routes each (token, expert) to the rank that owns the expert, into the
[E_local, max_num_tokens, K] batched format, via fixed-capacity
all_to_all_single collectives. Tokens over the per-rank capacity are dropped.
Linear expert placement: expert e lives on rank e // num_local_experts.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

logger = init_logger(__name__)


class NaiveLowLatencyPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Routed all-to-all EP dispatch/combine in the batched activation format."""

    def __init__(
        self,
        max_num_tokens: int,
        num_local_experts: int,
        num_dispatchers: int,
        rank: int,
        is_sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        # Per-rank capacity: batched buffer token dim and per-dest send capacity.
        # Not scaled by world size; overflow tokens are dropped.
        self.cap = max_num_tokens
        self.num_local_experts = num_local_experts
        self._num_dispatchers = num_dispatchers
        self.rank = rank
        self.is_sequence_parallel = is_sequence_parallel
        self._combine_ctx: dict | None = None

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def max_num_tokens_per_rank(self) -> int | None:
        return self.cap

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self._num_dispatchers

    def output_is_reduced(self) -> bool:
        # finalize() returns this rank's fully-combined tokens; no further
        # cross-rank reduction is needed (same as DeepEP-LL).
        return True

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
        if quant_config.quant_dtype is not None or defer_input_quant:
            raise NotImplementedError(
                f"{self.__class__.__name__} supports unquantized MoE only."
            )

        comm = get_ep_group().device_communicator
        assert comm is not None
        world = self._num_dispatchers
        e_local = self.num_local_experts
        cap = self.cap
        h = a1.size(1)
        num_tokens = a1.size(0)
        topk = topk_ids.size(1)
        dev = a1.device

        if apply_router_weight_on_input:
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        flat_expert = topk_ids.reshape(-1).to(torch.int64)
        flat_token = torch.arange(
            num_tokens, device=dev, dtype=torch.int64
        ).repeat_interleave(topk)
        flat_weight = topk_weights.reshape(-1)
        if apply_router_weight_on_input:
            flat_weight = torch.ones_like(flat_weight)
        dest = torch.div(flat_expert, e_local, rounding_mode="floor")

        # Fixed-capacity send buffers: one [cap] block per dest rank.
        send_x = torch.zeros((world, cap, h), dtype=a1.dtype, device=dev)
        send_eloc = torch.full((world, cap), -1, dtype=torch.int64, device=dev)
        send_meta = torch.full((world, cap), -1, dtype=torch.int64, device=dev)
        send_cnt = torch.zeros(world, dtype=torch.int64, device=dev)
        for r in range(world):
            idx = (dest == r).nonzero(as_tuple=True)[0][:cap]
            n = idx.numel()
            if n == 0:
                continue
            send_cnt[r] = n
            send_x[r, :n] = a1.index_select(0, flat_token[idx])
            send_eloc[r, :n] = flat_expert[idx] - r * e_local
            send_meta[r, :n] = idx

        # Fixed-size all-to-all (desync-tolerant). Fuse the two small int
        # metadata collectives (send_eloc + send_cnt) into a single
        # [world, cap + 1] packet: column 0 carries the per-dest count, the
        # rest carries the per-slot local expert ids. Two collectives instead
        # of three, no extra host sync (all shapes are static).
        send_meta_pkt = torch.empty((world, cap + 1), dtype=torch.int64, device=dev)
        send_meta_pkt[:, 0] = send_cnt
        send_meta_pkt[:, 1:] = send_eloc
        recv_x = comm.all_to_all_single(send_x.reshape(world, cap * h)).reshape(
            world, cap, h
        )
        recv_meta_pkt = comm.all_to_all_single(send_meta_pkt)
        recv_cnt = recv_meta_pkt[:, 0].contiguous()
        recv_eloc = recv_meta_pkt[:, 1:]

        # Flatten valid received rows (first recv_cnt[s] slots per source).
        slot_ar = torch.arange(cap, device=dev)
        recv_valid = slot_ar.view(1, -1) < recv_cnt.view(-1, 1)
        vsrc, vslot = recv_valid.nonzero(as_tuple=True)
        rows_x = recv_x[vsrc, vslot]
        rows_e = recv_eloc[vsrc, vslot].clamp_(0, e_local - 1)
        n_recv = rows_x.size(0)

        # Bucket into [E_local, cap, H] in received order (capped at cap).
        b_a1 = torch.zeros((e_local, cap, h), dtype=a1.dtype, device=dev)
        tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32, device=dev)
        if n_recv > 0:
            local_ids = torch.arange(e_local, device=dev).view(-1, 1)
            hits = rows_e.view(1, -1) == local_ids
            slots_all = hits.to(torch.int64).cumsum(dim=1) - 1
            row_slot = slots_all[rows_e, torch.arange(n_recv, device=dev)]
            keep = row_slot < cap
            b_a1[rows_e[keep], row_slot[keep]] = rows_x[keep]
            tokens_per_expert[:e_local] = hits.sum(dim=1).clamp(max=cap).to(torch.int32)
        else:
            row_slot = torch.zeros(0, dtype=torch.int64, device=dev)
            keep = torch.zeros(0, dtype=torch.bool, device=dev)

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=tokens_per_expert, expert_num_tokens_cpu=None
        )

        # Stash geometry so finalize() can reverse the dispatch (same idiom as
        # DeepEP-LL storing its dispatch handle).
        self._combine_ctx = {
            "world": world,
            "cap": cap,
            "h": h,
            "dtype": a1.dtype,
            "device": dev,
            "num_tokens": num_tokens,
            "vsrc": vsrc,
            "vslot": vslot,
            "rows_e": rows_e,
            "row_slot": row_slot,
            "keep": keep,
            "n_recv": n_recv,
            "send_cnt": send_cnt,
            "send_meta": send_meta,
            "flat_token": flat_token,
            "flat_weight": flat_weight,
        }
        return b_a1, None, expert_tokens_meta, None, None

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        ctx = self._combine_ctx
        assert ctx is not None, "finalize() called before prepare()"
        comm = get_ep_group().device_communicator
        assert comm is not None
        world, cap, h = ctx["world"], ctx["cap"], ctx["h"]
        dev, dtype = ctx["device"], ctx["dtype"]

        # Un-bucket expert output back into received-row order.
        rows_out = torch.zeros((ctx["n_recv"], h), dtype=dtype, device=dev)
        keep = ctx["keep"]
        if ctx["n_recv"] > 0 and keep.any():
            rows_out[keep] = fused_expert_output[
                ctx["rows_e"][keep], ctx["row_slot"][keep]
            ]

        # Scatter back into a [world, cap, H] per-source buffer.
        recv_out = torch.zeros((world, cap, h), dtype=dtype, device=dev)
        if ctx["n_recv"] > 0:
            recv_out[ctx["vsrc"], ctx["vslot"]] = rows_out

        # Reverse all-to-all: return each rank's tokens to their origin.
        send_out = comm.all_to_all_single(recv_out.reshape(world, cap * h)).reshape(
            world, cap, h
        )

        # Apply router weights and reduce over topk into the local tokens.
        slot_ar = torch.arange(cap, device=dev)
        send_valid = slot_ar.view(1, -1) < ctx["send_cnt"].view(-1, 1)
        gsrc, gslot = send_valid.nonzero(as_tuple=True)
        rows = send_out[gsrc, gslot]
        meta = ctx["send_meta"][gsrc, gslot]
        origin_token = ctx["flat_token"][meta]
        weight = ctx["flat_weight"][meta].to(rows.dtype)

        output.zero_()
        if rows.size(0) > 0:
            output.index_add_(
                0, origin_token, (rows * weight.unsqueeze(1)).to(output.dtype)
            )
        self._combine_ctx = None
