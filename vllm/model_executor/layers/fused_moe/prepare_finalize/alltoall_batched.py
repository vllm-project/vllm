# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig


class AllToAllBatchedPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Routed all-to-all EP dispatch/combine in the batched activation format,
    over fixed-capacity all_to_all_single collectives. A token is sent to a
    destination rank once, however many of its experts live there, carrying
    its local expert ids and router weights as metadata.
    """

    def __init__(
        self,
        max_num_tokens: int,
        num_local_experts: int,
        num_dispatchers: int,
        rank: int,
        is_sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.max_num_tokens = max_num_tokens
        self.num_local_experts = num_local_experts
        self._num_dispatchers = num_dispatchers
        self.rank = rank
        self.is_sequence_parallel = is_sequence_parallel
        self._combine_ctx: dict | None = None

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_num_tokens

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self._num_dispatchers

    def capacities(self, topk: int) -> tuple[int, int, int]:
        """Returns token slots per rank, expert slots per token, rows per expert."""
        tok_cap = self.max_num_tokens
        slots_per_token = min(topk, self.num_local_experts)
        expert_cap = self.max_num_tokens * self._num_dispatchers
        return tok_cap, slots_per_token, expert_cap

    def output_is_reduced(self) -> bool:
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
        h = a1.size(1)
        num_tokens = a1.size(0)
        topk = topk_ids.size(1)
        dev = a1.device

        assert num_tokens <= self.max_num_tokens, (
            f"{num_tokens} tokens exceeds max_num_tokens={self.max_num_tokens}"
        )
        cap, slots, expert_cap = self.capacities(topk)

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

        # Deduplicate (dest, token) pairs so send_x stays token-sized.
        order = torch.argsort(dest * (num_tokens + 1) + flat_token)
        d_sorted, t_sorted = dest[order], flat_token[order]
        new_slot = torch.ones_like(t_sorted, dtype=torch.bool)
        new_slot[1:] = (d_sorted[1:] != d_sorted[:-1]) | (t_sorted[1:] != t_sorted[:-1])
        slot_of_pair = new_slot.cumsum(0) - 1
        uniq_dest = d_sorted[new_slot]
        uniq_token = t_sorted[new_slot]
        base = torch.zeros(world + 1, dtype=torch.int64, device=dev)
        base[1:] = torch.bincount(uniq_dest, minlength=world).cumsum(0)
        send_cnt = base[1:] - base[:-1]
        local_slot = torch.arange(uniq_dest.numel(), device=dev) - base[uniq_dest]

        send_x = torch.zeros((world, cap, h), dtype=a1.dtype, device=dev)
        send_x[uniq_dest, local_slot] = a1.index_select(0, uniq_token)

        # `pick` indexes a pair within its (dest, token) group, so the k-th
        # expert a token wants on a rank lands in column k; -1 marks unused.
        pair_ar = torch.arange(slot_of_pair.numel(), device=dev)
        group_start = pair_ar[new_slot]
        pair_slot = local_slot[slot_of_pair]
        pick = pair_ar - group_start[slot_of_pair]
        send_eloc = torch.full((world, cap, slots), -1, dtype=torch.int64, device=dev)
        send_eloc[d_sorted, pair_slot, pick] = flat_expert[order] - d_sorted * e_local

        send_w = torch.zeros((world, cap, slots), dtype=torch.float32, device=dev)
        send_w[d_sorted, pair_slot, pick] = flat_weight[order].to(torch.float32)

        # Fixed-size collectives, no host sync.
        send_meta_pkt = torch.empty(
            (world, cap * slots + 1), dtype=torch.int64, device=dev
        )
        send_meta_pkt[:, 0] = send_cnt
        send_meta_pkt[:, 1:] = send_eloc.reshape(world, cap * slots)
        recv_x = comm.all_to_all_single(send_x.reshape(world, cap * h)).reshape(
            world, cap, h
        )
        recv_meta_pkt = comm.all_to_all_single(send_meta_pkt)
        recv_w = comm.all_to_all_single(send_w.reshape(world, cap * slots)).reshape(
            world, cap, slots
        )
        recv_cnt = recv_meta_pkt[:, 0].contiguous()
        recv_eloc = recv_meta_pkt[:, 1:].reshape(world, cap, slots)

        # Expand back to one row per (token, expert) pair.
        slot_ar = torch.arange(cap, device=dev)
        recv_valid = (slot_ar.view(1, -1) < recv_cnt.view(-1, 1)).unsqueeze(-1) & (
            recv_eloc >= 0
        )
        vsrc, vslot, vpick = recv_valid.nonzero(as_tuple=True)
        rows_x = recv_x[vsrc, vslot]
        rows_e = recv_eloc[vsrc, vslot, vpick].clamp_(0, e_local - 1)
        rows_w = recv_w[vsrc, vslot, vpick]
        n_recv = rows_x.size(0)

        b_a1 = torch.zeros((e_local, expert_cap, h), dtype=a1.dtype, device=dev)
        tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32, device=dev)
        if n_recv > 0:
            local_ids = torch.arange(e_local, device=dev).view(-1, 1)
            hits = rows_e.view(1, -1) == local_ids
            slots_all = hits.to(torch.int64).cumsum(dim=1) - 1
            row_slot = slots_all[rows_e, torch.arange(n_recv, device=dev)]
            keep = row_slot < expert_cap
            b_a1[rows_e[keep], row_slot[keep]] = rows_x[keep]
            tokens_per_expert[:e_local] = (
                hits.sum(dim=1).clamp(max=expert_cap).to(torch.int32)
            )
        else:
            row_slot = torch.zeros(0, dtype=torch.int64, device=dev)
            keep = torch.zeros(0, dtype=torch.bool, device=dev)

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=tokens_per_expert, expert_num_tokens_cpu=None
        )

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
            "rows_w": rows_w,
            "row_slot": row_slot,
            "keep": keep,
            "n_recv": n_recv,
            "send_cnt": send_cnt,
            "uniq_dest": uniq_dest,
            "uniq_token": uniq_token,
            "local_slot": local_slot,
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

        # Weight and reduce a token's experts on the owning rank, so the return
        # transfer is token-sized like the dispatch.
        recv_out = torch.zeros((world, cap, h), dtype=torch.float32, device=dev)
        keep = ctx["keep"]
        if ctx["n_recv"] > 0 and keep.any():
            rows_out = fused_expert_output[ctx["rows_e"][keep], ctx["row_slot"][keep]]
            rows_out = rows_out.to(torch.float32) * ctx["rows_w"][keep].unsqueeze(1)
            recv_out.index_put_(
                (ctx["vsrc"][keep], ctx["vslot"][keep]), rows_out, accumulate=True
            )

        send_out = comm.all_to_all_single(
            recv_out.to(dtype).reshape(world, cap * h)
        ).reshape(world, cap, h)

        # Sum each token's per-rank partials.
        output.zero_()
        output.index_add_(
            0,
            ctx["uniq_token"],
            send_out[ctx["uniq_dest"], ctx["local_slot"]].to(output.dtype),
        )
        self._combine_ctx = None
