# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Author: Vamsi Addanki @STyGIANet
# Nothing fancy, just plain torch distributed alltoall
# In view of being agnostic to Pcie,nvlink,rdma... NCCL takes care of it.

from dataclasses import dataclass

import torch
import torch.distributed as dist

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.utils.flashinfer import nvfp4_block_scale_interleave


@dataclass
class _NcclAllToAllHandle:
    send_counts: list[int]
    recv_counts: list[int]
    sent_token_indices: torch.Tensor
    sent_topk_weights: torch.Tensor


def _quantize_and_setup_dispatch(
    a1: torch.Tensor,
    quant_config: FusedMoEQuantConfig | None,
    defer_input_quant: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if defer_input_quant or quant_config is None:
        return a1, None

    input_sf = quant_config.a1_scale
    a1q, a1q_scale = moe_kernel_quantize_input(
        a1,
        input_sf,
        quant_dtype=quant_config.quant_dtype,
        per_act_token_quant=quant_config.per_act_token_quant,
        block_shape=quant_config.block_shape,
        is_scale_swizzled=False,
    )
    return a1q, a1q_scale


def _prepare_scales_for_moe(
    a1q_scale: torch.Tensor | None,
    quant_config: FusedMoEQuantConfig | None,
) -> torch.Tensor | None:
    if a1q_scale is None or quant_config is None:
        return a1q_scale

    if quant_config.quant_dtype == "nvfp4" and quant_config.is_scale_swizzled:
        if a1q_scale.element_size() == 1:
            a1q_scale = a1q_scale.view(torch.uint8)
        a1q_scale = nvfp4_block_scale_interleave(a1q_scale)
    return a1q_scale


class NcclAllToAllPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Real routed all2all implemented with torch.distributed.all_to_all_single.

    This keeps the integration local to the modular MoE path and avoids the
    extra connection/bootstrap machinery used by custom transports.
    """

    def __init__(
        self,
        num_dispatchers: int,
        global_to_physical: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_dispatchers_ = num_dispatchers
        self.global_to_physical = global_to_physical
        self.handle: _NcclAllToAllHandle | None = None
        self._validated_num_experts: int | None = None
        self._cached_expert_map_key: tuple[int, int] | None = None
        self._cached_local_num_experts: int | None = None

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    def _map_global_to_physical_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        if self.global_to_physical is None:
            return topk_ids
        return self.global_to_physical[topk_ids.long()].to(dtype=topk_ids.dtype)

    def _physical_expert_to_rank(
        self, physical_ids: torch.Tensor, num_experts: int
    ) -> torch.Tensor:
        world_size = self.num_dispatchers_
        base = num_experts // world_size
        remainder = num_experts % world_size
        if base == 0:
            return physical_ids.long()
        if remainder == 0:
            return torch.div(physical_ids.long(), base, rounding_mode="floor")
        split = (base + 1) * remainder
        physical_ids_long = physical_ids.long()
        return torch.where(
            physical_ids_long < split,
            torch.div(physical_ids_long, base + 1, rounding_mode="floor"),
            remainder
            + torch.div(physical_ids_long - split, base, rounding_mode="floor"),
        )

    def _validate_static_configuration(self, num_experts: int) -> None:
        if self._validated_num_experts == num_experts:
            return
        if self.global_to_physical is not None:
            max_physical = int(self.global_to_physical.max().item())
            if max_physical >= num_experts:
                raise NotImplementedError(
                    "nccl_alltoall does not support physical expert ids outside "
                    "the [0, num_experts) range. Redundant EPLB experts are not "
                    "supported by this backend yet."
                )
        self._validated_num_experts = num_experts

    def _local_num_experts(
        self, expert_map: torch.Tensor | None, num_experts: int
    ) -> int:
        if expert_map is None:
            return num_experts
        key = (expert_map.data_ptr(), expert_map.numel())
        if key != self._cached_expert_map_key:
            self._cached_expert_map_key = key
            self._cached_local_num_experts = int(
                torch.count_nonzero(expert_map >= 0).item()
            )
        assert self._cached_local_num_experts is not None
        return self._cached_local_num_experts

    def _all_to_all_counts(self, send_counts_tensor: torch.Tensor) -> torch.Tensor:
        ep_group = get_ep_group()
        group = ep_group.device_group
        assert group is not None
        recv_counts_tensor = torch.empty_like(send_counts_tensor)
        dist.all_to_all_single(recv_counts_tensor, send_counts_tensor, group=group)
        return recv_counts_tensor

    def _all_to_all_tensor(
        self,
        input_tensor: torch.Tensor,
        send_counts: list[int],
        recv_counts: list[int],
        async_op: bool = False,
    ) -> tuple[torch.Tensor, dist.Work | None]:
        group = get_ep_group().device_group
        assert group is not None
        output_shape = (sum(recv_counts),) + input_tensor.shape[1:]
        output = torch.empty(
            output_shape,
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )
        work = dist.all_to_all_single(
            output,
            input_tensor.contiguous(),
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=group,
            async_op=async_op,
        )
        return output, work

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig | None,
        defer_input_quant: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        mk.ExpertTokensMetadata,
        torch.Tensor,
        torch.Tensor,
    ]:
        self._validate_static_configuration(num_experts)

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        a1q, a1q_scale = _quantize_and_setup_dispatch(
            a1, quant_config, defer_input_quant
        )

        route_topk_ids = self._map_global_to_physical_ids(topk_ids)
        flat_route_topk_ids = route_topk_ids.reshape(-1)
        flat_topk_ids = topk_ids.reshape(-1, 1)
        flat_topk_weights = topk_weights.reshape(-1, 1)
        topk = topk_ids.size(1)

        if topk == 1:
            dest_ranks = self._physical_expert_to_rank(
                route_topk_ids.view(-1), num_experts
            )
            order = torch.argsort(dest_ranks)
            send_hidden_states = a1q.index_select(0, order)
            send_topk_ids = flat_topk_ids.index_select(0, order)
            sent_token_indices = order
            sent_topk_weights = flat_topk_weights.index_select(0, order)
            send_a1q_scale = None
            if a1q_scale is not None and a1q_scale.ndim != 0:
                send_a1q_scale = a1q_scale.index_select(0, order)
        else:
            token_indices = torch.arange(
                a1.size(0), device=a1.device, dtype=torch.int64
            )
            flat_token_indices = token_indices.view(-1, 1).expand_as(topk_ids)
            flat_token_indices = flat_token_indices.reshape(-1)
            dest_ranks = self._physical_expert_to_rank(
                flat_route_topk_ids, num_experts
            )
            order = torch.argsort(dest_ranks)
            ordered_token_indices = flat_token_indices.index_select(0, order)
            send_hidden_states = a1q.index_select(0, ordered_token_indices)
            send_topk_ids = flat_topk_ids.index_select(0, order)
            sent_token_indices = ordered_token_indices
            sent_topk_weights = flat_topk_weights.index_select(0, order)
            send_a1q_scale = None
            if a1q_scale is not None and a1q_scale.ndim != 0:
                send_a1q_scale = a1q_scale.index_select(0, ordered_token_indices)

        send_counts_tensor = torch.bincount(
            dest_ranks, minlength=self.num_dispatchers_
        )
        recv_counts_tensor = self._all_to_all_counts(send_counts_tensor)
        send_counts = send_counts_tensor.cpu().tolist()
        recv_counts = recv_counts_tensor.cpu().tolist()

        recv_hidden_states, hidden_work = self._all_to_all_tensor(
            send_hidden_states, send_counts, recv_counts, async_op=True
        )
        recv_topk_ids, topk_ids_work = self._all_to_all_tensor(
            send_topk_ids, send_counts, recv_counts, async_op=True
        )
        recv_a1q_scale = None
        scales_work = None
        if send_a1q_scale is not None:
            recv_a1q_scale, scales_work = self._all_to_all_tensor(
                send_a1q_scale, send_counts, recv_counts, async_op=True
            )
        else:
            recv_a1q_scale = a1q_scale

        if hidden_work is not None:
            hidden_work.wait()
        if topk_ids_work is not None:
            topk_ids_work.wait()
        if scales_work is not None:
            scales_work.wait()

        recv_a1q_scale = _prepare_scales_for_moe(recv_a1q_scale, quant_config)

        local_num_experts = self._local_num_experts(expert_map, num_experts)
        if expert_map is None:
            local_expert_ids = recv_topk_ids.view(-1)
        else:
            local_expert_ids = expert_map[recv_topk_ids.view(-1).long()].long()

        expert_num_tokens = torch.bincount(
            local_expert_ids, minlength=local_num_experts
        )
        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens.to(
                device=recv_hidden_states.device, dtype=torch.int32
            ),
            expert_num_tokens_cpu=expert_num_tokens.to(dtype=torch.int32).cpu(),
        )

        self.handle = _NcclAllToAllHandle(
            send_counts=send_counts,
            recv_counts=recv_counts,
            sent_token_indices=sent_token_indices,
            sent_topk_weights=sent_topk_weights,
        )

        return (
            recv_hidden_states,
            recv_a1q_scale,
            expert_tokens_meta,
            recv_topk_ids,
            (
                torch.ones_like(recv_topk_ids, dtype=topk_weights.dtype)
                if apply_router_weight_on_input
                else torch.ones_like(recv_topk_ids, dtype=topk_weights.dtype)
            ),
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: TopKWeightAndReduceDelegate,
    ) -> None:
        assert self.handle is not None

        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()

        local_output = weight_and_reduce_impl.apply(
            output=None,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
        if local_output.ndim == 3 and local_output.size(1) == 1:
            local_output = local_output.squeeze(1)

        recv_output, reverse_work = self._all_to_all_tensor(
            local_output,
            self.handle.recv_counts,
            self.handle.send_counts,
            async_op=True,
        )
        output.zero_()
        if reverse_work is not None:
            reverse_work.wait()

        if not apply_router_weight_on_input:
            recv_output.mul_(self.handle.sent_topk_weights.view(-1, 1))

        output.index_add_(0, self.handle.sent_token_indices, recv_output)
        self.handle = None
