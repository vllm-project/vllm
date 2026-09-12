# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from typing import Any

import nccl.ep as nccl_ep
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed.device_communicators.all2all import NcclEPGroupState
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input,
    normalize_batched_scales_shape,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_enabled,
    dbo_maybe_run_recv_hook,
    dbo_switch_to_comm,
    dbo_switch_to_compute,
    dbo_switch_to_compute_sync,
    dbo_yield_and_switch_from_comm_to_compute,
    dbo_yield_and_switch_from_compute_to_comm,
)


class NcclEPLLPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Prepare/finalize using NCCL EP low-latency kernels."""

    def __init__(
        self,
        state: NcclEPGroupState,
        num_dispatchers: int,
        global_to_physical: torch.Tensor | None = None,
        physical_to_global: torch.Tensor | None = None,
        local_expert_global_ids: torch.Tensor | None = None,
    ):
        super().__init__()
        self.state = state
        self.num_dispatchers_ = num_dispatchers
        self.global_to_physical = self._to_topk_dtype(global_to_physical)
        self.physical_to_global = self._to_topk_dtype(physical_to_global)
        self.local_expert_global_ids = self._to_topk_dtype(local_expert_global_ids)

    @staticmethod
    def _to_topk_dtype(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.to(dtype=torch.int64)

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def max_num_tokens_per_rank(self) -> int | None:
        return self.state.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64

    def _map_global_to_physical_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        if self.global_to_physical is None:
            return topk_ids
        return self.global_to_physical[topk_ids]

    def _update_handle(self, topk_ids: torch.Tensor, stream: torch.cuda.Stream) -> None:
        topk_tensor = nccl_ep.Tensor(topk_ids)
        if self.state.handle is None:
            self.state.handle = self.state.group.create_handle(
                nccl_ep.Layout.EXPERT_MAJOR,
                topk_tensor,
                stream=stream,
            )
        else:
            self.state.handle.update(topk_tensor, stream=stream)

    @staticmethod
    def _quantize(
        expert_x: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_experts, _, hidden_dim = expert_x.shape
        q_dtype = quant_config.quant_dtype
        if q_dtype == "nvfp4":
            q_dtype = None

        expert_x = expert_x.view(-1, hidden_dim)
        expert_x, expert_x_scales = moe_kernel_quantize_input(
            expert_x,
            quant_config.a1_scale,
            q_dtype,
            quant_config.per_act_token_quant,
            quant_config.block_shape,
        )
        expert_x = expert_x.view(num_experts, -1, hidden_dim)
        if q_dtype is not None:
            assert expert_x_scales is not None
            expert_x_scales = normalize_batched_scales_shape(
                expert_x_scales, num_experts
            )
        return expert_x, expert_x_scales

    def supports_async(self) -> bool:
        return True

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> tuple[Callable, mk.ReceiverType]:
        if defer_input_quant:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support defer_input_quant=True. "
                "Please select an MoE kernel that accepts quantized inputs."
            )
        if a1.size(1) % 16 != 0:
            raise ValueError(
                "NCCL EP low-latency dispatch requires the hidden dimension "
                f"to be a multiple of 16, but got {a1.size(1)}."
            )
        if apply_router_weight_on_input:
            assert topk_ids.size(1) == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        stream = torch.cuda.current_stream(a1.device)
        dispatch_topk_ids = self._map_global_to_physical_ids(topk_ids).contiguous()
        self._update_handle(dispatch_topk_ids, stream)

        expert_x = torch.empty(
            (
                self.state.num_local_experts,
                self.num_dispatchers_ * self.state.max_tokens_per_rank,
                a1.size(1),
            ),
            dtype=a1.dtype,
            device=a1.device,
        )
        expert_num_tokens = torch.empty(
            self.state.num_local_experts,
            dtype=torch.int32,
            device=a1.device,
        )
        assert self.state.handle is not None
        dispatch_inputs = nccl_ep.DispatchInputs(tokens=nccl_ep.Tensor(a1))
        dispatch_outputs = nccl_ep.DispatchOutputs(tokens=nccl_ep.Tensor(expert_x))
        layout_info = nccl_ep.LayoutInfo(
            expert_counters=nccl_ep.Tensor(expert_num_tokens)
        )
        self.state.handle.dispatch(
            dispatch_inputs,
            dispatch_outputs,
            layout_info=layout_info,
            config=nccl_ep.DispatchConfig(send_only=1),
            stream=stream,
        )

        def receiver() -> mk.PrepareResultType:
            expert_x_quant, expert_x_scales = self._quantize(expert_x, quant_config)
            expert_tokens_meta = mk.ExpertTokensMetadata(
                expert_num_tokens=expert_num_tokens,
                expert_num_tokens_cpu=None,
            )
            return expert_x_quant, expert_x_scales, expert_tokens_meta, None, None

        def complete() -> None:
            self.state.handle.complete(stream=stream)
            _ = dispatch_inputs, dispatch_outputs, layout_info

        return complete, receiver

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
        hook, receiver = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )
        hook()
        return receiver()

    def _finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
        send_only: bool,
    ) -> tuple[Callable, Callable]:
        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate), (
            "Weight application and reduction happens in the combine kernel."
        )
        assert self.state.handle is not None
        combine_weights = (
            torch.ones_like(topk_weights)
            if apply_router_weight_on_input
            else topk_weights
        )
        stream = torch.cuda.current_stream(output.device)
        dbo_maybe_run_recv_hook()
        combine_inputs = nccl_ep.CombineInputs(
            tokens=nccl_ep.Tensor(fused_expert_output)
        )
        combine_outputs = nccl_ep.CombineOutputs(
            tokens=nccl_ep.Tensor(output),
            topk_weights=nccl_ep.Tensor(combine_weights),
        )
        self.state.handle.combine(
            combine_inputs,
            combine_outputs,
            config=nccl_ep.CombineConfig(send_only=int(send_only)),
            stream=stream,
        )

        def hook() -> None:
            if send_only:
                self.state.handle.complete(stream=stream)
            _ = combine_inputs, combine_outputs

        return hook, lambda: None

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> tuple[Callable, Callable]:
        return self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            send_only=True,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            send_only=False,
        )


class NcclEPHTPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Prepare/finalize using NCCL EP high-throughput FLAT kernels."""

    def __init__(
        self,
        state: NcclEPGroupState,
        num_dispatchers: int,
        rank_expert_offset: int,
        num_experts: int,
        num_ubatches: int = 1,
        use_fp8_dispatch: bool = False,
        global_to_physical: torch.Tensor | None = None,
        physical_to_global: torch.Tensor | None = None,
    ):
        super().__init__()
        self.state = state
        self.num_dispatchers_ = num_dispatchers
        self.rank_expert_offset = rank_expert_offset
        self.num_experts = num_experts
        self.use_fp8_dispatch = use_fp8_dispatch
        self.global_to_physical = NcclEPLLPrepareAndFinalize._to_topk_dtype(
            global_to_physical
        )
        self.physical_to_global = NcclEPLLPrepareAndFinalize._to_topk_dtype(
            physical_to_global
        )
        self.handles: list[tuple[Any, ...] | None] = [None] * max(1, num_ubatches)

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return self.state.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64

    def supports_async(self) -> bool:
        return True

    def _map_global_to_physical_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        if self.global_to_physical is None:
            return topk_ids
        return self.global_to_physical[topk_ids]

    def _receive(
        self,
        expert_x: torch.Tensor,
        expert_x_scale: torch.Tensor | None,
        recv_topk_ids: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> mk.PrepareResultType:
        if expert_x_scale is not None and quant_config.is_block_quantized:
            assert quant_config.block_shape is not None
            block_size = quant_config.block_shape[1]
            num_groups = (expert_x.size(1) + block_size - 1) // block_size
            expert_x_scale = expert_x_scale[:, :1].expand(-1, num_groups).contiguous()

        valid = recv_topk_ids >= 0
        physical_topk_ids = recv_topk_ids + self.rank_expert_offset
        invalid_physical_id = (
            self.num_experts - 1 if self.rank_expert_offset == 0 else 0
        )
        physical_topk_ids = torch.where(
            valid,
            physical_topk_ids,
            invalid_physical_id,
        )
        global_topk_ids = (
            physical_topk_ids
            if self.physical_to_global is None
            else self.physical_to_global[physical_topk_ids]
        )
        if expert_x_scale is None and not defer_input_quant and expert_x.numel():
            q_dtype = quant_config.quant_dtype
            if q_dtype == "nvfp4":
                q_dtype = None
            expert_x, expert_x_scale = moe_kernel_quantize_input(
                expert_x,
                quant_config.a1_scale,
                q_dtype,
                quant_config.per_act_token_quant,
                quant_config.block_shape,
            )

        return (
            expert_x,
            expert_x_scale,
            None,
            global_topk_ids,
            recv_topk_weights,
        )

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.ReceiverType:
        if apply_router_weight_on_input:
            assert topk_ids.size(1) == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        use_fp8 = (
            self.use_fp8_dispatch
            and quant_config.is_block_quantized
            and not defer_input_quant
        )
        if use_fp8:
            tokens, token_scales = per_token_group_quant_fp8(
                a1,
                group_size=a1.size(1),
                dtype=torch.float8_e4m3fn,
            )
            token_scales = token_scales.view(a1.size(0), 1).expand(-1, 4).contiguous()
        else:
            tokens, token_scales = a1, None

        stream = torch.cuda.current_stream(a1.device)
        dbo_yield_and_switch_from_compute_to_comm()
        dispatch_topk_ids = self._map_global_to_physical_ids(topk_ids).contiguous()
        recv_total_counter = torch.empty(1, dtype=torch.int32, device=a1.device)
        layout_info = nccl_ep.LayoutInfo(
            recv_total_counter=nccl_ep.Tensor(recv_total_counter),
            recv_topk_idx_kind=nccl_ep.ExpertIdKind.LOCAL,
        )
        handle = self.state.group.create_handle(
            nccl_ep.Layout.FLAT,
            nccl_ep.Tensor(dispatch_topk_ids),
            layout_info=layout_info,
            stream=stream,
        )
        a2a_idx = dbo_current_ubatch_id()
        assert self.handles[a2a_idx] is None

        max_recv = int(recv_total_counter.item())
        max_recv_upper_bound = self.state.max_tokens_per_rank * self.num_dispatchers_
        if not 0 <= max_recv <= max_recv_upper_bound:
            handle.destroy()
            raise RuntimeError(
                "NCCL EP returned an invalid receive-token count: "
                f"{max_recv} is outside [0, {max_recv_upper_bound}]."
            )
        recv_tokens = torch.empty(
            (max_recv, tokens.size(1)), dtype=tokens.dtype, device=tokens.device
        )
        recv_topk_ids = torch.empty(
            (max_recv, topk_ids.size(1)),
            dtype=torch.int64,
            device=topk_ids.device,
        )
        recv_topk_weights = torch.empty(
            (max_recv, topk_weights.size(1)),
            dtype=torch.float32,
            device=topk_weights.device,
        )
        recv_scales = (
            torch.empty(
                (max_recv, token_scales.size(1)),
                dtype=token_scales.dtype,
                device=token_scales.device,
            )
            if token_scales is not None
            else None
        )
        dispatch_inputs = nccl_ep.DispatchInputs(
            tokens=nccl_ep.Tensor(tokens),
            topk_weights=nccl_ep.Tensor(topk_weights),
            scales=(nccl_ep.Tensor(token_scales) if token_scales is not None else None),
        )
        dispatch_outputs = nccl_ep.DispatchOutputs(
            tokens=nccl_ep.Tensor(recv_tokens),
            topk_weights=nccl_ep.Tensor(recv_topk_weights),
            topk_idx=nccl_ep.Tensor(recv_topk_ids),
            scales=nccl_ep.Tensor(recv_scales) if recv_scales is not None else None,
        )
        handle.dispatch(
            dispatch_inputs,
            dispatch_outputs,
            config=nccl_ep.DispatchConfig(
                quantization_recipe=(
                    nccl_ep.DispatchQuantizationRecipe.FWD
                    if token_scales is not None
                    else nccl_ep.DispatchQuantizationRecipe.NONE
                )
            ),
            stream=stream,
        )
        self.handles[a2a_idx] = (
            handle,
            dispatch_topk_ids,
            recv_total_counter,
            layout_info,
            dispatch_inputs,
            dispatch_outputs,
        )
        dbo_switch_to_compute_sync()
        return lambda: self._receive(
            recv_tokens,
            recv_scales,
            recv_topk_ids,
            recv_topk_weights,
            quant_config,
            defer_input_quant,
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
        receiver = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )
        return receiver()

    def _finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
        do_async: bool,
    ) -> Callable | None:
        a2a_idx = dbo_current_ubatch_id()
        handle_state = self.handles[a2a_idx]
        assert handle_state is not None
        handle = handle_state[0]
        if fused_expert_output.numel():
            if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
                weight_and_reduce_impl = TopKWeightAndReduceContiguous()
            fused_expert_output = weight_and_reduce_impl.apply(
                None,
                fused_expert_output,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
            )

        stream = torch.cuda.current_stream(output.device)
        dbo_yield_and_switch_from_compute_to_comm()
        combined_output = torch.empty_like(output)
        combine_inputs = nccl_ep.CombineInputs(
            tokens=nccl_ep.Tensor(fused_expert_output)
        )
        combine_outputs = nccl_ep.CombineOutputs(tokens=nccl_ep.Tensor(combined_output))
        handle.combine(
            combine_inputs,
            combine_outputs,
            config=nccl_ep.CombineConfig(),
            stream=stream,
        )
        dbo_switch_to_compute()

        def receiver() -> None:
            if do_async:
                dbo_switch_to_comm()
            output.copy_(combined_output, non_blocking=True)
            handle.destroy()
            self.handles[a2a_idx] = None
            _ = combine_inputs, combine_outputs, handle_state
            if do_async:
                dbo_yield_and_switch_from_comm_to_compute()

        if do_async:
            return receiver
        receiver()
        return None

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> Callable:
        receiver = self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            True,
        )
        assert receiver is not None
        return receiver

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        assert not dbo_enabled()
        self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            False,
        )
