# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import nccl.core as nccl_core  # type: ignore[import-not-found]
import nccl.ep as nccl_ep  # type: ignore[import-not-found]
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_enabled,
    dbo_switch_to_comm,
    dbo_switch_to_compute,
    dbo_switch_to_compute_sync,
    dbo_yield_and_switch_from_comm_to_compute,
    dbo_yield_and_switch_from_compute_to_comm,
)

logger = init_logger(__name__)

_TORCH_TO_NCCL_DTYPE = {
    torch.bfloat16: nccl_core.BFLOAT16,
    torch.float16: nccl_core.FLOAT16,
    torch.float32: nccl_core.FLOAT32,
    torch.int64: nccl_core.INT64,
    torch.int32: nccl_core.INT32,
    torch.float8_e4m3fn: nccl_core.UINT8,
}


def _to_nccl_tensor(t: torch.Tensor) -> nccl_ep.Tensor:
    dtype = _TORCH_TO_NCCL_DTYPE.get(t.dtype)
    assert dtype is not None, f"Unsupported dtype {t.dtype} for NCCL EP"
    return nccl_ep.Tensor(t.data_ptr(), dtype=int(dtype), shape=tuple(t.shape))


def _get_cuda_stream():
    return torch.cuda.current_stream().cuda_stream


class NcclEPStandardPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Prepare/Finalize using NCCL EP with FLAT layout
    (Standard activation format).
    """

    def __init__(
        self,
        ep_group: nccl_ep.Group,
        num_dispatchers: int,
        dp_size: int,
        rank_expert_offset: int,
        num_experts: int,
        num_topk: int,
        use_fp8_dispatch: bool = False,
    ):
        super().__init__()
        self.ep_group = ep_group
        self.num_dispatchers_ = num_dispatchers
        self.dp_size = dp_size
        self.rank_expert_offset = rank_expert_offset
        self.num_experts = num_experts
        self.num_topk = num_topk
        self.use_fp8_dispatch = use_fp8_dispatch

        self.handles: list = [None, None]

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64

    def _do_dispatch(
        self,
        tokens: torch.Tensor,
        token_scales: torch.Tensor | None,
        rank_topk_ids: torch.Tensor,
        rank_topk_weights: torch.Tensor,
        num_experts: int,
        a1_scale: torch.Tensor | None,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> Callable:
        num_tokens = tokens.shape[0]
        hidden = tokens.shape[1]
        topk = rank_topk_ids.shape[1]
        stream = _get_cuda_stream()

        dbo_yield_and_switch_from_compute_to_comm()

        ep_handle = self.ep_group.create_handle(
            nccl_ep.Layout.FLAT,
            _to_nccl_tensor(rank_topk_ids),
            config=nccl_ep.HandleConfig(),
            stream=stream,
        )

        max_recv = num_tokens * self.num_dispatchers_
        recv_tokens = torch.empty(
            (max_recv, hidden),
            dtype=tokens.dtype,
            device=tokens.device,
        )
        recv_topk_weights = torch.empty(
            (max_recv, topk),
            dtype=torch.float32,
            device=tokens.device,
        )
        recv_topk_idx = torch.empty(
            (max_recv, topk),
            dtype=torch.int64,
            device=tokens.device,
        )

        dispatch_inputs = nccl_ep.DispatchInputs(
            tokens=_to_nccl_tensor(tokens),
            topk_weights=_to_nccl_tensor(rank_topk_weights),
        )
        dispatch_outputs = nccl_ep.DispatchOutputs(
            tokens=_to_nccl_tensor(recv_tokens),
            topk_weights=_to_nccl_tensor(recv_topk_weights),
            topk_idx=_to_nccl_tensor(recv_topk_idx),
        )

        ep_handle.dispatch(
            dispatch_inputs,
            dispatch_outputs,
            config=nccl_ep.DispatchConfig(send_only=0),
            stream=stream,
        )
        ep_handle.complete(stream=stream)

        a2a_idx = dbo_current_ubatch_id()
        self.handles[a2a_idx] = ep_handle

        dbo_switch_to_compute_sync()

        return lambda: self._receiver(
            recv_tokens,
            recv_topk_idx,
            recv_topk_weights,
            num_experts,
            a1_scale,
            quant_config,
            defer_input_quant=defer_input_quant,
        )

    def _receiver(
        self,
        recv_x: torch.Tensor,
        recv_topk_idx: torch.Tensor | None,
        recv_topk_weights: torch.Tensor | None,
        num_experts: int,
        a1_scale: torch.Tensor | None,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> mk.PrepareResultType:
        expert_x = recv_x
        expert_x_scale = None

        if recv_topk_idx is not None:
            recv_topk_idx = recv_topk_idx + self.rank_expert_offset
            invalid = recv_topk_idx < 0
            if invalid.any():
                fill_val = (
                    num_experts - 1
                    if self.rank_expert_offset == 0
                    else 0
                )
                recv_topk_idx = torch.where(invalid, fill_val, recv_topk_idx)

        # Build per-expert token counts from topk_idx
        expert_num_tokens_list = []
        if recv_topk_idx is not None:
            local_ids = recv_topk_idx[:, 0] - self.rank_expert_offset
            num_local = self.num_experts // self.num_dispatchers_
            for e in range(num_local):
                expert_num_tokens_list.append(
                    int((local_ids == e).sum().item())
                )
        expert_tokens_meta = mk.ExpertTokensMetadata.make_from_list(
            expert_num_tokens_list, device=expert_x.device
        )

        if (not quant_config.is_block_quantized
                and not defer_input_quant
                and expert_x.numel() != 0):
            expert_x, expert_x_scale = moe_kernel_quantize_input(
                expert_x,
                a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=False,
                block_shape=quant_config.block_shape,
            )

        return (
            expert_x,
            expert_x_scale,
            expert_tokens_meta,
            recv_topk_idx,
            recv_topk_weights,
        )

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
    ) -> mk.ReceiverType:
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        if quant_config.is_block_quantized and not defer_input_quant:
            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                quant_config.a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
            )
            if a1q_scale is not None and a1q_scale.numel() == 1:
                a1q_scale = a1q_scale.view(1, 1)
            a1_post_scale = None
        else:
            a1q = a1
            a1q_scale = None
            a1_post_scale = (
                quant_config.a1_gscale
                if quant_config.quant_dtype == "nvfp4"
                else quant_config.a1_scale
            )

        return self._do_dispatch(
            tokens=a1q,
            token_scales=a1q_scale,
            rank_topk_ids=topk_ids,
            rank_topk_weights=topk_weights,
            num_experts=num_experts,
            a1_scale=a1_post_scale,
            quant_config=quant_config,
            defer_input_quant=defer_input_quant,
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
        ep_handle = self.handles[a2a_idx]
        assert ep_handle is not None

        if fused_expert_output.numel() != 0:
            if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
                weight_and_reduce_impl = TopKWeightAndReduceContiguous()
            fused_expert_output = weight_and_reduce_impl.apply(
                output=None,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        stream = _get_cuda_stream()

        dbo_yield_and_switch_from_compute_to_comm()

        assert fused_expert_output.dtype == torch.bfloat16, (
            f"NCCL EP combine requires bfloat16, "
            f"got {fused_expert_output.dtype}"
        )

        combined_output = torch.empty_like(output)

        combine_inputs = nccl_ep.CombineInputs(
            tokens=_to_nccl_tensor(fused_expert_output),
        )
        combine_outputs = nccl_ep.CombineOutputs(
            tokens=_to_nccl_tensor(combined_output),
        )

        ep_handle.combine(
            combine_inputs,
            combine_outputs,
            config=nccl_ep.CombineConfig(send_only=0),
            stream=stream,
        )
        ep_handle.complete(stream=stream)

        dbo_switch_to_compute()

        if do_async:

            def _receiver():
                dbo_switch_to_comm()
                output.copy_(combined_output, non_blocking=True)
                dbo_yield_and_switch_from_comm_to_compute()

            return _receiver
        else:
            assert not dbo_enabled()
            output.copy_(combined_output, non_blocking=True)
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
        self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            False,
        )
