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
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input,
    normalize_batched_scales_shape,
)
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_maybe_run_recv_hook,
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


class NcclEPBatchedPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Prepare/Finalize using NCCL EP with EXPERT_MAJOR layout
    (BatchedExperts activation format).
    """

    def __init__(
        self,
        ep_group: nccl_ep.Group,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        num_experts: int,
        num_local_experts: int,
        use_fp8_dispatch: bool = False,
        global_to_physical: torch.Tensor | None = None,
        physical_to_global: torch.Tensor | None = None,
        local_expert_global_ids: torch.Tensor | None = None,
    ):
        super().__init__()

        self.ep_group = ep_group
        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.num_dispatchers_ = num_dispatchers

        self.handles: list = [None, None]

        topk_indices_dtype = self.topk_indices_dtype()

        def _maybe_cast(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None or topk_indices_dtype is None:
                return tensor
            return tensor.to(dtype=topk_indices_dtype)

        self.global_to_physical = _maybe_cast(global_to_physical)
        self.physical_to_global = _maybe_cast(physical_to_global)
        self.local_expert_global_ids = _maybe_cast(local_expert_global_ids)

        self._recv_count_buf: torch.Tensor | None = None

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64

    def _map_global_to_physical_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        if self.global_to_physical is None:
            return topk_ids
        return self.global_to_physical[topk_ids]

    def _do_quant(
        self,
        x: torch.Tensor,
        a1_dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_experts, max_tokens, hidden_dim = x.size()
        q_dtype = quant_config.quant_dtype

        if q_dtype == "nvfp4":
            q_dtype = None
            logger.debug_once(
                "Using NCCL EP bfloat16 dispatch for NVFP4 MoE."
            )

        x = x.view((-1, hidden_dim))
        x, x_scales = moe_kernel_quantize_input(
            x,
            quant_config.a1_scale,
            q_dtype,
            quant_config.per_act_token_quant,
            quant_config.block_shape,
        )
        x = x.view((num_experts, -1, hidden_dim))

        if q_dtype is not None:
            assert x_scales is not None
            x_scales = normalize_batched_scales_shape(x_scales, num_experts)

        return x, x_scales

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
                f"{self.__class__.__name__} does not support "
                "defer_input_quant=True."
            )

        a2a_idx = dbo_current_ubatch_id()

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        hidden = a1.shape[1]
        stream = _get_cuda_stream()

        dispatch_topk_ids = self._map_global_to_physical_ids(topk_ids)

        ep_handle = self.ep_group.create_handle(
            nccl_ep.Layout.EXPERT_MAJOR,
            _to_nccl_tensor(dispatch_topk_ids),
            config=nccl_ep.HandleConfig(),
            stream=stream,
        )

        max_recv = self.max_tokens_per_rank * self.num_dispatchers_
        recv_tokens = torch.empty(
            (self.num_local_experts, max_recv, hidden),
            dtype=a1.dtype,
            device=a1.device,
        )
        recv_count = torch.empty(
            self.num_local_experts,
            dtype=torch.int32,
            device=a1.device,
        )

        dispatch_inputs = nccl_ep.DispatchInputs(
            tokens=_to_nccl_tensor(a1),
        )
        dispatch_outputs = nccl_ep.DispatchOutputs(
            tokens=_to_nccl_tensor(recv_tokens),
        )
        dispatch_layout = nccl_ep.LayoutInfo(
            expert_counters=_to_nccl_tensor(recv_count),
        )

        ep_handle.dispatch(
            dispatch_inputs,
            dispatch_outputs,
            layout_info=dispatch_layout,
            config=nccl_ep.DispatchConfig(send_only=1),
            stream=stream,
        )

        self.handles[a2a_idx] = ep_handle
        self._recv_count_buf = recv_count

        def hook():
            ep_handle.complete(stream=_get_cuda_stream())

        return (
            hook,
            lambda: self._receiver(
                recv_tokens,
                recv_count,
                a1.dtype,
                quant_config,
            ),
        )

    def _receiver(
        self,
        expert_x: torch.Tensor,
        expert_num_tokens: torch.Tensor,
        a1_dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        expert_x, expert_x_scale = self._do_quant(
            expert_x, a1_dtype, quant_config
        )

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=None,
        )

        return expert_x, expert_x_scale, expert_tokens_meta, None, None

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
        do_async: bool,
    ) -> tuple[Callable, Callable]:
        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate), (
            "Weight application and reduction happens in the combine kernel."
        )

        a2a_idx = dbo_current_ubatch_id()
        ep_handle = self.handles[a2a_idx]
        assert ep_handle is not None

        combine_topk_weights = topk_weights
        if apply_router_weight_on_input:
            combine_topk_weights = torch.ones_like(topk_weights)

        stream = _get_cuda_stream()

        dbo_maybe_run_recv_hook()

        combine_inputs = nccl_ep.CombineInputs(
            tokens=_to_nccl_tensor(fused_expert_output),
        )
        combine_outputs = nccl_ep.CombineOutputs(
            tokens=_to_nccl_tensor(output),
            topk_weights=_to_nccl_tensor(combine_topk_weights),
        )

        ep_handle.combine(
            combine_inputs,
            combine_outputs,
            config=nccl_ep.CombineConfig(send_only=1),
            stream=stream,
        )

        def recv_hook():
            ep_handle.complete(stream=_get_cuda_stream())

        return recv_hook, lambda: None

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
            do_async=True,
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
            do_async=False,
        )
