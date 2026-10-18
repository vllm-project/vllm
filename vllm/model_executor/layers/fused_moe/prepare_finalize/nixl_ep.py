# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from math import gcd

import nixl_ep
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.forward_context import get_forward_context, is_forward_context_available
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
    dbo_enabled,
    dbo_maybe_run_recv_hook,
    dbo_num_ubatches,
    dbo_switch_to_comm,
    dbo_switch_to_compute,
    dbo_yield_and_switch_from_comm_to_compute,
    dbo_yield_and_switch_from_compute_to_comm,
)

logger = init_logger(__name__)

# NIXL EP kernels quantize dispatch inputs in 128 element chunks.
NIXL_EP_QUANT_BLOCK_SIZE = 128
NIXL_EP_QUANT_BLOCK_SHAPE = [NIXL_EP_QUANT_BLOCK_SIZE, NIXL_EP_QUANT_BLOCK_SIZE]
NIXL_EP_TOPK_INDICES_DTYPE = getattr(nixl_ep, "topk_idx_t", torch.int64)
assert isinstance(NIXL_EP_TOPK_INDICES_DTYPE, torch.dtype)


def dequant_fp8(
    expert_x_fp8: torch.Tensor, expert_x_scales: torch.Tensor
) -> torch.Tensor:
    """Return dequantized tensor in fp32."""
    assert expert_x_fp8.is_contiguous()
    expert_x_scales = expert_x_scales.contiguous()
    num_experts = expert_x_fp8.size(0)

    expert_x_fp32 = expert_x_fp8.to(torch.float32).view(
        num_experts, -1, NIXL_EP_QUANT_BLOCK_SIZE
    )
    expert_x_scales = expert_x_scales.view(num_experts, -1, 1)
    return (expert_x_fp32 * expert_x_scales).view(expert_x_fp8.size())


class NixlEPPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Prepare/Finalize using NIXL EP kernels."""

    # NIXL EP kernels are compiled only for certain specific hidden sizes.
    # NOTE: Keep this list sorted, maybe_roundup_layer_hidden_size depends
    # on it.
    SUPPORTED_HIDDEN_SIZES = [2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192]
    assert sorted(set(SUPPORTED_HIDDEN_SIZES)) == SUPPORTED_HIDDEN_SIZES

    @staticmethod
    def maybe_roundup_layer_hidden_size(hidden_size: int) -> int:
        # Round up hidden size to the closest supported hidden size.
        _supported_hs = NixlEPPrepareAndFinalize.SUPPORTED_HIDDEN_SIZES

        for x in _supported_hs:
            if x >= hidden_size:
                return x

        raise ValueError(
            f"Hidden Size {hidden_size} is greater than the "
            f"maximum supported hidden size {_supported_hs[-1]}"
        )

    def __init__(
        self,
        buffer: nixl_ep.Buffer,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        expert_capacity: int,
        use_fp8_dispatch: bool = False,
        global_to_physical: torch.Tensor | None = None,
        physical_to_global: torch.Tensor | None = None,
        local_expert_global_ids: torch.Tensor | None = None,
        num_ubatches: int = 1,
    ):
        super().__init__()

        self.buffer = buffer
        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch
        # The dispatch function returns a handle that the combine function
        # requires. We store the handle here so it is available to the
        # combine function.
        self.handles: list[tuple | None] = [None] * num_ubatches
        self.num_dispatchers_ = num_dispatchers
        self.expert_capacity = expert_capacity

        topk_indices_dtype = self.topk_indices_dtype()

        def _maybe_cast(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None or topk_indices_dtype is None:
                return tensor
            return tensor.to(dtype=topk_indices_dtype)

        self.global_to_physical = _maybe_cast(global_to_physical)
        self.physical_to_global = _maybe_cast(physical_to_global)
        self.local_expert_global_ids = _maybe_cast(local_expert_global_ids)

        # We don't have enough information to determine if we should dispatch
        # activation scales in a packed ue8m0 format during object construction
        # time. This setting is handled by post_init_setup.
        self.use_ue8m0_dispatch = False

    def post_init_setup(self, fused_experts: mk.FusedMoEExperts):
        if not fused_experts.supports_packed_ue8m0_act_scales():
            # Early exit.
            return

        if self.use_fp8_dispatch:
            logger.debug_once(
                "Update NixlEPPrepareAndFinalize to do packed ue8m0 scales dispatch."
            )
            self.use_ue8m0_dispatch = True
        else:
            logger.warning_once(
                "NixlEPPrepareAndFinalize is setup to dispatch raw/unquantized "
                f"activations despite ({fused_experts.__class__.__name__}) being able "
                "to support quantized activations.",
            )

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def _get_runtime_dispatch_capacity(self) -> int:
        """Use the synchronized execution slice, retaining the reservation limit."""
        if not is_forward_context_available():
            return self.max_tokens_per_rank
        dp_metadata = get_forward_context().dp_metadata
        if dp_metadata is None:
            return self.max_tokens_per_rank
        # The MoE runner supplies EP-local sizes when sequence parallelism is
        # active. Both sources are CPU metadata, including during graph capture.
        if dp_metadata.local_sizes is not None:
            capacity = max(dp_metadata.local_sizes)
        else:
            capacity = int(dp_metadata.num_tokens_across_dp_cpu.max())
        capacity = max(1, capacity)
        # The NIXL TMA receive extent is capacity * num_dispatchers and must
        # be divisible by four. This only changes undersized warmup shapes;
        # normal aligned execution buckets (for example 64/128) are unchanged.
        alignment = 4 // gcd(self.num_dispatchers_, 4)
        capacity = (capacity + alignment - 1) // alignment * alignment
        assert capacity <= self.max_tokens_per_rank
        return capacity

    def topk_indices_dtype(self) -> torch.dtype | None:
        return NIXL_EP_TOPK_INDICES_DTYPE

    def _map_global_to_physical_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        if self.global_to_physical is None:
            return topk_ids
        return self.global_to_physical[topk_ids]

    def _do_quant(
        self,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        a1_dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_fp8_dispatch:
            block_k = (
                quant_config.block_shape[1]
                if quant_config.block_shape is not None
                else None
            )
            if block_k == NIXL_EP_QUANT_BLOCK_SIZE:
                # NIXL EP kernels did the quantization for us.
                x, x_scales = x
                return x, x_scales

            # Dequant to get back the tokens in the datatype we dispatched in.
            x_fp8, x_scales = x
            x = dequant_fp8(x_fp8, x_scales).to(dtype=a1_dtype)

        assert isinstance(x, torch.Tensor)

        num_experts, max_tokens, hidden_dim = x.size()

        x = x.view((-1, hidden_dim))
        q_dtype = quant_config.quant_dtype

        if q_dtype == "nvfp4":
            q_dtype = None
            logger.debug_once("Using NIXL EP bfloat16 dispatch for NVFP4 MoE.")

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
    ) -> tuple[Callable, mk.ReceiverType] | mk.ReceiverType:
        if defer_input_quant:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support defer_input_quant=True. "
                "Please select an MoE kernel that accepts quantized inputs."
            )

        hidden_size = a1.size(1)
        assert hidden_size in self.SUPPORTED_HIDDEN_SIZES, (
            f"Hidden Size {hidden_size} not in supported list of hidden sizes"
            f"{self.SUPPORTED_HIDDEN_SIZES}"
        )

        a2a_idx = dbo_current_ubatch_id()

        if self.use_fp8_dispatch:
            assert hidden_size % 128 == 0, (
                "NIXL EP kernels quantize the inputs in blocks of shape 128"
            )

        has_per_token_scales = (
            quant_config.a1_scale.numel() != 1
            if quant_config.a1_scale is not None
            else (
                quant_config.a2_scale.numel() != 1
                if quant_config.a2_scale is not None
                else False
            )
        )
        assert not has_per_token_scales, (
            "NIXL EP kernels don't support dispatching per-token scales"
        )

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        # Dispatch
        runtime_dispatch_capacity = self._get_runtime_dispatch_capacity()
        assert a1.size(0) <= runtime_dispatch_capacity
        dispatch_topk_ids = self._map_global_to_physical_ids(topk_ids)
        dbo_yield_and_switch_from_compute_to_comm()
        expert_x, expert_num_tokens, handle, _, hook = self.buffer.dispatch(
            a1,
            dispatch_topk_ids,
            runtime_dispatch_capacity,
            self.expert_capacity,
            use_fp8=self.use_fp8_dispatch,
            # round_scale needs to be set to dispatch in ue8m0
            round_scale=self.use_ue8m0_dispatch,
            use_ue8m0=self.use_ue8m0_dispatch,
            async_finish=False,
            # DBO already provides a separate stream for the full send/recv.
            return_recv_hook=not dbo_enabled(),
        )
        if dbo_num_ubatches() > 2:
            # NIXL reuses two receive buffers. Save their contents in the recv
            # hook, before the next microbatch dispatch can overwrite them.
            copies: list[tuple[torch.Tensor, torch.Tensor]] = []

            def snapshot(src: torch.Tensor) -> torch.Tensor:
                dst = torch.empty_like(src)
                copies.append((dst, src))
                return dst

            expert_x = (
                tuple(snapshot(x) for x in expert_x)
                if isinstance(expert_x, tuple)
                else snapshot(expert_x)
            )
            expert_num_tokens = snapshot(expert_num_tokens)
            handle = tuple(
                snapshot(value) if isinstance(value, torch.Tensor) else value
                for value in handle
            )
            recv_hook = hook

            def snapshot_recv_hook():
                if recv_hook is not None:
                    recv_hook()
                for dst, src in copies:
                    dst.copy_(src)

            hook = snapshot_recv_hook
        self.handles[a2a_idx] = handle

        def receiver():
            return self._receiver(
                expert_x,
                expert_num_tokens,
                quant_config.a1_scale,
                a1.dtype,
                quant_config,
            )

        if dbo_enabled():
            dbo_switch_to_compute()

            def receive_on_comm():
                dbo_switch_to_comm()
                if hook is not None:
                    hook()
                dbo_yield_and_switch_from_comm_to_compute()
                return receiver()

            # Snapshot copies precede the comm-done event. Keep the generic
            # MoE handoff from adding a second receive on the compute stream.
            return receive_on_comm
        return hook, receiver

    def _receiver(
        self,
        expert_x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        expert_num_tokens: torch.Tensor,
        a1_scale: torch.Tensor | None,
        a1_dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        expert_x, expert_x_scale = self._do_quant(expert_x, a1_dtype, quant_config)

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens, expert_num_tokens_cpu=None
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
        if defer_input_quant:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support defer_input_quant=True. "
                "Please select an MoE kernel that accepts quantized inputs."
            )
        result = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
        )
        if isinstance(result, tuple):
            hook, receiver = result
            hook()
            return receiver()
        return result()

    def _finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
        do_async: bool,
    ) -> tuple[Callable, Callable] | Callable:
        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate), (
            "Weight application and reduction happens in the combine kernel."
        )

        a2a_idx = dbo_current_ubatch_id()
        do_recv_hook = do_async and not dbo_enabled()
        handle = self.handles[a2a_idx]
        assert handle is not None

        combine_topk_weights = topk_weights
        if apply_router_weight_on_input:
            # weights have already been applied.
            combine_topk_weights = torch.ones_like(topk_weights)

        combine_topk_ids = self._map_global_to_physical_ids(topk_ids)
        # TODO (varun) : Enable zero copy mode
        dbo_maybe_run_recv_hook()
        dbo_yield_and_switch_from_compute_to_comm()
        _, _, recv_hook = self.buffer.combine(
            fused_expert_output,
            combine_topk_ids,
            combine_topk_weights,
            handle,
            async_finish=False,
            zero_copy=False,
            return_recv_hook=do_recv_hook,
            out=output,
        )

        if dbo_enabled():
            dbo_switch_to_compute()

            def receive_on_comm():
                dbo_switch_to_comm()
                dbo_yield_and_switch_from_comm_to_compute()

            return receive_on_comm
        return recv_hook, lambda: None

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> tuple[Callable, Callable] | Callable:
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
