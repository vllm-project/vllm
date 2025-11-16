# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import deep_ep
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
    dbo_enabled,
    dbo_maybe_run_recv_hook,
)

logger = init_logger(__name__)

# DeepEP kernels quantize dispatch inputs in 128 element chunks.
DEEPEP_QUANT_BLOCK_SIZE = 128
DEEPEP_QUANT_BLOCK_SHAPE = [DEEPEP_QUANT_BLOCK_SIZE, DEEPEP_QUANT_BLOCK_SIZE]


def dequant_fp8(
    expert_x_fp8: torch.Tensor, expert_x_scales: torch.Tensor
) -> torch.Tensor:
    """
    Return dequantized tensor in fp32
    """
    # TODO (varun) : Optimize leverage num_tokens_per_expert counts
    assert expert_x_fp8.is_contiguous()
    expert_x_scales = expert_x_scales.contiguous()
    num_experts = expert_x_fp8.size(0)

    expert_x_fp32 = expert_x_fp8.to(torch.float32).view(
        num_experts, -1, DEEPEP_QUANT_BLOCK_SIZE
    )
    expert_x_scales = expert_x_scales.view(num_experts, -1, 1)
    return (expert_x_fp32 * expert_x_scales).view(expert_x_fp8.size())


class DeepEPLLPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Prepare/Finalize using DeepEP low-latency kernels.
    """

    # DeepEP low-latency kernels are compiled only for certain
    # specific hidden sizes.
    # NOTE: Keep this list sorted, maybe_roundup_layer_hidden_size depends
    # on it.
    SUPPORTED_HIDDEN_SIZES = [2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192]

    @staticmethod
    def maybe_roundup_layer_hidden_size(hidden_size: int) -> int:
        # Round up hidden size to the closest supported hidden size.
        _supported_hs = DeepEPLLPrepareAndFinalize.SUPPORTED_HIDDEN_SIZES
        # Check sorted
        num_supported_hs = len(_supported_hs)
        assert all(
            [
                _supported_hs[i] < _supported_hs[i + 1]
                for i in range(num_supported_hs - 1)
            ]
        )

        for x in _supported_hs:
            if x >= hidden_size:
                return x

        raise ValueError(
            f"Hidden Size {hidden_size} is greater than the "
            f"maximum supported hidden size {_supported_hs[-1]}"
        )

    def __init__(
        self,
        buffer: deep_ep.Buffer,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        use_fp8_dispatch: bool = False,
    ):
        super().__init__()

        self.buffer = buffer
        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch
        # The dispatch function returns a handle that the combine function
        # requires. We store the handle here so it is available to the
        # combine function.
        self.handles: list[tuple | None] = [None, None]
        self.num_dispatchers_ = num_dispatchers

        # We don't have enough information to determine if we should dispatch
        # activation scales in a packed ue8m0 format during object construction
        # time. This setting is handled by post_init_setup.
        self.use_ue8m0_dispatch = False

    def post_init_setup(self, fused_experts: mk.FusedMoEPermuteExpertsUnpermute):
        if not fused_experts.supports_packed_ue8m0_act_scales():
            # Early exit.
            return

        if self.use_fp8_dispatch:
            logger.debug_once(
                "Update DeepEPLLPrepareFinalize to do packed ue8m0 scales dispatch."
            )
            self.use_ue8m0_dispatch = True
        else:
            logger.warning_once(
                "DeepEPLLPrepareAndFinalize is setup to dispatch raw/unquantized "
                f"activations despite ({fused_experts.__class__.__name__}) being able "
                "to support quantized activations.",
                scope="local",
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

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64

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
            if block_k == DEEPEP_QUANT_BLOCK_SIZE:
                # DeepEP kernels did the quantization for us.
                x, x_scales = x
                return x, x_scales

            # Dequant to get back the tokens in the datatype we dispatched in.
            x_fp8, x_scales = x
            x = dequant_fp8(x_fp8, x_scales).to(dtype=a1_dtype)

        assert isinstance(x, torch.Tensor)

        num_experts, max_tokens, hidden_dim = x.size()

        # TODO (varun): Optimization - Use a batched version of quant
        x = x.view((-1, hidden_dim))
        x, x_scales = moe_kernel_quantize_input(
            x,
            quant_config.a1_scale,
            quant_config.quant_dtype,
            quant_config.per_act_token_quant,
            quant_config.block_shape,
        )
        x = x.view((num_experts, -1, hidden_dim))

        if quant_config.quant_dtype is not None:
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
    ) -> tuple[Callable, mk.ReceiverType]:
        hidden_size = a1.size(1)
        assert hidden_size in self.SUPPORTED_HIDDEN_SIZES, (
            f"Hidden Size {hidden_size} not in supported list of hidden sizes"
            f"{self.SUPPORTED_HIDDEN_SIZES}"
        )

        a2a_idx = dbo_current_ubatch_id()

        if self.use_fp8_dispatch:
            assert hidden_size % 128 == 0, (
                "DeepEP kernels quantize the inputs in blocks of shape 128"
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
            "low_latency kernels doesn't support dispatching per-token scales"
        )

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        # Dispatch
        expert_x, expert_num_tokens, handle, _, hook = self.buffer.low_latency_dispatch(
            a1,
            topk_ids,
            self.max_tokens_per_rank,
            num_experts,
            use_fp8=self.use_fp8_dispatch,
            # round_scale needs to be set to dispatch in ue8m0
            round_scale=self.use_ue8m0_dispatch,
            use_ue8m0=self.use_ue8m0_dispatch,
            async_finish=False,
            return_recv_hook=True,
        )
        self.handles[a2a_idx] = handle

        return (
            hook,
            lambda: self._receiver(
                expert_x,
                expert_num_tokens,
                quant_config.a1_scale,
                a1.dtype,
                quant_config,
            ),
        )

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
        do_recv_hook = dbo_enabled() or do_async
        handle = self.handles[a2a_idx]
        assert handle is not None

        combine_topk_weights = topk_weights
        if apply_router_weight_on_input:
            # weights have already been applied.
            combine_topk_weights = torch.ones_like(topk_weights)

        # TODO (varun) : Enable zero copy mode
        dbo_maybe_run_recv_hook()
        _, _, recv_hook = self.buffer.low_latency_combine(
            fused_expert_output,
            topk_ids,
            combine_topk_weights,
            handle,
            async_finish=False,
            zero_copy=False,
            return_recv_hook=do_recv_hook,
            out=output,
        )

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
