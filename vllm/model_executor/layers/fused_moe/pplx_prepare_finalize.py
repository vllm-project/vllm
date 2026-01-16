# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import pplx_kernels as pplx
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _validate_scale_shape,
    moe_kernel_quantize_input,
)
from vllm.utils.math_utils import cdiv, round_up

logger = init_logger(__name__)


def pplx_hidden_dim_scale_bytes(
    max_num_tokens: int,
    hidden_dim: int,
    in_dtype: torch.dtype,
    quant_dtype: torch.dtype | str | None,
    per_act_token_quant: bool,
    block_shape: list[int] | None,
):
    # All pplx byte sizes must be 16-byte aligned.
    align = 16

    # For blocked per token: set to
    #   ceil_div(hidden_dim, block_size) * sizeof(float32)
    # For per-token: set to 4 * sizeof(float32) (x4 for alignment)
    if quant_dtype is not None:
        assert isinstance(quant_dtype, torch.dtype)
        assert quant_dtype.itemsize == 1
        hidden_dim_bytes = hidden_dim * quant_dtype.itemsize
        elem_size = torch.float32.itemsize

        if per_act_token_quant:
            # per-token (M x 1)
            assert block_shape is None
            hidden_scale_bytes = elem_size
        elif block_shape is not None:
            # per-group (M x K_tiles)
            block_size = block_shape[1]
            num_blocks = cdiv(hidden_dim, block_size)
            hidden_scale_bytes = num_blocks * elem_size
        else:
            # per-tensor (1 x 1)
            hidden_scale_bytes = elem_size
    else:
        hidden_dim_bytes = hidden_dim * in_dtype.itemsize
        hidden_scale_bytes = 0

    return (
        round_up(hidden_dim_bytes, align),
        round_up(hidden_scale_bytes, align),
    )


class PplxPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    def __init__(
        self,
        a2a: pplx.AllToAll,
        max_num_tokens: int,
        num_local_experts: int,
        num_dispatchers: int,
    ):
        super().__init__()
        assert max_num_tokens > 0
        assert num_local_experts > 0
        self.a2a = a2a
        self.max_num_tokens = max_num_tokens
        self.num_local_experts = num_local_experts
        self.num_dispatchers_ = num_dispatchers

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_num_tokens

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.uint32

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

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
        num_tokens = a1.size(0)  # M
        hidden_dim = a1.size(-1)  # K

        assert topk_ids.size(0) == num_tokens
        # expert_map should be None because with expert map, -1 id is used for
        # non-local token; this causes error when casting ids to the
        # topk_indices_dtype() int32
        #
        if expert_map is not None:
            logger.warning_once(
                "The PPLX backend does not support expert mapping. "
                "The provided `expert_map` will be ignored."
            )
        expert_map = None  # noqa: F841

        # Is this always going to be a1.device?
        device = a1.device

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        repeat_cols = 4
        repeat_rows = 1 if quant_config.per_act_token_quant else a1.size(0)
        # TODO(bnell): always pass quant_config.a1_scale?
        a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            (None if quant_config.per_act_token_quant else quant_config.a1_scale),
            quant_dtype=quant_config.quant_dtype,
            per_act_token_quant=quant_config.per_act_token_quant,
            block_shape=quant_config.block_shape,
        )

        _validate_scale_shape(
            a1q, a1q_scale, quant_config.per_act_token_quant, quant_config.block_shape
        )

        orig_a_scale_block_shape: int | None = None

        if a1q_scale is not None:
            scalar_scales = a1q_scale.numel() == 1

            # pplx requires 2-d scales even for scalar scales
            if a1q_scale.dim() <= 1:
                assert scalar_scales
                a1q_scale = a1q_scale.view(1, 1)

            orig_a_scale_block_shape = a1q_scale.shape[-1]

            if not quant_config.is_block_quantized:
                # TODO (bnell): use group_broadcast instead?
                a1q_scale = a1q_scale.repeat(repeat_rows, repeat_cols)

        assert a1q_scale is None or a1q_scale.ndim == 2, (
            f"{0 if a1q_scale is None else (a1q_scale.ndim, a1q_scale.shape)}"
        )

        expert_num_tokens = torch.empty(
            self.num_local_experts,
            dtype=torch.int32,
            device=device,
        )

        expert_x = torch.empty(
            (
                self.num_local_experts,
                self.max_num_tokens * self.num_dispatchers(),
                hidden_dim,
            ),
            dtype=a1q.dtype,
            device=device,
        )

        expert_x_scale: torch.Tensor | None = None
        if a1q.dtype.itemsize == 1:
            if quant_config.is_per_act_token:
                # (M x 1) -> (E x M x K)
                final_dim = expert_x.size(2)
            elif quant_config.is_per_tensor:
                # (1 x 1) -> (E x 1 x 1)
                final_dim = 1
            else:
                # (M x K_tiles) -> (E x M x K_tiles)
                assert quant_config.block_shape is not None
                num_blocks = cdiv(expert_x.size(2), quant_config.block_shape[1])
                final_dim = num_blocks

            expert_x_scale_shape = (
                self.num_local_experts,
                expert_x.size(1),
                round_up(final_dim, 4),  # round up for alignment
            )

            expert_x_scale = torch.empty(
                expert_x_scale_shape,
                dtype=torch.float32,
                device=expert_x.device,
            )

        # This argument is optional, defaults to indices.size(0)
        # There's not much point setting this unless it is != indices.size(0)
        bound_m: torch.Tensor | None = None

        self.a2a.dispatch(
            out_expert_num_tokens=expert_num_tokens,
            out_expert_x=expert_x,
            out_expert_x_scale=expert_x_scale,
            dp_x=a1q,
            dp_x_scale=a1q_scale,
            indices=topk_ids,
            bound_m=bound_m,
            do_send=True,
            do_recv=False,
        )

        hook = lambda: self.a2a.dispatch(
            out_expert_num_tokens=expert_num_tokens,
            out_expert_x=expert_x,
            out_expert_x_scale=expert_x_scale,
            dp_x=a1q,
            dp_x_scale=a1q_scale,
            indices=topk_ids,
            bound_m=bound_m,
            do_send=False,
            do_recv=True,
        )

        return (
            hook,
            lambda: self._receiver(
                expert_num_tokens,
                expert_x,
                expert_x_scale,
                orig_a_scale_block_shape,
            ),
        )

    def _receiver(
        self,
        expert_num_tokens: torch.Tensor,
        expert_x: torch.Tensor,
        expert_x_scale: torch.Tensor | None,
        orig_a_scale_block_shape: int | None,
    ) -> mk.PrepareResultType:
        if expert_x_scale is not None:
            expert_x_scale = expert_x_scale[:, :, :orig_a_scale_block_shape]
            assert expert_x_scale.ndim == 3

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

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> Callable:
        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate), (
            "Weight application and reduction happens in the combine kernel."
        )

        # This argument is optional
        # There's not much point setting this unless it is != topk_ids.size(0)
        bound_m: torch.Tensor | None = None

        # TODO (bnell): fails in test_pplx_moe.py, figure out what's going on
        # num_tokens = output.size(0)  # M
        # assert topk_ids.size(0) == num_tokens, (
        #    f"{topk_ids.size(0)} == {num_tokens}")
        assert topk_ids.size() == topk_weights.size(), (
            f"{topk_ids.size()} == {topk_weights.size()}"
        )
        assert output.size(0) <= self.max_num_tokens, (
            f"{output.size(0)} <= {self.max_num_tokens}"
        )
        assert output.size(1) == fused_expert_output.size(-1)

        # Set weights to 1 if we did them in dispatch. This is hacky.
        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        topk_ids_u32 = topk_ids.view(dtype=torch.uint32)

        self.a2a.combine(
            out_tokens=output,
            indices=topk_ids_u32,
            weights=topk_weights,
            expert_y=fused_expert_output,
            bound_m=bound_m,
            do_send=True,
            do_recv=False,
        )

        return lambda: self.a2a.combine(
            out_tokens=output,
            indices=topk_ids_u32,
            weights=topk_weights,
            expert_y=fused_expert_output,
            bound_m=bound_m,
            do_send=False,
            do_recv=True,
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
        receiver = self.finalize_async(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
        )
        receiver()
