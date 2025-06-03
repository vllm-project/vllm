# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import deep_ep
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)

# DeepEP kernels quantize dispatch inputs in 128 element chunks.
DEEPEP_QUANT_BLOCK_SIZE = 128


def dequant_fp8(expert_x_fp8: torch.Tensor,
                expert_x_scales: torch.Tensor) -> torch.Tensor:
    """
    Return dequantized tensor in fp32
    """
    # TODO (varun) : Optimize leverage num_tokens_per_expert counts
    assert expert_x_fp8.is_contiguous()
    expert_x_scales = expert_x_scales.contiguous()
    num_experts = expert_x_fp8.size(0)

    expert_x_fp32 = expert_x_fp8.to(torch.float32).view(
        num_experts, -1, DEEPEP_QUANT_BLOCK_SIZE)
    expert_x_scales = expert_x_scales.view(num_experts, -1, 1)
    return (expert_x_fp32 * expert_x_scales).view(expert_x_fp8.shape)


class DeepEPLLPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Prepare/Finalize using DeepEP low-latency kernels.
    """

    # DeepEP low-latency kernels are compiled only for certain
    # specific hidden sizes.
    SUPPORTED_HIDDEN_SIZES = [2560, 4096, 5120, 7168]

    def __init__(self,
                 buffer: deep_ep.Buffer,
                 world_size: int,
                 dp_size: int,
                 max_tokens_per_rank: int,
                 quant_dtype: Optional[torch.dtype] = None,
                 block_shape: Optional[list[int]] = None,
                 use_fp8_dispatch: bool = False):
        super().__init__()

        self.buffer = buffer
        self.world_size = world_size
        self.dp_size = dp_size
        self.quant_dtype = quant_dtype
        self.block_shape = block_shape
        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch
        # The dispatch function returns a handle that the combine function
        # requires. We store the handle here so it is available to the
        # combine function.
        self.handle = None

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        return torch.int64

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        rank_topk_weights: torch.Tensor,
        rank_topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor]]:

        hidden_size = a1.size(1)
        assert hidden_size in self.SUPPORTED_HIDDEN_SIZES, \
            (f"Hidden Size {hidden_size} not in supported list of hidden sizes"
            f"{self.SUPPORTED_HIDDEN_SIZES}")

        if self.use_fp8_dispatch:
            assert hidden_size % 128 == 0, \
            "DeepEP kernels quantize the inputs in blocks of shape 128"

        # Quantize
        per_act_token = a1_scale.numel() != 1 if a1_scale is not None else (
            a2_scale.numel() != 1 if a2_scale is not None else False)
        assert not per_act_token, (
            "low_latency kernels don't support per-act-token quant")

        if apply_router_weight_on_input:
            topk = rank_topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1")
            a1 = a1 * rank_topk_weights.to(a1.dtype)

        # Dispatch
        expert_x, expert_num_tokens, self.handle, event, hook = \
                self.buffer.low_latency_dispatch(a1,
                                                rank_topk_ids,
                                                self.max_tokens_per_rank,
                                                num_experts,
                                                use_fp8=self.use_fp8_dispatch,
                                                async_finish=False,
                                                return_recv_hook=False)

        if self.use_fp8_dispatch:
            # TODO (varun) : In the case of dynamic quantization, we could
            # probably skip the quant below and use the results directly.
            # Although note that the deepep quant is per token 128 elements.
            expert_x_fp8, expert_x_scales = expert_x
            expert_x = dequant_fp8(expert_x_fp8,
                                   expert_x_scales).to(dtype=a1.dtype)

        num_experts = expert_x.size(0)
        hidden_dim = expert_x.size(-1)

        expert_x = expert_x.view((-1, expert_x.size(-1)))
        expert_x, expert_x_scale = moe_kernel_quantize_input(
            expert_x, a1_scale, self.quant_dtype, per_act_token,
            self.block_shape)
        expert_x = expert_x.view((num_experts, -1, hidden_dim))

        return (expert_x, expert_x_scale, expert_num_tokens, None, None)

    def finalize(self, output: torch.Tensor, fused_expert_output: torch.Tensor,
                 topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                 apply_router_weight_on_input: bool) -> None:

        assert self.handle is not None

        combine_topk_weights = topk_weights
        if apply_router_weight_on_input:
            # weights have already been applied.
            combine_topk_weights = torch.ones_like(topk_weights)

        # TODO (varun) : Enable zero copy mode
        _, event, hook = self.buffer.low_latency_combine(
            fused_expert_output,
            topk_ids,
            combine_topk_weights,
            self.handle,
            async_finish=False,
            zero_copy=False,
            return_recv_hook=False,
            out=output)
