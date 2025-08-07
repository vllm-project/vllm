# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate)
from vllm.model_executor.layers.fused_moe.utils import extract_required_args
from vllm.utils.flashinfer import (flashinfer_cutlass_fused_moe,
                                   has_flashinfer_cutlass_fused_moe)

logger = init_logger(__name__)


def is_valid_flashinfer_cutlass_fused_moe(hidden_states: torch.Tensor,
                                          w1: torch.Tensor,
                                          w2: torch.Tensor) -> bool:
    """
    Check if the given problem size is supported by the FlashInfer CUTLASS MoE 
    kernel.
    """
    if not has_flashinfer_cutlass_fused_moe():
        logger.debug_once("FlashInferExperts disabled: "
                          "flashinfer_cutlass_fused_moe not available.")
        return False
    # Data type checks
    if (w1.dtype != torch.uint8 or w2.dtype != torch.uint8
            or hidden_states.dtype
            not in [torch.float32, torch.float16, torch.bfloat16]):
        logger.debug_once(
            "FlashInferExperts disabled: w1/w2 must be torch.uint8 "
            f"(got w1={w1.dtype}, w2={w2.dtype}), hidden_states must be "
            f"float32, float16, or bfloat16 (got {hidden_states.dtype}).")
        return False
    return True


class FlashInferExperts(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(
        self,
        use_nvfp4_w4a4: bool = False,
        use_fp8_w8a8: bool = False,
        use_dp: bool = False,
        ep_rank: int = 0,
        ep_size: int = 1,
        tp_rank: int = 0,
        tp_size: int = 1,
        num_dispatchers: Optional[int] = None,
        use_batched_format: bool = False,
    ):
        super().__init__(
            FusedMoEQuantConfig(
                quant_dtype=torch.uint8
                if not use_fp8_w8a8 else torch.float8_e4m3fn,
                per_act_token_quant=False,
                block_shape=None,
            ))
        self.use_nvfp4_w4a4 = use_nvfp4_w4a4
        self.use_fp8_w8a8 = use_fp8_w8a8
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.use_dp = use_dp
        assert not use_batched_format or num_dispatchers is not None
        self.num_dispatchers = num_dispatchers

    @property
    def activation_formats(
        self
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (mk.FusedMoEActivationFormat.Standard,
                mk.FusedMoEActivationFormat.Standard)

    def supports_expert_map(self) -> bool:
        return False

    def supports_chunking(self) -> bool:
        # This refers to TP chunking; DP chunking is handled separately.
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the impl.
        return TopKWeightAndReduceDelegate()

    def workspace_shapes(
        self,
        a: torch.Tensor,
        aq: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype]:
        # We use global_num_experts due to how moe_align_block_size handles
        # expert_maps.
        """
        Compute the shapes for the temporary and final outputs of the two gemms
        and activation in the fused expert function.  Since the gemms are
        independent, the workspace for the first gemm can be shared with the
        workspace for the last gemm.

        Returns a tuple of:
        - workspace13 shape tuple: must be large enough to hold the
          result of either expert gemm.
        - workspace2 shape tuple: must be large enough to hold the
          result of the activation function.
        - output shape tuple: must be exact size of the final gemm output.
        - Workspace type: The dtype to use for the workspace tensors.
        - Note: in order for activation chunking to work, the first dimension
          of each tuple must be the number of tokens.
        """
        aq_m, aq_n = aq.shape
        workspace2 = ()
        output_shape = (aq_m, aq_n * 2) if not self.use_fp8_w8a8 else (aq_m,
                                                                       aq_n)
        workspace_dtype = a.dtype
        workspace1 = output_shape
        # The workspace is determined by `aq`, since it comes after any
        # potential communication op and is involved in the expert computation.
        return (workspace1, workspace2, output_shape, workspace_dtype)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],  # Not used
        workspace13: Optional[torch.Tensor],
        workspace2: Optional[torch.Tensor],
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
        apply_router_weight_on_input: Optional[bool],
        extra_expert_args: Optional[dict[str, Any]],
    ):
        assert extra_expert_args is not None, \
            "extra_expert_args must be provided"

        assert w1_scale is not None
        assert w2_scale is not None

        if hidden_states.dtype == torch.float8_e4m3fn:
            required_keys = ['out_dtype']

            out_dtype = extract_required_args(extra_expert_args,
                                              required_keys)[0]
            assert a1q_scale is not None
            assert a2_scale is not None

            quant_scales = [
                w1_scale * a1q_scale, 1.0 / a2_scale, w2_scale * a2_scale,
                a1q_scale
            ]
            a1q_scale = None  # not passing input_sf in fp8
            fc1_expert_weights = w1
            fc2_expert_weights = w2
        else:
            required_keys = [
                'g1_alphas', 'g2_alphas', 'a1_gscale', 'a2_gscale', 'out_dtype'
            ]

            g1_alphas, g2_alphas, a1_gscale, a2_gscale, out_dtype = (
                extract_required_args(extra_expert_args, required_keys))

            quant_scales = [
                a1_gscale,
                w1_scale.view(torch.int32),
                g1_alphas,
                a2_gscale,
                w2_scale.view(torch.int32),
                g2_alphas,
            ]
            # FlashInfer API requires weight to be long for nvfp4
            fc1_expert_weights = w1.view(torch.long)
            fc2_expert_weights = w2.view(torch.long)

        # Flashinfer CUTLASS kernel takes scalar global scales,
        # min because inv_scale.
        # Ensure w1_scale and w2_scale are not None before calling view
        assert w1_scale is not None and w2_scale is not None, (
            "w1_scale and w2_scale must not "
            "be None for FlashInferExperts")

        _ = flashinfer_cutlass_fused_moe(
            input=hidden_states,
            token_selected_experts=topk_ids.to(torch.int),
            token_final_scales=topk_weights,
            fc1_expert_weights=fc1_expert_weights,
            fc2_expert_weights=fc2_expert_weights,
            output_dtype=out_dtype,
            quant_scales=quant_scales,
            input_sf=a1q_scale,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            output=output,
        )
