# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional, Dict

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_prepare_finalize import (
    FlashInferCutlassMoEPrepareAndFinalize)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

from vllm.utils import round_up

logger = init_logger(__name__)

from typing import TYPE_CHECKING

try:
    from flashinfer import fp4_quantize as fp4_quantize
    from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe
except ImportError:
    if not TYPE_CHECKING:
        cutlass_fused_moe = None

has_flashinfer_cutlass_fused_moe = flashinfer_cutlass_fused_moe is not None

#TODO(shuw): use this check
def _valid_flashinfer_fused_moe(hidden_states: torch.Tensor, w1: torch.Tensor,
                                w2: torch.Tensor) -> bool:
    """
    Check if the given problem size is supported by the DeepGemm grouped
    gemm kernel.  All of M, N, K and the quantization block_shape must be
    aligned by `dg.get_m_alignment_for_contiguous_layout()`.
    """
    if not has_flashinfer_cutlass_fused_moe:
        logger.debug(
            "FlashInferExperts disabled: flashinfer_cutlass_fused_moe not available."
        )
        return False
    # Data type checks
    if (w1.dtype != torch.uint8 or w2.dtype != torch.uint8
            or hidden_states.dtype
            not in [torch.float32, torch.float16, torch.bfloat16]):
        logger.debug(
            f"FlashInferExperts disabled: w1/w2 must be torch.uint8 (got w1={w1.dtype}, w2={w2.dtype}), "
            f"hidden_states must be float32, float16, or bfloat16 (got {hidden_states.dtype})."
        )
        return False
    return True

class FlashInferExperts(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(self,
        use_nvfp4_w4a4: bool = False,
        use_fp8_w8a8: bool = False,
        use_dp: bool=False,
        ep_rank: int=0,
        ep_size: int=1,
        tp_rank: int=0,
        tp_size: int=1,
    ):
        super().__init__(
            FusedMoEQuantConfig(
                quant_dtype=torch.uint8,
                per_act_token_quant=False,
                block_shape=None,
            ))
        self.use_nvfp4_w4a4 = use_nvfp4_w4a4
        self.use_fp8_w8a8 = use_fp8_w8a8
        self.ep_rank=ep_rank
        self.ep_size=ep_size
        self.tp_rank=tp_rank
        self.tp_size=tp_size
        self.use_dp=use_dp

    @property
    def activation_formats(
        self
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (mk.FusedMoEActivationFormat.Standard,
                mk.FusedMoEActivationFormat.Standard)

    def supports_expert_map(self) -> bool:
        return False
        
    def supports_chunking(self) -> bool:
        #TODO(shuw): support chunking later
        return False

    def workspace_shapes(
        self, a: torch.Tensor, aq: torch.Tensor, M: int, N: int, K: int,
        topk: int, global_num_experts: int, local_num_experts: int
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
        # num_experts = global_num_experts
        # block_m = self.block_shape[0]
        # M_sum = (M * topk) + num_experts * (block_m - 1)
        # M_sum = round_up(M_sum, block_m)
        # workspace1 = ()
        # TODO(shuw): This is nvfp4 specialized, add branch for other quant type.
        aq_m, aq_n = aq.shape
        workspace2 = ()
        output_shape = (aq_m, aq_n * 2)
        workspace_dtype = a.dtype
        workspace1 = output_shape
        # print(f"inside workspace_shape: workspace1:{workspace1} and output_shape:{output_shape} with type:{workspace_dtype}")
        # determined by aq, since aq is the one after possible communication op and participate in experts computation.
        return (workspace1, workspace2, output_shape, workspace_dtype)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13:Optional[torch.Tensor],
        workspace2:Optional[torch.Tensor],
        expert_num_tokens: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        g1_alphas: torch.Tensor,
        g2_alphas: torch.Tensor,
        a1_scale: torch.Tensor,
        out_dtype: torch.dtype,
    ):
        # Flashinfer CUTLASS kernel takes scalar global scales,
        # min because inv_scale.
        if self.use_nvfp4_w4a4:
            quant_scales = [
                a1_scale,
                w1_scale.view(torch.int32),
                g1_alphas,
                a2_scale,
                w2_scale.view(torch.int32),
                g2_alphas,
            ]
            # print(self.ep_size, self.ep_rank, self.tp_rank, self.tp_size)
            out = flashinfer_cutlass_fused_moe(
                hidden_states,
                topk_ids.to(torch.int),
                topk_weights,
                # FlashInfer API requires weight to be long for nvfp4
                w1.view(torch.long),
                w2.view(torch.long),
                output_dtype=out_dtype,
                quant_scales=quant_scales,
                input_sf=a1q_scale,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
            )[0]
            # print(f"callsite hidden_states:{hidden_states.shape}")
            # print(f"tmp:{out.shape}")
            # print(f"output:{output.shape}")
            output.copy_(out)
        else:
            raise ValueError("Only nvfp4 quantization is currently supported.")
