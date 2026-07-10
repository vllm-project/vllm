# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.hpc import has_hpc, hpc_fuse_moe, hpc_fuse_moe_blockwise

logger = init_logger(__name__)


class HPCExperts(mk.FusedMoEExpertsModular):
    """MoE implementation powered by [HPC](https://github.com/Tencent/hpc-ops).

    Only supported on NVIDIA Hopper GPUs (e.g. H20, H200), and currently limited to
    FP8 models such as Hy3-FP8, Qwen3-235B-A22B-FP8, etc.
    """

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)

        assert quant_config.weight_quant_dtype in (torch.float8_e4m3fn,), (
            "Only fp8 quantization is currently supported."
        )

        self.device = moe_config.device
        self.num_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        self.ep_size = moe_config.moe_parallel_config.ep_size
        self.tp_rank = moe_config.moe_parallel_config.tp_rank
        self.tp_size = moe_config.moe_parallel_config.tp_size
        self.out_dtype = moe_config.in_dtype

    @property
    def expects_unquantized_inputs(self) -> bool:
        return False

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and (p.is_device_capability(90) or p.is_device_capability_family(100))
            and has_hpc()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        scheme = (weight_key, activation_key)
        # The following are supported by HPCExperts:
        return scheme in [
            # fp8 static per-tensor on 9.0+
            (kFp8StaticTensorSym, kFp8StaticTensorSym),
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
        ]

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
        ]

    @staticmethod
    def _supports_shape(hidden_dim: int) -> bool:
        # HPC fused MoE kernels process hidden_size in blocks of 128:
        # block-wise fp8 requires hidden_size % 128 == 0 (per-128 quant), and
        # the group GEMM tiles N by 128. Require 128-alignment to cover all
        # code paths.
        return hidden_dim % 128 == 0

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def supports_chunking(self) -> bool:
        # This refers to TP chunking; DP chunking is handled separately.
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
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
        workspace1 = (M, K)
        workspace2 = (0,)
        output_shape = (M, K)
        # The workspace is determined by `aq`, since it comes after any
        # potential communication op and is involved in the expert computation.
        return (workspace1, workspace2, output_shape)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        assert self._supports_activation(activation), f"{activation=} not supported"
        assert self.quant_config.w1_scale is not None, (
            "w13_weight_scale must be provided"
        )
        assert self.quant_config.w2_scale is not None, (
            "w2_weight_scale must be provided"
        )

        if self.quant_config.is_block_quantized:
            hpc_fuse_moe_blockwise(
                x=hidden_states,
                x_scale=a1q_scale,
                gate_up_weight=w1,
                gate_up_weight_scale=self.quant_config.w1_scale,
                down_weight=w2,
                down_weight_scale=self.quant_config.w2_scale,
                topk_ids=topk_ids,
                topk_scale=topk_weights,
                rank_ep=self.ep_rank,
                num_expert_total=global_num_experts,
                output=output,
            )
        else:
            assert self.quant_config.a1_scale is not None, (
                "w13_input_scale must be provided"
            )
            assert self.quant_config.a2_scale is not None, (
                "w2_input_scale must be provided"
            )
            hpc_fuse_moe(
                x=hidden_states,
                gate_up_weight=w1,
                down_weight=w2,
                gate_up_scale=self.quant_config.g1_alphas,
                down_scale=self.quant_config.g2_alphas,
                act_and_mul_scale=self.quant_config.a2_gscale,
                topk_ids=topk_ids,
                topk_scale=topk_weights,
                rank_ep=self.ep_rank,
                num_expert_total=global_num_experts,
                output=output,
            )
