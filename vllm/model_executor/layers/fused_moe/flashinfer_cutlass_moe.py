# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
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
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_cutlass_fused_moe,
    has_flashinfer_cutlass_fused_moe,
)

logger = init_logger(__name__)


def is_valid_flashinfer_cutlass_fused_moe(
    hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> bool:
    """
    Check if the given problem size is supported by the FlashInfer CUTLASS MoE
    kernel.
    """
    if not has_flashinfer_cutlass_fused_moe():
        logger.debug_once(
            "FlashInferExperts disabled: flashinfer_cutlass_fused_moe not available."
        )
        return False
    # Data type checks
    if (
        w1.dtype != torch.uint8
        or w2.dtype != torch.uint8
        or hidden_states.dtype not in [torch.float32, torch.float16, torch.bfloat16]
    ):
        logger.debug_once(
            "FlashInferExperts disabled: w1/w2 must be torch.uint8 "
            f"(got w1={w1.dtype}, w2={w2.dtype}), hidden_states must be "
            f"float32, float16, or bfloat16 (got {hidden_states.dtype})."
        )
        return False
    return True


class FlashInferExperts(mk.FusedMoEModularExperts):
    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        assert quant_config.quant_dtype in ("nvfp4", torch.float8_e4m3fn, None), (
            "Only nvfp4, fp8, bfloat16 and"
            " float16 quantization are currently supported."
        )
        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        self.ep_size = moe_config.moe_parallel_config.ep_size
        self.tp_rank = moe_config.moe_parallel_config.tp_rank
        self.tp_size = moe_config.moe_parallel_config.tp_size
        self.out_dtype = moe_config.in_dtype
        self.use_dp = moe_config.moe_parallel_config.dp_size > 1
        # Enables DeepSeek-style FP8 block-scale path:
        # - pass per-block weight scales to the kernel
        # - skip input activation quantization (kernel applies scaling)
        self.use_deepseek_fp8_block_scale = quant_config.is_block_quantized

    @property
    def expects_unquantized_inputs(self) -> bool:
        return self.quant_config.use_fp8_w8a8 and self.quant_config.is_block_quantized

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cuda()
            and (
                current_platform.is_device_capability((9, 0))
                or current_platform.is_device_capability_family(100)
            )
            and has_flashinfer_cutlass_fused_moe()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # The following are supported by FlashInferExperts:
        #   * unquantized
        #   * fp8 static per-tensor on 9.0+
        #   * fp8 block on 9.0
        #   * nvfp4 on 10.0+

        p = current_platform
        scheme = (weight_key, activation_key)
        return (
            (
                scheme
                in [
                    (None, None),
                    (kFp8StaticTensorSym, kFp8StaticTensorSym),
                ]
            )
            or (
                (scheme == (kFp8Static128BlockSym, kFp8Dynamic128Sym))
                and (p.is_device_capability((9, 0)))
            )
            or (
                (scheme == (kNvfp4Static, kNvfp4Dynamic))
                and (p.is_device_capability_family(100))
            )
        )

    @staticmethod
    def _supports_activation(activation: str) -> bool:
        return activation in ["silu", "relu2_no_mul"]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # FLASHINFER_CUTLASS currently uses its down P/F, which does not
        # work with SP. This will be removed in follow up after we get
        # rid of the FlashInfer specific P/F function.
        # TODO: the per-tensor fp8 kernels don't work with MNNVL FI A2As.
        return not moe_parallel_config.is_sequence_parallel

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

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
        activation: str,
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
        # For NVFP4, the output is stored in a packed int8 format,
        # so the actual hidden dim is 2x the size of K here.
        output_shape = (M, K * 2 if self.quant_dtype == "nvfp4" else K)
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
        activation: str,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        from flashinfer.fused_moe.core import ActivationType

        activation_str_to_value_map = {
            "silu": ActivationType.Swiglu,  # This is the default
            "relu2_no_mul": ActivationType.Relu2,
        }
        assert activation in activation_str_to_value_map, (
            f"{activation=} missing from {activation_str_to_value_map.keys()=}"
        )

        # Select quantization metadata based on FP8 format/path
        if (
            self.quant_dtype == torch.float8_e4m3fn
            and not self.use_deepseek_fp8_block_scale
        ):
            # FP8 per-tensor path: use global alphas/scales; do not pass input_sf
            quant_scales = [
                self.g1_alphas,  # w13_weight_scale * w13_input_scale
                self.a2_gscale,  # 1.0 / w2_input_scale
                self.g2_alphas,  # w2_weight_scale * w2_input_scale
                self.a1_scale,
            ]

            a1q_scale = None  # not passing input_sf in fp8
            fc1_expert_weights = w1
            fc2_expert_weights = w2
        elif self.quant_dtype == "nvfp4":
            # Ensure w1_scale and w2_scale are not None before calling view
            assert self.w1_scale is not None and self.w2_scale is not None, (
                "w1_scale and w2_scale must not be None for FlashInferExperts"
            )
            # Flashinfer CUTLASS kernel takes scalar global scales,
            # min because inv_scale.
            quant_scales = [
                self.a1_gscale,
                self.w1_scale.view(torch.int32),
                self.g1_alphas,
                self.a2_gscale,
                self.w2_scale.view(torch.int32),
                self.g2_alphas,
            ]
            # FlashInfer API requires weight to be long for nvfp4
            fc1_expert_weights = w1.view(torch.long)
            fc2_expert_weights = w2.view(torch.long)
        elif self.use_deepseek_fp8_block_scale:
            # FP8 block-scale path: provide block-scale weights, omit a1q_scale
            quant_scales = [
                self.w1_scale,
                self.w2_scale,
            ]
            a1q_scale = None
            fc1_expert_weights = w1
            fc2_expert_weights = w2
        else:
            quant_scales = None
            a1q_scale = None
            fc1_expert_weights = w1
            fc2_expert_weights = w2

        _ = flashinfer_cutlass_fused_moe(
            input=hidden_states,
            token_selected_experts=topk_ids.to(torch.int),
            token_final_scales=topk_weights,
            fc1_expert_weights=fc1_expert_weights,
            fc2_expert_weights=fc2_expert_weights,
            output_dtype=self.out_dtype,
            quant_scales=quant_scales,
            input_sf=a1q_scale,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            output=output,
            activation_type=activation_str_to_value_map[activation],
            # Informs FlashInfer to use the block-scale decoding path when True
            use_deepseek_fp8_block_scale=self.use_deepseek_fp8_block_scale,
        )

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        # No support for LoRA in flashinfer_cutlass_fused_moe.
        # See TODOs in flashinfer functions runMoe and runMoeMinLantency.
        raise NotImplementedError("LoRA is not supported for flashinfer_cutlass_moe")
