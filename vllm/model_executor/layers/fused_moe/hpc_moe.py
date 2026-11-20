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
    kMxfp4Static,
    kMxfp8Dynamic,
)
from vllm.platforms import current_platform
from vllm.utils.hpc import (
    has_hpc,
    has_hpc_sm120_moe,
    hpc_fuse_moe,
    hpc_fuse_moe_blockwise,
    hpc_fuse_moe_mxfp8_sm120,
)

logger = init_logger(__name__)


def _unsupported_expert_placement(moe_config: mk.FusedMoEConfig) -> str | None:
    """The HPC kernels take the local experts to be the contiguous global range
    starting at `ep_rank * num_local_experts` and never read `expert_map`.

    EPLB keeps that layout: the router rewrites logical ids to global physical
    ids, and those physical slots stay a linear slice of the rank. Round-robin
    placement is not a contiguous range, so it stays unsupported.
    """
    from vllm.config import get_current_vllm_config

    parallel_config = moe_config.moe_parallel_config
    if not parallel_config.use_ep:
        return None
    placement = get_current_vllm_config().parallel_config.expert_placement_strategy
    if placement != "linear":
        return f"kernel does not support {placement} expert placement"
    return None


def _unsupported_sm120_config(moe_config: mk.FusedMoEConfig) -> str | None:
    """Limits of the SM120 fused MoE kernel, shared by both weight formats."""
    if moe_config.in_dtype != torch.bfloat16:
        return f"kernel does not support {moe_config.in_dtype} input/output dtype"
    if moe_config.num_local_experts > 512:
        return "kernel does not support more than 512 local experts"
    if moe_config.experts_per_token > 128:
        return "kernel does not support topk > 128"
    if moe_config.has_bias:
        return "kernel does not support bias"
    if moe_config.swiglu_limit is not None and moe_config.swiglu_limit < 0:
        return "kernel does not support negative swiglu_limit"
    if moe_config.swiglu_alpha is not None:
        return "kernel does not support swiglu_alpha"
    if moe_config.swiglu_beta is not None:
        return "kernel does not support swiglu_beta"
    return None


def interleave_hpc_sm120_gate_up(w13: torch.Tensor) -> torch.Tensor:
    """Interleave `[E, 2 * I, ...]` stacked `[gate; up]` rows 16 at a time,
    `[gate16, up16, gate16, ...]`, as the SM120 fused MoE reads them."""
    num_experts, rows = w13.shape[:2]
    halves = w13.view(num_experts, 2, rows // 32, 16, *w13.shape[2:])
    return halves.transpose(1, 2).reshape(w13.shape).contiguous()


def to_hpc_sm120_scale_layout(scale: torch.Tensor) -> torch.Tensor:
    """Move `[E, rows, K // 32]` UE8M0 scales into the per-expert flashinfer
    `layout_128x4`, as uint8. Rows and `K // 32` must be multiples of 128 and
    4."""
    num_experts, rows, scale_k = scale.shape
    return (
        scale.view(torch.uint8)
        .reshape(num_experts, rows // 128, 4, 32, scale_k // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .reshape(num_experts, rows, scale_k)
        .contiguous()
    )


def prepare_mx_moe_layer_for_hpc_sm120(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lay MXFP8 or MXFP4 expert weights and their `[E, rows, K // 32]` UE8M0
    scales out for the SM120 fused MoE: gate/up rows interleaved, scales in
    `layout_128x4`."""
    return (
        interleave_hpc_sm120_gate_up(w13),
        w2,
        to_hpc_sm120_scale_layout(interleave_hpc_sm120_gate_up(w13_scale)),
        to_hpc_sm120_scale_layout(w2_scale),
    )


def prepare_fp8_moe_layer_for_hpc_sm120(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert 128-block FP8 expert weights for the SM120 fused MoE: requantize
    them in place against UE8M0 block scales and repeat each scale over its
    block."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        requant_weight_ue8m0_inplace,
    )

    ue8m0 = []
    for weight, scale in ((w13, w13_scale), (w2, w2_scale)):
        requant_weight_ue8m0_inplace(weight, scale, block_size=(128, 128))
        rows, cols = weight.shape[1:]
        ue8m0.append(
            scale.to(torch.float8_e8m0fnu)
            .view(torch.uint8)
            .repeat_interleave(128, dim=1)
            .repeat_interleave(4, dim=2)[:, :rows, : cols // 32]
        )
    return prepare_mx_moe_layer_for_hpc_sm120(w13, w2, ue8m0[0], ue8m0[1])


class HPCExperts(mk.FusedMoEExpertsModular):
    """MoE implementation powered by [HPC](https://github.com/Tencent/hpc-ops).

    Supported on NVIDIA Hopper (e.g. H20, H200) for per-tensor and 128-block
    FP8 models such as Hy3-FP8 and Qwen3-235B-A22B-FP8, and on RTX PRO
    Blackwell (SM120) for 128-block FP8 models such as Qwen3.8-Flash-Next-FP8.
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
        self.is_sm120 = current_platform.is_device_capability(120)

    @property
    def expects_unquantized_inputs(self) -> bool:
        # BF16 activations are quantized to MXFP8 in hpc_fuse_moe_mxfp8_sm120.
        return self.is_sm120

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: mk.FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        if moe_config.swiglu_limit is not None:
            return False, "kernel does not support SwiGLU clamp"
        return mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        if not (p.is_cuda() and has_hpc()):
            return False
        if p.is_device_capability(90):
            return True
        return p.is_device_capability(120) and has_hpc_sm120_moe()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        scheme = (weight_key, activation_key)
        if current_platform.is_device_capability(120):
            return scheme == (kFp8Static128BlockSym, kFp8Dynamic128Sym)
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

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: mk.FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not supported or not current_platform.is_device_capability(120):
            return supported, reason
        if reason := _unsupported_expert_placement(moe_config):
            return False, reason
        # The SM120 kernel tiles N and K by 128. TP shards that split a
        # 128-block reach here already padded to whole blocks (see
        # pad_tp_shard_to_weight_blocks).
        if moe_config.intermediate_size_per_partition % 128 != 0:
            return False, (
                "kernel does not support intermediate_size_per_partition="
                f"{moe_config.intermediate_size_per_partition} (need a multiple "
                "of 128)"
            )
        reason = _unsupported_sm120_config(moe_config)
        return reason is None, reason

    def supports_expert_map(self) -> bool:
        return False

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
        """Compute the shapes for the temporary and final outputs of the two gemms
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

        if self.is_sm120:
            hpc_fuse_moe_mxfp8_sm120(
                x=hidden_states,
                x_scale=None,
                gate_up_weight=w1,
                gate_up_weight_scale=self.quant_config.w1_scale,
                down_weight=w2,
                down_weight_scale=self.quant_config.w2_scale,
                topk_ids=topk_ids,
                topk_scale=topk_weights,
                rank_ep=self.ep_rank,
                num_expert_total=global_num_experts,
                output=output,
                swiglu_limit=self.quant_config.gemm1_clamp_limit,
            )
        elif self.quant_config.is_block_quantized:
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


class HPCMxfp4Experts(mk.FusedMoEExpertsModular):
    """MXFP4 weight x MXFP8 activation MoE powered by
    [HPC](https://github.com/Tencent/hpc-ops).

    Supported on RTX PRO Blackwell (SM120) for MXFP4 MoE checkpoints such as
    DeepSeek-V4-Flash.
    """

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self.ep_rank = moe_config.moe_parallel_config.ep_rank

    @property
    def expects_unquantized_inputs(self) -> bool:
        # BF16 activations are quantized to MXFP8 in hpc_fuse_moe_mxfp8_sm120.
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return p.is_cuda() and p.is_device_capability(120) and has_hpc_sm120_moe()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp4Static, kMxfp8Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
        ]

    @staticmethod
    def _supports_shape(hidden_dim: int) -> bool:
        return hidden_dim % 128 == 0

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: mk.FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not supported:
            return supported, reason
        reason = _unsupported_expert_placement(moe_config) or _unsupported_sm120_config(
            moe_config
        )
        return reason is None, reason

    def supports_expert_map(self) -> bool:
        return False

    def supports_chunking(self) -> bool:
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
        workspace1 = (M, K)
        workspace2 = (0,)
        output_shape = (M, K)
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

        hpc_fuse_moe_mxfp8_sm120(
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
            swiglu_limit=self.quant_config.gemm1_clamp_limit,
        )
