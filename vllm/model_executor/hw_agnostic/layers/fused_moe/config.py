# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import namedtuple
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar, Union

import torch

from vllm.config import ParallelConfig, SchedulerConfig
from vllm.config.kernel import MoEBackend
from vllm.distributed import get_dp_group, get_pcp_group, get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.hw_agnostic.layers.fused_moe.activation import (
    MoEActivation,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_triton_kernels
from vllm.utils.math_utils import cdiv


class _GroupShape(namedtuple("_GroupShape", ["row", "col"])):
    pass


class GroupShape(_GroupShape):
    PER_TENSOR: ClassVar["GroupShape"]
    PER_TOKEN: ClassVar["GroupShape"]


GroupShape.PER_TENSOR = GroupShape(-1, -1)
GroupShape.PER_TOKEN = GroupShape(1, -1)

logger = init_logger(__name__)

if has_triton_kernels():
    try:
        from triton_kernels.matmul_ogs import PrecisionConfig
    except (ImportError, AttributeError) as e:
        logger.error(
            "Failed to import Triton kernels. Please make sure your triton "
            "version is compatible. Error: %s",
            e,
        )


def _get_config_dtype_str(
    dtype: torch.dtype,
    use_fp8_w8a8: bool = False,
    use_fp8_w8a16: bool = False,
    use_int8_w8a16: bool = False,
) -> str | None:
    """
    Return a string used to construct the filename that contains the
    tuning info for a particular quantization scheme.  See
    try_get_optimal_moe_config in fused_moe.py.
    """
    if use_fp8_w8a8:
        return "fp8_w8a8"
    if use_fp8_w8a16:
        return "fp8_w8a16"
    if use_int8_w8a16:
        return "int8_w8a16"
    if dtype == torch.float:
        # float32 MoE reuses the fp16/bfloat16 tuning configs.
        return "float32"
    return None


def _quant_flags_to_group_shape(
    quant_dtype: torch.dtype | str | None,
    per_act_token_quant: bool,
    per_out_ch_quant: bool,
    block_shape: list[int] | None,
) -> tuple[GroupShape | None, GroupShape | None]:
    """
    Convert MoE quantization flags into more generic GroupShapes.
    """
    a_shape: GroupShape | None
    w_shape: GroupShape | None
    if block_shape is not None:
        assert not per_act_token_quant
        assert not per_out_ch_quant
        a_shape = GroupShape(row=block_shape[0], col=block_shape[1])
        w_shape = GroupShape(row=block_shape[0], col=block_shape[1])
    else:
        w_shape = None
        a_shape = None if quant_dtype is None else GroupShape.PER_TENSOR

        if per_act_token_quant:
            a_shape = GroupShape.PER_TOKEN

        if per_out_ch_quant:
            w_shape = GroupShape.PER_TOKEN

    return a_shape, w_shape


# Keep enum values in sync with the FlashInfer counterpart at
# https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/trtllm/fused_moe/runner.h
class RoutingMethodType(IntEnum):
    Default = (0,)
    Renormalize = (1,)
    DeepSeekV3 = (2,)
    Llama4 = (3,)
    RenormalizeNaive = (4,)
    TopK = (5,)
    SigmoidRenorm = (6,)
    MiniMax2 = (7,)
    Sigmoid = (8,)
    Unspecified = (9,)
    # Routing types not passed to FlashInfer kernels:
    DeepseekV4 = (100,)
    Custom = (101,)
    Simulated = (102,)


def get_routing_method_type(
    scoring_func: str,
    top_k: int,
    renormalize: bool,
    num_expert_group: int | None,
    has_e_score_bias: bool,
    routed_scaling_factor: float | None = 1.0,
) -> RoutingMethodType:
    if scoring_func == "sqrtsoftplus":
        if renormalize:
            return RoutingMethodType.DeepseekV4
        else:
            return RoutingMethodType.Unspecified

    if has_e_score_bias:
        if scoring_func == "sigmoid":
            if not renormalize:
                return RoutingMethodType.Unspecified
            if (num_expert_group or 0) > 0:
                return RoutingMethodType.DeepSeekV3
            if routed_scaling_factor in (None, 1.0):
                return RoutingMethodType.MiniMax2
            return RoutingMethodType.Unspecified
        else:
            return RoutingMethodType.Unspecified

    if scoring_func == "sigmoid":
        if renormalize:
            return RoutingMethodType.SigmoidRenorm
        return RoutingMethodType.Sigmoid

    if scoring_func == "softmax":
        if renormalize:
            return RoutingMethodType.RenormalizeNaive
        else:
            return RoutingMethodType.Default

    return RoutingMethodType.Unspecified


@dataclass
class FusedMoEQuantDesc:
    """
    A quantization descriptor for fused MoE ops. This class can describe
    either activations or weights.
    """

    # The quantized type of this parameters.  None means unquantized or
    # already quantized.
    dtype: torch.dtype | str | None = None

    # A field that describes the quantization group shape, from quant_utils.py.
    #  * (-1, -1)   for per-tensor quantization
    #  * (1, -1)    for per-row quantization
    #  * (-1, 1)    for per-column quantization
    #  * (128, 128) for 128x128 deepseek style block quantization
    #  * (1, 128)   for deepseek style activation quantization
    #               (i.e. per-token-per-group)
    shape: GroupShape | None = None

    # Quantization scales.
    scale: Union[torch.Tensor, "PrecisionConfig", None] = None

    # Per-channel scales for W4A8 FP8.
    alpha_or_gscale: torch.Tensor | None = None

    # Zero points for int4/int8 types
    zp: torch.Tensor | None = None

    # Biases for GPT triton MoE
    bias: torch.Tensor | None = None


@dataclass
class FusedMoEQuantConfig:
    """Quantization parameters for a single FusedMoEMethodBase.

    Composed of four FusedMoEQuantDescs (a1/a2 for activations, w1/w2 for
    weights). Each FusedMoEMethodBase implements get_fused_moe_quant_config
    to build one. Constraints: paired activations / paired weights share
    the same GroupShape, activations are per-token / per-tensor / K-blocked
    only, weights need no GroupShape (they are already quantized).
    """

    _a1: FusedMoEQuantDesc
    _a2: FusedMoEQuantDesc
    _w1: FusedMoEQuantDesc
    _w2: FusedMoEQuantDesc

    # Clamp limit threaded to the silu+clamp fused activation.
    gemm1_clamp_limit: float | None = None

    def __post_init__(self):
        assert not self.per_act_token_quant or self.block_shape is None, (
            "illegal quantization"
        )

    #
    # Convenience accessors for various properties.
    #

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        return self._a1.dtype

    @property
    def weight_quant_dtype(self) -> torch.dtype | str | None:
        return self._w1.dtype

    @property
    def is_quantized(self) -> bool:
        return self.quant_dtype is not None

    @property
    def is_per_act_token(self) -> bool:
        return self._a1.shape == GroupShape.PER_TOKEN

    @property
    def per_act_token_quant(self) -> bool:
        return self._a1.shape == GroupShape.PER_TOKEN

    @property
    def per_out_ch_quant(self) -> bool:
        return self._w1.shape == GroupShape.PER_TOKEN

    @property
    def is_per_tensor(self) -> bool:
        return self._a1.shape == GroupShape.PER_TENSOR

    @property
    def block_shape(self) -> list[int] | None:
        if (
            self._a1.shape is not None
            and self._a1.shape != GroupShape.PER_TENSOR
            and self._a1.shape != GroupShape.PER_TOKEN
        ):
            return [self._a1.shape.row, self._a1.shape.col]
        else:
            return None

    @property
    def is_block_quantized(self) -> bool:
        return self.block_shape is not None

    @property
    def a1_scale(self) -> torch.Tensor | None:
        assert self._a1.scale is None or isinstance(self._a1.scale, torch.Tensor)
        return self._a1.scale

    @property
    def a2_scale(self) -> torch.Tensor | None:
        assert self._a2.scale is None or isinstance(self._a2.scale, torch.Tensor)
        return self._a2.scale

    @property
    def a1_gscale(self) -> torch.Tensor | None:
        return self._a1.alpha_or_gscale

    @property
    def a2_gscale(self) -> torch.Tensor | None:
        return self._a2.alpha_or_gscale

    @property
    def w1_scale(self) -> torch.Tensor | None:
        assert self._w1.scale is None or isinstance(self._w1.scale, torch.Tensor)
        return self._w1.scale

    @property
    def w1_zp(self) -> torch.Tensor | None:
        return self._w1.zp

    @property
    def w1_bias(self) -> torch.Tensor | None:
        return self._w1.bias

    @property
    def w1_precision(self) -> "PrecisionConfig | None":
        assert self._w1.scale is None or isinstance(self._w1.scale, PrecisionConfig)
        return self._w1.scale

    @property
    def g1_alphas(self) -> torch.Tensor | None:
        return self._w1.alpha_or_gscale

    @property
    def w2_scale(self) -> torch.Tensor | None:
        assert self._w2.scale is None or isinstance(self._w2.scale, torch.Tensor)
        return self._w2.scale

    @property
    def w2_zp(self) -> torch.Tensor | None:
        return self._w2.zp

    @property
    def w2_bias(self) -> torch.Tensor | None:
        return self._w2.bias

    @property
    def w2_precision(self) -> "PrecisionConfig | None":
        assert self._w2.scale is None or isinstance(self._w2.scale, PrecisionConfig)
        return self._w2.scale

    @property
    def g2_alphas(self) -> torch.Tensor | None:
        return self._w2.alpha_or_gscale

    @property
    def use_fp8_w8a8(self) -> bool:
        return self.quant_dtype == current_platform.fp8_dtype()

    @property
    def use_int8_w8a8(self) -> bool:
        return self.quant_dtype == torch.int8

    @property
    def use_int8_w8a16(self) -> bool:
        return self._a1.dtype is None and self._w1.dtype == torch.int8

    @property
    def use_fp8_w8a16(self) -> bool:
        return self._a1.dtype is None and self._w1.dtype == current_platform.fp8_dtype()

    def config_name(self, dtype: torch.dtype) -> str | None:
        """
        Return a string used to construct the filename that contains the
        tuning info for a particular quantization scheme.  See
        try_get_optimal_moe_config in fused_moe.py.
        """
        return _get_config_dtype_str(
            use_fp8_w8a8=self.use_fp8_w8a8,
            use_fp8_w8a16=self.use_fp8_w8a16,
            use_int8_w8a16=self.use_int8_w8a16,
            dtype=dtype,
        )

    def scale_shape(
        self,
        max_tokens: int,
        hidden_dim: int,
    ) -> tuple[int, int] | None:
        """
        Construct the proper activation scale shape for this
        config.
        """
        if self.is_quantized:
            if self.is_block_quantized:
                assert self.block_shape is not None
                _, block_k = self.block_shape
                k_tiles = cdiv(hidden_dim, block_k)
                return (max_tokens, k_tiles)
            elif self.is_per_act_token:
                return (max_tokens, 1)
            else:
                return (1, 1)
        else:
            return None

    def batched_scale_shape(
        self,
        num_experts: int,
        max_tokens: int,
        hidden_dim: int,
    ) -> tuple[int, int, int] | None:
        """
        Construct the proper activation batched scale shape for this
        config, e.g. (num experts, *scale_shape).
        """
        if self.is_quantized:
            scale_shape = self.scale_shape(max_tokens, hidden_dim)
            assert scale_shape is not None
            return (num_experts, *scale_shape)
        else:
            return None

    @staticmethod
    def make(
        quant_dtype: torch.dtype | str | None = None,
        per_act_token_quant: bool = False,
        per_out_ch_quant: bool = False,
        block_shape: list[int] | None = None,
        w1_scale: Union[torch.Tensor, "PrecisionConfig", None] = None,
        w2_scale: Union[torch.Tensor, "PrecisionConfig", None] = None,
        a1_scale: torch.Tensor | None = None,
        a2_scale: torch.Tensor | None = None,
        g1_alphas: torch.Tensor | None = None,
        g2_alphas: torch.Tensor | None = None,
        w1_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
        w1_zp: torch.Tensor | None = None,
        w2_zp: torch.Tensor | None = None,
        weight_dtype: torch.dtype | str | None = None,
        gemm1_clamp_limit: float | None = None,
    ) -> "FusedMoEQuantConfig":
        """General builder for a FusedMoEQuantConfig used by the Triton path.

        Supported dtypes: BF16/FP16 (``quant_dtype=None``), FP8 W8A8
        (per-tensor or block-quant via ``block_shape``), FP8 W8A16,
        INT8 W8A8/W8A16, INT4 W4A16. ``g{1,2}_alphas`` carry the
        per-channel scales / dq scales used by the FP8 W4A8 variants.
        """
        assert not isinstance(weight_dtype, str) or weight_dtype == "int4"
        assert not isinstance(quant_dtype, str), (
            f"Unsupported quant_dtype={quant_dtype!r} on the hw-agnostic FusedMoE path."
        )

        if weight_dtype is None:
            weight_dtype = quant_dtype

        a_shape, w_shape = _quant_flags_to_group_shape(
            quant_dtype, per_act_token_quant, per_out_ch_quant, block_shape
        )
        quant_config = FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(quant_dtype, a_shape, a1_scale),
            _a2=FusedMoEQuantDesc(quant_dtype, a_shape, a2_scale),
            _w1=FusedMoEQuantDesc(
                weight_dtype, w_shape, w1_scale, g1_alphas, w1_zp, w1_bias
            ),
            _w2=FusedMoEQuantDesc(
                weight_dtype, w_shape, w2_scale, g2_alphas, w2_zp, w2_bias
            ),
            gemm1_clamp_limit=gemm1_clamp_limit,
        )
        assert quant_config.per_act_token_quant == per_act_token_quant
        assert quant_config.per_out_ch_quant == per_out_ch_quant
        assert quant_config.block_shape == block_shape
        return quant_config


def fp8_w8a8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    per_act_token_quant: bool = False,
    per_out_ch_quant: bool = False,
    block_shape: list[int] | None = None,
    g1_alphas: torch.Tensor | None = None,
    g2_alphas: torch.Tensor | None = None,
    gemm1_clamp_limit: float | None = None,
) -> FusedMoEQuantConfig:
    """Quant config for FP8 activations and FP8 weights (per-tensor or block)."""
    return FusedMoEQuantConfig.make(
        current_platform.fp8_dtype(),
        w1_scale=w1_scale,
        g1_alphas=g1_alphas,
        w2_scale=w2_scale,
        g2_alphas=g2_alphas,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        per_act_token_quant=per_act_token_quant,
        per_out_ch_quant=per_out_ch_quant,
        block_shape=block_shape,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )


def mxfp4_w4a16_moe_quant_config(
    w1_scale: Union[torch.Tensor, "PrecisionConfig"],
    w2_scale: Union[torch.Tensor, "PrecisionConfig"],
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    gemm1_clamp_limit: float | None = None,
) -> FusedMoEQuantConfig:
    """Quant config for unquantized (BF16) activations and MXFP4 weights.

    The OAI Triton MXFP4 path stores the swizzled scale tensor inside a
    ``PrecisionConfig`` (a wrapper from ``triton_kernels.matmul_ogs``)
    rather than a plain ``torch.Tensor``; the modular pipeline only reads
    it back through ``FusedMoEQuantConfig.w{1,2}_precision``.
    """
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(),
        _a2=FusedMoEQuantDesc(),
        _w1=FusedMoEQuantDesc("mxfp4", None, w1_scale, None, None, w1_bias),
        _w2=FusedMoEQuantDesc("mxfp4", None, w2_scale, None, None, w2_bias),
        gemm1_clamp_limit=gemm1_clamp_limit,
    )


def int8_w8a8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    per_act_token_quant: bool = False,
) -> FusedMoEQuantConfig:
    """Quant config for INT8 activations and INT8 weights."""
    return FusedMoEQuantConfig.make(
        torch.int8,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        per_act_token_quant=per_act_token_quant,
        per_out_ch_quant=False,
        block_shape=None,
    )


def fp8_w8a16_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    gemm1_clamp_limit: float | None = None,
) -> FusedMoEQuantConfig:
    """Quant config for 16-bit float activations and fp8 weights."""
    group_shape = GroupShape(*block_shape) if block_shape is not None else None
    fp8_dtype = current_platform.fp8_dtype()
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(),
        _a2=FusedMoEQuantDesc(),
        _w1=FusedMoEQuantDesc(
            fp8_dtype,
            group_shape,
            w1_scale,
            None,
            None,
            w1_bias,
        ),
        _w2=FusedMoEQuantDesc(
            fp8_dtype,
            group_shape,
            w2_scale,
            None,
            None,
            w2_bias,
        ),
        gemm1_clamp_limit=gemm1_clamp_limit,
    )


def int8_w8a16_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
) -> FusedMoEQuantConfig:
    """Quant config for 16-bit float activations and INT8 weights."""
    group_shape = GroupShape(*block_shape) if block_shape is not None else None
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(shape=group_shape),
        _a2=FusedMoEQuantDesc(shape=group_shape),
        _w1=FusedMoEQuantDesc(torch.int8, group_shape, w1_scale, None, w1_zp, w1_bias),
        _w2=FusedMoEQuantDesc(torch.int8, group_shape, w2_scale, None, w2_zp, w2_bias),
    )


def biased_moe_quant_config(
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    gemm1_clamp_limit: float | None = None,
) -> FusedMoEQuantConfig:
    """Quant config for unquantized activations with biases."""
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(),
        _a2=FusedMoEQuantDesc(),
        _w1=FusedMoEQuantDesc(bias=w1_bias),
        _w2=FusedMoEQuantDesc(bias=w2_bias),
        gemm1_clamp_limit=gemm1_clamp_limit,
    )


# A FusedMoEQuantConfig constant for an unquantized MoE op.
FUSED_MOE_UNQUANTIZED_CONFIG: FusedMoEQuantConfig = FusedMoEQuantConfig.make()


@dataclass
class FusedMoEParallelConfig:
    tp_size: int
    pcp_size: int
    dp_size: int
    ep_size: int
    tp_rank: int
    pcp_rank: int
    dp_rank: int
    ep_rank: int
    sp_size: int

    use_ep: bool  # whether to use EP or not
    all2all_backend: str  # all2all backend for MoE communication
    enable_eplb: bool  # whether to enable expert load balancing

    @property
    def is_sequence_parallel(self) -> bool:
        return self.sp_size > 1

    @property
    def use_all2all_kernels(self):
        return self.dp_size > 1 and self.use_ep

    @staticmethod
    def flatten_tp_across_dp_and_pcp(
        tp_size: int, dp_size: int, dp_rank: int, pcp_size: int, pcp_rank: int
    ) -> tuple[int, int]:
        tp_rank = 0 if tp_size == 1 else get_tensor_model_parallel_rank()
        # There are actually dp_size * pcp_size * tp_size devices.
        # Update tp_size and tp_rank so we shard across all devices.
        flatten_tp_size = dp_size * pcp_size * tp_size
        flatten_tp_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank
        return flatten_tp_size, flatten_tp_rank

    @staticmethod
    def make(
        tp_size_: int,
        pcp_size_: int,
        dp_size_: int,
        sp_size_: int,
        vllm_parallel_config: ParallelConfig,
    ) -> "FusedMoEParallelConfig":
        """Compute the MoE parallel configuration.

        When ``enable_expert_parallel`` is set and there is more than one
        device across TP/DP/PCP, the MoE layer runs in EP mode (TP is
        flattened into EP across the DP+PCP+TP product). PCP plays the
        same role as DP for sharding purposes.
        """
        use_ep = (
            dp_size_ * pcp_size_ * tp_size_ > 1
            and vllm_parallel_config.enable_expert_parallel
        )

        dp_size = dp_size_
        dp_rank = get_dp_group().rank_in_group if dp_size > 1 else 0
        pcp_size = pcp_size_
        pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0
        tp_size, tp_rank = FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp(
            tp_size_, dp_size_, dp_rank, pcp_size_, pcp_rank
        )

        if not use_ep:
            return FusedMoEParallelConfig(
                tp_size=tp_size,
                tp_rank=tp_rank,
                pcp_size=pcp_size,
                pcp_rank=pcp_rank,
                dp_size=dp_size,
                dp_rank=dp_rank,
                ep_size=1,
                ep_rank=0,
                sp_size=sp_size_,
                use_ep=False,
                all2all_backend=vllm_parallel_config.all2all_backend,
                enable_eplb=vllm_parallel_config.enable_eplb,
            )
        # DP + EP / TP + EP / DP + TP + EP
        assert use_ep
        # In EP, each device owns a set of experts fully. There is no tensor
        # parallel update tp_size, tp_rank, ep_size and ep_rank to reflect that.
        ep_size = tp_size
        ep_rank = tp_rank
        return FusedMoEParallelConfig(
            tp_size=1,
            tp_rank=0,
            pcp_size=pcp_size,
            pcp_rank=pcp_rank,
            dp_size=dp_size,
            dp_rank=dp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            sp_size=sp_size_,
            use_ep=True,
            all2all_backend=vllm_parallel_config.all2all_backend,
            enable_eplb=vllm_parallel_config.enable_eplb,
        )

    @classmethod
    def make_no_parallel(cls) -> "FusedMoEParallelConfig":
        """For usage in CI/CD and testing."""
        return FusedMoEParallelConfig(
            tp_size=1,
            tp_rank=0,
            pcp_size=1,
            pcp_rank=0,
            dp_size=1,
            dp_rank=0,
            ep_size=1,
            ep_rank=0,
            sp_size=1,
            use_ep=False,
            all2all_backend="allgather_reducescatter",
            enable_eplb=False,
        )


# Adapted from pplx-kernels tests/all_to_all_utils.py
@dataclass
class FusedMoEConfig:
    num_experts: int
    experts_per_token: int
    hidden_dim: int
    intermediate_size: int
    num_local_experts: int
    num_logical_experts: int
    activation: MoEActivation
    device: torch.device | str
    routing_method: RoutingMethodType
    moe_parallel_config: FusedMoEParallelConfig

    # The activation type.
    in_dtype: torch.dtype

    # Defaults to in_dtype if not specified.
    router_logits_dtype: torch.dtype | None = None

    # Defaults to hidden_dim if not specified.
    hidden_dim_unpadded: int | None = None
    # Defaults to intermediate_size_per_partition if not specified.
    intermediate_size_per_partition_unpadded: int | None = None

    moe_backend: MoEBackend = "auto"
    max_num_tokens: int = SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP
    has_bias: bool = False
    is_lora_enabled: bool = False

    # Clamp limit threaded through to the silu+clamp fused activation.
    swiglu_limit: float | None = None

    max_capture_size: int = 0

    # Set by __post_init__.
    intermediate_size_per_partition: int = -1

    def __post_init__(self):
        tp_size = self.moe_parallel_config.tp_size
        assert self.intermediate_size % tp_size == 0
        self.intermediate_size_per_partition = self.intermediate_size // tp_size

        if self.dp_size > 1:
            logger.debug_once(
                "Using FusedMoEConfig::max_num_tokens=%d", self.max_num_tokens
            )

        assert self.max_num_tokens > 0

        if self.router_logits_dtype is None:
            self.router_logits_dtype = self.in_dtype

        if self.hidden_dim_unpadded is None:
            self.hidden_dim_unpadded = self.hidden_dim
        if self.intermediate_size_per_partition_unpadded is None:
            self.intermediate_size_per_partition_unpadded = (
                self.intermediate_size_per_partition
            )

    @property
    def is_act_and_mul(self) -> bool:
        return self.activation.is_gated

    @property
    def tp_size(self):
        return self.moe_parallel_config.tp_size

    @property
    def dp_size(self):
        return self.moe_parallel_config.dp_size

    @property
    def pcp_size(self):
        return self.moe_parallel_config.pcp_size

    @property
    def ep_size(self):
        return self.moe_parallel_config.ep_size

    @property
    def sp_size(self):
        return self.moe_parallel_config.sp_size

    @property
    def is_sequence_parallel(self):
        return self.moe_parallel_config.is_sequence_parallel

    @property
    def tp_rank(self):
        return self.moe_parallel_config.tp_rank

    @property
    def dp_rank(self):
        return self.moe_parallel_config.dp_rank

    @property
    def pcp_rank(self):
        return self.moe_parallel_config.pcp_rank

    @property
    def ep_rank(self):
        return self.moe_parallel_config.ep_rank

    @property
    def use_ep(self):
        return self.moe_parallel_config.use_ep
