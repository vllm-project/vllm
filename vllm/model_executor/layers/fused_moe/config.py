# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from enum import IntEnum
from typing import Union

import torch

import vllm.envs as envs
from vllm.config import ParallelConfig
from vllm.distributed import (
    get_dp_group,
    get_pcp_group,
    get_tensor_model_parallel_rank,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    OCP_MX_DTYPES,
    OCP_MX_Scheme,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_triton_kernels
from vllm.utils.math_utils import cdiv

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
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
) -> str | None:
    """
    Return a string used to construct the filename that contains the
    tuning info for a particular quantization scheme.  See
    try_get_optimal_moe_config in fused_moe.py.
    """
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_fp8_w8a16:
        return "fp8_w8a16"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif use_int4_w4a16:
        return "int4_w4a16"
    elif ocp_mx_scheme is not None:
        # The output of this function is passed to `try_get_optimal_moe_config`,
        # and as we only simulate OCP MX execution in fused_moe for now,
        # we will NOT look for `*,dtype=w_mxfp4_a_mxfp4.json` for now.
        return None
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
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
        # TODO(bnell): this is not quite right for activations since first
        # dim should be 1.
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


# The type of method in top-K routing
# Please keep this in sync with the counterpart defined in https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/trtllm/fused_moe/runner.h
class RoutingMethodType(IntEnum):
    # Default: Softmax -> TopK
    Default = (0,)
    # Renormalize: TopK -> Softmax/Sigmoid
    Renormalize = (1,)
    # DeepSeekV3: Sigmoid -> RoutingBiasAdd -> Top2 in group -> Top4 groups
    # -> Top8 experts from the Top4 groups
    DeepSeekV3 = (2,)
    # Llama4: Top1 -> Sigmoid
    Llama4 = (3,)
    # RenormalizeNaive: Softmax/Sigmoid -> TopK -> Renormalize
    RenormalizeNaive = (4,)
    # TopK: TopK (no softmax)
    TopK = (5,)
    # Custom
    Custom = (6,)
    # Simulated
    Simulated = (7,)
    # Unspecified
    Unspecified = 8.0


def get_routing_method_type(
    scoring_func: str, top_k: int, renormalize: bool
) -> RoutingMethodType:
    if scoring_func == "sigmoid":
        if top_k == 1:
            return RoutingMethodType.Llama4
        else:
            return RoutingMethodType.DeepSeekV3
    elif scoring_func == "softmax":
        if renormalize:
            return RoutingMethodType.Renormalize
        else:
            return RoutingMethodType.Default
    else:
        return RoutingMethodType.Unspecified


@dataclass
class FusedMoEQuantDesc:
    """
    A quantization descriptor for fused MoE ops. This class can describe
    either activations or weights.
    """

    # The quantized type of this parameters.  None means unquantized or
    # already quantized.
    # TODO (bnell): use scalar_type instead of Union.
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
    # TODO(bnell): maybe put PrecisionConfigs in subclass of QuantDesc?
    scale: Union[torch.Tensor, "PrecisionConfig", None] = None

    # Quantization alphas or gscales, used for nvfp4 types.
    # W4A8 FP8: used for per-channel scales
    # TODO(bnell): put some of these in subclasses
    alpha_or_gscale: torch.Tensor | None = None

    # Zero points for int4/int8 types
    zp: torch.Tensor | None = None

    # Biases for GPT triton MoE
    bias: torch.Tensor | None = None


# TODO(bnell): have subclasses for specific moe methods?
# e.g. for specific arguments bias, precision, etc.
@dataclass
class FusedMoEQuantConfig:
    """
    The FusedMoEQuantConfig contains all the quantization parameters for
    a single FusedMoEMethodBase operation.  It consists of four
    FusedMoEQuantDescs, one for each activation and set of weights.

    Each FusedMoEMethodBase must implement a get_fused_moe_quant_config
    method to construct a FusedMoEQuantConfig for use with that class.

    FusedMoEQuant configs are only used for modular kernels, fused_experts
    (from fused_moe.py), cutlass_moe_fp[48], rocm_aiter_fused_experts and
    triton_kernel_moe_forward.  Other MoE methods can ignore the
    FusedMoEQuantConfig (for now) and hardcode it to None.

    There are currently some restrictions on what can be expressed:
    - Most MoE ops only support similar quantization strategies for
      each parameter, e.g. both weights must have the same GroupShape
      and both activations must share the same GroupShape.  One exception to
      this is the cutlass moe which allows per channel quantization on the
      outputs.  Note: this restrictions are not always rigorously checked.
    - Not all fused MoE functions support all the parameters, e.g. zero points,
      global scales, alphas and biases are not universally supported.
    - Fully general GroupShapes are not allowed.  Activations only support
      per token, per tensor or K-blocked.
    - Weights are not required to have a GroupShape since they have already
      been quantized.

    Other notes:
    - PrecisionConfigs are specific to GPT OSS Triton.
    - As a follow up it would probably make sense to subclass FusedMoEQuantDesc
      or FusedMoEQuantConfig for particular FusedMoEMethodBase subclasses
      so that only the required quantization parameters are used/stored.
    """

    # TODO(bnell) make sure a1_scales/a2_scales don't interfere with chunking
    _a1: FusedMoEQuantDesc
    _a2: FusedMoEQuantDesc
    _w1: FusedMoEQuantDesc
    _w2: FusedMoEQuantDesc

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
    def a1_gscale(self) -> torch.Tensor | None:
        return self._a1.alpha_or_gscale

    @property
    def a2_scale(self) -> torch.Tensor | None:
        assert self._a2.scale is None or isinstance(self._a2.scale, torch.Tensor)
        return self._a2.scale

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
        return self.quant_dtype == torch.float8_e4m3fn

    @property
    def use_int8_w8a8(self) -> bool:
        return self.quant_dtype == torch.int8

    @property
    def use_int8_w8a16(self) -> bool:
        return self._a1.dtype is None and self._w1.dtype == torch.int8

    @property
    def use_fp8_w8a16(self) -> bool:
        return self._a1.dtype is None and self._w1.dtype == current_platform.fp8_dtype()

    @property
    def use_int4_w4a16(self) -> bool:
        return self._a1.dtype is None and self._w1.dtype == "int4"

    @property
    def use_nvfp4_w4a16(self) -> bool:
        return self._a1.dtype is None and self._w1.dtype == "nvfp4"

    @property
    def ocp_mx_scheme(self) -> str | None:
        if not hasattr(self, "_ocp_mx_scheme"):
            if (self._a1.dtype is not None and not isinstance(self._a1.dtype, str)) or (
                self._w1.dtype is not None and not isinstance(self._w1.dtype, str)
            ):
                self._ocp_mx_scheme = None
            else:
                ocp_mx_scheme = OCP_MX_Scheme.from_quant_dtype(
                    self._a1.dtype, self._w1.dtype
                )

                if ocp_mx_scheme is not None:
                    ocp_mx_scheme = ocp_mx_scheme.value

                self._ocp_mx_scheme = ocp_mx_scheme

        return self._ocp_mx_scheme

    @property
    def use_mxfp4_w4a16(self) -> bool:
        return self._a1.dtype is None and self._w1.dtype == "mxfp4"

    @property
    def use_mxfp4_w4a4(self) -> bool:
        return self._a1.dtype == "mxfp4" and self._w1.dtype == "mxfp4"

    @property
    def use_nvfp4_w4a4(self) -> bool:
        return self.quant_dtype == "nvfp4"

    @property
    def use_mxfp4_w4a8(self) -> bool:
        return self._a1.dtype == "fp8" and self._w1.dtype == "mxfp4"

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
            use_int4_w4a16=self.use_int4_w4a16,
            ocp_mx_scheme=self.ocp_mx_scheme,
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
        a1_gscale: torch.Tensor | None = None,
        a2_gscale: torch.Tensor | None = None,
        w1_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
        w1_zp: torch.Tensor | None = None,
        w2_zp: torch.Tensor | None = None,
        weight_dtype: torch.dtype | str | None = None,
    ) -> "FusedMoEQuantConfig":
        """
        General builder function for a FusedMoEQuantConfig.
        - quant_dtype: Optional quantization type. None if activations are
          unquantized or quantized prior to calling.  Note: "nvfp4", "mxfp4",
          "mxfp6_e3m2", "mxfp6_e2m3" are the only valid string values
          for quant_dtype.
        - per_act_token_quant: Activations have per token quantization.
        - per_out_ch_quant: Outputs have per channel quantization. (only
          for cutlass).
        - block_shape: Optional block size for block-wise quantization.
          Incompatible with per_act_token and per_out_ch quant.
        - w1_scale: Optional scale to be used for w1.
        - w2_scale: Optional scale to be used for w2.
        - a1_scale: Optional scale to be used for a1.
        - a2_scale: Optional scale to be used for a2.
        - g1_alphas: Optional global quantization scales for w1 (for nvfp4).
                     Optional per-channel scales for w1 (for W4A8 FP8).
                     Optional dq scale i.e. w_scale * a_scale (for W8A8 fp8).
        - g2_alphas: Optional global quantization scales for w2 (for nvfp4).
                     Optional per-channel scales for w2 (for W4A8 FP8).
                     Optional dq scale i.e. w_scale * a_scale (for W8A8 fp8).
        - a1_gscale: Optional global quantization scales for a1 (1.0 /a2_scale).
        - a2_gscale: Optional global quantization scales for a2 (1.0 /a2_scale).

        - w1_bias: Optional biases for w1 (GPT OSS Triton).
        - w2_bias: Optional biases for w1 (GPT OSS Triton).
        - w1_zp: Optional w1 zero points for int4/int8 quantization.
        - w2_zp: Optional w2 zero points for int4/int8 quantization.
        """
        assert not isinstance(quant_dtype, str) or quant_dtype in {
            "nvfp4",
            "mxfp4",
            "mxfp6_e3m2",
            "mxfp6_e2m3",
            "mxfp8",
        }
        assert not isinstance(weight_dtype, str) or weight_dtype in {
            "nvfp4",
            "mxfp4",
            "mxfp6_e3m2",
            "mxfp6_e2m3",
            "int4",
            "mxfp8",
        }

        if weight_dtype is None:
            weight_dtype = quant_dtype

        a_shape, w_shape = _quant_flags_to_group_shape(
            quant_dtype, per_act_token_quant, per_out_ch_quant, block_shape
        )
        quant_config = FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(quant_dtype, a_shape, a1_scale, a1_gscale),
            _a2=FusedMoEQuantDesc(quant_dtype, a_shape, a2_scale, a2_gscale),
            _w1=FusedMoEQuantDesc(
                weight_dtype, w_shape, w1_scale, g1_alphas, w1_zp, w1_bias
            ),
            _w2=FusedMoEQuantDesc(
                weight_dtype, w_shape, w2_scale, g2_alphas, w2_zp, w2_bias
            ),
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
    a1_gscale: torch.Tensor | None = None,
    a2_gscale: torch.Tensor | None = None,
    g1_alphas: torch.Tensor | None = None,
    g2_alphas: torch.Tensor | None = None,
) -> FusedMoEQuantConfig:
    """
    Construct a quant config for fp8 activations and fp8 weights.
    """
    return FusedMoEQuantConfig.make(
        torch.float8_e4m3fn,
        w1_scale=w1_scale,
        g1_alphas=g1_alphas,
        w2_scale=w2_scale,
        g2_alphas=g2_alphas,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        a1_scale=a1_scale,
        a1_gscale=a1_gscale,
        a2_scale=a2_scale,
        a2_gscale=a2_gscale,
        per_act_token_quant=per_act_token_quant,
        per_out_ch_quant=per_out_ch_quant,
        block_shape=block_shape,
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
    """
    Construct a quant config for int8 activations and int8 weights.
    """
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


def gptq_marlin_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    weight_bits: int,
    group_size: int,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
):
    """
    Construct a quant config for gptq marlin quantization.
    """
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    w_shape = None if group_size == -1 else GroupShape(row=1, col=group_size)

    # Activations are NOT quantized for GPTQ (fp16/bf16)
    a_shape = w_shape  # Same as weight shape for alignment

    # Determine weight dtype
    if weight_bits == 4:
        weight_dtype = "int4"
    elif weight_bits == 8:
        weight_dtype = torch.int8
    else:
        raise ValueError(f"Unsupported weight_bits: {weight_bits}")

    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(dtype=None, shape=a_shape),
        _a2=FusedMoEQuantDesc(dtype=None, shape=a_shape),
        _w1=FusedMoEQuantDesc(weight_dtype, w_shape, w1_scale, None, w1_zp, w1_bias),
        _w2=FusedMoEQuantDesc(weight_dtype, w_shape, w2_scale, None, w2_zp, w2_bias),
    )


def mxfp4_w4a16_moe_quant_config(
    w1_scale: Union[torch.Tensor, "PrecisionConfig"],
    w2_scale: Union[torch.Tensor, "PrecisionConfig"],
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> FusedMoEQuantConfig:
    """
    Construct a quant config for unquantized activations and mxfp4 weights.
    """
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(),
        _a2=FusedMoEQuantDesc(),
        _w1=FusedMoEQuantDesc("mxfp4", None, w1_scale, None, None, w1_bias),
        _w2=FusedMoEQuantDesc("mxfp4", None, w2_scale, None, None, w2_bias),
    )


def mxfp4_mxfp8_moe_quant_config(
    w1_scale: Union[torch.Tensor, "PrecisionConfig"],
    w2_scale: Union[torch.Tensor, "PrecisionConfig"],
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
) -> FusedMoEQuantConfig:
    """
    Construct a quant config for mxfp4 activations and mxfp4 weights.
    """
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc("mxfp8"),
        _a2=FusedMoEQuantDesc("mxfp8"),
        _w1=FusedMoEQuantDesc("mxfp4", None, w1_scale, None, None, w1_bias),
        _w2=FusedMoEQuantDesc("mxfp4", None, w2_scale, None, None, w2_bias),
    )


def mxfp4_w4a8_moe_quant_config(
    w1_scale: Union[torch.Tensor, "PrecisionConfig"],
    w2_scale: Union[torch.Tensor, "PrecisionConfig"],
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
) -> FusedMoEQuantConfig:
    """
    Construct a quant config for fp8 activations and mxfp4 weights.
    """
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc("fp8", None, a1_scale, None, None, None),
        _a2=FusedMoEQuantDesc("fp8", None, a2_scale, None, None, None),
        _w1=FusedMoEQuantDesc("mxfp4", None, w1_scale, None, None, w1_bias),
        _w2=FusedMoEQuantDesc("mxfp4", None, w2_scale, None, None, w2_bias),
    )


def ocp_mx_moe_quant_config(
    quant_dtype: str,
    w1_scale: Union[torch.Tensor, "PrecisionConfig"],
    w2_scale: Union[torch.Tensor, "PrecisionConfig"],
    weight_dtype: str | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
) -> FusedMoEQuantConfig:
    """
    Construct a quant config for mxfp4 activations and mxfp4 weights.
    """
    assert quant_dtype in OCP_MX_DTYPES
    return FusedMoEQuantConfig.make(
        quant_dtype=quant_dtype,
        weight_dtype=weight_dtype,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        per_act_token_quant=False,
        per_out_ch_quant=False,
        block_shape=block_shape,
    )


def nvfp4_moe_quant_config(
    g1_alphas: torch.Tensor,
    g2_alphas: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> FusedMoEQuantConfig:
    """
    Construct a quant config for mxfp4 activations and nvp4 weights.
    """
    return FusedMoEQuantConfig.make(
        "nvfp4",
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        per_act_token_quant=False,
        per_out_ch_quant=False,
        block_shape=None,
    )


def nvfp4_w4a16_moe_quant_config(
    g1_alphas: torch.Tensor,
    g2_alphas: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> FusedMoEQuantConfig:
    """
    Construct a quant config for 16-but activations and nvp4 weights.
    """
    return FusedMoEQuantConfig.make(
        quant_dtype=None,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        weight_dtype="nvfp4",
    )


def int4_w4a16_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_zp: torch.Tensor | None,
    w2_zp: torch.Tensor | None,
    block_shape: list[int] | None = None,
) -> FusedMoEQuantConfig:
    """
    Construct a quant config for 16-bit float activations and int4 weights.
    """
    group_shape = GroupShape(*block_shape) if block_shape is not None else None
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(shape=group_shape),
        _a2=FusedMoEQuantDesc(shape=group_shape),
        _w1=FusedMoEQuantDesc("int4", group_shape, w1_scale, None, w1_zp),
        _w2=FusedMoEQuantDesc("int4", group_shape, w2_scale, None, w2_zp),
    )


def fp8_w8a16_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    block_shape: list[int] | None = None,
) -> FusedMoEQuantConfig:
    """
    Construct a quant config for 16-bit float activations and fp8 weights.
    """
    group_shape = GroupShape(*block_shape) if block_shape is not None else None
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(),
        _a2=FusedMoEQuantDesc(),
        _w1=FusedMoEQuantDesc(
            current_platform.fp8_dtype(), group_shape, w1_scale, None, None
        ),
        _w2=FusedMoEQuantDesc(
            current_platform.fp8_dtype(), group_shape, w2_scale, None, None
        ),
    )


def int8_w8a16_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_zp: torch.Tensor | None,
    w2_zp: torch.Tensor | None,
    block_shape: list[int] | None = None,
) -> FusedMoEQuantConfig:
    """
    Construct a quant config for 16-bit float activations and int8 weights.
    """
    group_shape = GroupShape(*block_shape) if block_shape is not None else None
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(shape=group_shape),
        _a2=FusedMoEQuantDesc(shape=group_shape),
        _w1=FusedMoEQuantDesc(torch.int8, group_shape, w1_scale, None, w1_zp),
        _w2=FusedMoEQuantDesc(torch.int8, group_shape, w2_scale, None, w2_zp),
    )


def int4_w4afp8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    g1_alphas: torch.Tensor,
    g2_alphas: torch.Tensor,
    per_act_token_quant: bool = False,
    per_out_ch_quant: bool = False,
    block_shape: list[int] | None = None,
) -> FusedMoEQuantConfig:
    """
    Construct a quant config for fp8 activations and int4 weights.
    """
    return FusedMoEQuantConfig.make(
        torch.float8_e4m3fn,  # quant dtype for activations
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        per_act_token_quant=per_act_token_quant,
        per_out_ch_quant=per_out_ch_quant,
        block_shape=block_shape,
        weight_dtype="int4",  # weight dtype for weights
    )


def awq_marlin_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_zp: torch.Tensor | None,
    w2_zp: torch.Tensor | None,
    weight_bits: int,
    group_size: int,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> FusedMoEQuantConfig:
    """
    Construct a quant config for awq marlin quantization.
    """
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    w_shape = None if group_size == -1 else GroupShape(row=1, col=group_size)

    # Activations are NOT quantized for AWQ (fp16/bf16)
    a_shape = w_shape  # Same as weight shape for alignment

    # Determine weight dtype
    if weight_bits == 4:
        weight_dtype = "int4"
    elif weight_bits == 8:
        weight_dtype = torch.int8
    else:
        raise ValueError(f"Unsupported weight_bits: {weight_bits}")

    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(dtype=None, shape=a_shape),
        _a2=FusedMoEQuantDesc(dtype=None, shape=a_shape),
        _w1=FusedMoEQuantDesc(weight_dtype, w_shape, w1_scale, None, w1_zp, w1_bias),
        _w2=FusedMoEQuantDesc(weight_dtype, w_shape, w2_scale, None, w2_zp, w2_bias),
    )


def biased_moe_quant_config(
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
) -> FusedMoEQuantConfig:
    """
    Construct a quant config for unquantized activations with biases.
    """
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(),
        _a2=FusedMoEQuantDesc(),
        _w1=FusedMoEQuantDesc(bias=w1_bias),
        _w2=FusedMoEQuantDesc(bias=w2_bias),
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

    @property
    def use_pplx_kernels(self):
        return self.use_all2all_kernels and self.all2all_backend == "pplx"

    @property
    def use_deepep_ht_kernels(self):
        return (
            self.use_all2all_kernels
            and self.all2all_backend == "deepep_high_throughput"
        )

    @property
    def use_deepep_ll_kernels(self):
        return self.use_all2all_kernels and self.all2all_backend == "deepep_low_latency"

    @property
    def use_fi_all2allv_kernels(self):
        return (
            self.use_all2all_kernels and self.all2all_backend == "flashinfer_all2allv"
        )

    @property
    def use_batched_activation_format(self):
        return self.use_deepep_ll_kernels or self.use_pplx_kernels

    @property
    def use_naive_all2all_kernels(self):
        return self.use_all2all_kernels and (
            self.all2all_backend in ["naive", "allgather_reducescatter"]
        )

    @property
    def use_mori_kernels(self):
        return self.use_all2all_kernels and self.all2all_backend == "mori"

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
        """
        Determine MoE parallel configuration. Based on the input `tp_size_`,
        `dp_size_` and vllm's parallel config, determine what
        level's of parallelism to use in the fused moe layer.

        Args:
            tp_size_ (int): `tp_size` passed into the FusedMoE constructor.
            pcp_size_ (int): `pcp_size` passed into the FusedMoE constructor.
            dp_size_ (int): `dp_size` passed into the FusedMoE constructor.
            vllm_parallel_config (ParallelConfig): vLLM's parallel config
                object which contains the `enable_expert_parallel` flag.

        Examples:
            When there is no parallelism requested,
            i.e. `tp_size_` = `pcp_size_` = `dp_size_` = 1, we simply return the sizes
            unaltered and the ranks set to 0.

            Expert Parallelism is considered only when either `dp_size_`, `pcp_size_` or
            `tp_size_` is non trivial.

            Note that PCP serves the same function as DP here.

            When TP = 2, DP(PCP) = 1 and EP = False, the configuration on different
            devices:

            - device 0 : TP = {2, 0} DP = {1, 0} EP = {1, 0} //
                legend : {size, rank}
            - device 1 : TP = {2, 1} DP = {1, 0} EP = {1, 0}
            - Comment : Tensors are sharded across 2 devices.

            When TP = 1, DP(PCP) = 2 and EP = False, the configuration on different
                devices:

            - device 0 : TP = {2, 0} DP = {2, 0} EP = {1, 0}
            - device 1 : TP = {2, 1} DP = {2, 1} EP = {1, 0}
            - Comment: There are 2 engine instances and the tensors are sharded
                across 2 decvices.

            When TP = 2, DP(PCP) = 2 and EP = False, the configuration on different
                devices:

            - device 0: TP = {4, 0} DP = {2, 0} EP = {1, 0}
            - device 1: TP = {4, 1} DP = {2, 0} EP = {1, 0}
            - device 2: TP = {4, 2} DP = {2, 1} EP = {1, 0}
            - device 3: TP = {4, 3} DP = {2, 1} EP = {1, 0}
            - Comment: There are 2 engine instances and the tensors are sharded
                across 4 devices.

            When, TP = 2, DP(PCP) = 1 and EP = True, the configuration on different
                devices:

            - device 0: TP = {1, 0} DP = {1, 0} EP = {2, 0}
            - device 1: TP = {1, 0} DP = {1, 0} EP = {2, 1}
            - Comment: The experts are split between the 2 devices.

            When, TP = 1, DP(PCP) = 2 and EP = True, the configuration on different
                devices:

            - device 0: TP = {1, 0} DP = {2, 0} EP = {2, 0}
            - device 1: TP = {1, 0} DP = {2, 1} EP = {2, 1}
            - Comment: There are 2 engine instances and the experts are split
                between the 2 devices.

            When TP = 2, DP(PCP) = 2 and EP = True, the configuration on different
                devices:

            - device 0: TP = {1, 0} DP = {2, 0} EP = {4, 0}
            - device 1: TP = {1, 0} DP = {2, 0} EP = {4, 1}
            - device 2: TP = {1, 0} DP = {2, 1} EP = {4, 2}
            - device 3: TP = {1, 0} DP = {2, 1} EP = {4, 3}
            - Comment: There are 2 engine instances and the experts are split
                between the 4 devices.
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
            all2all_backend="naive",
            enable_eplb=False,
        )


# Adapted from pplx-kernels tests/all_to_all_utils.py
@dataclass
class FusedMoEConfig:
    num_experts: int
    experts_per_token: int
    hidden_dim: int
    intermediate_size_per_partition: int
    num_local_experts: int
    num_logical_experts: int
    activation: str
    device: torch.device | str
    routing_method: RoutingMethodType
    moe_parallel_config: FusedMoEParallelConfig

    # The activation type.
    in_dtype: torch.dtype

    # Defaults to in_dtype if not specified.
    router_logits_dtype: torch.dtype | None = None

    max_num_tokens: int = envs.VLLM_MOE_DP_CHUNK_SIZE
    has_bias: bool = False
    is_act_and_mul: bool = True
    is_lora_enabled: bool = False

    # This flag is used to disable the inplace optimization
    # in MoE kernels. If this flag is True then the kernel
    # should not be using inplace. If the flag is false, the
    # kernel is free to use inplace or not.
    disable_inplace: bool = True

    def __post_init__(self):
        if self.dp_size > 1:
            logger.debug_once(
                "Using FusedMoEConfig::max_num_tokens=%d", self.max_num_tokens
            )

        assert self.max_num_tokens > 0

        if self.router_logits_dtype is None:
            self.router_logits_dtype = self.in_dtype

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

    @property
    def use_pplx_kernels(self):
        return self.moe_parallel_config.use_pplx_kernels

    @property
    def use_deepep_ht_kernels(self):
        return self.moe_parallel_config.use_deepep_ht_kernels

    @property
    def use_deepep_ll_kernels(self):
        return self.moe_parallel_config.use_deepep_ll_kernels

    @property
    def use_mori_kernels(self):
        return self.moe_parallel_config.use_mori_kernels

    @property
    def use_fi_all2allv_kernels(self):
        return self.moe_parallel_config.use_fi_all2allv_kernels

    @property
    def use_naive_all2all_kernels(self):
        return self.moe_parallel_config.use_naive_all2all_kernels
