# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch

import vllm.envs as envs
from vllm.config import ParallelConfig
from vllm.distributed import get_dp_group, get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape)
from vllm.utils import cdiv, has_triton_kernels
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe

if TYPE_CHECKING and has_triton_kernels:
    from triton_kernels.matmul_ogs import PrecisionConfig

logger = init_logger(__name__)


def _get_config_dtype_str(
    dtype: torch.dtype,
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    use_mxfp4_w4a4: bool = False,
) -> Optional[str]:
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif use_int4_w4a16:
        return "int4_w4a16"
    elif use_mxfp4_w4a4:
        return "mxfp4_w4a4"
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None


def _get_config_quant_dtype(
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    use_mxfp4_w4a4: bool,
) -> Union[None, torch.dtype, str]:
    if use_fp8_w8a8:
        return torch.float8_e4m3fn
    elif use_int8_w8a8:
        return torch.int8
    elif use_mxfp4_w4a4:
        return "mxfp4"
    return None


def _quant_flags_to_group_shape(
    quant_dtype: Union[torch.dtype, str, None],
    per_act_token_quant: bool,
    per_out_ch_quant: bool,
    block_shape: Optional[list[int]],
) -> tuple[Optional[GroupShape], Optional[GroupShape]]:
    a_shape: Optional[GroupShape]
    w_shape: Optional[GroupShape]
    if block_shape is not None:
        assert not per_act_token_quant
        assert not per_out_ch_quant
        # This is not quite right since first dim should be 1.
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


@dataclass
class FusedMoEQuantDesc:
    # TODO (bnell): use scalar_type instead of Union.
    dtype: Union[torch.dtype, str, None] = None
    #  * (-1, -1)   for per-tensor quantization
    #  * (1, -1)    for per-row quantization
    #  * (-1, 1)    for per-column quantization
    #  * (128, 128) for 128x128 deepseek style block quantization
    #  * (1, 128)   for deepseek style activation quantization
    #               (i.e. per-token-per-group)
    shape: Optional[GroupShape] = None

    # TODO: maybe put PrecisionConfigs in subclass of QuantDesc?
    scale: Union[torch.Tensor, "PrecisionConfig", None] = None

    # TODO: put some of these in subclasses
    alpha_or_gscale: Optional[torch.Tensor] = None  # store as 1/gs or gs?
    zp: Optional[torch.Tensor] = None
    bias: Optional[torch.Tensor] = None


# TODO: have subclasses for specific moe methods?
# e.g. for specific arguments bias, precision, etc.
@dataclass
class FusedMoEQuantConfig:
    # TODO(bnell) make sure a1_scales/a2_scales don't interfere with chunking
    _a1: FusedMoEQuantDesc
    _a2: FusedMoEQuantDesc

    # Note: weights are not required to have a GroupShape since
    # they've already been quantized.
    _w1: FusedMoEQuantDesc
    _w2: FusedMoEQuantDesc

    def __post_init__(self):
        assert (not self.per_act_token_quant
                or self.block_shape is None), "illegal quantization"

    # TODO: add rest
    # - w1_scale (Optional[torch.Tensor]): Optional scale to be used for w1.
    # - w2_scale (Optional[torch.Tensor]): Optional scale to be used for w2.
    # - w1_zp (Optional[torch.Tensor]): Optional zero points to be used for
    #   w1.
    # - w2_zp (Optional[torch.Tensor]): Optional zero points to be used for
    #   w2.
    # - a1_scale (Optional[torch.Tensor]): Optional scale to be used for a1.
    # - a2_scale (Optional[torch.Tensor]): Optional scale to be used for a2.

    @property
    def quant_dtype(self) -> Union[torch.dtype, str, None]:
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
    def block_shape(self) -> Optional[list[int]]:
        if (self._a1.shape is not None
                and self._a1.shape != GroupShape.PER_TENSOR
                and self._a1.shape != GroupShape.PER_TOKEN):
            return [self._a1.shape.row, self._a1.shape.col]
        else:
            return None

    @property
    def is_block_quantized(self) -> bool:
        return self.block_shape is not None

    @property
    def a1_scale(self) -> Optional[torch.Tensor]:
        assert self._a1.scale is None or isinstance(self._a1.scale,
                                                    torch.Tensor)
        return self._a1.scale

    @property
    def a1_gscale(self) -> Optional[torch.Tensor]:
        return self._a1.alpha_or_gscale

    @property
    def a2_scale(self) -> Optional[torch.Tensor]:
        assert self._a2.scale is None or isinstance(self._a2.scale,
                                                    torch.Tensor)
        return self._a2.scale

    @property
    def a2_gscale(self) -> Optional[torch.Tensor]:
        return self._a2.alpha_or_gscale

    @property
    def w1_scale(self) -> Optional[torch.Tensor]:
        assert self._w1.scale is None or isinstance(self._w1.scale,
                                                    torch.Tensor)
        return self._w1.scale

    @property
    def w1_zp(self) -> Optional[torch.Tensor]:
        return self._w1.zp

    @property
    def w1_bias(self) -> Optional[torch.Tensor]:
        return self._w1.bias

    @property
    def w1_precision(self) -> Optional["PrecisionConfig"]:
        assert self._w1.scale is None or isinstance(self._w1.scale,
                                                    PrecisionConfig)
        return self._w1.scale

    @property
    def g1_alphas(self) -> Optional[torch.Tensor]:
        return self._w1.alpha_or_gscale

    @property
    def w2_scale(self) -> Optional[torch.Tensor]:
        assert self._w2.scale is None or isinstance(self._w2.scale,
                                                    torch.Tensor)
        return self._w2.scale

    @property
    def w2_zp(self) -> Optional[torch.Tensor]:
        return self._w2.zp

    @property
    def w2_bias(self) -> Optional[torch.Tensor]:
        return self._w2.bias

    @property
    def w2_precision(self) -> Optional["PrecisionConfig"]:
        assert self._w2.scale is None or isinstance(self._w2.scale,
                                                    PrecisionConfig)
        return self._w2.scale

    @property
    def g2_alphas(self) -> Optional[torch.Tensor]:
        return self._w2.alpha_or_gscale

    @property
    def use_fp8_w8a8(self) -> bool:
        return self.quant_dtype == torch.float8_e4m3fn

    @property
    def use_int8_w8a8(self) -> bool:
        return self.quant_dtype == torch.int8

    @property
    def use_int8_w8a16(self) -> bool:
        return (self._a1.dtype is None and self._w1.dtype == torch.int8)

    @property
    def use_int4_w4a16(self) -> bool:
        return (self._a1.dtype is None and self._w1.dtype == "int4")

    @property
    def use_mxfp4_w4a4(self) -> bool:
        return self.quant_dtype == "mxfp4"

    @property
    def use_nvfp4_w4a4(self) -> bool:
        return self.quant_dtype == "nvfp4"

    def config_name(self, dtype: torch.dtype) -> Optional[str]:
        return _get_config_dtype_str(
            use_fp8_w8a8=self.use_fp8_w8a8,
            use_int8_w8a16=self.use_int8_w8a16,
            use_int4_w4a16=self.use_int4_w4a16,
            use_mxfp4_w4a4=self.use_mxfp4_w4a4,
            dtype=dtype,
        )

    def scale_shape(
        self,
        max_tokens: int,
        hidden_dim: int,
    ) -> Optional[tuple[int, int]]:
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
    ) -> Optional[tuple[int, int, int]]:
        if self.is_quantized:
            scale_shape = self.scale_shape(max_tokens, hidden_dim)
            assert scale_shape is not None
            return (num_experts, *scale_shape)
        else:
            return None

    @staticmethod
    def make(
        quant_dtype: Union[torch.dtype, str, None] = None,
        per_act_token_quant: bool = False,
        per_out_ch_quant: bool = False,
        block_shape: Optional[list[int]] = None,
        w1_scale: Union[torch.Tensor, "PrecisionConfig", None] = None,
        w2_scale: Union[torch.Tensor, "PrecisionConfig", None] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        g1_alphas: Optional[torch.Tensor] = None,
        g2_alphas: Optional[torch.Tensor] = None,
        a1_gscale: Optional[torch.Tensor] = None,
        a2_gscale: Optional[torch.Tensor] = None,
        w1_bias: Optional[torch.Tensor] = None,
        w2_bias: Optional[torch.Tensor] = None,
    ) -> "FusedMoEQuantConfig":
        a_shape, w_shape = _quant_flags_to_group_shape(quant_dtype,
                                                       per_act_token_quant,
                                                       per_out_ch_quant,
                                                       block_shape)
        quant_config = FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(quant_dtype, a_shape, a1_scale, a1_gscale),
            _a2=FusedMoEQuantDesc(quant_dtype, a_shape, a2_scale, a2_gscale),
            _w1=FusedMoEQuantDesc(quant_dtype, w_shape, w1_scale, g1_alphas,
                                  None, w1_bias),
            _w2=FusedMoEQuantDesc(quant_dtype, w_shape, w2_scale, g2_alphas,
                                  None, w2_bias),
        )
        assert quant_config.per_act_token_quant == per_act_token_quant
        assert quant_config.per_out_ch_quant == per_out_ch_quant
        assert quant_config.block_shape == block_shape
        return quant_config


# TODO better doc


def fp8_w8a8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    per_act_token_quant: bool = False,
    per_out_ch_quant: bool = False,
    block_shape: Optional[list[int]] = None,
) -> FusedMoEQuantConfig:
    a_shape, w_shape = _quant_flags_to_group_shape(torch.float8_e4m3fn,
                                                   per_act_token_quant,
                                                   per_out_ch_quant,
                                                   block_shape)
    quant_config = FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(torch.float8_e4m3fn, a_shape, a1_scale),
        _a2=FusedMoEQuantDesc(torch.float8_e4m3fn, a_shape, a2_scale),
        _w1=FusedMoEQuantDesc(torch.float8_e4m3fn, w_shape, w1_scale),
        _w2=FusedMoEQuantDesc(torch.float8_e4m3fn, w_shape, w2_scale),
    )
    assert quant_config.quant_dtype == torch.float8_e4m3fn
    assert quant_config.per_act_token_quant == per_act_token_quant
    assert quant_config.per_out_ch_quant == per_out_ch_quant
    assert quant_config.block_shape == block_shape
    return quant_config


def int8_w8a8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: Optional[torch.Tensor],
    a2_scale: Optional[torch.Tensor],
    per_act_token_quant: bool = False,
) -> FusedMoEQuantConfig:
    a_shape, w_shape = _quant_flags_to_group_shape(torch.int8,
                                                   per_act_token_quant, False,
                                                   None)
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(torch.int8, a_shape, a1_scale),
        _a2=FusedMoEQuantDesc(torch.int8, a_shape, a2_scale),
        _w1=FusedMoEQuantDesc(torch.int8, w_shape, w1_scale),
        _w2=FusedMoEQuantDesc(torch.int8, w_shape, w2_scale),
    )


def mxfp4_w4a4_moe_quant_config(
    w1_scale: Union[torch.Tensor, "PrecisionConfig"],
    w2_scale: Union[torch.Tensor, "PrecisionConfig"],
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
) -> FusedMoEQuantConfig:
    a_shape, w_shape = _quant_flags_to_group_shape(
        "mxfp4",
        False,  #?
        False,  #?
        block_shape)
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc("mxfp4", a_shape, a1_scale),
        _a2=FusedMoEQuantDesc("mxfp4", a_shape, a2_scale),
        _w1=FusedMoEQuantDesc("mxfp4", w_shape, w1_scale, None, w1_bias),
        _w2=FusedMoEQuantDesc("mxfp4", w_shape, w2_scale, None, w2_bias),
    )


def nvfp4_moe_quant_config(
    g1_alphas: torch.Tensor,
    g2_alphas: torch.Tensor,
    a1_gscale: torch.Tensor,  # a1_scale?
    a2_gscale: torch.Tensor,  # a2_scale?
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> FusedMoEQuantConfig:
    a_shape, w_shape = _quant_flags_to_group_shape("nvfp4", False, False, None)
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc("nvfp4", a_shape, None, a1_gscale),
        _a2=FusedMoEQuantDesc("nvfp4", a_shape, None, a2_gscale),
        _w1=FusedMoEQuantDesc("nvfp4", w_shape, w1_scale, g1_alphas),
        _w2=FusedMoEQuantDesc("nvfp4", w_shape, w2_scale, g2_alphas),
    )


def int4_w4a16_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_zp: Optional[torch.Tensor],
    w2_zp: Optional[torch.Tensor],
    block_shape: Optional[list[int]] = None,
) -> FusedMoEQuantConfig:
    # Activations are pre-quantized
    group_shape = GroupShape(*block_shape) if block_shape is not None else None
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(shape=group_shape),
        _a2=FusedMoEQuantDesc(shape=group_shape),
        _w1=FusedMoEQuantDesc("int4", group_shape, w1_scale, None, w1_zp),
        _w2=FusedMoEQuantDesc("int4", group_shape, w2_scale, None, w2_zp),
    )


def int8_w8a16_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_zp: Optional[torch.Tensor],
    w2_zp: Optional[torch.Tensor],
    block_shape: Optional[list[int]] = None,
) -> FusedMoEQuantConfig:
    # Activations are pre-quantized
    group_shape = GroupShape(*block_shape) if block_shape is not None else None
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(shape=group_shape),
        _a2=FusedMoEQuantDesc(shape=group_shape),
        _w1=FusedMoEQuantDesc(torch.int8, group_shape, w1_scale, None, w1_zp),
        _w2=FusedMoEQuantDesc(torch.int8, group_shape, w2_scale, None, w2_zp),
    )


def biased_moe_quant_config(
    w1_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
) -> FusedMoEQuantConfig:
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(),
        _a2=FusedMoEQuantDesc(),
        _w1=FusedMoEQuantDesc(bias=w1_bias),
        _w2=FusedMoEQuantDesc(bias=w2_bias),
    )


FUSED_MOE_UNQUANTIZED_CONFIG: FusedMoEQuantConfig = FusedMoEQuantConfig.make()


@dataclass
class FusedMoEParallelConfig:
    tp_size: int
    dp_size: int
    ep_size: int
    tp_rank: int
    dp_rank: int
    ep_rank: int

    use_ep: bool  # whether to use EP or not

    @property
    def use_all2all_kernels(self):
        return self.dp_size > 1 and self.use_ep

    @property
    def use_pplx_kernels(self):
        return (self.use_all2all_kernels
                and envs.VLLM_ALL2ALL_BACKEND == "pplx")

    @property
    def use_deepep_ht_kernels(self):
        return (self.use_all2all_kernels
                and envs.VLLM_ALL2ALL_BACKEND == "deepep_high_throughput")

    @property
    def use_deepep_ll_kernels(self):
        return (self.use_all2all_kernels
                and envs.VLLM_ALL2ALL_BACKEND == "deepep_low_latency")

    @staticmethod
    def make(tp_size_: int, dp_size_: int,
             vllm_parallel_config: ParallelConfig) -> "FusedMoEParallelConfig":
        """
        Determine MoE parallel configuration. Based on the input `tp_size_`,
        `dp_size_` and vllm's parallel config, determine what
        level's of parallelism to use in the fused moe layer.

        Args:
            tp_size_ (int): `tp_size` passed into the FusedMoE constructor.
            dp_size_ (int): `dp_size` passed into the FusedMoE constructor.
            vllm_parallel_config (ParallelConfig): vLLM's parallel config
                object which contains the `enable_expert_parallel` flag.

        Examples:
            When there is no parallelism requested,
            i.e. `tp_size_` = `dp_size_` = 1, we simply return the sizes
            unaltered and the ranks set to 0.

            Expert Parallelism is considered only when either `dp_size_` or
            `tp_size_` is non trivial.

            When TP = 2, DP = 1 and EP = False, the configuration on different
            devices:

            - device 0 : TP = {2, 0} DP = {1, 0} EP = {1, 0} //
                legend : {size, rank}
            - device 1 : TP = {2, 1} DP = {1, 0} EP = {1, 0}
            - Comment : Tensors are sharded across 2 devices.

            When TP = 1, DP = 2 and EP = False, the configuration on different
                devices:

            - device 0 : TP = {2, 0} DP = {2, 0} EP = {1, 0}
            - device 1 : TP = {2, 1} DP = {2, 1} EP = {1, 0}
            - Comment: There are 2 engine instances and the tensors are sharded
                across 2 decvices.

            When TP = 2, DP = 2 and EP = False, the configuration on different
                devices:

            - device 0: TP = {4, 0} DP = {2, 0} EP = {1, 0}
            - device 1: TP = {4, 1} DP = {2, 0} EP = {1, 0}
            - device 2: TP = {4, 2} DP = {2, 1} EP = {1, 0}
            - device 3: TP = {4, 3} DP = {2, 1} EP = {1, 0}
            - Comment: There are 2 engine instances and the tensors are sharded
                across 4 devices.

            When, TP = 2, DP = 1 and EP = True, the configuration on different
                devices:

            - device 0: TP = {1, 0} DP = {1, 0} EP = {2, 0}
            - device 1: TP = {1, 0} DP = {1, 0} EP = {2, 1}
            - Comment: The experts are split between the 2 devices.

            When, TP = 1, DP = 2 and EP = True, the configuration on different
                devices:

            - device 0: TP = {1, 0} DP = {2, 0} EP = {2, 0}
            - device 1: TP = {1, 0} DP = {2, 1} EP = {2, 1}
            - Comment: There are 2 engine instances and the experts are split
                between the 2 devices.

            When TP = 2, DP = 2 and EP = True, the configuration on different
                devices:

            - device 0: TP = {1, 0} DP = {2, 0} EP = {4, 0}
            - device 1: TP = {1, 0} DP = {2, 0} EP = {4, 1}
            - device 2: TP = {1, 0} DP = {2, 1} EP = {4, 2}
            - device 3: TP = {1, 0} DP = {2, 1} EP = {4, 3}
            - Comment: There are 2 engine instances and the experts are split
                between the 4 devices.
        """

        def flatten_tp_across_dp(dp_rank: int):
            tp_rank = 0 if tp_size_ == 1 else get_tensor_model_parallel_rank()
            # There are actually dp_size_ * tp_size_ devices. Update tp_size
            # and tp_rank so we shard across all devices.
            tp_size = dp_size_ * tp_size_
            tp_rank = dp_rank * tp_size_ + tp_rank
            return tp_size, tp_rank

        use_ep = (dp_size_ * tp_size_ > 1
                  and vllm_parallel_config.enable_expert_parallel)

        dp_size = dp_size_
        dp_rank = get_dp_group().rank_in_group if dp_size > 1 else 0
        tp_size, tp_rank = flatten_tp_across_dp(dp_rank)

        if not use_ep:
            return FusedMoEParallelConfig(tp_size=tp_size,
                                          tp_rank=tp_rank,
                                          dp_size=dp_size,
                                          dp_rank=dp_rank,
                                          ep_size=1,
                                          ep_rank=0,
                                          use_ep=False)
        # DP + EP / TP + EP / DP + TP + EP
        assert use_ep
        # In EP, each device owns a set of experts fully. There is no tensor
        # parallel update tp_size, tp_rank, ep_size and ep_rank to reflect that.
        ep_size = tp_size
        ep_rank = tp_rank
        return FusedMoEParallelConfig(tp_size=1,
                                      tp_rank=0,
                                      dp_size=dp_size,
                                      dp_rank=dp_rank,
                                      ep_size=ep_size,
                                      ep_rank=ep_rank,
                                      use_ep=True)


# Adapted from pplx-kernels tests/all_to_all_utils.py
@dataclass
class FusedMoEConfig:
    num_experts: int
    experts_per_token: int
    hidden_dim: int

    num_local_experts: int
    moe_parallel_config: FusedMoEParallelConfig

    # The activation type.
    in_dtype: torch.dtype

    max_num_tokens: int = envs.VLLM_MOE_DP_CHUNK_SIZE

    has_bias: bool = False

    def __post_init__(self):
        if self.dp_size > 1:
            logger.debug_once("Using FusedMoEConfig::max_num_tokens=%d",
                              self.max_num_tokens)

        assert self.max_num_tokens > 0

    @property
    def tp_size(self):
        return self.moe_parallel_config.tp_size

    @property
    def dp_size(self):
        return self.moe_parallel_config.dp_size

    @property
    def ep_size(self):
        return self.moe_parallel_config.ep_size

    @property
    def tp_rank(self):
        return self.moe_parallel_config.tp_rank

    @property
    def dp_rank(self):
        return self.moe_parallel_config.dp_rank

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
    def use_flashinfer_cutlass_kernels(self):
        """
        Whether to use FlashInfer cutlass kernels for NVFP4 MoE.
        """
        assert False, "TBD quant check"
        # (self.quant_config is not None
        #  and self.quant_config.quant_dtype == "nvfp4"
        return (envs.VLLM_USE_FLASHINFER_MOE_FP4
                and has_flashinfer_cutlass_fused_moe()
                and envs.VLLM_FLASHINFER_MOE_BACKEND == "throughput")

