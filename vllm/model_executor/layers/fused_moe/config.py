# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional, Union

import torch
from compressed_tensors.quantization import (QuantizationArgs,
                                             QuantizationStrategy,
                                             QuantizationType)

import vllm.envs as envs
from vllm.config import ParallelConfig
from vllm.distributed import get_dp_group, get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.utils import cdiv
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe

logger = init_logger(__name__)


def _get_quant_config_quantization_args(
    quant_config: Optional[QuantizationConfig],
    prop_name: str,
) -> Optional[QuantizationArgs]:
    if (quant_config is not None and hasattr(quant_config, 'target_scheme_map')
            and "Linear" in quant_config.target_scheme_map and
            "input_activations" in quant_config.target_scheme_map["Linear"]):
        return quant_config.target_scheme_map["Linear"].get(prop_name)
    else:
        return None


def get_quant_config_input_quant(
        quant_config: Optional[QuantizationConfig]
) -> Optional[QuantizationArgs]:
    return _get_quant_config_quantization_args(quant_config,
                                               "input_activations")


def get_quant_config_weight_quant(
        quant_config: Optional[QuantizationConfig]
) -> Optional[QuantizationArgs]:
    return _get_quant_config_quantization_args(quant_config, "weights")


# TODO (bnell): use scalar_type instead of bools?
def get_config_quant_dtype(
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


@dataclass
class FusedMoEQuantConfig:
    # The post quantization activation type.
    quant_dtype: Optional[torch.dtype] = None
    per_act_token_quant: bool = False
    per_out_ch_quant: bool = False
    block_shape: Optional[list[int]] = None

    # TODO: add col major flag?
    # add detailed quant info for input, intermediates, weights, etc?

    def __post_init__(self):
        assert (not self.per_act_token_quant
                or self.block_shape is None), "illegal quantization"

    @property
    def is_quantized(self) -> bool:
        return self.quant_dtype is not None

    @property
    def is_per_act_token(self) -> bool:
        return self.per_act_token_quant

    @property
    def is_block_quantized(self) -> bool:
        return self.block_shape is not None

    @property
    def is_per_tensor(self) -> bool:
        return not self.per_act_token_quant and self.block_shape is None

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
        use_fp8_w8a8: bool = False,
        use_int8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        use_int4_w4a16: bool = False,
        use_mxfp4_w4a4: bool = False,
        per_act_token_quant: bool = False,
        per_out_ch_quant: bool = False,
        block_shape: Optional[list[int]] = None,
    ) -> "FusedMoEQuantConfig":
        assert sum([
            int(flag) for flag in [
                use_fp8_w8a8,
                use_int8_w8a8,
                use_int8_w8a16,
                use_int4_w4a16,
            ]
        ]) <= 1, "Quantization flags are mutually exclusive."

        quant_dtype = get_config_quant_dtype(
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            use_mxfp4_w4a4=use_mxfp4_w4a4,
        )
        return FusedMoEQuantConfig(
            quant_dtype,
            per_act_token_quant,
            per_out_ch_quant,
            block_shape,
        )


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

    @property
    def use_flashinfer_cutlass_kernels(self):
        return (envs.VLLM_USE_FLASHINFER_MOE_FP4
                and has_flashinfer_cutlass_fused_moe()
                and envs.VLLM_FLASHINFER_MOE_BACKEND == "throughput")

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

    quant_config: Optional[FusedMoEQuantConfig] = None

    max_num_tokens: int = envs.VLLM_MOE_DP_CHUNK_SIZE

    has_bias: bool = False

    def __post_init__(self):
        if self.dp_size > 1:
            logger.debug_once("Using FusedMoEConfig::max_num_tokens=%d",
                              self.max_num_tokens)

        assert self.max_num_tokens > 0

    @property
    def quant_dtype(self) -> Optional[torch.dtype]:
        if self.quant_config is not None:
            return self.quant_config.quant_dtype
        else:
            return None

    @property
    def block_shape(self) -> Optional[list[int]]:
        if self.quant_config is not None:
            return self.quant_config.block_shape
        else:
            return None

    @property
    def per_act_token_quant(self) -> bool:
        if self.quant_config is not None:
            return self.quant_config.per_act_token_quant
        else:
            return False

    @property
    def per_out_ch_quant(self) -> bool:
        if self.quant_config is not None:
            return self.quant_config.per_out_ch_quant
        else:
            return False

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
        return self.moe_parallel_config.use_flashinfer_cutlass_kernels

    @staticmethod
    def make(
        num_experts: int,
        experts_per_token: int,
        hidden_dim: int,
        num_local_experts: int,
        moe_parallel_config: FusedMoEParallelConfig,
        in_dtype: torch.dtype,
        max_num_tokens: int = envs.VLLM_MOE_DP_CHUNK_SIZE,
        quant_config: Optional[Union[FusedMoEQuantConfig,
                                     QuantizationConfig]] = None,
        has_bias: bool = False,
    ) -> "FusedMoEConfig":

        _quant_config: Optional[FusedMoEQuantConfig] = None

        if quant_config is not None and isinstance(quant_config,
                                                   QuantizationConfig):
            if hasattr(quant_config, 'weight_block_size'):
                block_shape = quant_config.weight_block_size
            else:
                block_shape = None
            per_act_token_quant = False
            per_out_ch_quant = False
            quant_dtype: Optional[torch.dtype] = None

            input_quant = get_quant_config_input_quant(quant_config)
            weight_quant = get_quant_config_weight_quant(quant_config)

            if input_quant is not None:
                per_act_token_quant = (input_quant.strategy
                                       == QuantizationStrategy.TOKEN
                                       if input_quant is not None else False)

                if input_quant.num_bits == 8:
                    if input_quant.type == QuantizationType.FLOAT:
                        quant_dtype = torch.float8_e4m3fn
                    elif input_quant.type == QuantizationType.INT:
                        quant_dtype = torch.int8

            from vllm.model_executor.layers.quantization.fp8 import Fp8Config
            if quant_dtype is None and isinstance(quant_config, Fp8Config):
                quant_dtype = torch.float8_e4m3fn

            from vllm.model_executor.layers.quantization.modelopt import (
                ModelOptNvFp4Config)
            if quant_dtype is None and isinstance(quant_config,
                                                  ModelOptNvFp4Config):
                quant_dtype = torch.uint8

            if weight_quant is not None:
                per_out_ch_quant = (
                    weight_quant.strategy == QuantizationStrategy.CHANNEL)

            if quant_dtype is not None:
                _quant_config = FusedMoEQuantConfig(
                    quant_dtype=quant_dtype,
                    per_act_token_quant=per_act_token_quant,
                    per_out_ch_quant=per_out_ch_quant,
                    block_shape=block_shape,
                )
            else:
                _quant_config = FusedMoEQuantConfig()
                if moe_parallel_config.dp_size > 1:
                    logger.warning_once("MoE DP setup unable to determine "
                                        "quantization scheme or unsupported "
                                        "quantization type. This model will "
                                        "not run with DP enabled.")
        else:
            _quant_config = quant_config

        return FusedMoEConfig(
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            hidden_dim=hidden_dim,
            num_local_experts=num_local_experts,
            moe_parallel_config=moe_parallel_config,
            in_dtype=in_dtype,
            quant_config=_quant_config,
            max_num_tokens=max_num_tokens,
            has_bias=has_bias,
        )
