# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.quantization import (ActivationOrdering,
                                             QuantizationStrategy)

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (FusedMoEConfig,
                                                  FusedMoEMethodBase)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import (  # noqa
    CompressedTensorsW4A4Fp4MoeMethod)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (  # noqa
    CompressedTensorsW8A8Fp8MoEMethod)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (  # noqa
    CompressedTensorsW8A8Int8MoEMethod)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa
    CompressedTensorsWNA16MarlinMoEMethod, CompressedTensorsWNA16MoEMethod)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_moe_marlin_supports_layer)

logger = init_logger(__name__)

__all__ = [
    "CompressedTensorsMoEMethod", "CompressedTensorsW8A8Fp8MoEMethod",
    "CompressedTensorsW8A8Int8MoEMethod",
    "CompressedTensorsWNA16MarlinMoEMethod", "CompressedTensorsWNA16MoEMethod",
    "CompressedTensorsW4A4Fp4MoeMethod"
]


class CompressedTensorsMoEMethod(FusedMoEMethodBase):

    def __init_(self, moe: FusedMoEConfig):
        super().__init__(moe)

    @staticmethod
    def get_moe_method(
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module,
    ) -> "CompressedTensorsMoEMethod":
        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        weight_quant = quant_config.target_scheme_map["Linear"].get("weights")
        input_quant = quant_config.target_scheme_map["Linear"].get(
            "input_activations")

        if quant_config._is_wNa16_group_channel(weight_quant, input_quant):
            # group_size=None means channelwise
            group_size = weight_quant.group_size or -1
            # Prefer to use the MarlinMoE kernel when it is supported.
            if not check_moe_marlin_supports_layer(layer, group_size):
                if (weight_quant.strategy in QuantizationStrategy.GROUP and
                        weight_quant.actorder in (ActivationOrdering.GROUP,
                                                  ActivationOrdering.DYNAMIC)):
                    raise ValueError(
                        "WNA16MoE is not supported with actorder=group/dynamic."
                    )
                logger.info_once("Using CompressedTensorsWNA16MoEMethod")
                return CompressedTensorsWNA16MoEMethod(quant_config,
                                                       layer.moe_config)
            else:
                logger.info_once("Using CompressedTensorsWNA16MarlinMoEMethod")
                return CompressedTensorsWNA16MarlinMoEMethod(
                    quant_config, layer.moe_config)
        elif quant_config._is_fp4a4_nvfp4(weight_quant, input_quant):
            logger.info_once("Using CompressedTensorsW4A4Fp4MoeMethod")
            return CompressedTensorsW4A4Fp4MoeMethod(layer.moe_config, layer)
        elif (quant_config._is_fp8_w8a8_sm90(weight_quant, input_quant)
              or quant_config._is_fp8_w8a8_sm100(weight_quant, input_quant)
              or quant_config._is_fp8_w8a8(weight_quant, input_quant)):
            logger.info_once("Using CompressedTensorsW8A8Fp8MoEMethod")
            return CompressedTensorsW8A8Fp8MoEMethod(quant_config,
                                                     layer.moe_config)
        elif quant_config._is_dynamic_token_w8a8(weight_quant, input_quant):
            logger.info_once("Using CompressedTensorsW8A8Int8MoEMethod")
            return CompressedTensorsW8A8Int8MoEMethod(quant_config,
                                                      layer.moe_config)
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}")
