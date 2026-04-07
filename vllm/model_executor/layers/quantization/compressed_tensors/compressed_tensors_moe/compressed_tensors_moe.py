# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationStrategy,
)

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa
    WNA16_SUPPORTED_BITS,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_moe_marlin_supports_layer,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


class CompressedTensorsMoEMethod(FusedMoEMethodBase):
    @staticmethod
    def get_moe_method(
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module,
        layer_name: str,
    ) -> FusedMoEMethodBase:
        # FusedMoE was made by combining multiple Linears so need to
        # make sure quantization config for Linear can target it
        quant_config._add_fused_moe_to_target_scheme_map()
        unfused_names = [
            layer_name + proj_name
            for proj_name in [".0.gate_proj", ".0.up_proj", ".0.down_proj"]
        ]
        # TODO: refactor this to use expert_mapping and check all layer numbers
        all_scheme_dicts = [
            quant_config.get_scheme_dict(layer, name) for name in unfused_names
        ]
        scheme_dict = all_scheme_dicts.pop()

        # multiple schemes found
        if not all([cur_dict == scheme_dict for cur_dict in all_scheme_dicts]):
            raise ValueError(
                "All MoE projections need to have same "
                "quantization scheme but found multiple"
            )

        if scheme_dict is None:  # ignored layer
            return UnquantizedFusedMoEMethod(layer.moe_config)

        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        weight_quant = scheme_dict.get("weights")
        input_quant = scheme_dict.get("input_activations")
        format = scheme_dict.get("format")

        if quant_config._is_mxfp4(weight_quant):
            from .compressed_tensors_moe_w4a4_mxfp4 import (
                CompressedTensorsW4A4Mxfp4MoEMethod,
            )

            return CompressedTensorsW4A4Mxfp4MoEMethod(layer.moe_config)

        if quant_config._is_wNa16_group_channel(weight_quant, input_quant):
            # group_size=None means channelwise
            group_size = weight_quant.group_size or -1

            valid_format_and_bits = (
                weight_quant.num_bits in WNA16_SUPPORTED_BITS
                and format == CompressionFormat.pack_quantized.value
            )

            if not valid_format_and_bits:
                raise ValueError(
                    "For Fused MoE layers, only format: ",
                    f"{CompressionFormat.pack_quantized.value} ",
                    f" and bits: {WNA16_SUPPORTED_BITS} is supported ",
                    f"but got format: {CompressionFormat.pack_quantized.value} "
                    f" and bits: {weight_quant.num_bits}",
                )

            # Prefer to use the MarlinMoE kernel when it is supported.
            if (
                not check_moe_marlin_supports_layer(layer, group_size)
                or current_platform.is_rocm()
            ):
                from .compressed_tensors_moe_wna16 import (
                    CompressedTensorsWNA16MoEMethod,
                )

                if (
                    weight_quant.strategy == QuantizationStrategy.GROUP
                    and weight_quant.actorder
                    in (ActivationOrdering.GROUP, ActivationOrdering.DYNAMIC)
                ):
                    raise ValueError(
                        "WNA16MoE is not supported with actorder=group/dynamic."
                    )
                logger.info_once("Using CompressedTensorsWNA16MoEMethod")
                return CompressedTensorsWNA16MoEMethod(
                    weight_quant, input_quant, layer.moe_config
                )
            else:
                from .compressed_tensors_moe_wna16_marlin import (
                    CompressedTensorsWNA16MarlinMoEMethod,
                )

                logger.info_once("Using CompressedTensorsWNA16MarlinMoEMethod")
                return CompressedTensorsWNA16MarlinMoEMethod(
                    weight_quant, input_quant, layer.moe_config
                )
        elif quant_config._is_nvfp4_format(weight_quant):
            from .compressed_tensors_moe_w4a4_nvfp4 import (
                CompressedTensorsW4A4Nvfp4MoEMethod,
            )

            _is_valid_nvfp4_activations = (
                quant_config._is_nvfp4_format(input_quant) or input_quant is None
            )
            if not _is_valid_nvfp4_activations:
                raise ValueError(
                    "For NVFP4 weights, input quantization must also be NVFP4 format ",
                    f"or None for NVFP4A16, found {input_quant}",
                )
            return CompressedTensorsW4A4Nvfp4MoEMethod(
                layer.moe_config, layer_name, use_a16=(input_quant is None)
            )
        elif (
            quant_config._is_fp8_w8a8_sm90(weight_quant, input_quant)
            or quant_config._is_fp8_w8a8_sm100(weight_quant, input_quant)
            or quant_config._is_fp8_w8a8(weight_quant, input_quant)
        ):
            from .compressed_tensors_moe_w8a8_fp8 import (
                CompressedTensorsW8A8Fp8MoEMethod,
            )

            return CompressedTensorsW8A8Fp8MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )
        elif quant_config._is_dynamic_token_w8a8(weight_quant, input_quant):
            from .compressed_tensors_moe_w8a8_int8 import (
                CompressedTensorsW8A8Int8MoEMethod,
            )

            return CompressedTensorsW8A8Int8MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )
        elif quant_config._is_fp8_w4a8_sm90(weight_quant, input_quant):
            from .compressed_tensors_moe_w4a8_fp8 import (
                CompressedTensorsW4A8Fp8MoEMethod,
            )

            logger.info_once("Using CompressedTensorsW4A8Fp8MoEMethod")
            return CompressedTensorsW4A8Fp8MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )
        elif quant_config._is_dynamic_token_w4a8_int(weight_quant, input_quant):
            from .compressed_tensors_moe_w4a8_int8 import (
                CompressedTensorsW4A8Int8MoEMethod,
            )

            return CompressedTensorsW4A8Int8MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}"
            )
