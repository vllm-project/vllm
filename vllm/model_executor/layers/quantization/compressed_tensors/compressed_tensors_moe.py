# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.quantization import ActivationOrdering, QuantizationStrategy

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A4Nvfp4MoeMethod,
    CompressedTensorsW4A8Int8MoEMethod,
    CompressedTensorsW8A8Fp8MoEMethod,
    CompressedTensorsW8A8Int8MoEMethod,
    CompressedTensorsWNA16MarlinMoEMethod,
    CompressedTensorsWNA16MoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_moe_marlin_supports_layer,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


__all__ = [
    "get_compressed_tensors_moe_method",
    "CompressedTensorsW8A8Fp8MoEMethod",
    "CompressedTensorsW8A8Int8MoEMethod",
    "CompressedTensorsWNA16MarlinMoEMethod",
    "CompressedTensorsWNA16MoEMethod",
    "CompressedTensorsW4A4Nvfp4MoeMethod",
    "CompressedTensorsW4A8Int8MoEMethod",
]


def get_compressed_tensors_moe_method(
    quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
    layer: torch.nn.Module,
) -> FusedMoEMethodBase:
    # TODO: @dsikka: refactor this to use schemes as other kernels
    # are supported + check if the layer is being ignored.
    # Check if a using "Linear" to select schemes
    if "Linear" in quant_config.target_scheme_map:
        matched_target = "Linear"
    else:
        # May have instead defined the linear layers in the fused model

        fused_layers = ["re:.*down_proj.*", "re:.*gate_proj.*", "re:.*up_proj.*"]
        current_scheme = None
        for fused_layer in fused_layers:
            # Check if one of the fused layers are defined in quant_config
            matched_target = find_matched_target(
                layer_name=fused_layer,
                module=layer,
                targets=quant_config.target_scheme_map.keys(),
                fused_mapping=quant_config.packed_modules_mapping,
            )

            # Only valid if down_proj, gate_proj, and up_proj
            # are mapped to the same quant scheme in the quant_config
            if current_scheme is None:
                current_scheme = quant_config.target_scheme_map.get(matched_target)
            else:
                assert current_scheme == quant_config.target_scheme_map.get(
                    matched_target
                )

    weight_quant = quant_config.target_scheme_map[matched_target].get("weights")
    input_quant = quant_config.target_scheme_map[matched_target].get(
        "input_activations"
    )

    if quant_config._is_wNa16_group_channel(weight_quant, input_quant):
        # group_size=None means channelwise
        group_size = weight_quant.group_size or -1
        # Prefer to use the MarlinMoE kernel when it is supported.
        if (
            not check_moe_marlin_supports_layer(layer, group_size)
            or current_platform.is_rocm()
        ):
            if (
                weight_quant.strategy == QuantizationStrategy.GROUP
                and weight_quant.actorder
                in (ActivationOrdering.GROUP, ActivationOrdering.DYNAMIC)
            ):
                raise ValueError(
                    "WNA16MoE is not supported with actorder=group/dynamic."
                )
            logger.info_once("Using CompressedTensorsWNA16MoEMethod")
            return CompressedTensorsWNA16MoEMethod(quant_config, layer.moe_config)
        else:
            logger.info_once("Using CompressedTensorsWNA16MarlinMoEMethod")
            return CompressedTensorsWNA16MarlinMoEMethod(quant_config, layer.moe_config)
    elif quant_config._is_fp4a4_nvfp4(weight_quant, input_quant):
        return CompressedTensorsW4A4Nvfp4MoeMethod(layer.moe_config)
    elif (
        quant_config._is_fp8_w8a8_sm90(weight_quant, input_quant)
        or quant_config._is_fp8_w8a8_sm100(weight_quant, input_quant)
        or quant_config._is_fp8_w8a8(weight_quant, input_quant)
    ):
        return CompressedTensorsW8A8Fp8MoEMethod(quant_config, layer.moe_config)
    elif quant_config._is_dynamic_token_w8a8(weight_quant, input_quant):
        return CompressedTensorsW8A8Int8MoEMethod(quant_config, layer.moe_config)
    elif quant_config._is_dynamic_token_w4a8_int(weight_quant, input_quant):
        return CompressedTensorsW4A8Int8MoEMethod(quant_config, layer.moe_config)
    else:
        raise RuntimeError(
            f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}"
        )
