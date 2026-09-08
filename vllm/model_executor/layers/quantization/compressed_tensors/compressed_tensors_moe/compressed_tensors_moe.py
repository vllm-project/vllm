# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import TYPE_CHECKING

import torch
from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationStrategy,
    QuantizationType,
)

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa
    WNA16_SUPPORTED_BITS,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig,
    )

logger = init_logger(__name__)

_MOE_PREFIX_ALIASES = ((".block_sparse_moe.experts", ".mlp.experts"),)
_MOE_PROJECTIONS = (
    ("gate_proj", "ckpt_gate_proj_name"),
    ("up_proj", "ckpt_up_proj_name"),
    ("down_proj", "ckpt_down_proj_name"),
)


def _get_moe_scheme_dicts(
    quant_config: "CompressedTensorsConfig",
    layer: torch.nn.Module,
    layer_name: str,
) -> list[dict | None]:
    """Resolve each expert projection before falling back to its module class."""
    prefixes = [layer_name]
    for left, right in _MOE_PREFIX_ALIASES:
        for source, target in ((left, right), (right, left)):
            if source in layer_name:
                prefixes.append(layer_name.replace(source, target))

    schemes = []
    for canonical_name, checkpoint_attr in _MOE_PROJECTIONS:
        projection_names = [canonical_name]
        checkpoint_name = getattr(layer, checkpoint_attr, canonical_name)
        if checkpoint_name not in projection_names:
            projection_names.append(checkpoint_name)
        candidates = [
            f"{prefix}.0.{projection_name}"
            for prefix in prefixes
            for projection_name in projection_names
        ]

        if any(
            should_ignore_layer(
                candidate,
                ignore=quant_config.ignore,
                fused_mapping=quant_config.packed_modules_mapping,
            )
            for candidate in candidates
        ):
            schemes.append(None)
            continue

        scheme = None
        for candidate in candidates:
            scheme = quant_config.get_scheme_dict(layer, candidate, match_module=False)
            if scheme is not None:
                break

        if scheme is None:
            scheme = quant_config.get_scheme_dict(layer, candidates[0])
        schemes.append(scheme)

    return schemes


class CompressedTensorsMoEMethod(FusedMoEMethodBase):
    @staticmethod
    def get_moe_method(
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module,
        layer_name: str,
    ) -> FusedMoEMethodBase:
        # RoutedExperts was made by combining multiple Linears so need to
        # make sure quantization config for Linear can target it
        quant_config._add_fused_moe_to_target_scheme_map()
        all_scheme_dicts = _get_moe_scheme_dicts(quant_config, layer, layer_name)
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

        if quant_config._is_mxfp8(weight_quant):
            from .compressed_tensors_moe_w8a8_mxfp8 import (
                CompressedTensorsW8A8Mxfp8MoEMethod,
            )

            return CompressedTensorsW8A8Mxfp8MoEMethod(layer.moe_config)

        if quant_config._is_wNa16_group_channel(weight_quant, input_quant):
            valid_format_and_bits = (
                weight_quant.num_bits in WNA16_SUPPORTED_BITS
                and format == CompressionFormat.pack_quantized.value
            )

            if not valid_format_and_bits:
                raise ValueError(
                    "For Fused MoE layers, only format: "
                    f"{CompressionFormat.pack_quantized.value} "
                    f"and bits: {WNA16_SUPPORTED_BITS} is supported "
                    f"but got format: {CompressionFormat.pack_quantized.value} "
                    f"and bits: {weight_quant.num_bits}"
                )

            # Native ROCm HIP kernels (RDNA3, etc.)
            if current_platform.is_rocm():
                from . import rocm_moe_rdna

                if rocm_moe_rdna.is_supported(weight_quant):
                    return rocm_moe_rdna.make_method(
                        weight_quant, input_quant, layer.moe_config
                    )
                from vllm.platforms.rocm import on_gfx950

                vllm_config = get_current_vllm_config()
                is_lora_disabled = vllm_config.lora_config is None
                moe_backend = vllm_config.kernel_config.moe_backend
                group_size = weight_quant.group_size or -1
                if (
                    weight_quant.strategy == QuantizationStrategy.GROUP
                    and weight_quant.type == QuantizationType.INT
                    and group_size == 32
                    and weight_quant.num_bits == 4
                    and is_lora_disabled
                    and on_gfx950()
                    and moe_backend == "flydsl"
                ):
                    from .compressed_tensors_moe_w4a16_flydsl import (
                        CompressedTensorsW4A16FlydslMoEMethod,
                    )

                    logger.info_once("Using CompressedTensorsW4A16FlydslMoEMethod")
                    return CompressedTensorsW4A16FlydslMoEMethod(
                        weight_quant, input_quant, layer.moe_config
                    )
                elif moe_backend == "emulation":
                    logger.info_once(
                        "Using CompressedTensorsWNA16MoEMethod "
                        "(emulation backend requested)"
                    )

            from .compressed_tensors_moe_wna16 import (
                CompressedTensorsWNA16MoEMethod,
            )

            logger.info_once("Using CompressedTensorsWNA16MoEMethod")
            return CompressedTensorsWNA16MoEMethod(
                weight_quant,
                input_quant,
                layer.moe_config,
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
                    "For NVFP4 weights, input quantization must also be NVFP4 "
                    f"format or None for NVFP4A16, found {input_quant}"
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
