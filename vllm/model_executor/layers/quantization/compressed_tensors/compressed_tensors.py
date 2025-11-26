# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import suppress
from typing import TYPE_CHECKING, Any, Literal, Optional, cast

import torch
from compressed_tensors.config import (
    CompressionFormat,
    SparsityCompressionConfig,
    SparsityStructure,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.transform import TransformConfig

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    W4A16SPARSE24_SUPPORTED_BITS,
    WNA16_SUPPORTED_BITS,
    CompressedTensors24,
    CompressedTensorsScheme,
    CompressedTensorsW4A4Fp4,
    CompressedTensorsW4A8Fp8,
    CompressedTensorsW4A8Int,
    CompressedTensorsW4A16Fp4,
    CompressedTensorsW4A16Sparse24,
    CompressedTensorsW8A8Fp8,
    CompressedTensorsW8A8Int8,
    CompressedTensorsW8A16Fp8,
    CompressedTensorsWNA16,
)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.linear import (  # noqa: E501
    CompressedTensorsLinearTransformMethod,
    get_linear_transform_schemes,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target,
    is_activation_quantization_format,
    should_ignore_layer,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    cutlass_fp4_supported,
)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

__all__ = ["CompressedTensorsLinearMethod"]

SPARSITY_CONFIG_NAME: Literal["sparsity_config"] = "sparsity_config"
QUANTIZATION_SCHEME_MAP_TYPE = dict[str, dict[str, QuantizationArgs] | None]


class CompressedTensorsConfig(QuantizationConfig):
    def __init__(
        self,
        target_scheme_map: dict[str, Any],
        ignore: list[str],
        quant_format: str,
        sparsity_scheme_map: dict[str, SparsityCompressionConfig],
        sparsity_ignore_list: list[str],
        kv_cache_scheme: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        transform_config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.ignore = ignore
        self.quant_format = quant_format
        # Map from [target -> scheme]
        self.target_scheme_map = target_scheme_map
        self.kv_cache_scheme = kv_cache_scheme
        self.sparsity_scheme_map = sparsity_scheme_map
        self.sparsity_ignore_list = sparsity_ignore_list
        self.config = config

        if transform_config:
            self.transform_config = TransformConfig.model_validate(transform_config)
        else:
            self.transform_config = None

    def get_linear_method(self) -> "CompressedTensorsLinearMethod":
        return CompressedTensorsLinearMethod(self)

    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def get_name(self) -> QuantizationMethods:
        return "compressed-tensors"

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        self.target_scheme_map = hf_to_vllm_mapper.apply_dict(self.target_scheme_map)
        self.ignore = hf_to_vllm_mapper.apply_list(self.ignore)
        self.sparsity_scheme_map = hf_to_vllm_mapper.apply_dict(
            self.sparsity_scheme_map
        )
        self.sparsity_ignore_list = hf_to_vllm_mapper.apply_list(
            self.sparsity_ignore_list
        )
        if self.kv_cache_scheme is not None:
            self.kv_cache_scheme = hf_to_vllm_mapper.apply_dict(self.kv_cache_scheme)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            # collect schemes
            quant_scheme = self.get_scheme(layer=layer, layer_name=prefix)
            input_tfms, output_tfms = get_linear_transform_schemes(
                layer, prefix, self.transform_config, self.packed_modules_mapping
            )

            # choose quantization method
            quant_method: LinearMethodBase = UnquantizedLinearMethod()
            if quant_scheme is not None:
                layer.scheme = quant_scheme
                quant_method = CompressedTensorsLinearMethod(self)

            # choose transform method
            if any((input_tfms, output_tfms)):
                return CompressedTensorsLinearTransformMethod.from_schemes(
                    quant_method, quant_scheme, input_tfms, output_tfms
                )

            else:
                return quant_method

        if isinstance(layer, Attention):
            return CompressedTensorsKVCacheMethod(self)
        if isinstance(layer, FusedMoE):
            return CompressedTensorsMoEMethod.get_moe_method(self, layer)
        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CompressedTensorsConfig":
        ignore: list[str] = cast(list[str], config.get("ignore", []))
        quant_format = cast(str, config.get("format"))
        target_scheme_map = cls._quantization_scheme_map_from_config(config=config)
        sparsity_scheme_map, sparsity_ignore_list = cls._parse_sparsity_config(
            config=config
        )
        transform_config = config.get("transform_config")

        return cls(
            target_scheme_map=target_scheme_map,
            ignore=ignore,
            quant_format=quant_format,
            sparsity_scheme_map=sparsity_scheme_map,
            sparsity_ignore_list=sparsity_ignore_list,
            config=config,
            transform_config=transform_config,
        )

    @classmethod
    def _parse_sparsity_config(
        cls, config: dict[str, Any]
    ) -> tuple[dict[str, SparsityCompressionConfig], list[str]]:
        """
        :param config: The `quantization_config` dictionary from config.json
        :return: A tuple with two elements
            1. A dictionary mapping target layer names to their corresponding
                sparsity_config
            2. A list of layer names to ignore for sparsity
        """
        if not (sparsity_config := config.get(SPARSITY_CONFIG_NAME)):
            return dict(), []

        sparsity_config = SparsityCompressionConfig.model_validate(sparsity_config)
        sparse_scheme_map: dict[str, SparsityCompressionConfig] = {
            target: sparsity_config for target in sparsity_config.targets or list()
        }
        sparsity_ignore_list = sparsity_config.ignore or list()
        return sparse_scheme_map, sparsity_ignore_list

    @classmethod
    def _quantization_scheme_map_from_config(
        cls, config: dict[str, Any]
    ) -> QUANTIZATION_SCHEME_MAP_TYPE:
        """
        :param config: The `quantization_config` dictionary from config.json
        :return: A dictionary mapping target layer names to their corresponding
            quantization_args for weights and input activations
        """
        target_scheme_map: dict[str, Any] = dict()
        quant_format = cast(str, config.get("format"))

        # The quant_config has multiple config_groups, each containing
        # an input_activations key with details about how the activations are
        # quantized, a weights key indicating how the weights are quantized,
        # and a list of targets under the `targets` key, dictating which
        # layers are impacted by the quantization details. The quantization
        # details follow the structure defined by the QuantizationArgs
        # pydantic model, which is used to verify the structure of the
        # quant_config and also store the details for later use.

        config_groups = config.get("config_groups", dict())
        for _, quant_config in config_groups.items():
            targets = quant_config.get("targets")
            for target in targets:
                target_scheme_map[target] = {}
                target_scheme_map[target]["weights"] = QuantizationArgs.model_validate(
                    quant_config.get("weights")
                )

                target_scheme_map[target]["input_activations"] = None
                target_scheme_map[target]["format"] = quant_config.get("format")
                format = target_scheme_map[target].get("format")
                # If no per-config format defined, use global format in config
                act_quant_format = (
                    is_activation_quantization_format(format)
                    if format is not None
                    else is_activation_quantization_format(quant_format)
                )
                # TODO(czhu): w4a8fp8 is in packed-quantized format
                # but needs input activation quantization
                input_activations = quant_config.get("input_activations")
                if act_quant_format or input_activations:
                    # The only case where we have activation quant supported
                    # but no input_activations provided in the config
                    # should be w8a16fp8 w8a16fp8 can also run for cases where
                    # there is an input_quant but it is ignored
                    if not input_activations:
                        assert (
                            target_scheme_map[target]["weights"].type
                            == QuantizationType.FLOAT
                        )
                    else:
                        target_scheme_map[target]["input_activations"] = (
                            QuantizationArgs.model_validate(
                                quant_config.get("input_activations")
                            )
                        )
        return target_scheme_map

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @staticmethod
    def _check_scheme_supported(
        min_capability: int, error: bool = True, match_exact: bool = False
    ) -> bool:
        capability_tuple = current_platform.get_device_capability()

        if capability_tuple is not None:
            capability = capability_tuple.to_int()
            if match_exact:
                supported = capability == min_capability
                if error and not supported:
                    raise RuntimeError(
                        "Quantization scheme is not supported for ",
                        "the current GPU. Required capability: ",
                        f"{min_capability}. Current capability: {capability}.",
                    )
            else:
                supported = capability >= min_capability
                if error and not supported:
                    raise RuntimeError(
                        "Quantization scheme is not supported for ",
                        f"the current GPU. Min capability: {min_capability}. ",
                        f"Current capability: {capability}.",
                    )
            return supported
        else:
            return False

    @staticmethod
    def _is_fp4a4_nvfp4(weight_quant: QuantizationArgs, input_quant: QuantizationArgs):
        if weight_quant is None or input_quant is None:
            return False

        is_tensor_group_quant = (
            weight_quant.strategy == QuantizationStrategy.TENSOR_GROUP.value
            and input_quant.strategy == QuantizationStrategy.TENSOR_GROUP.value
        )
        is_symmetric = weight_quant.symmetric and input_quant.symmetric

        is_group_size_16 = (
            weight_quant.group_size == 16 and input_quant.group_size == 16
        )
        is_float_type = (
            weight_quant.type == QuantizationType.FLOAT
            and input_quant.type == QuantizationType.FLOAT
        )
        is_4_bits = weight_quant.num_bits == 4 and input_quant.num_bits == 4

        return (
            is_tensor_group_quant
            and is_float_type
            and is_4_bits
            and is_group_size_16
            and is_symmetric
        )

    @staticmethod
    def _is_fp4a16_nvfp4(weight_quant: QuantizationArgs, input_quant: QuantizationArgs):
        is_weight_only = weight_quant is not None and input_quant is None
        is_tensor_group_quant = (
            weight_quant.strategy == QuantizationStrategy.TENSOR_GROUP.value
        )
        is_symmetric = weight_quant.symmetric

        is_group_size_16 = weight_quant.group_size == 16
        is_float_type = weight_quant.type == QuantizationType.FLOAT
        is_4_bits = weight_quant.num_bits == 4

        return (
            is_weight_only
            and is_tensor_group_quant
            and is_float_type
            and is_4_bits
            and is_group_size_16
            and is_symmetric
        )

    @staticmethod
    def _is_static_tensor_w8a8(
        weight_quant: QuantizationArgs, input_quant: QuantizationArgs
    ) -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.TENSOR.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
        )
        is_tensor = (
            weight_strategy
            and input_quant.strategy == QuantizationStrategy.TENSOR.value
        )
        is_static = not weight_quant.dynamic and not input_quant.dynamic

        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_8_bits and is_tensor and weight_quant.symmetric and is_static

    @staticmethod
    def _is_dynamic_token_w8a8(
        weight_quant: QuantizationArgs, input_quant: QuantizationArgs
    ) -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.TENSOR.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
        )
        is_token = (
            weight_strategy and input_quant.strategy == QuantizationStrategy.TOKEN.value
        )
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic

        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_8_bits and is_token and weight_quant.symmetric and is_dynamic

    @staticmethod
    def _is_dynamic_token_w4a8_int(
        weight_quant: QuantizationArgs, input_quant: QuantizationArgs
    ) -> bool:
        is_weight_4_bits = weight_quant.num_bits == 4
        is_activation_8_bits = input_quant.num_bits == 8
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.GROUP.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
        )
        is_token = (
            weight_strategy and input_quant.strategy == QuantizationStrategy.TOKEN.value
        )
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic

        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return (
            is_weight_4_bits
            and is_activation_8_bits
            and is_token
            and weight_quant.symmetric
            and is_dynamic
        )

    @staticmethod
    def _is_fp8_w8a8(
        weight_quant: QuantizationArgs, input_quant: QuantizationArgs
    ) -> bool:
        # Confirm weights and activations quantized.
        if weight_quant is None or input_quant is None:
            return False

        # Confirm weight scheme is supported.
        is_floating_point = (
            weight_quant.type == QuantizationType.FLOAT
            and input_quant.type == QuantizationType.FLOAT
        )
        is_symmetric_weight = weight_quant.symmetric
        is_static_weight = not weight_quant.dynamic
        is_tensor_or_channel_or_block_weight = weight_quant.strategy in [
            QuantizationStrategy.TENSOR,
            QuantizationStrategy.CHANNEL,
            QuantizationStrategy.BLOCK,
        ]
        if not (
            is_floating_point
            and is_symmetric_weight
            and is_static_weight
            and is_tensor_or_channel_or_block_weight
        ):
            return False

        # Dynamic quantization is always supported if weights supported.
        if input_quant.dynamic:
            return True

        # Confirm activation scheme is supported.
        is_symmetric_activation = input_quant.symmetric
        is_per_tensor_activation = input_quant.strategy == QuantizationStrategy.TENSOR
        return is_symmetric_activation and is_per_tensor_activation

    @staticmethod
    def _is_fp8_w4a8(
        weight_quant: QuantizationArgs, input_quant: QuantizationArgs
    ) -> bool:
        if not weight_quant or not input_quant:
            return False
        is_weight_4_bits = weight_quant.num_bits == 4
        is_activation_8_bits = input_quant.num_bits == 8
        weight_strategy = weight_quant.strategy == QuantizationStrategy.GROUP.value
        is_token = (
            weight_strategy and input_quant.strategy == QuantizationStrategy.TOKEN.value
        )
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic
        is_symmetric = weight_quant.symmetric and input_quant.symmetric
        # Only per-group symmetric weight (4bit)
        # + per-tok symmetric activation (8bit) quantization supported.
        return (
            is_weight_4_bits
            and is_activation_8_bits
            and is_token
            and is_symmetric
            and is_dynamic
        )

    @classmethod
    def _is_fp8_w4a8_sm90(
        cls, weight_quant: QuantizationArgs, input_quant: QuantizationArgs
    ) -> bool:
        return cls._check_scheme_supported(
            90, error=False, match_exact=True
        ) and cls._is_fp8_w4a8(weight_quant, input_quant)

    @classmethod
    def _is_fp8_w8a8_sm90(
        cls, weight_quant: QuantizationArgs, input_quant: QuantizationArgs
    ) -> bool:
        return cls._check_scheme_supported(
            90, error=False, match_exact=True
        ) and cls._is_fp8_w8a8(weight_quant, input_quant)

    @classmethod
    def _is_fp8_w8a8_sm100(
        cls, weight_quant: QuantizationArgs, input_quant: QuantizationArgs
    ) -> bool:
        return cls._check_scheme_supported(
            100, error=False, match_exact=True
        ) and cls._is_fp8_w8a8(weight_quant, input_quant)

    @staticmethod
    def _is_fp8_w8a16(
        weight_quant: QuantizationArgs, input_quant: QuantizationArgs
    ) -> bool:
        # Confirm weights quantized.
        if weight_quant is None:
            return False

        # Confirm we have floating points.
        if weight_quant.type != QuantizationType.FLOAT:
            return False

        # Confirm weight scheme is supported.
        is_symmetric_weight = weight_quant.symmetric
        is_static_weight = not weight_quant.dynamic
        is_tensor_or_channel_or_block_weight = weight_quant.strategy in [
            QuantizationStrategy.TENSOR,
            QuantizationStrategy.CHANNEL,
            QuantizationStrategy.BLOCK,
        ]
        return (
            is_symmetric_weight
            and is_static_weight
            and is_tensor_or_channel_or_block_weight
        )

    @staticmethod
    def _is_wNa16_group_channel(
        weight_quant: QuantizationArgs, input_quant: QuantizationArgs
    ) -> bool:
        input_quant_none = input_quant is None
        is_channel_group = (
            weight_quant.strategy == QuantizationStrategy.CHANNEL.value
            or weight_quant.strategy == QuantizationStrategy.GROUP.value
        )
        is_static = not weight_quant.dynamic

        return is_channel_group and input_quant_none and is_static

    def _get_scheme_from_parts(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        format: str | None = None,
    ) -> "CompressedTensorsScheme":
        # use the per-layer format if defined, otherwise, use global format
        format = format if format is not None else self.quant_format

        # Detect If Mixed Precision
        if self._is_fp4a16_nvfp4(weight_quant, input_quant):
            return CompressedTensorsW4A16Fp4()

        if self._is_fp8_w4a8_sm90(weight_quant, input_quant):
            return CompressedTensorsW4A8Fp8(
                num_bits=weight_quant.num_bits,
                strategy=weight_quant.strategy,
                symmetric=weight_quant.symmetric,
                group_size=weight_quant.group_size,
                actorder=weight_quant.actorder,
            )

        if self._is_wNa16_group_channel(weight_quant, input_quant):
            if (
                format == CompressionFormat.marlin_24.value
                and weight_quant.num_bits in W4A16SPARSE24_SUPPORTED_BITS
            ):
                assert weight_quant.symmetric
                return CompressedTensorsW4A16Sparse24(
                    strategy=weight_quant.strategy,
                    num_bits=weight_quant.num_bits,
                    group_size=weight_quant.group_size,
                )
            if (
                format == CompressionFormat.pack_quantized.value
                and weight_quant.num_bits in WNA16_SUPPORTED_BITS
            ):
                return CompressedTensorsWNA16(
                    num_bits=weight_quant.num_bits,
                    strategy=weight_quant.strategy,
                    symmetric=weight_quant.symmetric,
                    group_size=weight_quant.group_size,
                    actorder=weight_quant.actorder,
                )

        act_quant_format = is_activation_quantization_format(format)
        if act_quant_format:
            if self._is_fp4a4_nvfp4(weight_quant, input_quant):
                if cutlass_fp4_supported() or envs.VLLM_USE_NVFP4_CT_EMULATIONS:
                    return CompressedTensorsW4A4Fp4()
                else:
                    logger.warning_once(
                        "Current platform does not support cutlass NVFP4."
                        " Running CompressedTensorsW4A16Fp4."
                    )
                    return CompressedTensorsW4A16Fp4(has_input_global_scale=True)

            if self._is_fp8_w8a8(weight_quant, input_quant):
                is_fp8_w8a8_supported = self._check_scheme_supported(
                    CompressedTensorsW8A8Fp8.get_min_capability(), error=False
                )
                if is_fp8_w8a8_supported:
                    return CompressedTensorsW8A8Fp8(
                        weight_quant=weight_quant,
                        is_static_input_scheme=(
                            input_quant and not input_quant.dynamic
                        ),
                    )
                else:
                    # note: input_quant will be present for converted models;
                    # will be ignored during inference post loading
                    return CompressedTensorsW8A16Fp8(
                        strategy=weight_quant.strategy,
                        is_static_input_scheme=not input_quant.dynamic,
                    )

            # note: input_quant can be None
            if self._is_fp8_w8a16(weight_quant, input_quant):
                is_static_input_scheme = input_quant and not input_quant.dynamic
                return CompressedTensorsW8A16Fp8(
                    strategy=weight_quant.strategy,
                    is_static_input_scheme=is_static_input_scheme,
                )

            if self._is_static_tensor_w8a8(weight_quant, input_quant):
                return CompressedTensorsW8A8Int8(
                    strategy=weight_quant.strategy,
                    is_static_input_scheme=True,
                    input_symmetric=input_quant.symmetric,
                )

            if self._is_dynamic_token_w8a8(weight_quant, input_quant):
                return CompressedTensorsW8A8Int8(
                    strategy=weight_quant.strategy,
                    is_static_input_scheme=False,
                    input_symmetric=input_quant.symmetric,
                )

            if self._is_dynamic_token_w4a8_int(weight_quant, input_quant):
                is_static_input_scheme = input_quant and not input_quant.dynamic
                return CompressedTensorsW4A8Int(
                    num_bits=weight_quant.num_bits,
                    strategy=weight_quant.strategy,
                    group_size=weight_quant.group_size,
                    is_static_input_scheme=is_static_input_scheme,
                    input_symmetric=input_quant.symmetric,
                )

        raise NotImplementedError("No compressed-tensors compatible scheme was found.")

    def get_scheme(
        self, layer: torch.nn.Module, layer_name: str | None = None
    ) -> Optional["CompressedTensorsScheme"]:
        """
        compressed-tensors supports non uniform in the following way:

        targets of config_groups: There can be N config_groups which each
            have a quantization scheme. Each config_group has a list of targets
            which can be a full layer_name, a regex for a layer_name, or
            an nn.Module name.

        Detect whether a layer_name is found in any target and
        use the quantization scheme corresponding to the matched target
        to select the CompressedTensorsScheme used for inference.
        """

        # Find the "target" in the compressed-tensors config
        # that our layer conforms to.
        # TODO (@kylesayrs): support ignore module names with ct matching utils
        if should_ignore_layer(
            layer_name, ignore=self.ignore, fused_mapping=self.packed_modules_mapping
        ):
            return None

        # Will be empty for models with only sparsity
        weight_quant = input_quant = None
        if self.target_scheme_map:
            matched_target = find_matched_target(
                layer_name=layer_name,
                module=layer,
                targets=self.target_scheme_map.keys(),
                fused_mapping=self.packed_modules_mapping,
            )

            scheme_dict = self.target_scheme_map[matched_target]
            weight_quant = scheme_dict.get("weights")
            input_quant = scheme_dict.get("input_activations")
            format = scheme_dict.get("format")

        # Find the sparsity scheme of the layer
        # assume that fused layers inherit first component's sparsity scheme
        sparsity_targets = self.sparsity_scheme_map.keys() - set(
            self.sparsity_ignore_list
        )
        sparsity_scheme: SparsityCompressionConfig | None = None
        with suppress(ValueError):
            matched_target = find_matched_target(
                layer_name=layer_name,
                module=layer,
                targets=sparsity_targets,
                fused_mapping=self.packed_modules_mapping,
            )
            sparsity_scheme = self.sparsity_scheme_map[matched_target]

        if self.supports_cutlass_24(
            weight_quant=weight_quant,
            input_quant=input_quant,
            sparsity_scheme=sparsity_scheme,
        ):
            # Have a valid sparsity scheme
            # Validate layer is supported by Cutlass 2:4 Kernel
            model_compression_config = (
                None
                if sparsity_scheme is None or sparsity_scheme.format == "dense"
                else self.config
            )

            scheme = CompressedTensors24(
                quantized=weight_quant is not None or input_quant is not None,
                weight_quant=weight_quant,
                input_quant=input_quant,
                model_compression_config=model_compression_config,
            )
        elif weight_quant is None:
            logger.warning_once(
                "Acceleration for non-quantized schemes is "
                "not supported by Compressed Tensors. "
                "Falling back to UnquantizedLinearMethod"
            )
            return None

        else:
            # Find the quant_scheme
            scheme = self._get_scheme_from_parts(  # type: ignore
                weight_quant=weight_quant, input_quant=input_quant, format=format
            )

        # Raise error if device does not support the scheme
        # (e.g. fp8 needs ada lovelace)
        self._check_scheme_supported(scheme.get_min_capability())
        logger.debug("Using scheme: %s for %s", scheme.__class__.__name__, layer_name)
        return scheme

    def get_cache_scale(self, name: str) -> str | None:
        """
        Check whether the param name matches the format for k/v cache scales
        in compressed-tensors. If this is the case, return its equivalent
        param name expected by vLLM

        :param name: param name
        :return: matching param name for KV cache scale in vLLM
        """
        if name.endswith(".output_scale") and ".k_proj" in name:
            return name.replace(".k_proj.output_scale", ".attn.k_scale")
        if name.endswith(".output_scale") and ".v_proj" in name:
            return name.replace(".v_proj.output_scale", ".attn.v_scale")
        # If no matches, return None
        return None

    def has_blocked_weights(self) -> bool:
        for scheme in self.target_scheme_map.values():
            weight_quant = scheme.get("weights")
            if (
                weight_quant is not None
                and weight_quant.strategy == QuantizationStrategy.BLOCK
            ):
                return True
        return False

    @staticmethod
    def supports_cutlass_24(
        weight_quant: QuantizationArgs | None,
        input_quant: QuantizationArgs | None,
        sparsity_scheme: SparsityCompressionConfig | None = None,
    ) -> bool:
        """
        Check if the layer is supported by the Cutlass 2:4 Kernel
        Conditions:
            - Overarching condition: Sparsity Structure is 2:4
            - Unquantized cases are supported
            - Weight only quantization is not-supported
            - Supported weight quantization strategies are TENSOR and CHANNEL
            - Supported input quantization strategies are TENSOR and TOKEN
            - Only 8 bit quantization is supported

        :return: True if the layer is supported by the Cutlass 2:4 Kernel
            False otherwise
        """
        if sparsity_scheme is None:
            return False

        is_valid_sparsity_structure: bool = (
            sparsity_scheme.sparsity_structure == SparsityStructure.TWO_FOUR.value
        )

        valid_compressors = {
            CompressionFormat.dense.value,
            CompressionFormat.sparse_24_bitmask.value,
        }

        is_valid_sparsity = (
            is_valid_sparsity_structure and sparsity_scheme.format in valid_compressors
        )

        if not is_valid_sparsity:
            return False

        # Unquantized cases are supported
        if weight_quant is None and input_quant is None:
            return True

        # Weight only quantization is not-supported
        if weight_quant is not None and input_quant is None:
            return False

        supported_weight_quant_strategies = [
            QuantizationStrategy.TENSOR.value,
            QuantizationStrategy.CHANNEL.value,
        ]

        assert weight_quant is not None
        assert input_quant is not None
        if weight_quant.strategy not in supported_weight_quant_strategies:
            return False

        supported_input_quant_strategies = [
            QuantizationStrategy.TENSOR.value,
            QuantizationStrategy.TOKEN.value,
        ]

        if input_quant.strategy not in supported_input_quant_strategies:
            return False

        return weight_quant.num_bits == input_quant.num_bits == 8


class CompressedTensorsLinearMethod(LinearMethodBase):
    def __init__(self, quantization_config: CompressedTensorsConfig):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """
        Use the CompressedTensorsScheme associated with each layer to create
        the necessary parameters for the layer. See LinearMethodBase for param
        details
        """
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.scheme.create_weights(
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        """
        Use the output of create_weights and the CompressedTensorsScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details

        """
        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, x, bias=bias)


class CompressedTensorsKVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from compressed-tensors
    checkpoints.
    """

    def __init__(self, quant_config: CompressedTensorsConfig):
        self.validate_kv_cache_scheme(quant_config.kv_cache_scheme)
        super().__init__(quant_config)

    @staticmethod
    def validate_kv_cache_scheme(kv_cache_scheme: dict[str, Any] | None):
        """
        Validator for the kv cache scheme. Useful for controlling the
        kv cache quantization schemes, that are being supported in vLLM
        :param kv_cache_scheme: the compressed-tensors kv cache scheme
        """
        if kv_cache_scheme is None:
            return

        type_ = kv_cache_scheme.get("type")
        num_bits = kv_cache_scheme.get("num_bits")

        if type_ != "float" and num_bits != 8:
            raise NotImplementedError(
                "Currently supported kv cache quantization is "
                "num_bits=8, type=float, however "
                f"received num_bits={num_bits}, type={type_}"
            )

        strategy = kv_cache_scheme.get("strategy")
        if strategy != "tensor":
            raise NotImplementedError(
                "Only support per-tensor scaling factor "
                "for compressed-tensors KV cache. "
                f"Expected strategy: tensor, found strategy: {strategy}"
            )

        is_symmetric = kv_cache_scheme.get("symmetric")
        if not is_symmetric:
            raise NotImplementedError(
                "Only support symmetric scaling factor "
                "for compressed-tensors KV cache. "
                f"However found symmetric: {is_symmetric}"
            )
