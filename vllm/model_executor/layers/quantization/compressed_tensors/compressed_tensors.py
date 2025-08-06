# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any, Optional

import torch
from compressed_tensors import (KV_CACHE_SCHEME_NAME, SPARSITY_CONFIG_NAME,
                                TRANSFORM_CONFIG_NAME)
from compressed_tensors.config import (CompressionFormat,
                                       SparsityCompressionConfig,
                                       SparsityStructure)
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization import QuantizationConfig as CTQuantConfig
from compressed_tensors.quantization import (QuantizationStrategy,
                                             QuantizationType)
from compressed_tensors.transform import (TransformArgs, TransformConfig,
                                          TransformLocation, TransformScheme)

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    W4A16SPARSE24_SUPPORTED_BITS, WNA16_SUPPORTED_BITS, CompressedTensors24,
    CompressedTensorsScheme, CompressedTensorsW4A4Fp4,
    CompressedTensorsW4A8Int, CompressedTensorsW4A16Fp4,
    CompressedTensorsW4A16Sparse24, CompressedTensorsW8A8Fp8,
    CompressedTensorsW8A8Int8, CompressedTensorsW8A16Fp8,
    CompressedTensorsWNA16)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    is_activation_quantization_format, is_match)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    cutlass_fp4_supported)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

__all__ = ["CompressedTensorsLinearMethod"]


class CompressedTensorsConfig(QuantizationConfig):

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        q_config = config.copy()
        s_config = q_config.pop(SPARSITY_CONFIG_NAME, None)
        t_config = q_config.pop(TRANSFORM_CONFIG_NAME, None)
        kv_scheme = q_config.pop(KV_CACHE_SCHEME_NAME, None)

        self.quant_config = CTQuantConfig.model_validate(q_config)
        self.sparsity_config = SparsityCompressionConfig.model_validate(
            s_config) if s_config else None
        self.transform_config = TransformConfig.model_validate(
            t_config) if t_config else None
        self.kv_cache_scheme = QuantizationArgs.model_validate(
            kv_scheme) if kv_scheme else None

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
        # quantization
        for config_group in self.quant_config.config_groups.values():
            config_group.targets = hf_to_vllm_mapper.apply_list(
                config_group.targets)
        self.quant_config.ignore = hf_to_vllm_mapper.apply_list(
            self.quant_config.ignore)

        # sparsity
        if self.sparsity_config is not None:
            self.sparsity_config.targets = hf_to_vllm_mapper.apply_list(
                self.sparsity_config.targets)
            self.sparsity_config.ignore = hf_to_vllm_mapper.apply_list(
                self.sparsity_config.ignore)

        # transform
        if self.transform_config is not None:
            for scheme in self.transform_config.config_groups.values():
                for arg in scheme.apply:
                    arg.targets = hf_to_vllm_mapper.apply_list(arg.targets)
                    arg.ignore = hf_to_vllm_mapper.apply_list(arg.ignore)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            # TODO (@ksayers): maybe refactor this to live on the
            # LinearMethod, similar to MoEMethod. This makes clear
            # the separation of schemes by module type
            scheme = self.get_scheme(layer=layer, layer_name=prefix)
            if scheme is None:
                return UnquantizedLinearMethod()
            layer.scheme = scheme
            return CompressedTensorsLinearMethod(self)
        if isinstance(layer, Attention):
            return CompressedTensorsKVCacheMethod(self)
        if isinstance(layer, FusedMoE):
            return CompressedTensorsMoEMethod.get_moe_method(self,
                                                             layer=layer,
                                                             layer_name=prefix)
        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CompressedTensorsConfig":
        return cls(config)

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def _check_scheme_supported(self,
                                min_capability: int,
                                error: bool = True,
                                match_exact: bool = False) -> bool:
        capability_tuple = current_platform.get_device_capability()

        if capability_tuple is not None:
            capability = capability_tuple.to_int()
            if match_exact:
                supported = capability == min_capability
                if error and not supported:
                    raise RuntimeError(
                        "Quantization scheme is not supported for ",
                        "the current GPU. Required capability: ",
                        f"{min_capability}. Current capability: {capability}.")
            else:
                supported = capability >= min_capability
                if error and not supported:
                    raise RuntimeError(
                        "Quantization scheme is not supported for ",
                        f"the current GPU. Min capability: {min_capability}. ",
                        f"Current capability: {capability}.")
            return supported
        else:
            return False

    def _is_fp4a4_nvfp4(self, weight_quant: QuantizationArgs,
                        input_quant: Optional[QuantizationArgs]):

        if weight_quant is None or input_quant is None:
            return False

        is_tensor_group_quant = (weight_quant.strategy
                                 == QuantizationStrategy.TENSOR_GROUP.value
                                 and input_quant.strategy
                                 == QuantizationStrategy.TENSOR_GROUP.value)
        is_symmetric = weight_quant.symmetric and input_quant.symmetric

        is_group_size_16 = (weight_quant.group_size == 16
                            and input_quant.group_size == 16)
        is_float_type = (weight_quant.type == QuantizationType.FLOAT
                         and input_quant.type == QuantizationType.FLOAT.value)
        is_4_bits = weight_quant.num_bits == 4 and input_quant.num_bits == 4

        return (is_tensor_group_quant and is_float_type and is_4_bits
                and is_group_size_16 and is_symmetric)

    def _is_fp4a16_nvfp4(self, weight_quant: QuantizationArgs,
                         input_quant: Optional[QuantizationArgs]):

        is_weight_only = weight_quant is not None and input_quant is None
        is_tensor_group_quant = (
            weight_quant.strategy == QuantizationStrategy.TENSOR_GROUP.value)
        is_symmetric = weight_quant.symmetric

        is_group_size_16 = weight_quant.group_size == 16
        is_float_type = weight_quant.type == QuantizationType.FLOAT
        is_4_bits = weight_quant.num_bits == 4

        return (is_weight_only and is_tensor_group_quant and is_float_type
                and is_4_bits and is_group_size_16 and is_symmetric)

    def _is_static_tensor_w8a8(
            self, weight_quant: QuantizationArgs,
            input_quant: Optional[QuantizationArgs]) -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.TENSOR.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value)
        is_tensor = (weight_strategy and input_quant.strategy
                     == QuantizationStrategy.TENSOR.value)
        is_static = not weight_quant.dynamic and not input_quant.dynamic

        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_8_bits and is_tensor and weight_quant.symmetric and is_static

    def _is_dynamic_token_w8a8(
            self, weight_quant: QuantizationArgs,
            input_quant: Optional[QuantizationArgs]) -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.TENSOR.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value)
        is_token = (weight_strategy and input_quant.strategy
                    == QuantizationStrategy.TOKEN.value)
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic

        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_8_bits and is_token and weight_quant.symmetric and is_dynamic

    def _is_dynamic_token_w4a8_int(
            self, weight_quant: QuantizationArgs,
            input_quant: Optional[QuantizationArgs]) -> bool:
        is_weight_4_bits = weight_quant.num_bits == 4
        is_activation_8_bits = input_quant.num_bits == 8
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.GROUP.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value)
        is_token = (weight_strategy and input_quant.strategy
                    == QuantizationStrategy.TOKEN.value)
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic

        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return (is_weight_4_bits and is_activation_8_bits and is_token
                and weight_quant.symmetric and is_dynamic)

    def _is_fp8_w8a8(self, weight_quant: QuantizationArgs,
                     input_quant: Optional[QuantizationArgs]) -> bool:
        # Confirm weights and activations quantized.
        if weight_quant is None or input_quant is None:
            return False

        # Confirm weight scheme is supported.
        is_floating_point = (weight_quant.type == QuantizationType.FLOAT
                             and input_quant.type == QuantizationType.FLOAT)
        is_symmetric_weight = weight_quant.symmetric
        is_static_weight = not weight_quant.dynamic
        is_per_tensor_or_channel_weight = (weight_quant.strategy in [
            QuantizationStrategy.TENSOR, QuantizationStrategy.CHANNEL
        ])
        if not (is_floating_point and is_symmetric_weight and is_static_weight
                and is_per_tensor_or_channel_weight):
            return False

        # Dynamic quantization is always supported if weights supported.
        if input_quant.dynamic:
            return True

        # Confirm activation scheme is supported.
        is_symmetric_activation = input_quant.symmetric
        is_per_tensor_activation = (
            input_quant.strategy == QuantizationStrategy.TENSOR)
        return is_symmetric_activation and is_per_tensor_activation

    def _is_fp8_w8a8_sm90(self, weight_quant: QuantizationArgs,
                          input_quant: Optional[QuantizationArgs]) -> bool:
        return (self._check_scheme_supported(90, error=False, match_exact=True)
                and self._is_fp8_w8a8(weight_quant, input_quant))

    def _is_fp8_w8a8_sm100(self, weight_quant: QuantizationArgs,
                           input_quant: Optional[QuantizationArgs]) -> bool:
        return (self._check_scheme_supported(
            100, error=False, match_exact=True)
                and self._is_fp8_w8a8(weight_quant, input_quant))

    def _is_fp8_w8a16(self, weight_quant: QuantizationArgs,
                      input_quant: Optional[QuantizationArgs]) -> bool:
        # Confirm weights quantized.
        if weight_quant is None:
            return False

        # Confirm we have floating points.
        if weight_quant.type != QuantizationType.FLOAT:
            return False

        # Confirm weight scheme is supported.
        is_symmetric_weight = weight_quant.symmetric
        is_static_weight = not weight_quant.dynamic
        is_per_tensor_or_channel_weight = (weight_quant.strategy in [
            QuantizationStrategy.TENSOR, QuantizationStrategy.CHANNEL
        ])
        if not (is_symmetric_weight and is_static_weight  # noqa: SIM103
                and is_per_tensor_or_channel_weight):
            return False

        # All conditions satisfied.
        return True

    def _is_wNa16_group_channel(
            self, weight_quant: QuantizationArgs,
            input_quant: Optional[QuantizationArgs]) -> bool:
        input_quant_none = input_quant is None
        is_channel_group = (
            weight_quant.strategy == QuantizationStrategy.CHANNEL.value
            or weight_quant.strategy == QuantizationStrategy.GROUP.value)
        is_static = not weight_quant.dynamic

        return (is_channel_group and input_quant_none and is_static)

    def _get_scheme_from_parts(
        self, weight_quant: QuantizationArgs,
        input_quant: Optional[QuantizationArgs], quant_format: str,
        input_tfm: Optional[tuple[TransformScheme, TransformArgs]],
        output_tfm: Optional[tuple[TransformScheme, TransformArgs]]
    ) -> "CompressedTensorsScheme":
        # Detect If Mixed Precision
        if self._is_fp4a16_nvfp4(weight_quant, input_quant):
            return CompressedTensorsW4A16Fp4()

        if self._is_wNa16_group_channel(weight_quant, input_quant):
            if (quant_format == CompressionFormat.marlin_24.value
                    and weight_quant.num_bits in W4A16SPARSE24_SUPPORTED_BITS):
                assert weight_quant.symmetric
                return CompressedTensorsW4A16Sparse24(
                    strategy=weight_quant.strategy,
                    num_bits=weight_quant.num_bits,
                    group_size=weight_quant.group_size)
            if (quant_format == CompressionFormat.pack_quantized.value
                    and weight_quant.num_bits in WNA16_SUPPORTED_BITS):
                return CompressedTensorsWNA16(
                    num_bits=weight_quant.num_bits,
                    strategy=weight_quant.strategy,
                    symmetric=weight_quant.symmetric,
                    group_size=weight_quant.group_size,
                    actorder=weight_quant.actorder)

        if is_activation_quantization_format(quant_format):
            if self._is_fp4a4_nvfp4(weight_quant, input_quant):
                if cutlass_fp4_supported(
                ) or envs.VLLM_USE_NVFP4_CT_EMULATIONS:
                    return CompressedTensorsW4A4Fp4()
                else:
                    logger.warning_once(
                        "Current platform does not support cutlass NVFP4."
                        " Running CompressedTensorsW4A16Fp4.")
                    return CompressedTensorsW4A16Fp4(
                        has_input_global_scale=True)

            if self._is_fp8_w8a8(weight_quant, input_quant):
                is_fp8_w8a8_supported = self._check_scheme_supported(
                    CompressedTensorsW8A8Fp8.get_min_capability(), error=False)
                if is_fp8_w8a8_supported:
                    return CompressedTensorsW8A8Fp8(
                        strategy=weight_quant.strategy,
                        is_static_input_scheme=bool(
                            input_quant and not input_quant.dynamic))
                else:
                    # note: input_quant will be present for converted models;
                    # will be ignored during inference post loading
                    return CompressedTensorsW8A16Fp8(
                        strategy=weight_quant.strategy,
                        is_static_input_scheme=bool(not input_quant.dynamic))

            # note: input_quant can be None
            if self._is_fp8_w8a16(weight_quant, input_quant):
                is_static_input_scheme = bool(input_quant
                                              and not input_quant.dynamic)
                return CompressedTensorsW8A16Fp8(
                    strategy=weight_quant.strategy,
                    is_static_input_scheme=is_static_input_scheme)

            if self._is_static_tensor_w8a8(weight_quant, input_quant):
                return CompressedTensorsW8A8Int8(
                    strategy=weight_quant.strategy,
                    is_static_input_scheme=True,
                    input_symmetric=input_quant.symmetric)

            if self._is_dynamic_token_w8a8(weight_quant, input_quant):
                return CompressedTensorsW8A8Int8(
                    strategy=weight_quant.strategy,
                    is_static_input_scheme=False,
                    input_symmetric=input_quant.symmetric)

            if self._is_dynamic_token_w4a8_int(weight_quant, input_quant):
                is_static_input_scheme = bool(input_quant
                                              and not input_quant.dynamic)
                return CompressedTensorsW4A8Int(
                    num_bits=weight_quant.num_bits,
                    strategy=weight_quant.strategy,
                    group_size=weight_quant.group_size,
                    is_static_input_scheme=is_static_input_scheme,
                    input_symmetric=input_quant.symmetric)

        raise NotImplementedError(
            "No compressed-tensors compatible scheme was found.")

    def get_scheme(self, layer: torch.nn.Module,
                   layer_name: str) -> Optional["CompressedTensorsScheme"]:
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
        # Find quantization, sparsity, and transform args for this layer.
        # Because the config does not see the entire model structure,
        # we must match from layer to schemes (rather than schemes to layers)
        input_quant: Optional[QuantizationArgs] = None
        weight_quant: QuantizationArgs = None
        quant_format: str = self.quant_config.format  # TODO (@bdellabetta)
        sparsity_scheme: Optional[SparsityCompressionConfig] = None
        input_tfm: Optional[tuple[TransformScheme, TransformArgs]] = None
        output_tfm = Optional[tuple[TransformScheme, TransformArgs]] = None

        # throw an error if schemes overlap
        def replace_with_check(original, new):
            if new and original:
                raise ValueError(
                    "The provided compressed tensors config has overlapping "
                    f"config groups for the layer {layer_name}")
            return new or original

        # match quantization args
        for scheme in self.quant_config.config_groups.values():
            if is_match(layer_name,
                        layer,
                        scheme.targets,
                        self.quant_config.ignore,
                        fused=self.packed_modules_mapping):
                input_quant = replace_with_check(input_quant,
                                                 scheme.input_activations)
                weight_quant = replace_with_check(weight_quant, scheme.weights)

        # match sparsity args
        if self.sparsity_config and is_match(
                layer_name,
                layer,
                self.sparsity_config.targets,
                self.sparsity_config.ignore,
                fused=self.packed_modules_mapping):
            sparsity_scheme = self.sparsity_config

        # match transform args
        if self.transform_config is not None:
            for scheme in self.transform_config.config_groups.values():
                for args in scheme.apply:
                    if is_match(layer_name,
                                layer,
                                args.targets,
                                args.ignore,
                                fused=self.packed_modules_mapping):
                        if args.location == TransformLocation.INPUT:
                            input_tfm = replace_with_check(
                                input_tfm, (scheme, args))
                        if args.location == TransformLocation.OUTPUT:
                            output_tfm = replace_with_check(
                                output_tfm, (scheme, args))

        # TODO (@ksayers): Move this check into `_get_scheme_from_parts`
        if self.supports_cutlass_24(weight_quant=weight_quant,
                                    input_quant=input_quant,
                                    sparsity_scheme=sparsity_scheme):
            # Have a valid sparsity scheme
            # Validate layer is supported by Cutlass 2:4 Kernel
            config = (None if sparsity_scheme is None
                      or sparsity_scheme.format == "dense" else self)

            scheme = CompressedTensors24(
                quantized=weight_quant is not None or input_quant is not None,
                weight_quant=weight_quant,
                input_quant=input_quant,
                config=config,
            )

        # TODO (@ksayers): Move this check into `_get_scheme_from_parts`
        elif weight_quant is None:
            logger.warning_once("Acceleration for non-quantized schemes is "
                                "not supported by Compressed Tensors. "
                                "Falling back to UnquantizedLinearMethod")
            return None

        else:
            # Find the quant_scheme
            scheme = self._get_scheme_from_parts(weight_quant=weight_quant,
                                                 input_quant=input_quant,
                                                 quant_format=quant_format,
                                                 input_tfm=input_tfm,
                                                 output_tfm=output_tfm)

        # Raise error if device does not support the scheme
        # (e.g. fp8 needs ada lovelace)
        self._check_scheme_supported(scheme.get_min_capability())
        logger.debug("Using scheme: %s for %s", scheme.__class__.__name__,
                     layer_name)
        return scheme

    def get_cache_scale(self, name: str) -> Optional[str]:
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

    @staticmethod
    def supports_cutlass_24(
            weight_quant: QuantizationArgs,
            input_quant: Optional[QuantizationArgs],
            sparsity_scheme: Optional[SparsityCompressionConfig] = None
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
            sparsity_scheme.sparsity_structure ==
            SparsityStructure.TWO_FOUR.value)

        valid_compressors = {
            CompressionFormat.dense.value,
            CompressionFormat.sparse_24_bitmask.value
        }

        is_valid_sparsity = (is_valid_sparsity_structure
                             and sparsity_scheme.format in valid_compressors)

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
            QuantizationStrategy.CHANNEL.value
        ]

        assert weight_quant is not None
        assert input_quant is not None
        if weight_quant.strategy not in supported_weight_quant_strategies:
            return False

        supported_input_quant_strategies = [
            QuantizationStrategy.TENSOR.value, QuantizationStrategy.TOKEN.value
        ]

        if input_quant.strategy not in supported_input_quant_strategies:
            return False

        return weight_quant.num_bits == input_quant.num_bits == 8


class CompressedTensorsLinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: CompressedTensorsConfig):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
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
            weight_loader=weight_loader)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None):
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
    def validate_kv_cache_scheme(kv_cache_scheme: Optional[dict[str, Any]]):
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
                f"received num_bits={num_bits}, type={type_}")

        strategy = kv_cache_scheme.get("strategy")
        if strategy != "tensor":
            raise NotImplementedError(
                "Only support per-tensor scaling factor "
                "for compressed-tensors KV cache. "
                f"Expected strategy: tensor, found strategy: {strategy}")

        is_symmetric = kv_cache_scheme.get("symmetric")
        if not is_symmetric:
            raise NotImplementedError(
                "Only support symmetric scaling factor "
                "for compressed-tensors KV cache. "
                f"However found symmetric: {is_symmetric}")
