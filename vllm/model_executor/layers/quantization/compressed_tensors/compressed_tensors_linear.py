# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch
from compressed_tensors import CompressionFormat, SparsityCompressionConfig
from compressed_tensors.quantization import (QuantizationArgs,
                                             QuantizationConfig)
from compressed_tensors.transform import (TransformArgs, TransformLocation,
                                          TransformScheme)

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (  # noqa: E501
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_24 import (  # noqa: E501
    CompressedTensors24, supports_cutlass_24)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import (  # noqa: E501
    CompressedTensorsW4A4Fp4, is_fp4a4_nvfp4)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_int import (  # noqa: E501
    CompressedTensorsW4A8Int, is_dynamic_token_w4a8_int)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a16_24 import (  # noqa: E501
    W4A16SPARSE24_SUPPORTED_BITS, CompressedTensorsW4A16Sparse24)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a16_nvfp4 import (  # noqa: E501
    CompressedTensorsW4A16Fp4, is_fp4a16_nvfp4)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (  # noqa: E501
    CompressedTensorsW8A8Fp8, is_fp8_w8a8)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (  # noqa: E501
    CompressedTensorsW8A8Int8, is_dynamic_token_w8a8, is_static_tensor_w8a8)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a16_fp8 import (  # noqa: E501
    CompressedTensorsW8A16Fp8, is_fp8_w8a16)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa: E501
    WNA16_SUPPORTED_BITS, CompressedTensorsWNA16, is_wNa16_group_channel)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    is_activation_quantization_format, is_match)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    cutlass_fp4_supported)

logger = init_logger(__name__)


class CompressedTensorsLinearMethod(LinearMethodBase):
    """
    TODO: write a better docstring. This is essentially just a
    thin wrapper around the scheme. In the future, maybe each
    scheme could provide a linear method
    """

    @staticmethod
    def get_linear_method(
        cls, layer: torch.nn.Module, layer_name: str,
        config: CompressedTensorsConfig
    ) -> "CompressedTensorsLinearMethod" | "UnquantizedLinearMethod":
        scheme = get_scheme(layer_name, layer, config)
        if scheme is None:
            return UnquantizedLinearMethod()

        return cls()

    def __init__(self, scheme: CompressedTensorsScheme):
        self.scheme = scheme

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.scheme.process_weights_after_loading(layer)

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
        self.scheme.create_weights(
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
        return self.scheme.apply_weights(layer, x, bias=bias)


def get_scheme(config: CompressedTensorsConfig, layer: torch.nn.Module,
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
    quant_format: str = config.quant_config.format  # TODO (@bdellabetta)
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
    for scheme in config.quant_config.config_groups.values():
        if is_match(layer_name,
                    layer,
                    scheme.targets,
                    config.quant_config.ignore,
                    fused=config.packed_modules_mapping):
            input_quant = replace_with_check(input_quant,
                                             scheme.input_activations)
            weight_quant = replace_with_check(weight_quant, scheme.weights)

    # match sparsity args
    if config.sparsity_config and is_match(
            layer_name,
            layer,
            config.sparsity_config.targets,
            config.sparsity_config.ignore,
            fused=config.packed_modules_mapping):
        sparsity_scheme = config.sparsity_config

    # match transform args
    if config.transform_config is not None:
        for scheme in config.transform_config.config_groups.values():
            for args in scheme.apply:
                if is_match(layer_name,
                            layer,
                            args.targets,
                            args.ignore,
                            fused=config.packed_modules_mapping):
                    if args.location == TransformLocation.INPUT:
                        input_tfm = replace_with_check(input_tfm,
                                                       (scheme, args))
                    if args.location == TransformLocation.OUTPUT:
                        output_tfm = replace_with_check(
                            output_tfm, (scheme, args))

    # Find the scheme which determines LinearMethod behavior
    scheme: Optional[CompressedTensorsScheme] = get_scheme_from_parts(
        weight_quant=weight_quant,
        input_quant=input_quant,
        quant_format=quant_format,
        sparisty_scheme=sparsity_scheme,
        input_tfm=input_tfm,
        output_tfm=output_tfm,
        quant_config=config.quant_config)

    if scheme is not None:
        # Raise error if device does not support the scheme
        # (e.g. fp8 needs ada lovelace)
        scheme.check_scheme_supported(scheme.get_min_capability())
        logger.debug("Using scheme: %s for %s", scheme.__class__.__name__,
                     layer_name)
    return scheme


def get_scheme_from_parts(
        weight_quant: Optional[QuantizationArgs],
        input_quant: Optional[QuantizationArgs],
        quant_format: str,
        sparsity_scheme: Optional[SparsityCompressionConfig],
        input_tfm: Optional[tuple[TransformScheme, TransformArgs]],
        output_tfm: Optional[tuple[TransformScheme, TransformArgs]],
        quant_config: QuantizationConfig,  # used for cutlass_24
) -> Optional[CompressedTensorsScheme]:
    if weight_quant is None:
        logger.warning_once("Acceleration for non-quantized schemes is "
                            "not supported by Compressed Tensors. "
                            "Falling back to UnquantizedLinearMethod")
        return None

    if supports_cutlass_24(weight_quant=weight_quant,
                           input_quant=input_quant,
                           sparsity_scheme=sparsity_scheme):
        # Have a valid sparsity scheme
        # Validate layer is supported by Cutlass 2:4 Kernel
        config = (None if sparsity_scheme is None
                  or sparsity_scheme.format == "dense" else quant_config)

        return CompressedTensors24(
            quantized=weight_quant is not None or input_quant is not None,
            weight_quant=weight_quant,
            input_quant=input_quant,
            config=config,
        )

    # Detect If Mixed Precision
    if is_fp4a16_nvfp4(weight_quant, input_quant):
        return CompressedTensorsW4A16Fp4()

    if is_wNa16_group_channel(weight_quant, input_quant):
        if (quant_format == CompressionFormat.marlin_24.value
                and weight_quant.num_bits in W4A16SPARSE24_SUPPORTED_BITS):
            assert weight_quant.symmetric
            return CompressedTensorsW4A16Sparse24(
                strategy=weight_quant.strategy,
                num_bits=weight_quant.num_bits,
                group_size=weight_quant.group_size)
        if (quant_format == CompressionFormat.pack_quantized.value
                and weight_quant.num_bits in WNA16_SUPPORTED_BITS):
            return CompressedTensorsWNA16(num_bits=weight_quant.num_bits,
                                          strategy=weight_quant.strategy,
                                          symmetric=weight_quant.symmetric,
                                          group_size=weight_quant.group_size,
                                          actorder=weight_quant.actorder)

    if is_activation_quantization_format(quant_format):
        if is_fp4a4_nvfp4(weight_quant, input_quant):
            if cutlass_fp4_supported() or envs.VLLM_USE_NVFP4_CT_EMULATIONS:
                return CompressedTensorsW4A4Fp4()
            else:
                logger.warning_once(
                    "Current platform does not support cutlass NVFP4."
                    " Running CompressedTensorsW4A16Fp4.")
                return CompressedTensorsW4A16Fp4(has_input_global_scale=True)

        if is_fp8_w8a8(weight_quant, input_quant):
            is_fp8_w8a8_supported = CompressedTensorsScheme.check_scheme_supported(  # noqa: E501
                CompressedTensorsW8A8Fp8.get_min_capability(),
                error=False)
            if is_fp8_w8a8_supported:
                return CompressedTensorsW8A8Fp8(
                    strategy=weight_quant.strategy,
                    is_static_input_scheme=bool(input_quant
                                                and not input_quant.dynamic))
            else:
                # note: input_quant will be present for converted models;
                # will be ignored during inference post loading
                return CompressedTensorsW8A16Fp8(
                    strategy=weight_quant.strategy,
                    is_static_input_scheme=bool(not input_quant.dynamic))

        # note: input_quant can be None
        if is_fp8_w8a16(weight_quant, input_quant):
            is_static_input_scheme = bool(input_quant
                                          and not input_quant.dynamic)
            return CompressedTensorsW8A16Fp8(
                strategy=weight_quant.strategy,
                is_static_input_scheme=is_static_input_scheme)

        if is_static_tensor_w8a8(weight_quant, input_quant):
            return CompressedTensorsW8A8Int8(
                strategy=weight_quant.strategy,
                is_static_input_scheme=True,
                input_symmetric=input_quant.symmetric)

        if is_dynamic_token_w8a8(weight_quant, input_quant):
            return CompressedTensorsW8A8Int8(
                strategy=weight_quant.strategy,
                is_static_input_scheme=False,
                input_symmetric=input_quant.symmetric)

        if is_dynamic_token_w4a8_int(weight_quant, input_quant):
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
