from typing import Any, Dict, List, Literal, Optional, cast

import torch
from compressed_tensors.config import (CompressionFormat,
                                       SparsityCompressionConfig,
                                       SparsityStructure)
from compressed_tensors.quantization import (QuantizationArgs,
                                             QuantizationStrategy,
                                             QuantizationType)
from pydantic import BaseModel

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    W4A16SPARSE24_SUPPORTED_BITS, WNA16_SUPPORTED_BITS, CompressedTensors24,
    CompressedTensorsScheme, CompressedTensorsW4A16Sparse24,
    CompressedTensorsW8A8Fp8, CompressedTensorsW8A8Int8,
    CompressedTensorsW8A16Fp8, CompressedTensorsWNA16)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target, is_activation_quantization_format,
    should_ignore_layer)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.platforms import current_platform

__all__ = ["CompressedTensorsLinearMethod"]

SPARSITY_CONFIG_NAME: Literal["sparsity_config"] = "sparsity_config"
QUANTIZATION_SCHEME_MAP_TYPE = Dict[str, Optional[Dict[str, QuantizationArgs]]]


class CompressedTensorsConfig(QuantizationConfig):

    def __init__(
        self,
        target_scheme_map: Dict[str, Any],
        ignore: List[str],
        quant_format: str,
        sparsity_scheme_map: Dict[str, SparsityCompressionConfig],
        kv_cache_scheme: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):

        self.ignore = ignore
        self.quant_format = quant_format
        # Map from [target -> scheme]
        self.target_scheme_map = target_scheme_map
        self.kv_cache_scheme = kv_cache_scheme
        self.sparsity_scheme_map = sparsity_scheme_map
        self.config = config

    def get_linear_method(self) -> "CompressedTensorsLinearMethod":
        return CompressedTensorsLinearMethod(self)

    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def get_name(self) -> str:
        return "compressed_tensors"

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        # Check if the layer is skipped for quantization.
        # TODO (@robertgshaw2): support module names
        if should_ignore_layer(prefix, ignore=self.ignore):
            return UnquantizedLinearMethod()
        if isinstance(layer, LinearBase):
            scheme = self.get_scheme(layer=layer, layer_name=prefix)
            layer.scheme = scheme
            return CompressedTensorsLinearMethod(self)
        if isinstance(layer, Attention):
            return CompressedTensorsKVCacheMethod(self)
        if isinstance(layer, FusedMoE):
            return CompressedTensorsMoEMethod.get_moe_method(self)
        return None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CompressedTensorsConfig":
        ignore: List[str] = cast(List[str], config.get("ignore", []))
        quant_format = cast(str, config.get("format"))
        target_scheme_map = cls._quantization_scheme_map_from_config(
            config=config)
        sparsity_scheme_map = cls._sparsity_scheme_map_from_config(
            config=config)

        return cls(
            target_scheme_map=target_scheme_map,
            ignore=ignore,
            quant_format=quant_format,
            sparsity_scheme_map=sparsity_scheme_map,
            config=config,
        )

    @classmethod
    def _sparsity_scheme_map_from_config(
            cls, config: Dict[str,
                              Any]) -> Dict[str, SparsityCompressionConfig]:
        """
        :param config: The `quantization_config` dictionary from config.json
        :return: A dictionary mapping target layer names to their corresponding
            sparsity compression configurations
        """
        if (sparsity_config := config.get(SPARSITY_CONFIG_NAME)) is None:
            return dict()

        sparsity_config = SparsityCompressionConfig.model_validate(
            sparsity_config)
        sparse_scheme_map: Dict[str, SparsityCompressionConfig] = {
            target: sparsity_config
            for target in sparsity_config.targets or list()
        }
        return sparse_scheme_map

    @classmethod
    def _quantization_scheme_map_from_config(
            cls, config: Dict[str, Any]) -> QUANTIZATION_SCHEME_MAP_TYPE:
        """
        :param config: The `quantization_config` dictionary from config.json
        :return: A dictionary mapping target layer names to their corresponding
            quantization_args for weights and input activations
        """
        target_scheme_map: Dict[str, Any] = dict()
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
                target_scheme_map[target][
                    "weights"] = QuantizationArgs.model_validate(
                        quant_config.get("weights"))

                target_scheme_map[target]["input_activations"] = None
                if is_activation_quantization_format(quant_format):
                    input_activations = quant_config.get("input_activations")
                    # The only case where we have activation quant supported
                    # but no input_activations provided in the config
                    # should be w8a16fp8 w8a16fp8 can also run for cases where
                    # there is an input_quant but it is ignored
                    if not input_activations:
                        assert target_scheme_map[target][
                            "weights"].type == QuantizationType.FLOAT
                    else:
                        target_scheme_map[target][
                            "input_activations"] = QuantizationArgs.model_validate(  # noqa: E501
                                quant_config.get("input_activations"))
        return target_scheme_map

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    def _check_scheme_supported(self,
                                min_capability: int,
                                error: bool = True) -> bool:
        capability_tuple = current_platform.get_device_capability()

        if capability_tuple is not None:
            capability = capability_tuple.to_int()
            supported = capability >= min_capability
            if error and not supported:
                raise RuntimeError(
                    "Quantization scheme is not supported for ",
                    f"the current GPU. Min capability: {min_capability}. ",
                    f"Current capability: {capability}.")
            return supported
        else:
            return False

    def _is_static_tensor_w8a8(self, weight_quant: BaseModel,
                               input_quant: BaseModel) -> bool:
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

    def _is_dynamic_token_w8a8(self, weight_quant: BaseModel,
                               input_quant: BaseModel) -> bool:
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

    def _is_fp8_w8a8(self, weight_quant: BaseModel,
                     input_quant: BaseModel) -> bool:
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

    def _is_fp8_w8a16(self, weight_quant: BaseModel,
                      input_quant: BaseModel) -> bool:
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

    def _is_wNa16_group_channel(self, weight_quant: BaseModel,
                                input_quant: BaseModel) -> bool:
        input_quant_none = input_quant is None
        is_symmetric = weight_quant.symmetric
        is_channel_group = (
            weight_quant.strategy == QuantizationStrategy.CHANNEL.value
            or weight_quant.strategy == QuantizationStrategy.GROUP.value)
        is_static = not weight_quant.dynamic

        return (is_channel_group and input_quant_none and is_symmetric
                and is_static)

    def _get_scheme_from_parts(
            self, weight_quant: BaseModel,
            input_quant: BaseModel) -> "CompressedTensorsScheme":

        # Detect If Mixed Precision
        if self._is_wNa16_group_channel(weight_quant, input_quant):
            if (self.quant_format == CompressionFormat.marlin_24.value
                    and weight_quant.num_bits in W4A16SPARSE24_SUPPORTED_BITS):
                return CompressedTensorsW4A16Sparse24(
                    strategy=weight_quant.strategy,
                    num_bits=weight_quant.num_bits,
                    group_size=weight_quant.group_size)
            if (self.quant_format == CompressionFormat.pack_quantized.value
                    and weight_quant.num_bits in WNA16_SUPPORTED_BITS):
                return CompressedTensorsWNA16(
                    num_bits=weight_quant.num_bits,
                    strategy=weight_quant.strategy,
                    group_size=weight_quant.group_size,
                    actorder=weight_quant.actorder)

        if is_activation_quantization_format(self.quant_format):
            if self._is_fp8_w8a8(weight_quant, input_quant):
                is_fp8_w8a8_supported = self._check_scheme_supported(
                    CompressedTensorsW8A8Fp8.get_min_capability(), error=False)
                if is_fp8_w8a8_supported:
                    return CompressedTensorsW8A8Fp8(
                        strategy=weight_quant.strategy,
                        is_static_input_scheme=(input_quant
                                                and not input_quant.dynamic))
                else:
                    # note: input_quant will be present for converted models;
                    # will be ignored during inference post loading
                    return CompressedTensorsW8A16Fp8(
                        strategy=weight_quant.strategy,
                        is_static_input_scheme=not input_quant.dynamic)

            # note: input_quant can be None
            if self._is_fp8_w8a16(weight_quant, input_quant):
                is_static_input_scheme = (input_quant
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

        raise NotImplementedError(
            "No compressed-tensors compatible scheme was found.")

    def get_scheme(
            self,
            layer: torch.nn.Module,
            layer_name: Optional[str] = None) -> "CompressedTensorsScheme":
        """
        compressed-tensors supports non uniform in the following way:

        ignore: List of layer_names or nn.Module names to be ignored.
        targets of config_groups: There can be N config_groups which each
            have a quantization scheme. Each config_group has a list of targets
            which can be a full layer_name, a regex for a layer_name, or
            an nn.Module name.

        We first check whether a layer is in the ignore group and use
        CompressedTensorsUnquantized (i.e. fp16/bf16) scheme for the layer

        We then detect whether a layer_name is found in any target and
        use the quantization scheme corresponding to the matched target
        to select the CompressedTensorsScheme used for infernece.
        """

        # Find the "target" in the compressed-tensors config
        # that our layer conforms to.
        # TODO (@robertgshaw): add compressed-tensors as dep
        # so we do not have to re-write these functions
        # need to make accelerate optional in ct to do this

        # Will be empty for models with only sparsity
        if self.target_scheme_map:
            matched_target = find_matched_target(
                layer_name=layer_name,
                module=layer,
                targets=self.target_scheme_map.keys())

            scheme_dict = self.target_scheme_map[matched_target]
            weight_quant = scheme_dict.get("weights")
            input_quant = scheme_dict.get("input_activations")
        elif self.sparsity_scheme_map:
            matched_target = find_matched_target(
                layer_name=layer_name,
                module=layer,
                targets=self.sparsity_scheme_map.keys())
            weight_quant = None
            input_quant = None

        # For models with sparsity, assumes that the sparse layers are also
        # quantized for cutlass 2:4 support
        sparsity_scheme: Optional[
            SparsityCompressionConfig] = self.sparsity_scheme_map.get(
                matched_target)

        if self.supports_cutlass_24(weight_quant=weight_quant,
                                    input_quant=input_quant,
                                    sparsity_scheme=sparsity_scheme):
            # Have a valid sparsity scheme
            # Validate layer is supported by Cutlass 2:4 Kernel
            scheme = CompressedTensors24(quantized=weight_quant is not None
                                         or input_quant is not None,
                                         weight_quant=weight_quant,
                                         input_quant=input_quant)
        else:
            # Find the quant_scheme
            scheme = self._get_scheme_from_parts(  # type: ignore
                weight_quant=weight_quant,
                input_quant=input_quant,
            )

        # Raise error if device does not support the scheme
        # (e.g. fp8 needs ada lovelace)
        self._check_scheme_supported(scheme.get_min_capability())
        return scheme
    
    # move the get_compressed_tensors_cache_scale method from utils.py to instance
    # method of CompressedTensorsConfig class. By doing this, different
    # QuantizationConfig classes can implement their own get_cache_scale method.
    def get_cache_scale(self, name: str) -> Optional[List[str]]:
        """
        Check whether the param name matches the format for k/v cache scales
        in compressed-tensors. If this is the case, return its equivalent
        param name expected by vLLM

        :param name: param name
        :return: matching param name for KV cache scale in vLLM
        """
        if name.endswith(".output_scale") and ".k_proj" in name:
            return [name.replace(".k_proj.output_scale", ".attn.k_scale")]
        if name.endswith(".output_scale") and ".v_proj" in name:
            return [name.replace(".v_proj.output_scale", ".attn.v_scale")]
        # If no matches, return None
        return None

    @staticmethod
    def supports_cutlass_24(
            weight_quant: Optional[QuantizationArgs],
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
        is_valid_sparsity = (sparsity_scheme is not None
                             and sparsity_scheme.sparsity_structure
                             == SparsityStructure.TWO_FOUR.value
                             and sparsity_scheme.format == "dense")
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
                       output_partition_sizes: List[int], input_size: int,
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
    def validate_kv_cache_scheme(kv_cache_scheme: Optional[Dict[str, Any]]):
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
