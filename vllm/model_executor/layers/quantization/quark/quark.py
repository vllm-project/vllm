# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fnmatch
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import torch
from transformers import PretrainedConfig

from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.quark.quark_moe import (  # noqa: E501
    QuarkMoEMethod,
)
from vllm.model_executor.layers.quantization.quark.schemes import (
    QuarkNVFP4,
    QuarkOCP_MX,
    QuarkScheme,
    QuarkW4A8_MXFP4_FP8,
    QuarkW8A8Fp8,
    QuarkW8A8Fp8PerBlock,
    QuarkW8A8Int8,
)
from vllm.model_executor.layers.quantization.quark.utils import (
    QuarkQTensorHint,
    deep_compare,
    should_ignore_layer,
)
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    _ACTIVATION_QUANT_KEY_MAP,
    _WEIGHT_QUANT_KEY_MAP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockE8M0Sym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kInt4W4A8StaticChannelSym,
    kInt8DynamicTensorAsym,
    kInt8DynamicTensorSym,
    kInt8DynamicTokenAsym,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
    kInt8StaticTensorAsym,
    kInt8StaticTensorSym,
    kMxfp4Static,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

__all__ = ["QuarkLinearMethod"]

logger = init_logger(__name__)

# model_type values that use dynamic MXFP4 re-quantization for
# OCP MX fp4 Quark checkpoints
_DEEPSEEK_V3_FAMILY_MODEL_TYPES = frozenset({"deepseek_v3", "deepseek_v32"})


class QuantKeyMatch(NamedTuple):
    matches: bool
    activation_quant_key: QuantKey | None
    weight_quant_key: QuantKey | None

    def __bool__(self) -> bool:
        return self.matches


class QuarkConfig(QuantizationConfig):
    def __init__(
        self,
        quant_config: dict[str, Any],
        kv_cache_group: list[str] | None = None,
        kv_cache_config: dict[str, Any] | None = None,
        pack_method: str = "reorder",
    ):
        super().__init__()
        if kv_cache_group is None:
            kv_cache_group = []
        self.quant_config = quant_config
        # Copy the class-level default (which a subclass may override, e.g. the
        # DeepSeek-V4 Quark config) so per-instance edits don't mutate the class.
        # Read from the class rather than ``self`` because the base
        # ``QuantizationConfig.__init__`` sets an empty instance-level
        # ``packed_modules_mapping`` that would otherwise shadow the override.
        self.packed_modules_mapping = dict(
            getattr(type(self), "packed_modules_mapping", {})
        )
        self.kv_cache_group = kv_cache_group
        self.kv_cache_config = kv_cache_config
        self.pack_method = pack_method
        # Note : this flag is kept disabled because the overhead of
        # dynamic mxfp4 quantization negates the performance gains
        # that come from shifting to mxfp4. It is left here in case
        # we want to re-enable it in the future.
        self.dynamic_mxfp4_quant = False

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: PretrainedConfig | None = None,
        revision: str | None = None,
    ):
        """Enable dynamic MXFP4 only for DeepSeek-V3-family fp4 checkpoints."""

        if hf_config is None:
            return

        if (
            getattr(hf_config, "model_type", None)
            not in _DEEPSEEK_V3_FAMILY_MODEL_TYPES
        ):
            return

        quant_config = getattr(hf_config, "quantization_config", None)
        if isinstance(quant_config, dict):
            quant_dtype = (
                quant_config.get("global_quant_config", {})
                .get("weight", {})
                .get("dtype")
            )
            if quant_dtype == "fp4":
                self.dynamic_mxfp4_quant = True

    def get_linear_method(self) -> "QuarkLinearMethod":
        return QuarkLinearMethod(self)

    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def get_name(self) -> QuantizationMethods:
        return "quark"

    def apply_vllm_mapper(  # noqa: B027
        self, hf_to_vllm_mapper: "WeightsMapper"
    ):
        """
        Interface for models to update module names referenced in
        quantization configs in order to reflect the vllm model structure

        Args:
            hf_to_vllm_mapper: maps from hf model structure (the assumed
                structure of the qconfig) to vllm model structure
        """
        quant_config_with_hf_to_vllm_mapper: dict[str, Any] = {}

        for k, v in self.quant_config.items():
            if isinstance(v, list):
                quant_config_with_hf_to_vllm_mapper[k] = hf_to_vllm_mapper.apply_list(v)
            elif isinstance(v, dict):
                quant_config_with_hf_to_vllm_mapper[k] = hf_to_vllm_mapper.apply_dict(v)
            else:
                if isinstance(v, str):
                    mapped_v_list = hf_to_vllm_mapper.apply_list([v])
                    if mapped_v_list:
                        quant_config_with_hf_to_vllm_mapper[k] = mapped_v_list[0]
                else:
                    quant_config_with_hf_to_vllm_mapper[k] = v

        self.quant_config = quant_config_with_hf_to_vllm_mapper

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        weight_quant_key, activation_quant_key, method_cls = (
            self.get_quant_method_target(prefix, type(layer))
        )

        exclude_layers = cast(list[str], self.quant_config.get("exclude"))
        is_ignored = should_ignore_layer(
            prefix,
            ignore=exclude_layers,
            fused_mapping=self.packed_modules_mapping,
            check_children=isinstance(layer, RoutedExperts),
        )
        dynamic_mxfp4_quant = (
            is_ignored and "self_attn" in prefix and self.dynamic_mxfp4_quant
        )

        if method_cls is UnquantizedFusedMoEMethod:
            return UnquantizedFusedMoEMethod(layer.moe_config)
        if method_cls is UnquantizedLinearMethod:
            return UnquantizedLinearMethod()
        if method_cls is QuarkLinearMethod:
            _, _, scheme_cls = self.get_scheme_cls(type(layer), prefix)
            scheme = self.init_scheme(
                scheme_cls,
                weight_quant_key=weight_quant_key,
                activation_quant_key=activation_quant_key,
                dynamic_mxfp4_quant=dynamic_mxfp4_quant,
            )
            layer.scheme = scheme
            return QuarkLinearMethod(self)
        if method_cls is QuarkKVCacheMethod:
            return QuarkKVCacheMethod(self)
        if method_cls is not None and issubclass(method_cls, QuarkMoEMethod):
            return QuarkMoEMethod.get_moe_method(
                self,
                module=layer,
                method_cls=method_cls,
                weight_quant_key=weight_quant_key,
                activation_quant_key=activation_quant_key,
            )

        return None

    def get_quant_method_target(
        self, prefix: str, layer_type: type[torch.nn.Module]
    ) -> tuple[
        QuantKey | None,
        QuantKey | None,
        type[QuantizeMethodBase]
        | type[UnquantizedLinearMethod]
        | type[UnquantizedFusedMoEMethod]
        | None,
    ]:
        """Return weight key, activation key, and quant method class for
        the given ``prefix`` and ``layer_type``.

        This is the counterpart of ``get_quant_method`` without quant method
        instantiation.

        TODO: integrate as part of base QuantizationConfig.get_quant_method_target
        in the future.

        ``None`` denotes an unquantized/BF16 activation.
        """
        is_routed_experts = issubclass(layer_type, RoutedExperts)
        if should_ignore_layer(
            prefix,
            ignore=cast(list[str], self.quant_config.get("exclude")),
            fused_mapping=self.packed_modules_mapping,
            check_children=is_routed_experts,
        ):
            if is_routed_experts:
                return None, None, UnquantizedFusedMoEMethod
            if issubclass(layer_type, LinearBase):
                if "self_attn" in prefix and self.dynamic_mxfp4_quant:
                    weight_quant_key, activation_key, _ = self.get_scheme_cls(
                        layer_type, prefix
                    )
                    return weight_quant_key, activation_key, QuarkLinearMethod
                return None, None, UnquantizedLinearMethod
            return None, None, None
        if issubclass(layer_type, LinearBase):
            weight_quant_key, activation_key, _ = self.get_scheme_cls(
                layer_type, prefix
            )
            return weight_quant_key, activation_key, QuarkLinearMethod
        if issubclass(layer_type, Attention):
            return None, None, QuarkKVCacheMethod
        if is_routed_experts:
            weight_quant_key, activation_quant_key, quant_method_cls = (
                QuarkMoEMethod.get_moe_method_target(self, layer_type, prefix)
            )
            if quant_method_cls is None:
                return None, None, None
            return weight_quant_key, activation_quant_key, quant_method_cls
        return None, None, None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QuarkConfig":
        export_config = config.get("export")
        if export_config is None:
            raise ValueError(
                "The export key should be included in "
                "the configurations of Quark quantized model"
            )
        kv_cache_group = cast(list[str], export_config.get("kv_cache_group"))
        pack_method = cast(str, export_config.get("pack_method"))

        # In the export model of quark, the quantization configuration
        # of kv_cache is stored in layer_quant_config. First, it is
        # judged whether kv_cache_group exists, and then it is judged
        # whether layer_quant_config has a quantization configuration
        # that matches kv_cache.
        if len(kv_cache_group) == 0:
            kv_cache_config = None
        else:
            kv_cache_set = set(kv_cache_group)
            layer_quant_config = cast(dict[str, Any], config.get("layer_quant_config"))
            layer_quant_names = list(layer_quant_config.keys())
            layer_quant_set = set(layer_quant_names)

            if not (
                kv_cache_set.issubset(layer_quant_set)
                or any(
                    fnmatch.fnmatchcase(layer_quant, pat)
                    for layer_quant in list(layer_quant_set)
                    for pat in list(kv_cache_set)
                )
            ):
                raise ValueError(
                    "The Quark quantized model has the "
                    "kv_cache_group parameter setting, "
                    "but no kv_cache quantization settings "
                    "were found in the quantization "
                    "configuration."
                )

            q_configs = [
                quant_cfg
                for name, quant_cfg in layer_quant_config.items()
                if any(fnmatch.fnmatchcase(name, pattern) for pattern in kv_cache_group)
            ]

            if not all(
                deep_compare(q_config["output_tensors"], q_configs[0]["output_tensors"])
                for q_config in q_configs
            ):
                raise ValueError(
                    "The quantization method used for kv_cache should "
                    "be the same, but the quantization method for the "
                    "kv_cache layer in the config is different."
                )
            kv_cache_config = q_configs[0].get("output_tensors")
            if kv_cache_config is None:
                raise ValueError("The kv_cache quantization configuration is empty.")

            # Since we have already set kv_cache quantization configurations,
            # we will remove the quantization configuration for the
            # output_tensors corresponding to the kv_cache layer.
            for q_config in q_configs:
                q_config["output_tensors"] = None

            # In case q_proj output is also quantized, remove the configuration
            # to keep qkv consistency.
            q_proj_q_config = cast(dict[str, Any], layer_quant_config.get("*q_proj"))
            if q_proj_q_config is not None:
                q_proj_q_config["output_tensors"] = None

        return cls(
            quant_config=config,
            kv_cache_group=kv_cache_group,
            kv_cache_config=kv_cache_config,
            pack_method=pack_method,
        )

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def _check_scheme_supported(self, min_capability: int, error: bool = True) -> bool:
        capability_tuple = current_platform.get_device_capability()

        if capability_tuple is not None:
            capability = capability_tuple.to_int()
            supported = capability >= min_capability
            if error and not supported:
                raise RuntimeError(
                    "Quantization scheme is not supported for the current GPU. "
                    f"Min capability: {min_capability}. "
                    f"Current capability: {capability}."
                )
            return supported
        else:
            return False

    def _is_fp8_w4a8(
        self,
        weight_quant: QuarkQTensorHint,
        input_quant: QuarkQTensorHint,
    ) -> QuantKeyMatch:
        # Confirm weights and input quantized.
        if isinstance(input_quant, list):
            if len(input_quant) != 1:
                return QuantKeyMatch(False, None, None)
            input_quant = input_quant[0]
        if (
            not isinstance(weight_quant, list)
            or len(weight_quant) != 2
            or not isinstance(input_quant, dict)
        ):
            return QuantKeyMatch(False, None, None)

        # Confirm weight scheme is supported
        is_w4a8_dtype = (
            weight_quant[0].get("dtype") == "fp8_e4m3"
            and weight_quant[1].get("dtype") == "int4"
            and input_quant.get("dtype") == "fp8_e4m3"
        )
        is_static_weight = not weight_quant[0].get("is_dynamic") and not weight_quant[
            1
        ].get("is_dynamic")
        is_per_tensor_fp8_and_per_channel_int4_weight = (
            weight_quant[0].get("qscheme") == "per_tensor"
            and weight_quant[1].get("qscheme") == "per_channel"
            and weight_quant[1].get("symmetric") is True
            and weight_quant[1].get("ch_axis") == 0
        )

        if not (
            is_w4a8_dtype
            and is_static_weight
            and is_per_tensor_fp8_and_per_channel_int4_weight
        ):
            return QuantKeyMatch(False, None, None)

        # Dynamic quantization is always supported if weights supported.
        if input_quant.get("is_dynamic"):
            return QuantKeyMatch(True, kFp8DynamicTokenSym, kInt4W4A8StaticChannelSym)

        # Confirm activation scheme is supported.
        is_per_tensor_activation = input_quant.get("qscheme") == "per_tensor"
        if not is_per_tensor_activation:
            return QuantKeyMatch(False, None, None)
        return QuantKeyMatch(True, kFp8StaticTensorSym, kInt4W4A8StaticChannelSym)

    def _is_fp8_w8a8(
        self,
        weight_quant: dict[str, Any] | None,
        input_quant: dict[str, Any] | None,
    ) -> QuantKeyMatch:
        # Confirm weights and input quantized.
        if not isinstance(weight_quant, dict) or not isinstance(input_quant, dict):
            return QuantKeyMatch(False, None, None)

        # Confirm weight scheme is supported
        is_fp8_dtype = (
            weight_quant.get("dtype") == "fp8_e4m3"
            and input_quant.get("dtype") == "fp8_e4m3"
        )
        is_static_weight = not weight_quant.get("is_dynamic")
        is_per_tensor_or_channel_weight = weight_quant.get("qscheme") in [
            "per_tensor",
            "per_channel",
        ]
        block_size = list(weight_quant.get("block_size") or [])
        is_per_block_weight = (
            weight_quant.get("qscheme") == "per_block"
            and block_size == [128, 128]
            and weight_quant.get("symmetric") is True
        )

        if not is_fp8_dtype or not is_static_weight:
            return QuantKeyMatch(False, None, None)

        if is_per_block_weight:
            if not input_quant.get("is_dynamic"):
                return QuantKeyMatch(False, None, None)
            matches = (
                input_quant.get("qscheme") == "per_group"
                and input_quant.get("group_size") == block_size[1]
                and input_quant.get("symmetric") is True
            )
            if not matches:
                return QuantKeyMatch(False, None, None)
            weight_quant_key = (
                kFp8Static128BlockE8M0Sym
                if weight_quant.get("scale_type") == "float8_e8m0fnu"
                else kFp8Static128BlockSym
            )
            return QuantKeyMatch(True, kFp8Dynamic128Sym, weight_quant_key)

        if not is_per_tensor_or_channel_weight:
            return QuantKeyMatch(False, None, None)

        # Dynamic quantization is always supported if tensor/channel weights
        # are supported.
        if input_quant.get("is_dynamic"):
            activation_quant_key = (
                kFp8DynamicTokenSym
                if input_quant.get("qscheme") == "per_channel"
                else kFp8DynamicTensorSym
            )
            weight_quant_key = (
                kFp8StaticChannelSym
                if weight_quant.get("qscheme") == "per_channel"
                else kFp8StaticTensorSym
            )
            return QuantKeyMatch(True, activation_quant_key, weight_quant_key)

        # Confirm activation scheme is supported.
        is_per_tensor_activation = input_quant.get("qscheme") == "per_tensor"
        if not is_per_tensor_activation:
            return QuantKeyMatch(False, None, None)
        weight_quant_key = (
            kFp8StaticChannelSym
            if weight_quant.get("qscheme") == "per_channel"
            else kFp8StaticTensorSym
        )
        return QuantKeyMatch(True, kFp8StaticTensorSym, weight_quant_key)

    def _is_w8a8_int8(
        self,
        weight_quant: dict[str, Any] | None,
        input_quant: dict[str, Any] | None,
    ) -> QuantKeyMatch:
        # Confirm weights and input quantized.
        if not isinstance(weight_quant, dict) or not isinstance(input_quant, dict):
            return QuantKeyMatch(False, None, None)

        is_int8_dtype = (
            weight_quant.get("dtype") == "int8" and input_quant.get("dtype") == "int8"
        )

        is_valid_weight_scheme = weight_quant.get("qscheme") in [
            "per_tensor",
            "per_channel",
        ]
        is_static_input = input_quant.get(
            "qscheme"
        ) == "per_tensor" and not input_quant.get("is_dynamic")
        is_dynamic_input = (
            input_quant.get("qscheme") == "per_channel"
            and input_quant.get("is_dynamic") is True
        )
        is_weight_symmetric = weight_quant.get("symmetric") is True

        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        matches = (
            is_int8_dtype
            and is_valid_weight_scheme
            and is_weight_symmetric
            and (
                (is_static_input and not weight_quant.get("is_dynamic"))
                or is_dynamic_input
            )
        )
        if not matches:
            return QuantKeyMatch(False, None, None)
        weight_quant_key = (
            kInt8StaticChannelSym
            if weight_quant.get("qscheme") == "per_channel"
            else kInt8StaticTensorSym
        )
        if is_static_input:
            activation_quant_key = (
                kInt8StaticTensorSym
                if input_quant.get("symmetric") is True
                else kInt8StaticTensorAsym
            )
        elif weight_quant.get("qscheme") == "per_channel":
            activation_quant_key = (
                kInt8DynamicTokenSym
                if input_quant.get("symmetric") is True
                else kInt8DynamicTokenAsym
            )
        else:
            activation_quant_key = (
                kInt8DynamicTensorSym
                if input_quant.get("symmetric") is True
                else kInt8DynamicTensorAsym
            )
        return QuantKeyMatch(True, activation_quant_key, weight_quant_key)

    def _is_w4a8_mxfp4_fp8(
        self,
        weight_quant: dict[str, Any] | None,
        input_quant: dict[str, Any] | None,
    ) -> QuantKeyMatch:
        if weight_quant is None or input_quant is None:
            return QuantKeyMatch(False, None, None)

        is_weight_mxfp4 = (
            weight_quant.get("dtype") == "fp4"
            and weight_quant.get("qscheme") == "per_group"
            and weight_quant.get("group_size") == 32
            and weight_quant.get("scale_format") == "e8m0"
            and not weight_quant.get("is_dynamic")
        )

        is_input_fp8 = (
            input_quant.get("dtype") == "fp8_e4m3"
            and input_quant.get("qscheme") == "per_tensor"
            and not input_quant.get("is_dynamic")  # Static per-tensor
            and input_quant.get("symmetric") is True  # Symmetric quantization
        )

        if not (is_weight_mxfp4 and is_input_fp8):
            return QuantKeyMatch(False, None, None)
        return QuantKeyMatch(True, kFp8StaticTensorSym, kMxfp4Static)

    def _is_nvfp4(
        self,
        weight_quant: QuarkQTensorHint,
        input_quant: QuarkQTensorHint,
    ) -> QuantKeyMatch:
        # Confirm weights and input quantized.
        if weight_quant is None or input_quant is None:
            return QuantKeyMatch(False, None, None)

        # Confirm both weight_quant and input_quant are lists with 2 elements
        if not isinstance(weight_quant, list) or len(weight_quant) != 2:
            return QuantKeyMatch(False, None, None)
        if not isinstance(input_quant, list) or len(input_quant) != 2:
            return QuantKeyMatch(False, None, None)

        # First element should be fp4 with per_group quantization
        is_fp4_per_group_weight = (
            weight_quant[0].get("dtype") == "fp4"
            and weight_quant[0].get("qscheme") == "per_group"
            and weight_quant[0].get("group_size") == 16
            and not weight_quant[0].get("is_dynamic")
        )
        is_fp4_per_group_input = (
            input_quant[0].get("dtype") == "fp4"
            and input_quant[0].get("qscheme") == "per_group"
            and input_quant[0].get("group_size") == 16
            and input_quant[0].get("is_dynamic")
        )

        # Second element should be fp8_e4m3 with per_tensor quantization
        is_fp8_per_tensor_weight = (
            weight_quant[1].get("dtype") == "fp8_e4m3"
            and weight_quant[1].get("qscheme") == "per_tensor"
            and not weight_quant[1].get("is_dynamic")
        )
        is_fp8_per_tensor_input = (
            input_quant[1].get("dtype") == "fp8_e4m3"
            and input_quant[1].get("qscheme") == "per_tensor"
            and not input_quant[1].get("is_dynamic")
        )

        matches = (
            is_fp4_per_group_weight  # type: ignore[return-value]
            and is_fp4_per_group_input
            and is_fp8_per_tensor_weight
            and is_fp8_per_tensor_input
        )
        if not matches:
            return QuantKeyMatch(False, None, None)
        return QuantKeyMatch(True, kNvfp4Dynamic, kNvfp4Static)

    def _is_w_ocp_mx_a_x(
        self,
        weight_quant: dict[str, Any] | None,
        input_quant: dict[str, Any] | None,
        allow_static_fp8: bool = False,
    ) -> QuantKeyMatch:
        """
        This check returns True only if it is an OCP-MX weight quantization.
        The activation can be FP16/BF16, OCP MXFP4, MXFP8, FP8.
        The rationale for checking only the weight type is that
        the model loading concept and process primarily concerns the weights themselves.
        """
        # Confirm weights quantized.
        if not isinstance(weight_quant, dict):
            logger.debug(
                "Quark model's weight quantization is incompatible with OCP_MX format: "
                "weight_quant is not a dictionary."
            )
            return QuantKeyMatch(False, None, None)

        # Input and weight qscheme needs to be per group.
        if weight_quant.get("qscheme") != "per_group":
            logger.debug(
                "Quark model's weight quantization is incompatible with OCP MX format: "
                "weight is not per_group."
            )
            return QuantKeyMatch(False, None, None)

        # Input and weight group size needs to be 32.
        if weight_quant.get("group_size") != 32:
            logger.debug(
                "Quark model's weight quantization is incompatible with OCP MX format: "
                "group_size of weight is not 32."
            )
            return QuantKeyMatch(False, None, None)

        # Activations and weight scales need to be in e8m0 format.
        if weight_quant.get("scale_format") != "e8m0":
            logger.debug(
                "Quark model's weight quantization is incompatible with OCP MX format: "
                "scale_format of weight is not e8m0."
            )
            return QuantKeyMatch(False, None, None)

        # Input and weight dtypes need to be any of fp4,
        # fp6_e3m2 or fp6_e3m2, possibly mixed.
        if weight_quant.get("dtype") not in {
            "fp4",
            "fp6_e3m2",
            "fp6_e2m3",
        }:
            logger.debug(
                "Quark model's weight quantization is incompatible with OCP MX format: "
                "dtype is not in {fp4, fp6_e3m2, fp6_e2m3}."
            )
            return QuantKeyMatch(False, None, None)

        weight_dtype = weight_quant["dtype"].replace("fp", "mxfp")
        weight_quant_key = _WEIGHT_QUANT_KEY_MAP[weight_dtype]
        if input_quant is None:
            activation_quant_key = None
        elif not input_quant.get("is_dynamic"):
            if (
                allow_static_fp8
                and input_quant.get("dtype") == "fp8_e4m3"
                and input_quant.get("qscheme") == "per_tensor"
                and input_quant.get("symmetric") is True
            ):
                activation_quant_key = kFp8StaticTensorSym
            else:
                logger.debug(
                    "Quark model's OCP MX quantization is incompatible with static "
                    "input scales."
                )
                return QuantKeyMatch(False, None, None)
        elif input_quant["dtype"] == "fp8_e4m3":
            activation_quant_key = kFp8DynamicTensorSym
        else:
            input_dtype = input_quant["dtype"].replace("fp", "mxfp")
            if input_dtype not in _ACTIVATION_QUANT_KEY_MAP:
                raise ValueError(
                    f"Unsupported input_dtype={input_dtype} in Quark's vLLM "
                    "integration. Supported activation dtypes are "
                    f"{_ACTIVATION_QUANT_KEY_MAP.keys()}, or None for "
                    "weight-only quantization."
                )
            activation_quant_key = _ACTIVATION_QUANT_KEY_MAP[input_dtype]
        return QuantKeyMatch(True, activation_quant_key, weight_quant_key)

    @staticmethod
    def _unwrap_single_quant_config(
        quant_config: QuarkQTensorHint,
    ) -> dict[str, Any] | None:
        """Unwrap one-entry quantization-config lists."""
        if isinstance(quant_config, list):
            return quant_config[0] if len(quant_config) == 1 else None
        return quant_config

    def get_layer_quant_config_from_name(
        self, layer_name: str
    ) -> dict[str, Any] | None:
        proj_name = layer_name.split(".")[-1]
        if proj_name in self.packed_modules_mapping:
            shard_proj_names = self.packed_modules_mapping[proj_name]
            shard_configs = []
            for shard_proj_name in shard_proj_names:
                shard_name = layer_name.replace(proj_name, shard_proj_name)
                if shard_name != layer_name:
                    config = self.get_layer_quant_config_from_name(shard_name)
                else:
                    config = None
                shard_configs.append(config)

            matched_configs = [config for config in shard_configs if config is not None]
            if matched_configs and not all(
                deep_compare(config, matched_configs[0]) for config in matched_configs
            ):
                raise ValueError(
                    f"Found a different quantization configuration for "
                    f"{shard_proj_names} in {layer_name}. vLLM "
                    "requires all to use the same scheme."
                )
            if matched_configs:
                return matched_configs[0]
            return None
        else:
            layer_quant_config = cast(
                dict[str, Any], self.quant_config.get("layer_quant_config") or {}
            )
            for name_pattern, config in layer_quant_config.items():
                if "*" not in name_pattern:
                    matches = layer_name in name_pattern
                else:
                    matches = fnmatch.fnmatch(layer_name, name_pattern)
                if matches:
                    return config
            return None

    def _find_matched_config(
        self, layer_name: str, module: torch.nn.Module | type[torch.nn.Module]
    ) -> dict[str, Any]:
        # Priority order:
        # 1. layer_quant_config,
        # 2. layer_type_quant_config,
        # 3. global_quant_config.

        layer_type = cast(str, module if isinstance(module, type) else type(module))
        layer_type_quant_config = cast(
            dict[str, Any], self.quant_config.get("layer_type_quant_config")
        )
        global_quant_config = cast(
            dict[str, Any], self.quant_config.get("global_quant_config")
        )
        fallback_config = layer_type_quant_config.get(layer_type, global_quant_config)

        proj_name = layer_name.split(".")[-1]
        if proj_name in self.packed_modules_mapping:
            shard_proj_names = self.packed_modules_mapping[proj_name]
            shard_configs = []
            for shard_proj_name in shard_proj_names:
                shard_name = layer_name.replace(proj_name, shard_proj_name)
                if shard_name == layer_name:
                    config = fallback_config
                else:
                    config = self.get_layer_quant_config_from_name(shard_name)
                    if config is None:
                        config = fallback_config
                shard_configs.append(config)

            if not all(
                deep_compare(config, shard_configs[0]) for config in shard_configs
            ):
                raise ValueError(
                    f"Found a different quantization configuration for "
                    f"{shard_proj_names} in {layer_name}. vLLM requires all "
                    "to use the same scheme."
                )
            return shard_configs[0]
        else:
            layer_quant_config = self.get_layer_quant_config_from_name(layer_name)
            if layer_quant_config is not None:
                return layer_quant_config
            return fallback_config

    def _get_scheme_cls_from_config(
        self, config: dict[str, Any]
    ) -> tuple[QuantKey | None, QuantKey | None, type["QuarkScheme"]]:
        if config.get("output_tensors") or config.get("bias"):
            raise NotImplementedError(
                "Currently, Quark models with output_tensors "
                "and bias quantized are not supported"
            )
        weight_config = cast(QuarkQTensorHint, config.get("weight"))
        input_config = cast(QuarkQTensorHint, config.get("input_tensors"))

        if (
            isinstance(weight_config, list)
            and len(weight_config) > 1
            or isinstance(input_config, list)
            and len(input_config) > 1
        ):
            if match := self._is_nvfp4(weight_config, input_config):
                return (
                    match.weight_quant_key,
                    match.activation_quant_key,
                    QuarkNVFP4,
                )

            raise NotImplementedError(
                "Multi-entry weight or activation quantization configs are only "
                "supported for NVFP4."
            )

        weight_config = self._unwrap_single_quant_config(weight_config)
        input_config = self._unwrap_single_quant_config(input_config)

        if (
            isinstance(weight_config, dict)
            and weight_config.get("qscheme") == "per_block"
            and not weight_config.get("block_size")
        ):
            raise ValueError(
                "Quark W8A8 FP8 per-block weight quantization requires "
                "`block_size` in the weight quantization config."
            )

        if match := self._is_fp8_w8a8(weight_config, input_config):
            assert weight_config is not None
            if weight_config.get("qscheme") == "per_block":
                return (
                    match.weight_quant_key,
                    match.activation_quant_key,
                    QuarkW8A8Fp8PerBlock,
                )
            return match.weight_quant_key, match.activation_quant_key, QuarkW8A8Fp8
        elif match := self._is_w8a8_int8(weight_config, input_config):
            return (
                match.weight_quant_key,
                match.activation_quant_key,
                QuarkW8A8Int8,
            )
        elif match := self._is_w4a8_mxfp4_fp8(weight_config, input_config):
            return (
                match.weight_quant_key,
                match.activation_quant_key,
                QuarkW4A8_MXFP4_FP8,
            )
        elif match := self._is_w_ocp_mx_a_x(weight_config, input_config):
            return match.weight_quant_key, match.activation_quant_key, QuarkOCP_MX

        raise NotImplementedError(
            "No quark compatible scheme was found. "
            f"Weight config: {weight_config}, "
            f"Input config: {input_config}"
        )

    def get_scheme_cls(
        self, layer_type: type[torch.nn.Module], layer_name: str
    ) -> tuple[QuantKey | None, QuantKey | None, type["QuarkScheme"]]:
        """Return quantization keys and scheme class without initializing it."""
        return self._get_scheme_cls_from_config(
            self._find_matched_config(layer_name, layer_type)
        )

    def init_scheme(
        self,
        scheme_cls: type["QuarkScheme"],
        weight_quant_key: QuantKey | None,
        activation_quant_key: QuantKey | None,
        dynamic_mxfp4_quant: bool = False,
    ) -> "QuarkScheme":
        """Construct a Quark scheme selected by get_scheme_cls."""
        if scheme_cls not in (
            QuarkW8A8Fp8,
            QuarkW8A8Fp8PerBlock,
            QuarkW8A8Int8,
            QuarkOCP_MX,
            QuarkNVFP4,
            QuarkW4A8_MXFP4_FP8,
        ):
            raise AssertionError(f"Unsupported Quark scheme class: {scheme_cls}")

        kwargs: dict[str, Any] = {"activation_quant_key": activation_quant_key}

        # These classes handle different possible `weight_quant_key`.
        if scheme_cls in (
            QuarkW8A8Fp8,
            QuarkW8A8Fp8PerBlock,
            QuarkW8A8Int8,
            QuarkOCP_MX,
        ):
            kwargs["weight_quant_key"] = weight_quant_key

        if scheme_cls is QuarkOCP_MX:
            kwargs["dynamic_mxfp4_quant"] = dynamic_mxfp4_quant

        scheme = scheme_cls(**kwargs)

        # Raise error if device does not support the scheme
        # (e.g. fp8 needs ada lovelace)
        self._check_scheme_supported(scheme.get_min_capability())
        return scheme

    @staticmethod
    def get_cache_scale_mapper() -> "WeightsMapper":
        """Map Quark KV-cache scale names to vLLM names."""
        orig_to_new_suffix = {
            ".k_proj.output_scale": ".attn.k_scale",
            ".v_proj.output_scale": ".attn.v_scale",
            ".q_proj.output_scale": ".attn.q_scale",
            ".self_attn.prob_output_scale": ".self_attn.attn.prob_scale",
        }
        cache_scale_mapper = WeightsMapper(orig_to_new_suffix=orig_to_new_suffix)
        return cache_scale_mapper | QuantizationConfig.get_cache_scale_mapper()


class QuarkLinearMethod(LinearMethodBase):
    def __init__(self, quantization_config: QuarkConfig):
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


class QuarkKVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from quark checkpoints.
    """

    def __init__(self, quant_config: QuarkConfig):
        self.validate_kv_cache_config(quant_config.kv_cache_config)
        super().__init__(quant_config)

    @staticmethod
    def validate_kv_cache_config(kv_cache_config: dict[str, Any] | None):
        """
        Validator for the kv cache configuration. Useful for controlling the
        kv cache quantization schemes, that are being supported in vLLM

        Args:
            kv_cache_config: the quark kv cache scheme
        """
        if kv_cache_config is None:
            return

        dtype = kv_cache_config.get("dtype")
        if dtype != "fp8_e4m3":
            raise NotImplementedError(
                "Currently supported kv cache quantization is "
                f"dtype=fp8_e4m3, however received {dtype}"
            )

        qscheme = kv_cache_config.get("qscheme")
        if qscheme != "per_tensor":
            raise NotImplementedError(
                "Only support per-tensor scaling factor "
                "for quark KV cache. "
                f"Expected qscheme: per_tensor, found qscheme: {qscheme}"
            )
