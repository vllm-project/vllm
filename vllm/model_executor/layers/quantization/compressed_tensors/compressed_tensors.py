# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any, Optional

import torch
from compressed_tensors import KV_CACHE_SCHEME_NAME, SPARSITY_CONFIG_NAME
from compressed_tensors.config import SparsityCompressionConfig
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization import QuantizationConfig as CTQuantConfig
from compressed_tensors.transform import TransformConfig

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_kv import (  # noqa: E501
    CompressedTensorsKVCacheMethod)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_linear import (  # noqa: E501
    CompressedTensorsLinearMethod)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

__all__ = ["CompressedTensorsLinearMethod"]


class CompressedTensorsConfig(QuantizationConfig):
    quant_config: CTQuantConfig
    sparsity_config: Optional[SparsityCompressionConfig]
    transform_config: Optional[TransformConfig]
    kv_cache_scheme: Optional[QuantizationArgs]

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        q_config = config.copy()
        s_config = q_config.pop(SPARSITY_CONFIG_NAME, None)
        t_config = None  #q_config.pop(TRANSFORM_CONFIG_NAME, None)
        kv_scheme = q_config.pop(KV_CACHE_SCHEME_NAME, None)

        self.quant_config = CTQuantConfig.model_validate(q_config)
        self.sparsity_config = SparsityCompressionConfig.model_validate(
            s_config) if s_config else None
        self.transform_config = TransformConfig.model_validate(
            t_config) if t_config else None
        self.kv_cache_scheme = QuantizationArgs.model_validate(
            kv_scheme) if kv_scheme else None

    # NOTE: `QuantizationConfig.get_linear_method` doesn't
    # seem to be used and doesn't really seem to make sense.
    # Leave as undefined (as many qconfigs do)

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
            return CompressedTensorsLinearMethod.get_linear_method(
                self, layer=layer, layer_name=prefix)
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

    # base QuantizationConfig, needs to stay here
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
