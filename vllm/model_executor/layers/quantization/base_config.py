# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import regex as re
import torch
from torch import nn
from transformers import PretrainedConfig

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods
    from vllm.model_executor.models.utils import WeightsMapper
else:
    QuantizationMethods = str


class QuantizeMethodBase(ABC):
    """Base class for different quantized methods."""

    uses_meta_device: bool = False
    """
    Whether this method creates weights on meta device for online quantization.
    When True, weights are created on meta device and quantized layer-wise
    in process_weights_after_loading, reducing peak memory during loading.
    """

    @abstractmethod
    def create_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ):
        """Create weights for a layer.

        The weights will be set as attributes of the layer."""
        raise NotImplementedError

    @abstractmethod
    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        """Apply the weights in layer to the input tensor.

        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError

    # Not required functions
    def embedding(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        """Gather embeddings in the layer based on indices in the input tensor.

        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError

    # Not required functions
    def tie_weights(self, layer: torch.nn.Module, embed_tokens: torch.nn.Module):
        """Tie ``layer``'s weight to ``embed_tokens``' weight.

        The default shares the weight tensor, which is the standard behavior for
        tied word embeddings and matches what ``ParallelLMHead.tie_weights`` did
        directly before quantization methods became responsible for it.
        Quantization methods that need special weight handling (e.g. repacked
        weights) override this.

        Expects create_weights to have been called before on the layer."""
        layer.weight = embed_tokens.weight
        return layer

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Process the weight after loading.

        This can be used for example, to transpose weights for computation.
        """
        return


def method_has_implemented_embedding(method_class: type[QuantizeMethodBase]) -> bool:
    """
    Not all quant methods have embedding implemented, so we need to check that
    it exists for our given method. We check this by making sure the function
    has been changed from the base implementation.
    """
    base_embedding = inspect.getattr_static(QuantizeMethodBase, "embedding", None)
    class_embedding = inspect.getattr_static(method_class, "embedding", None)

    return class_embedding is not None and class_embedding is not base_embedding


class QuantizationConfig(ABC):
    """Base class for quantization configs."""

    _ignore_unexpected_suffixes = (
        ".q_scale",
        ".k_scale",
        ".v_scale",
        ".q_zero_point",
        ".k_zero_point",
        ".v_zero_point",
    )
    """Suffixes of quantization parameters that may be present in the checkpoint but
    not in the model, and should be ignored if unexpected during loading. These are used
    after remapping, so should be in vLLM format (e.g. .q_scale, not .q.scale)."""

    def __init__(self):
        super().__init__()
        # mapping is updated by models as they initialize
        self.packed_modules_mapping: dict[str, list[str]] = dict()

    @abstractmethod
    def get_name(self) -> QuantizationMethods:
        """Name of the quantization method."""
        raise NotImplementedError

    @abstractmethod
    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        """List of supported activation dtypes."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        """Minimum GPU capability to support the quantization method.

        E.g., 70 for Volta, 75 for Turing, 80 for Ampere.
        This requirement is due to the custom CUDA kernels used by the
        quantization method.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_config_filenames() -> list[str]:
        """List of filenames to search for in the model directory."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> "QuantizationConfig":
        """Create a config class from the model's quantization config."""
        raise NotImplementedError

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> QuantizationMethods | None:
        """
        Detects if this quantization method can support a given checkpoint
        format by overriding the user specified quantization method --
        this method should only be overwritten by subclasses in exceptional
        circumstances.

        Args:
            hf_quant_cfg: The checkpoint's quantization config dict.
            user_quant: The user-specified quantization method string.
            hf_config: The HuggingFace model config object (e.g. for
                model_type checks). May be None if not available.
        """
        return None

    @staticmethod
    def get_from_keys(config: dict[str, Any], keys: list[str]) -> Any:
        """Get a value from the model's quantization config."""
        for key in keys:
            if key in config:
                return config[key]
        raise ValueError(
            f"Cannot find any of {keys} in the model's quantization config."
        )

    @staticmethod
    def get_from_keys_or(config: dict[str, Any], keys: list[str], default: Any) -> Any:
        """Get an optional value from the model's quantization config."""
        try:
            return QuantizationConfig.get_from_keys(config, keys)
        except ValueError:
            return default

    @abstractmethod
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        """Get the quantize method to use for the quantized layer.

        Args:
            layer: The layer for the quant method.
            prefix: The full name of the layer in the state dict
        Returns:
            The quantize method. None if the given layer doesn't support quant
            method.
        """
        raise NotImplementedError

    @staticmethod
    def get_cache_scale_mapper() -> "WeightsMapper":
        """Mapping from checkpoint KV-cache scale names to vLLM scale names.

        Returning a mapper here causes `AutoWeightsLoader` to apply it to the
        weight stream automatically; individual model `load_weights` methods
        do not need to know about KV-cache scales.
        """
        from vllm.model_executor.models.utils import WeightsMapper

        orig_to_new_regex = {
            # Deprecated fused kv_scale -> attn.k_scale
            re.compile(r"\.kv_scale$"): r".attn.k_scale",
            # ModelOpt: .self_attn.{k,v}_proj.{k,v}_scale -> .self_attn.attn.*
            re.compile(r"\.self_attn\.[kv]_proj\.([kv])_scale$"): (
                r".self_attn.attn.\1_scale"
            ),
            # Fused QKV / qkqkv proj: .self_attn.qk(qk)v_proj.{k,v}_scale -> attn
            re.compile(r"\.self_attn\.qk(?:qk)?v_proj\.([kv])_scale$"): (
                r".self_attn.attn.\1_scale"
            ),
            # NemotronH: .mixer.{k,v}_proj.{k,v}_scale -> .mixer.attn.*
            re.compile(r"\.mixer\.[kv]_proj\.([kv])_scale$"): r".mixer.attn.\1_scale",
            # HYV3: .self_attn.q.scale -> .self_attn.attn.q_scale
            re.compile(r"\.self_attn\.q\.scale$"): r".self_attn.attn.q_scale",
            # HYV3: .self_attn.{k,v}_cache.scale -> .self_attn.attn.{k,v}_scale
            re.compile(r"\.self_attn\.([kv])_cache\.scale$"): (
                r".self_attn.attn.\1_scale"
            ),
            # Default: .{q,k,v}_scale -> .attn.{q,k,v}_scale (unless already .attn)
            re.compile(r"(?<!\.attn)\.([qkv])_scale$"): r".attn.\1_scale",
            re.compile(r"(?<!\.attn)\.([qkv])_zero_point$"): r".attn.\1_zero_point",
        }
        return WeightsMapper(orig_to_new_regex=orig_to_new_regex)

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
        # TODO (@kylesayrs): add implementations for all subclasses
        pass

    def maybe_update_config(  # noqa: B027
        self,
        model_name: str,
        hf_config: PretrainedConfig | None = None,
        revision: str | None = None,
    ):
        """
        Interface to update values after config initialization.

        Args:
            model_name: The name of the model
            hf_config: The Hugging Face config of the model
            revision: The revision of the model
        Returns:
        """
        # TODO: revision is never passed currently in vllm.py,
        # but is used in subclasses, should we remove this parameter?
        pass

    def is_mxfp4_quant(self, prefix: str, layer: torch.nn.Module) -> bool:
        """
        Determine if mxfp4 quantization will be used for this config.

        This allows hidden_size rounding to happen before moe_config creation
        without needing to instantiate quant_method first.

        Args:
            prefix: The layer prefix/name in the model
            layer: The layer module

        Returns:
            True if this config uses MXFP4 quantization, False otherwise
        """
        return False
