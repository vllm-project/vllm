# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fractions import Fraction
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig,
    QuantizationMethods,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.transformers_utils.config import get_safetensors_params_metadata

from .config_parser import INCConfigParser

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)


class INCConfig(QuantizationConfig):
    """Config class for Intel Neural Compressor (INC).
    Repo: https://github.com/intel/neural-compressor
    """

    SUPPORTED_BITS = {2, 3, 4, 8}
    SUPPORTED_DTYPES = {"int"}
    SUPPORTED_FORMATS = {"auto_round:auto_gptq", "auto_round:auto_awq"}
    SUPPORTED_BACKENDS = {
        "auto",
        "gptq",
        "gptq:marlin",
        "awq",
        "awq:marlin",
        "marlin",
    }

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        sym: bool = True,
        packing_format: str = "auto_round:auto_gptq",
        block_name_to_quantize: str | list[str] | None = None,
        extra_config: dict[str, Any] | None = None,
        data_type: str = "int",
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if weight_bits not in self.SUPPORTED_BITS:
            raise ValueError(
                f"Unsupported weight_bits: {weight_bits}, "
                f"currently only support {self.SUPPORTED_BITS}."
            )
        if data_type not in self.SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported data_type: {data_type},"
                f" currently only support  {self.SUPPORTED_DTYPES}."
            )
        if packing_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported packing_format: {packing_format}, "
                f"currently only support {self.SUPPORTED_FORMATS}."
            )
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend: {backend},  "
                f"currently only support {self.SUPPORTED_BACKENDS}."
            )

        self.weight_bits = weight_bits
        self.group_size = group_size
        self.sym = sym
        self.packing_format = packing_format
        self.block_name_to_quantize = (
            block_name_to_quantize.split(",")
            if isinstance(block_name_to_quantize, str)
            else block_name_to_quantize
        )
        self.extra_config = extra_config
        self.data_type = data_type
        self.backend = backend
        self.pack_factor = Fraction(32, weight_bits)
        self.config_parser = INCConfigParser(self)

        # Hybrid INT4+FP8: populated by maybe_update_config when the checkpoint
        # contains FP8 layers alongside INT4-quantized layers.
        self.fp8_config: Fp8Config | None = None
        self.fp8_layers: set[str] = set()

    def __repr__(self) -> str:
        return (
            f"INCConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, sym={self.sym})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "inc"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "INCConfig":
        return cls(
            weight_bits=cls.get_from_keys(config, ["bits"]),
            group_size=cls.get_from_keys(config, ["group_size"]),
            sym=cls.get_from_keys(config, ["sym"]),
            packing_format=cls.get_from_keys_or(
                config, ["packing_format"], "auto_round:auto_gptq"
            ),
            block_name_to_quantize=cls.get_from_keys_or(
                config, ["block_name_to_quantize", "to_quant_block_names"], None
            ),
            extra_config=cls.get_from_keys_or(config, ["extra_config"], None),
            data_type=cls.get_from_keys_or(config, ["data_type"], "int"),
            backend=cls.get_from_keys_or(config, ["backend", "vllm_backend"], "auto"),
        )

    def get_layer_config(self, layer, layer_name: str):
        return self.config_parser.get_layer_config(layer, layer_name)

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.block_name_to_quantize is not None:
            self.block_name_to_quantize = hf_to_vllm_mapper.apply_list(
                self.block_name_to_quantize
            )
        if self.extra_config is not None:
            self.extra_config = hf_to_vllm_mapper.apply_dict(self.extra_config)
        if self.fp8_layers:
            self.fp8_layers = set(
                hf_to_vllm_mapper.apply_list(list(self.fp8_layers))
            )

    def maybe_update_config(self, model_name: str, hf_config=None, revision: str | None = None):
        """Detect FP8 layers in hybrid INT4+FP8 AutoRound checkpoints.

        Some AutoRound checkpoints quantize expert FFN layers to INT4 while
        leaving attention and shared-expert layers in FP8 with per-block
        ``weight_scale_inv`` scales.  The base ``INCConfig`` has no way to
        know this from ``quantization_config.json`` alone, so we scan the
        safetensors metadata here and configure an ``Fp8Config`` for those
        layers so that ``Fp8LinearMethod`` is used instead of
        ``UnquantizedLinearMethod``.
        """
        metadata = get_safetensors_params_metadata(model_name, revision=revision)
        fp8_weights: dict[str, dict[str, Any]] = {}
        for param_name, info in metadata.items():
            dtype_str = info.get("dtype", None)
            if dtype_str is None:
                continue
            torch_dtype = _SAFETENSORS_TO_TORCH_DTYPE.get(dtype_str)
            if torch_dtype == torch.float8_e4m3fn and param_name.endswith(".weight"):
                scale_name = param_name.replace(".weight", ".weight_scale_inv")
                if scale_name in metadata:
                    fp8_weights[param_name] = info

        if not fp8_weights:
            return

        block_size = None
        for param_name, info in fp8_weights.items():
            scale_name = param_name.replace(".weight", ".weight_scale_inv")
            scale_info = metadata[scale_name]
            w_shape = info.get("shape", [])
            s_shape = scale_info.get("shape", [])
            if len(w_shape) == 2 and len(s_shape) == 2:
                block_size = [w_shape[0] // s_shape[0], w_shape[1] // s_shape[1]]
                break

        if block_size is None:
            return

        self.fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=block_size,
        )
        self.fp8_layers = {name.rsplit(".weight", 1)[0] for name in fp8_weights}
        logger.info(
            "INC hybrid checkpoint: detected %d FP8 layers (block_size=%s)",
            len(self.fp8_layers),
            block_size,
        )

    def _is_layer_fp8(self, prefix: str) -> bool:
        """Check if a layer prefix belongs to the FP8 set in a hybrid checkpoint."""
        if not self.fp8_layers:
            return False
        if prefix in self.fp8_layers:
            return True
        fused_mapping = getattr(self, "packed_modules_mapping", {})
        proj_name = prefix.split(".")[-1]
        if proj_name in fused_mapping:
            shard_prefixes = [
                prefix.replace(proj_name, shard)
                for shard in fused_mapping[proj_name]
            ]
            return all(
                any(fp8_layer in sp for fp8_layer in self.fp8_layers)
                for sp in shard_prefixes
            )
        return any(fp8_layer in prefix for fp8_layer in self.fp8_layers)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from .schemes.factory import resolve_scheme

        # Match original: check model.-prefixed names for unquantized layers
        if prefix and self.extra_config:
            for layer_name in self.extra_config:
                if (
                    layer_name == prefix or layer_name == f"model.{prefix}"
                ) and self.extra_config[layer_name].get("bits", 16) >= 16:
                    if isinstance(layer, RoutedExperts):
                        return UnquantizedFusedMoEMethod(layer.moe_config)
                    return UnquantizedLinearMethod()

        layer_config = self.config_parser.resolve(layer, prefix)
        if not layer_config.quantized:
            # Hybrid checkpoint: dispatch Fp8LinearMethod for layers detected
            # as FP8 by maybe_update_config.
            if self.fp8_config and self._is_layer_fp8(prefix):
                return Fp8LinearMethod(self.fp8_config)
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            if isinstance(layer, RoutedExperts):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return None

        logger.debug(
            "[%s] Type: %s, Bits: %s, Group Size: %s, Sym: %s",
            prefix,
            layer.__class__.__name__,
            layer_config.bits,
            layer_config.group_size,
            layer_config.sym,
        )

        scheme = resolve_scheme(layer_config)
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            return scheme.get_linear_method(self, layer, prefix, layer_config)
        if isinstance(layer, RoutedExperts):
            return scheme.get_moe_method(self, layer, prefix, layer_config)
        return None

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> "QuantizationMethods | None":
        """Override the `auto-round` method to `inc`."""
        is_auto_round_format = hf_quant_cfg.get("quant_method", None) == "auto-round"
        if is_auto_round_format:
            return cls.get_name()
        return None
