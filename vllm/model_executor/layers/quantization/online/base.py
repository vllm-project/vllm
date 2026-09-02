# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

import torch

from vllm.config.quantization import (
    _ONLINE_SHORTHANDS,
    QuantizationConfigArgs,
    QuantSpec,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
    Fp8PerBlockOnlineMoEMethod,
    Fp8PerTensorOnlineLinearMethod,
    Fp8PerTensorOnlineMoEMethod,
    Fp8PtpcOnlineLinearMethod,
    Fp8PtpcOnlineMoEMethod,
    OnlineLinearBase,
)
from vllm.model_executor.layers.quantization.online.int8 import (
    Int8OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.online.mxfp4 import (
    Mxfp4OnlineLinearMethod,
    Mxfp4OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.online.mxfp8 import (
    Mxfp8OnlineLinearMethod,
    Mxfp8OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.online.nvfp4 import (
    Nvfp4OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.config_utils import (
    find_matching_patterns,
    get_layer_name_after_index,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kInt8StaticChannelSym,
    kMxfp4Static,
    kMxfp8Dynamic,
    kNvfp4Static,
)

logger = init_logger(__name__)


# Online dispatch tables, keyed by the QuantSpec.weight QuantKey. The
# corresponding method class handles the activation choice via its
# `supported_activation_quant` set.
_ONLINE_LINEAR_METHODS: dict[QuantKey, type] = {
    kFp8StaticTensorSym: Fp8PerTensorOnlineLinearMethod,
    kFp8Static128BlockSym: Fp8PerBlockOnlineLinearMethod,
    kFp8StaticChannelSym: Fp8PtpcOnlineLinearMethod,
    kMxfp8Dynamic: Mxfp8OnlineLinearMethod,
    kMxfp4Static: Mxfp4OnlineLinearMethod,
}

_ONLINE_MOE_METHODS: dict[QuantKey, type] = {
    kFp8StaticTensorSym: Fp8PerTensorOnlineMoEMethod,
    kFp8Static128BlockSym: Fp8PerBlockOnlineMoEMethod,
    kFp8StaticChannelSym: Fp8PtpcOnlineMoEMethod,
    kMxfp8Dynamic: Mxfp8OnlineMoEMethod,
    kMxfp4Static: Mxfp4OnlineMoEMethod,
    kInt8StaticChannelSym: Int8OnlineMoEMethod,
    kNvfp4Static: Nvfp4OnlineMoEMethod,
}


def _find_matching_targets(
    prefix: str,
    targets: Mapping[str, str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
) -> list[str]:
    per_shard_matches = find_matching_patterns(
        prefix, targets, fused_mapping, use_fnmatch=True
    )
    if all(len(matches) == 0 for matches in per_shard_matches):
        return []
    if any(len(matches) == 0 for matches in per_shard_matches):
        raise ValueError(
            f"Found unmatched shards for {prefix}: {per_shard_matches}. vLLM "
            "requires all shards of a fused layer to match a target."
        )
    if any(len(matches) > 1 for matches in per_shard_matches):
        raise ValueError(
            f"Found multiple quantization_config.targets matches for the "
            f"shards of {prefix}: {per_shard_matches}. Each shard may match "
            "at most one target."
        )

    matched_patterns = [next(iter(matches)) for matches in per_shard_matches]
    quant_key_strs = {targets[pattern] for pattern in matched_patterns}
    if len(quant_key_strs) > 1:
        raise ValueError(
            f"Found different quantization_config.targets values for the "
            f"shards of {prefix}: {matched_patterns}. vLLM requires all "
            "shards of a fused layer to use the same target."
        )
    return [matched_patterns[0]]


class OnlineQuantizationConfig(QuantizationConfig):
    """Model-level config for online quantization (quantize fp16/bf16 weights
    during model loading, without requiring a pre-quantized checkpoint)."""

    def __init__(
        self,
        args: QuantizationConfigArgs,
    ) -> None:
        super().__init__()
        if args.linear is None and args.moe is None and args.targets is None:
            raise ValueError(
                "OnlineQuantizationConfig requires at least one of "
                "quantization_config.linear, quantization_config.moe, or "
                "quantization_config.targets to be set."
            )
        self.args = args
        self.ignored_layers: list[str] = args.ignore
        self.quantized_layers: dict[str, tuple[str, str, str | None]] = {}

    @property
    def quantized_layer_summaries(self) -> list[str]:
        counts: dict[tuple[str, str, str | None, str], int] = {}
        for layer_name, (
            source,
            quant_key_str,
            target_pattern,
        ) in self.quantized_layers.items():
            key = (
                get_layer_name_after_index(layer_name),
                source,
                target_pattern,
                quant_key_str,
            )
            counts[key] = counts.get(key, 0) + 1

        summaries = []
        # Build summary entries as
        # `self_attn.o_proj: 24 (from targets: re:.*self_attn\.o_proj, mxfp4`
        for (layer_type, source, target_pattern, quant_key_str), count in sorted(
            counts.items()
        ):
            pattern_prefix = f"{target_pattern}, " if target_pattern else ""
            summaries.append(
                f"{layer_type}: {count} "
                f"(from {source}: {pattern_prefix}{quant_key_str})"
            )
        return summaries

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "online"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # Note: as more online quant schemes will be added, this
        # value will become the minimum across all supported schemes.
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OnlineQuantizationConfig":
        raise NotImplementedError(
            "OnlineQuantizationConfig does not support loading from a "
            "checkpoint config. Use quantization_config or "
            "quantization='fp8_per_tensor'/'fp8_per_block' instead."
        )

    def _get_method_cls(
        self,
        spec: QuantSpec | None,
        table: dict[QuantKey, type],
        layer: torch.nn.Module,
    ) -> type | None:
        if spec is None or spec.weight is None:
            return None
        cls = table.get(spec.weight)
        if cls is None:
            raise ValueError(
                f"online quantization for {type(layer).__name__} with "
                f"weight={spec.weight} is not supported; supported weight "
                f"keys: {sorted(str(k) for k in table)}"
            )
        # Online method classes pick their own activation format internally.
        # Per-class activation overrides are not yet wired through; reject
        # explicit overrides until the relevant method class opts in.
        if spec.activation is not None:
            raise ValueError(
                f"activation override (activation={spec.activation}) is not "
                f"yet supported for online {cls.__name__}"
            )
        return cls

    def get_quantization_target(
        self, layer: torch.nn.Module, prefix: str
    ) -> tuple[str, QuantSpec, type] | None:
        """Return the QuantizeMethodBase subclass target
        without instantiating it."""
        if self.args.targets is not None:
            target = self._get_target_quantization_target(prefix, layer)
            if target is None:
                return None
            _, _, spec, cls = target
            return "targets", spec, cls

        if isinstance(layer, LinearBase):
            source = "linear"
            global_spec = self.args.linear
            table = _ONLINE_LINEAR_METHODS
        elif isinstance(layer, RoutedExperts):
            source = "moe"
            global_spec = self.args.moe
            table = _ONLINE_MOE_METHODS
        else:
            return None

        if should_ignore_layer(
            prefix,
            ignore=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
            use_fnmatch=True,
        ):
            return None

        method_cls = self._get_method_cls(global_spec, table, layer)
        if method_cls is None:
            return None
        assert global_spec is not None
        return source, global_spec, method_cls

    def _get_target_quantization_target(
        self, prefix: str, layer: torch.nn.Module
    ) -> tuple[str, str, QuantSpec, type] | None:
        assert self.args.targets is not None
        ignored = should_ignore_layer(
            prefix,
            ignore=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
            use_fnmatch=True,
        )
        matches = _find_matching_targets(
            prefix, self.args.targets, fused_mapping=self.packed_modules_mapping
        )
        if ignored and matches:
            raise ValueError(
                f"Layer {prefix} matches both quantization_config.ignore "
                f"and quantization_config.targets ({matches}); a layer may "
                f"not be referenced by both."
            )
        if ignored or not matches:
            return None
        if len(matches) > 1:
            raise ValueError(
                f"Layer {prefix} matches multiple quantization_config."
                f"targets patterns: {matches}. Each layer may match at most "
                f"one target."
            )
        quant_key_str = self.args.targets[matches[0]]
        shorthand = _ONLINE_SHORTHANDS[quant_key_str]
        if isinstance(layer, LinearBase):
            quant_spec = shorthand.linear
            table = _ONLINE_LINEAR_METHODS
        elif isinstance(layer, RoutedExperts):
            quant_spec = shorthand.moe
            table = _ONLINE_MOE_METHODS
        else:
            raise ValueError(
                f"Layer {prefix} was matched by quantization_config.targets "
                f"({matches[0]}), but online quantization is not supported for "
                f"{type(layer).__name__}."
            )
        if quant_spec is None:
            raise ValueError(
                f"targets pattern {matches[0]} = {quant_key_str} does "
                f"not define a QuantSpec for {type(layer).__name__} layers "
                f"(matched at {prefix})."
            )
        cls = self._get_method_cls(quant_spec, table, layer)
        if cls is None:
            return None
        return quant_key_str, matches[0], quant_spec, cls

    def _instantiate_method(
        self, cls: type, layer: torch.nn.Module
    ) -> "QuantizeMethodBase":
        if isinstance(layer, RoutedExperts):
            assert issubclass(cls, FusedMoEMethodBase)
            return cls(moe=layer.moe_config)
        assert issubclass(cls, OnlineLinearBase)
        linear_method_cls = cast(type[OnlineLinearBase], cls)
        return linear_method_cls()

    def _dispatch_target(
        self, prefix: str, layer: torch.nn.Module
    ) -> "QuantizeMethodBase | None":
        target = self._get_target_quantization_target(prefix, layer)
        if target is None:
            return None
        quant_key_str, pattern, _, cls = target
        self.quantized_layers[prefix] = ("targets", quant_key_str, pattern)
        return self._instantiate_method(cls, layer)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        # `targets` takes precedence over `moe` and `linear` and is exclusive.
        if self.args.targets is not None:
            method = self._dispatch_target(prefix, layer)
            if method is not None:
                return method

            if isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
            if isinstance(layer, RoutedExperts):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return None

        target = self.get_quantization_target(layer, prefix)
        if target is not None:
            source, spec, cls = target
            self.quantized_layers[prefix] = (source, str(spec), None)
            return self._instantiate_method(cls, layer)

        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        if isinstance(layer, RoutedExperts):
            return UnquantizedFusedMoEMethod(layer.moe_config)
        return None
