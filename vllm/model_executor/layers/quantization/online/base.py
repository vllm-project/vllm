# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Any

import torch

from vllm.config.quantization import (
    _ONLINE_SHORTHANDS,
    QUANT_KEY_NAMES,
    QuantizationConfigArgs,
    QuantSpec,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
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
from vllm.model_executor.layers.quantization.online.moe_base import OnlineMoEMethodBase
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
    kMxfp8Static,
    kNvfp4Static,
)

logger = init_logger(__name__)


class OnlineQuantizationSource(str, Enum):
    """Supported online quantization configuration sources."""

    linear = "linear"  # LinearBase
    moe = "moe"  # RoutedExperts
    targets = "targets"


# Online dispatch tables, keyed by the QuantSpec.weight QuantKey.
_ONLINE_LINEAR_METHODS: dict[QuantKey, type] = {
    kFp8StaticTensorSym: Fp8PerTensorOnlineLinearMethod,
    kFp8Static128BlockSym: Fp8PerBlockOnlineLinearMethod,
    kFp8StaticChannelSym: Fp8PtpcOnlineLinearMethod,
    kMxfp8Static: Mxfp8OnlineLinearMethod,
    kMxfp4Static: Mxfp4OnlineLinearMethod,
}

_ONLINE_MOE_METHODS: dict[QuantKey, type] = {
    kFp8StaticTensorSym: Fp8PerTensorOnlineMoEMethod,
    kFp8Static128BlockSym: Fp8PerBlockOnlineMoEMethod,
    kFp8StaticChannelSym: Fp8PtpcOnlineMoEMethod,
    kMxfp8Static: Mxfp8OnlineMoEMethod,
    kMxfp4Static: Mxfp4OnlineMoEMethod,
    kInt8StaticChannelSym: Int8OnlineMoEMethod,
    kNvfp4Static: Nvfp4OnlineMoEMethod,
}


def _find_matching_targets(
    prefix: str,
    targets: Mapping[str, str | QuantSpec],
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
    target_values = [targets[pattern] for pattern in matched_patterns]
    if any(target != target_values[0] for target in target_values[1:]):
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
        """Resolve the online method class for a layer's quantization spec.

        Args:
            spec: Quantization specification to resolve.
            table: Mapping from weight quantization keys to method classes.
            layer: Layer that will use the resolved method.

        Returns:
            The matching method class, or None when ``spec`` has no weight
            quantization.
        """
        if spec is None or spec.weight is None:
            return None
        cls = table.get(spec.weight)
        if cls is None:
            raise ValueError(
                f"online quantization for {type(layer).__name__} with "
                f"weight={spec.weight} is not supported; supported weight "
                f"keys: {sorted(str(k) for k in table)}"
            )
        return cls

    def resolve_quant_method_cls(
        self, layer: torch.nn.Module, prefix: str
    ) -> (
        tuple[
            OnlineQuantizationSource,
            str,
            str | None,
            QuantSpec,
            type,
            QuantKey | None,
        ]
        | None
    ):
        """Resolve quantization metadata and method class without instantiating it.

        Args:
            layer: Layer for which to resolve online quantization.
            prefix: Fully qualified layer name.

        Returns:
            A tuple of source, quantization key string, target pattern, spec,
            method class, and activation quantization key. Returns None when
            online quantization does not apply to the layer.
        """
        quant_spec: QuantSpec | None
        if self.args.targets is not None:
            resolved_pattern = self._resolve_targets_quant_method_metadata(
                prefix, layer
            )
            if resolved_pattern is None:
                return None
            source, quant_key_str, target_pattern, quant_spec, table = resolved_pattern
        else:
            if isinstance(layer, LinearBase):
                source = OnlineQuantizationSource.linear
                quant_spec = self.args.linear
                table = _ONLINE_LINEAR_METHODS
            elif isinstance(layer, RoutedExperts):
                source = OnlineQuantizationSource.moe
                quant_spec = self.args.moe
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
            quant_key_str = str(quant_spec)
            target_pattern = None

        quant_method_cls = self._get_method_cls(quant_spec, table, layer)
        if quant_method_cls is None:
            return None

        assert quant_spec is not None

        if isinstance(layer, RoutedExperts):
            assert issubclass(quant_method_cls, OnlineMoEMethodBase)
        else:
            assert isinstance(layer, LinearBase)
            assert issubclass(quant_method_cls, OnlineLinearBase)

        if "activation" not in quant_spec.fields_set:
            activation_quant_key = quant_method_cls.default_activation_quant_key
        else:
            activation_quant_key = quant_spec.activation

        return (
            source,
            quant_key_str,
            target_pattern,
            quant_spec,
            quant_method_cls,
            activation_quant_key,
        )

    def _resolve_targets_quant_method_metadata(
        self, prefix: str, layer: torch.nn.Module
    ) -> (
        tuple[OnlineQuantizationSource, str, str, QuantSpec, dict[QuantKey, type]]
        | None
    ):
        """Resolve target-pattern quantization metadata and dispatch table.

        Args:
            prefix: Fully qualified layer name.
            layer: Layer matched against configured target patterns.

        Returns:
            A tuple of source, quantization key string, target pattern, spec,
            and dispatch table. Returns None when no pattern applies or the
            layer is ignored.
        """
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
        target_pattern = matches[0]
        target = self.args.targets[target_pattern]
        if isinstance(target, str):
            shorthand = _ONLINE_SHORTHANDS[target]
            quant_key_str = target
            quant_spec = (
                shorthand.linear if isinstance(layer, LinearBase) else shorthand.moe
            )
        else:
            quant_spec = target
            target_config: dict[str, str | None] = {"weight": str(quant_spec)}

            # TODO: QuantKey itself should define `__str__`,
            # instead of having this logic.
            if "activation" in quant_spec.fields_set:
                target_config["activation"] = (
                    next(
                        (
                            name
                            for name, known_quant_key in QUANT_KEY_NAMES.items()
                            if known_quant_key == quant_spec.activation
                        ),
                        str(quant_spec.activation),
                    )
                    if quant_spec.activation is not None
                    else None
                )
            quant_key_str = json.dumps(target_config, separators=(",", ":"))
        if isinstance(layer, LinearBase):
            table = _ONLINE_LINEAR_METHODS
        elif isinstance(layer, RoutedExperts):
            table = _ONLINE_MOE_METHODS
        else:
            raise ValueError(
                f"Layer {prefix} was matched by quantization_config.targets "
                f"({target_pattern}), but online quantization is not supported for "
                f"{type(layer).__name__}."
            )
        if quant_spec is None:
            raise ValueError(
                f"targets pattern {target_pattern} = {target} does "
                f"not define a QuantSpec for {type(layer).__name__} layers "
                f"(matched at {prefix})."
            )
        return (
            OnlineQuantizationSource.targets,
            quant_key_str,
            target_pattern,
            quant_spec,
            table,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        # `targets` takes precedence over `moe` and `linear` and is exclusive.
        resolved = self.resolve_quant_method_cls(layer, prefix)
        if resolved is not None:
            (
                source,
                quant_key_str,
                target_pattern,
                _,
                quant_method_cls,
                activation_quant_key,
            ) = resolved
            self.quantized_layers[prefix] = (
                source.value,
                quant_key_str,
                target_pattern,
            )
            if isinstance(layer, RoutedExperts):
                assert issubclass(quant_method_cls, OnlineMoEMethodBase)
                return quant_method_cls(
                    moe=layer.moe_config,
                    activation_quant_key=activation_quant_key,
                )

            assert issubclass(quant_method_cls, OnlineLinearBase)
            return quant_method_cls(activation_quant_key=activation_quant_key)

        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        if isinstance(layer, RoutedExperts):
            return UnquantizedFusedMoEMethod(layer.moe_config)
        return None
