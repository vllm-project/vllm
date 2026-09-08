# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field
from typing import Annotated, Any, cast

import regex as re
from pydantic import (
    Field,
    GetPydanticSchema,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import core_schema

from vllm.config.utils import config
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kInt8StaticChannelSym,
    kMxfp4Dynamic,
    kMxfp4Static,
    kMxfp8Dynamic,
    kMxfp8Static,
    kNvfp4Static,
)

# User-facing names addressable from quantization_config.
QUANT_KEY_NAMES: dict[str, QuantKey] = {
    "fp8_per_tensor_static": kFp8StaticTensorSym,
    "fp8_per_tensor_dynamic": kFp8DynamicTensorSym,
    "fp8_per_token": kFp8DynamicTokenSym,
    "fp8_per_channel_static": kFp8StaticChannelSym,
    "fp8_per_block_static": kFp8Static128BlockSym,
    "fp8_per_block_dynamic": kFp8Dynamic128Sym,
    "mxfp8": kMxfp8Dynamic,
    "mxfp4": kMxfp4Dynamic,
    "int8_per_channel_static": kInt8StaticChannelSym,
}

# Ambiguous format names select the appropriate dynamic/static behavior
# for either activation (dynamic) or weight (static).
# Explicit ``*_static`` and ``*_dynamic`` names above retain
# their field-independent meanings.
# TODO: possibly deprecate the ``*_static`` and ``*_dynamic`` variants.
_WEIGHT_QUANT_KEY_NAMES: dict[str, QuantKey] = {
    "mxfp8": kMxfp8Static,
    "mxfp4": kMxfp4Static,
}


def _coerce_quant_key(v: Any) -> QuantKey | None:
    if v is None or isinstance(v, QuantKey):
        return v
    if not isinstance(v, str):
        raise TypeError(f"expected str or QuantKey, got {type(v).__name__}")
    try:
        return QUANT_KEY_NAMES[v]
    except KeyError:
        raise ValueError(
            f"unknown quantization name {v!r}; "
            f"expected one of {sorted(QUANT_KEY_NAMES)}"
        ) from None


# Stop pydantic from introspecting QuantKey: it transitively contains a
# NamedTuple with `ClassVar[GroupShape]` declarations that pydantic refuses.
QuantKeyField = Annotated[
    QuantKey | None,
    GetPydanticSchema(
        lambda _src, _handler: core_schema.no_info_plain_validator_function(
            _coerce_quant_key
        )
    ),
]

_UNSET = cast(QuantKey | None, object())


@config
class QuantSpec:
    """Quantization spec for one layer kind (linear or MoE).

    An omitted activation key lets the quantization implementation choose its
    default. An explicitly configured ``activation: null`` requests no
    activation quantization; methods that do not support that request raise an
    error.
    """

    weight: QuantKeyField = _UNSET
    """Weight quantization key, or a name from QUANT_KEY_NAMES."""

    activation: QuantKeyField = _UNSET
    """Activation quantization key, or a name from QUANT_KEY_NAMES."""

    _fields_set: frozenset[str] = field(init=False, repr=False, compare=False)
    """Names explicitly provided when constructing this spec."""

    @field_validator("weight", mode="before")
    @classmethod
    def _coerce_weight_quant_key(cls, v: Any) -> Any:
        if isinstance(v, str) and v in _WEIGHT_QUANT_KEY_NAMES:
            return _WEIGHT_QUANT_KEY_NAMES[v]
        return v

    def __post_init__(self) -> None:
        """
        We need a way to distinguish cases where `activation` is set by the
        user or is a default `None`, as `_ONLINE_SHORTHANDS` do not hold the
        default activation quant key, but may be overridden by users, including
        to `null`.
        """
        fields_set = set()
        for field_name in ("weight", "activation"):
            if getattr(self, field_name) is _UNSET:
                setattr(self, field_name, None)
            else:
                fields_set.add(field_name)
        self._fields_set = frozenset(fields_set)

    @property
    def fields_set(self) -> frozenset[str]:
        """Names explicitly provided when constructing this spec."""
        return self._fields_set

    def __str__(self) -> str:
        def quant_key_str(quant_key: QuantKey | None) -> str:
            if quant_key is None:
                return "None"
            return next(
                (
                    name
                    for name, known_quant_key in (
                        *QUANT_KEY_NAMES.items(),
                        *_WEIGHT_QUANT_KEY_NAMES.items(),
                    )
                    if known_quant_key == quant_key
                ),
                str(quant_key),
            )

        return quant_key_str(self.weight)


@config
class QuantizationConfigArgs:
    """User-facing quantization configuration.

    See `docs/features/quantization/online.md` for the schema and shorthand
    string forms accepted on `linear` and `moe`.
    """

    linear: QuantSpec | None = None
    """Spec applied to ``LinearBase`` layers."""

    moe: QuantSpec | None = None
    """Spec applied to ``FusedMoEFactory`` layers."""

    ignore: list[str] = Field(default_factory=list)
    """Layers to skip quantization for. Online quantization also supports
    fnmatch-style patterns."""

    targets: dict[str, str] | None = None
    """Per-layer online quantization overrides, keyed by exact layer name or
    regex patterns with a `re:`, or fnmatch-style patterns for online
    quantization, mapping to an online shorthand name (see
    `_ONLINE_SHORTHANDS`). A layer that matches no pattern is left unquantized.
    Mutually exclusive with `linear` and `moe`.
    """

    @field_validator("linear", "moe", mode="before")
    @classmethod
    def _coerce_spec(cls, v: Any, info: ValidationInfo) -> Any:
        if not isinstance(v, str):
            return v

        # e.g. `--quantization-config.moe mxfp4`
        field_name = info.field_name
        assert field_name is not None
        if v in _ONLINE_SHORTHANDS:
            spec = getattr(_ONLINE_SHORTHANDS[v], field_name)
            if spec is None:
                raise ValueError(
                    f"online shorthand {v!r} does not define a {field_name} spec"
                )
            return spec
        return QuantSpec(weight=_coerce_quant_key(v))

    @field_validator("targets", mode="before")
    @classmethod
    def _validate_targets(cls, v: Any) -> Any:
        if v is None:
            return v
        if not isinstance(v, dict):
            raise TypeError(f"targets must be a dict, got {type(v).__name__}")
        for pattern, shorthand in v.items():
            if not isinstance(pattern, str):
                raise ValueError(
                    f"targets keys must be strings, got {type(pattern).__name__}"
                )
            if not isinstance(shorthand, str) or shorthand not in _ONLINE_SHORTHANDS:
                raise ValueError(
                    f"targets[{pattern}] = {shorthand} is not a valid "
                    f"online shorthand name; expected one of "
                    f"{sorted(_ONLINE_SHORTHANDS)}"
                )
            if pattern.startswith("re:"):
                try:
                    re.compile(pattern[3:])
                except re.error as e:
                    raise ValueError(
                        f"targets key {pattern} is not a valid regex: {e}"
                    ) from e
        return v

    @model_validator(mode="after")
    def _validate_targets_exclusivity(self) -> "QuantizationConfigArgs":
        if self.targets is None:
            return self
        if self.linear is not None or self.moe is not None:
            raise ValueError(
                "quantization_config.targets is mutually exclusive with "
                f"quantization_config.linear/moe, got "
                f"targets={self.targets}, linear={self.linear}, "
                f"moe={self.moe}."
            )
        return self


# CLI shorthands accepted by `--quantization`. Each desugars to a full
# QuantizationConfigArgs; activation overrides go through quantization_config.
_ONLINE_SHORTHANDS: dict[str, QuantizationConfigArgs] = {
    "fp8_per_tensor": QuantizationConfigArgs(
        linear=QuantSpec(weight=kFp8StaticTensorSym),
        moe=QuantSpec(weight=kFp8StaticTensorSym),
    ),
    "fp8_per_block": QuantizationConfigArgs(
        linear=QuantSpec(weight=kFp8Static128BlockSym),
        moe=QuantSpec(weight=kFp8Static128BlockSym),
    ),
    # Per-output-channel weight scale + dynamic per-token activation.
    # Same shape as llmcompressor's FP8_DYNAMIC recipe.
    "fp8_per_channel": QuantizationConfigArgs(
        linear=QuantSpec(weight=kFp8StaticChannelSym),
        moe=QuantSpec(weight=kFp8StaticChannelSym),
    ),
    "mxfp8": QuantizationConfigArgs(
        linear=QuantSpec(weight=kMxfp8Static),
        moe=QuantSpec(weight=kMxfp8Static),
    ),
    "mxfp4": QuantizationConfigArgs(
        linear=QuantSpec(weight=kMxfp4Static),
        moe=QuantSpec(weight=kMxfp4Static),
    ),
    # INT8 weight-only on MoE; linear stays unquantized (no `linear` field).
    # TODO: this is broken since at least #41566, as Int8OnlineMoEMethod
    # defaults to activation_quant_key=kInt8DynamicTokenSym.
    "int8_per_channel_weight_only": QuantizationConfigArgs(
        moe=QuantSpec(weight=kInt8StaticChannelSym),
    ),
    # Online NVFP4 on MoE with per-token dynamic activation scales (Blackwell +
    # FlashInfer TRTLLM only); linear stays unquantized (no `linear` field).
    "nvfp4_per_token": QuantizationConfigArgs(
        moe=QuantSpec(weight=kNvfp4Static),
    ),
}


# Names accepted by `--quantization`; "online" means "use quantization_config".
ONLINE_QUANT_SHORTHAND_NAMES: tuple[str, ...] = (
    *_ONLINE_SHORTHANDS.keys(),
    "online",
)

# These names are also checkpoint quantization methods. Their online configs
# are resolved only when checkpoint quantization metadata is absent.
_DEFERRED_ONLINE_SHORTHANDS = frozenset(("mxfp4", "mxfp8"))


def resolve_quantization_config(
    quantization: str | None,
    quantization_config: dict[str, Any] | QuantizationConfigArgs | None,
) -> QuantizationConfigArgs | None:
    """Resolve `--quantization` shorthand and `--quantization-config` into a
    QuantizationConfigArgs.

    `quantization` is a CLI shorthand that desugars into a base config via
    `_ONLINE_SHORTHANDS`. `quantization_config` is a dict or pre-built args
    object. When both are given, fields explicitly set in `quantization_config`
    take precedence over the shorthand.
    """
    if quantization is not None and quantization not in ONLINE_QUANT_SHORTHAND_NAMES:
        # Pre-quantized checkpoints can be composed with online quantization
        # for layers that the base quant_method leaves unquantized. The
        # checkpoint quant_method remains the primary quantization method; composition
        # is performed after its config has been loaded.
        if quantization_config is None:
            return None

        # `quantization_config` may hold both:
        # 1. Base quantization method activation key override,
        # 2. online quantization config to apply on top of the base quant_method.
        if isinstance(quantization_config, dict):
            return QuantizationConfigArgs(**quantization_config)
        return quantization_config

    base = _ONLINE_SHORTHANDS.get(quantization) if quantization else None

    if quantization_config is None:
        if quantization in _DEFERRED_ONLINE_SHORTHANDS:
            return None
        return base

    if isinstance(quantization_config, dict):
        quantization_config = QuantizationConfigArgs(**quantization_config)

    if base is None:
        return quantization_config

    def merge_spec(
        base_spec: QuantSpec | None,
        override_spec: QuantSpec | None,
    ) -> QuantSpec | None:
        if override_spec is None:
            return base_spec
        if base_spec is None:
            return override_spec

        values = {
            "weight": (
                override_spec.weight
                if "weight" in override_spec.fields_set
                else base_spec.weight
            )
        }

        if "activation" in override_spec.fields_set:
            values["activation"] = override_spec.activation
        else:
            values["activation"] = base_spec.activation

        return QuantSpec(**values)

    return QuantizationConfigArgs(
        linear=merge_spec(base.linear, quantization_config.linear),
        moe=merge_spec(base.moe, quantization_config.moe),
        ignore=quantization_config.ignore or base.ignore,
        targets=quantization_config.targets or base.targets,
    )
