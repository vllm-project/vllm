# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
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
    kInt4Static32,
    kInt8Dynamic32Sym,
    kInt8StaticChannelSym,
    kLutBStatic,
    kMxfp4Dynamic,
    kMxfp4Static,
    kMxfp8Dynamic,
    kMxfp8Static,
    kNvfp4DynamicToken,
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
    "nvfp4_per_token": kNvfp4DynamicToken,
    "int8_per_channel_static": kInt8StaticChannelSym,
    "lut_b": kLutBStatic,
    # Dynamic per-32-element int8 activations (RDNA3 MXFP4 W4A8 GEMV).
    "int8_per_group_32": kInt8Dynamic32Sym,
    # Load-time MXFP4-to-int4 for Kimi-K3 on gfx942.
    "int4_per_group_32": kInt4Static32,
}

# Ambiguous format names select the dynamic behavior for activations and the
# static one for weights; `*_static`/`*_dynamic` names stay field-independent.
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


def quant_key_name(quant_key: QuantKey | None) -> str | None:
    """User-facing name of `quant_key`, falling back to `str(quant_key)`."""
    # TODO: QuantKey itself should define `__str__`, instead of having this logic.
    if quant_key is None:
        return None
    names = (*QUANT_KEY_NAMES.items(), *_WEIGHT_QUANT_KEY_NAMES.items())
    return next((name for name, key in names if key == quant_key), str(quant_key))


@config
class QuantSpec:
    """Quantization spec for one layer kind (linear or MoE).

    An omitted activation key lets the quantization implementation choose its
    default. An explicitly configured `activation: null` requests no
    activation quantization; methods that do not support that request raise an
    error.
    """

    weight: QuantKeyField = _UNSET
    """Weight quantization key, or a name from QUANT_KEY_NAMES."""

    activation: QuantKeyField = _UNSET
    """Activation quantization key, or a name from QUANT_KEY_NAMES."""

    hadamard_size: int | None = None
    """Block size (a power of two) of an orthonormal Hadamard rotation applied
    along the input dimension of both the weight and the activation before
    quantization (see `InputHadamardLinearMethod`). The GEMM result is
    unchanged up to quantization error, which the rotation shrinks by spreading
    outliers across each block. Linear layers only; validated with MXFP4."""

    fields_set: frozenset[str] = field(init=False, repr=False, compare=False)
    """Names explicitly provided when constructing this spec."""

    @field_validator("weight", mode="before")
    @classmethod
    def _coerce_weight_quant_key(cls, v: Any) -> Any:
        if isinstance(v, str) and v in _WEIGHT_QUANT_KEY_NAMES:
            return _WEIGHT_QUANT_KEY_NAMES[v]
        return v

    @field_validator("hadamard_size")
    @classmethod
    def _check_hadamard_size(cls, size: int | None) -> int | None:
        if size is not None and (size <= 0 or size & (size - 1)):
            raise ValueError(f"hadamard_size must be a power of two, got {size}")
        return size

    def __post_init__(self) -> None:
        """Record the explicitly set fields; unset ones become `None`."""
        fields_set = set()
        for field_name in ("weight", "activation"):
            if getattr(self, field_name) is _UNSET:
                setattr(self, field_name, None)
            else:
                fields_set.add(field_name)
        # `None` means "no rotation", so a set size is always explicit.
        if self.hadamard_size is not None:
            fields_set.add("hadamard_size")
        self.fields_set = frozenset(fields_set)

    def __str__(self) -> str:
        """The spec as configured, e.g. `mxfp4` or
        `{"weight":"mxfp4","activation":null}`."""
        if "activation" not in self.fields_set:
            name = str(quant_key_name(self.weight))
        else:
            spec = {
                "weight": quant_key_name(self.weight),
                "activation": quant_key_name(self.activation),
            }
            name = json.dumps(spec, separators=(",", ":"))
        if self.hadamard_size is not None:
            name = f"{name}_hadamard{self.hadamard_size}"
        return name


@config
class QuantizationConfigArgs:
    """User-facing quantization configuration.

    See `docs/features/quantization/online.md` for the schema and shorthand
    string forms accepted on `linear` and `moe`.
    """

    linear: QuantSpec | None = None
    """Spec applied to `LinearBase` layers."""

    moe: QuantSpec | None = None
    """Spec applied to `FusedMoEFactory` layers."""

    ignore: list[str] = Field(default_factory=list)
    """Layers to skip quantization for. Online quantization also supports
    fnmatch-style patterns."""

    targets: dict[str, str | QuantSpec] | None = None
    """Per-layer online quantization overrides, keyed by exact layer name or
    regex patterns with a `re:`, or fnmatch-style patterns for online
    quantization. A target can map to an online shorthand name (see
    `_ONLINE_SHORTHANDS`) or a quantization spec containing `weight` and
    `activation` keys. A layer that matches no pattern is left unquantized.
    Mutually exclusive with `linear` and `moe`.
    """

    @field_validator("linear", "moe", mode="before")
    @classmethod
    def _coerce_spec(cls, v: Any, info: ValidationInfo) -> Any:
        if not isinstance(v, str):
            return v

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
        targets: dict[str, str | QuantSpec] = {}
        for pattern, target in v.items():
            if not isinstance(pattern, str):
                raise ValueError(
                    f"targets keys must be strings, got {type(pattern).__name__}"
                )
            if isinstance(target, str):
                if target not in _ONLINE_SHORTHANDS:
                    raise ValueError(
                        f"targets[{pattern}] = {target} is not a valid "
                        f"online shorthand name; expected one of "
                        f"{sorted(_ONLINE_SHORTHANDS)}"
                    )
            elif isinstance(target, dict):
                target = QuantSpec(**target)
            elif not isinstance(target, QuantSpec):
                raise ValueError(
                    f"targets[{pattern}] must be an online shorthand name or "
                    f"a quantization spec, got {type(target).__name__}"
                )
            if pattern.startswith("re:"):
                try:
                    re.compile(pattern[3:])
                except re.error as e:
                    raise ValueError(
                        f"targets key {pattern} is not a valid regex: {e}"
                    ) from e
            targets[pattern] = target
        return targets

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
    # MXFP4 linear with a 32-wide Hadamard rotation of weights and activations;
    # MoE stays unquantized (no `moe` field).
    "mxfp4_hadamard32": QuantizationConfigArgs(
        linear=QuantSpec(weight=kMxfp4Static, hadamard_size=32),
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
    # LUT-B routed experts with MXFP8 activations; native on Rubin (SM107),
    # emulated elsewhere. Dense layers stay unquantized.
    "lut_b_moe": QuantizationConfigArgs(
        moe=QuantSpec(weight=kLutBStatic),
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

    # A deferred shorthand may name the checkpoint format: its defaults are only
    # merged in by `get_quant_config` once the model is known to be unquantized,
    # so that an override alone does not quantize layers a checkpoint skips.
    if base is None or quantization in _DEFERRED_ONLINE_SHORTHANDS:
        return quantization_config
    return merge_quantization_config(base, quantization_config)


def merge_quantization_config(
    base: QuantizationConfigArgs, override: QuantizationConfigArgs | None
) -> QuantizationConfigArgs:
    """Apply the fields explicitly set in `override` on top of a shorthand's
    `base` config."""
    if override is None:
        return base

    def merge_spec(
        base_spec: QuantSpec | None,
        override_spec: QuantSpec | None,
    ) -> QuantSpec | None:
        if base_spec is None or override_spec is None:
            return override_spec or base_spec
        if (
            base_spec.hadamard_size is not None
            and "weight" in override_spec.fields_set
            and override_spec.weight != base_spec.weight
        ):
            raise ValueError(
                "quantization_config cannot change the weight of a "
                f"Hadamard-rotated shorthand ({base_spec} -> {override_spec}); "
                "use --quantization online and set `hadamard_size` on the spec "
                "instead."
            )
        return QuantSpec(
            **{name: getattr(base_spec, name) for name in base_spec.fields_set}
            | {name: getattr(override_spec, name) for name in override_spec.fields_set}
        )

    return QuantizationConfigArgs(
        linear=merge_spec(base.linear, override.linear),
        moe=merge_spec(base.moe, override.moe),
        ignore=override.ignore or base.ignore,
        targets=override.targets or base.targets,
    )


# `quantization_config.targets` spec matching the pre-quantized layer whose
# checkpoint quant method is being built (see `resolve_quant_method`).
_layer_target_spec: ContextVar[QuantSpec | None] = ContextVar(
    "layer_target_spec", default=None
)


@contextmanager
def layer_target_spec_scope(spec: QuantSpec | None) -> Iterator[None]:
    """Apply a layer's `targets` spec to `user_activation_override`."""
    token = _layer_target_spec.set(spec)
    try:
        yield
    finally:
        _layer_target_spec.reset(token)


def user_activation_override(kind: str, config: Any) -> tuple[QuantKey | None, bool]:
    """The user's activation override for the layer being built, and whether it
    was set (explicit `null` included): the layer's `targets` spec if any, else
    `quantization_config.<kind>` in `config` (a `VllmConfig` or None; `kind` is
    `linear` or `moe`)."""
    spec = _layer_target_spec.get()
    if spec is None:
        model_config = config.model_config if config else None
        args = model_config.quantization_config if model_config else None
        is_args = isinstance(args, QuantizationConfigArgs)
        spec = getattr(args, kind) if is_args else None
    if spec is None:
        return None, False
    return spec.activation, "activation" in spec.fields_set
