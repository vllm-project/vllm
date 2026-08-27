# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache
from typing import TYPE_CHECKING, Annotated, Any, TypeAlias

from pydantic import Field, GetPydanticSchema, ValidationInfo, field_validator
from pydantic_core import core_schema

from vllm.config.utils import config

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
else:
    QuantKey: TypeAlias = object

__all__ = [
    "ONLINE_QUANT_SHORTHAND_NAMES",  # noqa: F822 - resolved by __getattr__
    "QUANT_KEY_NAMES",  # noqa: F822 - resolved by __getattr__
    "QuantSpec",
    "QuantizationConfigArgs",
    "resolve_quantization_config",
]


@cache
def _quant_keys() -> dict[str, "QuantKey"]:
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
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
        kNvfp4Static,
    )

    return {
        "fp8_per_tensor_static": kFp8StaticTensorSym,
        "fp8_per_tensor_dynamic": kFp8DynamicTensorSym,
        "fp8_per_token": kFp8DynamicTokenSym,
        "fp8_per_channel_static": kFp8StaticChannelSym,
        "fp8_per_block_static": kFp8Static128BlockSym,
        "fp8_per_block_dynamic": kFp8Dynamic128Sym,
        "mxfp8": kMxfp8Dynamic,
        "mxfp4": kMxfp4Dynamic,
        "int8_per_channel_static": kInt8StaticChannelSym,
        "mxfp4_static": kMxfp4Static,
        "nvfp4_static": kNvfp4Static,
    }


@cache
def _quant_key_names() -> dict[str, "QuantKey"]:
    keys = _quant_keys()
    return {
        name: value
        for name, value in keys.items()
        if name not in {"mxfp4_static", "nvfp4_static"}
    }


def _coerce_quant_key(v: Any) -> QuantKey | None:
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

    if v is None or isinstance(v, QuantKey):
        return v
    if not isinstance(v, str):
        raise TypeError(f"expected str or QuantKey, got {type(v).__name__}")
    try:
        return _quant_key_names()[v]
    except KeyError:
        raise ValueError(
            f"unknown quantization name {v!r}; "
            f"expected one of {sorted(_quant_key_names())}"
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


@config
class QuantSpec:
    """Quantization spec for one layer kind (linear or MoE).

    `None` on either side means the method class falls back to its own default
    (typically inherited from the checkpoint, or unquantized for online).
    """

    weight: QuantKeyField = None
    """Weight quantization key, or a name from QUANT_KEY_NAMES."""

    activation: QuantKeyField = None
    """Activation quantization key, or a name from QUANT_KEY_NAMES."""


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
    """Layers to skip quantization for."""

    @field_validator("linear", "moe", mode="before")
    @classmethod
    def _coerce_spec(cls, v: Any, info: ValidationInfo) -> Any:
        if not isinstance(v, str):
            return v
        field_name = info.field_name
        assert field_name is not None
        shorthands = _online_shorthands()
        if v in shorthands:
            spec = getattr(shorthands[v], field_name)
            if spec is None:
                raise ValueError(
                    f"online shorthand {v!r} does not define a {field_name} spec"
                )
            return spec
        return QuantSpec(weight=_coerce_quant_key(v))


@cache
def _online_shorthands() -> dict[str, QuantizationConfigArgs]:
    keys = _quant_keys()

    return {
        "fp8_per_tensor": QuantizationConfigArgs(
            linear=QuantSpec(weight=keys["fp8_per_tensor_static"]),
            moe=QuantSpec(weight=keys["fp8_per_tensor_static"]),
        ),
        "fp8_per_block": QuantizationConfigArgs(
            linear=QuantSpec(weight=keys["fp8_per_block_static"]),
            moe=QuantSpec(weight=keys["fp8_per_block_static"]),
        ),
        "fp8_per_channel": QuantizationConfigArgs(
            linear=QuantSpec(weight=keys["fp8_per_channel_static"]),
            moe=QuantSpec(weight=keys["fp8_per_channel_static"]),
        ),
        "mxfp8": QuantizationConfigArgs(
            linear=QuantSpec(weight=keys["mxfp8"]),
            moe=QuantSpec(weight=keys["mxfp8"]),
        ),
        "mxfp4": QuantizationConfigArgs(
            linear=QuantSpec(weight=keys["mxfp4_static"]),
            moe=QuantSpec(weight=keys["mxfp4_static"]),
        ),
        "int8_per_channel_weight_only": QuantizationConfigArgs(
            moe=QuantSpec(weight=keys["int8_per_channel_static"]),
        ),
        "nvfp4_per_token": QuantizationConfigArgs(
            moe=QuantSpec(weight=keys["nvfp4_static"]),
        ),
    }


@cache
def _online_quant_shorthand_names() -> tuple[str, ...]:
    return (*_online_shorthands(), "online")


def __getattr__(name: str):
    value: object
    if name == "QUANT_KEY_NAMES":
        value = _quant_key_names()
    elif name == "_ONLINE_SHORTHANDS":
        value = _online_shorthands()
    elif name == "ONLINE_QUANT_SHORTHAND_NAMES":
        value = _online_quant_shorthand_names()
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    return value


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
    shorthand_names = _online_quant_shorthand_names()
    if quantization is not None and quantization not in shorthand_names:
        if quantization_config is not None:
            raise ValueError(
                f"quantization_config is only supported when quantization is "
                f"one of {sorted(shorthand_names)}, "
                f"got quantization={quantization!r}"
            )
        return None

    base = _online_shorthands().get(quantization) if quantization else None

    if quantization_config is None:
        return base

    if isinstance(quantization_config, dict):
        quantization_config = QuantizationConfigArgs(**quantization_config)

    if base is None:
        return quantization_config

    return QuantizationConfigArgs(
        linear=quantization_config.linear or base.linear,
        moe=quantization_config.moe or base.moe,
        ignore=quantization_config.ignore or base.ignore,
    )
