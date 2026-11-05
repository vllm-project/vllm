# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from typing import Annotated, Any, Literal

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
    kMxfp4Dynamic,
    kMxfp4Static,
    kMxfp8Dynamic,
    kNvfp4DynamicTensor,
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
    # Dynamic per-32-element int8 activations (RDNA3 MXFP4 W4A8 GEMV).
    "int8_per_group_32": kInt8Dynamic32Sym,
    # Load-time MXFP4-to-int4 for Kimi-K3 on gfx942.
    "int4_per_group_32": kInt4Static32,
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


# Activation keys of PhaseSpec formats; None is unquantized.
_PHASE_ACTIVATION_KEYS: dict[str, QuantKey | None] = {
    "unquantized": None,
    "nvfp4": kNvfp4DynamicTensor,
    "mxfp8": kMxfp8Dynamic,
}


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
class PhaseSpec:
    """Activation formats for decode and prefill, served from one weight copy.

    A linear call with at most `max_decode_tokens` input rows (normally the
    tokens scheduled in the step, after CUDA graph padding) uses `decode`;
    larger calls use `prefill`. The two may differ in precision but not in
    quantized format: a phase applies to NVFP4 linears when it uses "nvfp4" or
    only "unquantized", and to MXFP8 linears when it uses "mxfp8" or only
    "unquantized".
    """

    prefill: Literal["unquantized", "nvfp4", "mxfp8"] | None = None
    """Activations of large calls (required)."""

    decode: Literal["unquantized", "nvfp4", "mxfp8"] = "unquantized"
    """Activations of small calls: "unquantized" (default) keeps the model
    dtype."""

    max_decode_tokens: int | None = Field(default=None, ge=1)
    """Largest call, in rows, that uses `decode`. Required when `decode` and
    `prefill` differ."""

    scale: Literal["dynamic_tensor"] | None = None
    """Global scale of NVFP4 activations: "dynamic_tensor", the default with
    "nvfp4", shares one scale per call, computed from the absolute maximum of
    its rows. MXFP8 activations only have per-block scales; leave it unset."""

    @model_validator(mode="after")
    def _validate_formats(self) -> "PhaseSpec":
        if self.prefill is None:
            raise ValueError("phase.prefill is required")
        if self.decode != self.prefill and self.max_decode_tokens is None:
            raise ValueError(
                "phase.max_decode_tokens is required when decode and prefill differ"
            )
        quantized = {self.decode, self.prefill} - {"unquantized"}
        if len(quantized) > 1:
            raise ValueError(
                f"phase.decode={self.decode} and phase.prefill={self.prefill} "
                "need two weight copies; use one quantized format or unquantized"
            )
        if "nvfp4" in quantized:
            self.scale = self.scale or "dynamic_tensor"
        elif self.scale is not None:
            raise ValueError("phase.scale only applies to nvfp4 activations")
        return self

    @property
    def quantized_format(self) -> str | None:
        """The quantized activation format the phase uses, if any."""
        return next(
            (f for f in (self.decode, self.prefill) if f != "unquantized"), None
        )

    def activation_modes(self) -> tuple[QuantKey | None, QuantKey | None]:
        """(decode, prefill) activation keys; None is unquantized."""
        assert self.prefill is not None
        return (
            _PHASE_ACTIVATION_KEYS[self.decode],
            _PHASE_ACTIVATION_KEYS[self.prefill],
        )

    def __str__(self) -> str:
        return (
            f"decode={self.decode}, prefill={self.prefill}, "
            f"max_decode_tokens={self.max_decode_tokens}, scale={self.scale}"
        )


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

    hadamard_size: int | None = None
    """Block size (a power of two) of an orthonormal Hadamard rotation applied
    along the input dimension of both the weight and the activation before
    quantization (see `InputHadamardLinearMethod`). The GEMM result is
    unchanged up to quantization error, which the rotation shrinks by spreading
    outliers across each block. Linear layers only; validated with MXFP4."""

    phase: PhaseSpec | None = None
    """Separate decode and prefill activation formats for one weight copy.
    Linear only; mutually exclusive with `activation`. Online-quantized linears
    only support it with MXFP8 weights."""

    @field_validator("hadamard_size")
    @classmethod
    def _check_hadamard_size(cls, size: int | None) -> int | None:
        if size is not None and (size <= 0 or size & (size - 1)):
            raise ValueError(f"hadamard_size must be a power of two, got {size}")
        return size

    @model_validator(mode="after")
    def _validate_phase(self) -> "QuantSpec":
        if self.phase is not None and self.activation is not None:
            raise ValueError("phase and activation are mutually exclusive")
        return self

    def __str__(self) -> str:
        def quant_key_str(quant_key: QuantKey | None) -> str:
            if quant_key is None:
                return "None"
            return next(
                (
                    name
                    for name, known_quant_key in QUANT_KEY_NAMES.items()
                    if known_quant_key == quant_key
                ),
                str(quant_key),
            )

        name = quant_key_str(self.weight)
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
    def _validate_moe_phase(self) -> "QuantizationConfigArgs":
        if self.moe is not None and self.moe.phase is not None:
            raise ValueError("quantization_config.moe.phase is not supported")
        return self

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
        linear=QuantSpec(weight=kMxfp8Dynamic),
        moe=QuantSpec(weight=kMxfp8Dynamic),
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

    linear = quantization_config.linear
    if (
        linear is not None
        and base.linear is not None
        and linear.weight is None
        and linear.activation is None
        and linear.phase is not None
    ):
        # A spec that sets a phase but no format keeps the shorthand's format.
        linear = dataclasses.replace(
            linear,
            weight=base.linear.weight,
            hadamard_size=linear.hadamard_size or base.linear.hadamard_size,
        )
    return QuantizationConfigArgs(
        linear=linear or base.linear,
        moe=quantization_config.moe or base.moe,
        ignore=quantization_config.ignore or base.ignore,
        targets=quantization_config.targets or base.targets,
    )
