# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum
from typing import Any

from pydantic import Field, field_validator

from vllm.config.utils import config


class OnlineQuantScheme(Enum):
    """Supported online quantization schemes."""

    # fp8, weights and activations scaled per-tensor
    FP8_PER_TENSOR = "fp8_per_tensor"

    # fp8, activations scaled in blocks of 1x128 elements, weights scaled in
    # blocks of 128x128 elements (popularized by DeepSeek)
    FP8_PER_BLOCK = "fp8_per_block"

    # int8, weight-only per-channel quantization for MoE expert weights.
    # Linear layers remain unquantized.
    INT8_PER_CHANNEL_WEIGHT_ONLY = "int8_per_channel_weight_only"

    # mxfp8, weights scaled in blocks of 1x32 elements (microscaling FP8)
    MXFP8 = "mxfp8"


@config
class OnlineQuantizationConfigArgs:
    """Configuration for online quantization.

    Controls how ``OnlineQuantizationConfig`` is applied to a model.
    At least one of ``global_scheme``, ``linear_scheme_override``, or
    ``moe_scheme_override`` must be set.
    """

    global_scheme: OnlineQuantScheme | None = None
    """Quantization scheme applied to every supported layer."""

    linear_scheme_override: OnlineQuantScheme | None = None
    """Quantization scheme override for ``LinearBase`` layers."""

    moe_scheme_override: OnlineQuantScheme | None = None
    """Quantization scheme override for ``FusedMoE`` layers."""

    ignore: list[str] = Field(default_factory=list)
    """Layers to skip quantization for. Supports exact names and regex
    patterns with ``re:`` prefix (e.g. ``re:.*attn.*``), consistent with
    compressed_tensors layer skipping."""

    @field_validator(
        "global_scheme", "linear_scheme_override", "moe_scheme_override", mode="before"
    )
    @classmethod
    def _coerce_scheme(
        cls, v: str | OnlineQuantScheme | None
    ) -> OnlineQuantScheme | None:
        if isinstance(v, str):
            return OnlineQuantScheme(v)
        return v


def resolve_online_quant_config(
    quantization: str | None,
    quantization_config: dict[str, Any] | OnlineQuantizationConfigArgs | None,
) -> OnlineQuantizationConfigArgs | None:
    """Resolve online quant scheme shorthand into a quantization config.

    If ``quantization`` is an online quant scheme (e.g. ``'fp8_per_tensor'``),
    ensures ``quantization_config`` has a matching ``global_scheme`` and casts
    it to :class:`OnlineQuantizationConfigArgs` if needed.
    """
    online_quant_values = {s.value for s in OnlineQuantScheme}
    valid_quantization_values = online_quant_values | {"online"}
    if quantization not in valid_quantization_values:
        if quantization_config is not None:
            raise ValueError(
                f"quantization_config is only supported when quantization "
                f"is one of {sorted(valid_quantization_values)}, "
                f"got quantization={quantization!r}"
            )
        return None

    if quantization in online_quant_values:
        scheme = OnlineQuantScheme(quantization)

        if quantization_config is None:
            quantization_config = {
                "global_scheme": scheme.value,
            }
        elif isinstance(quantization_config, OnlineQuantizationConfigArgs):
            if quantization_config.global_scheme is None:
                quantization_config.global_scheme = scheme
            elif quantization_config.global_scheme != scheme:
                raise ValueError(
                    f"quantization={quantization!r} conflicts with "
                    f"quantization_config.global_scheme="
                    f"{quantization_config.global_scheme.value!r}. "
                    f"These must match when both are specified."
                )
        elif isinstance(quantization_config, dict):
            existing = quantization_config.get("global_scheme")
            if existing is None:
                quantization_config["global_scheme"] = scheme.value
            else:
                # Coerce to enum for comparison
                existing_scheme = (
                    OnlineQuantScheme(existing)
                    if isinstance(existing, str)
                    else existing
                )
                if existing_scheme != scheme:
                    raise ValueError(
                        f"quantization={quantization!r} conflicts "
                        f"with quantization_config"
                        f"['global_scheme']={existing!r}. "
                        f"These must match when both are specified."
                    )

    # Cast dict to OnlineQuantizationConfigArgs
    if isinstance(quantization_config, dict):
        quantization_config = OnlineQuantizationConfigArgs(**quantization_config)

    return quantization_config
