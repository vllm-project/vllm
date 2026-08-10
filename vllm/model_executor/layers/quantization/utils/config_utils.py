# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


def is_shared_expert_quant_fse_compatible(
    quant_config: "QuantizationConfig | None",
    expert_prefix: str,
    shared_expert_prefix: str,
) -> tuple[bool, str | None]:
    """Check whether quantization permits fused shared-expert execution.

    Returns:
        A compatibility flag and, when incompatible, the reason.
    """
    if quant_config is None:
        return True, None

    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    if isinstance(quant_config, OnlineQuantizationConfig):
        if (
            quant_config.args.moe is not None
            and quant_config.args.linear is not None
            and quant_config.args.moe == quant_config.args.linear
        ):
            return True, None
        return (
            False,
            "online quantization requires identical non-empty moe and linear "
            f"quantization configurations; got moe={quant_config.args.moe!r}, "
            f"linear={quant_config.args.linear!r}",
        )

    if isinstance(quant_config, QuarkConfig):
        # TODO: Check on `layer_quant_config`. There could be cases where
        # `expert_prefix` and `shared_expert_prefix` have a different per-layer
        # quantization config through `layer_quant_config`
        is_compatible = not any(
            "shared_expert" in str(entry)
            for entry in quant_config.quant_config.get("exclude", [])
        )
        if is_compatible:
            return True, None
        return False, f"Quark excludes shared experts at {shared_expert_prefix}"

    # TODO: Extend FSE support detection to other quantization methods. Typically,
    # one would check that the experts and shared_experts use the same
    # quantization config. This may be refactored as part of QuantizationConfig later.

    return (
        False,
        "shared-expert FSE quantization compatibility is not implemented for "
        f"{type(quant_config).__name__}",
    )
