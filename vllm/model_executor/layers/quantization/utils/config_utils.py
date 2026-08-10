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
    from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config

    if isinstance(quant_config, DeepseekV4FP8Config):
        from vllm.config import get_current_vllm_config

        if quant_config.expert_dtype != "fp4":
            return False, "DeepSeek-V4 routed experts are not MXFP4"

        quantization_config = getattr(
            get_current_vllm_config().model_config.hf_config,
            "quantization_config",
            None,
        )
        if quantization_config is None:
            return False, "DeepSeek-V4 has no quantization configuration"

        shared_expert_prefix = shared_expert_prefix.removeprefix("model.")
        if any(
            entry.startswith(shared_expert_prefix)
            for entry in quantization_config.get("exclude") or []
            if isinstance(entry, str)
        ):
            return (
                False,
                f"DeepSeek-V4 excludes shared experts at {shared_expert_prefix}",
            )

        layer_config = (quantization_config.get("layer_quant_config") or {}).get(
            f"{shared_expert_prefix}.w1"
        )
        shared_weight_config = (
            layer_config or quantization_config.get("global_quant_config") or {}
        ).get("weight") or {}
        if shared_weight_config.get("dtype") == "fp4":
            return True, None
        return (
            False,
            f"DeepSeek-V4 shared experts at {shared_expert_prefix} are not MXFP4",
        )

    if isinstance(quant_config, OnlineQuantizationConfig):
        from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
            should_ignore_layer,
        )

        ignored_layers = quant_config.args.ignore
        expert_is_ignored = should_ignore_layer(
            expert_prefix,
            ignore=ignored_layers,
            fused_mapping=quant_config.packed_modules_mapping,
        )
        shared_expert_is_ignored = any(
            should_ignore_layer(
                f"{shared_expert_prefix}.{projection}",
                ignore=ignored_layers,
                fused_mapping=quant_config.packed_modules_mapping,
            )
            for projection in ("gate_up_proj", "down_proj")
        )
        if expert_is_ignored or shared_expert_is_ignored:
            ignored_prefix = (
                expert_prefix if expert_is_ignored else shared_expert_prefix
            )
            return (
                False,
                f"online quantization excludes FSE weights at {ignored_prefix}",
            )

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
