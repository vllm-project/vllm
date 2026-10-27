# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Read model head constraints without loading model weights."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HeadConstraint:
    name: str
    attention_heads: int
    kv_heads: int

    def supports(self, tp: int) -> bool:
        if self.attention_heads % tp:
            return False
        if self.kv_heads >= tp:
            return self.kv_heads % tp == 0
        return tp % self.kv_heads == 0


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Cannot determine a positive integer for {name}: {value!r}")
    return value


def extract_head_constraints(
    hf_config: Any, *, include_vision: bool = True
) -> tuple[HeadConstraint, ...]:
    """Extract decoder, global-attention, hybrid and encoder head constraints."""
    constraints = []
    if getattr(hf_config, "model_type", None) == "whisper":
        for component in ("encoder", "decoder"):
            heads = _positive_int(
                getattr(hf_config, f"{component}_attention_heads", None), component
            )
            constraints.append(HeadConstraint(component, heads, heads))
        return tuple(constraints)

    text = hf_config.get_text_config()
    # Prefer resolved per-layer values for heterogeneous attention. Accessing
    # global KV-head attributes may raise on recent Transformers configurations.
    per_layer = getattr(text, "per_layer_config", None)
    if per_layer is not None:
        layers = (
            per_layer.items() if hasattr(per_layer, "items") else enumerate(per_layer)
        )
    else:
        layers = (("text", text),)

    text_constraint_count = 0
    for name, layer in layers:
        heads_value = getattr(layer, "num_attention_heads", None)
        kv_value = getattr(layer, "num_key_value_heads", None)
        # Non-attention layers may not specify either value.
        if per_layer is not None and heads_value is None and kv_value is None:
            continue
        heads = _positive_int(heads_value, f"{name} attention heads")
        kv = _positive_int(heads if kv_value is None else kv_value, f"{name} KV heads")
        constraints.append(HeadConstraint(str(name), heads, kv))
        text_constraint_count += 1

        global_kv = getattr(layer, "num_global_key_value_heads", None)
        if global_kv is not None:
            constraints.append(
                HeadConstraint(
                    f"{name} global attention",
                    heads,
                    _positive_int(global_kv, f"{name} global KV"),
                )
            )

        # Hybrid linear-attention layers can impose additional head constraints.
        for field in ("linear_num_key_heads", "linear_num_value_heads"):
            count = getattr(layer, field, None)
            if count is not None:
                count = _positive_int(count, f"{name} {field}")
                constraints.append(HeadConstraint(f"{name} {field}", count, count))

    if not text_constraint_count:
        raise ValueError("No attention-head constraints found in model layers")

    vision = getattr(hf_config, "vision_config", None)
    if include_vision and vision is not None:
        vision_heads = getattr(vision, "num_attention_heads", None)
        if vision_heads is None:
            vision_heads = getattr(vision, "num_heads", None)
        vision_heads = _positive_int(vision_heads, "vision heads")
        constraints.append(HeadConstraint("vision", vision_heads, vision_heads))
    return tuple(constraints)


def load_head_constraints(config: dict[str, Any]) -> tuple[HeadConstraint, ...]:
    """Load the effective serving configuration through vLLM's config loader."""
    from vllm.transformers_utils.config import get_config

    hf_overrides = config.get("hf-overrides")
    if isinstance(hf_overrides, str):
        hf_overrides = json.loads(hf_overrides)
    if hf_overrides is not None and not isinstance(hf_overrides, dict):
        raise ValueError("Head detection requires hf-overrides to be a JSON object")
    hf_config = get_config(
        config["model"],
        trust_remote_code=config.get("trust-remote-code", False),
        revision=config.get("revision"),
        code_revision=config.get("code-revision"),
        config_format=config.get("config-format", "auto"),
        hf_overrides_kw=hf_overrides,
    )
    return extract_head_constraints(
        hf_config, include_vision=config.get("mm-encoder-tp-mode") != "data"
    )


def suggest_tensor_parallel_size(
    candidate: int, constraints: tuple[HeadConstraint, ...]
) -> int:
    """Reduce an automatic power-of-two candidate until all head checks pass."""
    _positive_int(candidate, "TP candidate")
    if candidate & (candidate - 1) or not constraints:
        raise ValueError("Expected a power-of-two TP candidate and model constraints")
    while not all(constraint.supports(candidate) for constraint in constraints):
        candidate //= 2
    return candidate
