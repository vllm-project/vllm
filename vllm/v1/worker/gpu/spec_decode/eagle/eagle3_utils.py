# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import cast

import torch.nn as nn

from vllm.config import SpeculativeConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsEagle3, supports_eagle3

logger = init_logger(__name__)


def set_eagle3_aux_hidden_state_layers(
    model: nn.Module,
    spec_config: SpeculativeConfig,
) -> None:
    if not supports_eagle3(model):
        raise RuntimeError("Model does not support EAGLE3 interface")
    # mypy may infer the class-level overload for supports_eagle3.
    # Narrow explicitly to the runtime protocol instance.
    if isinstance(model, type):
        raise RuntimeError("Expected model instance for EAGLE3 configuration")
    eagle3_model = cast(SupportsEagle3, model)

    aux_layers = get_eagle3_aux_layers_from_config(spec_config)
    if aux_layers:
        logger.info("Using Eagle3 auxiliary layers from config: %s", aux_layers)
    else:
        aux_layers = eagle3_model.get_eagle3_default_aux_hidden_state_layers()
        logger.info("Using Eagle3 auxiliary layers from model: %s", aux_layers)
    eagle3_model.set_aux_hidden_state_layers(aux_layers)


def resolve_eagle3_aux_layer_semantics(
    spec_config: SpeculativeConfig,
) -> str:
    """Resolve the EAGLE-3 aux-layer index convention to use.

    Resolution order:
      1. Explicit `spec_config.eagle_aux_layer_semantics` override.
      2. `eagle_aux_layer_semantics` declared on the draft model's HF config.
      3. Default to ``"vllm"`` (historical behavior, indices used as-is).
    """
    semantics = getattr(spec_config, "eagle_aux_layer_semantics", None)
    if semantics is None and spec_config.draft_model_config is not None:
        hf_config = spec_config.draft_model_config.hf_config
        semantics = getattr(hf_config, "eagle_aux_layer_semantics", None)
    return semantics or "vllm"


def apply_eagle3_aux_layer_semantics(
    layer_ids: tuple[int, ...],
    spec_config: SpeculativeConfig,
) -> tuple[int, ...]:
    """Map config-declared aux layer ids onto vLLM's capture slots.

    vLLM captures aux hidden states *after* each decoder layer, so a stored
    index `k` yields the output of layer `k - 1`. SGLang-trained draft heads
    declare indices referring to the output of layer `k`, so a `+1` shift is
    applied to keep both frameworks in sync.

    The SGLang/vLLM index-convention ambiguity only affects EAGLE-3
    checkpoints. Other methods (e.g. DFlash) normalize their aux-layer
    indices to vLLM's convention at config-load time, so the shift must not
    be re-applied for them; `eagle_aux_layer_semantics` is rejected for those
    methods in `SpeculativeConfig`.
    """
    if spec_config.method != "eagle3":
        return layer_ids
    if resolve_eagle3_aux_layer_semantics(spec_config) == "sglang":
        return tuple(v + 1 for v in layer_ids)
    return layer_ids


def get_eagle3_aux_layers_from_config(
    spec_config: SpeculativeConfig,
) -> tuple[int, ...] | None:
    if not (spec_config and spec_config.draft_model_config):
        return None
    hf_config = spec_config.draft_model_config.hf_config
    layer_ids = getattr(hf_config, "eagle_aux_hidden_state_layer_ids", None)
    if not layer_ids:
        dflash_config = getattr(hf_config, "dflash_config", None)
        if dflash_config and isinstance(dflash_config, dict):
            # Add 1 to convert DFlash's aux layer id semantics
            layer_ids = [i + 1 for i in (dflash_config.get("target_layer_ids") or [])]
    if not layer_ids:
        dspark_layer_ids = getattr(hf_config, "dspark_target_layer_ids", None)
        if dspark_layer_ids:
            layer_ids = [i + 1 for i in dspark_layer_ids]
    if not layer_ids:
        # Dense DSpark (e.g. Qwen3) also uses different aux layer semantics.
        target_layer_ids = getattr(hf_config, "target_layer_ids", None)
        if target_layer_ids:
            layer_ids = [i + 1 for i in target_layer_ids]
    if layer_ids and isinstance(layer_ids, (list, tuple)):
        return apply_eagle3_aux_layer_semantics(tuple(layer_ids), spec_config)
    return None
