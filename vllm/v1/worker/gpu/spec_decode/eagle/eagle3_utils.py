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
        aux_layers = eagle3_model.get_eagle3_aux_hidden_state_layers()
        logger.info("Using Eagle3 auxiliary layers from model: %s", aux_layers)
    eagle3_model.set_aux_hidden_state_layers(aux_layers)


def get_eagle3_aux_layers_from_config(
    spec_config: SpeculativeConfig,
) -> tuple[int, ...] | None:
    if not (spec_config and spec_config.draft_model_config):
        return None
    hf_config = spec_config.draft_model_config.hf_config
    if not hasattr(hf_config, "eagle_aux_hidden_state_layer_ids"):
        return None
    layer_ids = hf_config.eagle_aux_hidden_state_layer_ids
    if layer_ids and isinstance(layer_ids, (list, tuple)):
        return tuple(layer_ids)
    return None
