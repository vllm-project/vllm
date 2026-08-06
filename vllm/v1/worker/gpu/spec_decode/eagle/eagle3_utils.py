# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import cast

import torch
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
    reserve_aux_intermediate_tensor_slots(model)


def _inner_decoder(model: nn.Module) -> nn.Module | None:
    parent_ref = model
    if hasattr(model, "get_language_model"):
        parent_ref = model.get_language_model()
    elif hasattr(model, "language_model"):
        parent_ref = model.language_model
    return getattr(parent_ref, "model", None)


def supports_aux_hidden_states_over_pp(model: nn.Module) -> bool:
    """Whether `model` carries its aux hidden states across pipeline stages.

    EAGLE3-style drafting runs on the last PP rank but taps layers that may sit
    on earlier stages, so the target has to get those tensors to that rank.
    Models that do opt in via
    `EagleModelMixin.supports_aux_hidden_states_over_pp`.
    """
    inner = _inner_decoder(model)
    return bool(getattr(inner, "supports_aux_hidden_states_over_pp", False))


def reserve_aux_intermediate_tensor_slots(model: nn.Module) -> None:
    """Declare the aux slots the last PP stage receives inline.

    The stage before the last one ships its taps inside the pipeline handoff
    (`EagleModelMixin.pack_local_aux_for_last`). The V2 runner does not forward
    a received tensor dict straight to the model: it copies it into a
    persistent buffer built once from `make_empty_intermediate_tensors`, so the
    CUDA graphs keep seeing the same addresses, and any key that buffer does
    not already have is silently dropped. Unless those slots are declared here
    the taps never reach the drafter, which then reads whatever the buffer held
    and loses acceptance without failing.
    """
    from vllm.distributed.parallel_state import get_pp_group

    pp = get_pp_group()
    if pp.world_size < 2 or not pp.is_last_rank:
        return
    inner = _inner_decoder(model)
    if not getattr(inner, "supports_aux_hidden_states_over_pp", False):
        return

    num_taps = inner._num_local_taps_on_rank(pp.world_size - 2, pp.world_size)
    if num_taps == 0:
        return

    key = inner.AUX_HIDDEN_STATE_KEY
    hidden_size = inner.config.hidden_size
    make_empty = model.make_empty_intermediate_tensors

    def make_empty_with_aux(batch_size, dtype, device):
        tensors = make_empty(batch_size, dtype, device)
        for i in range(num_taps):
            tensors[f"{key}{i}"] = torch.zeros(
                (batch_size, hidden_size), dtype=dtype, device=device
            )
        return tensors

    model.make_empty_intermediate_tensors = make_empty_with_aux
    logger.info(
        "Reserved %d aux hidden-state slot(s) in the PP handoff for taps "
        "produced by stage %d.",
        num_taps,
        pp.world_size - 2,
    )


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
    if not layer_ids:
        for config_name in ("dflash_config", "eagle_config"):
            drafter_config = getattr(hf_config, config_name, None)
            if drafter_config and isinstance(drafter_config, dict):
                layer_ids = drafter_config.get("layer_ids")
                if layer_ids:
                    break
    if layer_ids and isinstance(layer_ids, (list, tuple)):
        return tuple(layer_ids)
    return None
