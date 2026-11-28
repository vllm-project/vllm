# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sized
from typing import cast

import torch
import torch.nn as nn

from vllm.config import SpeculativeConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    SupportsEagle3,
    _get_decoder_holder,
    get_eagle_decoder,
    supports_eagle3,
)

logger = init_logger(__name__)


def _get_max_aux_hidden_state_layer_id(model: nn.Module) -> int | None:
    holder = _get_decoder_holder(model)
    config = getattr(holder, "config", None)
    num_visits = getattr(config, "num_layer_visits", None)
    if isinstance(num_visits, int) and not isinstance(num_visits, bool):
        return num_visits

    layers = getattr(holder, "layers", None)
    if layers is None:
        decoder = getattr(holder, "decoder", None)
        layers = getattr(decoder, "layers", None)
    if layers is not None and isinstance(layers, Sized):
        return len(layers)

    if hasattr(holder, "_layer_names"):
        text_config = getattr(holder, "text_config", None)
        num_layers = getattr(text_config, "num_hidden_layers", None)
        if isinstance(num_layers, int) and not isinstance(num_layers, bool):
            return num_layers
        layer_names = getattr(holder, "_layer_names", None)
        if isinstance(layer_names, dict) and layer_names:
            return max(layer_names) + 1

    return None


def _validate_eagle3_aux_hidden_state_layers(
    model: nn.Module,
    aux_layers: tuple[int, ...],
    zero_based_source: tuple[str, tuple[int, ...]] | None = None,
) -> None:
    max_layer_id = _get_max_aux_hidden_state_layer_id(model)

    if zero_based_source is not None:
        field_name, raw_ids = zero_based_source
        invalid_raw = [
            layer_id
            for layer_id in raw_ids
            if not isinstance(layer_id, int)
            or isinstance(layer_id, bool)
            or layer_id < 0
            or (max_layer_id is not None and layer_id >= max_layer_id)
        ]
        if invalid_raw:
            if max_layer_id is None:
                raise ValueError(
                    f"{field_name} must contain non-negative 0-based layer "
                    f"indices for {type(model).__name__}; got {list(raw_ids)} "
                    f"(invalid: {invalid_raw})."
                )
            if max_layer_id > 0:
                valid_range = f"0-based layer indices in [0, {max_layer_id - 1}]"
                hint = (
                    " Note that target_layer_ids use 0-based decoder layer "
                    f"indices (e.g. final layer is {max_layer_id - 1}, not "
                    f"{max_layer_id})."
                )
            else:
                valid_range = "empty because model has 0 layers"
                hint = ""
            raise ValueError(
                f"{type(model).__name__} has {max_layer_id} decoder layers, so "
                f"{field_name} must contain {valid_range}; got {list(raw_ids)} "
                f"(out of range: {invalid_raw}).{hint}"
            )
        return

    invalid = [
        layer_id
        for layer_id in aux_layers
        if not isinstance(layer_id, int)
        or isinstance(layer_id, bool)
        or layer_id < 0
        or (max_layer_id is not None and layer_id > max_layer_id)
    ]
    if invalid:
        upper = max_layer_id if max_layer_id is not None else "..."
        raise ValueError(
            f"Auxiliary hidden state layer indices {invalid} from {aux_layers} "
            f"are out of range for {type(model).__name__} (expected [0, {upper}])."
        )


def set_eagle3_aux_hidden_state_layers(
    model: nn.Module, spec_config: SpeculativeConfig, use_pp: bool = False
) -> None:
    if not supports_eagle3(model):
        raise RuntimeError("Model does not support EAGLE3 interface")
    if isinstance(model, type):
        raise RuntimeError("Expected model instance for EAGLE3 configuration")
    eagle3_model = cast(SupportsEagle3, model)

    aux_layers, zero_based_source = _resolve_eagle3_aux_layers_from_config(spec_config)
    if aux_layers:
        logger.info("Using Eagle3 auxiliary layers from config: %s", aux_layers)
    else:
        aux_layers = eagle3_model.get_eagle3_default_aux_hidden_state_layers()
        zero_based_source = None
        logger.info("Using Eagle3 auxiliary layers from model: %s", aux_layers)
    _validate_eagle3_aux_hidden_state_layers(model, aux_layers, zero_based_source)
    eagle3_model.set_aux_hidden_state_layers(aux_layers)
    reserve_aux_intermediate_tensor_slots(model)

    if use_pp:
        assert spec_config.method is not None
        verify_supports_aux_hidden_states_over_pp(model, spec_config.method)


def verify_supports_aux_hidden_states_over_pp(model: nn.Module, method: str) -> None:
    decoder = get_eagle_decoder(model)
    if decoder is None or not decoder.supports_aux_hidden_states_over_pp:
        raise ValueError(
            f"{type(model).__name__} does not support {method} with "
            "pipeline parallelism"
        )


def reserve_aux_intermediate_tensor_slots(model: nn.Module) -> None:
    from vllm.distributed.parallel_state import model_parallel_is_initialized

    if not model_parallel_is_initialized():
        return
    decoder = get_eagle_decoder(model)
    if decoder is None or not decoder.supports_aux_hidden_states_over_pp:
        return
    upstream_keys, _ = decoder.aux_hidden_state_slot_keys()
    if not upstream_keys:
        return

    hidden_size = cast(nn.Module, decoder).config.hidden_size
    make_empty = model.make_empty_intermediate_tensors

    def make_empty_with_aux(batch_size, dtype, device):
        tensors = make_empty(batch_size, dtype, device)
        for key in upstream_keys:
            tensors[key] = torch.zeros(
                (batch_size, hidden_size), dtype=dtype, device=device
            )
        return tensors

    model.make_empty_intermediate_tensors = make_empty_with_aux


def _resolve_eagle3_aux_layers_from_config(
    spec_config: SpeculativeConfig,
) -> tuple[tuple[int, ...] | None, tuple[str, tuple[int, ...]] | None]:
    if not (spec_config and spec_config.draft_model_config):
        return None, None
    hf_config = spec_config.draft_model_config.hf_config
    zero_based_source: tuple[str, tuple[int, ...]] | None = None
    layer_ids = getattr(hf_config, "eagle_aux_hidden_state_layer_ids", None)
    if not layer_ids:
        dflash_config = getattr(hf_config, "dflash_config", None)
        if dflash_config and isinstance(dflash_config, dict):
            raw_ids = dflash_config.get("target_layer_ids")
            if raw_ids:
                # Add 1 to convert DFlash's aux layer id semantics
                layer_ids = [i + 1 for i in raw_ids]
                zero_based_source = ("dflash_config.target_layer_ids", tuple(raw_ids))
    if not layer_ids:
        dspark_layer_ids = getattr(hf_config, "dspark_target_layer_ids", None)
        if dspark_layer_ids:
            if getattr(hf_config, "model_type", None) == "deepseek_v41":
                # v4.1 reads the attention *inputs* of its target layers, and
                # the target model captures the entry stream of layer L when
                # idx+1 == L, so the ids are used as-is. (v4's ids are in
                # capture-after semantics and keep the +1.)
                layer_ids = list(dspark_layer_ids)
            else:
                layer_ids = [i + 1 for i in dspark_layer_ids]
                zero_based_source = (
                    "dspark_target_layer_ids",
                    tuple(dspark_layer_ids),
                )
    if not layer_ids:
        # Dense DSpark (e.g. Qwen3) also uses different aux layer semantics.
        target_layer_ids = getattr(hf_config, "target_layer_ids", None)
        if target_layer_ids:
            layer_ids = [i + 1 for i in target_layer_ids]
            zero_based_source = ("target_layer_ids", tuple(target_layer_ids))
    if not layer_ids:
        for config_name in ("dflash_config", "eagle_config"):
            drafter_config = getattr(hf_config, config_name, None)
            if drafter_config and isinstance(drafter_config, dict):
                layer_ids = drafter_config.get("layer_ids")
                if layer_ids:
                    break
    if layer_ids and isinstance(layer_ids, (list, tuple)):
        return tuple(layer_ids), zero_based_source
    return None, None


def get_eagle3_aux_layers_from_config(
    spec_config: SpeculativeConfig,
) -> tuple[int, ...] | None:
    aux_layers, _ = _resolve_eagle3_aux_layers_from_config(spec_config)
    return aux_layers
