# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Centralized PP propagation for Eagle3 aux hidden states.

Eagle3 speculative decoding collects auxiliary hidden states from specific
target-model layers (e.g. layers 2, N/2, N-3).  When Pipeline Parallelism
(PP > 1) splits the model across multiple pipeline stages, those aux layers
may reside on different PP ranks.  Without propagation, each non-last PP rank
silently discards its locally-collected aux states and the drafter (running on
the last PP rank) receives fewer states than expected, causing a shape
mismatch in ``combine_hidden_states``.

This module solves the problem **without modifying any model forward methods**.
It uses ``register_forward_hook`` on the relevant decoder layers and a thin
wrapper around the inner model's ``forward`` / ``make_empty_intermediate_tensors``
to transparently pack/unpack aux states into/out of ``IntermediateTensors``.

The wrapper is only installed when ``PP > 1``; the ``PP == 1`` code path is
completely unaffected.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.models.interfaces import (
    AUX_HIDDEN_STATE_TENSOR_PREFIX,
    extract_aux_hidden_states,
)
from vllm.model_executor.models.utils import PPMissingLayer

logger = logging.getLogger(__name__)


def install_eagle3_pp_aux_propagation(inner_model: nn.Module) -> bool:
    """Install forward hooks + forward/make_empty wrappers for PP aux
    propagation on *inner_model* (e.g. ``LlamaModel`` or
    ``DeepseekV2Model``).

    Must be called **after** :meth:`set_aux_hidden_state_layers`.

    Returns ``True`` if the wrapper was installed, ``False`` if skipped
    (PP == 1, no aux layers, or already installed).
    """
    pp_group = get_pp_group()
    if pp_group.world_size == 1:
        return False

    if getattr(inner_model, "_eagle3_pp_aux_installed", False):
        return False

    aux_layers = getattr(inner_model, "aux_hidden_state_layers", ())
    if not aux_layers:
        return False

    start_layer = inner_model.start_layer
    end_layer = inner_model.end_layer
    hidden_size = inner_model.config.hidden_size

    # ------------------------------------------------------------------
    # 1.  Forward hooks – capture aux hidden states from local layers.
    #
    # aux_layer N refers to the output of global layer N-1.
    # All vLLM decoder layers return (hidden_states, residual[, ...]),
    # so the aux value = hidden_states + residual.
    # ------------------------------------------------------------------
    captured: dict[int, torch.Tensor] = {}

    def _make_hook(aux_idx: int):
        def _hook(module, inputs, output):
            if isinstance(output, tuple) and len(output) >= 2:
                hs = output[0]
                res = output[1]
            else:
                hs = output
                res = None
            captured[aux_idx] = hs + res if res is not None else hs

        return _hook

    num_local_hooks = 0
    for aux_idx in aux_layers:
        target = aux_idx - 1  # 0-indexed global layer index
        if target < 0:
            # aux_layer 0 = embedding-level state (no layer to hook).
            continue
        if start_layer <= target < end_layer:
            layer = inner_model.layers[target]
            if not isinstance(layer, PPMissingLayer):
                layer.register_forward_hook(_make_hook(aux_idx))
                num_local_hooks += 1

    # ------------------------------------------------------------------
    # 2.  Wrap forward – accumulate aux across PP stages.
    # ------------------------------------------------------------------
    original_forward = inner_model.forward

    def pp_aux_forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        captured.clear()
        incoming_aux = extract_aux_hidden_states(intermediate_tensors)
        result = original_forward(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

        local_aux = [captured[k] for k in sorted(captured)]
        all_aux = incoming_aux + local_aux

        if not pp_group.is_last_rank:
            # Non-last rank – pack aux states into IntermediateTensors so
            # the next PP stage can receive them.
            for i, t in enumerate(all_aux):
                result.tensors[f"{AUX_HIDDEN_STATE_TENSOR_PREFIX}{i}"] = t
            return result

        # Last rank – override the model's own aux return with the complete
        # list (incoming + hook-captured).  The model's own aux collection may
        # be wrong because of local-vs-global index mismatch under PP.
        hidden_states = result[0] if isinstance(result, tuple) else result
        if all_aux:
            return hidden_states, all_aux
        return hidden_states

    inner_model.forward = pp_aux_forward.__get__(inner_model, type(inner_model))

    # ------------------------------------------------------------------
    # 3.  Wrap make_empty_intermediate_tensors – pre-allocate placeholders.
    # ------------------------------------------------------------------
    original_make_empty = inner_model.make_empty_intermediate_tensors

    def pp_make_empty(batch_size, dtype, device):
        result = original_make_empty(batch_size, dtype, device)
        if not pp_group.is_first_rank:
            num_incoming = sum(
                1
                for layer_idx in aux_layers
                if layer_idx > 0 and (layer_idx - 1) < start_layer
            )
            for i in range(num_incoming):
                result.tensors[f"{AUX_HIDDEN_STATE_TENSOR_PREFIX}{i}"] = torch.zeros(
                    (batch_size, hidden_size),
                    dtype=dtype,
                    device=device,
                )
        return result

    inner_model.make_empty_intermediate_tensors = pp_make_empty
    inner_model._eagle3_pp_aux_installed = True

    logger.info(
        "Installed Eagle3 PP aux propagation on %s "
        "(aux_layers=%s, start=%d, end=%d, local_hooks=%d).",
        type(inner_model).__name__,
        aux_layers,
        start_layer,
        end_layer,
        num_local_hooks,
    )
    return True
