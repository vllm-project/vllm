# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Observation Hook - Integration layer between Observation Plugin Manager
and vLLM internals.

Handles:
- Installing PyTorch forward hooks
- Capturing hidden states
- Calling plugin callbacks
- Handling plugin responses for ABORT/PREEMPT

This is internal implementation - users should use ObservationPlugin.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from vllm.plugins.observation.interface import (
    ObservationResult,
    PluginManager,
    RequestContext,
)

logger = logging.getLogger(__name__)


class ObservationHook:
    def __init__(
        self,
        plugin_manager: PluginManager,
        model: nn.Module,
    ):
        self.plugin_manager = plugin_manager
        self.model = model

        self.observation_layers = self.plugin_manager.get_observation_layers()
        self._captured_hidden_states: dict[int, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

    def install_hooks(self) -> None:
        if not self.observation_layers:
            return

        layers = self._get_model_layers()
        num_layers = len(layers)

        for layer_idx in self.observation_layers:
            # Handle negative indices
            norm_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx
            if norm_idx < 0 or norm_idx >= num_layers:
                logger.warning(
                    "Observation layer %d (normalized %d) "
                    "out of range. Model has %d layers. Skipping.",
                    layer_idx,
                    norm_idx,
                    num_layers,
                )
                continue

            hook = layers[norm_idx].register_forward_hook(
                self._create_hook_fn(norm_idx)
            )
            self._hooks.append(hook)

    def _get_model_layers(self) -> nn.ModuleList:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "layers"):
            return self.model.layers
        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            return self.model.transformer.h
        elif hasattr(self.model, "h"):
            return self.model.h
        else:
            raise AttributeError(
                "Could not find transformer layers in model. "
                "Model architecture may not be supported for observation."
            )

    def _create_hook_fn(self, layer_idx: int):
        def hook_fn(module: nn.Module, input: tuple, output: Any) -> None:
            hidden_states = output[0] if isinstance(output, tuple) else output
            # Clone so later operations don't modify the state we observed
            self._captured_hidden_states[layer_idx] = hidden_states.detach().clone()

        return hook_fn

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def process_step(
        self,
        request_contexts: list[RequestContext],
    ) -> list[ObservationResult]:
        """
        Process a batch step.
        Takes the request contexts for the current batch and passes them
        and the captured hidden states to the plugin manager.
        """
        if not self.observation_layers:
            return [ObservationResult()] * len(request_contexts)

        results = self.plugin_manager.on_step_batch(
            self._captured_hidden_states, request_contexts
        )

        # Clear after processing step
        self._captured_hidden_states.clear()

        return results
