# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Activation collector for extracting intermediate layer activations during inference."""

from typing import Dict, Optional, Set

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)


class ActivationCollector:
    """
    Collects activations from specific layers during model forward pass.
    Uses PyTorch forward hooks to capture intermediate outputs.
    """

    def __init__(
        self, model: nn.Module, layer_indices: Optional[Set[int]] = None
    ):
        """
        Args:
            model: The transformer model (e.g., LlamaForCausalLM)
            layer_indices: Set of layer indices to collect from. None = all layers
        """
        self.model = model
        self.layer_indices = layer_indices
        self.activations: Dict[int, torch.Tensor] = {}
        self.hooks = []

    def _make_hook(self, layer_idx: int):
        """Create a hook function for a specific layer."""

        def hook(module, input, output):
            logger.info(
                f"🔔 Hook CALLED for layer {layer_idx}! output type={type(output)}, "
                f"is_tuple={isinstance(output, tuple)}"
            )
            try:
                # Store the activation (typically the hidden state)
                # output could be a tuple, so handle that
                if isinstance(output, tuple):
                    logger.info(
                        f"Layer {layer_idx}: output is tuple with {len(output)} elements"
                    )
                    activation = output[0]  # First element is usually hidden states
                else:
                    activation = output

                # Validate that we got a tensor
                if not isinstance(activation, torch.Tensor):
                    logger.warning(
                        f"Layer {layer_idx} output is not a tensor: {type(activation)}"
                    )
                    return

                logger.info(
                    f"Layer {layer_idx}: activation shape={activation.shape}, "
                    f"dtype={activation.dtype}, device={activation.device}"
                )
                # Clone and detach to avoid affecting gradients/memory
                # Move to CPU to avoid serialization issues
                self.activations[layer_idx] = activation.detach().cpu().clone()
                logger.info(
                    f"✓ Collected activation from layer {layer_idx}, "
                    f"shape={self.activations[layer_idx].shape}"
                )
            except Exception as e:
                logger.error(
                    f"Error collecting activation from layer {layer_idx}: {e}",
                    exc_info=True,
                )

        return hook

    def _get_layers(self):
        """Get the transformer layers from the model."""
        # Try different common model structures
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
            logger.info(
                f"Found layers at model.model.layers: {len(layers)} layers, "
                f"type: {type(layers)}"
            )
        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            layers = self.model.transformer.h
            logger.info(
                f"Found layers at model.transformer.h: {len(layers)} layers"
            )
        elif hasattr(self.model, "layers"):
            layers = self.model.layers
            logger.info(f"Found layers at model.layers: {len(layers)} layers")
        else:
            logger.error(
                f"Could not find transformer layers. Model type: {type(self.model)}, "
                f"has model: {hasattr(self.model, 'model')}, "
                f"has transformer: {hasattr(self.model, 'transformer')}, "
                f"has layers: {hasattr(self.model, 'layers')}"
            )
            if hasattr(self.model, "model"):
                logger.error(
                    f"model.model type: {type(self.model.model)}, "
                    f"has layers: {hasattr(self.model.model, 'layers')}"
                )
            raise ValueError(
                "Could not find transformer layers in model. "
                "Model structure not supported for activation extraction. "
                f"Model type: {type(self.model)}, "
                f"Model attributes: {[attr for attr in dir(self.model) if not attr.startswith('_')]}"
            )
        
        # Validate layer indices if specified
        if self.layer_indices is not None:
            num_layers = len(layers)
            invalid_layers = [idx for idx in self.layer_indices if idx >= num_layers]
            if invalid_layers:
                logger.warning(
                    f"Some requested layer indices are out of range: {invalid_layers}. "
                    f"Model has {num_layers} layers (0-{num_layers-1}). "
                    f"Only valid layers will be collected."
                )
                self.layer_indices = {
                    idx for idx in self.layer_indices if idx < num_layers
                }
        
        return layers

    def register_hooks(self):
        """Register hooks on the target layers."""
        layers = self._get_layers()
        num_layers = len(layers)
        logger.info(
            f"Registering hooks: num_layers={num_layers}, "
            f"layer_indices={self.layer_indices}"
        )
        
        # Check if model is compiled (torch.compile bypasses hooks)
        is_compiled = hasattr(self.model, "_compiled_callable") or hasattr(
            self.model, "compiled"
        )
        if is_compiled:
            logger.warning(
                "⚠ Model appears to be compiled. PyTorch hooks may not work with "
                "compiled models. Activation extraction may fail. "
                "Consider disabling compilation when using activation extraction."
            )

        layers_to_hook = []
        for layer_idx, layer in enumerate(layers):
            if self.layer_indices is None or layer_idx in self.layer_indices:
                hook = layer.register_forward_hook(self._make_hook(layer_idx))
                self.hooks.append(hook)
                layers_to_hook.append(layer_idx)
                logger.info(f"✓ Registered hook on layer {layer_idx}")
        
        logger.info(f"Hooked {len(layers_to_hook)} layers: {layers_to_hook}")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_activations(self) -> Dict[int, torch.Tensor]:
        """Get collected activations and clear the buffer."""
        activations = self.activations.copy()
        self.activations.clear()
        return activations

    def __enter__(self):
        """Context manager entry."""
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.remove_hooks()

