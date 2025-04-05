# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class ControlVectorMapping:
    layer_mapping: dict[int, torch.Tensor]


class BaseLayerWithControlVector(nn.Module):
    pass


class MLPWithControlVector(BaseLayerWithControlVector):

    def __init__(self, base_layer) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.normalize = True
        self.control_vectors: dict[int, torch.Tensor | int] = {}
        self.active_vector: Optional[torch.Tensor] = None

    def set_normalization(self, normalize: bool) -> None:
        self.normalize = normalize

    def set_layer_id(self, layer_id: int) -> None:
        """assign the layer id of this MLP layer"""
        self.layer_id = layer_id

    def set_control_vector(self, index: int, cv_vector: torch.Tensor):
        """Set a control vector at a specific index."""
        self.control_vectors[index] = cv_vector

    def get_control_vector(self, index: int) -> Optional[torch.Tensor]:
        """Get a control vector by index."""
        return self.control_vectors.get(index)

    def reset_control_vector(self, index: int):
        """Reset a control vector to zero at a specific index."""
        if index in self.control_vectors:
            self.control_vectors[index] = 0

    def set_active_tensor(self, index: int):
        """Sets the active vector"""
        if index is not None and index in self.control_vectors:
            self.active_vector = self.control_vectors[index]
        else:
            self.active_vector = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional application of control vectors."""
        hidden_states = self.base_layer(hidden_states)

        norm_pre = torch.norm(hidden_states, dim=-1, keepdim=True)

        cv = self.active_vector

        if cv is not None and cv.numel() > 0:
            hidden_states += cv

            if self.normalize:
                norm_post = torch.norm(hidden_states, dim=-1, keepdim=True)
                hidden_states = hidden_states * norm_pre / norm_post

        return hidden_states
