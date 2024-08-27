from dataclasses import dataclass, field
from typing import Optional, Dict, List

import torch
from torch import nn

from vllm.adapter_commons.layers import AdapterMapping
from vllm.control_vectors.request import ControlVectorRequest


@dataclass
class ControlVectorMapping:
    layer_mapping: Dict[int, torch.Tensor]


class BaseLayerWithControlVector(nn.Module):
    pass


class MLPWithControlVector(BaseLayerWithControlVector):

    def __init__(self, base_layer) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.control_vectors = {}
        self.normalize = False
        self.active_vector: torch.Tensor = None

    def set_normalization(self, normalize: bool) -> None:
        self.normalize = normalize

    def set_layer_id(self, layer_id: int) -> None:
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
                print("HERE")
                hidden_states = hidden_states * norm_pre / torch.norm(
                    hidden_states, dim=-1, keepdim=True)

        return hidden_states
